from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from scipy import stats
import warnings
from dataclasses import dataclass
from math import log, sqrt, exp
from scipy.stats import norm

warnings.filterwarnings('ignore')


def download_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download data for multiple tickers efficiently."""
    print(f"Downloading data for {len(tickers)} assets...")

    batch_size = 10
    all_data = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)

            if len(batch) == 1:
                if not data.empty:
                    all_data[batch[0]] = data
            else:
                for ticker in batch:
                    if ticker in data:
                        if not data[ticker].empty:
                            all_data[ticker] = data[ticker]
        except Exception as e:
            print(f"  Warning: Could not download batch {batch}: {e}")
            continue

    processed = {}
    for ticker, df in all_data.items():
        try:
            if df.empty:
                continue

            if 'Adj Close' in df.columns:
                df['Adj Close'] = df['Adj Close']
            elif 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            else:
                continue

            required = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in required:
                if col not in df.columns:
                    if col == 'Adj Close':
                        df['Adj Close'] = df['Close'] if 'Close' in df.columns else df['Open']
                    elif col in ['Open', 'High', 'Low', 'Close']:
                        df[col] = df['Adj Close']
                    else:
                        df[col] = np.nan

            df = df[required]
            processed[ticker] = df
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    print(f"\nSuccessfully downloaded {len(processed)}/{len(tickers)} assets")
    return processed


def find_cointegrated_pairs(prices_df: pd.DataFrame, pvalue_threshold: float = 0.05) -> List[Tuple[str, str, float]]:
    """Find cointegrated pairs using Engle-Granger test."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        print("  Installing statsmodels for cointegration test...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'statsmodels'])
        from statsmodels.tsa.stattools import coint

    n = len(prices_df.columns)
    pairs = []

    print(f"  Testing {n * (n - 1) // 2} possible pairs...")

    for i in range(n):
        for j in range(i + 1, n):
            ticker1 = prices_df.columns[i]
            ticker2 = prices_df.columns[j]

            series1 = prices_df[ticker1].dropna()
            series2 = prices_df[ticker2].dropna()
            common_idx = series1.index.intersection(series2.index)

            if len(common_idx) < 100:
                continue

            try:
                score, pvalue, _ = coint(series1.loc[common_idx], series2.loc[common_idx])
                if pvalue < pvalue_threshold:
                    X = series1.loc[common_idx].values.reshape(-1, 1)
                    y = series2.loc[common_idx].values
                    hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

                    pairs.append((ticker1, ticker2, pvalue, hedge_ratio, score))
            except Exception as e:
                continue

    pairs.sort(key=lambda x: x[2])
    print(f"  Found {len(pairs)} cointegrated pairs")

    return [(p[0], p[1], p[3]) for p in pairs[:20]]


def statistical_arbitrage_strategy(
    prices_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    target_volatility: float = 0.10,
    max_position_pct: float = 0.10,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float = 3.0,
    lookback_days: int = 60,
    rebalance_days: int = 5,
    max_active_pairs: int = 5,
    max_holding_days: int = 60,
    vol_lookback_days: int = 20,
    vol_scalar_bounds: Tuple[float, float] = (0.25, 2.0),
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    prices_df = prices_df.sort_index()
    dates = prices_df.index
    tickers = prices_df.columns.tolist()

    print("Finding cointegrated pairs...")
    pairs = find_cointegrated_pairs(prices_df)

    if len(pairs) == 0:
        print("Warning: No cointegrated pairs found. Using sector pairs as fallback.")
        sector_pairs = [
            ('XLF', 'XLK', 1.0),
            ('XLE', 'XLU', 1.0),
            ('XLV', 'XLP', 1.0),
            ('XLY', 'XLI', 1.0),
        ]
        pairs = [(a, b, hr) for a, b, hr in sector_pairs
                 if a in prices_df.columns and b in prices_df.columns]

    portfolio = pd.DataFrame(index=dates, data={
        "equity": float(initial_capital),
        "cash": float(initial_capital),
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
        "active_pairs": 0,
        "return": 0.0,
    })
    positions = pd.DataFrame(0.0, index=dates, columns=tickers)

    active_pairs: Dict[str, Dict[str, Any]] = {}

    def _zscore(t1: str, t2: str, hedge_ratio: float, end_i: int) -> float | None:
        start = max(0, end_i - lookback_days)
        window = prices_df.iloc[start:end_i][[t1, t2]]
        if len(window) < max(30, lookback_days // 2):
            return None
        if window.isna().any().any():
            return None

        spread = window[t1] - hedge_ratio * window[t2]
        std = spread.std(ddof=1)
        if std is None or std == 0 or np.isnan(std):
            return None
        z = (spread.iloc[-1] - spread.mean()) / std
        if np.isnan(z) or np.isinf(z):
            return None
        return float(z)

    def _mark_to_market(pos_row: pd.Series, price_row: pd.Series, cash: float) -> float:
        return float(cash + (pos_row * price_row).sum())

    def _apply_target_position(
        pos_row: pd.Series, cash: float, ticker: str, target_shares: float, px: float
    ) -> float:
        cur = float(pos_row[ticker])
        delta = float(target_shares - cur)
        if delta != 0.0:
            cash -= delta * float(px)
            pos_row[ticker] = float(target_shares)
        return cash

    def _vol_scalar(i: int) -> float:
        if target_volatility <= 0:
            return 1.0
        if i < lookback_days + 2:
            return 1.0

        start = max(0, i - vol_lookback_days)
        rets = portfolio["return"].iloc[start:i].dropna()
        if len(rets) < 5:
            return 1.0

        realized = float(rets.std(ddof=1) * np.sqrt(252))
        if realized <= 1e-12 or np.isnan(realized):
            return 1.0

        s = target_volatility / realized
        lo, hi = vol_scalar_bounds
        return float(np.clip(s, lo, hi))

    for i in range(lookback_days, len(dates)):
        current_date = dates[i]
        prev_date = dates[i - 1]

        positions.loc[current_date] = positions.loc[prev_date]
        cash = float(portfolio.at[prev_date, "cash"])

        price_today = prices_df.loc[current_date]
        pos_today = positions.loc[current_date]

        for pair_key, info in list(active_pairs.items()):
            t1, t2 = info["t1"], info["t2"]
            hr = float(info["hedge_ratio"])

            z = _zscore(t1, t2, hr, end_i=i)
            if z is None:
                continue

            exit_signal = (
                abs(z) < exit_z
                or abs(z) > stop_loss_z
                or ((z > 0 and info["entry_z"] < 0) or (z < 0 and info["entry_z"] > 0))
                or (i - int(info["entry_i"]) >= max_holding_days)
            )

            if exit_signal:
                cash = _apply_target_position(pos_today, cash, t1, 0.0, float(price_today[t1]))
                cash = _apply_target_position(pos_today, cash, t2, 0.0, float(price_today[t2]))
                del active_pairs[pair_key]

        if rebalance_days > 0 and (i % rebalance_days == 0):
            equity_now = _mark_to_market(pos_today, price_today, cash)
            if equity_now <= 0:
                portfolio.at[current_date, "cash"] = cash
                portfolio.at[current_date, "equity"] = equity_now
                continue

            vs = _vol_scalar(i)

            candidates = []
            for (t1, t2, hr) in pairs:
                key = f"{t1}__{t2}"
                if key in active_pairs:
                    continue
                if t1 not in prices_df.columns or t2 not in prices_df.columns:
                    continue
                z = _zscore(t1, t2, float(hr), end_i=i)
                if z is None:
                    continue
                if abs(z) >= entry_z:
                    candidates.append((abs(z), z, t1, t2, float(hr)))

            candidates.sort(reverse=True, key=lambda x: x[0])

            for _, z, t1, t2, hr in candidates:
                if len(active_pairs) >= max_active_pairs:
                    break

                px1 = float(price_today[t1])
                px2 = float(price_today[t2])
                if not np.isfinite(px1) or not np.isfinite(px2) or px1 <= 0 or px2 <= 0:
                    continue

                equity_now = _mark_to_market(pos_today, price_today, cash)
                pair_notional = float(max_position_pct * equity_now * vs)

                base_shares_t1 = pair_notional / px1
                if z > 0:
                    target_t1 = -base_shares_t1
                    target_t2 = +hr * base_shares_t1
                else:
                    target_t1 = +base_shares_t1
                    target_t2 = -hr * base_shares_t1

                max_leg_notional = float(max_position_pct * equity_now * vs)
                leg1_notional = abs(target_t1 * px1)
                leg2_notional = abs(target_t2 * px2)
                scale = 1.0
                if leg1_notional > max_leg_notional:
                    scale = min(scale, max_leg_notional / (leg1_notional + 1e-12))
                if leg2_notional > max_leg_notional:
                    scale = min(scale, max_leg_notional / (leg2_notional + 1e-12))
                if scale <= 0:
                    continue
                target_t1 *= scale
                target_t2 *= scale

                cash = _apply_target_position(pos_today, cash, t1, target_t1, px1)
                cash = _apply_target_position(pos_today, cash, t2, target_t2, px2)

                active_pairs[f"{t1}__{t2}"] = {
                    "t1": t1,
                    "t2": t2,
                    "hedge_ratio": hr,
                    "entry_z": float(z),
                    "entry_i": int(i),
                }

        equity = _mark_to_market(pos_today, price_today, cash)
        portfolio.at[current_date, "cash"] = cash
        portfolio.at[current_date, "equity"] = equity

        pos_values = pos_today * price_today
        long_exposure = float(pos_values[pos_values > 0].sum())
        short_exposure = float((-pos_values[pos_values < 0]).sum())
        gross = long_exposure + short_exposure

        if equity != 0:
            portfolio.at[current_date, "gross_exposure"] = gross / abs(equity)
            portfolio.at[current_date, "net_exposure"] = (long_exposure - short_exposure) / equity
        portfolio.at[current_date, "active_pairs"] = len(active_pairs)

        prev_equity = float(portfolio.at[prev_date, "equity"])
        if prev_equity != 0:
            portfolio.at[current_date, "return"] = (equity - prev_equity) / prev_equity
        else:
            portfolio.at[current_date, "return"] = 0.0

    portfolio["equity"] = portfolio["equity"].ffill()
    portfolio["cash"] = portfolio["cash"].ffill()
    portfolio["return"] = portfolio["return"].fillna(0.0)

    return portfolio, positions


def merger_arbitrage_strategy(prices_df: pd.DataFrame,
                              merger_etfs: List[str] = ['MNA', 'MRGR'],
                              initial_capital: float = 100_000.0) -> pd.DataFrame:
    available_etfs = [etf for etf in merger_etfs if etf in prices_df.columns]

    if not available_etfs:
        low_vol_assets = ['SHY', 'IEI', 'TIP', 'BIL']
        available_etfs = [a for a in low_vol_assets if a in prices_df.columns]

    if not available_etfs:
        portfolio = pd.DataFrame(index=prices_df.index)
        portfolio['equity'] = initial_capital
        portfolio['return'] = 0.0
        return portfolio

    portfolio = pd.DataFrame(index=prices_df.index)

    returns = {}
    for etf in available_etfs:
        returns[etf] = prices_df[etf].pct_change().fillna(0)

    etf_returns = pd.DataFrame(returns)
    portfolio_return = etf_returns.mean(axis=1)

    portfolio['return'] = portfolio_return
    portfolio['equity'] = initial_capital * (1 + portfolio_return).cumprod()

    return portfolio


def risk_parity_strategy(prices_df: pd.DataFrame,
                         rp_assets: List[str] = ['TLT', 'GLD', 'TIP', 'IEF', 'LQD'],
                         initial_capital: float = 100_000.0,
                         lookback: int = 63) -> pd.DataFrame:

    available_assets = [asset for asset in rp_assets if asset in prices_df.columns]

    if len(available_assets) < 3:
        print(f"Warning: Only {len(available_assets)} risk parity assets available")
        if len(available_assets) == 0:
            portfolio = pd.DataFrame(index=prices_df.index)
            portfolio['equity'] = initial_capital
            portfolio['return'] = 0.0
            return portfolio

    returns_df = pd.DataFrame()
    for asset in available_assets:
        returns_df[asset] = prices_df[asset].pct_change().fillna(0)

    portfolio = pd.DataFrame(index=returns_df.index)

    weights_history = []

    for i in range(lookback, len(returns_df)):
        current_date = returns_df.index[i]

        rolling_returns = returns_df.iloc[i - lookback:i]
        volatilities = rolling_returns.std() * np.sqrt(252)
        volatilities = volatilities.replace(0, np.nan)

        if volatilities.isna().all():
            weights = pd.Series(1 / len(available_assets), index=available_assets)
        else:
            inv_vol = 1 / volatilities
            weights = inv_vol / inv_vol.sum()

        weights_history.append(weights)

    if weights_history:
        weights_df = pd.DataFrame(weights_history, index=returns_df.index[lookback:])
        weights_df = weights_df.reindex(returns_df.index).ffill().bfill()
    else:
        weights_df = pd.DataFrame(1 / len(available_assets),
                                  index=returns_df.index,
                                  columns=available_assets)

    portfolio_return = (returns_df * weights_df).sum(axis=1)
    portfolio['return'] = portfolio_return
    portfolio['equity'] = initial_capital * (1 + portfolio_return).cumprod()

    return portfolio


@dataclass
class CCConfig:
    underlying: str = "SPY"
    initial_capital: float = 100_000.0
    target_dte: int = 35
    min_roll_dte: int = 5
    otm_pct: float = 0.01
    iv_lookback: int = 20
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.0
    stock_slippage_bps: float = 1.0
    option_slippage_pct: float = 0.02
    option_fee_per_contract: float = 0.65
    shares_per_contract: int = 100
    max_contracts: int = 999999

def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(days=14)

def next_monthly_expiry(date: pd.Timestamp, target_dte: int) -> pd.Timestamp:
    candidates = []
    for m in range(0, 4):
        dt = (date + pd.DateOffset(months=m))
        exp = third_friday(dt.year, dt.month)
        if exp <= date:
            continue
        dte = (exp - date).days
        candidates.append((abs(dte - target_dte), dte, exp))
    candidates.sort()
    return candidates[0][2]

def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def covered_call_strategy(prices_df, underlying="SPY", initial_capital=100_000.0):
    cfg = CCConfig(underlying=underlying, initial_capital=initial_capital)
    return covered_call_backtest_strike_expiry_aware(prices_df, cfg)


def covered_call_backtest_strike_expiry_aware(prices_df: pd.DataFrame, cfg: CCConfig) -> pd.DataFrame:
    if cfg.underlying not in prices_df.columns:
        raise ValueError(f"{cfg.underlying} not found in prices_df")

    px = prices_df[cfg.underlying].dropna().copy()
    dates = px.index

    rets = px.pct_change().fillna(0.0)
    rv = rets.rolling(cfg.iv_lookback).std() * np.sqrt(252)
    rv = rv.bfill()

    cash = cfg.initial_capital
    shares = 0
    opt_short = 0
    opt_strike = None
    opt_expiry = None
    opt_entry_price = 0.0

    equity_curve = []

    def stock_trade_cost(notional):
        return abs(notional) * (cfg.stock_slippage_bps / 10000.0)

    def option_trade_cost(premium_total):
        slip = abs(premium_total) * cfg.option_slippage_pct
        fee = abs(opt_short) * cfg.option_fee_per_contract if opt_short != 0 else 0.0
        return slip + fee

    for t, dt in enumerate(dates):
        S = float(px.loc[dt])
        sigma = float(rv.loc[dt])
        r = cfg.risk_free_rate
        q = cfg.dividend_yield

        if shares == 0:
            lot_cost = cfg.shares_per_contract * S
            n_lots = int(cash // lot_cost)
            n_lots = min(n_lots, cfg.max_contracts)
            if n_lots > 0:
                buy_shares = n_lots * cfg.shares_per_contract
                notional = buy_shares * S
                cost = notional + stock_trade_cost(notional)
                cash -= cost
                shares += buy_shares

        if shares > 0 and opt_short == 0:
            opt_expiry = next_monthly_expiry(dt, cfg.target_dte)
            dte = (opt_expiry - dt).days
            T = dte / 365.0
            opt_strike = round(S * (1.0 + cfg.otm_pct), 2)

            call_mid = bs_call_price(S, opt_strike, T, r, sigma, q)
            contracts = shares // cfg.shares_per_contract
            opt_short = -int(contracts)

            premium_total = abs(opt_short) * call_mid * cfg.shares_per_contract
            cash += premium_total
            cash -= option_trade_cost(premium_total)
            opt_entry_price = call_mid

        opt_value = 0.0
        if opt_short != 0 and opt_expiry is not None:
            dte = (opt_expiry - dt).days
            T = max(dte, 0) / 365.0
            call_mid = bs_call_price(S, opt_strike, T, r, sigma, q)
            opt_value = opt_short * call_mid * cfg.shares_per_contract

        if opt_short != 0 and opt_expiry is not None:
            dte = (opt_expiry - dt).days
            if dte <= cfg.min_roll_dte and dte > 0:
                T = dte / 365.0
                buyback = bs_call_price(S, opt_strike, T, r, sigma, q)
                buyback_total = abs(opt_short) * buyback * cfg.shares_per_contract
                cash -= buyback_total
                cash -= (abs(buyback_total) * cfg.option_slippage_pct + abs(opt_short) * cfg.option_fee_per_contract)

                opt_expiry = next_monthly_expiry(dt, cfg.target_dte)
                new_dte = (opt_expiry - dt).days
                T2 = new_dte / 365.0
                opt_strike = round(S * (1.0 + cfg.otm_pct), 2)
                sell_price = bs_call_price(S, opt_strike, T2, r, sigma, q)
                sell_total = abs(opt_short) * sell_price * cfg.shares_per_contract
                cash += sell_total
                cash -= (abs(sell_total) * cfg.option_slippage_pct + abs(opt_short) * cfg.option_fee_per_contract)

        if opt_short != 0 and opt_expiry is not None and dt >= opt_expiry:
            intrinsic = max(0.0, S - opt_strike)
            settlement = abs(opt_short) * intrinsic * cfg.shares_per_contract

            if intrinsic > 0 and shares > 0:
                shares_to_deliver = abs(opt_short) * cfg.shares_per_contract
                shares_to_deliver = min(shares_to_deliver, shares)
                proceeds = shares_to_deliver * opt_strike
                cash += proceeds
                cash -= stock_trade_cost(proceeds)
                shares -= shares_to_deliver

            cash -= settlement

            opt_short = 0
            opt_strike = None
            opt_expiry = None
            opt_entry_price = 0.0
            opt_value = 0.0

        stock_value = shares * S
        equity = cash + stock_value + opt_value
        equity_curve.append((dt, equity))

    out = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    out["return"] = out["equity"].pct_change().fillna(0.0)
    return out


def multi_strategy_portfolio(prices_df: pd.DataFrame,
                             initial_capital: float = 100_000.0,
                             strategy_weights: Dict[str, float] = None) -> Dict:

    if strategy_weights is None:
        strategy_weights = {
            'stat_arb': 0.35,
            'merger_arb': 0.25,
            'risk_parity': 0.20,
            'covered_call': 0.20
        }

    print("\n" + "=" * 80)
    print("RUNNING MULTI-STRATEGY PORTFOLIO")
    print("=" * 80)

    strategies = {}

    print("\n1. Statistical Arbitrage Strategy")
    if strategy_weights.get('stat_arb', 0) > 0:
        stat_arb_capital = initial_capital * strategy_weights['stat_arb']
        stat_arb_portfolio, stat_arb_positions = statistical_arbitrage_strategy(
            prices_df=prices_df,
            initial_capital=stat_arb_capital,
            target_volatility=0.10,
            max_position_pct=0.10,
            entry_z=2.0,
            exit_z=0.5,
            stop_loss_z=3.0,
            lookback_days=60,
            rebalance_days=5,
            max_active_pairs=5
        )
    else:
        stat_arb_portfolio = pd.DataFrame(index=prices_df.index)
        stat_arb_portfolio['equity'] = 0.0
        stat_arb_portfolio['return'] = 0.0
        stat_arb_positions = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    strategies['stat_arb'] = stat_arb_portfolio

    print("\n2. Merger Arbitrage Strategy")
    merger_arb_capital = initial_capital * strategy_weights['merger_arb']
    merger_arb_portfolio = merger_arbitrage_strategy(
        prices_df=prices_df,
        merger_etfs=['MNA'],
        initial_capital=merger_arb_capital
    )
    strategies['merger_arb'] = merger_arb_portfolio

    print("\n3. Risk Parity Strategy")
    risk_parity_capital = initial_capital * strategy_weights['risk_parity']
    risk_parity_portfolio = risk_parity_strategy(
        prices_df=prices_df,
        rp_assets=['TLT', 'GLD', 'TIP', 'IEF', 'LQD', 'BND'],
        initial_capital=risk_parity_capital
    )
    strategies['risk_parity'] = risk_parity_portfolio

    print("\n4. Covered Call Strategy")
    covered_call_capital = initial_capital * strategy_weights['covered_call']
    covered_call_portfolio = covered_call_strategy(
        prices_df=prices_df,
        underlying='SPY',
        initial_capital=covered_call_capital
    )
    strategies['covered_call'] = covered_call_portfolio

    common_dates = prices_df.index
    for strategy_name in strategies:
        strategies[strategy_name] = strategies[strategy_name].reindex(common_dates).ffill().bfill()

    combined = pd.DataFrame(index=common_dates)

    for strategy_name, portfolio in strategies.items():
        if 'equity' in portfolio.columns:
            combined[strategy_name] = portfolio['equity']
        else:
            combined[strategy_name] = initial_capital * strategy_weights[strategy_name]

    combined['total'] = combined.sum(axis=1)
    combined['return'] = combined['total'].pct_change().fillna(0)

    for strategy_name in strategies:
        if 'return' in strategies[strategy_name].columns:
            combined[f'{strategy_name}_return'] = strategies[strategy_name]['return']

    return {
        'combined': combined,
        'strategies': strategies,
        'positions': stat_arb_positions,
        'strategy_weights': strategy_weights
    }


def calculate_performance_metrics(
    portfolio: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> Dict:

    returns = portfolio['return'].dropna()

    if 'total' in portfolio.columns:
        equity = portfolio['total'].dropna()
    elif 'equity' in portfolio.columns:
        equity = portfolio['equity'].dropna()
    else:
        return {}

    if len(returns) < 10 or len(equity) < 10:
        return {}

    metrics = {}

    start, end = equity.index[0], equity.index[-1]
    years = max((end - start).days / 365.25, 1e-9)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    metrics['Total Return'] = total_return
    metrics['CAGR'] = cagr

    daily_vol = float(returns.std(ddof=1))
    annual_vol = daily_vol * np.sqrt(periods_per_year)
    metrics['Annual Volatility'] = annual_vol

    downside = returns[returns < 0]
    downside_vol_daily = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    downside_vol_annual = downside_vol_daily * np.sqrt(periods_per_year) if downside_vol_daily > 0 else 0.0
    metrics['Downside Volatility'] = downside_vol_annual

    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_daily

    ex_mean = float(excess.mean())
    ex_std = float(excess.std(ddof=1))
    metrics['Sharpe Ratio'] = (ex_mean / ex_std) * np.sqrt(periods_per_year) if ex_std > 0 else 0.0

    downside_excess = excess[excess < 0]
    de_std = float(downside_excess.std(ddof=1))
    metrics['Sortino Ratio'] = (ex_mean / de_std) * np.sqrt(periods_per_year) if de_std > 0 else 0.0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    metrics['Max Drawdown'] = float(drawdown.min())
    metrics['Avg Drawdown'] = float(drawdown[drawdown < 0].mean())

    metrics['Win Rate'] = float((returns > 0).mean())

    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(abs(returns[returns < 0].sum()))
    metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf

    metrics['Calmar Ratio'] = metrics['CAGR'] / abs(metrics['Max Drawdown']) if metrics['Max Drawdown'] < 0 else 0.0

    if benchmark_returns is not None:
        bench = benchmark_returns.dropna()
        common_idx = returns.index.intersection(bench.index)
        if len(common_idx) > 10:
            r = returns.loc[common_idx]
            b = bench.loc[common_idx]

            metrics['Market Correlation'] = float(r.corr(b))
            cov = np.cov(r, b)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            metrics['Beta'] = float(beta)

            bench_equity = (1.0 + b).cumprod()
            years_b = max((common_idx[-1] - common_idx[0]).days / 365.25, 1e-9)
            bench_cagr = float(bench_equity.iloc[-1] ** (1.0 / years_b) - 1.0)

            metrics['Alpha'] = float(cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate)))

    metrics['Daily VaR 95%'] = float(np.percentile(returns, 5))
    metrics['Expected Shortfall 95%'] = float(returns[returns <= metrics['Daily VaR 95%']].mean())
    metrics['Skewness'] = float(stats.skew(returns))
    metrics['Kurtosis'] = float(stats.kurtosis(returns))

    monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    metrics['Monthly Win Rate'] = float((monthly > 0).mean()) if len(monthly) else 0.0
    metrics['Positive Months %'] = metrics['Monthly Win Rate'] * 100.0

    return metrics


def plot_results(results: Dict, prices_df: pd.DataFrame):
    portfolio = results['combined']
    strategies = results['strategies']

    fig = plt.figure(figsize=(18, 14))

    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(portfolio.index, portfolio['total'], label='Multi-Strategy',
             linewidth=3, color='darkblue', alpha=0.9)

    colors = ['green', 'red', 'orange', 'purple']
    for idx, (strat_name, strat_data) in enumerate(strategies.items()):
        if 'equity' in strat_data.columns:
            ax1.plot(strat_data.index, strat_data['equity'],
                     label=strat_name.replace('_', ' ').title(),
                     alpha=0.6, linestyle='--', color=colors[idx % len(colors)])

    if 'SPY' in prices_df.columns:
        spy_returns = prices_df['SPY'].pct_change().fillna(0)
        spy_equity = 100000 * (1 + spy_returns).cumprod()
        ax1.plot(prices_df.index, spy_equity, label='SPY',
                 linewidth=2, color='gray', alpha=0.5)

    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    equity = portfolio['total']
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100
    ax2.fill_between(portfolio.index, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(portfolio.index, drawdown, color='red', linewidth=1)
    ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    monthly_returns = portfolio['return'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

    if len(monthly_returns) > 0:
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'returns': monthly_returns.values
        })

        heatmap_data = monthly_df.pivot(index='year', columns='month', values='returns')

        im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                        vmin=-10, vmax=10, interpolation='nearest')
        ax3.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month', fontsize=11)
        ax3.set_ylabel('Year', fontsize=11)
        plt.colorbar(im, ax=ax3)

    ax4 = plt.subplot(3, 3, 4)
    returns = portfolio['return'].dropna() * 100
    ax4.hist(returns, bins=50, alpha=0.7, color='steelblue',
             edgecolor='black', density=True)

    x = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = stats.norm.pdf(x, returns.mean(), returns.std())
    ax4.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal')

    ax4.axvline(returns.mean(), color='green', linestyle='--',
                label=f'Mean: {returns.mean():.2f}%')
    ax4.axvline(0, color='black', linestyle='-', alpha=0.3)

    ax4.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Daily Return (%)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    rolling_window = 126
    if len(portfolio['return']) > rolling_window:
        rolling_mean = portfolio['return'].rolling(rolling_window).mean()
        rolling_std = portfolio['return'].rolling(rolling_window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)

        ax5.plot(portfolio.index, rolling_sharpe, color='darkgreen', linewidth=2)
        ax5.axhline(rolling_sharpe.mean(), color='red', linestyle='--',
                    alpha=0.7, label=f'Avg: {rolling_sharpe.mean():.2f}')
        ax5.axhline(0, color='black', linestyle='-', alpha=0.3)

        ax5.set_title('Rolling 6-Month Sharpe Ratio', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio', fontsize=11)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)

    allocation_data = {}
    for strat_name, strat_df in strategies.items():
        if 'equity' in strat_df.columns:
            allocation_data[strat_name] = strat_df['equity']

    if allocation_data:
        allocation_df = pd.DataFrame(allocation_data)
        allocation_pct = allocation_df.div(allocation_df.sum(axis=1), axis=0) * 100

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax6.stackplot(allocation_pct.index, allocation_pct.T,
                      labels=[s.replace('_', ' ').title() for s in allocation_pct.columns],
                      colors=colors[:len(allocation_pct.columns)], alpha=0.8)

        ax6.set_title('Strategy Allocation (%)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Allocation %', fontsize=11)
        ax6.legend(loc='upper left', fontsize=9)
        ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)

    strat_returns = {}
    for strat_name, strat_df in strategies.items():
        if 'return' in strat_df.columns:
            strat_returns[strat_name] = strat_df['return'].dropna()

    if len(strat_returns) > 1:
        returns_df = pd.DataFrame(strat_returns)
        corr_matrix = returns_df.corr()

        im = ax7.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                        aspect='auto')
        ax7.set_title('Strategy Correlation Matrix', fontsize=14, fontweight='bold')

        tick_labels = [s.replace('_', '\n').title() for s in corr_matrix.columns]
        ax7.set_xticks(range(len(corr_matrix.columns)))
        ax7.set_yticks(range(len(corr_matrix.columns)))
        ax7.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
        ax7.set_yticklabels(tick_labels, fontsize=9)

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax7.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                         fontsize=8)

        plt.colorbar(im, ax=ax7)

    ax8 = plt.subplot(3, 3, 8)
    if 'positions' in results:
        positions = results['positions']
        active_positions = (positions.abs() > 0).sum(axis=1)
        ax8.plot(positions.index, active_positions, color='darkorange', linewidth=2)
        ax8.set_title('Active Statistical Arb Positions', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Number of Positions', fontsize=11)
        ax8.set_xlabel('Date', fontsize=11)
        ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    rolling_vol = portfolio['return'].rolling(21).std() * np.sqrt(252) * 100
    ax9.plot(portfolio.index, rolling_vol, color='purple', linewidth=2)
    ax9.axhline(rolling_vol.mean(), color='red', linestyle='--',
                alpha=0.7, label=f'Avg: {rolling_vol.mean():.1f}%')
    ax9.set_title('Rolling 1-Month Annualized Volatility', fontsize=14, fontweight='bold')
    ax9.set_ylabel('Volatility (%)', fontsize=11)
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    tickers = [
        'XLF', 'XLV', 'XLK', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB',
        'EFA', 'EEM',
        'TLT', 'IEF', 'SHY', 'LQD', 'BND', 'TIP',
        'GLD', 'SLV',
        'MNA',
        'VXX',
        'SPY', 'AGG',
    ]

    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    initial_capital = 100_000.0
    risk_free_rate = 0.02

    strategy_weights = {
        'stat_arb': 0.45,
        'merger_arb': 0.05,
        'risk_parity': 0.25,
        'covered_call': 0.25
    }

    wsum = sum(strategy_weights.values())
    if wsum <= 0:
        raise ValueError("strategy_weights must sum to a positive number.")
    if abs(wsum - 1.0) > 1e-9:
        strategy_weights = {k: v / wsum for k, v in strategy_weights.items()}

    print("=" * 80)
    print("MULTI-STRATEGY PORTFOLIO")
    print("=" * 80)
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Number of Assets (requested): {len(tickers)}")
    print("=" * 80)

    data = download_data(tickers, start_date, end_date)

    if len(data) < 10:
        print(f"\nWarning: Only {len(data)} assets downloaded. Strategy may be limited.")

    prices_list = []
    for ticker, df in data.items():
        if (df is not None) and (not df.empty) and ('Adj Close' in df.columns):
            prices_list.append(pd.DataFrame({ticker: df['Adj Close']}))

    if not prices_list:
        print("No valid price data available")
        return

    prices_df = pd.concat(prices_list, axis=1)

    prices_df = prices_df.ffill(limit=5).dropna()

    if len(prices_df) < 252:
        print(f"Warning: Only {len(prices_df)} trading days available. Need at least 252.")
        return

    print(f"\nData range (prices_df): {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    print(f"Trading days: {len(prices_df)}")
    print(f"Available assets: {len(prices_df.columns)}")

    results = multi_strategy_portfolio(
        prices_df=prices_df,
        initial_capital=initial_capital,
        strategy_weights=strategy_weights
    )

    benchmark_returns = None
    spy_portfolio = None
    if 'SPY' in prices_df.columns:
        spy_px = prices_df['SPY'].reindex(results['combined'].index).dropna()
        spy_rets = spy_px.pct_change().dropna()

        benchmark_returns = spy_rets

        spy_equity = initial_capital * (1.0 + spy_rets).cumprod()
        spy_portfolio = pd.DataFrame(index=spy_equity.index)
        spy_portfolio['equity'] = spy_equity
        spy_portfolio['return'] = spy_rets

        common_idx = spy_portfolio.index
        print(f"\nSPY window used: {common_idx[0].date()} to {common_idx[-1].date()}  (days={len(common_idx)})")

    metrics = calculate_performance_metrics(
        results['combined'],
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    spy_metrics = {}
    if spy_portfolio is not None and len(spy_portfolio) > 10:
        spy_metrics = calculate_performance_metrics(
            spy_portfolio,
            benchmark_returns=None,
            risk_free_rate=risk_free_rate
        )

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS (Annualized)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 80)

    portfolio_metrics = {
        'CAGR': metrics.get('CAGR', 0),
        'Annual Vol': metrics.get('Annual Volatility', 0),
        'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
        'Sortino Ratio': metrics.get('Sortino Ratio', 0),
        'Max Drawdown': metrics.get('Max Drawdown', 0),
        'Win Rate': metrics.get('Win Rate', 0),
        'Market Correlation': metrics.get('Market Correlation', 0),
        'Beta': metrics.get('Beta', 0),
    }

    spy_metrics_print = {
        'CAGR': spy_metrics.get('CAGR', 0),
        'Annual Vol': spy_metrics.get('Annual Volatility', 0),
        'Sharpe Ratio': spy_metrics.get('Sharpe Ratio', 0),
        'Sortino Ratio': spy_metrics.get('Sortino Ratio', 0),
        'Max Drawdown': spy_metrics.get('Max Drawdown', 0),
        'Win Rate': spy_metrics.get('Win Rate', 0),
        'Market Correlation': 1.0 if spy_portfolio is not None else 0,
        'Beta': 1.0 if spy_portfolio is not None else 0,
    }

    for metric, port_val in portfolio_metrics.items():
        spy_val = spy_metrics_print.get(metric, 0)

        if metric in ['CAGR', 'Annual Vol', 'Max Drawdown']:
            port_fmt = f"{port_val:.2%}" if port_val is not None else "N/A"
            spy_fmt = f"{spy_val:.2%}" if spy_val is not None else "N/A"
        elif metric in ['Sharpe Ratio', 'Sortino Ratio', 'Beta']:
            port_fmt = f"{port_val:.3f}" if port_val is not None else "N/A"
            spy_fmt = f"{spy_val:.3f}" if spy_val is not None else "N/A"
        elif metric in ['Win Rate']:
            port_fmt = f"{port_val:.1%}" if port_val is not None else "N/A"
            spy_fmt = f"{spy_val:.1%}" if spy_val is not None else "N/A"
        elif metric in ['Market Correlation']:
            port_fmt = f"{port_val:.3f}" if port_val is not None else "N/A"
            spy_fmt = f"{spy_val:.3f}" if spy_val is not None else "N/A"
        else:
            port_fmt = f"{port_val:.3f}" if port_val is not None else "N/A"
            spy_fmt = f"{spy_val:.3f}" if spy_val is not None else "N/A"

        print(f"{metric:<25} {port_fmt:>12} {spy_fmt:>12}")

    print("-" * 80)

    print(f"{'Positive Months':<25} {metrics.get('Positive Months %', 0):>12.1f}%")
    print(f"{'Profit Factor':<25} {metrics.get('Profit Factor', 0):>12.2f}")
    print(f"{'Calmar Ratio':<25} {metrics.get('Calmar Ratio', 0):>12.2f}")
    print(f"{'Alpha':<25} {metrics.get('Alpha', 0):>12.2%}")

    print("=" * 80)

    print("\nSTRATEGY CORRELATIONS:")
    print("-" * 40)

    strat_returns = {}
    for strat_name, strat_df in results['strategies'].items():
        if isinstance(strat_df, pd.DataFrame) and ('return' in strat_df.columns):
            r = strat_df['return'].dropna()
            if len(r) > 2:
                strat_returns[strat_name] = r

    if len(strat_returns) > 1:
        corr_df = pd.DataFrame(strat_returns).dropna()
        if not corr_df.empty and corr_df.shape[1] > 1:
            corr_matrix = corr_df.corr()

            strat_names = [s.replace('_', ' ').title() for s in corr_matrix.columns]
            print(" " * 15, end="")
            for name in strat_names:
                print(f"{name[:10]:>12}", end="")
            print()

            for i, row_name in enumerate(strat_names):
                print(f"{row_name[:15]:<15}", end="")
                for j in range(len(corr_matrix.columns)):
                    print(f"{corr_matrix.iloc[i, j]:>12.3f}", end="")
                print()

    print("\nRISK METRICS:")
    print("-" * 40)
    print(f"Daily VaR 95%: {metrics.get('Daily VaR 95%', 0):.2%}")
    print(f"Expected Shortfall 95%: {metrics.get('Expected Shortfall 95%', 0):.2%}")
    print(f"Skewness: {metrics.get('Skewness', 0):.3f}")
    print(f"Kurtosis: {metrics.get('Kurtosis', 0):.3f}")

    print("\nGenerating performance charts...")
    plot_results(results, prices_df)


if __name__ == "__main__":
    main()