import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data using yfinance."""
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]

    return df


def compute_indicators(
    df: pd.DataFrame,
    sma_long: int = 200,
    mom_lookback: int = 252,   # ~12m
    mom_skip: int = 21,        # skip last month (12-1 momentum)
    vol_lookback: int = 20,    # for risk parity sizing
) -> pd.DataFrame:
    df = df.copy()
    close = df["Adj Close"]

    # Long trend filter
    df["SMA_200"] = close.rolling(sma_long, min_periods=sma_long // 2).mean()
    df["trend_ok"] = (close > df["SMA_200"]).astype(int)

    # 12-1 momentum (classic, avoids short-term mean reversion)
    df["mom_12_1"] = close.pct_change(mom_lookback - mom_skip)

    # Volatility estimate for sizing
    df["vol_ann"] = close.pct_change().rolling(vol_lookback).std() * np.sqrt(252)

    # ATR optional (kept from your old code idea)
    high = df["High"]
    low = df["Low"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=7).mean()
    df["ATR_pct"] = df["ATR"] / close

    return df



def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # “Signal” here just means eligibility + score; actual portfolio rules decide weights.
    df["eligible"] = df["trend_ok"]  # 1 if above SMA_200 else 0
    df["score"] = df["mom_12_1"]     # cross-sectional rank uses this

    return df



def download_all_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download and process data for all tickers."""
    data = {}
    print(f"\nDownloading data for {len(tickers)} assets...")

    for ticker in tickers:
        try:
            print(f"  {ticker}...", end=" ")
            df = download_data(ticker, start, end)
            df = compute_indicators(df)
            df = generate_signals(df)
            data[ticker] = df
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    return data


def align_data(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_dates = None
    for _, df in data.items():
        all_dates = set(df.index) if all_dates is None else all_dates.intersection(set(df.index))
    common_dates = sorted(list(all_dates))

    prices, rets = {}, {}
    sig_cols = []

    for ticker, df in data.items():
        df_aligned = df.loc[common_dates]
        prices[ticker] = df_aligned["Adj Close"]
        rets[ticker] = df_aligned["Adj Close"].pct_change().fillna(0.0)

        # multi-field signal frame
        sig_cols.append(((ticker, "eligible"), df_aligned["eligible"]))
        sig_cols.append(((ticker, "score"), df_aligned["score"]))

    prices_df = pd.DataFrame(prices, index=common_dates)
    returns_df = pd.DataFrame(rets, index=common_dates)

    signals_df = pd.concat([s for _, s in sig_cols], axis=1)
    signals_df.columns = pd.MultiIndex.from_tuples([c for c, _ in sig_cols], names=["ticker", "field"])

    return prices_df, signals_df, returns_df



def calculate_correlations(returns_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling correlations between assets."""
    return returns_df.rolling(window).corr()


def unified_portfolio_strategy(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,   # expects columns like "<ticker>_eligible" etc? see note below
    returns_df: pd.DataFrame,
    risk_assets: List[str],
    defensive_assets: List[str],
    regime_ticker: str = "SPY",
    initial_capital: float = 100_000.0,
    trading_fee_bp: float = 5.0,
    target_volatility: float = 0.10,   # lower vol target
    max_gross_leverage: float = 1.0,   # long-only, no leverage by default
    top_n: int = 4,
    rebalance_weekday: int = 0,        # Monday=0
    vol_lookback: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Long-only trend + momentum rotation with regime filter and vol targeting.

    Requires signals_df to contain for each ticker:
      - eligible: 0/1
      - score: momentum value
    We'll pass those in as MultiIndex columns: (ticker, field).
    """

    dates = prices_df.index
    tickers = list(prices_df.columns)

    # sanity
    for t in risk_assets + defensive_assets + [regime_ticker]:
        if t not in tickers:
            raise ValueError(f"{t} not found in prices/returns universe.")

    # Rolling asset vol for inverse-vol weights
    asset_vol = returns_df.rolling(vol_lookback).std() * np.sqrt(252)

    portfolio = pd.DataFrame(index=dates)
    portfolio["equity"] = initial_capital
    positions = pd.DataFrame(0.0, index=dates, columns=tickers)

    def is_rebalance_day(dt) -> bool:
        # Weekly rebalance on chosen weekday; always rebalance first valid day.
        return dt.weekday() == rebalance_weekday

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        prev_pos = positions.loc[prev_date].copy()

        # default: carry positions unless rebalance day
        new_pos = prev_pos.copy()

        if is_rebalance_day(date):
            # --- regime filter based on regime_ticker eligibility ---
            regime_eligible = signals_df.loc[date, (regime_ticker, "eligible")] == 1

            if regime_eligible:
                # RISK-ON: pick top N eligible risk assets by momentum score
                eligible = []
                for t in risk_assets:
                    if signals_df.loc[date, (t, "eligible")] == 1:
                        eligible.append(t)

                if len(eligible) > 0:
                    scores = signals_df.loc[date, [(t, "score") for t in eligible]]
                    scores.index = [x[0] for x in scores.index]  # flatten
                    top = scores.sort_values(ascending=False).head(top_n).index.tolist()
                else:
                    top = []

                # inverse-vol weights among top picks
                weights = pd.Series(0.0, index=tickers)
                if len(top) > 0:
                    vols = asset_vol.loc[prev_date, top].replace(0, np.nan)
                    inv = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if inv.sum() > 0:
                        w_top = inv / inv.sum()
                        weights.loc[top] = w_top

                # small ballast in defensive assets (optional; keep it tiny in risk-on)
                # weights.loc[defensive_assets] = 0.0

                new_pos = weights

            else:
                # RISK-OFF: allocate to defensive assets that are eligible; otherwise to the “least bad”
                eligible_def = []
                for t in defensive_assets:
                    if signals_df.loc[date, (t, "eligible")] == 1:
                        eligible_def.append(t)

                weights = pd.Series(0.0, index=tickers)

                if len(eligible_def) > 0:
                    vols = asset_vol.loc[prev_date, eligible_def].replace(0, np.nan)
                    inv = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if inv.sum() > 0:
                        w_def = inv / inv.sum()
                        weights.loc[eligible_def] = w_def
                else:
                    # if nothing is eligible, go to “cash proxy”: IEF if present else TLT else stay flat
                    if "IEF" in tickers:
                        weights.loc["IEF"] = 1.0
                    elif "TLT" in tickers:
                        weights.loc["TLT"] = 1.0
                    else:
                        weights[:] = 0.0

                new_pos = weights

            # --- Vol targeting (scale gross exposure to target_volatility) ---
            # Estimate portfolio vol using last vol_lookback returns with the new weights.
            # (cheap approximation: sqrt(w^2 * vol^2) ignoring correlations)
            vols = asset_vol.loc[prev_date].fillna(0.0)
            est_port_vol = np.sqrt(((new_pos ** 2) * (vols ** 2)).sum())

            if est_port_vol > 1e-8:
                scale = target_volatility / est_port_vol
                scale = float(np.clip(scale, 0.0, max_gross_leverage))
                new_pos = new_pos * scale
            else:
                new_pos = new_pos * 0.0

            # final cap (safety)
            gross = new_pos.abs().sum()
            if gross > max_gross_leverage and gross > 1e-8:
                new_pos = new_pos * (max_gross_leverage / gross)

        # trading cost on turnover
        turnover = (new_pos - prev_pos).abs().sum()
        trading_cost = turnover * (trading_fee_bp / 10000.0)

        # portfolio return uses prev positions (entered at close prev day)
        r = (prev_pos * returns_df.loc[date]).sum() - trading_cost

        portfolio.loc[date, "return"] = r
        portfolio.loc[date, "equity"] = portfolio.loc[prev_date, "equity"] * (1 + r)
        portfolio.loc[date, "gross_exposure"] = prev_pos.abs().sum()
        portfolio.loc[date, "net_exposure"] = prev_pos.sum()
        portfolio.loc[date, "num_positions"] = (prev_pos.abs() > 0.01).sum()

        positions.loc[date] = new_pos

    if "SPY" in returns_df.columns:
        portfolio["spy_equity"] = initial_capital * (1 + returns_df["SPY"]).cumprod()

    return portfolio, positions



def performance_stats(portfolio: pd.DataFrame) -> Dict:
    """Calculate portfolio performance statistics."""
    stats = {}

    returns = portfolio["return"].dropna()
    equity = portfolio["equity"]

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    stats["Total Return %"] = total_return

    # CAGR
    n_years = len(equity) / 252
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100
    stats["CAGR %"] = cagr

    # Volatility
    annual_vol = returns.std() * np.sqrt(252) * 100
    stats["Annual Volatility %"] = annual_vol

    # Sharpe ratio
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    stats["Sharpe Ratio"] = sharpe

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100
    stats["Max Drawdown %"] = drawdown.min()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100
    stats["Win Rate %"] = win_rate

    # Calmar ratio
    calmar = cagr / abs(drawdown.min()) if drawdown.min() < 0 else 0
    stats["Calmar Ratio"] = calmar

    # Average exposures
    stats["Avg Gross Exposure %"] = portfolio["gross_exposure"].mean() * 100
    stats["Avg Net Exposure %"] = portfolio["net_exposure"].mean() * 100
    stats["Avg Num Positions"] = portfolio["num_positions"].mean()

    return stats


def plot_results(portfolio: pd.DataFrame, positions: pd.DataFrame):
    """Plot portfolio performance and positions."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # 1. Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(portfolio.index, portfolio["equity"], label="Unified Multi-Asset Strategy",
             linewidth=2.5, color='darkblue')
    if "spy_equity" in portfolio.columns:
        ax1.plot(portfolio.index, portfolio["spy_equity"], label="SPY Buy & Hold",
                 linewidth=2.5, linestyle='--', color='red', alpha=0.7)
    ax1.set_title("Portfolio Equity: Unified Multi-Asset Strategy vs S&P 500", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Equity ($)")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    equity = portfolio["equity"]
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100
    ax2.fill_between(portfolio.index, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(portfolio.index, drawdown, color='red', linewidth=1)
    ax2.set_title("Portfolio Drawdown", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # 3. Gross & Net Exposure
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(portfolio.index, portfolio["gross_exposure"] * 100, label="Gross Exposure", linewidth=1.5)
    ax3.plot(portfolio.index, portfolio["net_exposure"] * 100, label="Net Exposure", linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title("Portfolio Exposure", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Exposure (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Number of positions
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(portfolio.index, portfolio["num_positions"], linewidth=1.5, color='green')
    ax4.set_title("Number of Active Positions", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Count")
    ax4.grid(True, alpha=0.3)

    # 5. Position heatmap (sample recent period)
    ax5 = fig.add_subplot(gs[3, :])
    recent_positions = positions.iloc[-252:].T  # Last year
    im = ax5.imshow(recent_positions, aspect='auto', cmap='RdYlGn', vmin=-0.25, vmax=0.25)
    ax5.set_title("Position Heatmap (Last 252 Days)", fontsize=12, fontweight='bold')
    ax5.set_ylabel("Assets")
    ax5.set_yticks(range(len(recent_positions)))
    ax5.set_yticklabels(recent_positions.index, fontsize=8)
    plt.colorbar(im, ax=ax5, label="Position Weight")

    plt.tight_layout()
    plt.show()


def main():
    # === CONFIG ===
    tickers = [
        # Equities
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq
        "IWM",  # Russell 2000
        "EFA",  # International Developed
        "EEM",  # Emerging Markets

        # Sectors
        "XLF",  # Financials
        "XLE",  # Energy
        "XLK",  # Technology
        "XLV",  # Healthcare

        # Bonds (defensive)
        "TLT",  # Long-term Treasury
        "IEF",  # Mid-term Treasury

        # Commodities (defensive diversifiers)
        "GLD",  # Gold
        "SLV",  # Silver
    ]

    # Universe split for the new strategy
    risk_assets = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLF", "XLE", "XLK", "XLV"]
    defensive_assets = ["TLT", "IEF", "GLD", "SLV"]

    # Backtest range
    start = "2015-01-01"
    end = datetime.today().strftime("%Y-%m-%d")
    initial_capital = 100_000.0

    # Download data + indicators + signals (new versions)
    data = download_all_data(tickers, start, end)

    if len(data) < 3:
        print("\nInsufficient data downloaded. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(data)} assets.")

    # Align data (new MultiIndex signals_df)
    print("\nAligning data across common dates...")
    prices_df, signals_df, returns_df = align_data(data)
    print(f"Common date range: {prices_df.index[0]} to {prices_df.index[-1]}")
    print(f"Total trading days: {len(prices_df)}")

    # Run new unified strategy v2 (long-only, regime + momentum, vol targeting)
    print("\nRunning unified portfolio strategy v2 (regime + momentum + vol targeting)...")
    portfolio, positions = unified_portfolio_strategy(
        prices_df=prices_df,
        signals_df=signals_df,
        returns_df=returns_df,
        risk_assets=risk_assets,
        defensive_assets=defensive_assets,
        regime_ticker="SPY",
        initial_capital=initial_capital,
        trading_fee_bp=5.0,
        target_volatility=0.10,     # smoother than 0.15
        max_gross_leverage=1.0,     # long-only, no leverage
        top_n=4,                    # rotate into top 4 risk assets in risk-on
        rebalance_weekday=0,        # Monday rebalance
        vol_lookback=20,
    )

    # Calculate stats
    print("\nCalculating performance statistics...")
    stats = performance_stats(portfolio)

    # Compare to SPY
    spy_stats = {}
    if "SPY" in returns_df.columns and "spy_equity" in portfolio.columns:
        spy_equity = portfolio["spy_equity"]
        spy_returns = returns_df["SPY"].dropna()

        spy_stats["Total Return %"] = (spy_equity.iloc[-1] / spy_equity.iloc[0] - 1) * 100
        n_years = len(spy_equity) / 252
        spy_stats["CAGR %"] = ((spy_equity.iloc[-1] / spy_equity.iloc[0]) ** (1 / n_years) - 1) * 100
        spy_stats["Annual Volatility %"] = spy_returns.std() * np.sqrt(252) * 100
        spy_stats["Sharpe Ratio"] = (spy_returns.mean() / spy_returns.std()) * np.sqrt(252) if spy_returns.std() > 0 else 0

        spy_cummax = spy_equity.cummax()
        spy_dd = ((spy_equity / spy_cummax - 1) * 100).min()
        spy_stats["Max Drawdown %"] = spy_dd
        spy_stats["Calmar Ratio"] = spy_stats["CAGR %"] / abs(spy_dd) if spy_dd < 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("UNIFIED MULTI-ASSET STRATEGY V2 PERFORMANCE")
    print("=" * 80)
    print(f"{'Metric':<30s} {'Strategy':>15s} {'SPY':>15s} {'Difference':>15s}")
    print("-" * 80)

    compare_metrics = ["Total Return %", "CAGR %", "Annual Volatility %",
                       "Sharpe Ratio", "Max Drawdown %", "Calmar Ratio"]

    for metric in compare_metrics:
        strat_val = stats.get(metric, np.nan)
        spy_val = spy_stats.get(metric, np.nan)
        diff = strat_val - spy_val if not np.isnan(spy_val) else np.nan

        print(f"{metric:<30s} {strat_val:>15.2f} {spy_val:>15.2f} {diff:>15.2f}")

    print("-" * 80)
    print(f"{'Avg Gross Exposure %':<30s} {stats['Avg Gross Exposure %']:>15.2f}")
    print(f"{'Avg Net Exposure %':<30s} {stats['Avg Net Exposure %']:>15.2f}")
    print(f"{'Avg Num Positions':<30s} {stats['Avg Num Positions']:>15.2f}")
    print("=" * 80)

    # Plot results
    print("\nGenerating plots...")
    plot_results(portfolio, positions)



if __name__ == "__main__":
    main()