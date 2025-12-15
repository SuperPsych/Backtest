import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def download_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download data for multiple tickers efficiently."""
    print(f"Downloading data for {len(tickers)} assets...")

    # Download in batches to avoid timeout
    batch_size = 10
    all_data = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)

            # Handle single ticker vs multiple tickers format
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

    # Process each dataframe
    processed = {}
    for ticker, df in all_data.items():
        try:
            if df.empty:
                continue

            # Standardize column names
            if 'Adj Close' in df.columns:
                df['Adj Close'] = df['Adj Close']
            elif 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            else:
                continue

            # Ensure we have all required columns
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

            # Get common non-NaN data
            series1 = prices_df[ticker1].dropna()
            series2 = prices_df[ticker2].dropna()
            common_idx = series1.index.intersection(series2.index)

            if len(common_idx) < 100:  # Need sufficient data
                continue

            try:
                score, pvalue, _ = coint(series1.loc[common_idx], series2.loc[common_idx])
                if pvalue < pvalue_threshold:
                    # Calculate optimal hedge ratio using OLS
                    X = series1.loc[common_idx].values.reshape(-1, 1)
                    y = series2.loc[common_idx].values
                    hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

                    pairs.append((ticker1, ticker2, pvalue, hedge_ratio, score))
            except Exception as e:
                continue

    # Sort by p-value (most significant first)
    pairs.sort(key=lambda x: x[2])
    print(f"  Found {len(pairs)} cointegrated pairs")

    # Return top pairs with hedge ratio
    return [(p[0], p[1], p[3]) for p in pairs[:20]]


def statistical_arbitrage_strategy(prices_df: pd.DataFrame,
                                   initial_capital: float = 100_000.0,
                                   target_volatility: float = 0.10,
                                   max_position_pct: float = 0.10,
                                   entry_z: float = 2.0,
                                   exit_z: float = 0.5,
                                   stop_loss_z: float = 3.0,
                                   lookback_days: int = 60,
                                   rebalance_days: int = 5,
                                   max_active_pairs: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Statistical arbitrage strategy focusing on mean reversion of cointegrated pairs.
    """

    print("Finding cointegrated pairs...")
    pairs = find_cointegrated_pairs(prices_df)

    if len(pairs) == 0:
        print("Warning: No cointegrated pairs found. Using sector pairs as fallback.")
        # Fallback to sector pairs
        sector_pairs = [
            ('XLF', 'XLK', 1.0),  # Financials vs Tech
            ('XLE', 'XLU', 1.0),  # Energy vs Utilities
            ('XLV', 'XLP', 1.0),  # Healthcare vs Consumer Staples
            ('XLY', 'XLI', 1.0),  # Consumer Discretionary vs Industrials
        ]
        pairs = [(p[0], p[1], p[2]) for p in sector_pairs
                 if p[0] in prices_df.columns and p[1] in prices_df.columns]

    dates = prices_df.index
    tickers = prices_df.columns.tolist()

    # Initialize portfolio
    portfolio = pd.DataFrame(index=dates)
    portfolio['equity'] = initial_capital
    portfolio['cash'] = initial_capital
    portfolio['gross_exposure'] = 0.0
    portfolio['net_exposure'] = 0.0
    portfolio['active_pairs'] = 0

    positions = pd.DataFrame(0.0, index=dates, columns=tickers)

    # Track active pairs
    active_pairs = {}

    for i in range(lookback_days, len(dates)):
        current_date = dates[i]
        prev_date = dates[i - 1]

        # Rebalance logic
        if i % rebalance_days == 0:
            # Close pairs that have been open too long (60 days max)
            pairs_to_close = []
            for pair_key, pair_info in list(active_pairs.items()):
                days_open = (current_date - pair_info['entry_date']).days
                if days_open > 60:
                    pairs_to_close.append(pair_key)

            for pair_key in pairs_to_close:
                t1, t2 = pair_key.split('_')

                # Close positions
                pos1 = positions.at[prev_date, t1]
                pos2 = positions.at[prev_date, t2]

                # Update cash (selling positions)
                portfolio.at[current_date, 'cash'] = portfolio.at[prev_date, 'cash'] + (
                        abs(pos1) * prices_df.at[current_date, t1] +
                        abs(pos2) * prices_df.at[current_date, t2]
                )

                positions.at[current_date, t1] = 0
                positions.at[current_date, t2] = 0
                del active_pairs[pair_key]

            # Check for new entries if we have capacity
            if len(active_pairs) < max_active_pairs:
                for pair_idx, (t1, t2, hedge_ratio) in enumerate(pairs):
                    if len(active_pairs) >= max_active_pairs:
                        break

                    pair_key = f"{t1}_{t2}"
                    if pair_key in active_pairs:
                        continue

                    # Calculate spread z-score
                    lookback_data = prices_df.iloc[max(0, i - lookback_days):i]
                    if len(lookback_data) < 30:
                        continue

                    spread = lookback_data[t1] - hedge_ratio * lookback_data[t2]
                    current_spread = spread.iloc[-1]
                    spread_mean = spread.mean()
                    spread_std = spread.std()

                    if spread_std == 0 or pd.isna(spread_std):
                        continue

                    zscore = (current_spread - spread_mean) / spread_std

                    # Entry signal
                    if abs(zscore) > entry_z:
                        # Determine trade direction
                        if zscore > entry_z:
                            # Spread is wide: short t1, long t2
                            size_t1 = -max_position_pct * initial_capital / prices_df.at[current_date, t1]
                            size_t2 = (max_position_pct * initial_capital * abs(hedge_ratio)) / prices_df.at[
                                current_date, t2]
                        else:  # zscore < -entry_z
                            # Spread is narrow: long t1, short t2
                            size_t1 = max_position_pct * initial_capital / prices_df.at[current_date, t1]
                            size_t2 = (-max_position_pct * initial_capital * abs(hedge_ratio)) / prices_df.at[
                                current_date, t2]

                        # Check margin/cash availability
                        required_cash = abs(size_t1 * prices_df.at[current_date, t1]) + abs(
                            size_t2 * prices_df.at[current_date, t2])
                        if required_cash <= portfolio.at[prev_date, 'cash'] * 0.8:  # Use 80% of cash
                            positions.at[current_date, t1] = size_t1
                            positions.at[current_date, t2] = size_t2
                            active_pairs[pair_key] = {
                                'entry_z': zscore,
                                'hedge_ratio': hedge_ratio,
                                'entry_date': current_date,
                                'direction': 'short_t1_long_t2' if zscore > 0 else 'long_t1_short_t2'
                            }

                            # Update cash (buying positions)
                            portfolio.at[current_date, 'cash'] = portfolio.at[prev_date, 'cash'] - required_cash

        # Check exit conditions for active pairs
        for pair_key, pair_info in list(active_pairs.items()):
            t1, t2 = pair_key.split('_')

            # Calculate current z-score
            lookback_data = prices_df.iloc[max(0, i - lookback_days):i]
            spread = lookback_data[t1] - pair_info['hedge_ratio'] * lookback_data[t2]
            current_spread = spread.iloc[-1]
            spread_mean = spread.mean()
            spread_std = spread.std()

            if spread_std == 0:
                continue

            current_z = (current_spread - spread_mean) / spread_std

            # Exit conditions
            exit_signal = False

            # 1. Mean reversion (spread returned to mean)
            if abs(current_z) < exit_z:
                exit_signal = True

            # 2. Stop loss
            elif abs(current_z) > stop_loss_z:
                exit_signal = True

            # 3. Direction changed (spread crossed zero)
            elif (current_z > 0 and pair_info['entry_z'] < 0) or (current_z < 0 and pair_info['entry_z'] > 0):
                exit_signal = True

            if exit_signal:
                # Close position
                pos1 = positions.at[prev_date, t1]
                pos2 = positions.at[prev_date, t2]

                # Update cash (selling positions)
                portfolio.at[current_date, 'cash'] = portfolio.at[prev_date, 'cash'] + (
                        abs(pos1) * prices_df.at[current_date, t1] +
                        abs(pos2) * prices_df.at[current_date, t2]
                )

                positions.at[current_date, t1] = 0
                positions.at[current_date, t2] = 0
                del active_pairs[pair_key]

        # Carry forward unchanged positions
        for col in positions.columns:
            if pd.isna(positions.at[current_date, col]):
                positions.at[current_date, col] = positions.at[prev_date, col]

        # Calculate portfolio value
        position_values = []
        for ticker in tickers:
            pos = positions.at[current_date, ticker]
            if pos != 0:
                value = pos * prices_df.at[current_date, ticker]
                position_values.append(value)

        total_position_value = sum(position_values) if position_values else 0
        portfolio.at[current_date, 'equity'] = portfolio.at[current_date, 'cash'] + total_position_value

        # Calculate exposures
        long_exposure = sum([v for v in position_values if v > 0])
        short_exposure = sum([abs(v) for v in position_values if v < 0])
        total_exposure = long_exposure + short_exposure

        if portfolio.at[current_date, 'equity'] > 0:
            portfolio.at[current_date, 'gross_exposure'] = total_exposure / portfolio.at[current_date, 'equity']
            portfolio.at[current_date, 'net_exposure'] = (long_exposure - short_exposure) / portfolio.at[
                current_date, 'equity']

        portfolio.at[current_date, 'active_pairs'] = len(active_pairs)

        # Calculate daily return
        if i > lookback_days:
            prev_equity = portfolio.at[prev_date, 'equity']
            curr_equity = portfolio.at[current_date, 'equity']
            if prev_equity > 0:
                portfolio.at[current_date, 'return'] = (curr_equity - prev_equity) / prev_equity

    # Fill forward any missing equity values
    portfolio['equity'].ffill(inplace=True)
    portfolio['return'].fillna(0, inplace=True)

    return portfolio, positions


def merger_arbitrage_strategy(prices_df: pd.DataFrame,
                              merger_etfs: List[str] = ['MNA', 'MRGR'],
                              initial_capital: float = 100_000.0) -> pd.DataFrame:
    """
    Merger arbitrage strategy using actual merger arbitrage ETFs.
    """

    # Try to get actual merger arbitrage ETFs
    available_etfs = [etf for etf in merger_etfs if etf in prices_df.columns]

    if not available_etfs:
        print("No merger arbitrage ETFs available. Using low-volatility blend.")
        # Create synthetic low-volatility portfolio
        low_vol_assets = ['SHY', 'IEI', 'TIP', 'BIL']
        available_etfs = [a for a in low_vol_assets if a in prices_df.columns]

    if not available_etfs:
        # Last resort: use cash
        portfolio = pd.DataFrame(index=prices_df.index)
        portfolio['equity'] = initial_capital
        portfolio['return'] = 0.0
        return portfolio

    # Calculate equal-weighted portfolio of available ETFs
    portfolio = pd.DataFrame(index=prices_df.index)

    # Get returns for each ETF
    returns = {}
    for etf in available_etfs:
        returns[etf] = prices_df[etf].pct_change().fillna(0)

    # Equal weight returns
    etf_returns = pd.DataFrame(returns)
    portfolio_return = etf_returns.mean(axis=1)

    # Calculate equity
    portfolio['return'] = portfolio_return
    portfolio['equity'] = initial_capital * (1 + portfolio_return).cumprod()

    # Smooth returns (merger arb should be low volatility)
    portfolio['return'] = portfolio['return'].rolling(5).mean().fillna(portfolio['return'])

    return portfolio


def risk_parity_strategy(prices_df: pd.DataFrame,
                         rp_assets: List[str] = ['TLT', 'GLD', 'TIP', 'IEF', 'LQD'],
                         initial_capital: float = 100_000.0,
                         lookback: int = 63) -> pd.DataFrame:
    """
    Risk parity strategy using inverse volatility weighting.
    """

    available_assets = [asset for asset in rp_assets if asset in prices_df.columns]

    if len(available_assets) < 3:
        print(f"Warning: Only {len(available_assets)} risk parity assets available")
        if len(available_assets) == 0:
            portfolio = pd.DataFrame(index=prices_df.index)
            portfolio['equity'] = initial_capital
            portfolio['return'] = 0.0
            return portfolio

    # Get returns
    returns_df = pd.DataFrame()
    for asset in available_assets:
        returns_df[asset] = prices_df[asset].pct_change().fillna(0)

    portfolio = pd.DataFrame(index=returns_df.index)

    # Calculate dynamic weights based on inverse volatility
    weights_history = []

    for i in range(lookback, len(returns_df)):
        current_date = returns_df.index[i]

        # Calculate rolling volatility
        rolling_returns = returns_df.iloc[i - lookback:i]
        volatilities = rolling_returns.std() * np.sqrt(252)

        # Avoid division by zero
        volatilities = volatilities.replace(0, np.nan)

        if volatilities.isna().all():
            # Equal weight if all volatilities are NaN
            weights = pd.Series(1 / len(available_assets), index=available_assets)
        else:
            # Inverse volatility weighting
            inv_vol = 1 / volatilities
            weights = inv_vol / inv_vol.sum()

        weights_history.append(weights)

    # Create weights DataFrame
    if weights_history:
        weights_df = pd.DataFrame(weights_history, index=returns_df.index[lookback:])
        weights_df = weights_df.reindex(returns_df.index).ffill().bfill()
    else:
        # Equal weights as fallback
        weights_df = pd.DataFrame(1 / len(available_assets),
                                  index=returns_df.index,
                                  columns=available_assets)

    # Calculate portfolio returns
    portfolio_return = (returns_df * weights_df).sum(axis=1)
    portfolio['return'] = portfolio_return
    portfolio['equity'] = initial_capital * (1 + portfolio_return).cumprod()

    return portfolio


def covered_call_strategy(prices_df: pd.DataFrame,
                          underlying: str = 'SPY',
                          initial_capital: float = 100_000.0) -> pd.DataFrame:
    """
    Simulated covered call strategy for income generation.
    This is a simplified simulation assuming:
    - Own 100 shares of SPY
    - Sell monthly at-the-money calls
    - Collect premium
    """

    if underlying not in prices_df.columns:
        portfolio = pd.DataFrame(index=prices_df.index)
        portfolio['equity'] = initial_capital
        portfolio['return'] = 0.0
        return portfolio

    portfolio = pd.DataFrame(index=prices_df.index)

    # Get underlying returns
    underlying_returns = prices_df[underlying].pct_change().fillna(0)

    # Simulate covered call returns:
    # Base return = underlying return
    # Plus monthly premium income ~0.5-1.5% per month
    # Minus capped upside (call premium limits gains)

    monthly_dates = pd.date_range(start=prices_df.index[0],
                                  end=prices_df.index[-1],
                                  freq='MS')

    # Initialize
    portfolio_return = underlying_returns.copy() * 0.7  # Reduced beta due to calls

    # Add premium income (simplified)
    for i in range(len(monthly_dates)):
        month_start = monthly_dates[i]
        if month_start in portfolio_return.index:
            # Add premium income (annualized ~8-12%)
            premium = 0.008  # 0.8% per month ~ 9.6% annualized
            portfolio_return.loc[month_start] = portfolio_return.loc[month_start] + premium

    # Reduce volatility (covered calls reduce risk)
    portfolio_return = portfolio_return * 0.8

    # Calculate equity
    portfolio['return'] = portfolio_return.fillna(0)
    portfolio['equity'] = initial_capital * (1 + portfolio['return']).cumprod()

    return portfolio


def multi_strategy_portfolio(prices_df: pd.DataFrame,
                             initial_capital: float = 100_000.0,
                             strategy_weights: Dict[str, float] = None) -> Dict:
    """
    Combine multiple uncorrelated strategies.
    """

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

    # 1. Statistical Arbitrage
    print("\n1. Statistical Arbitrage Strategy")
    if strategy_weights.get('stat_arb', 0) > 0:
        print("\n1. Statistical Arbitrage Strategy")
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
        # keep shapes consistent so downstream code/plots don’t break
        stat_arb_portfolio = pd.DataFrame(index=prices_df.index)
        stat_arb_portfolio['equity'] = 0.0
        stat_arb_portfolio['return'] = 0.0
        stat_arb_positions = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    strategies['stat_arb'] = stat_arb_portfolio

    # 2. Merger Arbitrage
    print("\n2. Merger Arbitrage Strategy")
    merger_arb_capital = initial_capital * strategy_weights['merger_arb']
    merger_arb_portfolio = merger_arbitrage_strategy(
        prices_df=prices_df,
        merger_etfs=['MNA'],  # IQ Merger Arbitrage ETF
        initial_capital=merger_arb_capital
    )
    strategies['merger_arb'] = merger_arb_portfolio

    # 3. Risk Parity
    print("\n3. Risk Parity Strategy")
    risk_parity_capital = initial_capital * strategy_weights['risk_parity']
    risk_parity_portfolio = risk_parity_strategy(
        prices_df=prices_df,
        rp_assets=['TLT', 'GLD', 'TIP', 'IEF', 'LQD', 'BND'],
        initial_capital=risk_parity_capital
    )
    strategies['risk_parity'] = risk_parity_portfolio

    # 4. Covered Call Strategy
    print("\n4. Covered Call Strategy")
    covered_call_capital = initial_capital * strategy_weights['covered_call']
    covered_call_portfolio = covered_call_strategy(
        prices_df=prices_df,
        underlying='SPY',
        initial_capital=covered_call_capital
    )
    strategies['covered_call'] = covered_call_portfolio

    # Align all strategies to common dates
    common_dates = prices_df.index
    for strategy_name in strategies:
        strategies[strategy_name] = strategies[strategy_name].reindex(common_dates).ffill().bfill()

    # Combine strategies
    combined = pd.DataFrame(index=common_dates)

    for strategy_name, portfolio in strategies.items():
        if 'equity' in portfolio.columns:
            combined[strategy_name] = portfolio['equity']
        else:
            combined[strategy_name] = initial_capital * strategy_weights[strategy_name]

    # Calculate total portfolio
    combined['total'] = combined.sum(axis=1)
    combined['return'] = combined['total'].pct_change().fillna(0)

    # Calculate individual strategy returns
    for strategy_name in strategies:
        if 'return' in strategies[strategy_name].columns:
            combined[f'{strategy_name}_return'] = strategies[strategy_name]['return']

    return {
        'combined': combined,
        'strategies': strategies,
        'positions': stat_arb_positions,
        'strategy_weights': strategy_weights
    }


def calculate_performance_metrics(portfolio: pd.DataFrame,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.02) -> Dict:
    """Calculate comprehensive performance metrics."""

    returns = portfolio['return'].dropna()
    equity = portfolio['total']

    if len(returns) < 10:
        return {}

    metrics = {}

    # Basic returns
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
    metrics['Total Return'] = total_return
    metrics['CAGR'] = ((1 + total_return) ** (252 / len(returns)) - 1)

    # Risk metrics
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    metrics['Annual Volatility'] = annual_vol

    # Downside risk
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    metrics['Downside Volatility'] = downside_vol

    # Risk-adjusted returns
    excess_return = metrics['CAGR'] - risk_free_rate
    metrics['Sharpe Ratio'] = excess_return / annual_vol if annual_vol > 0 else 0

    metrics['Sortino Ratio'] = excess_return / downside_vol if downside_vol > 0 else 0

    # Maximum drawdown
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1)
    metrics['Max Drawdown'] = drawdown.min()
    metrics['Avg Drawdown'] = drawdown[drawdown < 0].mean()

    # Win rate
    metrics['Win Rate'] = (returns > 0).mean()

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar ratio
    metrics['Calmar Ratio'] = metrics['CAGR'] / abs(metrics['Max Drawdown']) if metrics['Max Drawdown'] < 0 else 0

    # Market correlation
    if benchmark_returns is not None:
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 10:
            correlation = returns.loc[common_idx].corr(benchmark_returns.loc[common_idx])
            metrics['Market Correlation'] = correlation

            # Beta calculation
            cov_matrix = np.cov(returns.loc[common_idx], benchmark_returns.loc[common_idx])
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
            metrics['Beta'] = beta

            # Alpha
            benchmark_cagr = ((1 + benchmark_returns.loc[common_idx]).prod() - 1) * (252 / len(common_idx))
            metrics['Alpha'] = metrics['CAGR'] - (risk_free_rate + beta * (benchmark_cagr - risk_free_rate))

    # Value at Risk
    metrics['Daily VaR 95%'] = np.percentile(returns, 5)
    metrics['Expected Shortfall 95%'] = returns[returns <= metrics['Daily VaR 95%']].mean()

    # Skewness and Kurtosis
    metrics['Skewness'] = stats.skew(returns)
    metrics['Kurtosis'] = stats.kurtosis(returns)

    # Monthly consistency
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    metrics['Monthly Win Rate'] = (monthly_returns > 0).mean()
    metrics['Positive Months %'] = metrics['Monthly Win Rate'] * 100

    return metrics


def plot_results(results: Dict, prices_df: pd.DataFrame):
    """Generate performance visualizations."""

    portfolio = results['combined']
    strategies = results['strategies']

    fig = plt.figure(figsize=(18, 14))

    # 1. Equity curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(portfolio.index, portfolio['total'], label='Multi-Strategy',
             linewidth=3, color='darkblue', alpha=0.9)

    # Plot individual strategies
    colors = ['green', 'red', 'orange', 'purple']
    for idx, (strat_name, strat_data) in enumerate(strategies.items()):
        if 'equity' in strat_data.columns:
            ax1.plot(strat_data.index, strat_data['equity'],
                     label=strat_name.replace('_', ' ').title(),
                     alpha=0.6, linestyle='--', color=colors[idx % len(colors)])

    # SPY comparison
    if 'SPY' in prices_df.columns:
        spy_returns = prices_df['SPY'].pct_change().fillna(0)
        spy_equity = 100000 * (1 + spy_returns).cumprod()
        ax1.plot(prices_df.index, spy_equity, label='SPY',
                 linewidth=2, color='gray', alpha=0.5)

    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = plt.subplot(3, 3, 2)
    equity = portfolio['total']
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100
    ax2.fill_between(portfolio.index, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(portfolio.index, drawdown, color='red', linewidth=1)
    ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Monthly returns heatmap
    ax3 = plt.subplot(3, 3, 3)
    monthly_returns = portfolio['return'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

    if len(monthly_returns) > 0:
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'returns': monthly_returns.values
        })

        # Pivot for heatmap
        heatmap_data = monthly_df.pivot(index='year', columns='month', values='returns')

        im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                        vmin=-10, vmax=10, interpolation='nearest')
        ax3.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month', fontsize=11)
        ax3.set_ylabel('Year', fontsize=11)
        plt.colorbar(im, ax=ax3)

    # 4. Returns distribution
    ax4 = plt.subplot(3, 3, 4)
    returns = portfolio['return'].dropna() * 100
    ax4.hist(returns, bins=50, alpha=0.7, color='steelblue',
             edgecolor='black', density=True)

    # Add normal distribution for comparison
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

    # 5. Rolling Sharpe ratio (6-month)
    ax5 = plt.subplot(3, 3, 5)
    rolling_window = 126  # 6 months
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

    # 6. Strategy allocation over time
    ax6 = plt.subplot(3, 3, 6)

    # Calculate allocation percentages
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

    # 7. Correlation matrix (strategies)
    ax7 = plt.subplot(3, 3, 7)

    # Get strategy returns
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

        # Set tick labels
        tick_labels = [s.replace('_', '\n').title() for s in corr_matrix.columns]
        ax7.set_xticks(range(len(corr_matrix.columns)))
        ax7.set_yticks(range(len(corr_matrix.columns)))
        ax7.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
        ax7.set_yticklabels(tick_labels, fontsize=9)

        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax7.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                         fontsize=8)

        plt.colorbar(im, ax=ax7)

    # 8. Active positions (from stat arb)
    ax8 = plt.subplot(3, 3, 8)
    if 'positions' in results:
        positions = results['positions']
        active_positions = (positions.abs() > 0).sum(axis=1)
        ax8.plot(positions.index, active_positions, color='darkorange', linewidth=2)
        ax8.set_title('Active Statistical Arb Positions', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Number of Positions', fontsize=11)
        ax8.set_xlabel('Date', fontsize=11)
        ax8.grid(True, alpha=0.3)

    # 9. Rolling volatility (1-month)
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
        # Equity ETFs for pair trading
        'XLF', 'XLV', 'XLK', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB',

        # International for diversification
        'EFA', 'EEM',

        # Fixed Income (for risk parity)
        'TLT', 'IEF', 'SHY', 'LQD', 'BND', 'TIP',

        # Alternatives
        'GLD', 'SLV',

        # Merger Arbitrage ETF
        'MNA',

        # Volatility/Alternative
        'VXX',

        # Benchmarks
        'SPY', 'AGG',
    ]

    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    initial_capital = 100_000.0
    risk_free_rate = 0.02

    # Strategy weights (stat arb removed; shifted to covered calls)
    strategy_weights = {
        'stat_arb': 0.0,
        'merger_arb': 0.1,
        'risk_parity': 0.2,
        'covered_call': 0.7
    }
    # Normalize in case of typos / future edits
    wsum = sum(strategy_weights.values())
    if wsum <= 0:
        raise ValueError("strategy_weights must sum to a positive number.")
    if abs(wsum - 1.0) > 1e-9:
        strategy_weights = {k: v / wsum for k, v in strategy_weights.items()}

    print("=" * 80)
    print("MARKET-NEUTRAL MULTI-STRATEGY PORTFOLIO")
    print("=" * 80)
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Number of Assets (requested): {len(tickers)}")
    print("=" * 80)

    # Download data
    data = download_data(tickers, start_date, end_date)

    if len(data) < 10:
        print(f"\nWarning: Only {len(data)} assets downloaded. Strategy may be limited.")

    # Create prices DataFrame
    prices_list = []
    for ticker, df in data.items():
        if (df is not None) and (not df.empty) and ('Adj Close' in df.columns):
            prices_list.append(pd.DataFrame({ticker: df['Adj Close']}))

    if not prices_list:
        print("No valid price data available")
        return

    prices_df = pd.concat(prices_list, axis=1)

    # Forward fill missing values (up to 5 days), then drop remaining NAs
    prices_df = prices_df.ffill(limit=5).dropna()

    if len(prices_df) < 252:
        print(f"Warning: Only {len(prices_df)} trading days available. Need at least 252.")
        return

    print(f"\nData range (prices_df): {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    print(f"Trading days: {len(prices_df)}")
    print(f"Available assets: {len(prices_df.columns)}")

    # Run multi-strategy portfolio
    results = multi_strategy_portfolio(
        prices_df=prices_df,
        initial_capital=initial_capital,
        strategy_weights=strategy_weights
    )

    # Benchmark returns (align to portfolio dates to avoid mismatched windows)
    benchmark_returns = None
    if 'SPY' in prices_df.columns:
        benchmark_returns = prices_df['SPY'].pct_change().fillna(0.0)
        # Align benchmark to combined portfolio index
        benchmark_returns = benchmark_returns.reindex(results['combined'].index).fillna(0.0)

        common_idx = results['combined'].index
        print(f"\nSPY window used: {common_idx[0].date()} to {common_idx[-1].date()}  (days={len(common_idx)})")

    # Portfolio metrics (uses your existing function)
    metrics = calculate_performance_metrics(
        results['combined'],
        benchmark_returns if benchmark_returns is not None else None,
        risk_free_rate=risk_free_rate
    )

    # Print results
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS (Annualized)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 80)

    # Portfolio metrics from calculate_performance_metrics()
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

    # --- Correct SPY metrics (true CAGR, not linear annualization) ---
    spy_metrics = {}
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        spy_returns = benchmark_returns.dropna()
        n = len(spy_returns)

        # True geometric CAGR based on compounded total return
        spy_total_growth = float((1.0 + spy_returns).prod())
        spy_cagr = spy_total_growth ** (252.0 / n) - 1.0 if n > 0 else 0.0

        spy_vol = float(spy_returns.std() * np.sqrt(252.0))
        spy_sharpe = (spy_cagr - risk_free_rate) / spy_vol if spy_vol > 0 else 0.0

        # Drawdown
        spy_equity = (1.0 + spy_returns).cumprod()
        spy_cummax = spy_equity.cummax()
        spy_drawdown = float((spy_equity / spy_cummax - 1.0).min())

        # Sortino (optional but easy)
        downside = spy_returns[spy_returns < 0]
        downside_vol = float(downside.std() * np.sqrt(252.0)) if len(downside) > 0 else 0.0
        spy_sortino = (spy_cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

        spy_metrics = {
            'CAGR': spy_cagr,
            'Annual Vol': spy_vol,
            'Sharpe Ratio': spy_sharpe,
            'Sortino Ratio': spy_sortino,
            'Max Drawdown': spy_drawdown,
            'Win Rate': float((spy_returns > 0).mean()),
            'Market Correlation': 1.0,
            'Beta': 1.0,
        }

    for metric, port_val in portfolio_metrics.items():
        spy_val = spy_metrics.get(metric, 0)

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

    # Additional metrics (portfolio only)
    print(f"{'Positive Months':<25} {metrics.get('Positive Months %', 0):>12.1f}%")
    print(f"{'Profit Factor':<25} {metrics.get('Profit Factor', 0):>12.2f}")
    print(f"{'Calmar Ratio':<25} {metrics.get('Calmar Ratio', 0):>12.2f}")
    print(f"{'Alpha':<25} {metrics.get('Alpha', 0):>12.2%}")

    print("=" * 80)

    # Strategy correlations
    print("\nSTRATEGY CORRELATIONS:")
    print("-" * 40)

    strat_returns = {}
    for strat_name, strat_df in results['strategies'].items():
        if isinstance(strat_df, pd.DataFrame) and ('return' in strat_df.columns):
            # keep only if it has variability / non-empty
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

    # Risk metrics
    print("\nRISK METRICS:")
    print("-" * 40)
    print(f"Daily VaR 95%: {metrics.get('Daily VaR 95%', 0):.2%}")
    print(f"Expected Shortfall 95%: {metrics.get('Expected Shortfall 95%', 0):.2%}")
    print(f"Skewness: {metrics.get('Skewness', 0):.3f}")
    print(f"Kurtosis: {metrics.get('Kurtosis', 0):.3f}")

    # Plot results
    print("\nGenerating performance charts...")
    plot_results(results, prices_df)


if __name__ == "__main__":
    main()