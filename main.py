import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data using yfinance."""
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        raise ValueError("No data downloaded. Check ticker and date range.")

    # Ensure Adj Close exists; if not, use Close as a proxy.
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]

    return df


def compute_indicators(
    df: pd.DataFrame,
    fast_sma: int = 50,
    slow_sma: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """
    Add multiple technical indicators and create a composite trading signal.
    Long-only, trend + momentum + volatility filter.
    """
    df = df.copy()
    close = df["Adj Close"]
    high = df["High"]
    low = df["Low"]

    # === Trend indicators ===
    df["SMA_fast"] = close.rolling(fast_sma, min_periods=fast_sma // 2).mean()
    df["SMA_slow"] = close.rolling(slow_sma, min_periods=slow_sma // 2).mean()
    df["EMA_trend"] = close.ewm(span=21, adjust=False).mean()

    # === RSI (Wilder-style) ===
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # === ATR ===
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(atr_period, min_periods=atr_period // 2).mean()
    df["ATR_pct"] = df["ATR"] / close

    # === MACD (12, 26, 9) ===
    ema_fast_macd = close.ewm(span=12, adjust=False).mean()
    ema_slow_macd = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast_macd - ema_slow_macd
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # === Bollinger Bands (20, 2) ===
    bb_mid = close.rolling(bb_period, min_periods=bb_period // 2).mean()
    bb_std_series = close.rolling(bb_period, min_periods=bb_period // 2).std()

    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_mid + bb_std * bb_std_series
    df["BB_lower"] = bb_mid - bb_std * bb_std_series
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid

    # === Composite trading logic ===

    # 1) Market regime filter: only trade in bullish regime
    df["bull_regime"] = close > df["SMA_slow"]

    # 2) Trend score (0–2)
    trend_score = (
        (close > df["SMA_fast"]).astype(int)
        + (df["EMA_trend"] > df["SMA_fast"]).astype(int)
    )

    # 3) Momentum score (0–2)
    rsi_ok = (df["RSI"] > 50) & (df["RSI"] < 70)
    macd_ok = df["MACD_hist"] > 0
    momentum_score = rsi_ok.astype(int) + macd_ok.astype(int)

    # 4) Volatility filter
    vol_filter = (df["ATR_pct"] < 0.04) & (df["BB_width"] > 0.02)

    # Long when:
    # - Bullish regime
    # - At least some trend confirmation
    # - Some positive momentum
    # - Volatility is not extreme or dead
    df["long_signal_raw"] = (
        df["bull_regime"]
        & (trend_score >= 1)
        & (momentum_score >= 1)
        & vol_filter
    )

    # Convert to numeric signal 1 (long) or 0 (flat)
    df["signal"] = 0
    df.loc[df["long_signal_raw"], "signal"] = 1

    # Position = yesterday's signal (trade at next day's open)
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    trading_fee_bp: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorized backtest engine:
    - Uses 'position' column in df (0 or 1).
    - Applies trading fees when position changes.
    """
    df = df.copy()

    # Daily returns of the asset
    df["asset_ret"] = df["Adj Close"].pct_change().fillna(0.0)

    # Gross strategy returns
    df["strategy_ret_gross"] = df["position"] * df["asset_ret"]

    # Trading costs: whenever we change position
    df["trade"] = df["position"].diff().fillna(0.0).abs()
    fee_rate = trading_fee_bp / 10_000.0  # basis points → decimal
    df["trading_cost"] = df["trade"] * fee_rate

    # Net strategy returns
    df["strategy_ret_net"] = df["strategy_ret_gross"] - df["trading_cost"]

    # Equity curves
    df["equity_strategy"] = initial_capital * (1 + df["strategy_ret_net"]).cumprod()
    df["equity_buy_hold"] = initial_capital * (1 + df["asset_ret"]).cumprod()

    return df


def performance_stats(df: pd.DataFrame) -> dict:
    """Calculate common performance statistics."""
    stats = {}

    trading_days_per_year = 252
    total_days = (df.index[-1] - df.index[0]).days
    years = total_days / 365.25 if total_days > 0 else 0

    initial = df["equity_strategy"].iloc[0]
    final = df["equity_strategy"].iloc[-1]

    stats["Total Return %"] = (final / initial - 1) * 100 if initial > 0 else np.nan
    stats["CAGR %"] = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 and initial > 0 else np.nan

    # Max drawdown
    equity = df["equity_strategy"]
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    stats["Max Drawdown %"] = drawdown.min() * 100

    # Sharpe
    daily_ret = df["strategy_ret_net"]
    avg_daily = daily_ret.mean()
    std_daily = daily_ret.std()
    sharpe = (avg_daily / std_daily) * np.sqrt(trading_days_per_year) if std_daily > 0 else np.nan
    stats["Sharpe Ratio"] = sharpe

    # Win rate & profit factor
    positive = daily_ret[daily_ret > 0]
    negative = daily_ret[daily_ret < 0]
    stats["Win Rate %"] = (len(positive) / len(daily_ret) * 100) if len(daily_ret) > 0 else np.nan
    stats["Profit Factor"] = (positive.sum() / abs(negative.sum())) if negative.sum() < 0 else np.nan

    return stats


def plot_results(df: pd.DataFrame, ticker: str):
    """Plot price with key indicators and the equity curves."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Adj Close"], label=f"{ticker} Price")
    plt.plot(df.index, df["SMA_fast"], label="SMA Fast (50)")
    plt.plot(df.index, df["SMA_slow"], label="SMA Slow (200)")
    plt.plot(df.index, df["EMA_trend"], label="EMA Trend (21)", linestyle=":")

    # Mark buy/sell points
    buys = (df["position"] == 1) & (df["position"].shift(1) == 0)
    sells = (df["position"] == 0) & (df["position"].shift(1) == 1)

    plt.scatter(df.index[buys], df["Adj Close"][buys], marker="^", s=80, label="Buy", zorder=5)
    plt.scatter(df.index[sells], df["Adj Close"][sells], marker="v", s=80, label="Sell", zorder=5)

    plt.title(f"{ticker} Price & Multi-Indicator Strategy Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity_strategy"], label="Strategy Equity")
    plt.plot(df.index, df["equity_buy_hold"], label="Buy & Hold Equity", linestyle="--")
    plt.title("Equity Curve: Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # === CONFIG ===
    ticker = "SPY"
    start = "2015-01-01"
    end = datetime.today().strftime("%Y-%m-%d")
    initial_capital = 10_000.0
    trading_fee_bp = 1.0  # 1 bp per trade

    print(f"Downloading data for {ticker} from {start} to {end}...")
    df = download_data(ticker, start, end)

    print("Computing indicators and signals...")
    df = compute_indicators(df)

    print("Running backtest...")
    df = backtest(df, initial_capital=initial_capital, trading_fee_bp=trading_fee_bp)

    print("Calculating performance stats...")
    stats = performance_stats(df)

    print("\n=== Performance Summary (Multi-Indicator Strategy) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:8.2f}")
        else:
            print(f"{k:20s}: {v}")

    print("\nPlotting results...")
    plot_results(df, ticker)


if __name__ == "__main__":
    main()
