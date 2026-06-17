"""Technical indicators computed deterministically with pandas/numpy.

Each function takes an OHLCV pandas DataFrame with columns:
    ['date', 'open', 'high', 'low', 'close', 'volume']
and returns either a pandas Series (full history) or a float (latest value).

Formulas follow standard definitions (Wilder's smoothing for RSI/ATR/ADX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When avg_loss == 0 the asset only gained -> RSI 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower})


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    tr = _true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = _true_range(df)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (cumulative over the supplied window)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0.0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


def latest(series: pd.Series) -> float | None:
    """Return the last non-NaN value of a series as a float, or None."""
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def compute_snapshot(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute the latest value of every indicator for a quick agent-facing summary."""
    close = df["close"]
    macd_df = macd(close)
    bb = bollinger_bands(close)
    adx_df = adx(df)
    return {
        "close": latest(close),
        "sma_20": latest(sma(close, 20)),
        "sma_50": latest(sma(close, 50)),
        "sma_200": latest(sma(close, 200)),
        "ema_20": latest(ema(close, 20)),
        "rsi_14": latest(rsi(close)),
        "macd": latest(macd_df["macd"]),
        "macd_signal": latest(macd_df["signal"]),
        "macd_hist": latest(macd_df["hist"]),
        "bb_upper": latest(bb["upper"]),
        "bb_mid": latest(bb["mid"]),
        "bb_lower": latest(bb["lower"]),
        "atr_14": latest(atr(df)),
        "adx_14": latest(adx_df["adx"]),
        "plus_di": latest(adx_df["plus_di"]),
        "minus_di": latest(adx_df["minus_di"]),
        "obv": latest(obv(df)),
        "vwap": latest(vwap(df)),
    }
