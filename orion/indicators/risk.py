"""Risk and return metrics computed deterministically.

All functions take a close-price Series (and optionally a benchmark Series) and
return floats. Annualization assumes daily candles (252 trading days/year).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_volatility(close: pd.Series) -> float | None:
    r = daily_returns(close)
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(close: pd.Series, periods_per_year: int = TRADING_DAYS) -> float | None:
    s = close.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return None
    years = len(s) / periods_per_year
    if years <= 0:
        return None
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)


def max_drawdown(close: pd.Series) -> float | None:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.32)."""
    s = close.dropna()
    if s.empty:
        return None
    running_max = s.cummax()
    drawdown = (s - running_max) / running_max
    return float(drawdown.min())


def sharpe_ratio(close: pd.Series, risk_free_rate: float = 0.0) -> float | None:
    r = daily_returns(close)
    if len(r) < 2:
        return None
    excess = r - risk_free_rate / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0:
        return None
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(close: pd.Series, risk_free_rate: float = 0.0) -> float | None:
    r = daily_returns(close)
    if len(r) < 2:
        return None
    excess = r - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return None
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def beta(close: pd.Series, benchmark_close: pd.Series) -> float | None:
    """Beta of the asset vs a benchmark, aligned on index."""
    a = daily_returns(close)
    b = daily_returns(benchmark_close)
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return None
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    var_b = cov[1, 1]
    if var_b == 0:
        return None
    return float(cov[0, 1] / var_b)


def historical_var(close: pd.Series, confidence: float = 0.95) -> float | None:
    """Historical Value-at-Risk on daily returns (negative fraction)."""
    r = daily_returns(close)
    if len(r) < 2:
        return None
    return float(np.percentile(r, (1 - confidence) * 100))


def compute_snapshot(
    close: pd.Series, benchmark_close: pd.Series | None = None
) -> dict[str, float | None]:
    out = {
        "annualized_volatility": annualized_volatility(close),
        "cagr": cagr(close),
        "max_drawdown": max_drawdown(close),
        "sharpe": sharpe_ratio(close),
        "sortino": sortino_ratio(close),
        "var_95": historical_var(close, 0.95),
    }
    if benchmark_close is not None:
        out["beta"] = beta(close, benchmark_close)
    return out
