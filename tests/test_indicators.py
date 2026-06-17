"""Verify indicator/risk math against hand-checked reference values."""
import numpy as np
import pandas as pd

from orion.indicators import risk, technical as t


def _ohlcv(close):
    close = pd.Series(close, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=len(close)),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )


def test_sma_last_value():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert t.latest(t.sma(s, 2)) == 4.5


def test_ema_constant_series():
    s = pd.Series([7.0] * 30)
    assert abs(t.latest(t.ema(s, 10)) - 7.0) < 1e-9


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40), dtype=float)  # strictly increasing
    assert abs(t.latest(t.rsi(s, 14)) - 100.0) < 1e-6


def test_rsi_in_bounds():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    vals = t.rsi(s, 14).dropna()
    assert (vals >= 0).all() and (vals <= 100).all()


def test_bollinger_mid_equals_sma():
    s = pd.Series(np.arange(1, 60), dtype=float)
    bb = t.bollinger_bands(s, 20)
    assert abs(t.latest(bb["mid"]) - t.latest(t.sma(s, 20))) < 1e-9


def test_macd_hist_is_macd_minus_signal():
    s = pd.Series(np.arange(1, 100), dtype=float)
    m = t.macd(s)
    row = m.dropna().iloc[-1]
    assert abs(row["hist"] - (row["macd"] - row["signal"])) < 1e-9


def test_atr_positive():
    df = _ohlcv(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 100)))
    assert t.latest(t.atr(df, 14)) > 0


def test_max_drawdown_known():
    s = pd.Series([100, 120, 90, 130], dtype=float)
    assert abs(risk.max_drawdown(s) - (-0.25)) < 1e-9


def test_cagr_doubling_one_year():
    s = pd.Series([100.0, 200.0])
    assert abs(risk.cagr(s, periods_per_year=2) - 1.0) < 1e-9


def test_sharpe_positive_for_uptrend():
    s = pd.Series(100 * (1.001 ** np.arange(300)))
    assert risk.sharpe_ratio(s) > 0


def test_snapshots_have_expected_keys():
    df = _ohlcv(100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 300)))
    tech = t.compute_snapshot(df)
    assert {"rsi_14", "macd", "adx_14", "vwap"} <= tech.keys()
    rsnap = risk.compute_snapshot(df["close"])
    assert {"sharpe", "max_drawdown", "annualized_volatility"} <= rsnap.keys()
