"""Deterministic synthetic data provider for offline development and tests.

Generates reproducible OHLCV candles (seeded per symbol) so the full pipeline can
run end-to-end without paid Kite Connect credentials. Clearly NOT real market data.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .provider import DataProvider, Instrument


class SampleDataProvider(DataProvider):
    name = "sample"

    def __init__(self, seed_base: int = 1337, start_price: float = 1000.0):
        self.seed_base = seed_base
        self.start_price = start_price

    def _seed_for(self, symbol: str) -> int:
        # Stable across processes (builtin hash() is salted per run).
        digest = hashlib.sha256(symbol.encode()).hexdigest()
        return self.seed_base + (int(digest[:8], 16) % 100000)

    def get_candles(
        self, instrument: Instrument, interval: str = "day", days: int = 400
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed_for(instrument.symbol))
        # Mild upward drift + noise -> realistic-looking geometric random walk.
        drift = 0.0004
        vol = 0.015
        returns = rng.normal(drift, vol, days)
        close = self.start_price * np.cumprod(1 + returns)

        # Build OHLC around the close path.
        intraday = np.abs(rng.normal(0, vol, days)) * close
        open_ = np.concatenate([[self.start_price], close[:-1]])
        high = np.maximum(open_, close) + intraday
        low = np.minimum(open_, close) - intraday
        low = np.clip(low, 1e-6, None)
        volume = rng.integers(100_000, 5_000_000, days).astype(float)

        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
        return pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def get_ltp(self, instrument: Instrument) -> float | None:
        df = self.get_candles(instrument, days=2)
        return float(df["close"].iloc[-1])

    def get_holdings(self) -> list[dict]:
        # A small illustrative portfolio so Portfolio & Sizing has something to chew on.
        return [
            {"tradingsymbol": "INFY", "quantity": 50, "average_price": 1450.0,
             "last_price": 1500.0},
            {"tradingsymbol": "TCS", "quantity": 20, "average_price": 3600.0,
             "last_price": 3700.0},
        ]

    def get_margins(self) -> dict:
        return {"equity": {"available": {"live_balance": 250_000.0}}}
