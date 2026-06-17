"""DataProvider interface.

Decouples the committee from the data source so the app can run on cached/sample
candles during development and swap in live Kite Connect for production.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Instrument:
    symbol: str
    exchange: str = "NSE"

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}"


class DataProvider(ABC):
    """Read interface plus a guarded write (order placement)."""

    name: str = "abstract"

    @abstractmethod
    def get_candles(
        self, instrument: Instrument, interval: str = "day", days: int = 400
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with columns
        ['date','open','high','low','close','volume'], oldest first."""

    @abstractmethod
    def get_ltp(self, instrument: Instrument) -> float | None:
        """Latest traded price."""

    @abstractmethod
    def get_holdings(self) -> list[dict]:
        """Equity holdings (empty list if unavailable)."""

    @abstractmethod
    def get_margins(self) -> dict:
        """Available funds/margins (empty dict if unavailable)."""

    def place_order(self, **kwargs) -> str:
        """Place an order. Only providers that support live trading implement this."""
        raise NotImplementedError(
            f"{self.name} provider does not support order placement"
        )
