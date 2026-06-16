"""Snapshot data provider — serves a frozen JSON snapshot of real account data.

Used for the "in-session Zerodha bridge": Claude pulls live data via the MCP
tools, writes it to a JSON file, and the committee runs on real numbers without
the app needing its own (paid) Kite Connect credentials.

Snapshot JSON shape:
{
  "symbol": "INFY",
  "exchange": "NSE",
  "ltp": 1500.0,
  "candles": [{"date": "...", "open": .., "high": .., "low": .., "close": .., "volume": ..}, ...],
  "holdings": [...],
  "margins": {...}
}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .provider import DataProvider, Instrument


class SnapshotDataProvider(DataProvider):
    name = "snapshot"

    def __init__(self, snapshot_file: str | Path):
        path = Path(snapshot_file)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {path}")
        self._data = json.loads(path.read_text())

    def get_candles(
        self, instrument: Instrument, interval: str = "day", days: int = 400
    ) -> pd.DataFrame:
        candles = self._data.get("candles", [])
        if not candles:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(candles)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]]

    def get_ltp(self, instrument: Instrument) -> float | None:
        ltp = self._data.get("ltp")
        return float(ltp) if ltp is not None else None

    def get_holdings(self) -> list[dict]:
        return self._data.get("holdings", [])

    def get_margins(self) -> dict:
        return self._data.get("margins", {})
