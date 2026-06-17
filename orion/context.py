"""AnalysisContext: the bundle of computed, data-grounded facts shared by agents."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data.provider import Instrument


@dataclass
class AnalysisContext:
    instrument: Instrument
    candles: pd.DataFrame
    technical: dict[str, float | None]
    risk: dict[str, float | None]
    ltp: float | None = None
    holdings: list[dict] = field(default_factory=list)
    margins: dict = field(default_factory=dict)

    @property
    def available_cash(self) -> float:
        try:
            return float(self.margins["equity"]["available"]["live_balance"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def held_quantity(self) -> int:
        for h in self.holdings:
            if h.get("tradingsymbol") == self.instrument.symbol:
                return int(h.get("quantity", 0))
        return 0
