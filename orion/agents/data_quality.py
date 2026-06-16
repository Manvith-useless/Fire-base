"""Data & Quality module — gatekeeper that validates the inputs.

Nothing downstream should be trusted if the data is thin, stale, or full of gaps.
Emits no scoring dimension; instead it sets confidence and a reanalysis flag.
"""
from __future__ import annotations

import pandas as pd

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport
from .base import Agent

MIN_CANDLES = 200  # needed for a 200-day SMA / meaningful risk stats


class DataQualityAgent(Agent):
    name = "Data & Quality"

    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        df = ctx.candles
        evidence: list[str] = []
        risks: list[str] = []
        n = len(df)
        evidence.append(f"Candles available: {n}")

        nan_cols = [c for c in df.columns if df[c].isna().any()]
        reanalysis = False
        confidence = 90.0

        if n < MIN_CANDLES:
            risks.append(
                f"Only {n} candles (< {MIN_CANDLES}); long-horizon indicators unreliable"
            )
            confidence -= 30
            reanalysis = True
        if nan_cols:
            risks.append(f"Missing values in columns: {nan_cols}")
            confidence -= 15

        # Freshness check (daily candles should be recent).
        if n:
            last_date = pd.to_datetime(df["date"].iloc[-1])
            age_days = (pd.Timestamp.now(tz=last_date.tz) - last_date).days
            evidence.append(f"Most recent candle age: {age_days} day(s)")
            if age_days > 5:
                risks.append(f"Data is stale ({age_days} days old)")
                confidence -= 20
        if ctx.ltp is None:
            risks.append("No live price (LTP) available")
            confidence -= 10

        if not risks:
            evidence.append("Data passed freshness, completeness and depth checks")

        return [
            self._report(
                dimension=None,
                score=confidence,  # informational only
                confidence=confidence,
                evidence=evidence,
                risks=risks or ["None material"],
                reanalysis=reanalysis,
            )
        ]
