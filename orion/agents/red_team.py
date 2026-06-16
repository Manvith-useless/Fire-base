"""Red Team module — actively attacks the emerging thesis.

Emits no scoring dimension; it surfaces disconfirming evidence and can force a
re-analysis when it finds contradictions among the signals.
"""
from __future__ import annotations

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport
from .base import Agent, fmt


class RedTeamAgent(Agent):
    name = "Red Team"

    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        t, r = ctx.technical, ctx.risk
        attacks: list[str] = []
        ev: list[str] = []

        close, sma200 = t.get("close"), t.get("sma_200")
        rsi = t.get("rsi_14")
        macd_hist = t.get("macd_hist")

        # Contradiction hunting.
        if rsi is not None and rsi > 70 and macd_hist is not None and macd_hist > 0:
            attacks.append("Momentum is bullish but RSI is overbought — chasing risk at extension")
        if close and sma200 and close > sma200 and macd_hist is not None and macd_hist < 0:
            attacks.append("Price above SMA200 yet MACD turning down — possible trend exhaustion")

        vol = r.get("annualized_volatility")
        if vol and vol > 0.35:
            attacks.append(f"Elevated volatility {vol*100:.1f}% can wipe out the edge via whipsaw")

        mdd = r.get("max_drawdown")
        if mdd is not None and mdd < -0.3:
            attacks.append(f"History shows {mdd*100:.0f}% drawdowns — assume it can recur")

        sharpe = r.get("sharpe")
        if sharpe is not None and sharpe < 0.5:
            attacks.append(f"Weak Sharpe {fmt(sharpe)} — reward may not justify the risk")

        # Always remind of structural limits of this technical-only system.
        attacks.append(
            "Thesis is price/technical only — no fundamentals, earnings or news; a "
            "negative surprise is unmodelled here"
        )
        ev.append("Red Team applied: hunted contradictions across trend/momentum/risk")

        # If many independent attacks land, demand re-analysis.
        reanalysis = len(attacks) >= 4

        return [self._report(dimension=None, score=50.0, confidence=70,
                            evidence=ev, risks=attacks, reanalysis=reanalysis)]
