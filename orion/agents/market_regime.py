"""Market Regime module — trend regime + volatility regime from price history."""
from __future__ import annotations

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport
from .base import Agent, fmt


class MarketRegimeAgent(Agent):
    name = "Market Regime"

    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        t, r = ctx.technical, ctx.risk
        score = 50.0
        ev: list[str] = []
        risks: list[str] = []

        sma50, sma200 = t.get("sma_50"), t.get("sma_200")
        if sma50 and sma200:
            if sma50 > sma200:
                score += 15
                ev.append(f"SMA50 {fmt(sma50)} > SMA200 {fmt(sma200)} — golden-cross / uptrend regime")
            else:
                score -= 15
                risks.append(f"SMA50 {fmt(sma50)} < SMA200 {fmt(sma200)} — death-cross / downtrend regime")

        adx = t.get("adx_14")
        if adx is not None:
            if adx >= 25:
                ev.append(f"ADX {fmt(adx)} — trending regime (signals more reliable)")
            else:
                score -= 5
                risks.append(f"ADX {fmt(adx)} — choppy/range regime (trend signals less reliable)")

        vol = r.get("annualized_volatility")
        if vol is not None:
            pct = vol * 100
            if vol < 0.20:
                score += 10
                ev.append(f"Annualized volatility {pct:.1f}% — calm regime")
            elif vol < 0.40:
                ev.append(f"Annualized volatility {pct:.1f}% — normal regime")
            else:
                score -= 12
                risks.append(f"Annualized volatility {pct:.1f}% — high-volatility regime")

        return [self._report("market_regime", score, confidence=72,
                            evidence=ev or ["Regime indeterminate"],
                            risks=risks or ["None material"])]
