"""Risk, Scenarios & Stress module.

Scores the risk_volatility dimension and produces bull/base/bear scenarios plus
black-swan flags from the computed risk metrics.
"""
from __future__ import annotations

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport
from .base import Agent, fmt


class RiskScenariosAgent(Agent):
    name = "Risk, Scenarios & Stress"

    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        r = ctx.risk
        score = 60.0
        ev: list[str] = []
        risks: list[str] = []
        reanalysis = False

        vol = r.get("annualized_volatility")
        mdd = r.get("max_drawdown")
        sharpe = r.get("sharpe")
        sortino = r.get("sortino")
        var95 = r.get("var_95")

        if vol is not None:
            if vol > 0.50:
                score -= 20
                risks.append(f"Very high annualized volatility {vol*100:.1f}%")
            elif vol > 0.35:
                score -= 8
            else:
                score += 8
                ev.append(f"Contained volatility {vol*100:.1f}%")

        if mdd is not None:
            ev.append(f"Historical max drawdown {mdd*100:.1f}%")
            if mdd < -0.40:
                score -= 15
                risks.append(f"Severe historical drawdown {mdd*100:.1f}% — stress risk")
            elif mdd < -0.25:
                score -= 6

        if sharpe is not None:
            ev.append(f"Sharpe {fmt(sharpe)}")
            score += 10 if sharpe > 1 else (-10 if sharpe < 0 else 0)
        if sortino is not None:
            ev.append(f"Sortino {fmt(sortino)}")

        # Scenario construction grounded in volatility/VaR.
        if ctx.ltp and vol is not None:
            price = ctx.ltp
            ev.append(
                f"Scenarios (1y, vol-based): "
                f"Bull +{vol*100:.0f}% ≈ {price*(1+vol):.0f} | "
                f"Base ≈ {price:.0f} | "
                f"Bear -{vol*100:.0f}% ≈ {price*(1-vol):.0f}"
            )
        if var95 is not None:
            risks.append(f"1-day 95% VaR {var95*100:.2f}% (historical)")

        # Black-swan flag.
        if (vol is not None and vol > 0.6) or (mdd is not None and mdd < -0.5):
            risks.append("BLACK-SWAN FLAG: extreme tail risk in history — escalate")
            reanalysis = True

        return [self._report("risk_volatility", score, confidence=75,
                            evidence=ev or ["Insufficient risk data"],
                            risks=risks or ["None material"],
                            reanalysis=reanalysis)]
