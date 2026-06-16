"""Portfolio & Sizing module.

Uses the user's real holdings and available cash to judge portfolio fit, position
sizing and opportunity cost.
"""
from __future__ import annotations

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport
from .base import Agent

# Risk-budget: max fraction of available cash to put in a single new position.
MAX_SINGLE_POSITION = 0.10


class PortfolioSizingAgent(Agent):
    name = "Portfolio & Sizing"

    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        score = 55.0
        ev: list[str] = []
        risks: list[str] = []

        cash = ctx.available_cash
        held = ctx.held_quantity()
        price = ctx.ltp or ctx.technical.get("close")

        if cash > 0:
            ev.append(f"Available cash: {cash:,.0f}")
            budget = cash * MAX_SINGLE_POSITION
            if price:
                max_qty = int(budget // price)
                ev.append(
                    f"Risk-budgeted size: {max_qty} share(s) "
                    f"(≤{MAX_SINGLE_POSITION*100:.0f}% of cash ≈ {budget:,.0f})"
                )
                if max_qty == 0:
                    score -= 15
                    risks.append("Position too small to be meaningful given cash")
        else:
            score -= 10
            risks.append("No available-cash data — sizing is indicative only")

        if held > 0:
            ev.append(f"Existing position: {held} share(s) already held")
            vol = ctx.risk.get("annualized_volatility")
            if vol and vol > 0.4:
                score -= 8
                risks.append("Adding to an already-held high-volatility name raises concentration risk")
            else:
                score += 5
        else:
            ev.append("No existing position — fresh allocation")

        # Opportunity cost: a positive risk-adjusted profile competes well for capital.
        sharpe = ctx.risk.get("sharpe")
        if sharpe is not None:
            if sharpe > 1:
                score += 10
                ev.append(f"Attractive risk-adjusted return (Sharpe {sharpe:.2f}) vs holding cash")
            elif sharpe < 0:
                score -= 10
                risks.append(f"Negative risk-adjusted return (Sharpe {sharpe:.2f}) — cash may be better")

        return [self._report("portfolio_fit", score, confidence=68,
                            evidence=ev or ["Insufficient portfolio data"],
                            risks=risks or ["None material"])]
