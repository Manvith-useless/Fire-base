"""CIO / Decision module — aggregates the committee into a final verdict.

Combines module scores by the re-weighted framework (renormalized over the
dimensions actually present), applies the spec's escalation rules, and produces a
CIO-style verdict with bull/base/bear and entry/exit considerations.
"""
from __future__ import annotations

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import (
    SCORING_WEIGHTS,
    AgentReport,
    FinalVerdict,
    Recommendation,
    recommendation_from_score,
    score_label,
)
from .base import fmt

# Confidence floor from the spec's escalation rules.
CONFIDENCE_FLOOR = 60.0


class CIO:
    name = "CIO / Decision"

    def decide(
        self, ctx: AnalysisContext, reports: list[AgentReport], llm: LLM
    ) -> FinalVerdict:
        # ---- Weighted aggregation over present dimensions --------------------
        dim_reports = {r.dimension: r for r in reports if r.dimension}
        total_w = sum(SCORING_WEIGHTS[d] for d in dim_reports)
        if total_w > 0:
            weighted = sum(
                SCORING_WEIGHTS[d] * rep.investment_score
                for d, rep in dim_reports.items()
            ) / total_w
        else:
            weighted = 0.0

        # Confidence = data-quality confidence blended with mean module confidence.
        scoring_conf = (
            sum(r.confidence for r in dim_reports.values()) / len(dim_reports)
            if dim_reports
            else 0.0
        )
        data_report = next(
            (r for r in reports if r.agent_name == "Data & Quality"), None
        )
        data_conf = data_report.confidence if data_report else scoring_conf
        confidence = round(0.5 * scoring_conf + 0.5 * data_conf, 1)

        # Conviction reflects agreement: penalize when modules disagree widely.
        scores = [r.investment_score for r in dim_reports.values()]
        spread = (max(scores) - min(scores)) if scores else 0.0
        conviction = round(max(0.0, confidence - spread * 0.5), 1)

        # ---- Escalation rules ------------------------------------------------
        escalations: list[str] = []
        reanalysis = any(r.reanalysis_required for r in reports)
        if confidence < CONFIDENCE_FLOOR:
            escalations.append(
                f"Confidence {confidence} < {CONFIDENCE_FLOOR:.0f} — re-analysis advised"
            )
            reanalysis = True
        if spread >= 40:
            escalations.append(
                f"Modules disagree (score spread {spread:.0f}) — arbitration applied"
            )
        for r in reports:
            for risk in r.risks:
                if "BLACK-SWAN" in risk:
                    escalations.append(f"{r.agent_name}: {risk}")

        verdict = recommendation_from_score(weighted)
        # Safety overrides: never issue BUY under low confidence or unresolved reanalysis.
        if verdict == Recommendation.BUY and (
            confidence < CONFIDENCE_FLOOR or reanalysis
        ):
            verdict = Recommendation.HOLD
            escalations.append("BUY downgraded to HOLD pending higher-confidence re-analysis")
        if weighted == 0.0:
            verdict = Recommendation.NO_ACTION

        risks_all = [risk for r in reports for risk in r.risks if risk != "None material"]

        bull, base, bear = self._scenarios(ctx)
        thesis = self._thesis(ctx, weighted, verdict, dim_reports)

        verdict_obj = FinalVerdict(
            symbol=ctx.instrument.symbol,
            market_environment=self._regime_text(dim_reports),
            sector_outlook="Not assessed (technical-only system; no sector/fundamental data)",
            thesis=thesis,
            bull_case=bull,
            base_case=base,
            bear_case=bear,
            expected_risks=risks_all[:8] or ["None material identified"],
            confidence_score=confidence,
            conviction_score=conviction,
            position_size_guidance=self._sizing_text(dim_reports),
            entry_considerations=self._entry_text(ctx),
            exit_considerations=self._exit_text(ctx),
            weighted_score=weighted,
            verdict=verdict,
            rationale=self._rationale(weighted, verdict, escalations),
            module_reports=reports,
        )

        # Optional LLM polish of the narrative (never changes the numbers).
        self._maybe_enrich(verdict_obj, llm)
        return verdict_obj

    # ---- narrative builders (deterministic fallbacks) -----------------------
    def _scenarios(self, ctx: AnalysisContext) -> tuple[str, str, str]:
        price = ctx.ltp or ctx.technical.get("close")
        vol = ctx.risk.get("annualized_volatility")
        if price and vol:
            return (
                f"Trend continuation; ~+{vol*100:.0f}% to ≈{price*(1+vol):.0f} over ~1y.",
                f"Range-bound around {price:.0f}; returns track the broad market.",
                f"Trend break; ~-{vol*100:.0f}% to ≈{price*(1-vol):.0f}, deeper if regime turns.",
            )
        return ("Upside on trend continuation.", "Sideways.", "Downside on trend break.")

    def _regime_text(self, dim_reports: dict) -> str:
        r = dim_reports.get("market_regime")
        if r:
            return "; ".join(r.evidence[:2])
        return "Regime not assessed"

    def _thesis(self, ctx, weighted, verdict, dim_reports) -> str:
        return (
            f"{ctx.instrument.symbol}: composite technical score {weighted:.1f}/100 "
            f"({score_label(weighted)}) → {verdict.value}. Driven by trend/momentum, "
            f"risk and regime signals computed from price history only."
        )

    def _sizing_text(self, dim_reports: dict) -> str:
        r = dim_reports.get("portfolio_fit")
        if r:
            sizing = [e for e in r.evidence if "size" in e.lower() or "budget" in e.lower()]
            if sizing:
                return sizing[0]
        return "Size conservatively; cap single-name exposure (≤10% of cash)."

    def _entry_text(self, ctx: AnalysisContext) -> str:
        t = ctx.technical
        sma20 = t.get("sma_20")
        atr = t.get("atr_14")
        parts = []
        if sma20:
            parts.append(f"Prefer entries near SMA20 ({fmt(sma20)}) on pullbacks")
        if atr:
            parts.append(f"Size stops around ATR ({fmt(atr)})")
        return "; ".join(parts) or "Stagger entries; avoid chasing extended prices."

    def _exit_text(self, ctx: AnalysisContext) -> str:
        atr = ctx.technical.get("atr_14")
        mdd = ctx.risk.get("max_drawdown")
        parts = ["Define a stop before entry"]
        if atr:
            parts.append(f"e.g. 2×ATR ({fmt(2*atr)}) below entry")
        if mdd is not None:
            parts.append(f"history shows drawdowns to {mdd*100:.0f}%")
        return "; ".join(parts)

    def _rationale(self, weighted, verdict, escalations) -> str:
        base = f"Weighted technical score {weighted:.1f} maps to {verdict.value}."
        if escalations:
            return base + " Escalations: " + " | ".join(escalations)
        return base + " No escalations triggered."

    def _maybe_enrich(self, v: FinalVerdict, llm: LLM) -> None:
        if not llm.available:
            return
        facts = "\n".join(
            f"- {r.agent_name} [{r.dimension or 'qualitative'}]: score "
            f"{r.investment_score}, conf {r.confidence}; "
            f"evidence: {'; '.join(r.evidence[:3])}; risks: {'; '.join(r.risks[:3])}"
            for r in v.module_reports
        )
        system = (
            "You are the CIO of a disciplined investment committee. You interpret "
            "ONLY the numbers given; never invent data. Be concise and candid about "
            "uncertainty. Output 2-4 sentences for the investment thesis."
        )
        prompt = (
            f"Symbol {v.symbol}. Weighted score {v.weighted_score:.1f} -> "
            f"{v.verdict.value}. Module findings:\n{facts}\n\n"
            "Write a tight CIO thesis paragraph."
        )
        text = llm.interpret(system, prompt, model=llm.settings.cio_model)
        if text:
            v.thesis = text
