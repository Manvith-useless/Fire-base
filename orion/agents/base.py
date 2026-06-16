"""Agent base class.

Scoring is deterministic (derived from Python-computed indicators). The LLM is used
only to enrich evidence/risk narratives; when unavailable, a rule-based fallback is
used so the pipeline runs fully offline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport, Recommendation, recommendation_from_score


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def fmt(value: float | None, nd: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{nd}f}"


class Agent(ABC):
    name: str = "Agent"

    @abstractmethod
    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        """Return one or more AgentReports."""

    # Shared helper to build a report with optional LLM-enriched narrative.
    def _report(
        self,
        dimension: str | None,
        score: float,
        confidence: float,
        evidence: list[str],
        risks: list[str],
        reanalysis: bool = False,
    ) -> AgentReport:
        return AgentReport(
            agent_name=self.name,
            dimension=dimension,
            investment_score=round(clamp(score), 1),
            confidence=round(clamp(confidence), 1),
            evidence=evidence,
            risks=risks,
            recommendation=recommendation_from_score(score)
            if dimension
            else Recommendation.HOLD,
            reanalysis_required=reanalysis,
        )
