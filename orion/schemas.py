"""Shared data structures and the scoring framework for Orion Capital.

The original "Orion Capital" spec demanded fundamental data (P/E, balance sheet,
earnings, management quality) that Zerodha cannot provide. This system is scoped
to price/technical data only, so the scoring weights are re-based onto factors we
can actually measure from price history and the user's portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Recommendation(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    AVOID = "AVOID"
    NO_ACTION = "NO ACTION"


# Re-weighted scoring framework (technical-only). Keys map to module dimensions.
SCORING_WEIGHTS: dict[str, float] = {
    "trend_momentum": 0.30,   # Technical & Quant
    "risk_volatility": 0.25,  # Risk, Scenarios & Stress
    "market_regime": 0.15,    # Market Regime
    "portfolio_fit": 0.15,    # Portfolio & Sizing
    "liquidity": 0.10,        # Volume / liquidity
    "catalysts": 0.05,        # Price-action catalysts (breakouts/levels)
}

assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

# Short-term / swing weighting: lean into momentum, catalysts and liquidity;
# de-emphasize the long-term (200-DMA) regime so a strong bounce isn't vetoed.
SHORT_TERM_WEIGHTS: dict[str, float] = {
    "trend_momentum": 0.40,
    "catalysts": 0.15,
    "liquidity": 0.15,
    "risk_volatility": 0.20,
    "market_regime": 0.05,
    "portfolio_fit": 0.05,
}

assert abs(sum(SHORT_TERM_WEIGHTS.values()) - 1.0) < 1e-9, "short weights must sum to 1.0"


def weights_for(horizon: str) -> dict[str, float]:
    return SHORT_TERM_WEIGHTS if horizon == "short" else SCORING_WEIGHTS


def recommendation_from_score(score: float) -> Recommendation:
    """Map a 0-100 score to a recommendation per the spec's scale."""
    if score >= 75:
        return Recommendation.BUY
    if score >= 50:
        return Recommendation.HOLD
    return Recommendation.AVOID


def score_label(score: float) -> str:
    """Human-readable band for a 0-100 score."""
    if score >= 90:
        return "Exceptional Opportunity"
    if score >= 85:
        return "Strong Buy"
    if score >= 75:
        return "Buy"
    if score >= 65:
        return "Watchlist"
    if score >= 50:
        return "Hold"
    return "Avoid"


@dataclass
class AgentReport:
    """Standard report every module emits (matches the spec's communication format)."""

    agent_name: str
    # Which scoring dimension this report contributes to (key in SCORING_WEIGHTS),
    # or None for modules that don't directly score (e.g. Data & Quality, Red Team).
    dimension: str | None
    investment_score: float  # 0-100
    confidence: float        # 0-100
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommendation: Recommendation = Recommendation.HOLD
    reanalysis_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recommendation"] = self.recommendation.value
        return d


@dataclass
class OrderProposal:
    """A structured trade proposal. Never executed without explicit confirmation."""

    symbol: str
    exchange: str
    transaction_type: str  # BUY / SELL
    quantity: int
    order_type: str        # MARKET / LIMIT
    product: str           # CNC / MIS / NRML
    price: float | None
    rationale: str
    confirm_token: str

    def summary(self) -> str:
        px = f"@ {self.price}" if self.price is not None else "@ MARKET"
        return (
            f"{self.transaction_type} {self.quantity} {self.exchange}:{self.symbol} "
            f"{px} ({self.order_type}/{self.product})"
        )


@dataclass
class FinalVerdict:
    symbol: str
    market_environment: str
    sector_outlook: str
    thesis: str
    bull_case: str
    base_case: str
    bear_case: str
    expected_risks: list[str]
    confidence_score: float
    conviction_score: float
    position_size_guidance: str
    entry_considerations: str
    exit_considerations: str
    weighted_score: float
    verdict: Recommendation
    rationale: str
    module_reports: list[AgentReport] = field(default_factory=list)
    order_proposal: OrderProposal | None = None
    # Plain-language call + concrete sizing for the user.
    action_line: str = ""
    suggested_quantity: int = 0
    suggested_amount: float = 0.0
    # Concrete trade levels (ATR-based long plan).
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    target3: float = 0.0
    risk_per_share: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weighted_score": round(self.weighted_score, 2),
            "verdict": self.verdict.value,
            "action_line": self.action_line,
            "suggested_quantity": self.suggested_quantity,
            "suggested_amount": round(self.suggested_amount, 2),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "targets": [self.target1, self.target2, self.target3],
            "confidence_score": round(self.confidence_score, 2),
            "conviction_score": round(self.conviction_score, 2),
            "thesis": self.thesis,
            "expected_risks": self.expected_risks,
            "module_reports": [r.to_dict() for r in self.module_reports],
        }
