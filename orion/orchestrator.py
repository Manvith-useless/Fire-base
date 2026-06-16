"""Pipeline: build the data-grounded context, run the committee, return a verdict."""
from __future__ import annotations

from .agents.cio import CIO
from .agents.data_quality import DataQualityAgent
from .agents.learning import record_decision
from .agents.market_regime import MarketRegimeAgent
from .agents.portfolio_sizing import PortfolioSizingAgent
from .agents.red_team import RedTeamAgent
from .agents.risk_scenarios import RiskScenariosAgent
from .agents.technical_quant import TechnicalQuantAgent
from .config import Settings
from .context import AnalysisContext
from .data.provider import DataProvider, Instrument
from .indicators import risk as risk_ind
from .indicators import technical as tech_ind
from .llm import LLM
from .schemas import FinalVerdict

# Order matters only for presentation; scoring is independent per module.
ANALYSIS_AGENTS = [
    DataQualityAgent(),
    TechnicalQuantAgent(),
    MarketRegimeAgent(),
    RiskScenariosAgent(),
    PortfolioSizingAgent(),
    RedTeamAgent(),
]


def build_context(
    provider: DataProvider, instrument: Instrument, days: int = 400
) -> AnalysisContext:
    candles = provider.get_candles(instrument, interval="day", days=days)
    technical = tech_ind.compute_snapshot(candles) if not candles.empty else {}
    risk = risk_ind.compute_snapshot(candles["close"]) if not candles.empty else {}

    # Portfolio data is best-effort (may be empty if not authenticated).
    try:
        holdings = provider.get_holdings()
    except Exception:
        holdings = []
    try:
        margins = provider.get_margins()
    except Exception:
        margins = {}
    try:
        ltp = provider.get_ltp(instrument)
    except Exception:
        ltp = technical.get("close")

    return AnalysisContext(
        instrument=instrument,
        candles=candles,
        technical=technical,
        risk=risk,
        ltp=ltp,
        holdings=holdings,
        margins=margins,
    )


def analyze(
    provider: DataProvider,
    instrument: Instrument,
    settings: Settings,
    persist: bool = True,
    horizon: str = "position",
    capital: float | None = None,
) -> FinalVerdict:
    llm = LLM(settings)
    ctx = build_context(provider, instrument)

    # Let the user model a specific bankroll (e.g. a planned top-up).
    if capital is not None:
        ctx.margins = {"equity": {"available": {"live_balance": float(capital)}}}

    reports = []
    for agent in ANALYSIS_AGENTS:
        reports.extend(agent.run(ctx, llm))

    verdict = CIO().decide(ctx, reports, llm, horizon=horizon)
    if persist:
        record_decision(verdict, provider.name)
    return verdict
