"""The safety-critical guarantee: no order is placed without explicit confirmation."""
import pytest

from orion.data.provider import DataProvider, Instrument
from orion.execution import (
    OrderRejected,
    build_order_proposal,
    execute_proposal,
)
from orion.schemas import FinalVerdict, Recommendation


class RecordingProvider(DataProvider):
    name = "recording"

    def __init__(self):
        self.placed = []

    def get_candles(self, instrument, interval="day", days=400):
        raise NotImplementedError

    def get_ltp(self, instrument):
        return 100.0

    def get_holdings(self):
        return []

    def get_margins(self):
        return {}

    def place_order(self, **kwargs):
        self.placed.append(kwargs)
        return "ORDER123"


def _verdict(rec: Recommendation) -> FinalVerdict:
    return FinalVerdict(
        symbol="INFY", market_environment="", sector_outlook="", thesis="",
        bull_case="", base_case="", bear_case="", expected_risks=[],
        confidence_score=70, conviction_score=60, position_size_guidance="",
        entry_considerations="", exit_considerations="", weighted_score=80,
        verdict=rec, rationale="test",
    )


def test_hold_produces_no_proposal():
    inst = Instrument("INFY")
    assert build_order_proposal(_verdict(Recommendation.HOLD), inst, 10) is None
    assert build_order_proposal(_verdict(Recommendation.NO_ACTION), inst, 10) is None


def test_buy_produces_proposal_but_does_not_place():
    inst = Instrument("INFY")
    provider = RecordingProvider()
    proposal = build_order_proposal(_verdict(Recommendation.BUY), inst, 10)
    assert proposal is not None
    assert proposal.transaction_type == "BUY"
    # Building a proposal must never touch the broker.
    assert provider.placed == []


def test_wrong_token_rejected_and_nothing_placed():
    inst = Instrument("INFY")
    provider = RecordingProvider()
    proposal = build_order_proposal(_verdict(Recommendation.BUY), inst, 10)
    with pytest.raises(OrderRejected):
        execute_proposal(provider, proposal, "wrong-token")
    with pytest.raises(OrderRejected):
        execute_proposal(provider, proposal, "")
    assert provider.placed == []


def test_correct_token_places_exactly_one_order():
    inst = Instrument("INFY")
    provider = RecordingProvider()
    proposal = build_order_proposal(_verdict(Recommendation.BUY), inst, 10)
    order_id = execute_proposal(provider, proposal, proposal.confirm_token)
    assert order_id == "ORDER123"
    assert len(provider.placed) == 1
    assert provider.placed[0]["transaction_type"] == "BUY"
    assert provider.placed[0]["quantity"] == 10
