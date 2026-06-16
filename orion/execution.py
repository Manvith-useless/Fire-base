"""Execution gate.

A trade is NEVER placed automatically. The flow is:
  1. build_order_proposal(...) -> OrderProposal (with a one-time confirm_token)
  2. the user is shown the proposal and must echo the exact token
  3. execute_proposal(...) verifies the token, THEN calls the provider

This module enforces "suggest orders, I confirm each" — the agreed safety model.
"""
from __future__ import annotations

import secrets

from .data.provider import DataProvider, Instrument
from .schemas import FinalVerdict, OrderProposal, Recommendation


class OrderRejected(Exception):
    """Raised when an order is attempted without a valid confirmation token."""


def build_order_proposal(
    verdict: FinalVerdict,
    instrument: Instrument,
    quantity: int,
    order_type: str = "MARKET",
    product: str = "CNC",
    price: float | None = None,
) -> OrderProposal | None:
    """Translate a BUY/SELL verdict into a concrete, unconfirmed proposal.

    Returns None for HOLD/AVOID/NO ACTION — there is nothing to execute.
    """
    if verdict.verdict not in (Recommendation.BUY, Recommendation.SELL):
        return None
    side = "BUY" if verdict.verdict == Recommendation.BUY else "SELL"
    return OrderProposal(
        symbol=instrument.symbol,
        exchange=instrument.exchange,
        transaction_type=side,
        quantity=quantity,
        order_type=order_type,
        product=product,
        price=price,
        rationale=verdict.rationale,
        confirm_token=secrets.token_hex(4),
    )


def execute_proposal(
    provider: DataProvider,
    proposal: OrderProposal,
    supplied_token: str,
) -> str:
    """Place the order ONLY if supplied_token matches the proposal's token.

    Raises OrderRejected otherwise. Returns the broker order id on success.
    """
    if not supplied_token or supplied_token.strip() != proposal.confirm_token:
        raise OrderRejected(
            "Confirmation token mismatch — order NOT placed. "
            "Re-run and confirm the exact token to execute."
        )
    return provider.place_order(
        variety="regular",
        exchange=proposal.exchange,
        tradingsymbol=proposal.symbol,
        transaction_type=proposal.transaction_type,
        quantity=proposal.quantity,
        product=proposal.product,
        order_type=proposal.order_type,
        price=proposal.price,
        tag="orion",
    )
