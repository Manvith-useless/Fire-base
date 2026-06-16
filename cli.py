#!/usr/bin/env python3
"""Orion Capital CLI.

Usage:
  python cli.py analyze INFY [--provider sample|kite] [--exchange NSE] [--order-qty N]
  python cli.py login
  python cli.py review [SYMBOL]
"""
from __future__ import annotations

import argparse
import sys

from orion.config import load_settings
from orion.data.provider import Instrument
from orion.data.sample_provider import SampleDataProvider
from orion.execution import OrderRejected, build_order_proposal, execute_proposal
from orion.schemas import FinalVerdict, score_label


def _get_provider(name: str, settings, snapshot_file: str | None = None):
    if name == "sample":
        return SampleDataProvider()
    if name == "snapshot":
        if not snapshot_file:
            raise SystemExit("--snapshot-file is required for --provider snapshot")
        from orion.data.snapshot_provider import SnapshotDataProvider

        return SnapshotDataProvider(snapshot_file)
    if name == "kite":
        from orion.data.kite_client import KiteClient

        return KiteClient(settings)
    raise SystemExit(f"Unknown provider: {name}")


def _print_verdict(v: FinalVerdict) -> None:
    line = "=" * 70
    print(f"\n{line}\nORION CAPITAL — {v.symbol}\n{line}")
    print(f"Verdict           : {v.verdict.value}  ({score_label(v.weighted_score)})")
    print(f"Weighted score    : {v.weighted_score:.1f}/100")
    print(f"Confidence        : {v.confidence_score:.1f}/100")
    print(f"Conviction        : {v.conviction_score:.1f}/100")
    if v.action_line:
        print(f"\n>>> ACTION        : {v.action_line}")
    print(f"\nMarket environment: {v.market_environment}")
    print(f"Thesis            : {v.thesis}")
    print(f"\nBull case : {v.bull_case}")
    print(f"Base case : {v.base_case}")
    print(f"Bear case : {v.bear_case}")
    print("\nExpected risks:")
    for r in v.expected_risks:
        print(f"  - {r}")
    print(f"\nPosition size : {v.position_size_guidance}")
    print(f"Entry         : {v.entry_considerations}")
    print(f"Exit          : {v.exit_considerations}")
    print(f"Rationale     : {v.rationale}")
    if v.entry_price:
        rpct = (v.risk_per_share / v.entry_price * 100) if v.entry_price else 0
        print("\n--- Trade plan (long, ATR-based) ---")
        print(f"  Entry      : ~{v.entry_price}")
        print(f"  Stop-loss  : {v.stop_loss}   (risk {v.risk_per_share}/sh, -{rpct:.1f}%)")
        print(f"  Target 1   : {v.target1}   (+{v.risk_per_share:.1f}, 1:1 R:R)")
        print(f"  Target 2   : {v.target2}   (+{2*v.risk_per_share:.1f}, 1:2 R:R)")
        print(f"  Target 3   : {v.target3}   (+{3*v.risk_per_share:.1f}, 1:3 R:R)")
    print("\n--- Module reports ---")
    for rep in v.module_reports:
        dim = f"[{rep.dimension}]" if rep.dimension else "[qualitative]"
        print(f"\n{rep.agent_name} {dim}")
        print(f"  Score {rep.investment_score} | Confidence {rep.confidence} "
              f"| {rep.recommendation.value}"
              f"{' | REANALYSIS' if rep.reanalysis_required else ''}")
        for e in rep.evidence:
            print(f"  + {e}")
        for r in rep.risks:
            print(f"  ! {r}")
    print(line)


def cmd_analyze(args) -> int:
    settings = load_settings()
    provider = _get_provider(args.provider, settings, getattr(args, "snapshot_file", None))
    instrument = Instrument(symbol=args.symbol.upper(), exchange=args.exchange)

    from orion.orchestrator import analyze

    verdict = analyze(provider, instrument, settings)
    _print_verdict(verdict)

    if args.order_qty:
        proposal = build_order_proposal(verdict, instrument, quantity=args.order_qty)
        if proposal is None:
            print(f"\nNo order proposed (verdict is {verdict.verdict.value}).")
            return 0
        print("\n" + "#" * 70)
        print("ORDER PROPOSAL (NOT YET PLACED)")
        print(f"  {proposal.summary()}")
        print(f"  Rationale: {proposal.rationale}")
        print(f"  Confirmation token: {proposal.confirm_token}")
        print("#" * 70)
        if not sys.stdin.isatty():
            print("Non-interactive session — order NOT placed.")
            return 0
        entered = input("Type the token to place this LIVE order (Enter to skip): ")
        try:
            order_id = execute_proposal(provider, proposal, entered)
            print(f"Order placed. Broker order id: {order_id}")
        except OrderRejected as e:
            print(f"Skipped: {e}")
        except NotImplementedError as e:
            print(f"Cannot place order with this provider: {e}")
    return 0


def cmd_login(args) -> int:
    settings = load_settings()
    if not settings.kite_api_key or not settings.kite_api_secret:
        print("Set KITE_API_KEY and KITE_API_SECRET in .env first.")
        return 1
    from orion.data.kite_client import KiteClient

    print("1. Open this URL and log in to Zerodha:")
    print("   " + KiteClient.login_url(settings))
    print("2. After login you are redirected to your app's redirect URL with a")
    print("   `request_token=...` query parameter. Paste that token below.")
    request_token = input("request_token: ").strip()
    KiteClient.complete_login(settings, request_token)
    print("Access token cached in .kite_session.json (valid for the trading day).")
    return 0


def cmd_review(args) -> int:
    from orion.agents.learning import load_decisions

    decisions = load_decisions(args.symbol.upper() if args.symbol else None)
    if not decisions:
        print("No past decisions recorded yet.")
        return 0
    for d in decisions:
        print(f"{d.get('timestamp', '?')} | {d.get('symbol')} | "
              f"{d.get('verdict')} | score {d.get('weighted_score')} | "
              f"conf {d.get('confidence_score')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orion Capital — technical investment committee")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Analyze a symbol")
    p_an.add_argument("symbol")
    p_an.add_argument("--provider", default="sample",
                      choices=["sample", "kite", "snapshot"])
    p_an.add_argument("--snapshot-file", default=None,
                      help="Path to a JSON snapshot (for --provider snapshot)")
    p_an.add_argument("--exchange", default="NSE")
    p_an.add_argument("--order-qty", type=int, default=0,
                      help="Propose an order of this quantity (still requires confirmation)")
    p_an.set_defaults(func=cmd_analyze)

    p_login = sub.add_parser("login", help="Authenticate with Zerodha Kite")
    p_login.set_defaults(func=cmd_login)

    p_rev = sub.add_parser("review", help="Show past decisions")
    p_rev.add_argument("symbol", nargs="?")
    p_rev.set_defaults(func=cmd_review)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
