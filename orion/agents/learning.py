"""Learning / Memory module — persists decisions and (later) their outcomes.

Append-only JSONL so the committee can review recurring mistakes over time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import FinalVerdict

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
DECISIONS_FILE = MEMORY_DIR / "decisions.jsonl"


def record_decision(verdict: FinalVerdict, provider: str) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        **verdict.to_dict(),
    }
    with DECISIONS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_decisions(symbol: str | None = None) -> list[dict]:
    if not DECISIONS_FILE.exists():
        return []
    out = []
    for line in DECISIONS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if symbol is None or rec.get("symbol") == symbol:
            out.append(rec)
    return out
