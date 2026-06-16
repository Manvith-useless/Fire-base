# Orion Capital — Technical Investment Committee

A disciplined, **technical-only** investment committee for Indian equities, wired to
your **Zerodha (Kite)** account. It pulls live price history, computes indicators
**deterministically in Python** (never by an LLM), runs a small committee of
reasoning modules, and produces a CIO-style verdict — with optional, **confirmation-
gated** order placement.

> This is decision-support software, not financial advice. It analyses price and
> portfolio data only. It does **not** model fundamentals, earnings, news or macro,
> so a negative surprise outside price action is unmodelled. Trade at your own risk.

## Why it looks different from the original 39-agent spec

The original "Orion Capital" concept listed 39 agents, most of which need data
Zerodha cannot supply (P/E, balance sheet, earnings, management quality, macro). A
raw LLM asked for those numbers will **fabricate** them. This implementation makes
three deliberate adjustments:

1. **Scope to price/technical data** — every number is data-grounded and honest.
2. **Compute all metrics in Python** — the LLM only *interprets*, never calculates.
3. **Consolidate 39 agents → 8 modules** — cheaper, faster, no redundant theater.

### The 8 modules
CIO/Decision · Data & Quality · Technical & Quant · Market Regime ·
Risk/Scenarios/Stress · Portfolio & Sizing · Red Team · Learning/Memory

### Re-weighted scoring (technical-only)
Trend & Momentum 30% · Risk/Volatility 25% · Market Regime 15% ·
Portfolio Fit 15% · Liquidity 10% · Catalysts 5%

## Install

```bash
pip install -r requirements.txt
```

## Run offline (no account needed)

The `sample` provider generates reproducible synthetic candles so you can try the
full pipeline immediately:

```bash
python cli.py analyze INFY --provider sample
python cli.py review INFY        # past decisions log
```

## Connect your Zerodha account (live data)

Live data needs **Kite Connect** API credentials — a **paid** developer
subscription (~₹2000/month) created at <https://kite.trade>. This is separate from
your normal Zerodha login.

1. Create a Kite Connect app → copy `api_key` and `api_secret`.
2. `cp .env.example .env` and fill in `KITE_API_KEY`, `KITE_API_SECRET`,
   `ANTHROPIC_API_KEY`.
3. Authenticate (once per trading day):
   ```bash
   python cli.py login        # prints a login URL; paste back the request_token
   ```
4. Analyze with live data:
   ```bash
   python cli.py analyze INFY --provider kite
   ```

The access token is cached in `.kite_session.json` (git-ignored) for the day.

## Placing trades — you confirm every order

By design the system **never** auto-trades. With `--order-qty N` it prepares an
order proposal and prints a one-time confirmation token; the order is placed only
if you type that exact token in an interactive session:

```bash
python cli.py analyze INFY --provider kite --order-qty 10
```

HOLD / AVOID / NO ACTION verdicts propose nothing. The safety gate is enforced in
`orion/execution.py` and covered by `tests/test_execution_gate.py`.

## Configuration

Set in `.env` (see `.env.example`): `KITE_API_KEY`, `KITE_API_SECRET`,
`KITE_ACCESS_TOKEN` (auto), `ANTHROPIC_API_KEY`, and optional model overrides
`ORION_MODEL`, `ORION_CIO_MODEL`. Without `ANTHROPIC_API_KEY` the committee still
runs fully on its deterministic rule-based narratives.

## Tests

```bash
pytest -q
```

Covers indicator math against hand-checked values and the order-confirmation gate.

## Layout

```
cli.py                 entrypoint (analyze / login / review)
orion/
  config.py            env/settings
  schemas.py           reports, verdict, scoring weights
  llm.py               Anthropic wrapper (graceful offline fallback)
  context.py           AnalysisContext (computed facts shared by modules)
  data/                provider interface, kite client, sample provider
  indicators/          technical.py, risk.py (deterministic math)
  agents/              the 8 modules + CIO aggregation
  orchestrator.py      runs the pipeline
  execution.py         confirmation-gated order placement
  memory/decisions.jsonl   learning log (git-ignored)
tests/
```
