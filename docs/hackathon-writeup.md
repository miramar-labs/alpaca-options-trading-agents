# Alpaca AI Trading Agents Hackathon — Submission Write-up

**Event:** Alpaca AI Trading Agents Hackathon (lablab.ai), 28 Aug – 4 Sep 2026
**Repo:** [miramar-labs/alpaca-options-trading-agents](https://github.com/miramar-labs/alpaca-options-trading-agents)
**Account:** Alpaca's competition $100k paper account

## What it is

A three-agent trading floor that autonomously trades **options** on US equities. Every
morning it picks a watchlist; every ten minutes during market hours it decides BUY/HOLD/SELL
on each symbol, and — the core of this submission — hands every non-HOLD signal to an
**Alpaca MCP tool-calling agent** that browses the live option chain and picks a specific
contract (strike, expiration, right) to trade, instead of trading the underlying stock
directly. A separate execution service places the order, watches it, and manages its own
synthetic stop-loss/take-profit/expiration exits since Alpaca has no server-side bracket orders
for options.

Everything runs live, continuously, on a Kubernetes cluster, against Alpaca's competition
paper account — not a backtest, not a demo script run once for a screen recording.

## What's new here

This builds on an existing multi-agent equity-trading framework of mine — the Analyst /
Dealer / Floor Broker structure, the layered risk gates, and the config-as-URL loader
predate the hackathon and trade the underlying stock directly. The hackathon work is the
options capability: the Alpaca MCP contract-selection agent (the core of this submission),
the options execution path in the Floor Broker, options-specific risk gates and the synthetic
stop-loss / take-profit / expiration exits, the end-of-day Slack report, and packaging the
whole thing as a live, continuously-trading deployment. Everything in this repo's
[commit history](https://github.com/miramar-labs/alpaca-options-trading-agents/commits/main)
is that work, done during the competition week. Crypto trading, a backtest harness, and a
nightly power scheduler exist in the upstream framework but were left out as out of scope.

## The problem

Turning a directional stock call ("this looks like a buy") into an actual options trade is a
second, harder decision: which expiration, which strike, is the premium sane, does the delta
match the intended exposure. That's normally a human options trader's job. This project puts
an LLM agent in that seat, giving it live read access to Alpaca's option chain/quotes/Greeks
through MCP and letting it reason its way to a specific contract — while a deterministic,
non-LLM layer underneath re-validates every constraint (DTE window, delta window, notional cap,
duplicate check) before anything actually executes, so a bad or hallucinated model output can't
directly place a bad trade.

## Architecture at a glance

```
Analyst (CronJob, 08:55 ET)  →  portfolio ConfigMap  →  Dealer (poll loop, every 600s)
                                                              │
                                                   BUY/HOLD/SELL signal (LLM)
                                                              │
                                              non-HOLD  ──────┘
                                                 │
                                    Alpaca MCP tool-calling loop
                                    (option chain → pick a contract)
                                                 │
                                    Floor Broker (FastAPI) → Alpaca
                                    server-side re-validation, order
                                    placement, synthetic SL/TP/DTE exit
```

Full technical detail — every node, every gate, the config reference, the DB schema — is in
[`docs/architecture.md`](architecture.md). The short version:

- **Analyst** — a LangGraph pipeline: screens Alpaca movers + a fixed large-cap list, pulls
  news and TAAPI technical indicators, reads back its own recent track record from Postgres,
  and asks an LLM to pick the day's watchlist with a budget and rationale per symbol.
- **Dealer** — a long-running poll loop. For each watchlisted symbol it fetches indicators and
  OHLCV-derived market structure, asks an LLM for BUY/HOLD/SELL, and for any non-HOLD signal
  launches the MCP contract-selection agent below.
- **Floor Broker** — a stateless FastAPI service that is the only component allowed to touch
  Alpaca's trading API. It re-validates every constraint server-side, places the order, and
  runs three background poll loops: fill detection, synthetic option stop-loss/take-profit/
  DTE-force-close, and restart recovery (rebuilding all in-memory tracking from Alpaca + Postgres
  if the pod restarts mid-position).
- **EOD Report** — a daily CronJob that posts a plain-language account/trade recap to Slack,
  independent of the trading loop above.

## The Alpaca MCP integration

This is the feature the whole submission is built around, so it's worth calling out on its
own. `src/dealer/mcp_options.py` launches the `alpaca-mcp-server` package over stdio
(`langchain-mcp-adapters`' `MultiServerMCPClient`) with a **read-only** toolset
(`assets,options-data,account`) — the Dealer can look at the option chain, quotes, and Greeks,
but it can never place an order itself; only Floor Broker can do that, over a separate plain
REST call.

Given the underlying symbol, the desired direction (call for a bullish signal, put for a
bearish one — options here are always a long position, never a short), and the config's DTE
and delta windows, the LLM drives a bounded tool-calling loop (≤6 rounds) to pull the live
chain and settle on one contract, returned as structured output: strike, expiration, delta,
mid-price premium, and its reasoning. That pick is then:

1. **Validated** — rejected outright if it claims the wrong option type for the signal's
   direction, or names a contract that was never actually seen in a chain response (a
   hallucination).
2. **Reconciled** — its strike/expiration/delta/premium are overwritten with what the chain
   actually showed for that exact contract; only the reasoning text is trusted from the model.
3. **Re-validated a second time**, server-side in Floor Broker, against the live DTE/delta
   windows and a live re-quote of the ask price before an order is submitted.

## Risk management

Layered gates sit between "the LLM said BUY" and "an order reaches Alpaca": a confidence
threshold on the signal itself, a macro-event blackout calendar (FOMC/CPI/jobs/PCE dates), a
same-symbol stop-loss cooldown, a portfolio win-rate throttle, a daily profit-target/loss-limit
halt, an operator-flippable kill switch (no redeploy needed), a DTE/delta window
re-validation, and a hard notional cap enforced against the market rather than the model's
claimed price. Once a position is open, Alpaca has no server-side bracket order for options, so
Floor Broker runs its own 30-second poll loop that force-closes on a 50% loss, a 175% gain, or
expiration dropping to 3 days out — whichever comes first, and regardless of whether the
system's ability to *open* new positions is currently gated.

## Deployment

The whole stack runs on a DGX Spark k3s cluster, deployed by a GitHub Actions CI/CD chain on a
self-hosted runner: every push runs tests + lint, a green run on `main` builds and pushes four
container images to GHCR, and a green build triggers a rolling deploy to the
`alpaca-options-trader` namespace. Config (`config.yaml`) is fetched fresh from GitHub at
runtime by every pod every 60 seconds rather than baked into the image, so most tuning —
risk thresholds, blackout dates, even the LLM model — is a plain `git push`, no rebuild or
redeploy required. The README's live P/L and "Dealer LLM" badges are refreshed the same way, by
small scheduled jobs that read the account and `config.yaml` and commit the result.

## Tech stack

- **LangGraph** for both agents' decision graphs; **LangChain** (`ChatOpenAI` against an
  OpenAI-compatible endpoint) plus `langchain-mcp-adapters` for the MCP tool-calling loop
- **Alpaca** (`alpaca-py`) — trading, market data, news, options chain/quotes/Greeks, all
  against the one paper account
- **alpaca-mcp-server** — the official MCP server exposing Alpaca's tools, run locally as a
  stdio subprocess per Dealer poll
- **Ollama**, self-hosted on a DGX Spark, serving `qwen3.6:35b-a3b` as the LLM behind every
  agent decision — no external LLM API calls
- **FastAPI** (Floor Broker), **Postgres** (shared platform instance — audit trail, track
  record, DB-backed reconciliation), **Kubernetes/k3s**, **GitHub Actions** (CI/CD + badges),
  **TAAPI.io** (technical indicators), **Finnhub** (earnings calendar), **Slack** (notifications)

## Challenges & learnings

**Structured output on a local model, under an agentic loop, is the hard part — not the MCP
plumbing itself.** The MCP tool-calling loop genuinely works end-to-end: the agent calls the
right tools, pulls real chain data, and reasons about it. But `qwen3.6:35b-a3b`'s
`.with_structured_output()` call at the *end* of that loop sometimes returns an empty result
rather than a populated `OptionContractPick`, even though the tool-call history it just
produced contains everything needed to answer. This looks like an Ollama constrained-decoding
interaction with a long, tool-heavy context more than a pure model-capability gap — smaller,
single-turn structured-output calls (the Dealer's own BUY/HOLD/SELL signal, for instance) don't
show the same failure. Rather than let that empty result mean "no trade happens," the system
falls back to a deterministic contract picker (`_fallback_pick`) that selects directly from the
chain rows the loop already fetched, using the same delta/DTE/quote gates the LLM was asked to
apply — so a structured-output miss degrades to "the code picks the contract the same way a
human would skim the chain" rather than silently doing nothing. A larger model and/or moving
from Ollama's JSON-mode to a guided-decoding backend (vLLM/SGLang/NIM) is the likely real fix;
that evaluation was underway but didn't land inside the competition window, so this submission
ships with the fallback as a first-class, expected code path rather than a rare edge case.

**Everything downstream of a model call needs to assume the model is adversarial, not just
wrong.** The validate → reconcile → re-validate chain around the option pick wasn't originally
written with "the LLM is hostile" in mind — it grew out of ordinary defensive programming
against a locally-hosted model's occasional bad output. But by the time it was done, it also
happens to block the more interesting failure modes: a pick that names a real contract but
invents a fabricated premium, or a call-labeled pick whose OCC symbol is actually a put. None of
that trust boundary is unique to a competition setting — it's the same posture the pre-existing
non-hackathon version of this system already needed for real paper-money trades.

## Results

*Filled in once several trading days' worth of live contracts have opened and closed on the
competition account — see `options_trades` in Postgres for the ledger this section is sourced
from. Deploy went live 2 Sep 2026; check back after 3-4 Sep for realized entries/exits, win
rate, and the live P/L badges above in the README.*

## Repo

MIT-licensed, public: [github.com/miramar-labs/alpaca-options-trading-agents](https://github.com/miramar-labs/alpaca-options-trading-agents).
Development happened as real, dated commits across the competition week — see the
[commit history](https://github.com/miramar-labs/alpaca-options-trading-agents/commits/main)
and the live Test/Lint + Build/Push CI badges in the README.
