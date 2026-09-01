# alpaca-options-trading-agents

[![Test and Lint](https://github.com/miramar-labs/alpaca-options-trading-agents/actions/workflows/test-lint.yaml/badge.svg)](https://github.com/miramar-labs/alpaca-options-trading-agents/actions/workflows/test-lint.yaml)
[![Build and Push](https://github.com/miramar-labs/alpaca-options-trading-agents/actions/workflows/build-push.yaml/badge.svg)](https://github.com/miramar-labs/alpaca-options-trading-agents/actions/workflows/build-push.yaml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Today's P/L](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/miramar-labs/alpaca-options-trading-agents/main/badges/today-pl.json)](https://app.alpaca.markets/paper/dashboard/overview)
[![YTD P/L](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/miramar-labs/alpaca-options-trading-agents/main/badges/ytd-pl.json)](https://app.alpaca.markets/paper/dashboard/overview)
[![Dealer LLM](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/miramar-labs/alpaca-options-trading-agents/main/badges/model.json)](config.yaml)

A 3-agent options-trading floor for the **Alpaca AI Trading Agents Hackathon**
(28 Aug – 4 Sep 2026):

- **Analyst** — each morning, screens the market and picks the day's tradeable universe.
- **Dealer** — polls each symbol for a BUY / SELL / HOLD signal, then uses an
  Alpaca MCP tool-calling loop to select the specific option contract to trade.
- **Floor Broker** — a FastAPI executor that places the order on Alpaca and
  manages synthetic stop-loss / take-profit exits.

Risk gates, an end-of-day Slack recap, and live P/L + model badges round it out.
The stack runs on the DGX k3s cluster against an Alpaca paper account.

## Development

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pytest -q
.venv/bin/ruff check .
```

More detail — architecture, deployment, and the hackathon write-up — lands in
`docs/` as the build progresses.
