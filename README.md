# alpaca-options-trading-agents

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
