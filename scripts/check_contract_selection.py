"""Live diagnostic: does the LLM actually pick an option contract, or does it fall back?

Runs the real Dealer contract-selection code path -- ``_select_option_contract_async``
-- against the live Alpaca option chain, exactly as a non-HOLD Dealer signal would:
the ``alpaca-mcp-server`` subprocess over stdio, the read-only tool set, the bounded
tool-calling loop, and the closing ``with_structured_output(OptionContractPick)`` call.

It bypasses the signal / duplicate / risk gates because it calls the selector
directly, so it never places an order and does not need the profit-target halt
lifted. Use it to confirm a model change (see ``docs/models.md``) before trusting
it in the live loop.

It ships in the dealer image. Run it inside the dealer pod, where the MCP server,
Ollama endpoint and Alpaca credentials are already wired up::

    kubectl -n alpaca-options-trader exec deploy/dealer -- \\
        python scripts/check_contract_selection.py NVDA BUY

``fell_back=False`` means the model returned a valid structured pick. ``fell_back=True``
means ``_fallback_pick`` had to salvage it (empty or rejected structured output) --
the same silent degradation that ``qwen3.6:35b-a3b`` hit on every cycle.

Args: ``SYMBOL`` (default AAPL), ``ACTION`` (BUY -> call, SELL -> put; default BUY),
optional ``--model`` to override ``config.yaml``'s ``llm.model`` for the run.
"""

import argparse
import asyncio
import time

from src.common.config import load_config
from src.dealer.graph import _select_option_contract_async

_FALLBACK_MARKER = "deterministic fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("symbol", nargs="?", default="AAPL")
    parser.add_argument("action", nargs="?", default="BUY", choices=["BUY", "SELL"])
    parser.add_argument(
        "--model",
        default=None,
        help="override config.yaml llm.model for this run (e.g. a candidate model)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg.llm.model = args.model
    print(f">>> model={cfg.llm.model} symbol={args.symbol} action={args.action}", flush=True)

    state = {"symbol": args.symbol}
    signal = {
        "action": args.action,
        "reasoning": (
            "Diagnostic run: technicals point the stated direction "
            "(RSI mid-range, MACD crossed, constructive structure)."
        ),
    }

    t0 = time.time()
    try:
        pick = asyncio.run(_select_option_contract_async(state, cfg, signal))
    except Exception as exc:  # noqa: BLE001 -- diagnostic; surface any failure verbatim
        print(f">>> ERROR in {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
        return 2

    dt = time.time() - t0
    if pick is None:
        print(f">>> RESULT in {dt:.1f}s: None (no pick at all)", flush=True)
        return 1

    fell_back = _FALLBACK_MARKER in (pick.reasoning or "")
    print(f">>> RESULT in {dt:.1f}s  fell_back={fell_back}", flush=True)
    print(f">>> {pick.model_dump_json()}", flush=True)
    return 1 if fell_back else 0


if __name__ == "__main__":
    raise SystemExit(main())
