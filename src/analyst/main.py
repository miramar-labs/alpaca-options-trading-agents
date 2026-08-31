import os
from datetime import datetime

import pytz

from src.analyst.graph import build_graph
from src.common import langsmith, slack
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.market_calendar import is_stock_market_open

log = get_logger("ANALYST")


def main():
    cfg = load_config()
    langsmith.configure(cfg)

    is_midday_run = os.environ.get("ANALYST_RUN_LABEL") == "midday"
    if is_midday_run and not cfg.analyst.midday_run.enabled:
        log("⏭️ midday run disabled via config — exiting without a graph run")
        return

    today = datetime.now(pytz.timezone("US/Eastern")).date()
    stock_market_open = is_stock_market_open(today)
    if not stock_market_open:
        # Unlike EOD Report, Analyst must not skip its whole run on a closed day -- crypto
        # trades 24/7 and still needs today's picks. Only the stock half of discover_candidates
        # is gated on this; see AnalystState.stock_market_open.
        log(f"📅 {today} stock market is closed — stocks skipped, crypto picks still run")

    try:
        graph = build_graph()
        result = graph.invoke(
            {
                "raw_candidates": [],
                "research_text": "",
                "indicator_text": "",
                "track_record_text": "",
                "pnl_text": "",
                "selection": None,
                "stock_market_open": stock_market_open,
                "is_midday_run": is_midday_run,
            },
            config={"tags": ["analyst", "midday"] if is_midday_run else ["analyst"]},
        )
        selection = result.get("selection") or {}
        log(f"wrote portfolio with {len(selection.get('symbols', []))} symbols")
    except Exception as exc:
        log(f"💥 Analyst run failed: {exc}")
        slack.notify_error("ANALYST", str(exc))
        raise


if __name__ == "__main__":
    main()
