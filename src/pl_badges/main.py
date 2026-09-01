import json
from datetime import datetime
from pathlib import Path

import pytz

from src.common.logging import get_logger
from src.common.market_calendar import is_stock_market_open
from src.common.pl_badges import build_badge_payload, fetch_pl_summary

log = get_logger("PL-BADGES")

BADGES_DIR = Path(__file__).resolve().parents[2] / "badges"
HISTORY_FILE = BADGES_DIR / "pl_history.json"


def _now_eastern() -> datetime:
    return datetime.now(pytz.timezone("US/Eastern"))


def _load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {}
    return json.loads(HISTORY_FILE.read_text())


def main():
    today = _now_eastern().date()

    if not is_stock_market_open(today):
        log(f"📅 {today} was not a trading day — leaving badges unchanged.")
        return

    history_pl = _load_history()
    summary = fetch_pl_summary(today, history_pl)
    BADGES_DIR.mkdir(exist_ok=True)
    (BADGES_DIR / "today-pl.json").write_text(json.dumps(build_badge_payload("Today's P/L", summary["today_pl"])))
    (BADGES_DIR / "ytd-pl.json").write_text(json.dumps(build_badge_payload("YTD P/L", summary["ytd_pl"])))

    history_pl[today.isoformat()] = summary["today_pl"]
    HISTORY_FILE.write_text(json.dumps(history_pl, indent=2, sort_keys=True))

    log(f"✅ badges written — today {summary['today_pl']:+.2f}, YTD {summary['ytd_pl']:+.2f}")


if __name__ == "__main__":
    main()
