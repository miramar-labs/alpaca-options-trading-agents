import json
import time
from datetime import datetime, time as dtime, timedelta
from uuid import uuid4

import pytz
from kubernetes.client.exceptions import ApiException

from src.common import langsmith, slack, symbols
from src.common.alpaca_client import trading_client
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.portfolio_state import merge_held_positions, read_portfolio
from src.dealer.graph import build_graph
from src.dealer.mcp_options import reset_options_tools_cache

log = get_logger("DEALER")

_last_market_open = None  # edge-detects open/closed transitions so Slack gets one notice per
                          # transition, not a repeat every poll cycle while the market stays closed
_last_symbol_bases_refresh = 0.0  # monotonic timestamp; gates refresh_symbol_bases_if_due() below


def _now_et() -> datetime:
    return datetime.now(pytz.timezone("US/Eastern"))


def is_after_open_buffer(buffer_minutes: int) -> bool:
    eastern = pytz.timezone("US/Eastern")
    now_et = _now_et()

    market_open_naive = datetime.combine(now_et.date(), dtime(9, 30))
    market_open_et = eastern.localize(market_open_naive)

    adj_open_et = market_open_et + timedelta(minutes=buffer_minutes)
    return now_et >= adj_open_et


def should_process_entry(entry: dict, cfg) -> bool:
    is_crypto = entry["exchange"] != "stocks"
    if is_crypto and not cfg.trading.crypto.enabled:
        return False
    if not is_crypto and not cfg.trading.stocks.enabled:
        return False
    return True


def market_is_open(cfg, log) -> bool:
    global _last_market_open

    if cfg.trading.market_override:
        log("📈 OVERRIDE: stock market is OPEN.")
        _last_market_open = True
        return True

    clock = trading_client.get_clock()
    if clock.is_open:
        _last_market_open = True
        if not is_after_open_buffer(cfg.trading.buffer):
            log(f"📈 stock market is OPEN but we are waiting {cfg.trading.buffer} minutes to avoid volatility")
            return False
        log("📈 stock market is OPEN.")
        return True

    log("🔒 stock market is CLOSED.")
    log(f"next open: {clock.next_open}")
    log(f"next close: {clock.next_close}")
    if _last_market_open is not False:
        eastern = pytz.timezone("US/Eastern")
        next_open_et = clock.next_open.astimezone(eastern).strftime("%Y-%m-%d %I:%M %p %Z")
        slack.notify_stock_market_closed(next_open_et)
    _last_market_open = False
    return False


def refresh_symbol_bases_if_due() -> None:
    """Dealer has no dedicated poll thread (unlike Floor Broker's poll_symbol_bases()) -- it's a
    single loop already gated by cfg.trading.pollsecs (10 min by default), well under
    symbols.REFRESH_INTERVAL_S. This throttles the Alpaca asset-list refresh to that shared
    interval so merge_held_positions()'s crypto-symbol canonicalization stays correct without
    hitting Alpaca on every dealer poll cycle."""
    global _last_symbol_bases_refresh
    now = time.monotonic()
    if now - _last_symbol_bases_refresh < symbols.REFRESH_INTERVAL_S:
        return
    try:
        count = symbols.refresh_known_usd_crypto_bases_from_alpaca()
        log(f"🔄  refreshed known USD crypto bases from Alpaca ({count} base(s))")
    except Exception as exc:
        log(f"💥 symbol-bases refresh failed: {exc}")
    _last_symbol_bases_refresh = now


def main():
    langsmith.configure(load_config())

    graph = build_graph()

    while True:
        cfg = load_config()  # reloaded every poll cycle so a live config change never needs a restart
        reset_options_tools_cache()  # MCP tool list is stable within a cycle; rebuild once per cycle
        refresh_symbol_bases_if_due()
        if market_is_open(cfg, log):
            try:
                portfolio = merge_held_positions(read_portfolio(), cfg)
            except (ApiException, json.JSONDecodeError) as exc:
                log(f"⚠️ BAD portfolio read .. cannot proceed: {exc}")
                slack.notify_error("DEALER", f"portfolio read failed: {exc}")
                time.sleep(60)
                continue

            taapi_calls = 0
            for entry in portfolio.get("symbols", []):
                if not should_process_entry(entry, cfg):
                    continue
                if taapi_calls > 0:
                    # TAAPI's free plan allows 1 request/15s -- fetch_indicators makes exactly
                    # one bulk request per symbol, so spacing symbols out here keeps the whole
                    # loop under whatever plan's rate limit is configured.
                    time.sleep(cfg.taapi.min_request_interval_secs)
                taapi_calls += 1
                try:
                    state = {
                        "symbol": entry["symbol"],
                        "exchange": entry["exchange"],
                        "budget": entry["budget"],
                        "indicator_names": entry["indicators"],
                        "indicators_text": "",
                        "cycle_id": str(uuid4()),
                        "raw_bars": {},
                        "ohlcv_features_text": "",
                        "signal": None,
                        "option_pick": None,
                        "execution_result": None,
                    }
                    graph.invoke(state, config={"tags": ["dealer"]})
                except Exception as exc:
                    log(f"💥 failed processing {entry['symbol']}: {exc}")
                    slack.notify_error("DEALER", f"{entry['symbol']}: {exc}")
                    continue

        log(f"------------------ PAUSED FOR {cfg.trading.pollsecs}s -----------------------")
        time.sleep(cfg.trading.pollsecs)


if __name__ == "__main__":
    main()
