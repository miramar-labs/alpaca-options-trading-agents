import threading
import time

import uvicorn

from src.common import db, kill_switch, slack, symbols
from src.common.logging import get_logger
from src.floor_broker import execution

log = get_logger("FLOOR")

BRACKET_FILL_POLL_INTERVAL_S = 30
KILL_SWITCH_POLL_INTERVAL_S = 30
PENDING_FILL_POLL_INTERVAL_S = 30
RECONCILIATION_RETRY_INTERVAL_S = 60
EOD_FLATTEN_POLL_INTERVAL_S = 60
SYMBOL_BASES_POLL_INTERVAL_S = symbols.REFRESH_INTERVAL_S


def poll_bracket_fills():
    """Runs for the lifetime of the process, watching for TP/SL bracket legs that fill
    asynchronously on Alpaca's side -- outside of any /execute request/response cycle, so this is
    the only place those fills ever get observed and reported. Also reports a bracket that goes
    terminal with no fill at all (both legs canceled/expired/rejected) -- that's a silent gap
    otherwise, since no /execute response ever covers it.

    Also checks tracked crypto synthetic stop-loss/take-profit levels on the same cadence --
    crypto has no server-side bracket equivalent (Alpaca's bracket orders are equity-only), so
    this poller is the only place a crypto exit is ever triggered."""
    while True:
        try:
            for event in execution.check_bracket_fills():
                if event["kind"] == "fill":
                    log(f"🎯 {event['reason']} filled for {event['symbol']} @ {event['fill_price']}")
                    slack.notify_floor_broker_result(
                        event["symbol"],
                        "SELL",
                        "executed",
                        f"{event['reason']} leg filled: {event['order_id']}",
                        reason=event["reason"],
                        fill_price=event["fill_price"],
                    )
                    db.record_floor_broker_event(
                        event["symbol"], "fill", f"{event['reason']} leg filled: {event['order_id']}",
                        qty=event.get("qty"),
                        price=event["fill_price"],
                    )
                else:
                    statuses = "/".join(event["leg_statuses"])
                    log(f"⚠️ bracket {event['order_id']} for {event['symbol']} closed with no fill: {statuses}")
                    slack.notify_floor_broker_result(
                        event["symbol"],
                        "SELL",
                        "no_fill",
                        f"bracket legs closed with no fill ({statuses}): {event['order_id']}",
                    )
                    db.record_floor_broker_event(
                        event["symbol"], "no_fill", f"bracket legs closed with no fill ({statuses}): {event['order_id']}"
                    )
            for event in execution.check_crypto_stops():
                log(f"🎯 synthetic {event['reason']} triggered for {event['symbol']} @ {event['bid_price']}")
                slack.notify_floor_broker_result(
                    event["symbol"],
                    "SELL",
                    event["sell_result"]["status"],
                    f"synthetic {event['reason']} triggered @ {event['bid_price']}: {event['sell_result']['detail']}",
                    reason=event["reason"],
                )
                db.record_floor_broker_event(
                    event["symbol"],
                    f"synthetic_{event['reason']}",
                    event["sell_result"]["detail"],
                    price=event["bid_price"],
                )
            for event in execution.check_option_stops():
                log(f"🎯 synthetic {event['reason']} triggered for {event['contract_symbol']} @ {event['premium']}")
                slack.notify_floor_broker_result(
                    event["symbol"],
                    "SELL",
                    event["sell_result"]["status"],
                    f"synthetic {event['reason']} triggered @ {event['premium']}: {event['sell_result']['detail']}",
                    asset_class="option",
                    reason=event["reason"],
                )
                db.record_floor_broker_event(
                    event["symbol"],
                    f"synthetic_{event['reason']}",
                    event["sell_result"]["detail"],
                    price=event["premium"],
                )
        except Exception as exc:
            log(f"💥 bracket-fill poll failed: {exc}")
        time.sleep(BRACKET_FILL_POLL_INTERVAL_S)


def poll_pending_fills():
    """Runs for the lifetime of the process, watching for the fill of orders buy()/sell()
    themselves submitted (ROADMAP P0.14) -- /execute now returns status="submitted" before the
    fill is known, so this is the only place that fill is ever observed and reported. Also
    reports an order that goes terminal with no fill (canceled/expired/rejected) -- otherwise
    that outcome is silent, since no /execute response ever covers it either."""
    while True:
        try:
            for event in execution.check_pending_fills():
                if event["kind"] == "fill":
                    log(f"💰 {event['action']} filled for {event['symbol']} @ {event['fill_price']}")
                    slack.notify_floor_broker_result(
                        event["symbol"],
                        event["action"],
                        "executed",
                        f"{event['reason']} order filled: {event['order_id']}",
                        reason=event["reason"],
                        fill_price=event["fill_price"],
                        sl_price=event.get("sl_price"),
                        tp_price=event.get("tp_price"),
                    )
                    db.record_floor_broker_event(
                        event["symbol"],
                        "fill",
                        f"{event['reason']} order filled: {event['order_id']}",
                        qty=event.get("qty"),
                        price=event["fill_price"],
                    )
                    if event["action"] == "BUY":
                        db.record_position_opened(event["symbol"])
                    else:
                        db.record_position_closed(event["symbol"])
                else:
                    log(f"⚠️ {event['action']} {event['symbol']} closed with no fill: {event['order_status']}")
                    slack.notify_floor_broker_result(
                        event["symbol"],
                        event["action"],
                        "no_fill",
                        f"order {event['order_status']}, never filled: {event['order_id']}",
                        reason=event["reason"],
                    )
                    db.record_floor_broker_event(
                        event["symbol"], "no_fill", f"order {event['order_status']}, never filled: {event['order_id']}"
                    )
        except Exception as exc:
            log(f"💥 pending-fill poll failed: {exc}")
        time.sleep(PENDING_FILL_POLL_INTERVAL_S)


def poll_pending_option_fills():
    """Runs for the lifetime of the process, watching for the fill of option BUY orders
    buy_option() itself submitted -- mirrors poll_pending_fills() but tracks option contracts
    separately (by OCC symbol / synthetic exit mechanism), via
    execution.check_pending_option_fills(). _option_positions and the
    options_trades DB row are only written on a confirmed fill (see check_pending_option_fills()'s
    docstring); an order that never fills produces no phantom tracked state or DB row."""
    while True:
        try:
            for event in execution.check_pending_option_fills():
                action = event.get("action", "BUY")
                reason = event["reason"] if action == "SELL" else "opening_position"
                if event["kind"] == "fill":
                    log(f"💰 option {action} filled for {event['contract_symbol']} @ {event['fill_price']}")
                    slack.notify_floor_broker_result(
                        event["symbol"],
                        action,
                        "executed",
                        f"option {action.lower()} order filled: {event['order_id']}",
                        asset_class="option",
                        reason=reason,
                        fill_price=event["fill_price"],
                    )
                    db.record_floor_broker_event(
                        event["symbol"],
                        "fill",
                        f"option {action.lower()} order filled: {event['order_id']}",
                        qty=event.get("qty"),
                        price=event["fill_price"],
                    )
                else:
                    log(f"⚠️ option {action} {event['contract_symbol']} closed with no fill: {event['order_status']}")
                    slack.notify_floor_broker_result(
                        event["symbol"],
                        action,
                        "no_fill",
                        f"option order {event['order_status']}, never filled: {event['order_id']}",
                        asset_class="option",
                        reason=reason,
                    )
                    db.record_floor_broker_event(
                        event["symbol"], "no_fill", f"option order {event['order_status']}, never filled: {event['order_id']}"
                    )
        except Exception as exc:
            log(f"💥 pending-option-fill poll failed: {exc}")
        time.sleep(PENDING_FILL_POLL_INTERVAL_S)


def poll_eod_flatten():
    """Runs for the lifetime of the process. When strategy config enables eod_flatten and Alpaca's
    live clock reports the market is close to closing, sells every open stock position (crypto is
    24/7 and excluded, see execution.check_eod_flatten()). The immediate submit is reported here,
    the same way poll_bracket_fills() reports a triggered synthetic crypto stop -- the eventual
    fill/no-fill is reported separately by poll_pending_fills(), which already tracks every order
    sell() submits."""
    while True:
        try:
            for event in execution.check_eod_flatten():
                log(f"🌇 eod-flatten selling {event['symbol']}: {event['sell_result']['detail']}")
                slack.notify_floor_broker_result(
                    event["symbol"],
                    "SELL",
                    event["sell_result"]["status"],
                    event["sell_result"]["detail"],
                    reason=event["reason"],
                )
                db.record_floor_broker_event(event["symbol"], "eod_flatten", event["sell_result"]["detail"])
        except Exception as exc:
            log(f"💥 eod-flatten poll failed: {exc}")
        time.sleep(EOD_FLATTEN_POLL_INTERVAL_S)


def poll_kill_switch():
    """Watches the buy-kill-switch ConfigMap for a state change (ROADMAP P0.5) and posts a Slack
    notice only on transition, not on every poll -- `/execute` itself already re-checks the
    switch fresh on each BUY request, so this loop exists purely to surface the change as an
    operational event. `last_state` starts at None (not yet observed) so the very first poll,
    which is just discovering whatever state the switch was seeded/left in, never fires a
    transition notice on its own."""
    last_state = None
    while True:
        try:
            active = kill_switch.buy_kill_switch_active()
            if last_state is not None and active != last_state:
                log(f"🛑 BUY kill switch changed: {'ACTIVE' if active else 'inactive'}")
                slack.notify_buy_kill_switch(active)
            last_state = active
        except Exception as exc:
            log(f"💥 kill-switch poll failed: {exc}")
        time.sleep(KILL_SWITCH_POLL_INTERVAL_S)


def poll_reconciliation():
    """Runs only while startup's bounded reconstruct_tracked_state() retries were all exhausted --
    exits on its own as soon as reconciliation succeeds. Exists so a transient Alpaca outage that
    spans all of startup's retries doesn't leave BUY execution rejected for the pod's entire
    lifetime; keeps retrying reconcile_tracked_state_once() until Alpaca recovers."""
    while not execution.is_state_reconciled():
        time.sleep(RECONCILIATION_RETRY_INTERVAL_S)
        try:
            execution.reconcile_tracked_state_once()
        except Exception as exc:
            log(f"💥 background reconciliation attempt failed: {exc}")


def poll_symbol_bases():
    """Runs for the lifetime of the process, periodically refreshing the known-USD-crypto-base
    set (see src.common.symbols) from Alpaca's own tradable crypto asset list -- keeps
    canonical_crypto_symbol()/is_usd_crypto_symbol() correct as Alpaca lists new coins, without
    ever blocking the hot paths that call them on a live Alpaca request."""
    while True:
        try:
            count = symbols.refresh_known_usd_crypto_bases_from_alpaca()
            log(f"🔄  refreshed known USD crypto bases from Alpaca ({count} base(s))")
        except Exception as exc:
            log(f"💥 symbol-bases poll failed: {exc}")
        time.sleep(SYMBOL_BASES_POLL_INTERVAL_S)


def main():
    execution.reconstruct_tracked_state()
    threading.Thread(target=poll_reconciliation, daemon=True).start()
    threading.Thread(target=poll_bracket_fills, daemon=True).start()
    threading.Thread(target=poll_pending_fills, daemon=True).start()
    threading.Thread(target=poll_pending_option_fills, daemon=True).start()
    threading.Thread(target=poll_kill_switch, daemon=True).start()
    threading.Thread(target=poll_eod_flatten, daemon=True).start()
    threading.Thread(target=poll_symbol_bases, daemon=True).start()
    uvicorn.run("src.floor_broker.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
