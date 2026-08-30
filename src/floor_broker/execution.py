import json
import re
import threading
import time
from datetime import datetime

import pytz
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from alpaca.trading.requests import (
    GetOrderByIdRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.common import db, kill_switch
from src.common.alpaca_client import (
    get_current_ask_price,
    get_current_bid_price,
    get_current_option_ask_price,
    get_current_option_mid_price,
    trading_client,
)
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.symbols import alpaca_order_symbol, canonical_crypto_symbol, is_usd_crypto_symbol

log = get_logger("FLOOR")

MIN_CRYPTO_NOTIONAL = 10.0  # Alpaca rejects a crypto notional below this (code 40310000)

ORDER_NOT_FOUND_CODE = 40410000  # Alpaca's code for "no order exists with that id"

_TERMINAL_NO_FILL = {OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}

# Options terminal/non-terminal classification for check_pending_option_fills(). Deliberately NOT
# shared with _TERMINAL_NO_FILL (the stock/crypto check_pending_fills() uses that one and is out
# of scope for the options mechanism). Everything NOT in this set is terminal for
# options pending-fill purposes -- fix-loop round 2, finding 1 (remaining half): the prior
# `!= PARTIALLY_FILLED` check correctly handled the with-fill branch but the no-fill branch's
# `in _TERMINAL_NO_FILL` only caught CANCELED/EXPIRED/REJECTED, leaking any other zero-fill
# terminal status (DONE_FOR_DAY, REPLACED, STOPPED, SUSPENDED) forever. Verified against every
# member of alpaca.trading.enums.OrderStatus (17 total).
_OPTION_NON_TERMINAL = {
    OrderStatus.NEW,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.ACCEPTED,
    OrderStatus.PENDING_NEW,
    OrderStatus.ACCEPTED_FOR_BIDDING,
    OrderStatus.HELD,
    OrderStatus.PENDING_CANCEL,
    OrderStatus.PENDING_REPLACE,
    OrderStatus.PENDING_REVIEW,
}

_RECONCILE_MAX_STARTUP_ATTEMPTS = 5
_RECONCILE_BACKOFF_BASE_S = 5.0


class InvalidOrderParameters(Exception):
    """Raised when a stock bracket order's computed quantity/prices fail the invariant checks
    below before submission (ROADMAP P0.9) -- a stale/zero quote or an inverted SL/TP
    relationship should never reach Alpaca as a live order."""


class InsufficientQuantity(InvalidOrderParameters):
    """The specific InvalidOrderParameters case where the budget affords less than one whole
    share at the reference price -- not a bug, just too little budget for the current price, so
    buy() turns this into a normal status="skipped" outcome rather than propagating."""


class NoAskQuote(Exception):
    """Raised when Alpaca has no executable ask for a symbol. This is a market-data/no-liquidity
    condition, not a malformed order parameter."""


# Tracks the parent order id of each open bracket BUY, keyed by symbol, so the fill-watcher
# (check_bracket_fills) can later find out which of its TP/SL legs eventually filled. Value is
# either a plain order-id string (the normal case) or, once a poll has hit a transient error for
# that symbol, {"order_id": str, "poll_failures": int}. In-memory only -- reconstruct_tracked_state()
# rebuilds this from Alpaca's own open-orders state on process start, but only for brackets still
# open on Alpaca; a bracket that fills in the gap between the pod dying and reconstruct running
# still produces no Slack notice for that one fill (the trade itself executes fine either way).
_tracked_brackets: dict[str, str | dict] = {}

# Tracks every order buy()/sell() itself submitted, keyed by order id, so check_pending_fills()
# can later report that order's own fill (ROADMAP P0.14) -- distinct from _tracked_brackets
# above, which is only for a bracket's *child* TP/SL legs. Same in-memory + reconstruct-on-start
# caveat as _tracked_brackets: an order that fills in the gap between the pod dying and
# reconstruct_tracked_state() running on the new pod still produces no Slack notice for that fill.
_pending_fills: dict[str, dict] = {}

# Synthetic stop-loss/take-profit for open crypto positions, keyed by symbol, value (sl_price,
# tp_price). Alpaca's bracket/OCO orders are equity-only (alpaca.trading.enums.OrderClass:
# "Crypto trading: simple (or \"\")"), so crypto has no server-side SL/TP at all -- this dict plus
# check_crypto_stops() below is the entire mechanism. In-memory only, same restart caveat as
# _tracked_brackets/_pending_fills: a Floor Broker restart drops tracking for any crypto position
# still open at the time, silently losing its stop/target until the next manual Dealer SELL.
_crypto_stops: dict[str, tuple[float, float]] = {}

# Tracks every open option position this process itself opened, keyed by contract_symbol, value
# {symbol, right, strike, expiration, delta, entry_premium, qty}. Options have no native bracket/
# OCO support any more than crypto does, so check_option_stops() below is the entire synthetic
# exit mechanism for options -- same in-memory-only, no-restart-recovery caveat as _crypto_stops.
_option_positions: dict[str, dict] = {}

# Tracks every option BUY order buy_option() itself submitted, keyed by order id, so
# check_pending_option_fills() can later confirm the fill and only then commit the position into
# _option_positions and record it to the DB -- mirrors _pending_fills/check_pending_fills() for the
# stock/crypto path. It's a separate dict/poller because option fills carry the OCC contract symbol
# and feed the synthetic option-exit mechanism, not because they're on a different account (every
# order is on the one live account). In-memory only, same restart caveat as _pending_fills: an
# option BUY that fills in the gap between the pod dying and reconstruct_tracked_state() running
# still produces no Slack notice for that fill.
_pending_option_fills: dict[str, dict] = {}

# False from process start until reconcile_tracked_state_once() has succeeded at least once.
# buy() refuses new BUYs while this is False (see below) -- submitting a fresh order before
# Alpaca's live open-order state has been reconciled into _pending_fills/_tracked_brackets risks
# losing track of it exactly like the restart gap this whole mechanism exists to close.
_state_reconciled = False
_state_lock = threading.RLock()


def _is_order_not_found(exc: APIError) -> bool:
    """True only for a confirmed "no such order" response. Any other shape -- a genuinely
    different code, or a non-JSON/malformed error body (e.g. a raw network exception) -- must be
    treated as transient rather than assumed to mean not-found, since dropping tracked state
    should require positive confirmation, not just an unparseable error."""
    try:
        return exc.code == ORDER_NOT_FOUND_CODE
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return False


def _is_position_not_found_error(exc: APIError) -> bool:
    """True only for a confirmed HTTP 404 (no such position). Any other status -- or an error shape
    with no discoverable HTTP status -- must be treated as transient rather than assumed to mean
    not-found, since dropping tracked state should require positive confirmation, not just an
    unparseable or non-404 error."""
    return exc.status_code == 404


def _pending_fills_snapshot() -> list[tuple[str, dict]]:
    with _state_lock:
        return [(order_id, ctx.copy()) for order_id, ctx in _pending_fills.items()]


def _tracked_brackets_snapshot() -> list[tuple[str, str | dict]]:
    with _state_lock:
        return [(symbol, entry.copy() if isinstance(entry, dict) else entry) for symbol, entry in _tracked_brackets.items()]


def _crypto_stops_snapshot() -> list[tuple[str, tuple[float, float]]]:
    with _state_lock:
        return list(_crypto_stops.items())


def _drop_pending_fill(order_id: str) -> None:
    with _state_lock:
        _pending_fills.pop(order_id, None)


def _increment_pending_poll_failures(order_id: str) -> int:
    with _state_lock:
        ctx = _pending_fills.get(order_id)
        if ctx is None:
            return 0
        ctx["poll_failures"] = ctx.get("poll_failures", 0) + 1
        return ctx["poll_failures"]


def _clear_pending_poll_failures(order_id: str) -> None:
    with _state_lock:
        ctx = _pending_fills.get(order_id)
        if ctx is not None:
            ctx.pop("poll_failures", None)


def _track_crypto_stop(symbol: str, sl_price: float, tp_price: float) -> None:
    with _state_lock:
        _crypto_stops[symbol] = (sl_price, tp_price)


def _drop_crypto_stop_if_current(symbol: str, sl_price: float, tp_price: float) -> bool:
    with _state_lock:
        if _crypto_stops.get(symbol) != (sl_price, tp_price):
            return False
        _crypto_stops.pop(symbol, None)
        return True


def _drop_tracked_bracket(symbol: str) -> None:
    with _state_lock:
        _tracked_brackets.pop(symbol, None)


def _set_tracked_bracket(symbol: str, entry: str | dict) -> None:
    with _state_lock:
        _tracked_brackets[symbol] = entry


def _bracket_poll_failures(symbol: str) -> int:
    with _state_lock:
        entry = _tracked_brackets.get(symbol)
        return entry["poll_failures"] if isinstance(entry, dict) else 0


def _set_pending_fill(order_id: str, ctx: dict) -> None:
    with _state_lock:
        _pending_fills[order_id] = ctx


def _pending_option_fills_snapshot() -> list[tuple[str, dict]]:
    with _state_lock:
        return [(order_id, ctx.copy()) for order_id, ctx in _pending_option_fills.items()]


def _drop_pending_option_fill(order_id: str) -> None:
    with _state_lock:
        _pending_option_fills.pop(order_id, None)


def _increment_pending_option_poll_failures(order_id: str) -> int:
    with _state_lock:
        ctx = _pending_option_fills.get(order_id)
        if ctx is None:
            return 0
        ctx["poll_failures"] = ctx.get("poll_failures", 0) + 1
        return ctx["poll_failures"]


def _clear_pending_option_poll_failures(order_id: str) -> None:
    with _state_lock:
        ctx = _pending_option_fills.get(order_id)
        if ctx is not None:
            ctx.pop("poll_failures", None)


def _set_pending_option_fill(order_id: str, ctx: dict) -> None:
    with _state_lock:
        _pending_option_fills[order_id] = ctx


def _drop_pending_option_fills_for_contract(contract_symbol: str) -> None:
    """Drops every _pending_option_fills entry tracking contract_symbol -- called at every
    sell_option() exit that drops (or, since external review finding 1, is about to replace) tracking
    for a contract (fix-loop round 1, finding 3). A partially-filled BUY order stays in
    _pending_option_fills until fully terminal (finding 1), so there's a window where
    check_option_stops() can sell the already-filled portion of a position via sell_option() while an
    outstanding partial-fill BUY for that same contract is still tracked here. Left alone, the next
    check_pending_option_fills() poll would resurrect _option_positions for a contract whose sell is
    already in flight (or, worse, already sold and DB-closed), leaving a phantom in-memory-only
    position with no DB backing. Deliberately does not touch the underlying Alpaca order -- no
    cancel_related_orders() call -- this is in-memory tracking cleanup only; the order itself is
    left to resolve on its own (self-limiting: the next poll or sell_option() call finds no position
    and pops).

    Must be called while _state_lock is already held (every call site in sell_option() already holds
    it around this call)."""
    stale_order_ids = [order_id for order_id, ctx in _pending_option_fills.items() if ctx.get("contract_symbol") == contract_symbol]
    for order_id in stale_order_ids:
        _pending_option_fills.pop(order_id, None)


def _stop_tracking_symbol(symbol: str) -> None:
    with _state_lock:
        _tracked_brackets.pop(symbol, None)
        _crypto_stops.pop(symbol, None)


def check_pending_fills() -> list[dict]:
    """Polls every order buy()/sell() submitted for its own fill -- /execute now returns
    status="submitted" before this is known (ROADMAP P0.14), so this is the only place a
    submitted order's fill is ever observed and reported.

    A transient APIError (rate limit, timeout, Alpaca-side 5xx) must not drop the entry -- that
    would silently stop watching a live order. Only a confirmed 404 (the order genuinely no
    longer exists) removes it without a fill ever being observed."""
    events = []
    for order_id, ctx in _pending_fills_snapshot():
        try:
            order = trading_client.get_order_by_id(order_id)
        except APIError as exc:
            if _is_order_not_found(exc):
                log(f"⚠️  pending order {order_id} ({ctx['symbol']}) no longer exists on Alpaca -- dropping")
                _drop_pending_fill(order_id)
            else:
                failures = _increment_pending_poll_failures(order_id)
                log(f"💥  poll failure #{failures} for pending order {order_id} ({ctx['symbol']}): {exc}")
            continue

        _clear_pending_poll_failures(order_id)
        ctx.pop("poll_failures", None)

        if order.filled_avg_price is not None:
            fill_price = float(order.filled_avg_price)
            if ctx.get("crypto_slP") is not None:
                sl_price = fill_price * ctx["crypto_slP"]
                tp_price = fill_price * ctx["crypto_tpP"]
                _track_crypto_stop(ctx["symbol"], sl_price, tp_price)
                log(f"🎯  tracking synthetic stop/target for {ctx['symbol']}: {(sl_price, tp_price)}")
            event = {**ctx, "kind": "fill", "order_id": order_id, "fill_price": fill_price}
            if getattr(order, "filled_qty", None):
                event["qty"] = float(order.filled_qty)
            events.append(event)
            _drop_pending_fill(order_id)
        elif order.status in _TERMINAL_NO_FILL:
            events.append({**ctx, "kind": "terminal", "order_id": order_id, "order_status": order.status.value})
            _drop_pending_fill(order_id)

    return events


def check_pending_option_fills() -> list[dict]:
    """Polls every pending option order for its own fill -- both BUY orders buy_option() itself
    submitted and, since external review finding 1, SELL orders sell_option() submitted -- mirroring
    check_pending_fills(). It's a separate dict/poller because option fills carry the OCC contract
    symbol and feed the synthetic option-exit mechanism, not because they're on a different account
    (every order is on the one live account). Entries are distinguished by ctx["action"] ("BUY" or
    "SELL"), each handled by its own branch below.

    BUY branch: _option_positions and the options_trades DB row are only written here, on a confirmed
    fill, using the real fill_price -- never in buy_option() itself, which only knows the pre-trade
    quoted ask. This is what stops a rejected/canceled/unfilled option BUY from leaving behind phantom
    tracked state, and what keeps check_option_stops()'s synthetic SL/TP math anchored to the real
    entry price rather than a stale pre-trade quote.

    SELL branch (external review finding 1): mirrors the BUY branch's confirmed-fill gating so an
    option exit is only reported/DB-closed once Alpaca actually confirms the fill, not on submit --
    sell_option() itself no longer pops _option_positions or closes the DB row synchronously; it only
    registers the pending SELL here. A terminal no-fill SELL (canceled/rejected/expired) leaves
    _option_positions tracked, exactly as if the sell had never been attempted. Unlike the BUY branch,
    a PARTIALLY_FILLED SELL whose order is still non-terminal is left alone entirely -- no DB write,
    no event, no state change -- until the order finishes filling or goes terminal; closing on the
    first partial fill would drop protection for real still-open qty at Alpaca (external review
    finding 2, 2026-08-26 round 2). And once such a SELL does go terminal without having filled the
    whole tracked qty (e.g. canceled after only 1 of 2 contracts sold), the position is not popped
    wholesale -- only the sold qty is subtracted from _option_positions, the DB row is updated (not
    closed) with the remaining qty, and the position stays tracked/protected for the remainder
    (external review finding 1, 2026-08-26 round 3).

    Same transient-vs-terminal distinction as check_pending_fills: a non-404 APIError keeps the order
    tracked and just records the failure; only a confirmed 404 drops it without a fill ever observed.

    A PARTIALLY_FILLED BUY order updates _option_positions and the DB row on every poll but stays in
    _pending_option_fills -- the first observation of any fill (partial or full) does the initial
    DB INSERT (db.record_options_trade_opened) and emits the one "fill" event/Slack notification for
    this order id; every later observation of the same still-open order instead does a DB UPDATE
    (db.record_options_trade_updated) and emits no new event, keyed off ctx["db_row_opened"] (which
    may already be seeded True by reconcile_tracked_state_once() if this order was reconstructed for
    a contract that already had an open DB row -- see finding 2, fix-loop round 1). The entry is only
    dropped from _pending_option_fills once order.status leaves _OPTION_NON_TERMINAL (fix-loop round
    2, finding 1's remaining half) -- this now covers every real Alpaca terminal status, with or
    without a fill (e.g. done_for_day, replaced, stopped, suspended), not just
    FILLED/CANCELED/EXPIRED/REJECTED, so an unusual status never leaks this entry forever. The same
    set is used for the zero-fill branch below, so PENDING_CANCEL/PENDING_REPLACE/PENDING_REVIEW are
    correctly kept non-terminal in both branches.

    check_pending_option_fills() re-checks _pending_option_fills membership under _state_lock
    immediately before writing _option_positions or making the terminal-drop/reseed decision
    (fix-loop round 2, new issue A) -- get_order_by_id() above is a network call made outside any
    lock, so a concurrent sell_option() can drop this order's tracking in that window; if it did,
    this poll skips the order entirely rather than resurrecting a phantom position or orphaning a
    DB row."""
    events = []
    for order_id, ctx in _pending_option_fills_snapshot():
        try:
            order = trading_client.get_order_by_id(order_id)
        except APIError as exc:
            if _is_order_not_found(exc):
                log(f"⚠️  pending option order {order_id} ({ctx['contract_symbol']}) no longer exists on Alpaca -- dropping")
                _drop_pending_option_fill(order_id)
            else:
                failures = _increment_pending_option_poll_failures(order_id)
                log(f"💥  poll failure #{failures} for pending option order {order_id} ({ctx['contract_symbol']}): {exc}")
            continue

        _clear_pending_option_poll_failures(order_id)
        ctx.pop("poll_failures", None)

        if ctx.get("action") == "SELL":
            # Only a confirmed *terminal* fill (or a zero-fill terminal status) resolves this SELL --
            # a PARTIALLY_FILLED SELL whose order is still open (still in _OPTION_NON_TERMINAL) leaves
            # real qty still open at Alpaca; treating that first partial fill as done would drop
            # check_option_stops()'s protection for the unsold remainder and report the trade closed
            # while it isn't (external review finding 2, 2026-08-26 round 2 -- the first version of
            # this fix, external review finding 1, 2026-08-26, closed _option_positions/the DB row on
            # ANY observed fill, partial or not). This entry is deliberately left tracked and
            # untouched in that case -- the next poll re-checks the same order until it either
            # finishes filling or goes terminal.
            is_terminal = order.status not in _OPTION_NON_TERMINAL
            if order.filled_avg_price is not None:
                if not is_terminal:
                    continue
                fill_price = float(order.filled_avg_price)
                filled_qty = int(float(order.filled_qty)) if getattr(order, "filled_qty", None) else None
                with _state_lock:
                    if order_id not in _pending_option_fills:
                        # A concurrent poll or another sell_option() call already handled this order
                        # -- same resurrection guard as the BUY branch below (fix-loop round 2, new
                        # issue A).
                        continue
                    position = _option_positions.get(ctx["contract_symbol"])
                    tracked_qty = position["qty"] if position else None
                    entry_premium = position["entry_premium"] if position else None
                    # A terminal SELL (e.g. the remainder got canceled/expired after only partially
                    # filling) doesn't necessarily mean the whole tracked position sold -- sell_option()
                    # always sizes its order to the full tracked qty at submit time, so filled_qty short
                    # of tracked_qty means real qty is still open at Alpaca (external review finding 1,
                    # 2026-08-26 round 3). remaining_qty defaults to 0 (treated as a full close) if
                    # either qty is unknown -- the same "when in doubt, don't leave a phantom untracked
                    # position half-closed" default the pre-round-3 code always used.
                    remaining_qty = tracked_qty - filled_qty if tracked_qty is not None and filled_qty is not None else 0
                    if remaining_qty > 0:
                        position["qty"] = remaining_qty
                        fully_closed = False
                    else:
                        _option_positions.pop(ctx["contract_symbol"], None)
                        fully_closed = True
                    _pending_option_fills.pop(order_id, None)
                if fully_closed:
                    db.record_options_trade_closed(ctx["contract_symbol"], ctx["reason"], fill_price)
                else:
                    log(
                        f"⚠️  option SELL {order_id} for {ctx['contract_symbol']} went terminal after only "
                        f"partially filling ({filled_qty}/{tracked_qty}) -- {remaining_qty} still open at "
                        "Alpaca, position stays tracked for the remainder"
                    )
                    # entry_premium (cost basis of the still-open remainder), not fill_price (this
                    # SELL's exit price), is what belongs in the still-open row -- record_options_trade_
                    # updated() sets both columns together (it's designed for a BUY's running average
                    # cost as it fills further), so entry_premium must be re-passed unchanged here.
                    db.record_options_trade_updated(ctx["contract_symbol"], entry_premium, remaining_qty)
                event = {**ctx, "kind": "fill", "order_id": order_id, "fill_price": fill_price}
                if filled_qty is not None:
                    event["qty"] = float(filled_qty)
                events.append(event)
            elif is_terminal:
                with _state_lock:
                    _pending_option_fills.pop(order_id, None)
                events.append({**ctx, "kind": "terminal", "order_id": order_id, "order_status": order.status.value})
            continue

        if order.filled_avg_price is not None:
            fill_price = float(order.filled_avg_price)
            filled_qty = int(float(order.filled_qty)) if getattr(order, "filled_qty", None) else ctx["qty"]
            db_row_was_open = ctx.get("db_row_opened", False)
            # Snapshot the fill event now, before ctx is possibly mutated below (db_row_opened is
            # set True inside the lock for a still-open partial fill) -- this preserves the
            # pre-round-2 event shape (no db_row_opened key ever leaking into the emitted event).
            fill_event = None if db_row_was_open else {**ctx, "kind": "fill", "order_id": order_id, "fill_price": fill_price, "qty": filled_qty}
            # Terminal here means order.status has left _OPTION_NON_TERMINAL: a full fill, or a
            # partial fill that will never grow further because the order itself resolved
            # (canceled/expired/rejected/done_for_day/replaced/stopped/suspended after partially
            # filling). Fix-loop round 2, finding 1's remaining half.
            is_terminal = order.status not in _OPTION_NON_TERMINAL
            with _state_lock:
                if order_id not in _pending_option_fills:
                    # A concurrent sell_option() (or another poll) already dropped this contract's
                    # tracking between the get_order_by_id() call above and this lock -- the
                    # position was sold out from under this poll. Do not resurrect
                    # _option_positions/_pending_option_fills, and do not write a DB row for it
                    # (fix-loop round 2, new issue A).
                    continue
                _option_positions[ctx["contract_symbol"]] = {
                    "symbol": ctx["symbol"],
                    "right": ctx["right"],
                    "strike": ctx["strike"],
                    "expiration": ctx["expiration"],
                    "delta": ctx["delta"],
                    "entry_premium": fill_price,
                    "qty": filled_qty,
                }
                if is_terminal:
                    _pending_option_fills.pop(order_id, None)
                else:
                    ctx["db_row_opened"] = True
                    _pending_option_fills[order_id] = ctx

            if db_row_was_open:
                db.record_options_trade_updated(ctx["contract_symbol"], fill_price, filled_qty)
            else:
                # Re-check _option_positions right before the INSERT (external review finding 2,
                # 2026-08-26): the in-memory write above happened under _state_lock, released before
                # this DB call, so a concurrent sell_option() + confirmed SELL fill could already have
                # closed this same contract in the gap, whose db.record_options_trade_closed() UPDATE
                # then no-ops (no open row exists yet) -- inserting here afterward would leave a
                # phantom "open" row for a position that's actually closed. Skipping the INSERT when
                # the position's no longer tracked avoids that phantom row; the tradeoff is losing
                # this one trade's DB record entirely in that narrow window, which is judged better
                # than a permanently-wrong open row. Still emits the fill event either way so the
                # BUY's own fill is never silently dropped from Slack/notifications.
                with _state_lock:
                    still_open = ctx["contract_symbol"] in _option_positions
                if still_open:
                    db.record_options_trade_opened(
                        ctx["symbol"], ctx["contract_symbol"], ctx["right"], ctx["strike"], ctx["expiration"],
                        ctx["delta"], fill_price, filled_qty, ctx["reasoning"], ctx["cycle_id"],
                    )
                else:
                    log(f"⚠️  option {ctx['contract_symbol']} BUY fill confirmed but position was already closed by a concurrent SELL -- skipping stale DB insert")
                # Emit the fill event (and therefore the one Slack/DB notification) only on the
                # first fill observed for this order -- otherwise a slowly-filling partial order
                # would re-notify on every poll tick until it finishes.
                events.append(fill_event)
        elif order.status not in _OPTION_NON_TERMINAL:
            events.append({**ctx, "kind": "terminal", "order_id": order_id, "order_status": order.status.value})
            _drop_pending_option_fill(order_id)

    return events


def check_bracket_fills() -> list[dict]:
    """Polls every tracked bracket BUY for a TP or SL leg that has since filled. A bracket's two
    child legs are OCO (one-cancels-other) on Alpaca's side -- once either fills, the other is
    auto-cancelled, so a symbol is untracked as soon as either outcome is observed.

    Same transient-vs-terminal distinction as check_pending_fills: a non-404 APIError keeps the
    symbol tracked and just records the failure."""
    events = []
    for symbol, entry in _tracked_brackets_snapshot():
        order_id = entry["order_id"] if isinstance(entry, dict) else entry
        try:
            order = trading_client.get_order_by_id(order_id, filter=GetOrderByIdRequest(nested=True))
        except APIError as exc:
            if _is_order_not_found(exc):
                log(f"⚠️  tracked bracket order {order_id} ({symbol}) no longer exists on Alpaca -- dropping")
                _drop_tracked_bracket(symbol)
            else:
                failures = _bracket_poll_failures(symbol) + 1
                _set_tracked_bracket(symbol, {"order_id": order_id, "poll_failures": failures})
                log(f"💥  poll failure #{failures} for tracked bracket {order_id} ({symbol}): {exc}")
            continue

        legs = order.legs or []
        filled_leg = next((leg for leg in legs if leg.status == OrderStatus.FILLED), None)
        if filled_leg is not None:
            events.append(
                {
                    "kind": "fill",
                    "symbol": symbol,
                    "order_id": filled_leg.id,
                    "reason": "take_profit" if filled_leg.type == OrderType.LIMIT else "stop_loss",
                    "fill_price": float(filled_leg.filled_avg_price) if filled_leg.filled_avg_price else None,
                    "qty": float(filled_leg.filled_qty) if filled_leg.filled_qty else None,
                }
            )
            _drop_tracked_bracket(symbol)
        elif legs and all(leg.status in _TERMINAL_NO_FILL for leg in legs):
            events.append(
                {
                    "kind": "terminal",
                    "symbol": symbol,
                    "order_id": order_id,
                    "leg_statuses": [leg.status.value for leg in legs],
                }
            )
            _drop_tracked_bracket(symbol)
        else:
            _set_tracked_bracket(symbol, order_id)

    return events


def check_crypto_stops() -> list[dict]:
    """Polls every tracked crypto position's synthetic stop-loss/take-profit against its current
    bid (the price an immediate market SELL would realize). A transient price-fetch failure just
    skips that symbol this round -- it stays tracked and gets checked again on the next poll."""
    events = []
    for symbol, (sl_price, tp_price) in _crypto_stops_snapshot():
        try:
            bid = get_current_bid_price(symbol)
        except APIError as exc:
            log(f"💥  failed to fetch bid for tracked crypto stop {symbol}: {exc}")
            continue

        if bid <= sl_price or bid >= tp_price:
            reason = "stop_loss" if bid <= sl_price else "take_profit"
            if not _drop_crypto_stop_if_current(symbol, sl_price, tp_price):
                continue
            result = sell(symbol, reason=reason)
            events.append({"symbol": symbol, "reason": reason, "bid_price": bid, "sell_result": result})

    return events


def check_option_stops() -> list[dict]:
    """Runs unconditionally regardless of options_trading.enabled -- that flag only gates opening
    NEW option positions (see select_option_contract/call_floor_broker_option in dealer/graph.py); a
    position that's already open must stay protected by its synthetic SL/TP/DTE-force-close even
    after the flag is flipped off as an emergency rollback. Naturally a no-op whenever
    _option_positions is empty, matching check_crypto_stops(), which has no enabled-flag gate at
    all."""
    cfg = load_config()
    events = []
    with _state_lock:
        tracked = list(_option_positions.items())

    today = datetime.now(pytz.timezone("US/Eastern")).date()
    for contract_symbol, ctx in tracked:
        try:
            mid = get_current_option_mid_price(contract_symbol)
        except (APIError, KeyError, TypeError) as exc:
            log(f"💥  failed to fetch quote for tracked option {contract_symbol}: {exc}")
            continue

        try:
            expiration = datetime.strptime(ctx["expiration"], "%Y-%m-%d").date()
        except (ValueError, TypeError) as exc:
            log(f"💥  malformed expiration {ctx['expiration']!r} for tracked option {contract_symbol}: {exc}")
            continue
        dte = (expiration - today).days
        entry_premium = ctx["entry_premium"]
        sl_price = entry_premium * cfg.options_trading.options_slP
        tp_price = entry_premium * cfg.options_trading.options_tpP

        if dte <= cfg.options_trading.dte_force_close:
            reason = "dte_force_close"
        elif mid <= sl_price:
            reason = "stop_loss"
        elif mid >= tp_price:
            reason = "take_profit"
        else:
            continue

        result = sell_option(contract_symbol, reason=reason)
        events.append({"symbol": ctx["symbol"], "contract_symbol": contract_symbol, "reason": reason, "premium": mid, "sell_result": result})

    return events


def check_eod_flatten() -> list[dict]:
    """Feature-gated (strategy config eod_flatten.enabled, off by default). When enabled and
    Alpaca's live clock reports the market is within eod_flatten.minutes_before_close minutes of
    closing, decides which open stock positions to sell -- crypto is 24/7 so "end of day" doesn't
    apply to it. Uses the live clock (not a fixed schedule) so early/half-trading-close days are
    handled correctly with no special-casing. No "already flattened today" bookkeeping is needed:
    once a symbol is sold, trading_client.get_all_positions() simply stops returning it, so later
    polls within the same closing window are cheap no-ops.

    Non-conditional mode (eod_flatten.conditional: false, the default) sells every open stock
    position, same as always. Conditional mode sells everything only if the aggregate unrealized
    P&L across all open stock positions is >= 0 ("UP"); if it's negative ("DOWN"), positions are
    held overnight instead -- except any individual position held >= eod_flatten.max_days_held_loss
    days, which is force-flattened regardless of the aggregate sign so a single loser can't ride
    forever."""
    cfg = load_config()  # fresh (within its own refresh window), same live-reload pattern buy() uses
    if not cfg.eod_flatten.enabled:
        return []

    clock = trading_client.get_clock()
    if not clock.is_open:
        return []

    minutes_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60
    if minutes_to_close > cfg.eod_flatten.minutes_before_close:
        return []

    positions = [p for p in trading_client.get_all_positions() if p.asset_class == AssetClass.US_EQUITY]

    if cfg.eod_flatten.get("conditional", False):
        aggregate_pl = sum(float(p.unrealized_pl) for p in positions if p.unrealized_pl is not None)
        if aggregate_pl >= 0:
            to_sell = positions
        else:
            max_days = cfg.eod_flatten.get("max_days_held_loss", 5)
            to_sell = []
            for position in positions:
                opened_at = db.fetch_position_opened_at(position.symbol)
                days_held = (clock.timestamp - opened_at).days if opened_at else 0
                if days_held >= max_days:
                    to_sell.append(position)
    else:
        to_sell = positions

    events = []
    for position in to_sell:
        result = sell(position.symbol, reason="eod_flatten")
        if result["status"] != "skipped":
            events.append({"symbol": position.symbol, "reason": "eod_flatten", "sell_result": result})
    return events


def flatten_all_crypto(reason: str = "power_down_flatten") -> list[dict]:
    """Force-sells every open crypto position, no market-clock gating (crypto is 24/7). Used by
    power_scheduler right before it scales floor-broker to 0 -- crypto's stop-loss/take-profit is
    only enforced by this process's own check_crypto_stops() poll loop, so an open crypto position
    left behind while the pod is scaled down would be completely unprotected overnight."""
    positions = [p for p in trading_client.get_all_positions() if _is_crypto_position(p)]

    events = []
    for position in positions:
        result = sell(position.symbol, reason=reason)
        if result["status"] != "skipped":
            events.append({"symbol": position.symbol, "reason": reason, "sell_result": result})
    return events


def flatten_all_options(reason: str = "power_down_flatten") -> list[dict]:
    """Force-sells every open LONG option position, no market-clock gating. Used by power_scheduler
    right before it scales floor-broker to 0 -- check_option_stops()'s SL/TP/DTE-force-close
    protection is only enforced by this process's own poll loop, so an open option position left
    behind while the pod is scaled down would be completely unprotected until the next session. This
    system never itself opens a short option position, so a short reaching this list would only be
    some other actor's position -- skipped rather than sold, since selling a short would open MORE
    short instead of closing it (sell_option() itself refuses this too, as defense in depth)."""
    positions = [
        p
        for p in trading_client.get_all_positions()
        if p.asset_class == AssetClass.US_OPTION and getattr(p, "side", PositionSide.LONG) == PositionSide.LONG
    ]

    events = []
    for position in positions:
        result = sell_option(position.symbol, reason=reason)
        if result["status"] != "skipped":
            events.append({"symbol": position.symbol, "reason": reason, "sell_result": result})
    return events


def is_state_reconciled() -> bool:
    with _state_lock:
        return _state_reconciled


def _set_state_reconciled(value: bool) -> None:
    global _state_reconciled
    with _state_lock:
        _state_reconciled = value


def _is_crypto_position(position) -> bool:
    return position.asset_class == AssetClass.CRYPTO


def _crypto_reference_price(position) -> float | None:
    for attr in ("avg_entry_price", "current_price"):
        value = getattr(position, attr, None)
        if value is not None:
            return float(value)
    return None


def _rebuild_crypto_stops_from_positions(positions, cfg) -> int:
    restored = 0
    for position in positions:
        if not _is_crypto_position(position):
            continue
        symbol = canonical_crypto_symbol(position.symbol)
        reference_price = _crypto_reference_price(position)
        if reference_price is None or reference_price <= 0:
            log(f"⚠️  cannot reconstruct crypto stop/target for {symbol}: no usable reference price")
            continue
        with _state_lock:
            has_pending_buy = any(
                ctx.get("symbol") == symbol and ctx.get("action") == "BUY" for ctx in _pending_fills.values()
            )
            if symbol in _crypto_stops or has_pending_buy:
                continue
            _crypto_stops[symbol] = (reference_price * cfg.strategy.crypto_slP, reference_price * cfg.strategy.crypto_tpP)
        restored += 1
    return restored


_OCC_SYMBOL_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def _parse_occ_contract_symbol(contract_symbol: str) -> dict | None:
    """Parses an OCC-standard option contract symbol (e.g. "AAPL250117C00200000" -> root AAPL,
    expiration 2025-01-17, call, strike $200.00) with no DB dependency, mirroring
    _crypto_reference_price's DB-free reconstruction for crypto. Returns None if the symbol doesn't
    match the OCC format."""
    m = _OCC_SYMBOL_RE.match(contract_symbol)
    if m is None:
        return None
    root, yymmdd, right_code, strike_digits = m.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "symbol": root,
        "right": "call" if right_code == "C" else "put",
        "strike": int(strike_digits) / 1000,
        "expiration": expiration.isoformat(),
    }


def _rebuild_option_positions_from_positions(positions, open_trades: list[dict]) -> int:
    """Rebuilds _option_positions from Alpaca's own open-positions state, cross-referenced against
    options_trades for the fields Alpaca's Position object doesn't carry (right, delta, reasoning
    isn't needed here). Postgres returns NUMERIC as decimal.Decimal and DATE as datetime.date --
    both must be explicitly cast, or check_option_stops()'s float/Decimal arithmetic and
    datetime.strptime() on a date object will raise TypeError the first time this reconstructed
    entry is polled. When no options_trades row matches (e.g. the row predates this table, or the DB
    write failed after the Alpaca order filled), falls back to a DB-free reconstruction from the
    OCC-standard contract symbol plus Alpaca's own avg_entry_price -- mirroring
    _crypto_reference_price's DB-free pattern for crypto, so a missing DB row never leaves a live
    position with zero stop-loss/take-profit protection. delta is None in the fallback path;
    check_option_stops() never reads ctx["delta"], so this is a safe degradation. Only skips entirely
    when the OCC parse fails or avg_entry_price is missing/<= 0 -- both cases with no reliable way to
    protect the position at all.

    A short option position (this system never opens one itself, so any short reaching this function
    is some other actor's position) is skipped entirely, for the same reason flatten_all_options()
    skips one -- tracking it as if it were long would eventually cause sell_option() to sell it again
    rather than close it."""
    trades_by_symbol = {trade["contract_symbol"]: trade for trade in open_trades}
    restored = 0
    for position in positions:
        contract_symbol = position.symbol
        with _state_lock:
            already_tracked = contract_symbol in _option_positions
        if already_tracked:
            continue

        if getattr(position, "side", PositionSide.LONG) != PositionSide.LONG:
            log(
                f"⚠️  skipping option position {contract_symbol}: not a long position -- this system "
                "only opens long option positions, and reconstructing a short as tracked state would "
                "cause sell_option() to double it instead of closing it"
            )
            continue

        trade = trades_by_symbol.get(contract_symbol)
        if trade is not None:
            expiration = trade["expiration"]
            with _state_lock:
                _option_positions[contract_symbol] = {
                    "symbol": trade["symbol"],
                    "right": trade["right"],
                    "strike": float(trade["strike"]),
                    "expiration": expiration.isoformat() if hasattr(expiration, "isoformat") else expiration,
                    "delta": float(trade["delta"]) if trade["delta"] is not None else None,
                    "entry_premium": float(trade["entry_premium"]),
                    "qty": abs(int(float(position.qty))),
                }
            restored += 1
            continue

        parsed = _parse_occ_contract_symbol(contract_symbol)
        avg_entry_price = getattr(position, "avg_entry_price", None)
        if parsed is None or avg_entry_price is None or float(avg_entry_price) <= 0:
            log(
                f"⚠️  cannot reconstruct option position {contract_symbol}: no matching open row in "
                "options_trades and no usable OCC-symbol/avg_entry_price fallback"
            )
            continue

        log(
            f"⚠️  reconstructing option position {contract_symbol} from OCC symbol + avg_entry_price "
            "(degraded: no matching options_trades row, delta unknown)"
        )
        with _state_lock:
            _option_positions[contract_symbol] = {
                "symbol": parsed["symbol"],
                "right": parsed["right"],
                "strike": parsed["strike"],
                "expiration": parsed["expiration"],
                "delta": None,
                "entry_premium": float(avg_entry_price),
                "qty": abs(int(float(position.qty))),
            }
        restored += 1
    return restored


def reconcile_tracked_state_once() -> bool:
    """Rebuilds _pending_fills and _tracked_brackets from Alpaca's own open-orders state -- both
    dicts are in-memory only, so a Floor Broker restart otherwise loses track of every order/
    bracket that was still open at the moment it went down, silently dropping their eventual fill
    notifications. Returns True and marks state reconciled on success; False (never raises) on
    any APIError, leaving existing tracked state and is_state_reconciled() untouched so a later
    retry can still succeed.

    GetOrdersRequest(status="open") queries at the order-*family* level (per Alpaca's own
    semantics, distinct from an individual leg's OrderStatus) -- a bracket whose entry already
    filled but whose TP/SL legs are still live is still "open" here, so nested=True correctly
    surfaces it as a parent with its legs attached, the same shape check_bracket_fills expects.
    Only orders still open on Alpaca are restored -- by definition nothing has filled yet here, so
    no notification could have been missed by this reconciliation itself. An order that fills in
    the gap between the pod dying and this running is a separate, narrower gap this cannot close
    -- Alpaca no longer reports it as "open" once filled -- the trade itself is still correct at
    Alpaca, only its Slack fill notice is missed."""
    try:
        open_orders = trading_client.get_orders(GetOrdersRequest(status="open", nested=True))
    except APIError as exc:
        log(f"💥  failed to fetch open orders from Alpaca while reconciling tracked state: {exc}")
        return False

    restored_pending = 0
    restored_brackets = 0
    for order in open_orders:
        # Option orders live on this same account now but are reconstructed separately below
        # (into _pending_option_fills, keyed by OCC symbol) -- keep them out of the stock/crypto
        # _pending_fills / _tracked_brackets rebuild.
        if order.asset_class == AssetClass.US_OPTION:
            continue
        symbol = canonical_crypto_symbol(order.symbol) if "/" in order.symbol or is_usd_crypto_symbol(order.symbol) else order.symbol
        legs = order.legs or []
        if legs:
            if any(leg.status not in _TERMINAL_NO_FILL for leg in legs):
                _set_tracked_bracket(symbol, order.id)
                restored_brackets += 1
        elif order.filled_avg_price is None and order.status not in _TERMINAL_NO_FILL:
            _set_pending_fill(order.id, {
                "symbol": symbol,
                "action": "BUY" if order.side == OrderSide.BUY else "SELL",
                "reason": "reconstructed_after_restart",
                "sl_price": None,
                "tp_price": None,
            })
            restored_pending += 1

    log(f"🔄  reconstructed {restored_pending} pending order(s) and {restored_brackets} bracket(s) from Alpaca")

    # One get_all_positions() call for the whole floor -- stocks, crypto and options all live on
    # the same account now, so a single fetch feeds the crypto-stop rebuild, the position_opens
    # backfill and the option-positions rebuild below.
    #
    # A failed read here is FATAL (return False, don't mark reconciled): a partial Alpaca read
    # that still set _state_reconciled=True would leave existing option positions unrebuilt (so
    # check_option_stops() protects nothing), skip the crypto-stop rebuild and position_opens
    # backfill, re-enable BUYs via _buy_preflight_skip(), and stop poll_reconciliation() retrying
    # (it loops only while unreconciled). The retry-with-backoff in reconstruct_tracked_state()
    # plus the 60s background loop exist precisely to ride out a transient outage here.
    try:
        all_positions = trading_client.get_all_positions()
    except APIError as exc:
        log(f"💥  failed to fetch open positions from Alpaca while reconciling tracked state: {exc}")
        return False

    # Single options_trades read for the whole reconcile: it feeds both the _option_positions
    # rebuild (recovering BUY-side fields Alpaca's Position doesn't carry) and the
    # open_contract_symbols set that seeds db_row_opened on reconstructed pending BUY orders. A DB
    # outage here is fatal for the same reason as the position read above -- option-stop protection
    # depends on _option_positions being fully rebuilt before state is marked reconciled.
    try:
        open_option_trades = db.fetch_open_options_trades()
    except Exception as exc:
        log(f"💥  failed to fetch open options_trades while reconciling tracked state: {exc}")
        return False

    # Backfill position_opens for every currently-open position -- ON CONFLICT DO NOTHING means
    # this is a no-op for a symbol already tracked, and only seeds an opened_at for a position
    # this process has never observed a BUY fill for (e.g. one that predates this feature, or was
    # opened in a gap while the pod was down). The crypto-stop rebuild is best-effort within this
    # step: a bad config or one malformed position must not sink the whole reconcile.
    try:
        crypto_stops_restored = _rebuild_crypto_stops_from_positions(all_positions, load_config())
    except Exception as exc:
        crypto_stops_restored = 0
        log(f"💥  failed to reconstruct crypto stops while reconciling tracked state: {exc}")
    for position in all_positions:
        # Option contracts are tracked in _option_positions (OCC symbols), not position_opens --
        # keep them out of the equity/crypto backfill so db.record_position_opened() never sees
        # an OCC symbol.
        if position.asset_class == AssetClass.US_OPTION:
            continue
        symbol = canonical_crypto_symbol(position.symbol) if _is_crypto_position(position) else position.symbol
        db.record_position_opened(symbol)
    if crypto_stops_restored:
        log(f"🔄  reconstructed {crypto_stops_restored} crypto synthetic stop/target(s) from open positions")

    # Fatal on failure (return False): a half-rebuilt _option_positions marked reconciled leaves
    # pre-restart option positions with no synthetic SL/TP/DTE protection from check_option_stops().
    try:
        option_positions = [p for p in all_positions if p.asset_class == AssetClass.US_OPTION]
        options_restored = _rebuild_option_positions_from_positions(option_positions, open_option_trades)
    except Exception as exc:
        log(f"💥  failed to reconstruct option positions while reconciling tracked state: {exc}")
        return False
    if options_restored:
        log(f"🔄  reconstructed {options_restored} option position(s) from Alpaca + options_trades")

    # No side filter here (unlike the pre-fix version, which only fetched BUY) -- a Floor Broker
    # restart between sell_option() submitting a SELL and that SELL's fill would otherwise leave
    # the SELL permanently unobserved: nothing else ever re-registers it, so options_trades would
    # stay open forever even after Alpaca fills or cancels it (external review finding 1,
    # 2026-08-26). Filter to US_OPTION -- this account also carries stock/crypto orders, which the
    # nested get_orders() call above already reconstructs. Fatal on failure (return False): a
    # still-open option BUY order missing from _pending_option_fills can allow a duplicate
    # submission after restart, and a missed SELL stays unobserved forever.
    try:
        open_option_orders = [
            o
            for o in trading_client.get_orders(GetOrdersRequest(status="open"))
            if o.asset_class == AssetClass.US_OPTION
        ]
    except APIError as exc:
        log(f"💥  failed to fetch open option orders from Alpaca while reconciling tracked state: {exc}")
        return False

    # Seed db_row_opened from whether the contract already has an open options_trades row (from the
    # single fetch above) -- otherwise the eventual fill re-INSERTs a duplicate open row and
    # re-fires the first-fill Slack notification for a contract that's already tracked (fix-loop
    # round 1, finding 2). open_option_trades is guaranteed bound here: its fetch is fatal above.
    open_contract_symbols = {trade["contract_symbol"] for trade in open_option_trades}

    restored_pending_options = 0
    for order in open_option_orders:
        with _state_lock:
            already_pending = order.id in _pending_option_fills
        if already_pending:
            continue
        if order.side == OrderSide.SELL:
            # A pending SELL has no BUY-side fields to reconstruct (right/strike/delta/etc.) and
            # doesn't need them -- check_pending_option_fills()'s SELL branch only reads
            # contract_symbol/symbol/reason. _option_positions itself is reconstructed separately
            # above (_rebuild_option_positions_from_positions), from Alpaca's still-open position,
            # so it's already tracked here independent of this order.
            with _state_lock:
                underlying_symbol = _option_positions.get(order.symbol, {}).get("symbol")
            _set_pending_option_fill(order.id, {
                "contract_symbol": order.symbol,
                "symbol": underlying_symbol,
                "action": "SELL",
                "reason": "reconstructed_after_restart",
            })
            restored_pending_options += 1
            continue
        parsed = _parse_occ_contract_symbol(order.symbol)
        if parsed is None:
            log(f"⚠️  cannot reconstruct pending option order {order.id} ({order.symbol}): symbol doesn't match OCC format")
            continue
        _set_pending_option_fill(order.id, {
            "contract_symbol": order.symbol,
            "symbol": parsed["symbol"],
            "action": "BUY",
            "right": parsed["right"],
            "strike": parsed["strike"],
            "expiration": parsed["expiration"],
            "delta": None,
            "qty": int(float(order.qty)),
            "reasoning": "reconstructed_after_restart",
            "cycle_id": None,
            "db_row_opened": order.symbol in open_contract_symbols,
        })
        restored_pending_options += 1
    if restored_pending_options:
        log(f"🔄  reconstructed {restored_pending_options} pending option order(s) from Alpaca")

    _set_state_reconciled(True)
    return True


def reconstruct_tracked_state(
    max_attempts: int = _RECONCILE_MAX_STARTUP_ATTEMPTS, backoff_base_s: float = _RECONCILE_BACKOFF_BASE_S
) -> None:
    """Runs once at Floor Broker startup, before poll threads start. Retries
    reconcile_tracked_state_once() with exponential backoff -- a transient Alpaca outage at
    exactly boot time shouldn't permanently strand the service with empty tracking dicts from a
    single failed attempt. If every attempt fails, is_state_reconciled() stays False -- buy()
    refuses new BUYs (see below) until main.poll_reconciliation() succeeds in the background."""
    for attempt in range(1, max_attempts + 1):
        if reconcile_tracked_state_once():
            return
        if attempt < max_attempts:
            backoff = backoff_base_s * (2 ** (attempt - 1))
            log(f"🔄  reconciliation attempt {attempt}/{max_attempts} failed, retrying in {backoff:.0f}s")
            time.sleep(backoff)

    log(
        f"🚨  exhausted {max_attempts} startup reconciliation attempts -- "
        "BUY execution will be rejected until state reconciles; retrying in background"
    )


def _fetch_open_position(symbol: str):
    try:
        return trading_client.get_open_position(alpaca_order_symbol(symbol))
    except APIError:
        return None


def get_open_position(symbol: str) -> float:
    symbol = canonical_crypto_symbol(symbol) if "/" in symbol or is_usd_crypto_symbol(symbol) else symbol
    is_stock = symbol.find("/") == -1
    position = _fetch_open_position(symbol)
    if position is None:
        return 0
    qty = int(float(position.qty)) if is_stock else float(position.qty)
    log(f"📈  open position: {qty} of {symbol.replace('/', '')}")
    return qty


def cancel_related_orders(order_ids: list[str]) -> None:
    for oid in order_ids:
        try:
            trading_client.cancel_order_by_id(oid)
            log(f"✅  cancelled conflicting order {oid}")
        except APIError as exc:
            if exc.code != ORDER_NOT_FOUND_CODE:
                raise


def get_qty(ask: float, budget: float) -> int:
    if ask <= 0:
        return 0
    return int(budget // ask)


def _round_to_tick(price: float) -> float:
    # SEC Rule 612 / Alpaca: sub-$1 stocks are quoted in $0.0001 increments, not $0.01 --
    # rounding to 2dp for these can land TP/SL on the same cent as base_price and get rejected.
    return round(price, 4) if price < 1.0 else round(price, 2)


def _validate_bracket_order(ask: float, qty: int, budget: float, stop_loss_px: float, take_profit_px: float) -> None:
    """Invariant checks (ROADMAP P0.9) run just before a stock bracket order is built, using the
    same single reference price (`ask`) that sized `qty` and priced `stop_loss_px`/
    `take_profit_px` -- see P0.8: these three values must all derive from one quote, not
    independent ones fetched moments apart."""
    if ask <= 0:
        raise InvalidOrderParameters(f"non-positive reference price: {ask}")
    if budget <= 0:
        raise InvalidOrderParameters(f"non-positive budget: {budget}")
    if qty < 1:
        raise InsufficientQuantity(f"budget {budget} affords < 1 share at reference price {ask}")
    if not (0 < stop_loss_px < ask < take_profit_px):
        raise InvalidOrderParameters(
            f"invalid SL/TP relationship: stop_loss={stop_loss_px} ask={ask} take_profit={take_profit_px}"
        )
    estimated_notional = qty * ask
    if estimated_notional > budget:
        raise InvalidOrderParameters(f"estimated notional {estimated_notional} exceeds authorized budget {budget}")


def bracket_buy_with_SLTP(
    symbol: str, budget: float, slP: float, tpP: float, base_price: float | None = None
) -> MarketOrderRequest:
    # A bracket BUY fills near the ask, not the bid/ask mid -- pricing TP/SL off mid understates
    # the real entry price and can fall below Alpaca's `base_price + 0.01` floor on wide-spread
    # symbols, causing the whole order to be rejected. `base_price`, when given, is Alpaca's own
    # rejection-supplied reference price for a retry (see `buy()`) -- it takes priority over a
    # fresh client-side quote, which can diverge from Alpaca's reference on thin, low-priced
    # symbols (e.g. our free-tier IEX-only feed missing the true NBBO).
    ask = base_price if base_price is not None else get_current_ask_price(symbol)
    if ask <= 0:
        raise NoAskQuote(f"no executable ask quote for {symbol}: {ask}")

    # Alpaca also enforces an absolute $0.01 minimum distance between TP/SL and base_price,
    # regardless of stock price -- on sub-~$0.50 stocks, slP/tpP's percentage move (e.g. 2%/5%)
    # doesn't reach a full cent, so the percentage-based price must be clamped to that floor.
    # Clamp to $0.02, not the bare $0.01 minimum, for a small safety margin against price
    # movement in the moments between quoting `ask` and Alpaca validating the order.
    take_profit_px = max(_round_to_tick(ask * tpP), _round_to_tick(ask + 0.02))
    stop_loss_px = min(_round_to_tick(ask * slP), _round_to_tick(ask - 0.02))

    # Quantity is sized off the same `ask` used for TP/SL above (P0.8) -- previously get_qty()
    # fetched its own independent quote, so qty and bracket prices could be based on different
    # market snapshots.
    qty = get_qty(ask, budget)

    log(f"📈  ask-price {ask:.2f} => TP {take_profit_px}  |  SL {stop_loss_px}  |  qty {qty}")

    _validate_bracket_order(ask, qty, budget, stop_loss_px, take_profit_px)

    return MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        stop_loss=StopLossRequest(stop_price=stop_loss_px),
        take_profit=TakeProfitRequest(limit_price=take_profit_px),
    )


def _buy_preflight_skip(symbol: str, cfg) -> dict | None:
    if not is_state_reconciled():
        log(f"🛑  BUY {symbol} rejected -- tracked state not yet reconciled with Alpaca")
        return {
            "status": "skipped",
            "reason": "state_not_reconciled",
            "detail": "tracked state not yet reconciled with Alpaca after restart",
        }

    if kill_switch.buy_kill_switch_active():
        log(f"🛑  BUY kill switch active -- skipping BUY {symbol}")
        return {"status": "skipped", "reason": "buy_kill_switch_active", "detail": "BUY kill switch is active"}

    account = trading_client.get_account()
    daily_pnl = float(account.equity) - float(account.last_equity)
    if daily_pnl >= cfg.strategy.daily_profit_target_usd:
        log(f"🛑  daily profit target reached (${daily_pnl:.2f}) -- skipping BUY {symbol}")
        return {
            "status": "skipped",
            "reason": "daily_profit_target_reached",
            "detail": f"daily P&L ${daily_pnl:.2f} >= target ${cfg.strategy.daily_profit_target_usd}",
        }
    if daily_pnl <= -cfg.strategy.daily_loss_limit_usd:
        log(f"🛑  daily loss limit reached (${daily_pnl:.2f}) -- skipping BUY {symbol}")
        return {
            "status": "skipped",
            "reason": "daily_loss_limit_reached",
            "detail": f"daily P&L ${daily_pnl:.2f} <= -limit ${cfg.strategy.daily_loss_limit_usd}",
        }

    return None


def _remaining_budget_or_skip(symbol: str, budget: float) -> tuple[float | None, dict | None]:
    position = _fetch_open_position(symbol)
    if position is None:
        return budget, None

    if position.market_value is None:
        log(f"⚠️  {symbol} existing position market_value unavailable - aborting BUY")
        return None, {
            "status": "skipped",
            "reason": "market_value_unavailable",
            "detail": "existing position market_value unavailable",
        }

    existing_value = float(position.market_value)
    if existing_value >= budget:
        log(f"⚠️  {symbol} position (${existing_value:.2f}) already at/above budget (${budget:.2f}) - skipping BUY")
        return None, {
            "status": "skipped",
            "reason": "budget_exhausted",
            "detail": f"existing position value ${existing_value:.2f} >= budget ${budget:.2f}",
        }

    remaining_budget = budget - existing_value
    log(f"📈  {symbol} position (${existing_value:.2f}) below budget (${budget:.2f}) - topping up ${remaining_budget:.2f}")
    return remaining_budget, None


def _risk_based_budget_cap(slP: float, cfg) -> float | None:
    """Returns the largest budget consistent with strategy.risk_per_trade_usd given a stop-loss
    fraction slP (a BUY sized at that budget loses budget * (1 - slP) if stopped out), or None if
    risk-based sizing isn't active -- "flat_budget" (the default) leaves the caller's requested
    budget untouched. Only ever caps the budget down, never up, so this can shrink an
    Analyst-authorized budget toward the risk target but never inflate exposure beyond what was
    authorized."""
    if cfg.strategy.position_sizing != "risk_based":
        return None
    risk_usd = cfg.strategy.risk_per_trade_usd
    if not risk_usd or not (0 < slP < 1):
        return None
    return risk_usd / (1 - slP)


def _max_concurrent_positions_skip(symbol: str, cfg) -> dict | None:
    """Refuses a new BUY once the number of currently-open positions is at/above
    strategy.max_concurrent_positions -- topping up a symbol that's already open doesn't add a
    new position, so it's exempt (checked via _fetch_open_position, same as
    _remaining_budget_or_skip's own top-up check). Stocks, crypto and options all share this cap
    now -- one account, one open-position count."""
    if _fetch_open_position(symbol) is not None:
        return None

    open_count = len(trading_client.get_all_positions())
    limit = cfg.strategy.max_concurrent_positions
    if open_count >= limit:
        log(f"🛑  max concurrent positions reached ({open_count}/{limit}) -- skipping new BUY {symbol}")
        return {
            "status": "skipped",
            "reason": "max_concurrent_positions_reached",
            "detail": f"{open_count} open position(s) at/above cap of {limit}",
        }
    return None


def _duplicate_option_buy_skip(contract_symbol: str) -> dict | None:
    """Refuses a second BUY for a contract this process already holds or already has a BUY order in
    flight for. Options have no top-up concept (unlike the stock path): every Dealer option_pick is
    a brand-new entry, _fallback_pick is deterministic so a slow-filling BUY gets re-picked
    identically on the next Dealer cycle, and _option_positions is keyed by OCC symbol and
    overwritten wholesale on each observed fill (see check_pending_option_fills) -- so a doubled BUY
    both over-positions the account and corrupts the tracked qty / entry premium that synthetic
    SL/TP/DTE protection sizes off. The reconcile scan (gated by is_state_reconciled(), which
    _buy_preflight_skip enforces) seeds _pending_option_fills from every open Alpaca option order,
    so the in-memory view here is authoritative; _fetch_open_position is a final cross-check against
    Alpaca. Mirrors buy()'s open-orders guard for the stock/crypto path."""
    with _state_lock:
        buy_in_flight = any(
            ctx.get("contract_symbol") == contract_symbol and ctx.get("action") == "BUY"
            for ctx in _pending_option_fills.values()
        )
        tracked = contract_symbol in _option_positions
    if buy_in_flight:
        log(f"🛑  option BUY {contract_symbol} skipped -- a BUY for this contract is already in flight")
        return {
            "status": "skipped",
            "reason": "option_buy_in_flight",
            "detail": f"a BUY order for {contract_symbol} is already in flight; not submitting a duplicate",
        }
    if tracked or _fetch_open_position(contract_symbol) is not None:
        log(f"🛑  option BUY {contract_symbol} skipped -- this contract is already an open position")
        return {
            "status": "skipped",
            "reason": "already_holding_contract",
            "detail": f"{contract_symbol} is already an open position; not submitting a duplicate BUY",
        }
    return None


def option_exposure_contract_symbols() -> list[str]:
    """Every option contract this process currently holds or has a BUY order in flight for. The
    Dealer reads this (GET /option-exposure) before a new option entry so it can skip a contract
    that would be a duplicate; buy_option() enforces the same rule authoritatively via
    _duplicate_option_buy_skip -- this is only the earlier, quieter check."""
    with _state_lock:
        held = set(_option_positions)
        pending_buys = {
            ctx["contract_symbol"]
            for ctx in _pending_option_fills.values()
            if ctx.get("action") == "BUY" and ctx.get("contract_symbol")
        }
    return sorted(held | pending_buys)


def _spread_reference_price_or_skip(symbol: str, cfg) -> tuple[float | None, dict | None]:
    max_spread_pct = cfg.strategy.get("max_bid_ask_spread_pct")
    if max_spread_pct is None:
        return None, None
    ask = get_current_ask_price(symbol)
    bid = get_current_bid_price(symbol)
    if ask <= 0:
        return None, {
            "status": "skipped",
            "reason": "no_ask_quote",
            "detail": f"no executable ask quote for {symbol}: {ask}",
        }
    if bid <= 0 or bid >= ask:
        return None, {
            "status": "skipped",
            "reason": "invalid_bid_ask_quote",
            "detail": f"invalid bid/ask quote for {symbol}: bid={bid} ask={ask}",
        }
    spread_pct = (ask - bid) / ask
    if spread_pct > max_spread_pct:
        log(f"⚠️  spread {spread_pct:.2%} exceeds max {max_spread_pct:.2%} for {symbol} -- skipping BUY")
        return None, {
            "status": "skipped",
            "reason": "wide_bid_ask_spread",
            "detail": f"bid/ask spread {spread_pct:.2%} exceeds max {max_spread_pct:.2%}",
        }
    return ask, None


def _stock_buy_request_or_skip(
    symbol: str,
    budget: float,
    slP: float,
    tpP: float,
    cfg,
    base_price: float | None = None,
):
    try:
        if base_price is None:
            base_price, skip = _spread_reference_price_or_skip(symbol, cfg)
            if skip is not None:
                return None, skip
        return bracket_buy_with_SLTP(symbol, budget, slP, tpP, base_price=base_price), None
    except NoAskQuote as exc:
        log(f"⚠️  {exc} -- skipping BUY")
        return None, {"status": "skipped", "reason": "no_ask_quote", "detail": str(exc)}
    except InsufficientQuantity as exc:
        log(f"⚠️  {exc} -- skipping BUY")
        return None, {"status": "skipped", "reason": "insufficient_qty", "detail": str(exc)}
    except InvalidOrderParameters as exc:
        log(f"⚠️  {exc} -- skipping BUY")
        return None, {"status": "skipped", "reason": "invalid_order_parameters", "detail": str(exc)}


def _crypto_buy_request_or_skip(symbol: str, budget: float):
    if not is_usd_crypto_symbol(symbol):
        log(f"⚠️  crypto BUY {symbol} is not USD-quoted -- skipping")
        return None, {
            "status": "skipped",
            "reason": "non_usd_crypto_pair",
            "detail": f"crypto BUY {symbol} is not quoted in USD",
        }

    notional = round(budget, 2)
    if notional < MIN_CRYPTO_NOTIONAL:
        log(f"⚠️  budget {notional} below Alpaca's ${MIN_CRYPTO_NOTIONAL:.0f} crypto minimum -- skipping")
        return None, {
            "status": "skipped",
            "reason": "budget_below_minimum",
            "detail": f"budget {notional} below ${MIN_CRYPTO_NOTIONAL:.0f} crypto minimum",
        }

    return (
        MarketOrderRequest(
            symbol=symbol,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        ),
        None,
    )


def _submit_buy_order(req, symbol: str, exchange: str, budget: float, slP: float, tpP: float):
    try:
        return trading_client.submit_order(req), None, req
    except APIError as exc:
        if exchange != "stocks":
            raise
        try:
            err = json.loads(str(exc))
        except json.JSONDecodeError:
            raise

        base_price = err.get("base_price")
        if err.get("code") != 42210000 or base_price is None:
            raise

        log(f"🔄  retrying BUY {symbol} priced off Alpaca's own base_price {base_price} ...")
        retry_req, skip = _stock_buy_request_or_skip(
            symbol,
            budget,
            slP,
            tpP,
            load_config(),
            base_price=float(base_price),
        )
        if skip is not None:
            return None, skip, retry_req
        return trading_client.submit_order(retry_req), None, retry_req


def buy(symbol: str, exchange: str, budget: float, slP: float, tpP: float) -> dict:
    if exchange != "stocks":
        symbol = canonical_crypto_symbol(symbol)

    cfg = load_config()  # fresh (within its own refresh window) so a live strategy change never needs a restart

    effective_slP = slP if exchange == "stocks" else cfg.strategy.crypto_slP
    risk_cap = _risk_based_budget_cap(effective_slP, cfg)
    if risk_cap is not None and risk_cap < budget:
        log(f"📉  risk-based sizing caps {symbol} budget ${budget:.2f} -> ${risk_cap:.2f} (risk_per_trade_usd=${cfg.strategy.risk_per_trade_usd}, slP={effective_slP})")
        budget = risk_cap

    skip = _buy_preflight_skip(symbol, cfg)
    if skip is not None:
        return skip

    oo = trading_client.get_orders(GetOrdersRequest(status="open"))
    matching_orders = [order for order in oo if alpaca_order_symbol(order.symbol) == alpaca_order_symbol(symbol)]

    if matching_orders:
        log(f"⚠️  open orders exist for {symbol} - aborting BUY")
        return {"status": "skipped", "reason": "open_orders_exist", "detail": "open orders exist for symbol"}

    skip = _max_concurrent_positions_skip(symbol, cfg)
    if skip is not None:
        return skip

    budget, skip = _remaining_budget_or_skip(symbol, budget)
    if skip is not None:
        return skip

    if exchange == "stocks":
        req, skip = _stock_buy_request_or_skip(symbol, budget, slP, tpP, cfg)
    else:
        req, skip = _crypto_buy_request_or_skip(symbol, budget)
    if skip is not None:
        return skip

    order, skip, req = _submit_buy_order(req, symbol, exchange, budget, slP, tpP)
    if skip is not None:
        return skip

    log(f"✅  buy order submitted: {order.id}")

    if exchange == "stocks":
        _set_tracked_bracket(symbol, order.id)
        sl_price = req.stop_loss.stop_price
        tp_price = req.take_profit.limit_price
        crypto_slP = crypto_tpP = None
    else:
        sl_price = None
        tp_price = None
        # No fill price is known yet for a notional market order -- store the strategy.crypto_slP/
        # crypto_tpP multipliers here so check_pending_fills() can compute the actual sl_price/
        # tp_price once the fill (and its real fill_price) is observed.
        crypto_slP = cfg.strategy.crypto_slP
        crypto_tpP = cfg.strategy.crypto_tpP

    _set_pending_fill(order.id, {
        "symbol": symbol,
        "action": "BUY",
        "reason": "opening_position",
        "sl_price": sl_price,
        "tp_price": tp_price,
        "crypto_slP": crypto_slP,
        "crypto_tpP": crypto_tpP,
    })

    return {
        "status": "submitted",
        "reason": "opening_position",
        "detail": f"buy order submitted: {order.id}",
        "order_id": str(order.id),
        "sl_price": sl_price,
        "tp_price": tp_price,
    }


def sell(symbol: str, reason: str = "dealer_signal") -> dict:
    symbol = canonical_crypto_symbol(symbol) if "/" in symbol or is_usd_crypto_symbol(symbol) else symbol
    qty = get_open_position(symbol)

    if qty <= 0:
        log(f"⚠️  no open position of {symbol} to sell")
        return {"status": "skipped", "detail": "no open position"}

    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)

    # An explicit SELL closes the position, which also cancels any still-open TP/SL bracket legs
    # on Alpaca's side -- stop watching for a bracket fill on this symbol immediately rather than
    # waiting for check_bracket_fills() to notice the legs went terminal with no fill. Same idea
    # for a tracked synthetic crypto stop/target -- a manual/Dealer-driven SELL already closes the
    # position, so check_crypto_stops() must not also try to sell it.
    _stop_tracking_symbol(symbol)

    try:
        order = trading_client.submit_order(req)
        log(f"✅  sell order submitted: {order.id}")
        _set_pending_fill(order.id, {"symbol": symbol, "action": "SELL", "reason": reason, "sl_price": None, "tp_price": None})
        return {
            "status": "submitted",
            "reason": reason,
            "detail": f"sell order submitted: {order.id}",
            "order_id": str(order.id),
        }
    except APIError as exc:
        try:
            err = json.loads(str(exc))
        except json.JSONDecodeError:
            raise

        if err.get("code") != 40310000:
            raise

        log(f"⚠️  {err.get('message')} for {err.get('symbol')}")
        cancel_related_orders(err.get("related_orders", []))

        # The qty available to sell can change once the blocking orders are
        # cleared, so the retry must recompute qty and rebuild req rather than
        # resubmitting the stale req captured before cleanup.
        qty = get_open_position(symbol)
        if qty <= 0:
            log(f"⚠️  no qty of {symbol} remaining after cleanup")
            return {"status": "skipped", "detail": "no qty remaining after cleanup"}

        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)

        try:
            log("🔄  retrying after clean-up ...")
            order = trading_client.submit_order(req)
            log(f"✅  sell order submitted: {order.id}")
            _set_pending_fill(order.id, {"symbol": symbol, "action": "SELL", "reason": reason, "sl_price": None, "tp_price": None})
            return {
                "status": "submitted",
                "reason": reason,
                "detail": f"sell order submitted: {order.id}",
                "order_id": str(order.id),
            }
        except APIError as retry_exc:
            log(f"💥  sell retry failed for {symbol}: {retry_exc}")
            raise


def buy_option(
    contract_symbol: str,
    qty: int,
    entry_premium: float,
    right: str,
    strike: float,
    expiration: str,
    delta: float | None,
    reasoning: str | None,
    symbol: str,
    cycle_id: str | None,
) -> dict:
    cfg = load_config()  # fresh (within its own refresh window) so a live strategy change never needs a restart

    skip = _buy_preflight_skip(contract_symbol, cfg)
    if skip is not None:
        return skip

    skip = _duplicate_option_buy_skip(contract_symbol)
    if skip is not None:
        return skip

    skip = _max_concurrent_positions_skip(contract_symbol, cfg)
    if skip is not None:
        return skip

    try:
        live_ask = get_current_option_ask_price(contract_symbol)
    except APIError as exc:
        log(f"💥  failed to re-quote {contract_symbol} before BUY: {exc}")
        return {"status": "error", "detail": f"failed to fetch live quote for {contract_symbol}: {exc}"}

    if live_ask <= 0:
        log(f"⚠️  no executable ask quote for {contract_symbol} -- skipping BUY")
        return {
            "status": "skipped",
            "reason": "no_ask_quote",
            "detail": f"no executable ask quote for {contract_symbol}: {live_ask}",
        }

    notional = qty * live_ask * 100
    if notional > cfg.options_trading.max_notional_usd:
        log(f"🛑  option BUY notional (${notional:.2f}) exceeds cap (${cfg.options_trading.max_notional_usd}) -- skipping {contract_symbol}")
        return {
            "status": "skipped",
            "reason": "notional_cap_exceeded",
            "detail": f"live-quoted notional ${notional:.2f} (qty={qty} @ live ask ${live_ask:.2f}) exceeds cap ${cfg.options_trading.max_notional_usd}",
        }

    req = MarketOrderRequest(symbol=contract_symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    try:
        order = trading_client.submit_order(req)
    except APIError as exc:
        log(f"💥  option buy order failed for {contract_symbol}: {exc}")
        return {"status": "error", "detail": str(exc)}

    log(f"✅  option buy order submitted: {order.id}")
    _set_pending_option_fill(order.id, {
        "contract_symbol": contract_symbol,
        "symbol": symbol,
        "action": "BUY",
        "right": right,
        "strike": strike,
        "expiration": expiration,
        "delta": delta,
        "qty": qty,
        "reasoning": reasoning,
        "cycle_id": cycle_id,
    })

    return {
        "status": "submitted",
        "reason": "opening_position",
        "detail": f"option buy order submitted: {order.id}",
        "order_id": str(order.id),
    }


def sell_option(contract_symbol: str, reason: str = "dealer_signal") -> dict:
    """Submits a market SELL for the full open qty of contract_symbol. Unlike the pre-fix version,
    this does NOT clear _option_positions or close the DB row synchronously on submit -- a submitted
    SELL can still be canceled/rejected or simply never fill, and clearing state early would drop
    synthetic SL/TP/DTE protection and report the trade closed for a position that's still open.
    Mirrors buy_option()/check_pending_option_fills(): the submit here only registers a pending SELL
    entry in _pending_option_fills; check_pending_option_fills() is the sole place that pops
    _option_positions and calls db.record_options_trade_closed, once the SELL's own fill (or
    terminal no-fill) is confirmed (external review finding 1, 2026-08-26).

    Also cancels (best-effort) any still-open BUY order this process is tracking for contract_symbol
    BEFORE fetching the position to size this SELL -- e.g. a partially-filled BUY that's still
    non-terminal when check_option_stops() sells the already-filled portion. Canceling first (rather
    than after sizing/submitting the SELL) matters: if the cancel loses the race because the BUY's
    remaining qty fills right before Alpaca can cancel it, that fill has already landed by the time
    get_open_position() is called below, so this SELL is sized to include it -- no leftover qty is
    ever left both unsold and untracked (external review finding 2, 2026-08-26; tightened further,
    external review finding 1, 2026-08-26 round 2, after the first version of this fix cancelled only
    after sizing/submitting the SELL, which left the exact same gap open).

    Only a BUY order this call actually confirmed canceled has its tracking dropped below. A BUY
    whose cancel attempt itself failed (e.g. a transient API error, as opposed to the order having
    already filled) is deliberately left tracked -- it may still be genuinely live at Alpaca, and
    dropping it here would mean nothing ever watches it again; check_pending_option_fills() resolves
    it normally on a later poll instead, whichever way it actually goes (external review finding 2,
    2026-08-26 round 3)."""
    with _state_lock:
        if any(ctx.get("action") == "SELL" and ctx.get("contract_symbol") == contract_symbol for ctx in _pending_option_fills.values()):
            return {"status": "skipped", "detail": "sell already pending"}
        stale_buy_order_ids = [
            oid for oid, ctx in _pending_option_fills.items()
            if ctx.get("contract_symbol") == contract_symbol and ctx.get("action") == "BUY"
        ]

    canceled_buy_order_ids = []
    for buy_order_id in stale_buy_order_ids:
        try:
            trading_client.cancel_order_by_id(buy_order_id)
            log(f"🛑  canceled stale pending option BUY {buy_order_id} for {contract_symbol} ahead of its SELL")
            canceled_buy_order_ids.append(buy_order_id)
        except APIError as exc:
            log(
                f"💥  failed to cancel stale pending option BUY {buy_order_id} for {contract_symbol}, "
                f"leaving it tracked: {exc}"
            )

    try:
        position = trading_client.get_open_position(contract_symbol)
    except APIError as exc:
        if _is_position_not_found_error(exc):
            log(f"⚠️  no open option position of {contract_symbol} to sell -- dropping tracking")
            with _state_lock:
                _option_positions.pop(contract_symbol, None)
                _drop_pending_option_fills_for_contract(contract_symbol)
            return {"status": "skipped", "detail": "no open position"}
        log(f"💥  failed to fetch option position {contract_symbol}: {exc}")
        return {"status": "error", "detail": str(exc)}

    qty = int(float(position.qty))
    if qty <= 0:
        if qty < 0:
            log(f"⚠️  {contract_symbol} is a short position (qty={qty}) -- refusing to sell, dropping tracking")
        with _state_lock:
            _option_positions.pop(contract_symbol, None)
            _drop_pending_option_fills_for_contract(contract_symbol)
        return {"status": "skipped", "detail": "no open position"}

    req = MarketOrderRequest(symbol=contract_symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    try:
        order = trading_client.submit_order(req)
    except APIError as exc:
        log(f"💥  option sell order failed for {contract_symbol}: {exc}")
        return {"status": "error", "detail": str(exc)}

    log(f"✅  option sell order submitted: {order.id}")
    with _state_lock:
        underlying_symbol = _option_positions.get(contract_symbol, {}).get("symbol")
        # Only drop the pending-fill entries for BUYs actually confirmed canceled above -- a BUY
        # whose cancel attempt failed stays tracked (see docstring) rather than being blanket-dropped
        # via _drop_pending_option_fills_for_contract() (external review finding 2, 2026-08-26 round
        # 3). _option_positions itself is deliberately left tracked here too: check_pending_option_
        # fills() clears it once this SELL's fill (or terminal no-fill) is confirmed (fix-loop round
        # 1, finding 3; external review finding 1, 2026-08-26).
        for buy_order_id in canceled_buy_order_ids:
            _pending_option_fills.pop(buy_order_id, None)
        _pending_option_fills[order.id] = {
            "contract_symbol": contract_symbol,
            "symbol": underlying_symbol,
            "action": "SELL",
            "reason": reason,
        }

    return {
        "status": "submitted",
        "reason": reason,
        "detail": f"option sell order submitted: {order.id}",
        "order_id": str(order.id),
    }
