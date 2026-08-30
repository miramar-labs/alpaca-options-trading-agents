import pytest

from src.floor_broker import main as fb_main


class _StopLoop(Exception):
    """Raised from a mocked time.sleep() to break out of poll_bracket_fills()'s infinite loop
    after exactly one iteration, so it can be tested without actually running forever."""


def test_poll_bracket_fills_posts_slack_notification_for_each_event(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_bracket_fills",
        lambda: [{"kind": "fill", "symbol": "MGN", "order_id": "leg-1", "reason": "take_profit", "fill_price": 15.0, "qty": 10.0}],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_bracket_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "MGN"
    assert args[1] == "SELL"
    assert kwargs["reason"] == "take_profit"
    assert kwargs["fill_price"] == 15.0


def test_poll_bracket_fills_posts_no_fill_notice_for_a_terminal_event(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_bracket_fills",
        lambda: [{"kind": "terminal", "symbol": "MGN", "order_id": "parent-1", "leg_statuses": ["canceled", "canceled"]}],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_bracket_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "MGN"
    assert args[1] == "SELL"
    assert args[2] == "no_fill"


def test_poll_bracket_fills_posts_nothing_when_no_events(monkeypatch):
    monkeypatch.setattr(fb_main.execution, "check_bracket_fills", lambda: [])
    monkeypatch.setattr(fb_main.execution, "check_crypto_stops", lambda: [])
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_bracket_fills()

    assert posted == []


def test_poll_bracket_fills_survives_an_exception_and_reaches_the_next_sleep(monkeypatch):
    """A transient Alpaca error inside one poll iteration must not kill the background thread --
    it should be caught, logged, and the loop must still reach time.sleep() to try again."""

    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.execution, "check_bracket_fills", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_bracket_fills()


def test_poll_bracket_fills_posts_slack_notification_for_a_crypto_stop_event(monkeypatch):
    """check_crypto_stops() runs on the same poll cadence as check_bracket_fills() (no dedicated
    thread) -- a triggered synthetic stop-loss/take-profit must be reported the same way a bracket
    fill is."""
    monkeypatch.setattr(fb_main.execution, "check_bracket_fills", lambda: [])
    monkeypatch.setattr(
        fb_main.execution,
        "check_crypto_stops",
        lambda: [
            {
                "symbol": "BTC/USD",
                "reason": "stop_loss",
                "bid_price": 49000.0,
                "sell_result": {"status": "submitted", "detail": "market sell submitted"},
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_bracket_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "BTC/USD"
    assert args[1] == "SELL"
    assert args[2] == "submitted"
    assert kwargs["reason"] == "stop_loss"
    assert "49000.0" in args[3]


def test_poll_bracket_fills_survives_a_crypto_stop_check_exception(monkeypatch):
    """A transient error from check_crypto_stops() (e.g. a failed price fetch that wasn't already
    swallowed inside it) must not kill the shared poll_bracket_fills thread."""
    monkeypatch.setattr(fb_main.execution, "check_bracket_fills", lambda: [])

    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.execution, "check_crypto_stops", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_bracket_fills()


def test_poll_pending_fills_posts_slack_notification_for_each_event(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_fills",
        lambda: [
            {
                "kind": "fill",
                "symbol": "MGN",
                "action": "BUY",
                "reason": "opening_position",
                "order_id": "order-1",
                "fill_price": 10.05,
                "sl_price": 9.8,
                "tp_price": 10.5,
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "MGN"
    assert args[1] == "BUY"
    assert kwargs["reason"] == "opening_position"
    assert kwargs["fill_price"] == 10.05
    assert kwargs["sl_price"] == 9.8
    assert kwargs["tp_price"] == 10.5


def test_poll_pending_fills_records_position_opened_on_a_buy_fill(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_fills",
        lambda: [
            {
                "kind": "fill",
                "symbol": "MGN",
                "action": "BUY",
                "reason": "opening_position",
                "order_id": "order-1",
                "fill_price": 10.05,
                "sl_price": 9.8,
                "tp_price": 10.5,
            }
        ],
    )
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: None)
    opened, closed = [], []
    monkeypatch.setattr(fb_main.db, "record_position_opened", lambda symbol: opened.append(symbol))
    monkeypatch.setattr(fb_main.db, "record_position_closed", lambda symbol: closed.append(symbol))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_fills()

    assert opened == ["MGN"]
    assert closed == []


def test_poll_pending_fills_records_position_closed_on_a_sell_fill(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_fills",
        lambda: [
            {
                "kind": "fill",
                "symbol": "MGN",
                "action": "SELL",
                "reason": "dealer_signal",
                "order_id": "order-1",
                "fill_price": 10.05,
                "sl_price": None,
                "tp_price": None,
            }
        ],
    )
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: None)
    opened, closed = [], []
    monkeypatch.setattr(fb_main.db, "record_position_opened", lambda symbol: opened.append(symbol))
    monkeypatch.setattr(fb_main.db, "record_position_closed", lambda symbol: closed.append(symbol))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_fills()

    assert closed == ["MGN"]
    assert opened == []


def test_poll_pending_fills_posts_no_fill_notice_for_a_terminal_event(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_fills",
        lambda: [
            {
                "kind": "terminal",
                "symbol": "MGN",
                "action": "BUY",
                "reason": "opening_position",
                "order_id": "order-1",
                "order_status": "rejected",
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "MGN"
    assert args[1] == "BUY"
    assert args[2] == "no_fill"
    assert kwargs["reason"] == "opening_position"


def test_poll_pending_fills_posts_nothing_when_no_events(monkeypatch):
    monkeypatch.setattr(fb_main.execution, "check_pending_fills", lambda: [])
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_fills()

    assert posted == []


def test_poll_pending_fills_survives_an_exception_and_reaches_the_next_sleep(monkeypatch):
    """A transient Alpaca error inside one poll iteration must not kill the background thread --
    it should be caught, logged, and the loop must still reach time.sleep() to try again."""

    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.execution, "check_pending_fills", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_fills()


def test_poll_pending_option_fills_posts_slack_notification_for_a_fill(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_option_fills",
        lambda: [
            {
                "contract_symbol": "AAPL250117C00200000",
                "symbol": "AAPL",
                "kind": "fill",
                "order_id": "order-opt-1",
                "fill_price": 3.35,
                "qty": 2,
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_option_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "AAPL"
    assert args[1] == "BUY"
    assert args[2] == "executed"
    assert kwargs["reason"] == "opening_position"
    assert kwargs["fill_price"] == 3.35


def test_poll_pending_option_fills_posts_no_fill_notice_for_a_terminal_event(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_option_fills",
        lambda: [
            {
                "contract_symbol": "AAPL250117C00200000",
                "symbol": "AAPL",
                "kind": "terminal",
                "order_id": "order-opt-1",
                "order_status": "rejected",
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_option_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "AAPL"
    assert args[1] == "BUY"
    assert args[2] == "no_fill"


def test_poll_pending_option_fills_posts_slack_notification_for_a_sell_fill(monkeypatch):
    """Regression (external review finding 1, 2026-08-26): an option SELL fill event now flows
    through the same poller as a BUY fill, keyed off event["action"] == "SELL"."""
    monkeypatch.setattr(
        fb_main.execution,
        "check_pending_option_fills",
        lambda: [
            {
                "contract_symbol": "AAPL250117C00200000",
                "symbol": "AAPL",
                "action": "SELL",
                "reason": "take_profit",
                "kind": "fill",
                "order_id": "order-opt-sell-1",
                "fill_price": 4.50,
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_option_fills()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "AAPL"
    assert args[1] == "SELL"
    assert args[2] == "executed"
    assert kwargs["reason"] == "take_profit"
    assert kwargs["fill_price"] == 4.50


def test_poll_pending_option_fills_posts_nothing_when_no_events(monkeypatch):
    monkeypatch.setattr(fb_main.execution, "check_pending_option_fills", lambda: [])
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_option_fills()

    assert posted == []


def test_poll_pending_option_fills_survives_an_exception_and_reaches_the_next_sleep(monkeypatch):
    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.execution, "check_pending_option_fills", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_pending_option_fills()


def test_poll_eod_flatten_posts_slack_notification_for_each_event(monkeypatch):
    monkeypatch.setattr(
        fb_main.execution,
        "check_eod_flatten",
        lambda: [
            {
                "symbol": "MGN",
                "reason": "eod_flatten",
                "sell_result": {"status": "submitted", "detail": "sell order submitted: order-1"},
            }
        ],
    )
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_eod_flatten()

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "MGN"
    assert args[1] == "SELL"
    assert args[2] == "submitted"
    assert "order-1" in args[3]
    assert kwargs["reason"] == "eod_flatten"


def test_poll_eod_flatten_posts_nothing_when_no_events(monkeypatch):
    monkeypatch.setattr(fb_main.execution, "check_eod_flatten", lambda: [])
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_floor_broker_result", lambda *a, **k: posted.append((a, k)))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_eod_flatten()

    assert posted == []


def test_poll_eod_flatten_survives_an_exception_and_reaches_the_next_sleep(monkeypatch):
    """A transient Alpaca error inside one poll iteration must not kill the background thread --
    it should be caught, logged, and the loop must still reach time.sleep() to try again."""

    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.execution, "check_eod_flatten", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: (_ for _ in ()).throw(_StopLoop))

    with pytest.raises(_StopLoop):
        fb_main.poll_eod_flatten()


def test_poll_reconciliation_exits_once_state_is_reconciled(monkeypatch):
    """poll_reconciliation() only exists to retry reconciliation after startup's bounded attempts
    were all exhausted -- once execution.is_state_reconciled() flips True (whether from this loop
    or elsewhere), the loop must exit rather than keep polling forever."""
    states = iter([False, False, True])
    monkeypatch.setattr(fb_main.execution, "is_state_reconciled", lambda: next(states))
    attempts = []
    monkeypatch.setattr(fb_main.execution, "reconcile_tracked_state_once", lambda: attempts.append(1))
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: None)

    fb_main.poll_reconciliation()

    assert len(attempts) == 2


def test_poll_reconciliation_survives_an_exception_and_keeps_retrying(monkeypatch):
    states = iter([False, False, True])
    monkeypatch.setattr(fb_main.execution, "is_state_reconciled", lambda: next(states))

    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.execution, "reconcile_tracked_state_once", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", lambda s: None)

    fb_main.poll_reconciliation()


def _stop_after(n):
    """Lets poll_kill_switch() run for exactly `n` iterations (n calls to time.sleep()) before
    breaking out via _StopLoop, so a multi-iteration transition sequence can be tested without an
    infinite loop."""
    calls = {"count": 0}

    def _sleep(seconds):
        calls["count"] += 1
        if calls["count"] >= n:
            raise _StopLoop

    return _sleep


def test_poll_kill_switch_posts_nothing_on_first_observation(monkeypatch):
    """ROADMAP P0.5: the first poll only discovers whatever state the switch was seeded/left in
    -- that's not a transition, and must not fire a Slack notice on its own."""
    monkeypatch.setattr(fb_main.kill_switch, "buy_kill_switch_active", lambda: True)
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_buy_kill_switch", lambda active: posted.append(active))
    monkeypatch.setattr(fb_main.time, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        fb_main.poll_kill_switch()

    assert posted == []


def test_poll_kill_switch_notifies_only_on_transition(monkeypatch):
    states = iter([False, False, True, True, False])
    monkeypatch.setattr(fb_main.kill_switch, "buy_kill_switch_active", lambda: next(states))
    posted = []
    monkeypatch.setattr(fb_main.slack, "notify_buy_kill_switch", lambda active: posted.append(active))
    monkeypatch.setattr(fb_main.time, "sleep", _stop_after(5))

    with pytest.raises(_StopLoop):
        fb_main.poll_kill_switch()

    assert posted == [True, False], "must notify on False->True and True->False, and nothing else"


def test_poll_kill_switch_survives_an_exception_and_reaches_the_next_sleep(monkeypatch):
    def _raise():
        raise RuntimeError("apiserver unavailable")

    monkeypatch.setattr(fb_main.kill_switch, "buy_kill_switch_active", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        fb_main.poll_kill_switch()


def test_poll_symbol_bases_refreshes_on_each_iteration(monkeypatch):
    calls = []
    monkeypatch.setattr(fb_main.symbols, "refresh_known_usd_crypto_bases_from_alpaca", lambda: calls.append(1) or 12)
    monkeypatch.setattr(fb_main.time, "sleep", _stop_after(2))

    with pytest.raises(_StopLoop):
        fb_main.poll_symbol_bases()

    assert len(calls) == 2


def test_poll_symbol_bases_survives_an_exception_and_reaches_the_next_sleep(monkeypatch):
    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(fb_main.symbols, "refresh_known_usd_crypto_bases_from_alpaca", _raise)
    monkeypatch.setattr(fb_main.time, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        fb_main.poll_symbol_bases()
