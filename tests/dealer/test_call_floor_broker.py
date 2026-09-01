from datetime import datetime

import pytz
from omegaconf import OmegaConf

from src.dealer import graph


def _cfg():
    return OmegaConf.create(
        {
            "trading": {"slP": 0.98, "tpP": 1.05},
            "floor_broker": {"base_url": "http://floor-broker.test:8000"},
            "macro_blackout": {"enabled": False, "dates": []},
            "strategy": {
                "min_confidence": 0.6,
                "win_rate_throttle": {"enabled": False},
                "symbol_stop_cooldown": {"enabled": False},
                "dealer_memory": {"enabled": False},
            },
            "analyst": {"track_record_days": 5},
        }
    )


def _state(action: str, budget: float, size_hint: float = 1.0, confidence: float = 1.0) -> dict:
    return {
        "symbol": "MGN",
        "exchange": "stocks",
        "budget": budget,
        "cycle_id": "cycle-1",
        "raw_bars": {},
        "ohlcv_features_text": "",
        "signal": {"action": action, "reasoning": "test", "size_hint": size_hint, "confidence": confidence},
        "execution_result": None,
    }


def _silence_slack(monkeypatch):
    # These tests target call_floor_broker's own HTTP dispatch, not Slack notifications --
    # silence both so the environment's real SLACK_WEBHOOK_URL2 (if configured) can't cause an
    # actual network call, and so its use of the shared `requests` module doesn't clobber the
    # fake `requests.post` these tests install for the Floor Broker call.
    monkeypatch.setattr(graph.slack, "notify_dealer_signal", lambda *a, **k: None)
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)


def test_call_floor_broker_records_ohlcv_audit_fields(monkeypatch):
    recorded = {}
    monkeypatch.setattr(graph.slack, "notify_dealer_signal", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: recorded.update(args=a, kwargs=k))

    result = graph.call_floor_broker({**_state("HOLD", budget=5000.0), "ohlcv_features_text": "features"}, _cfg())

    assert result["execution_result"] == {"status": "skipped", "detail": "HOLD"}
    assert recorded["kwargs"] == {"ohlcv_enrichment_active": True, "cycle_id": "cycle-1"}


def test_buy_on_held_only_position_is_skipped_without_calling_floor_broker(monkeypatch):
    """Regression: merge_held_positions() gives held-only entries budget=0.0 -- a BUY signal on
    one of these must be refused locally, never forwarded to Floor Broker sized off a market
    value that was never authorized new-BUY capital."""
    _silence_slack(monkeypatch)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called for a zero-budget BUY")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(_state("BUY", budget=0.0), _cfg())

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "no_authorized_budget"


def test_buy_with_authorized_budget_is_forwarded_to_floor_broker(monkeypatch):
    """A normal Analyst-authorized BUY (nonzero budget) must still reach Floor Broker."""
    _silence_slack(monkeypatch)
    posted = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    def _fake_post(url, json, timeout):
        posted["url"] = url
        posted["json"] = json
        return FakeResponse()

    monkeypatch.setattr(graph.requests, "post", _fake_post)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), _cfg())

    assert posted["json"]["budget"] == 5000.0
    assert result["execution_result"]["status"] == "executed"


def test_buy_scales_forwarded_budget_by_size_hint(monkeypatch):
    """The Dealer LLM's size_hint (fraction of budget to deploy) was previously captured in the
    schema but never actually applied -- ROADMAP P1.7. A 0.5 hint on a $5000 budget must reach
    Floor Broker as $2500, not the full $5000."""
    _silence_slack(monkeypatch)
    posted = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    def _fake_post(url, json, timeout):
        posted["json"] = json
        return FakeResponse()

    monkeypatch.setattr(graph.requests, "post", _fake_post)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0, size_hint=0.5), _cfg())

    assert posted["json"]["budget"] == 2500.0
    assert result["execution_result"]["status"] == "executed"


def test_buy_with_zero_size_hint_is_skipped_without_calling_floor_broker(monkeypatch):
    """A size_hint of exactly 0.0 scales the authorized budget to $0 -- ExecuteRequest requires
    budget > 0 (src/floor_broker/app.py), so this must be refused locally rather than forwarded
    to a request that would fail Pydantic validation."""
    _silence_slack(monkeypatch)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called for a size_hint-zeroed BUY")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0, size_hint=0.0), _cfg())

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "size_hint_zero"


def test_sell_forwards_budget_unscaled_by_size_hint(monkeypatch):
    """size_hint is documented as a BUY-only sizing hint (src/dealer/schema.py) -- Floor Broker's
    sell() ignores budget entirely, but confirm call_floor_broker doesn't scale it for SELL
    either, in case that ever changes."""
    _silence_slack(monkeypatch)
    posted = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "sell order submitted: order-456"}

    def _fake_post(url, json, timeout):
        posted["json"] = json
        return FakeResponse()

    monkeypatch.setattr(graph.requests, "post", _fake_post)

    graph.call_floor_broker(_state("SELL", budget=5000.0, size_hint=0.1), _cfg())

    assert posted["json"]["budget"] == 5000.0


def test_execution_result_fields_are_forwarded_to_the_slack_notification(monkeypatch):
    """execution.py's reason/fill_price/sl_price/tp_price must reach the Slack notice, not just
    status/detail -- confirms call_floor_broker doesn't drop them on the way through."""
    monkeypatch.setattr(graph.slack, "notify_dealer_signal", lambda *a, **k: None)
    posted = {}
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: posted.update(kwargs=k))

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": "executed",
                "detail": "buy order submitted: order-123",
                "reason": "opening_position",
                "order_id": "order-123",
                "fill_price": 10.05,
                "sl_price": 9.8,
                "tp_price": 10.5,
            }

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    graph.call_floor_broker(_state("BUY", budget=5000.0), _cfg())

    assert posted["kwargs"]["reason"] == "opening_position"
    assert posted["kwargs"]["fill_price"] == 10.05
    assert posted["kwargs"]["sl_price"] == 9.8
    assert posted["kwargs"]["tp_price"] == 10.5


def test_buy_is_skipped_during_macro_blackout(monkeypatch):
    """A BUY on a day matching macro_blackout.dates must be refused locally, never forwarded to
    Floor Broker. `_is_quad_witching_day` is forced False so this test isolates the hand-
    maintained date-list path from the auto-computed quad-witching path (tested separately)."""
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    today = datetime.now(pytz.timezone("US/Eastern")).date().isoformat()
    cfg = _cfg()
    cfg.macro_blackout.enabled = True
    cfg.macro_blackout.dates = [{"date": today, "label": "CPI release"}]

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called during a macro blackout")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "macro_blackout"


def test_buy_is_skipped_on_quad_witching_day(monkeypatch):
    """Quad witching is auto-detected (src/dealer/graph.py:_is_quad_witching_day), not a
    config.yaml entry -- forced True here to test that path independent of the current date."""
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: True)
    cfg = _cfg()
    cfg.macro_blackout.enabled = True

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called on a quad witching day")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "macro_blackout"


def test_buy_is_not_skipped_when_macro_blackout_date_is_not_today(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    posted = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    def _fake_post(url, json, timeout):
        posted["json"] = json
        return FakeResponse()

    monkeypatch.setattr(graph.requests, "post", _fake_post)

    cfg = _cfg()
    cfg.macro_blackout.enabled = True
    cfg.macro_blackout.dates = [{"date": "1999-01-01", "label": "not today"}]

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert result["execution_result"]["status"] == "executed"


def test_buy_is_skipped_when_confidence_below_minimum(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called for a low-confidence BUY")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0, confidence=0.4), _cfg())

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "low_confidence"


def test_buy_proceeds_when_confidence_at_or_above_minimum(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    result = graph.call_floor_broker(_state("BUY", budget=5000.0, confidence=0.6), _cfg())

    assert result["execution_result"]["status"] == "executed"


def test_buy_missing_confidence_defaults_to_full_confidence(monkeypatch):
    """A signal dict without a confidence key (e.g. an older cached state) must not be gated
    out -- matches Signal's own default of 1.0."""
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    state = _state("BUY", budget=5000.0)
    del state["signal"]["confidence"]

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    result = graph.call_floor_broker(state, _cfg())

    assert result["execution_result"]["status"] == "executed"


def test_classify_exit_event_stock_bracket_fills():
    assert graph._classify_exit_event({"event_type": "fill", "detail": "take_profit leg filled: o-1"}) == "win"
    assert graph._classify_exit_event({"event_type": "fill", "detail": "stop_loss leg filled: o-1"}) == "loss"


def test_classify_exit_event_crypto_synthetic_exits():
    assert graph._classify_exit_event({"event_type": "synthetic_take_profit", "detail": "sold"}) == "win"
    assert graph._classify_exit_event({"event_type": "synthetic_stop_loss", "detail": "sold"}) == "loss"


def test_classify_exit_event_ignores_non_exit_events():
    """BUY opens, manual SELLs, eod_flatten, errors, and skips have no reliable win/loss outcome
    without the original entry price -- must be excluded from the win-rate sample, not miscounted."""
    assert graph._classify_exit_event({"event_type": "buy_executed", "detail": "buy order submitted: o-1"}) is None
    assert graph._classify_exit_event({"event_type": "sell_executed", "detail": "sell order submitted: o-1"}) is None
    assert graph._classify_exit_event({"event_type": "error", "detail": "boom"}) is None
    assert graph._classify_exit_event({"event_type": "skip", "detail": "macro blackout"}) is None
    assert graph._classify_exit_event({"event_type": "fill", "detail": "unrelated fill text"}) is None


def _win_rate_cfg(min_win_rate=0.3, win_rate_min_sample=5):
    cfg = _cfg()
    cfg.strategy.win_rate_throttle.enabled = True
    cfg.strategy.win_rate_throttle_scope = "global"
    cfg.strategy.min_win_rate = min_win_rate
    cfg.strategy.win_rate_min_sample = win_rate_min_sample
    return cfg


def _fill_event(reason: str) -> dict:
    return {"event_type": "fill", "detail": f"{reason} leg filled: order-1"}


def _synthetic_event(reason: str) -> dict:
    return {"event_type": f"synthetic_{reason}", "detail": "sold"}


def test_buy_is_skipped_when_win_rate_below_minimum_with_sufficient_sample(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    events = [_fill_event("stop_loss")] * 4 + [_fill_event("take_profit")] * 1  # 20% win rate
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since_date: events)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called while the win-rate throttle is active")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(
        _state("BUY", budget=5000.0),
        _win_rate_cfg(min_win_rate=0.3, win_rate_min_sample=5),
    )

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "win_rate_throttle"


def test_buy_is_skipped_when_symbol_recently_stopped_out(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    cfg = _cfg()
    cfg.strategy.symbol_stop_cooldown.enabled = True
    cfg.strategy.symbol_stop_cooldown_days = 1
    cfg.strategy.max_symbol_stop_losses = 1
    monkeypatch.setattr(
        graph.db,
        "fetch_symbol_floor_broker_events_since",
        lambda symbol, since_date, limit=100: [_fill_event("stop_loss")],
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called while symbol cooldown is active")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "symbol_stop_cooldown"


def test_buy_proceeds_when_symbol_cooldown_has_no_recent_stop(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    cfg = _cfg()
    cfg.strategy.symbol_stop_cooldown.enabled = True
    monkeypatch.setattr(
        graph.db,
        "fetch_symbol_floor_broker_events_since",
        lambda symbol, since_date, limit=100: [_fill_event("take_profit")],
    )

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert result["execution_result"]["status"] == "executed"


def test_buy_proceeds_when_win_rate_at_or_above_minimum(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    events = [_fill_event("take_profit")] * 4 + [_synthetic_event("stop_loss")] * 1  # 80% win rate
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since_date: events)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    result = graph.call_floor_broker(
        _state("BUY", budget=5000.0),
        _win_rate_cfg(min_win_rate=0.3, win_rate_min_sample=5),
    )

    assert result["execution_result"]["status"] == "executed"


def test_buy_proceeds_when_exit_sample_size_is_below_the_minimum(monkeypatch):
    """A poor win rate on too few completed exits (e.g. 0/1) must not trip the throttle -- avoids
    overreacting to noise early in the trailing window."""
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    events = [_fill_event("stop_loss")]  # 0% win rate, but only 1 sample
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since_date: events)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    result = graph.call_floor_broker(
        _state("BUY", budget=5000.0),
        _win_rate_cfg(min_win_rate=0.3, win_rate_min_sample=5),
    )

    assert result["execution_result"]["status"] == "executed"


def test_buy_proceeds_when_win_rate_throttle_disabled(monkeypatch):
    """win_rate_throttle.enabled: false must be a config-only no-op, matching macro_blackout's own
    feature-gate precedent -- the db call is never even made."""
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not query floor_broker_events while the throttle is disabled")

    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", _fail_if_called)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "buy order submitted: order-123"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())

    cfg = _win_rate_cfg(min_win_rate=0.99, win_rate_min_sample=0)
    cfg.strategy.win_rate_throttle.enabled = False

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert result["execution_result"]["status"] == "executed"


def test_symbol_scoped_win_rate_throttle_queries_same_symbol_events(monkeypatch):
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    cfg = _win_rate_cfg(min_win_rate=0.3, win_rate_min_sample=5)
    cfg.strategy.win_rate_throttle_scope = "symbol"
    seen = {}

    def _symbol_events(symbol, since_date, limit=100):
        seen["symbol"] = symbol
        return [_fill_event("stop_loss")] * 5

    monkeypatch.setattr(graph.db, "fetch_symbol_floor_broker_events_since", _symbol_events)
    monkeypatch.setattr(
        graph.db,
        "fetch_floor_broker_events_since",
        lambda since_date: (_ for _ in ()).throw(AssertionError("global events must not be queried")),
    )
    monkeypatch.setattr(
        graph.requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Floor Broker must not be called")),
    )

    result = graph.call_floor_broker(_state("BUY", budget=5000.0), cfg)

    assert seen["symbol"] == "MGN"
    assert result["execution_result"]["reason"] == "win_rate_throttle"


def test_sell_forwards_even_during_macro_blackout(monkeypatch):
    """The macro blackout gate only wraps the BUY branch -- SELL must reach Floor Broker
    normally regardless, since risk management (exiting positions) shouldn't itself be paused."""
    _silence_slack(monkeypatch)
    monkeypatch.setattr(graph, "_is_quad_witching_day", lambda d: False)
    posted = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "executed", "detail": "sell order submitted: order-456"}

    def _fake_post(url, json, timeout):
        posted["json"] = json
        return FakeResponse()

    monkeypatch.setattr(graph.requests, "post", _fake_post)

    today = datetime.now(pytz.timezone("US/Eastern")).date().isoformat()
    cfg = _cfg()
    cfg.macro_blackout.enabled = True
    cfg.macro_blackout.dates = [{"date": today, "label": "CPI release"}]

    graph.call_floor_broker(_state("SELL", budget=5000.0), cfg)

    assert posted["json"]["budget"] == 5000.0


def test_is_quad_witching_day_matches_third_friday_of_quarter_end_months():
    from datetime import date

    assert graph._is_quad_witching_day(date(2026, 3, 20))  # third Friday of March 2026
    assert graph._is_quad_witching_day(date(2026, 6, 19))  # third Friday of June 2026
    assert graph._is_quad_witching_day(date(2026, 9, 18))  # third Friday of September 2026
    assert graph._is_quad_witching_day(date(2026, 12, 18))  # third Friday of December 2026


def test_is_quad_witching_day_false_for_other_fridays_and_months():
    from datetime import date

    assert not graph._is_quad_witching_day(date(2026, 3, 13))  # second Friday of March
    assert not graph._is_quad_witching_day(date(2026, 3, 27))  # fourth Friday of March
    assert not graph._is_quad_witching_day(date(2026, 4, 17))  # third Friday of April (not a quarter-end month)
    assert not graph._is_quad_witching_day(date(2026, 9, 17))  # a Thursday, not a Friday
