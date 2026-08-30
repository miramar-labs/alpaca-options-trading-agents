import pytest
from alpaca.common.exceptions import APIError
from fastapi.testclient import TestClient

from src.floor_broker import app as app_module


def _client():
    return TestClient(app_module.app)


def _payload(action="BUY", **overrides):
    payload = {"symbol": "MGN", "exchange": "stocks", "action": action, "budget": 5000.0, "slP": 0.98, "tpP": 1.05}
    payload.update(overrides)
    return payload


def test_execute_response_includes_sl_tp_and_reason_for_a_buy(monkeypatch):
    monkeypatch.setattr(
        app_module.execution,
        "buy",
        lambda symbol, exchange, budget, slP, tpP: {
            "status": "submitted",
            "reason": "opening_position",
            "detail": "buy order submitted: order-123",
            "order_id": "order-123",
            "sl_price": 9.8,
            "tp_price": 10.5,
        },
    )

    response = _client().post("/execute", json=_payload("BUY"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "submitted"
    assert body["reason"] == "opening_position"
    assert body["order_id"] == "order-123"
    assert body["fill_price"] is None
    assert body["sl_price"] == 9.8
    assert body["tp_price"] == 10.5


def test_execute_response_includes_dealer_signal_reason_for_a_sell(monkeypatch):
    monkeypatch.setattr(
        app_module.execution,
        "sell",
        lambda symbol: {
            "status": "submitted",
            "reason": "dealer_signal",
            "detail": "sell order submitted: order-456",
            "order_id": "order-456",
        },
    )

    response = _client().post("/execute", json=_payload("SELL"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "submitted"
    assert body["reason"] == "dealer_signal"
    assert body["fill_price"] is None
    assert body["sl_price"] is None
    assert body["tp_price"] is None


def test_execute_response_omits_optional_fields_when_skipped(monkeypatch):
    monkeypatch.setattr(app_module.execution, "sell", lambda symbol: {"status": "skipped", "detail": "no open position"})

    response = _client().post("/execute", json=_payload("SELL"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "skipped"
    assert body["reason"] is None
    assert body["fill_price"] is None
    assert body["sl_price"] is None
    assert body["tp_price"] is None


def test_execute_returns_200_not_500_for_a_buy_while_state_unreconciled(monkeypatch):
    """Regression test: execution.buy()'s state_not_reconciled short-circuit must return a
    status value ExecuteResponse's Literal actually permits (see app.py) -- exercises the real,
    unmocked execution.buy() through the live /execute endpoint so a response-model mismatch
    here (which previously surfaced as a 500, not a validation failure at the execution layer)
    is caught the same way a real caller would hit it."""
    monkeypatch.setattr(app_module.execution, "_state_reconciled", False)

    response = _client().post("/execute", json=_payload("BUY"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "skipped"
    assert body["reason"] == "state_not_reconciled"


def test_execute_sell_with_zero_budget_reaches_execution_sell(monkeypatch):
    """Regression: a held-only position (merge_held_positions()) carries budget=0.0, and the
    Dealer's SELL signal on it forwards that budget as-is (only BUY is guarded/scaled locally in
    src/dealer/graph.py). Previously ExecuteRequest required budget > 0 for every action, so this
    422'd before execution.sell() ever ran -- a held-only SELL could never actually execute."""
    received = {}

    def _fake_sell(symbol):
        received["symbol"] = symbol
        return {"status": "submitted", "reason": "dealer_signal", "detail": "sell order submitted: order-789", "order_id": "order-789"}

    monkeypatch.setattr(app_module.execution, "sell", _fake_sell)

    response = _client().post("/execute", json=_payload("SELL", budget=0.0))

    assert response.status_code == 200
    assert received["symbol"] == "MGN"
    assert response.json()["status"] == "submitted"


def test_execute_rejects_buy_with_zero_budget(monkeypatch):
    """budget > 0 is still enforced for BUY -- only SELL was over-restricted."""
    response = _client().post("/execute", json=_payload("BUY", budget=0.0))

    assert response.status_code == 422


def test_execute_rejects_negative_budget_for_sell(monkeypatch):
    """budget=0.0 is a legitimate held-only-position value, but negative is still nonsensical
    regardless of action."""
    response = _client().post("/execute", json=_payload("SELL", budget=-100.0))

    assert response.status_code == 422


def test_execute_normalizes_symbol_and_exchange_case(monkeypatch):
    """ROADMAP P0.3: symbol/exchange are normalized (stripped, cased) before reaching
    execution.buy() -- a lowercase symbol or upper-case "STOCKS" exchange from a caller must
    still compare correctly against execution.py's `if exchange == "stocks"` check."""
    received = {}

    def _fake_buy(symbol, exchange, budget, slP, tpP):
        received["symbol"] = symbol
        received["exchange"] = exchange
        return {"status": "executed", "reason": "opening_position", "detail": "ok", "order_id": "order-123"}

    monkeypatch.setattr(app_module.execution, "buy", _fake_buy)

    response = _client().post("/execute", json=_payload("BUY", symbol=" mgn ", exchange="STOCKS"))

    assert response.status_code == 200
    assert received["symbol"] == "MGN"
    assert received["exchange"] == "stocks"


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": ""},
        {"symbol": "TOO/MANY/SLASHES"},
        {"symbol": "TOO.MANY.DOTS"},
        {"symbol": "HAS SPACE"},
        {"exchange": ""},
        {"exchange": "has space"},
        {"budget": 0},
        {"budget": -100.0},
        {"budget": app_module.MAX_BUDGET + 0.01},
        {"slP": 0},
        {"slP": 1},
        {"slP": 1.01},
        {"slP": -0.1},
        {"tpP": 1},
        {"tpP": 2},
        {"tpP": 0.99},
        {"unexpected_field": "nope"},
    ],
    ids=lambda v: str(v),
)
def test_execute_rejects_invalid_request_fields(overrides):
    """ROADMAP P0.3: invalid requests must fail FastAPI's own request validation (422) before
    execution.buy()/sell() is ever called -- no execution.* monkeypatch needed here since a
    valid call should never happen."""
    response = _client().post("/execute", json=_payload("BUY", **overrides))

    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"budget": app_module.MAX_BUDGET},
        {"slP": 0.01},
        {"slP": 0.99},
        {"tpP": 1.01},
        {"tpP": 1.99},
        {"symbol": "DSX.WS"},
        {"symbol": "BRK.B"},
    ],
    ids=lambda v: str(v),
)
def test_execute_accepts_boundary_valid_values(monkeypatch, overrides):
    monkeypatch.setattr(
        app_module.execution,
        "buy",
        lambda symbol, exchange, budget, slP, tpP: {
            "status": "executed",
            "reason": "opening_position",
            "detail": "ok",
            "order_id": "order-123",
        },
    )

    response = _client().post("/execute", json=_payload("BUY", **overrides))

    assert response.status_code == 200


def test_flatten_crypto_returns_ok_with_events(monkeypatch):
    monkeypatch.setattr(
        app_module.execution,
        "flatten_all_crypto",
        lambda: [{"symbol": "BTC/USD", "reason": "power_down_flatten", "sell_result": {"status": "submitted"}}],
    )

    response = _client().post("/flatten-crypto")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["events"] == [{"symbol": "BTC/USD", "reason": "power_down_flatten", "sell_result": {"status": "submitted"}}]


def test_flatten_crypto_returns_ok_with_no_events_when_nothing_open(monkeypatch):
    monkeypatch.setattr(app_module.execution, "flatten_all_crypto", lambda: [])

    response = _client().post("/flatten-crypto")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["events"] == []


def test_flatten_crypto_returns_error_status_on_api_error(monkeypatch):
    def _raise():
        raise APIError("broker rejected")

    monkeypatch.setattr(app_module.execution, "flatten_all_crypto", _raise)

    response = _client().post("/flatten-crypto")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["events"] == []


def test_flatten_crypto_notifies_slack_and_reraises_on_unexpected_error(monkeypatch):
    def _raise():
        raise ValueError("boom")

    monkeypatch.setattr(app_module.execution, "flatten_all_crypto", _raise)
    calls = {}
    monkeypatch.setattr(app_module.slack, "notify_error", lambda *a, **k: calls.setdefault("error", (a, k)))

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post("/flatten-crypto")

    assert response.status_code == 500
    assert "error" in calls


def test_flatten_options_returns_ok_with_events(monkeypatch):
    monkeypatch.setattr(
        app_module.execution,
        "flatten_all_options",
        lambda: [{"symbol": "AAPL250117C00200000", "reason": "power_down_flatten", "sell_result": {"status": "submitted"}}],
    )

    response = _client().post("/flatten-options")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["events"] == [{"symbol": "AAPL250117C00200000", "reason": "power_down_flatten", "sell_result": {"status": "submitted"}}]


def test_flatten_options_returns_ok_with_no_events_when_nothing_open(monkeypatch):
    monkeypatch.setattr(app_module.execution, "flatten_all_options", lambda: [])

    response = _client().post("/flatten-options")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["events"] == []


def test_flatten_options_returns_error_status_on_api_error(monkeypatch):
    def _raise():
        raise APIError("broker rejected")

    monkeypatch.setattr(app_module.execution, "flatten_all_options", _raise)

    response = _client().post("/flatten-options")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["events"] == []


def test_flatten_options_notifies_slack_and_reraises_on_unexpected_error(monkeypatch):
    def _raise():
        raise ValueError("boom")

    monkeypatch.setattr(app_module.execution, "flatten_all_options", _raise)
    calls = {}
    monkeypatch.setattr(app_module.slack, "notify_error", lambda *a, **k: calls.setdefault("error", (a, k)))

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post("/flatten-options")

    assert response.status_code == 500
    assert "error" in calls


def test_execute_option_returns_result_from_buy_option(monkeypatch):
    captured = {}

    def _fake_buy_option(contract_symbol, qty, premium, right, strike, expiration, delta, reasoning, symbol, cycle_id):
        captured["args"] = (contract_symbol, qty, premium, right, strike, expiration, delta, reasoning, symbol, cycle_id)
        return {"status": "submitted", "reason": "opening_position", "detail": "option buy order submitted: order-1", "order_id": "order-1"}

    monkeypatch.setattr(app_module.execution, "buy_option", _fake_buy_option)

    response = _client().post(
        "/execute-option",
        json={
            "contract_symbol": "AAPL250117C00200000",
            "side": "BUY",
            "qty": 2,
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "premium": 3.20,
            "reasoning": "test",
            "cycle_id": "cycle-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "submitted"
    assert captured["args"][0] == "AAPL250117C00200000"
    assert captured["args"][1] == 2


def test_option_exposure_returns_contract_symbols_from_execution(monkeypatch):
    monkeypatch.setattr(
        app_module.execution,
        "option_exposure_contract_symbols",
        lambda: ["AAPL250117C00200000", "MSFT250117C00400000"],
    )

    response = _client().get("/option-exposure")

    assert response.status_code == 200
    assert response.json() == {"contracts": ["AAPL250117C00200000", "MSFT250117C00400000"]}


def test_option_exposure_empty_when_nothing_held(monkeypatch):
    monkeypatch.setattr(app_module.execution, "option_exposure_contract_symbols", lambda: [])

    response = _client().get("/option-exposure")

    assert response.status_code == 200
    assert response.json() == {"contracts": []}


def test_execute_option_rejects_non_buy_side():
    response = _client().post(
        "/execute-option",
        json={
            "contract_symbol": "AAPL250117C00200000",
            "side": "SELL",
            "qty": 2,
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "premium": 3.20,
        },
    )

    assert response.status_code == 422


def test_execute_option_rejects_notional_above_ceiling():
    response = _client().post(
        "/execute-option",
        json={
            "contract_symbol": "AAPL250117C00200000",
            "side": "BUY",
            "qty": 1000,
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "premium": app_module.MAX_OPTION_NOTIONAL / (1000 * 100) + 0.01,
            "reasoning": "test",
            "cycle_id": "cycle-1",
        },
    )

    assert response.status_code == 422


def test_execute_option_accepts_notional_at_ceiling(monkeypatch):
    monkeypatch.setattr(
        app_module.execution,
        "buy_option",
        lambda *a, **k: {"status": "submitted", "reason": "opening_position", "detail": "ok", "order_id": "order-1"},
    )

    response = _client().post(
        "/execute-option",
        json={
            "contract_symbol": "AAPL250117C00200000",
            "side": "BUY",
            "qty": 1000,
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "premium": app_module.MAX_OPTION_NOTIONAL / (1000 * 100),
            "reasoning": "test",
            "cycle_id": "cycle-1",
        },
    )

    assert response.status_code == 200


def test_execute_option_rejects_malformed_expiration():
    response = _client().post(
        "/execute-option",
        json={
            "contract_symbol": "AAPL250117C00200000",
            "side": "BUY",
            "qty": 2,
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "13/45/2025",
            "delta": 0.45,
            "premium": 3.20,
            "reasoning": "test",
            "cycle_id": "cycle-1",
        },
    )

    assert response.status_code == 422
