import json
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytz
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AssetClass, OrderSide, OrderStatus, OrderType, PositionSide
from omegaconf import OmegaConf

from src.common import db
from src.floor_broker import execution

# buy() now calls load_config() itself (fetched live from GitHub in production, see
# src/common/config.py) rather than reading a module-level cfg captured at import time --
# without this fixture every test in this file would attempt a real network fetch, and would be
# coupled to whatever config.yaml on `main` happens to contain (e.g. Task D's
# daily_loss_limit_usd change) instead of the fixed values these tests assert against.
_FAKE_CFG = OmegaConf.create(
    {
        "strategy": {
            "daily_profit_target_usd": 1000,
            "daily_loss_limit_usd": 500,
            "crypto_slP": 0.98,
            "crypto_tpP": 1.03,
            "max_concurrent_positions": 10,
            "position_sizing": "flat_budget",
            "risk_per_trade_usd": None,
        },
        "options_trading": {"max_notional_usd": 2000},
        "eod_flatten": {
            "enabled": False,
            "minutes_before_close": 10,
        },
    }
)


@pytest.fixture(autouse=True)
def _fake_cfg(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _FAKE_CFG)


def _api_error(payload: dict) -> APIError:
    err = APIError.__new__(APIError)
    err.args = (json.dumps(payload),)
    err._error = json.dumps(payload)
    return err


@pytest.fixture(autouse=True)
def _clear_tracked_brackets():
    """_tracked_brackets, _pending_fills, _crypto_stops, _option_positions, and
    _pending_option_fills are module-level state shared across every test in this file -- clear
    them before and after each test so tests can't leak tracking into one another."""
    execution._tracked_brackets.clear()
    execution._pending_fills.clear()
    execution._crypto_stops.clear()
    execution._option_positions.clear()
    execution._pending_option_fills.clear()
    yield
    execution._tracked_brackets.clear()
    execution._pending_fills.clear()
    execution._crypto_stops.clear()
    execution._option_positions.clear()
    execution._pending_option_fills.clear()


@pytest.fixture(autouse=True)
def _kill_switch_inactive(monkeypatch):
    """ROADMAP P0.5: every test in this file exercises buy() without a live k8s API to read the
    real ConfigMap against -- default the switch to inactive so existing BUY-path tests are
    unaffected; tests that care about the switch itself override this explicitly."""
    monkeypatch.setattr(execution.kill_switch, "buy_kill_switch_active", lambda: False)


@pytest.fixture(autouse=True)
def _state_reconciled_by_default(monkeypatch):
    """buy() refuses to submit while tracked state hasn't been reconciled with Alpaca (see
    reconcile_tracked_state_once()) -- default every test in this file to reconciled=True so
    existing BUY-path tests are unaffected; tests that care about this gate override it
    explicitly."""
    monkeypatch.setattr(execution, "_state_reconciled", True)


class FakeAccount:
    """Stands in for alpaca-py's TradeAccount. Defaults to flat daily P&L (equity ==
    last_equity) so the daily profit/loss halt in buy() is a no-op unless a test overrides it."""

    def __init__(self, equity="100000.0", last_equity="100000.0"):
        self.equity = equity
        self.last_equity = last_equity


class FakeTradingClient:
    """Stands in for alpaca-py's TradingClient. `submit_order` raises the given rejection(s)
    in order, then succeeds -- lets tests replay real Alpaca rejection payloads without any
    network access."""

    def __init__(self, rejections=(), account=None, open_positions_count=0):
        self._rejections = list(rejections)
        self.submitted = []
        self._account = account or FakeAccount()
        self._open_positions_count = open_positions_count

    def get_account(self):
        return self._account

    def get_open_position(self, symbol):
        raise APIError("no position")

    def get_orders(self, request):
        return []

    def get_all_positions(self):
        return [object()] * self._open_positions_count

    def submit_order(self, req):
        self.submitted.append(req)
        if self._rejections:
            raise _api_error(self._rejections.pop(0))
        return type("Order", (), {"id": "order-123"})()


class FakeLeg:
    def __init__(self, id, status, type_, filled_avg_price=None, filled_qty=None):
        self.id = id
        self.status = status
        self.type = type_
        self.filled_avg_price = filled_avg_price
        self.filled_qty = filled_qty


class FakeBracketOrder:
    def __init__(self, legs):
        self.legs = legs


class FakeBracketTradingClient:
    """Stands in for alpaca-py's TradingClient for check_bracket_fills() -- `orders_by_id` maps
    a parent order id to either a FakeBracketOrder or an exception instance to raise."""

    def __init__(self, orders_by_id):
        self._orders_by_id = orders_by_id

    def get_order_by_id(self, order_id, filter=None):
        result = self._orders_by_id[order_id]
        if isinstance(result, Exception):
            raise result
        return result


def test_bracket_pricing_uses_ask_not_mid(monkeypatch):
    """Regression for the HYFM rejection: a bracket BUY fills near the ask, not the bid/ask
    mid -- pricing off mid understated the real entry price and got the order rejected.
    TP/SL must be derived from the ask, not from (ask+bid)/2."""
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 2.32)

    req = execution.bracket_buy_with_SLTP("HYFM", budget=5000.0, slP=0.98, tpP=1.05)

    assert req.take_profit.limit_price == pytest.approx(2.32 * 1.05, abs=0.01)
    assert req.stop_loss.stop_price == pytest.approx(2.32 * 0.98, abs=0.01)


def test_round_to_tick_sub_dollar_precision():
    """Regression for MGN base_price=0.1616 rejection: rounding TP/SL to 2 decimals on a
    sub-$1 stock collapses distinct prices onto the same cent (SEC Rule 612 requires
    $0.0001 increments below $1); 4-decimal rounding must be used instead."""
    assert execution._round_to_tick(0.158368) == 0.1584
    assert execution._round_to_tick(2.32 * 0.98) == 2.27


def test_bracket_buy_uses_a_single_quote_for_qty_and_prices(monkeypatch):
    """Regression for P0.8: get_qty() used to fetch its own independent ask, so quantity and
    TP/SL could be priced off different market snapshots. get_current_ask_price() must now be
    called exactly once per bracket-BUY attempt, and the resulting qty must be consistent with
    that single ask."""
    calls = []

    def _fake_ask(symbol):
        calls.append(symbol)
        return 10.0

    monkeypatch.setattr(execution, "get_current_ask_price", _fake_ask)

    req = execution.bracket_buy_with_SLTP("MGN", budget=5000.0, slP=0.98, tpP=1.05)

    assert calls == ["MGN"], "get_current_ask_price must be called exactly once"
    assert req.qty == 500  # int(5000.0 // 10.0)


def test_bracket_buy_raises_on_zero_price_quote(monkeypatch):
    """A zero-price quote means Alpaca has no executable ask; it must not be reported as an
    invalid order-parameter bug."""
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 0.0)

    with pytest.raises(execution.NoAskQuote):
        execution.bracket_buy_with_SLTP("MGN", budget=5000.0, slP=0.98, tpP=1.05)


def test_buy_skips_with_no_ask_quote_reason_when_quote_is_zero(monkeypatch):
    """Regression for live TDCL Slack BUY noise: Alpaca returned ask_price=0.0, which used to be
    surfaced as invalid_order_parameters. It should be a clean no-quote skip instead."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 0.0)

    result = execution.buy("TDCL", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "no_ask_quote"
    assert fake_client.submitted == []


def test_bracket_buy_raises_when_stop_loss_goes_non_positive(monkeypatch):
    """P0.9: on an extremely low-priced symbol, `ask - 0.02` can go to zero or negative --
    stop_loss_px must stay strictly positive and below the reference price, or the order must
    be rejected before submission rather than sent to Alpaca."""
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 0.01)

    with pytest.raises(execution.InvalidOrderParameters):
        execution.bracket_buy_with_SLTP("MGN", budget=5000.0, slP=0.98, tpP=1.05)


def test_buy_skips_with_insufficient_qty_reason_when_budget_affords_less_than_one_share(monkeypatch):
    """P0.9: qty < 1 is a normal, expected outcome (budget too small for the current price at
    all) rather than a bug -- must produce a status="skipped" result, not an exception, and
    must never call submit_order."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10000.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_qty"
    assert fake_client.submitted == []


def test_buy_skips_with_insufficient_qty_reason_on_retry(monkeypatch):
    """Same insufficient-qty skip, but reached via the base_price retry path -- a divergent
    Alpaca base_price can itself push the affordable quantity below one share."""
    rejection = {"base_price": "10000.00", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}
    fake_client = FakeTradingClient(rejections=[rejection])
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 1.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_qty"
    assert len(fake_client.submitted) == 1, "the initial (rejected) attempt submits once; the retry must not submit at all"


def test_buy_skips_with_invalid_order_parameters_reason_when_stop_loss_goes_non_positive(monkeypatch):
    """Regression for a live ANSCW 500: bracket_buy_with_SLTP's $0.02 floor/ceiling clamp can
    push stop_loss_px negative on an extremely low-priced symbol (ask=$0.0097) -- this used to
    propagate InvalidOrderParameters straight out of buy(), which app.py's generic exception
    handler turned into a bare 500 instead of a clean skip. Must return status="skipped" and
    never reach submit_order."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 0.0097)

    result = execution.buy("ANSCW", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "invalid_order_parameters"
    assert fake_client.submitted == []


def test_buy_skips_with_invalid_order_parameters_reason_on_retry(monkeypatch):
    """Same invalid-order-parameters skip, but reached via the base_price retry path -- Alpaca's
    own reported base_price can itself be low enough to hit the same non-positive stop_loss
    clamp on the retry attempt, even though the initial (higher, client-quoted) ask was fine."""
    rejection = {"base_price": "0.0097", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}
    fake_client = FakeTradingClient(rejections=[rejection])
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 0.05)

    result = execution.buy("ANSCW", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "invalid_order_parameters"
    assert len(fake_client.submitted) == 1, "the initial (rejected) attempt submits once; the retry must not submit at all"


def test_get_qty_uses_the_given_ask_not_a_fresh_quote():
    assert execution.get_qty(ask=10.0, budget=5000.0) == 500
    assert execution.get_qty(ask=0.0, budget=5000.0) == 0
    assert execution.get_qty(ask=-1.0, budget=5000.0) == 0


def test_bracket_clamps_tp_sl_to_minimum_cent_distance(monkeypatch):
    """Regression for MGN base_price=0.1577 rejection: tpP/slP's percentage move (5%/2%) is
    less than $0.01 on stocks priced under ~$0.50, so the percentage-based price alone isn't
    enough -- it must be clamped to at least a $0.02 buffer past the reference price."""
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 0.1577)

    req = execution.bracket_buy_with_SLTP("MGN", budget=5000.0, slP=0.98, tpP=1.05)

    assert req.take_profit.limit_price >= 0.1577 + 0.01
    assert req.stop_loss.stop_price <= 0.1577 - 0.01


@pytest.mark.parametrize(
    "symbol, rejection, observed_ask",
    [
        # MGN: small drift, absorbed by the $0.02 buffer alone (kept for historical coverage).
        ("MGN", {"base_price": "0.1616", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}, 0.1616 + 0.02),
        ("MGN", {"base_price": "0.1577", "code": 42210000, "message": "take_profit.limit_price must be >= base_price + 0.01"}, 0.1577 + 0.02),
        ("MGN", {"base_price": "0.1486", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}, 0.1486 + 0.02),
        # MGN: real production drift (ask 0.16 vs base_price 0.1473) that outran the $0.02 buffer.
        ("MGN", {"base_price": "0.1473", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}, 0.16),
        # DFNS: real production drift of ~$1.25 (ask 43.94 vs base_price 42.69) -- proves the fix
        # isn't bounded to penny stocks or to any fixed-cent buffer size.
        ("DFNS", {"base_price": "42.69", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}, 43.94),
        # STKH: real production drift of ~$0.47 (ask 3.35 vs base_price 2.88).
        ("STKH", {"base_price": "2.88", "code": 42210000, "message": "stop_loss.stop_price must be <= base_price - 0.01"}, 3.35),
    ],
    ids=lambda v: f"{v}" if not isinstance(v, dict) else f"base_price={v['base_price']}",
)
def test_buy_retries_with_alpacas_base_price_on_bracket_rejection(monkeypatch, symbol, rejection, observed_ask):
    """Regression for repeated bracket-BUY rejections even after the tick-size and
    percentage-floor fixes: our client-quoted ask can diverge from Alpaca's own base_price on
    thin symbols (e.g. a lagging free-tier quote feed) by anywhere from a few cents (MGN) to
    over a dollar (DFNS) -- no fixed buffer size can bound this. buy() must retry once, pricing
    TP/SL off the base_price Alpaca actually reports in the rejection, regardless of the gap."""
    fake_client = FakeTradingClient(rejections=[rejection])
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: observed_ask)

    result = execution.buy(symbol, "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert len(fake_client.submitted) == 2, "expected one rejected attempt + one retry"

    retry_req = fake_client.submitted[1]
    base_price = float(rejection["base_price"])

    # Same inequalities Alpaca itself enforces server-side.
    assert retry_req.take_profit.limit_price >= base_price + 0.01
    assert retry_req.stop_loss.stop_price <= base_price - 0.01


def test_crypto_buy_rounds_notional_to_2_decimal_places(monkeypatch):
    """Regression for a live BUY BTCUSD rejection: {"code":42210000,"message":"notional value
    must be limited to 2 decimal places"} -- a `budget` with more precision than that (e.g. a
    merged position's market_value) must be rounded before being sent as notional."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 123.456789, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].notional == 123.46


def test_crypto_buy_canonicalizes_live_alpaca_position_symbol_shape(monkeypatch):
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTCUSD", "binance", 123.45, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].symbol == "BTC/USD"
    assert execution._pending_fills["order-123"]["symbol"] == "BTC/USD"


def test_crypto_buy_skips_notional_below_alpacas_minimum(monkeypatch):
    """A budget below Alpaca's crypto minimum notional (code 40310000, "cost basis must be >=
    minimal amount of order 10") must be skipped, not clamped up -- clamping would silently
    submit an order larger than the caller's intended budget."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 3.5, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "budget_below_minimum"
    assert fake_client.submitted == []


def test_crypto_buy_executes_at_exactly_the_minimum_notional(monkeypatch):
    """The minimum itself must still be accepted -- only strictly-below values are skipped."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", execution.MIN_CRYPTO_NOTIONAL, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].notional == execution.MIN_CRYPTO_NOTIONAL


def test_crypto_buy_skips_non_usd_quoted_pair(monkeypatch):
    """Regression for live SHIB/USDT BUY errors: Alpaca paper accounts are funded in USD, so
    submitting a USDT-quoted pair makes Alpaca reject the order for insufficient USDT balance.
    Floor Broker must skip that stale portfolio entry before it reaches Alpaca."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("SHIB/USDT", "binance", 100.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "non_usd_crypto_pair"
    assert fake_client.submitted == []


def test_buy_reraises_on_unrelated_api_error(monkeypatch):
    """buy()'s retry is specific to the base_price mismatch (code 42210000 with a base_price
    field) -- any other rejection must propagate, not be silently retried/swallowed."""
    other_rejection = {"code": 40310000, "message": "unrelated conflict"}
    fake_client = FakeTradingClient(rejections=[other_rejection])
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 1.0)

    with pytest.raises(APIError):
        execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert len(fake_client.submitted) == 1, "must not retry on an unrelated rejection"


class FakePosition:
    def __init__(self, market_value, qty="1"):
        self.qty = qty
        self.market_value = market_value


class FakeExistingPositionTradingClient(FakeTradingClient):
    """Like FakeTradingClient, but get_open_position() returns an existing position instead of
    raising -- used to exercise buy()'s top-up-toward-budget branch."""

    def __init__(self, market_value, rejections=(), account=None):
        super().__init__(rejections=rejections, account=account)
        self._market_value = market_value

    def get_open_position(self, symbol):
        return FakePosition(self._market_value)


def test_buy_skips_with_open_orders_exist_reason_when_a_matching_order_is_open(monkeypatch):
    """An in-flight order for the symbol (BUY not yet filled, or a pending SELL) must still be a
    hard, unconditional skip -- layering a new BUY on top of it is racy regardless of how much
    budget headroom the caller thinks is available."""

    class FakeOpenOrderTradingClient(FakeTradingClient):
        def get_orders(self, request):
            return [type("Order", (), {"symbol": "MGN"})()]

    fake_client = FakeOpenOrderTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "open_orders_exist"
    assert fake_client.submitted == []


def test_buy_skips_with_market_value_unavailable_reason_when_position_market_value_is_none(monkeypatch):
    """market_value is Optional[str] on Alpaca's own Position model. This is a trading-money
    gate, so it must fail closed (skip) rather than guess at remaining budget headroom -- the
    opposite of the Analyst's informational P&L snapshot, which fails open on the same field."""
    fake_client = FakeExistingPositionTradingClient(market_value=None)
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "market_value_unavailable"
    assert fake_client.submitted == []


def test_buy_skips_with_budget_exhausted_reason_when_existing_position_already_meets_budget(monkeypatch):
    fake_client = FakeExistingPositionTradingClient(market_value="5000.00")
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "budget_exhausted"
    assert fake_client.submitted == []


def test_buy_tops_up_existing_stock_position_with_only_the_remaining_budget(monkeypatch):
    """Core fix: budget=$5000, existing position worth $4000 -- buy() must submit a bracket
    order sized off the $1000 remainder, not the full $5000 budget and not a skip."""
    fake_client = FakeExistingPositionTradingClient(market_value="4000.00")
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].qty == 100  # int(1000.0 // 10.0), not int(5000.0 // 10.0)


def test_buy_tops_up_existing_crypto_position_with_only_the_remaining_budget(monkeypatch):
    fake_client = FakeExistingPositionTradingClient(market_value="40.00")
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 100.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].notional == 60.0  # 100.0 - 40.0, not the full 100.0


def test_buy_skips_with_insufficient_qty_reason_when_remaining_budget_affords_less_than_one_share(monkeypatch):
    """The reduced "remaining budget" must flow into the existing InsufficientQuantity path
    unchanged -- no new sizing logic needed for this edge case."""
    fake_client = FakeExistingPositionTradingClient(market_value="4990.00")
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10000.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_qty"
    assert fake_client.submitted == []


def test_buy_skips_with_budget_below_minimum_reason_when_remaining_budget_below_crypto_minimum(monkeypatch):
    """Same idea for crypto: the remainder must flow into the existing MIN_CRYPTO_NOTIONAL check
    unchanged."""
    fake_client = FakeExistingPositionTradingClient(market_value="95.00")
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 100.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "budget_below_minimum"
    assert fake_client.submitted == []


def test_stock_buy_returns_reason_sl_tp_and_tracks_the_bracket_and_pending_fill(monkeypatch):
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert result["reason"] == "opening_position"
    assert result["order_id"] == "order-123"
    assert "fill_price" not in result
    assert result["sl_price"] == pytest.approx(9.8, abs=0.01)
    assert result["tp_price"] == pytest.approx(10.5, abs=0.01)
    assert execution._tracked_brackets["MGN"] == "order-123"
    assert execution._pending_fills["order-123"] == {
        "symbol": "MGN",
        "action": "BUY",
        "reason": "opening_position",
        "sl_price": pytest.approx(9.8, abs=0.01),
        "tp_price": pytest.approx(10.5, abs=0.01),
        "crypto_slP": None,
        "crypto_tpP": None,
    }


def test_stock_buy_coerces_a_uuid_order_id_to_str(monkeypatch):
    """Alpaca's real SDK returns order.id as a uuid.UUID, not a str -- the returned dict's
    order_id must be a str so it validates against ExecuteResponse.order_id: str | None in
    app.py. A UUID leaking through here previously crashed /execute with a 500 on every
    successful BUY, even though the order had already been submitted and would go on to fill."""
    order_uuid = uuid.uuid4()

    class FakeUUIDOrderClient(FakeTradingClient):
        def submit_order(self, req):
            self.submitted.append(req)
            return type("Order", (), {"id": order_uuid})()

    fake_client = FakeUUIDOrderClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["order_id"] == str(order_uuid)
    assert isinstance(result["order_id"], str)


def test_crypto_buy_has_no_sl_tp_price_and_is_not_tracked_as_a_bracket(monkeypatch):
    """Crypto BUYs are plain notional market orders, not brackets -- there's no TP/SL leg to
    later watch for a fill."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 100.0, slP=0.98, tpP=1.05)

    assert result["sl_price"] is None
    assert result["tp_price"] is None
    assert "BTC/USD" not in execution._tracked_brackets


def test_crypto_buy_stores_strategy_crypto_slp_tpp_for_the_pending_fill(monkeypatch):
    """Crypto has no fill price known at submission time (a plain notional market order), so the
    strategy.crypto_slP/crypto_tpP multipliers -- not yet a concrete sl_price/tp_price -- must be
    stashed on the pending-fill entry for check_pending_fills() to apply once the fill lands."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 100.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    pending = execution._pending_fills["order-123"]
    assert pending["crypto_slP"] == _FAKE_CFG.strategy.crypto_slP
    assert pending["crypto_tpP"] == _FAKE_CFG.strategy.crypto_tpP


def test_sell_returns_dealer_signal_reason_and_untracks_bracket_and_tracks_pending_fill(monkeypatch):
    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "5"})()

        def submit_order(self, req):
            return type("Order", (), {"id": "sell-order-1"})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())
    execution._tracked_brackets["MGN"] = "parent-order-should-be-cleared"

    result = execution.sell("MGN")

    assert result["status"] == "submitted"
    assert result["reason"] == "dealer_signal"
    assert result["order_id"] == "sell-order-1"
    assert "fill_price" not in result
    assert "MGN" not in execution._tracked_brackets
    assert execution._pending_fills["sell-order-1"] == {
        "symbol": "MGN",
        "action": "SELL",
        "reason": "dealer_signal",
        "sl_price": None,
        "tp_price": None,
    }


def test_sell_coerces_a_uuid_order_id_to_str(monkeypatch):
    """Same UUID-leak regression as the BUY path (see test_stock_buy_coerces_a_uuid_order_id_to_str)
    -- Alpaca's real order.id is a uuid.UUID, and the returned order_id must be a str."""
    order_uuid = uuid.uuid4()

    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "5"})()

        def submit_order(self, req):
            return type("Order", (), {"id": order_uuid})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())

    result = execution.sell("MGN")

    assert result["order_id"] == str(order_uuid)
    assert isinstance(result["order_id"], str)


def test_sell_accepts_a_reason_override_and_clears_tracked_crypto_stop(monkeypatch):
    """check_crypto_stops() calls sell(symbol, reason=...) so the eventual fill notice says
    stop_loss/take_profit rather than the default dealer_signal -- and the triggering sell must
    itself clear the symbol's _crypto_stops entry so the poller can't double-sell it."""
    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "1"})()

        def submit_order(self, req):
            return type("Order", (), {"id": "sell-order-1"})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())
    execution._crypto_stops["BTC/USD"] = (98.0, 103.0)

    result = execution.sell("BTC/USD", reason="stop_loss")

    assert result["status"] == "submitted"
    assert result["reason"] == "stop_loss"
    assert execution._pending_fills["sell-order-1"]["reason"] == "stop_loss"
    assert "BTC/USD" not in execution._crypto_stops


def test_check_crypto_stops_sells_when_bid_drops_to_or_below_stop_loss(monkeypatch):
    execution._crypto_stops["BTC/USD"] = (98.0, 103.0)
    monkeypatch.setattr(execution, "get_current_bid_price", lambda symbol: 98.0)

    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "1"})()

        def submit_order(self, req):
            return type("Order", (), {"id": "sell-order-1"})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())

    events = execution.check_crypto_stops()

    assert len(events) == 1
    assert events[0]["symbol"] == "BTC/USD"
    assert events[0]["reason"] == "stop_loss"
    assert events[0]["bid_price"] == 98.0
    assert events[0]["sell_result"]["status"] == "submitted"
    assert "BTC/USD" not in execution._crypto_stops


def test_check_crypto_stops_sells_when_bid_rises_to_or_above_take_profit(monkeypatch):
    execution._crypto_stops["BTC/USD"] = (98.0, 103.0)
    monkeypatch.setattr(execution, "get_current_bid_price", lambda symbol: 103.0)

    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "1"})()

        def submit_order(self, req):
            return type("Order", (), {"id": "sell-order-1"})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())

    events = execution.check_crypto_stops()

    assert events[0]["reason"] == "take_profit"
    assert "BTC/USD" not in execution._crypto_stops


def test_check_crypto_stops_keeps_tracking_when_bid_is_within_bounds(monkeypatch):
    execution._crypto_stops["BTC/USD"] = (98.0, 103.0)
    monkeypatch.setattr(execution, "get_current_bid_price", lambda symbol: 100.0)

    events = execution.check_crypto_stops()

    assert events == []
    assert execution._crypto_stops["BTC/USD"] == (98.0, 103.0)


def test_check_crypto_stops_keeps_tracking_symbol_on_transient_price_fetch_error(monkeypatch):
    execution._crypto_stops["BTC/USD"] = (98.0, 103.0)

    def _raise(symbol):
        raise APIError("unreachable")

    monkeypatch.setattr(execution, "get_current_bid_price", _raise)

    events = execution.check_crypto_stops()

    assert events == []
    assert execution._crypto_stops["BTC/USD"] == (98.0, 103.0)


def test_check_crypto_stops_skips_sell_if_stop_changed_after_snapshot(monkeypatch):
    execution._crypto_stops["BTC/USD"] = (98.0, 103.0)

    def _bid_after_concurrent_update(symbol):
        execution._crypto_stops["BTC/USD"] = (90.0, 110.0)
        return 98.0

    sells = []
    monkeypatch.setattr(execution, "get_current_bid_price", _bid_after_concurrent_update)
    monkeypatch.setattr(execution, "sell", lambda symbol, reason: sells.append((symbol, reason)))

    events = execution.check_crypto_stops()

    assert events == []
    assert sells == []
    assert execution._crypto_stops["BTC/USD"] == (90.0, 110.0)


class FakeClock:
    def __init__(self, is_open, minutes_to_close=5):
        self.is_open = is_open
        self.timestamp = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
        self.next_close = self.timestamp + timedelta(minutes=minutes_to_close)


class FakeEodPosition:
    def __init__(
        self,
        symbol,
        asset_class,
        qty="1",
        unrealized_pl=None,
        avg_entry_price=None,
        current_price=None,
    ):
        self.symbol = symbol
        self.asset_class = asset_class
        self.qty = qty
        self.unrealized_pl = unrealized_pl
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price


class FakeEodFlattenTradingClient:
    """Stands in for alpaca-py's TradingClient for check_eod_flatten() -- also backs the real
    sell() it calls per-symbol, so get_open_position() must resolve every symbol get_all_positions()
    returns."""

    def __init__(self, clock, positions):
        self._clock = clock
        self._positions = positions
        self.submitted = []

    def get_clock(self):
        return self._clock

    def get_all_positions(self):
        return self._positions

    def get_open_position(self, symbol):
        for position in self._positions:
            if position.symbol.replace("/", "") == symbol:
                return position
        raise APIError("no position")

    def submit_order(self, req):
        self.submitted.append(req)
        return type("Order", (), {"id": f"order-{req.symbol}"})()


def _eod_flatten_cfg(enabled=True, minutes_before_close=10, conditional=False, max_days_held_loss=5):
    return OmegaConf.create(
        {
            "strategy": _FAKE_CFG.strategy,
            "eod_flatten": {
                "enabled": enabled,
                "minutes_before_close": minutes_before_close,
                "conditional": conditional,
                "max_days_held_loss": max_days_held_loss,
            },
        }
    )


def test_check_eod_flatten_is_a_noop_when_disabled_by_config(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(enabled=False))
    fake_client = FakeEodFlattenTradingClient(FakeClock(is_open=True), [FakeEodPosition("MGN", AssetClass.US_EQUITY)])
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_eod_flatten()

    assert events == []
    assert fake_client.submitted == []


def test_check_eod_flatten_is_a_noop_when_market_is_closed(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg())
    fake_client = FakeEodFlattenTradingClient(FakeClock(is_open=False), [FakeEodPosition("MGN", AssetClass.US_EQUITY)])
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_eod_flatten()

    assert events == []
    assert fake_client.submitted == []


def test_check_eod_flatten_is_a_noop_when_not_yet_within_the_closing_window(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(minutes_before_close=10))
    fake_client = FakeEodFlattenTradingClient(
        FakeClock(is_open=True, minutes_to_close=30), [FakeEodPosition("MGN", AssetClass.US_EQUITY)]
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_eod_flatten()

    assert events == []
    assert fake_client.submitted == []


def test_check_eod_flatten_sells_stock_positions_and_skips_crypto(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(minutes_before_close=10))
    fake_client = FakeEodFlattenTradingClient(
        FakeClock(is_open=True, minutes_to_close=5),
        [FakeEodPosition("MGN", AssetClass.US_EQUITY), FakeEodPosition("BTC/USD", AssetClass.CRYPTO)],
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_eod_flatten()

    assert len(events) == 1
    assert events[0]["symbol"] == "MGN"
    assert events[0]["reason"] == "eod_flatten"
    assert events[0]["sell_result"]["status"] == "submitted"
    assert [req.symbol for req in fake_client.submitted] == ["MGN"]


def test_check_eod_flatten_excludes_a_skipped_sell_from_the_returned_events(monkeypatch):
    """sell() returns status="skipped" when get_open_position() resolves to qty<=0 -- shouldn't
    happen for a symbol get_all_positions() itself just returned, but if it does (e.g. a race with
    a fill that already closed it), it must not be reported as a flatten event."""
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(minutes_before_close=10))
    fake_client = FakeEodFlattenTradingClient(
        FakeClock(is_open=True, minutes_to_close=5), [FakeEodPosition("MGN", AssetClass.US_EQUITY, qty="0")]
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_eod_flatten()

    assert events == []
    assert fake_client.submitted == []


def test_check_eod_flatten_conditional_flattens_everything_when_aggregate_pl_is_up(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(conditional=True))
    fake_client = FakeEodFlattenTradingClient(
        FakeClock(is_open=True, minutes_to_close=5),
        [
            FakeEodPosition("MGN", AssetClass.US_EQUITY, unrealized_pl="50"),
            FakeEodPosition("ACME", AssetClass.US_EQUITY, unrealized_pl="-20"),
        ],
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_eod_flatten()

    assert {e["symbol"] for e in events} == {"MGN", "ACME"}
    assert {req.symbol for req in fake_client.submitted} == {"MGN", "ACME"}


def test_check_eod_flatten_conditional_holds_everything_when_aggregate_pl_is_down(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(conditional=True, max_days_held_loss=5))
    fake_client = FakeEodFlattenTradingClient(
        FakeClock(is_open=True, minutes_to_close=5),
        [
            FakeEodPosition("MGN", AssetClass.US_EQUITY, unrealized_pl="-50"),
            FakeEodPosition("ACME", AssetClass.US_EQUITY, unrealized_pl="20"),
        ],
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(db, "fetch_position_opened_at", lambda symbol: None)  # untracked -> 0 days held

    events = execution.check_eod_flatten()

    assert events == []
    assert fake_client.submitted == []


def test_check_eod_flatten_conditional_force_flattens_a_position_past_the_days_held_cap(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _eod_flatten_cfg(conditional=True, max_days_held_loss=3))
    clock = FakeClock(is_open=True, minutes_to_close=5)
    fake_client = FakeEodFlattenTradingClient(
        clock,
        [
            FakeEodPosition("MGN", AssetClass.US_EQUITY, unrealized_pl="-10"),
            FakeEodPosition("ACME", AssetClass.US_EQUITY, unrealized_pl="-5"),
        ],
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    def fake_opened_at(symbol):
        if symbol == "MGN":
            return clock.timestamp - timedelta(days=4)  # past the 3-day cap
        return clock.timestamp - timedelta(days=1)  # under the cap

    monkeypatch.setattr(db, "fetch_position_opened_at", fake_opened_at)

    events = execution.check_eod_flatten()

    assert [e["symbol"] for e in events] == ["MGN"]
    assert [req.symbol for req in fake_client.submitted] == ["MGN"]


def test_flatten_all_crypto_sells_crypto_positions_and_skips_stock(monkeypatch):
    fake_client = FakeEodFlattenTradingClient(
        FakeClock(is_open=True),
        [FakeEodPosition("BTC/USD", AssetClass.CRYPTO), FakeEodPosition("MGN", AssetClass.US_EQUITY)],
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.flatten_all_crypto()

    assert len(events) == 1
    assert events[0]["symbol"] == "BTC/USD"
    assert events[0]["reason"] == "power_down_flatten"
    assert events[0]["sell_result"]["status"] == "submitted"
    assert [req.symbol for req in fake_client.submitted] == ["BTC/USD"]


def test_flatten_all_crypto_is_a_noop_with_no_open_crypto_positions(monkeypatch):
    fake_client = FakeEodFlattenTradingClient(FakeClock(is_open=True), [FakeEodPosition("MGN", AssetClass.US_EQUITY)])
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.flatten_all_crypto()

    assert events == []
    assert fake_client.submitted == []


def test_flatten_all_crypto_excludes_a_skipped_sell_from_the_returned_events(monkeypatch):
    fake_client = FakeEodFlattenTradingClient(FakeClock(is_open=True), [FakeEodPosition("BTC/USD", AssetClass.CRYPTO, qty="0")])
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.flatten_all_crypto()

    assert events == []
    assert fake_client.submitted == []


def test_flatten_all_crypto_uses_a_custom_reason(monkeypatch):
    fake_client = FakeEodFlattenTradingClient(FakeClock(is_open=True), [FakeEodPosition("BTC/USD", AssetClass.CRYPTO)])
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.flatten_all_crypto(reason="test_reason")

    assert events[0]["reason"] == "test_reason"


def test_flatten_all_options_skips_short_positions(monkeypatch):
    """Regression: this system never opens a short option position itself, so a short reaching
    flatten_all_options() must be skipped, not sold -- selling it would open MORE short instead of
    closing it."""

    class FakeLongPosition:
        symbol = "AAPL250117C00200000"
        side = PositionSide.LONG
        asset_class = AssetClass.US_OPTION

    class FakeShortPosition:
        symbol = "MSFT250117P00300000"
        side = PositionSide.SHORT
        asset_class = AssetClass.US_OPTION

    class FakeTradingClient2:
        def get_all_positions(self):
            return [FakeLongPosition(), FakeShortPosition()]

    sold = []
    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())
    monkeypatch.setattr(execution, "sell_option", lambda contract_symbol, reason: sold.append(contract_symbol) or {"status": "submitted"})

    execution.flatten_all_options()

    assert sold == ["AAPL250117C00200000"]


def test_check_bracket_fills_reports_take_profit_leg_filled(monkeypatch):
    execution._tracked_brackets["MGN"] = "parent-1"
    tp_leg = FakeLeg("tp-leg-1", OrderStatus.FILLED, OrderType.LIMIT, filled_avg_price="13.50", filled_qty="10")
    sl_leg = FakeLeg("sl-leg-1", OrderStatus.CANCELED, OrderType.STOP)
    fake_client = FakeBracketTradingClient({"parent-1": FakeBracketOrder([tp_leg, sl_leg])})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_bracket_fills()

    assert events == [
        {"kind": "fill", "symbol": "MGN", "order_id": "tp-leg-1", "reason": "take_profit", "fill_price": 13.50, "qty": 10.0}
    ]
    assert "MGN" not in execution._tracked_brackets


def test_check_bracket_fills_reports_stop_loss_leg_filled(monkeypatch):
    execution._tracked_brackets["MGN"] = "parent-1"
    tp_leg = FakeLeg("tp-leg-1", OrderStatus.CANCELED, OrderType.LIMIT)
    sl_leg = FakeLeg("sl-leg-1", OrderStatus.FILLED, OrderType.STOP, filled_avg_price="9.80", filled_qty="10")
    fake_client = FakeBracketTradingClient({"parent-1": FakeBracketOrder([tp_leg, sl_leg])})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_bracket_fills()

    assert events == [
        {"kind": "fill", "symbol": "MGN", "order_id": "sl-leg-1", "reason": "stop_loss", "fill_price": 9.80, "qty": 10.0}
    ]
    assert "MGN" not in execution._tracked_brackets


def test_check_bracket_fills_keeps_tracking_while_both_legs_still_open(monkeypatch):
    execution._tracked_brackets["MGN"] = "parent-1"
    legs = [FakeLeg("tp-leg-1", OrderStatus.NEW, OrderType.LIMIT), FakeLeg("sl-leg-1", OrderStatus.NEW, OrderType.STOP)]
    fake_client = FakeBracketTradingClient({"parent-1": FakeBracketOrder(legs)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_bracket_fills()

    assert events == []
    assert execution._tracked_brackets["MGN"] == "parent-1"


def test_check_bracket_fills_untracks_symbol_when_both_legs_end_without_a_fill(monkeypatch):
    """e.g. the position was closed some other way and Alpaca cancelled both legs -- must stop
    polling rather than track forever, and must report the no-fill outcome rather than going
    silent (nothing else ever covers this case)."""
    execution._tracked_brackets["MGN"] = "parent-1"
    legs = [FakeLeg("tp-leg-1", OrderStatus.CANCELED, OrderType.LIMIT), FakeLeg("sl-leg-1", OrderStatus.CANCELED, OrderType.STOP)]
    fake_client = FakeBracketTradingClient({"parent-1": FakeBracketOrder(legs)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_bracket_fills()

    assert events == [
        {"kind": "terminal", "symbol": "MGN", "order_id": "parent-1", "leg_statuses": ["canceled", "canceled"]}
    ]
    assert "MGN" not in execution._tracked_brackets


def test_buy_skips_with_kill_switch_reason_when_active_and_touches_no_client(monkeypatch):
    """ROADMAP P0.5: an active BUY kill switch must block the BUY before any position/order
    lookup or submission -- no `trading_client` monkeypatch is set up here on purpose, so this
    test would error out trying to reach the real Alpaca API if the early-exit didn't fire
    first."""
    monkeypatch.setattr(execution.kill_switch, "buy_kill_switch_active", lambda: True)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result == {
        "status": "skipped",
        "reason": "buy_kill_switch_active",
        "detail": "BUY kill switch is active",
    }


def test_sell_still_permitted_when_kill_switch_active(monkeypatch):
    """ROADMAP P0.5: the switch only ever blocks new BUY exposure -- SELL must be completely
    unaffected by its state."""
    monkeypatch.setattr(execution.kill_switch, "buy_kill_switch_active", lambda: True)

    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "5"})()

        def submit_order(self, req):
            return type("Order", (), {"id": "sell-order-1"})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())

    result = execution.sell("MGN")

    assert result["status"] == "submitted"


def test_buy_skips_when_daily_profit_target_reached(monkeypatch):
    account = FakeAccount(equity="101000.0", last_equity="100000.0")  # +$1000, == strategy.daily_profit_target_usd
    fake_client = FakeTradingClient(account=account)
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "daily_profit_target_reached"
    assert fake_client.submitted == []


def test_buy_skips_when_daily_loss_limit_reached(monkeypatch):
    account = FakeAccount(equity="99500.0", last_equity="100000.0")  # -$500, == strategy.daily_loss_limit_usd
    fake_client = FakeTradingClient(account=account)
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "daily_loss_limit_reached"
    assert fake_client.submitted == []


def test_buy_proceeds_when_daily_pnl_is_within_bounds(monkeypatch):
    account = FakeAccount(equity="100200.0", last_equity="100000.0")  # +$200, under target
    fake_client = FakeTradingClient(account=account)
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"


def test_buy_skips_when_max_concurrent_positions_reached(monkeypatch):
    fake_client = FakeTradingClient(open_positions_count=10)  # == _FAKE_CFG.strategy.max_concurrent_positions
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "max_concurrent_positions_reached"
    assert fake_client.submitted == []


def test_buy_proceeds_when_below_max_concurrent_positions(monkeypatch):
    fake_client = FakeTradingClient(open_positions_count=9)
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"


def test_stock_buy_skips_when_bid_ask_spread_exceeds_configured_cap(monkeypatch):
    cfg = OmegaConf.create(_FAKE_CFG)
    cfg.strategy.max_bid_ask_spread_pct = 0.03
    monkeypatch.setattr(execution, "load_config", lambda: cfg)
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)
    monkeypatch.setattr(execution, "get_current_bid_price", lambda symbol: 9.5)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "skipped"
    assert result["reason"] == "wide_bid_ask_spread"
    assert fake_client.submitted == []


def test_stock_buy_uses_single_spread_checked_ask_for_order_pricing(monkeypatch):
    cfg = OmegaConf.create(_FAKE_CFG)
    cfg.strategy.max_bid_ask_spread_pct = 0.03
    monkeypatch.setattr(execution, "load_config", lambda: cfg)
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)
    monkeypatch.setattr(execution, "get_current_bid_price", lambda symbol: 9.9)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].qty == 500


def test_buy_tops_up_existing_position_without_regard_to_max_concurrent_positions(monkeypatch):
    """Topping up a symbol that's already open must never be blocked by the concurrent-positions
    cap -- it isn't a new position. FakeExistingPositionTradingClient has no get_all_positions
    override, so this also proves the cap check never even calls it for a top-up."""
    fake_client = FakeExistingPositionTradingClient(market_value="4000.00")
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"


def _risk_based_cfg(risk_per_trade_usd=100):
    return OmegaConf.create(
        {
            "strategy": {
                "daily_profit_target_usd": 1000,
                "daily_loss_limit_usd": 500,
                "crypto_slP": 0.98,
                "crypto_tpP": 1.03,
                "max_concurrent_positions": 10,
                "position_sizing": "risk_based",
                "risk_per_trade_usd": risk_per_trade_usd,
            },
            "eod_flatten": {"enabled": False, "minutes_before_close": 10},
        }
    )


def test_risk_based_sizing_caps_budget_to_risk_per_trade_over_stop_distance(monkeypatch):
    """risk_per_trade_usd=100, slP=0.95 -> the stop loses 5% of the budget, so the largest budget
    that risks exactly $100 at the stop is 100 / 0.05 = $2000."""
    monkeypatch.setattr(execution, "load_config", lambda: _risk_based_cfg(risk_per_trade_usd=100))
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 10000.0, slP=0.95, tpP=1.05)

    assert result["status"] == "submitted"
    # ~200 shares (100 / 0.05 / 10) -- computed the same way the production code does rather than
    # hardcoded, since 1 - 0.95 isn't exactly representable in floating point.
    assert fake_client.submitted[0].qty == int((100 / (1 - 0.95)) // 10.0)


def test_risk_based_sizing_never_increases_budget_above_authorized(monkeypatch):
    """The risk cap only ever shrinks the requested budget -- a generous risk_per_trade_usd must
    never inflate exposure beyond what the Analyst/Dealer already authorized."""
    monkeypatch.setattr(execution, "load_config", lambda: _risk_based_cfg(risk_per_trade_usd=1000))
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 500.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].qty == 50  # int(500.0 // 10.0), unchanged -- the $50000 risk cap is far above it


def test_risk_based_sizing_uses_crypto_slp_for_crypto_buys(monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda: _risk_based_cfg(risk_per_trade_usd=1))
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)

    result = execution.buy("BTC/USD", "binance", 100.0, slP=0.98, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].notional == 50.0  # 1 / (1 - 0.98) == 50, capped down from the $100 request


def test_flat_budget_sizing_leaves_budget_unchanged(monkeypatch):
    """Default mode (_FAKE_CFG.strategy.position_sizing == "flat_budget") -- confirms the risk
    cap is a no-op unless risk_based is explicitly active, even with a stop wide enough that a
    risk cap would otherwise bite hard."""
    fake_client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution, "get_current_ask_price", lambda symbol: 10.0)

    result = execution.buy("MGN", "stocks", 5000.0, slP=0.5, tpP=1.05)

    assert result["status"] == "submitted"
    assert fake_client.submitted[0].qty == 500  # int(5000.0 // 10.0), unaffected by the wide stop


def test_sell_still_permitted_when_daily_loss_limit_reached(monkeypatch):
    """The halt only ever blocks new BUY exposure -- SELL must be completely unaffected,
    matching the existing kill-switch precedent."""
    class FakeSellClient:
        def get_open_position(self, symbol):
            return type("Position", (), {"qty": "5"})()

        def submit_order(self, req):
            return type("Order", (), {"id": "sell-order-1"})()

    monkeypatch.setattr(execution, "trading_client", FakeSellClient())

    result = execution.sell("MGN")

    assert result["status"] == "submitted"


def test_check_bracket_fills_untracks_symbol_on_confirmed_not_found(monkeypatch):
    """A cancelled/expired parent order can eventually 404 -- treat a *confirmed* not-found as
    nothing left to watch rather than retrying forever."""
    execution._tracked_brackets["MGN"] = "parent-1"
    fake_client = FakeBracketTradingClient({"parent-1": _api_error({"code": execution.ORDER_NOT_FOUND_CODE, "message": "order not found"})})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_bracket_fills()

    assert events == []
    assert "MGN" not in execution._tracked_brackets


def test_check_bracket_fills_keeps_tracking_symbol_on_transient_api_error(monkeypatch):
    """A rate limit / 5xx / network blip must not drop tracking -- that would silently stop
    watching a still-live bracket. The symbol stays tracked and the failure is recorded so it
    can be observed, distinct from a confirmed not-found."""
    execution._tracked_brackets["MGN"] = "parent-1"
    fake_client = FakeBracketTradingClient({"parent-1": _api_error({"code": 50000000, "message": "internal server error"})})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_bracket_fills()

    assert events == []
    assert execution._tracked_brackets["MGN"] == {"order_id": "parent-1", "poll_failures": 1}

    events = execution.check_bracket_fills()

    assert events == []
    assert execution._tracked_brackets["MGN"] == {"order_id": "parent-1", "poll_failures": 2}


class FakePendingOrder:
    def __init__(self, filled_avg_price=None, status=OrderStatus.FILLED):
        self.filled_avg_price = filled_avg_price
        self.status = status


class FakePendingFillTradingClient:
    """Stands in for alpaca-py's TradingClient for check_pending_fills() -- `orders_by_id` maps
    an order id to either a FakePendingOrder or an exception instance to raise."""

    def __init__(self, orders_by_id):
        self._orders_by_id = orders_by_id

    def get_order_by_id(self, order_id, filter=None):
        result = self._orders_by_id[order_id]
        if isinstance(result, Exception):
            raise result
        return result


def test_check_pending_fills_reports_a_filled_order(monkeypatch):
    execution._pending_fills["order-1"] = {"symbol": "MGN", "action": "BUY", "reason": "opening_position", "sl_price": 9.8, "tp_price": 10.5}
    fake_client = FakePendingFillTradingClient({"order-1": FakePendingOrder(filled_avg_price="10.05")})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_fills()

    assert events == [
        {
            "symbol": "MGN",
            "action": "BUY",
            "reason": "opening_position",
            "sl_price": 9.8,
            "tp_price": 10.5,
            "kind": "fill",
            "order_id": "order-1",
            "fill_price": 10.05,
        }
    ]
    assert "order-1" not in execution._pending_fills


def test_check_pending_fills_starts_tracking_a_crypto_stop_on_buy_fill(monkeypatch):
    execution._pending_fills["order-1"] = {
        "symbol": "BTC/USD",
        "action": "BUY",
        "reason": "opening_position",
        "sl_price": None,
        "tp_price": None,
        "crypto_slP": 0.98,
        "crypto_tpP": 1.03,
    }
    fake_client = FakePendingFillTradingClient({"order-1": FakePendingOrder(filled_avg_price="100.0")})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    execution.check_pending_fills()

    assert execution._crypto_stops["BTC/USD"] == pytest.approx((98.0, 103.0))


def test_check_pending_fills_does_not_track_a_crypto_stop_for_a_stock_fill(monkeypatch):
    """A stock BUY's pending-fill entry has crypto_slP/crypto_tpP left at None (see buy()) --
    must not be mistaken for a crypto fill."""
    execution._pending_fills["order-1"] = {
        "symbol": "MGN",
        "action": "BUY",
        "reason": "opening_position",
        "sl_price": 9.8,
        "tp_price": 10.5,
        "crypto_slP": None,
        "crypto_tpP": None,
    }
    fake_client = FakePendingFillTradingClient({"order-1": FakePendingOrder(filled_avg_price="10.05")})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    execution.check_pending_fills()

    assert execution._crypto_stops == {}


def test_check_pending_fills_keeps_tracking_an_unfilled_order(monkeypatch):
    execution._pending_fills["order-1"] = {"symbol": "MGN", "action": "BUY", "reason": "opening_position", "sl_price": None, "tp_price": None}
    fake_client = FakePendingFillTradingClient({"order-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.NEW)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_fills()

    assert events == []
    assert "order-1" in execution._pending_fills


def test_check_pending_fills_untracks_order_on_terminal_no_fill_status(monkeypatch):
    """Rejected/canceled/expired must be reported, not go silent -- no /execute response ever
    covers this outcome since it's only known after the fact."""
    execution._pending_fills["order-1"] = {"symbol": "MGN", "action": "BUY", "reason": "opening_position", "sl_price": None, "tp_price": None}
    fake_client = FakePendingFillTradingClient({"order-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.CANCELED)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_fills()

    assert events == [
        {
            "symbol": "MGN",
            "action": "BUY",
            "reason": "opening_position",
            "sl_price": None,
            "tp_price": None,
            "kind": "terminal",
            "order_id": "order-1",
            "order_status": "canceled",
        }
    ]
    assert "order-1" not in execution._pending_fills


def test_check_pending_fills_untracks_order_on_confirmed_not_found(monkeypatch):
    execution._pending_fills["order-1"] = {"symbol": "MGN", "action": "BUY", "reason": "opening_position", "sl_price": None, "tp_price": None}
    fake_client = FakePendingFillTradingClient({"order-1": _api_error({"code": execution.ORDER_NOT_FOUND_CODE, "message": "order not found"})})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_fills()

    assert events == []
    assert "order-1" not in execution._pending_fills


def test_check_pending_fills_keeps_tracking_order_on_transient_api_error(monkeypatch):
    """A rate limit / 5xx / network blip must not drop tracking of a still-live order -- the
    entry stays and the failure count is recorded on it."""
    execution._pending_fills["order-1"] = {"symbol": "MGN", "action": "BUY", "reason": "opening_position", "sl_price": None, "tp_price": None}
    fake_client = FakePendingFillTradingClient({"order-1": _api_error({"code": 50000000, "message": "internal server error"})})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_fills()

    assert events == []
    assert execution._pending_fills["order-1"]["poll_failures"] == 1

    events = execution.check_pending_fills()

    assert events == []
    assert execution._pending_fills["order-1"]["poll_failures"] == 2


def test_check_pending_fills_clears_poll_failures_once_order_is_reachable_again(monkeypatch):
    """A transient failure must not leave a stale poll_failures count behind once the order is
    successfully observed again."""
    execution._pending_fills["order-1"] = {
        "symbol": "MGN",
        "action": "BUY",
        "reason": "opening_position",
        "sl_price": None,
        "tp_price": None,
        "poll_failures": 3,
    }
    fake_client = FakePendingFillTradingClient({"order-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.NEW)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    execution.check_pending_fills()

    assert "poll_failures" not in execution._pending_fills["order-1"]


def test_check_pending_option_fills_reports_a_filled_order_and_commits_tracked_state(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-1": FakePendingOrder(filled_avg_price="3.35")})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_opened", lambda *a, **k: recorded_db.setdefault("opened", (a, k)))

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000",
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "qty": 2,
            "reasoning": "test reasoning",
            "cycle_id": "cycle-1",
            "kind": "fill",
            "order_id": "order-opt-1",
            "fill_price": 3.35,
        }
    ]
    assert "order-opt-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"] == {
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "entry_premium": 3.35,
            "qty": 2,
        }
    assert recorded_db["opened"] == (
        ("AAPL", "AAPL250117C00200000", "call", 200.0, "2025-01-17", 0.45, 3.35, 2, "test reasoning", "cycle-1"),
        {},
    )


def test_check_pending_option_fills_uses_actual_filled_qty_when_available(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
    }
    order = FakePendingOrder(filled_avg_price="3.35")
    order.filled_qty = "1"  # partial fill
    fake_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution.db, "record_options_trade_opened", lambda *a, **k: None)

    events = execution.check_pending_option_fills()

    assert events[0]["qty"] == 1
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 1


def test_check_pending_option_fills_keeps_tracking_a_partial_fill_first_observation(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
    }
    order = FakePendingOrder(filled_avg_price="3.35", status=OrderStatus.PARTIALLY_FILLED)
    order.filled_qty = "1"
    fake_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_opened", lambda *a, **k: recorded_db.setdefault("opened", (a, k)))
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: recorded_db.setdefault("updated", (a, k)))

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000",
            "symbol": "AAPL",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "qty": 1,
            "reasoning": "test reasoning",
            "cycle_id": "cycle-1",
            "kind": "fill",
            "order_id": "order-opt-1",
            "fill_price": 3.35,
        }
    ]
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 1
        assert execution._option_positions["AAPL250117C00200000"]["entry_premium"] == 3.35
    assert "opened" in recorded_db
    assert "updated" not in recorded_db
    assert "order-opt-1" in execution._pending_option_fills
    assert execution._pending_option_fills["order-opt-1"]["db_row_opened"] is True


def test_check_pending_option_fills_updates_tracked_state_on_second_partial_fill_observation(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
        "db_row_opened": True,
    }
    order = FakePendingOrder(filled_avg_price="3.50", status=OrderStatus.PARTIALLY_FILLED)
    order.filled_qty = "2"
    fake_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_opened", lambda *a, **k: recorded_db.setdefault("opened", (a, k)))
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: recorded_db.setdefault("updated", (a, k)))

    events = execution.check_pending_option_fills()

    assert events == []
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 2
        assert execution._option_positions["AAPL250117C00200000"]["entry_premium"] == 3.50
    assert recorded_db["updated"] == (("AAPL250117C00200000", 3.50, 2), {})
    assert "opened" not in recorded_db
    assert "order-opt-1" in execution._pending_option_fills


def test_check_pending_option_fills_finalizes_partial_fill_once_order_goes_terminal(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
        "db_row_opened": True,
    }
    order = FakePendingOrder(filled_avg_price="3.50", status=OrderStatus.CANCELED)
    order.filled_qty = "1"
    fake_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: None)

    events = execution.check_pending_option_fills()

    assert events == []
    assert "order-opt-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 1
        assert execution._option_positions["AAPL250117C00200000"]["entry_premium"] == 3.50


def test_check_pending_option_fills_treats_unusual_fill_carrying_status_as_terminal(monkeypatch):
    """Regression (fix-loop round 1, finding 1): a fill-carrying order can land on a status that is
    neither FILLED nor in _TERMINAL_NO_FILL -- e.g. DONE_FOR_DAY, a real Alpaca status, plausible
    for options since they are DAY-only. Before the fix, `order.status == FILLED or status in
    _TERMINAL_NO_FILL` would never be true here, so the entry would poll and re-update forever.
    `order.status != PARTIALLY_FILLED` must treat this as terminal and drop it."""
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
        "db_row_opened": True,
    }
    order = FakePendingOrder(filled_avg_price="3.50", status=OrderStatus.DONE_FOR_DAY)
    order.filled_qty = "1"
    fake_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: None)

    events = execution.check_pending_option_fills()

    assert events == []
    assert "order-opt-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 1
        assert execution._option_positions["AAPL250117C00200000"]["entry_premium"] == 3.50


def test_check_pending_option_fills_untracks_zero_fill_order_on_unusual_terminal_status(monkeypatch):
    """Regression (fix-loop round 2, finding 1's remaining half): a zero-fill order can land on a
    terminal status that is neither FILLED nor in the small _TERMINAL_NO_FILL set (CANCELED/
    EXPIRED/REJECTED) -- e.g. DONE_FOR_DAY, a real Alpaca status, plausible for options since they
    are DAY-only. Before the fix, `elif order.status in _TERMINAL_NO_FILL:` would never be true
    here, so the entry would poll forever and never emit the "terminal" event. The new
    `elif order.status not in _OPTION_NON_TERMINAL:` must treat this as terminal and drop it."""
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
        "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.DONE_FOR_DAY)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
            "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
            "kind": "terminal", "order_id": "order-opt-1", "order_status": "done_for_day",
        }
    ]
    assert "order-opt-1" not in execution._pending_option_fills


def test_check_pending_option_fills_untracks_zero_fill_order_on_suspended_status(monkeypatch):
    """Same regression as above, second status (SUSPENDED) to cover more than one unusual
    zero-fill terminal case per the fix-loop round 2 brief."""
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
        "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.SUSPENDED)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
            "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
            "kind": "terminal", "order_id": "order-opt-1", "order_status": "suspended",
        }
    ]
    assert "order-opt-1" not in execution._pending_option_fills


def test_check_pending_option_fills_keeps_tracking_pending_cancel_with_partial_fill(monkeypatch):
    """Free side-effect fix confirmed (fix-loop round 1's new issue B): PENDING_CANCEL with a
    partial fill must NOT be treated as terminal -- a further fill could still be observed before
    the cancel actually resolves. _OPTION_NON_TERMINAL includes PENDING_CANCEL, so
    `is_terminal = order.status not in _OPTION_NON_TERMINAL` stays False here."""
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
        "db_row_opened": True,
    }
    order = FakePendingOrder(filled_avg_price="3.50", status=OrderStatus.PENDING_CANCEL)
    order.filled_qty = "1"
    fake_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: None)

    events = execution.check_pending_option_fills()

    assert events == []
    assert "order-opt-1" in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 1
        assert execution._option_positions["AAPL250117C00200000"]["entry_premium"] == 3.50


class RacyGetOrderTradingClient:
    """Stands in for trading_client to simulate the fix-loop round 2 (new issue A) race: a
    concurrent sell_option() drops _pending_option_fills/_option_positions tracking for the
    contract in the window between get_order_by_id() returning and check_pending_option_fills()
    acquiring _state_lock. The drop happens as a side effect of get_order_by_id() itself, which is
    the cleanest way to make the interleaving deterministic in a single-threaded test."""

    def __init__(self, order_id, order, contract_symbol):
        self._order_id = order_id
        self._order = order
        self._contract_symbol = contract_symbol

    def get_order_by_id(self, order_id, filter=None):
        execution._pending_option_fills.pop(order_id, None)
        execution._option_positions.pop(self._contract_symbol, None)
        return self._order


def test_check_pending_option_fills_does_not_resurrect_state_dropped_by_concurrent_sell(monkeypatch):
    """Regression (fix-loop round 2, new issue A): if sell_option() drops both
    _pending_option_fills and _option_positions for a contract in the window between the
    get_order_by_id() network call and check_pending_option_fills() acquiring _state_lock, this
    poll must skip the order entirely -- no _option_positions resurrection, no DB write, no
    event."""
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": 0.45,
        "qty": 2,
        "reasoning": "test reasoning",
        "cycle_id": "cycle-1",
    }
    order = FakePendingOrder(filled_avg_price="3.35", status=OrderStatus.FILLED)
    fake_client = RacyGetOrderTradingClient("order-opt-1", order, "AAPL250117C00200000")
    monkeypatch.setattr(execution, "trading_client", fake_client)

    def _fail_if_called(*a, **k):
        raise AssertionError("db.record_options_trade_opened must not be called when the entry was concurrently dropped")

    monkeypatch.setattr(execution.db, "record_options_trade_opened", _fail_if_called)
    monkeypatch.setattr(execution.db, "record_options_trade_updated", _fail_if_called)

    events = execution.check_pending_option_fills()

    assert events == []
    assert "order-opt-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_check_pending_option_fills_closes_position_and_db_on_confirmed_sell_fill(monkeypatch):
    """Regression (external review finding 1, 2026-08-26): a pending SELL entry (registered by
    sell_option(), which no longer clears state synchronously) is only resolved -- position dropped,
    DB row closed -- once check_pending_option_fills() observes the SELL's own confirmed fill."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
    execution._pending_option_fills["order-opt-sell-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "action": "SELL",
        "reason": "take_profit",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-sell-1": FakePendingOrder(filled_avg_price="4.50")})
    monkeypatch.setattr(execution, "trading_client", fake_client)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_closed", lambda *a, **k: recorded_db.setdefault("closed", (a, k)))

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000",
            "symbol": "AAPL",
            "action": "SELL",
            "reason": "take_profit",
            "kind": "fill",
            "order_id": "order-opt-sell-1",
            "fill_price": 4.5,
        }
    ]
    assert recorded_db["closed"][0] == ("AAPL250117C00200000", "take_profit", 4.5)
    assert "order-opt-sell-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_check_pending_option_fills_leaves_position_and_pending_sell_tracked_on_partial_sell_fill(monkeypatch):
    """Regression (external review finding 2, 2026-08-26 round 2): the first version of this fix
    closed _option_positions/the DB row on ANY observed SELL fill, including a PARTIALLY_FILLED order
    that's still open on Alpaca and could fill (or be canceled with qty still open) further. A
    still-non-terminal partial fill must leave _option_positions, the pending SELL entry, and the DB
    untouched -- no event either -- until the order is actually done."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
    execution._pending_option_fills["order-opt-sell-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "action": "SELL",
        "reason": "take_profit",
    }
    fake_client = FakePendingFillTradingClient(
        {"order-opt-sell-1": FakePendingOrder(filled_avg_price="4.50", status=OrderStatus.PARTIALLY_FILLED)}
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    def _fail_if_called(*a, **k):
        raise AssertionError("db.record_options_trade_closed must not be called on a still-open partial SELL fill")

    monkeypatch.setattr(execution.db, "record_options_trade_closed", _fail_if_called)

    events = execution.check_pending_option_fills()

    assert events == []
    assert execution._pending_option_fills["order-opt-sell-1"]["action"] == "SELL"
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 2


def test_check_pending_option_fills_leaves_position_tracked_on_sell_terminal_no_fill(monkeypatch):
    """Regression (external review finding 1, 2026-08-26): if the SELL order is canceled/rejected
    without ever filling, the position must stay tracked (protection stays live) -- only the pending
    SELL entry itself is dropped."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
    execution._pending_option_fills["order-opt-sell-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "action": "SELL",
        "reason": "stop_loss",
    }
    fake_client = FakePendingFillTradingClient(
        {"order-opt-sell-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.REJECTED)}
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    def _fail_if_called(*a, **k):
        raise AssertionError("db.record_options_trade_closed must not be called on a zero-fill terminal SELL")

    monkeypatch.setattr(execution.db, "record_options_trade_closed", _fail_if_called)

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000",
            "symbol": "AAPL",
            "action": "SELL",
            "reason": "stop_loss",
            "kind": "terminal",
            "order_id": "order-opt-sell-1",
            "order_status": "rejected",
        }
    ]
    assert "order-opt-sell-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert "AAPL250117C00200000" in execution._option_positions


def test_check_pending_option_fills_leaves_remaining_qty_tracked_on_terminal_partial_sell_fill(monkeypatch):
    """Regression (external review finding 1, 2026-08-26 round 3): a partially-filled SELL that
    reaches a terminal status (e.g. canceled after only 1 of 2 tracked contracts sold) still has a
    non-null filled_avg_price, so it must not be treated as a full close -- the unsold remainder is
    still open at Alpaca and must stay tracked (and the DB row must stay open), not get dropped
    alongside the fully-filled portion."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
    execution._pending_option_fills["order-opt-sell-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "action": "SELL",
        "reason": "take_profit",
    }
    fake_order = FakePendingOrder(filled_avg_price="4.50", status=OrderStatus.CANCELED)
    fake_order.filled_qty = "1"
    fake_client = FakePendingFillTradingClient({"order-opt-sell-1": fake_order})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    def _fail_if_called(*a, **k):
        raise AssertionError("db.record_options_trade_closed must not be called on a terminal partial SELL fill")

    monkeypatch.setattr(execution.db, "record_options_trade_closed", _fail_if_called)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: recorded_db.setdefault("updated", a))

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000",
            "symbol": "AAPL",
            "action": "SELL",
            "reason": "take_profit",
            "kind": "fill",
            "order_id": "order-opt-sell-1",
            "fill_price": 4.5,
            "qty": 1.0,
        }
    ]
    assert recorded_db["updated"] == ("AAPL250117C00200000", 3.20, 1)
    assert "order-opt-sell-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions["AAPL250117C00200000"]["qty"] == 1


def test_check_pending_option_fills_keeps_tracking_an_unfilled_order(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
        "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.NEW)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_option_fills()

    assert events == []
    assert "order-opt-1" in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions == {}


def test_check_pending_option_fills_untracks_order_on_terminal_no_fill_status(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
        "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-1": FakePendingOrder(filled_avg_price=None, status=OrderStatus.CANCELED)})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_option_fills()

    assert events == [
        {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
            "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
            "kind": "terminal", "order_id": "order-opt-1", "order_status": "canceled",
        }
    ]
    assert "order-opt-1" not in execution._pending_option_fills
    with execution._state_lock:
        assert execution._option_positions == {}


def test_check_pending_option_fills_untracks_order_on_confirmed_not_found(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
        "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient(
        {"order-opt-1": _api_error({"code": execution.ORDER_NOT_FOUND_CODE, "message": "order not found"})}
    )
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_option_fills()

    assert events == []
    assert "order-opt-1" not in execution._pending_option_fills


def test_check_pending_option_fills_keeps_tracking_order_on_transient_api_error(monkeypatch):
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "right": "call", "strike": 200.0,
        "expiration": "2025-01-17", "delta": 0.45, "qty": 2, "reasoning": "r", "cycle_id": "cycle-1",
    }
    fake_client = FakePendingFillTradingClient({"order-opt-1": _api_error({"code": 50000000, "message": "internal server error"})})
    monkeypatch.setattr(execution, "trading_client", fake_client)

    events = execution.check_pending_option_fills()

    assert events == []
    assert execution._pending_option_fills["order-opt-1"]["poll_failures"] == 1

    events = execution.check_pending_option_fills()

    assert events == []
    assert execution._pending_option_fills["order-opt-1"]["poll_failures"] == 2


class FakeOrder:
    def __init__(self, id, symbol, side, status, filled_avg_price=None, legs=None, asset_class=AssetClass.US_EQUITY):
        self.id = id
        self.symbol = symbol
        self.side = side
        self.status = status
        self.filled_avg_price = filled_avg_price
        self.legs = legs
        self.asset_class = asset_class


class FakeReconstructTradingClient:
    """Single live account: stocks, crypto and options all resolve through this one client.
    reconcile_tracked_state_once() calls get_orders(nested=True) for the stock/crypto rebuild,
    get_all_positions() once for everything, and get_orders() (no nest) for the option-order
    rebuild -- so the fake branches get_orders on the request's `nested` flag and unions the
    position lists."""

    def __init__(
        self, open_orders, positions=None, option_positions=(), option_orders=(), option_orders_error=None
    ):
        self._open_orders = open_orders
        self._positions = positions or []
        self._option_positions = list(option_positions)
        self._option_orders = list(option_orders)
        self._option_orders_error = option_orders_error

    def get_orders(self, request):
        if getattr(request, "nested", None):
            return self._open_orders
        if self._option_orders_error is not None:
            raise self._option_orders_error
        return self._option_orders

    def get_all_positions(self):
        return [*self._positions, *self._option_positions]


def test_reconcile_tracked_state_once_restores_a_still_open_pending_order(monkeypatch):
    order = FakeOrder("order-1", "BTC/USD", OrderSide.BUY, OrderStatus.NEW, filled_avg_price=None, legs=None)
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([order]))
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    assert execution._pending_fills["order-1"] == {
        "symbol": "BTC/USD",
        "action": "BUY",
        "reason": "reconstructed_after_restart",
        "sl_price": None,
        "tp_price": None,
    }
    assert execution.is_state_reconciled() is True


def test_reconcile_tracked_state_once_restores_a_bracket_with_a_still_open_leg(monkeypatch):
    legs = [FakeLeg("tp-leg-1", OrderStatus.NEW, OrderType.LIMIT), FakeLeg("sl-leg-1", OrderStatus.NEW, OrderType.STOP)]
    order = FakeOrder("parent-1", "MGN", OrderSide.BUY, OrderStatus.FILLED, filled_avg_price="10.05", legs=legs)
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([order]))
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    assert execution._tracked_brackets["MGN"] == "parent-1"
    assert "parent-1" not in execution._pending_fills


def test_reconcile_tracked_state_once_skips_a_bracket_whose_legs_are_all_terminal(monkeypatch):
    """Shouldn't happen given the status="open" family-level query, but must not crash or
    mistrack if it ever does."""
    legs = [FakeLeg("tp-leg-1", OrderStatus.CANCELED, OrderType.LIMIT), FakeLeg("sl-leg-1", OrderStatus.CANCELED, OrderType.STOP)]
    order = FakeOrder("parent-1", "MGN", OrderSide.BUY, OrderStatus.FILLED, filled_avg_price="10.05", legs=legs)
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([order]))
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    assert "MGN" not in execution._tracked_brackets


def test_reconcile_tracked_state_once_backfills_position_opens_for_open_positions(monkeypatch):
    positions = [FakeEodPosition("MGN", AssetClass.US_EQUITY), FakeEodPosition("BTC/USD", AssetClass.CRYPTO)]
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], positions=positions))
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])
    recorded = []
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: recorded.append(symbol))

    assert execution.reconcile_tracked_state_once() is True

    assert recorded == ["MGN", "BTC/USD"]


def test_reconcile_tracked_state_once_backfill_skips_option_contracts(monkeypatch):
    """Options live on the same account now, so get_all_positions() returns option contracts too --
    they must NOT get db.record_position_opened(<OCC symbol>); they are tracked in _option_positions
    (rebuilt separately) keyed by OCC symbol, not in position_opens."""
    positions = [
        FakeEodPosition("MGN", AssetClass.US_EQUITY),
        FakeEodPosition("AAPL250117C00200000", AssetClass.US_OPTION),
    ]
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], positions=positions))
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])
    recorded = []
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: recorded.append(symbol))

    assert execution.reconcile_tracked_state_once() is True

    assert recorded == ["MGN"]


def test_reconcile_tracked_state_once_rebuilds_crypto_stop_from_live_alpaca_position_symbol(monkeypatch):
    positions = [FakeEodPosition("BTCUSD", AssetClass.CRYPTO, avg_entry_price="100.0")]
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], positions=positions))
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)

    assert execution.reconcile_tracked_state_once() is True

    assert execution._crypto_stops["BTC/USD"] == pytest.approx((98.0, 103.0))


def test_reconcile_tracked_state_once_does_not_rebuild_crypto_stop_for_pending_buy(monkeypatch):
    execution._pending_fills["order-1"] = {"symbol": "BTC/USD", "action": "BUY"}
    positions = [FakeEodPosition("BTCUSD", AssetClass.CRYPTO, avg_entry_price="100.0")]
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], positions=positions))
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)

    assert execution.reconcile_tracked_state_once() is True

    assert execution._crypto_stops == {}


def test_reconcile_tracked_state_once_returns_false_when_get_all_positions_fails(monkeypatch):
    """A failed position read is fatal: marking state reconciled after only a partial Alpaca read
    would leave existing option positions unrebuilt (check_option_stops() protects nothing), skip
    the crypto-stop rebuild and position_opens backfill, re-enable BUYs via _buy_preflight_skip(),
    and stop poll_reconciliation() from retrying (it loops only while unreconciled)."""

    class FailingPositionsClient(FakeReconstructTradingClient):
        def get_all_positions(self):
            raise APIError("unreachable")

    monkeypatch.setattr(execution, "trading_client", FailingPositionsClient([]))
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is False
    assert execution.is_state_reconciled() is False


def test_reconcile_tracked_state_once_rebuilds_option_position_from_matching_options_trades_row(monkeypatch):
    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        qty = "2"
        asset_class = AssetClass.US_OPTION

    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_positions=[FakeOptionPosition()]))
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    open_trades = [{
        "symbol": "AAPL", "contract_symbol": "AAPL250117C00200000", "right": "call",
        "strike": Decimal("200.0"), "expiration": date(2025, 1, 17), "delta": Decimal("0.45"),
        "entry_premium": Decimal("3.20"), "qty": 2,
    }]
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: open_trades)

    assert execution.reconcile_tracked_state_once() is True

    with execution._state_lock:
        restored = execution._option_positions["AAPL250117C00200000"]
    assert restored == {
        "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
        "delta": 0.45, "entry_premium": 3.20, "qty": 2,
    }
    assert isinstance(restored["strike"], float)
    assert isinstance(restored["entry_premium"], float)


def test_reconcile_tracked_state_once_skips_short_option_position(monkeypatch):
    """Regression: a short option position must never be reconstructed into _option_positions --
    tracking it as if long would eventually cause sell_option() to sell it again instead of closing
    it. Uses a matching options_trades row on purpose, to prove the side check runs before either
    reconstruction branch, not just the OCC-fallback one."""
    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        qty = "-2"
        side = PositionSide.SHORT
        asset_class = AssetClass.US_OPTION

    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_positions=[FakeOptionPosition()]))
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    open_trades = [{
        "symbol": "AAPL", "contract_symbol": "AAPL250117C00200000", "right": "call",
        "strike": Decimal("200.0"), "expiration": date(2025, 1, 17), "delta": Decimal("0.45"),
        "entry_premium": Decimal("3.20"), "qty": 2,
    }]
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: open_trades)

    assert execution.reconcile_tracked_state_once() is True

    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_reconcile_tracked_state_once_reconstructs_option_position_from_occ_symbol_when_no_matching_trades_row(monkeypatch):
    """Regression: a missing options_trades row (e.g. the DB write failed after the Alpaca order
    filled) must not leave a live position with zero protection -- fall back to parsing the
    OCC-standard contract symbol plus Alpaca's own avg_entry_price."""
    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        qty = "2"
        avg_entry_price = "3.20"
        asset_class = AssetClass.US_OPTION

    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_positions=[FakeOptionPosition()]))
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    with execution._state_lock:
        restored = execution._option_positions["AAPL250117C00200000"]
    assert restored == {
        "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
        "delta": None, "entry_premium": 3.20, "qty": 2,
    }


def test_reconcile_tracked_state_once_skips_option_position_when_occ_fallback_has_no_usable_avg_entry_price(monkeypatch):
    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        qty = "2"
        avg_entry_price = None
        asset_class = AssetClass.US_OPTION

    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_positions=[FakeOptionPosition()]))
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_reconcile_tracked_state_once_does_not_reconstruct_option_position_already_tracked(monkeypatch):
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 999.0, "expiration": "2025-01-17",
            "delta": 0.1, "entry_premium": 1.0, "qty": 1,
        }

    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        qty = "2"
        asset_class = AssetClass.US_OPTION

    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_positions=[FakeOptionPosition()]))
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    open_trades = [{
        "symbol": "AAPL", "contract_symbol": "AAPL250117C00200000", "right": "call",
        "strike": Decimal("200.0"), "expiration": date(2025, 1, 17), "delta": Decimal("0.45"),
        "entry_premium": Decimal("3.20"), "qty": 2,
    }]
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: open_trades)

    assert execution.reconcile_tracked_state_once() is True

    with execution._state_lock:
        # untouched -- the already-tracked entry must not be clobbered by a stale/duplicate row
        assert execution._option_positions["AAPL250117C00200000"]["strike"] == 999.0


class FakeOptionOrder:
    def __init__(self, id, symbol, qty, side=OrderSide.BUY):
        self.id = id
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.asset_class = AssetClass.US_OPTION


def test_reconcile_tracked_state_once_restores_a_pending_option_order_from_open_orders(monkeypatch):
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_orders=[FakeOptionOrder("order-opt-1", "AAPL250117C00200000", "2")]))

    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    assert execution._pending_option_fills["order-opt-1"] == {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "action": "BUY",
        "right": "call",
        "strike": 200.0,
        "expiration": "2025-01-17",
        "delta": None,
        "qty": 2,
        "reasoning": "reconstructed_after_restart",
        "cycle_id": None,
        "db_row_opened": False,
    }


def test_reconcile_tracked_state_once_restores_a_pending_option_sell_order(monkeypatch):
    """Regression (external review finding 1, 2026-08-26): a restart between sell_option()
    submitting a SELL and that SELL's fill must not permanently lose the fill, or options_trades
    stays open forever. Reconciliation now fetches open orders of both sides and reconstructs a
    pending SELL entry (action=SELL) for one still open on Alpaca, using _option_positions
    (reconstructed separately, above, from Alpaca's own still-open position) to recover the
    underlying symbol."""
    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        asset_class = AssetClass.US_OPTION
        qty = "1"
        avg_entry_price = "3.20"

    monkeypatch.setattr(
        execution,
        "trading_client",
        FakeReconstructTradingClient(
            [],
            option_positions=[FakeOptionPosition()],
            option_orders=[FakeOptionOrder("order-opt-sell-1", "AAPL250117C00200000", "1", side=OrderSide.SELL)],
        ),
    )

    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [
        {"symbol": "AAPL", "contract_symbol": "AAPL250117C00200000", "right": "call",
         "strike": Decimal("200.0"), "expiration": date(2025, 1, 17), "delta": Decimal("0.45"),
         "entry_premium": Decimal("3.20"), "qty": 1}
    ])

    assert execution.reconcile_tracked_state_once() is True

    assert execution._pending_option_fills["order-opt-sell-1"] == {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "action": "SELL",
        "reason": "reconstructed_after_restart",
    }


def test_reconcile_tracked_state_once_seeds_db_row_opened_for_contract_already_open_in_db(monkeypatch):
    """Regression (fix-loop round 1, finding 2): a reconstructed pending-fill entry for a contract
    that already has an open options_trades row (e.g. it had already partially filled and been
    DB-recorded before the restart, or a race with the poll thread) must be seeded with
    db_row_opened=True -- otherwise the eventual fill re-INSERTs a duplicate open row and re-fires
    the first-fill Slack notification. Also proves the downstream fill in
    check_pending_option_fills() honors that seed: UPDATE not INSERT, and no fill event/Slack
    notification fires."""
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_orders=[FakeOptionOrder("order-opt-1", "AAPL250117C00200000", "2")]))

    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [
        {"contract_symbol": "AAPL250117C00200000", "symbol": "AAPL"}
    ])

    assert execution.reconcile_tracked_state_once() is True

    assert execution._pending_option_fills["order-opt-1"]["db_row_opened"] is True

    # Now drive the reconstructed entry to its first observed fill and confirm it does an UPDATE,
    # not a duplicate INSERT, and does not emit a fill event (no duplicate Slack notification).
    order = FakePendingOrder(filled_avg_price="3.35", status=OrderStatus.PARTIALLY_FILLED)
    order.filled_qty = "1"
    fake_pending_client = FakePendingFillTradingClient({"order-opt-1": order})
    monkeypatch.setattr(execution, "trading_client", fake_pending_client)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_opened", lambda *a, **k: recorded_db.setdefault("opened", (a, k)))
    monkeypatch.setattr(execution.db, "record_options_trade_updated", lambda *a, **k: recorded_db.setdefault("updated", (a, k)))

    events = execution.check_pending_option_fills()

    assert events == []
    assert "opened" not in recorded_db
    assert recorded_db["updated"] == (("AAPL250117C00200000", 3.35, 1), {})


def test_reconcile_tracked_state_once_returns_false_when_open_options_trades_fetch_fails(monkeypatch):
    """The single options_trades read feeds both the _option_positions rebuild and the
    open_contract_symbols seed for reconstructed pending BUYs. A DB outage here is fatal for the
    same reason a failed position read is -- option-stop protection depends on _option_positions
    being fully rebuilt before state is marked reconciled."""
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_orders=[FakeOptionOrder("order-opt-1", "AAPL250117C00200000", "2")]))
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)

    def _raise():
        raise Exception("db unreachable")

    monkeypatch.setattr(db, "fetch_open_options_trades", _raise)

    assert execution.reconcile_tracked_state_once() is False
    assert execution.is_state_reconciled() is False
    assert "order-opt-1" not in execution._pending_option_fills


def test_reconcile_tracked_state_once_returns_false_when_option_positions_rebuild_raises(monkeypatch):
    """If _rebuild_option_positions_from_positions() itself raises (e.g. a deterministic bug or a
    malformed Alpaca Position), state must not be marked reconciled -- a half-rebuilt
    _option_positions leaves pre-restart option positions unprotected by check_option_stops()."""

    class FakeOptionPosition:
        symbol = "AAPL250117C00200000"
        qty = "2"
        asset_class = AssetClass.US_OPTION

    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_positions=[FakeOptionPosition()]))
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    def _boom(*a, **k):
        raise RuntimeError("rebuild bug")

    monkeypatch.setattr(execution, "_rebuild_option_positions_from_positions", _boom)

    assert execution.reconcile_tracked_state_once() is False
    assert execution.is_state_reconciled() is False


def test_reconcile_tracked_state_once_skips_pending_option_order_with_non_occ_symbol(monkeypatch):
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_orders=[FakeOptionOrder("order-opt-1", "NOT-AN-OCC-SYMBOL", "2")]))

    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    assert "order-opt-1" not in execution._pending_option_fills


def test_reconcile_tracked_state_once_does_not_overwrite_an_already_pending_option_order(monkeypatch):
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_orders=[FakeOptionOrder("order-opt-1", "AAPL250117C00200000", "2")]))
    execution._pending_option_fills["order-opt-1"] = {
        "contract_symbol": "AAPL250117C00200000",
        "symbol": "AAPL",
        "right": "call",
        "strike": 999.0,
        "expiration": "2099-01-17",
        "delta": 0.99,
        "qty": 99,
        "reasoning": "already tracked, should not be overwritten",
        "cycle_id": "cycle-existing",
    }

    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is True

    assert execution._pending_option_fills["order-opt-1"]["strike"] == 999.0
    assert execution._pending_option_fills["order-opt-1"]["reasoning"] == "already tracked, should not be overwritten"


def test_reconcile_tracked_state_once_returns_false_when_open_option_orders_fetch_fails(monkeypatch):
    """A failed open-option-orders read is fatal: a still-open option BUY order left out of
    _pending_option_fills can allow a duplicate submission after restart, and a missed SELL stays
    unobserved forever."""
    monkeypatch.setattr(execution, "trading_client", FakeReconstructTradingClient([], option_orders_error=APIError("unreachable")))
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(db, "record_position_opened", lambda symbol: None)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])

    assert execution.reconcile_tracked_state_once() is False
    assert execution.is_state_reconciled() is False



def test_reconcile_tracked_state_once_handles_api_error_without_raising(monkeypatch):
    class FailingClient:
        def get_orders(self, request):
            raise APIError("unreachable")

    monkeypatch.setattr(execution, "trading_client", FailingClient())
    monkeypatch.setattr(execution, "_state_reconciled", False)

    assert execution.reconcile_tracked_state_once() is False

    assert execution._pending_fills == {}
    assert execution._tracked_brackets == {}
    assert execution.is_state_reconciled() is False


class FakeFlakyReconstructTradingClient:
    """Raises APIError on the first `fail_times` calls to get_orders(), then returns
    `open_orders` -- lets tests exercise reconstruct_tracked_state()'s retry-with-backoff loop
    without a live Alpaca outage."""

    def __init__(self, open_orders, fail_times):
        self._open_orders = open_orders
        self._fail_times = fail_times
        self.calls = 0

    def get_orders(self, request):
        if not getattr(request, "nested", None):
            return []  # the option-order rebuild pass -- not what this test exercises
        self.calls += 1
        if self.calls <= self._fail_times:
            raise APIError("unreachable")
        return self._open_orders

    def get_all_positions(self):
        return []


def test_reconstruct_tracked_state_retries_with_backoff_until_success(monkeypatch):
    client = FakeFlakyReconstructTradingClient(open_orders=[], fail_times=2)
    monkeypatch.setattr(execution, "trading_client", client)
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(db, "fetch_open_options_trades", lambda: [])
    sleeps = []
    monkeypatch.setattr(execution.time, "sleep", lambda s: sleeps.append(s))

    execution.reconstruct_tracked_state(max_attempts=5, backoff_base_s=1)

    assert client.calls == 3
    assert sleeps == [1, 2]
    assert execution.is_state_reconciled() is True


def test_reconstruct_tracked_state_gives_up_after_max_attempts_and_stays_unreconciled(monkeypatch):
    client = FakeFlakyReconstructTradingClient(open_orders=[], fail_times=99)
    monkeypatch.setattr(execution, "trading_client", client)
    monkeypatch.setattr(execution, "_state_reconciled", False)
    monkeypatch.setattr(execution.time, "sleep", lambda s: None)

    execution.reconstruct_tracked_state(max_attempts=3, backoff_base_s=1)

    assert client.calls == 3
    assert execution.is_state_reconciled() is False


def test_buy_skips_when_state_not_reconciled(monkeypatch):
    # status="skipped", not a distinct "rejected" -- ExecuteResponse's status Literal
    # (src/floor_broker/app.py) doesn't permit "rejected"; see test_app.py for the
    # end-to-end regression test covering that API-contract constraint directly.
    monkeypatch.setattr(execution, "_state_reconciled", False)
    client = FakeTradingClient()
    monkeypatch.setattr(execution, "trading_client", client)

    result = execution.buy("MGN", "stocks", budget=100.0, slP=0.98, tpP=1.05)

    assert result == {
        "status": "skipped",
        "reason": "state_not_reconciled",
        "detail": "tracked state not yet reconciled with Alpaca after restart",
    }
    assert client.submitted == []


def test_buy_option_submits_order_and_defers_tracking_until_fill(monkeypatch):
    """buy_option() must not write _option_positions or call record_options_trade_opened itself --
    that's deferred to check_pending_option_fills() until a real fill is observed (Task 27)."""
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)
    monkeypatch.setattr(execution, "get_current_option_ask_price", lambda contract_symbol: 3.20)
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_opened", lambda *a, **k: recorded_db.setdefault("opened", (a, k)))

    class FakeOrder:
        id = "order-opt-1"

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return []

        def submit_order(self, req):
            recorded_db["req"] = req
            return FakeOrder()

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "submitted"
    assert result["order_id"] == "order-opt-1"
    assert "opened" not in recorded_db
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions
        assert execution._pending_option_fills["order-opt-1"] == {
            "contract_symbol": "AAPL250117C00200000",
            "symbol": "AAPL",
            "action": "BUY",
            "right": "call",
            "strike": 200.0,
            "expiration": "2025-01-17",
            "delta": 0.45,
            "qty": 2,
            "reasoning": "test reasoning",
            "cycle_id": "cycle-1",
        }


def test_buy_option_refuses_when_state_not_reconciled(monkeypatch):
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: False)

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "state_not_reconciled"


def test_buy_option_refuses_when_kill_switch_active(monkeypatch):
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)
    monkeypatch.setattr(execution.kill_switch, "buy_kill_switch_active", lambda: True)

    class FakeTradingClient2:
        pass  # kill switch trips before any client method is called

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "buy_kill_switch_active"


def test_buy_option_refuses_when_daily_loss_limit_reached(monkeypatch):
    """Option BUYs share the live account's daily-P&L halt with the stock/crypto path -- one
    account, one balance, one limit (no options carve-out)."""
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    class FakeLiveClient:
        def get_account(self):
            return FakeAccount(equity="99400.0", last_equity="100000.0")  # -$600, past -$500 limit in _FAKE_CFG

    monkeypatch.setattr(execution, "trading_client", FakeLiveClient())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "daily_loss_limit_reached"


def test_buy_option_refuses_when_max_concurrent_positions_reached(monkeypatch):
    """Option BUYs count against the same strategy.max_concurrent_positions cap as stocks/crypto --
    the open-position count is the live account's whole book."""
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    class FakeLiveClient:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return [object()] * 10  # _FAKE_CFG.strategy.max_concurrent_positions == 10

    monkeypatch.setattr(execution, "trading_client", FakeLiveClient())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "max_concurrent_positions_reached"


def test_buy_option_blocked_when_stock_heavy_book_fills_the_concurrent_positions_cap(monkeypatch):
    """Regression for the one-live-account unification: a book already at the cap with nothing but
    stock positions blocks a brand-new option BUY -- the cap is the whole account, not per asset
    class."""
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    class FakeStockPosition:
        asset_class = AssetClass.US_EQUITY

    class FakeLiveClient:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return [FakeStockPosition()] * 10  # _FAKE_CFG.strategy.max_concurrent_positions == 10

    monkeypatch.setattr(execution, "trading_client", FakeLiveClient())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "max_concurrent_positions_reached"


def test_buy_option_refuses_when_contract_already_held_at_alpaca(monkeypatch):
    """Options have no top-up concept: a contract already open at Alpaca must not get a second BUY --
    that over-positions the account and corrupts _option_positions (keyed by OCC, overwritten on
    each fill)."""
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    def _submit_must_not_be_called(req):
        raise AssertionError("must not submit a duplicate BUY for an already-held contract")

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            return object()  # contract already open

        def get_all_positions(self):
            return []

        submit_order = _submit_must_not_be_called

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "already_holding_contract"


def test_buy_option_refuses_when_contract_in_option_positions(monkeypatch):
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")  # Alpaca doesn't show it yet, but we already track it

        def get_all_positions(self):
            return []

        def submit_order(self, req):
            raise AssertionError("must not submit a duplicate BUY for a tracked contract")

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.10, "qty": 1,
        }

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "already_holding_contract"


def test_buy_option_refuses_when_a_buy_is_already_in_flight(monkeypatch):
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return []

        def submit_order(self, req):
            raise AssertionError("must not submit a second BUY while one is in flight")

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())
    execution._set_pending_option_fill("order-inflight-1", {
        "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "action": "BUY",
        "right": "call", "strike": 200.0, "expiration": "2025-01-17", "delta": 0.45, "qty": 1,
    })

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "option_buy_in_flight"


def test_option_exposure_contract_symbols_unions_held_and_pending_buys():
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {"symbol": "AAPL", "qty": 1}
    execution._set_pending_option_fill("o-buy", {"contract_symbol": "MSFT250117C00400000", "action": "BUY"})
    execution._set_pending_option_fill("o-sell", {"contract_symbol": "NVDA250117C00900000", "action": "SELL"})

    assert execution.option_exposure_contract_symbols() == [
        "AAPL250117C00200000",
        "MSFT250117C00400000",
    ]  # sorted; a pending SELL is not "exposure to acquire" so it's excluded


def test_buy_option_refuses_when_live_notional_exceeds_cap(monkeypatch):
    """The claimed entry_premium argument is deliberately NOT what's checked against the cap --
    a hallucinated low premium (which is what sized qty in the Dealer in the first place) must not
    be able to bypass this gate. Only the live re-quoted ask price is used here."""
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)
    monkeypatch.setattr(execution, "get_current_option_ask_price", lambda contract_symbol: 50.0)  # 10 * 50 * 100 = $50,000, above the $2,000 cap

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return []

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.buy_option(
        "AAPL250117C00200000", 10, 0.02, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "notional_cap_exceeded"
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_buy_option_refuses_when_live_ask_quote_is_not_executable(monkeypatch):
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)
    monkeypatch.setattr(execution, "get_current_option_ask_price", lambda contract_symbol: 0.0)

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return []

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_ask_quote"
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_buy_option_returns_error_when_requote_fails(monkeypatch):
    monkeypatch.setattr(execution, "is_state_reconciled", lambda: True)

    def _raise(contract_symbol):
        raise APIError("no quote available")

    monkeypatch.setattr(execution, "get_current_option_ask_price", _raise)

    class FakeTradingClient2:
        def get_account(self):
            return FakeAccount()

        def get_open_position(self, symbol):
            raise APIError("no position")

        def get_all_positions(self):
            return []

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.buy_option(
        "AAPL250117C00200000", 2, 3.20, "call", 200.0, "2025-01-17", 0.45, "test reasoning", "AAPL", "cycle-1"
    )

    assert result["status"] == "error"


def test_sell_option_submits_order_and_registers_pending_sell(monkeypatch):
    """Regression (external review finding 1, 2026-08-26): sell_option() must not clear tracking or
    close the DB row synchronously on submit -- a submitted SELL can still be canceled/rejected or
    never fill. State is only cleared once check_pending_option_fills() confirms the fill."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
    recorded_db = {}
    monkeypatch.setattr(execution.db, "record_options_trade_closed", lambda *a, **k: recorded_db.setdefault("closed", (a, k)))

    class FakePosition:
        qty = "2"
        current_price = "4.50"

    class FakeOrder:
        id = "order-opt-2"

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            return FakePosition()

        def submit_order(self, req):
            return FakeOrder()

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000", reason="take_profit")

    assert result["status"] == "submitted"
    assert result["order_id"] == "order-opt-2"
    assert "closed" not in recorded_db
    with execution._state_lock:
        assert "AAPL250117C00200000" in execution._option_positions
        pending = execution._pending_option_fills["order-opt-2"]
        assert pending["contract_symbol"] == "AAPL250117C00200000"
        assert pending["symbol"] == "AAPL"
        assert pending["action"] == "SELL"
        assert pending["reason"] == "take_profit"


def test_sell_option_skips_when_sell_already_pending(monkeypatch):
    """Regression (external review finding 1, 2026-08-26): check_option_stops()/flatten_all_options()
    re-evaluate every tracked position each poll cycle -- since a submitted SELL no longer drops
    _option_positions immediately, sell_option() must refuse to submit a second SELL for a contract
    that already has one in flight, or every poll would resubmit a duplicate order."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
        execution._pending_option_fills["order-opt-inflight"] = {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "action": "SELL", "reason": "stop_loss",
        }

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            raise AssertionError("must not re-fetch position when a sell is already pending")

        def submit_order(self, req):
            raise AssertionError("must not submit a duplicate sell order")

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000", reason="take_profit")

    assert result == {"status": "skipped", "detail": "sell already pending"}


def test_sell_option_refuses_to_sell_short_position_and_drops_tracking(monkeypatch):
    """Regression: sell_option() must never submit a SELL for a short position -- that would open
    MORE short instead of closing anything. Defense in depth: this must hold even if a short position
    somehow reaches sell_option() directly, bypassing the upstream flatten/reconcile filters."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    class FakePosition:
        qty = "-2"
        current_price = "4.50"

    submitted = []

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            return FakePosition()

        def submit_order(self, req):
            submitted.append(req)
            raise AssertionError("must not submit an order for a short position")

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000")

    assert result["status"] == "skipped"
    assert submitted == []
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_sell_option_drops_tracking_on_confirmed_position_not_found(monkeypatch):
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    class FakeHttpError:
        class response:
            status_code = 404

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            raise APIError('{"code": 40410000, "message": "position does not exist"}', http_error=FakeHttpError())

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000")

    assert result["status"] == "skipped"
    with execution._state_lock:
        assert "AAPL250117C00200000" not in execution._option_positions


def test_sell_option_drops_matching_pending_option_fill_on_successful_sell(monkeypatch):
    """Regression (fix-loop round 1, finding 3; updated for external review finding 2, 2026-08-26,
    which added cancellation of the stale BUY order itself): a partially-filled BUY order for a
    contract can stay in _pending_option_fills (finding 1's fix) after check_option_stops() sells
    the already-filled portion via sell_option(). Left tracked, the next
    check_pending_option_fills() poll would resurrect _option_positions for a contract that was
    just sold and whose DB row is now closed -- a phantom in-memory-only position. sell_option()
    must cancel and drop the matching _pending_option_fills entry (and only the matching one) in
    its successful-sell branch, leaving an unrelated contract's tracked entry untouched."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
        execution._pending_option_fills["order-opt-stale"] = {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "action": "BUY", "qty": 1,
        }
        execution._pending_option_fills["order-opt-other"] = {
            "contract_symbol": "MSFT250117C00300000", "symbol": "MSFT", "action": "BUY", "qty": 1,
        }
    monkeypatch.setattr(execution.db, "record_options_trade_closed", lambda *a, **k: None)

    class FakePosition:
        qty = "2"
        current_price = "4.50"

    class FakeOrder:
        id = "order-opt-2"

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            return FakePosition()

        def submit_order(self, req):
            return FakeOrder()

        def cancel_order_by_id(self, order_id):
            pass

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000", reason="take_profit")

    assert result["status"] == "submitted"
    with execution._state_lock:
        assert "order-opt-stale" not in execution._pending_option_fills
        assert "order-opt-other" in execution._pending_option_fills
        assert execution._pending_option_fills["order-opt-2"]["action"] == "SELL"


def test_sell_option_cancels_stale_pending_buy_order_before_registering_sell(monkeypatch):
    """Regression (external review finding 2, 2026-08-26): a partially-filled BUY order can still be
    non-terminal (tracked in _pending_option_fills with action=BUY) when check_option_stops() sells
    the already-filled portion. Dropping the BUY's in-memory tracking alone doesn't stop it from
    filling further on Alpaca's side -- sell_option() must also cancel the order itself."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 1,
        }
        execution._pending_option_fills["order-opt-buy-stale"] = {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "action": "BUY", "qty": 2,
        }
    monkeypatch.setattr(execution.db, "record_options_trade_closed", lambda *a, **k: None)

    class FakePosition:
        qty = "1"
        current_price = "4.50"

    class FakeOrder:
        id = "order-opt-sell-1"

    cancelled_order_ids = []

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            return FakePosition()

        def submit_order(self, req):
            return FakeOrder()

        def cancel_order_by_id(self, order_id):
            cancelled_order_ids.append(order_id)

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000", reason="take_profit")

    assert result["status"] == "submitted"
    assert cancelled_order_ids == ["order-opt-buy-stale"]
    with execution._state_lock:
        assert "order-opt-buy-stale" not in execution._pending_option_fills


def test_sell_option_sizes_sell_off_position_fetched_after_stale_buy_cancel_attempt(monkeypatch):
    """Regression (external review finding 1, 2026-08-26 round 2): the first version of this fix
    canceled the stale BUY only after sizing/submitting the SELL from an already-stale position
    snapshot -- if the cancel then failed because the BUY's remaining qty had just filled, that
    newly-filled remainder was left both unsold and untracked. Canceling the stale BUY BEFORE
    fetching the position (as this now does) fixes the exact failure the earlier test locked in:
    even when the cancel fails for exactly that reason, get_open_position() is called AFTER the
    cancel attempt and so already reflects the just-filled remainder, and the SELL is sized off
    that -- covering the whole position, not just the earlier partial fill.

    Also covers external review finding 2, 2026-08-26 round 3: since a failed cancel can't be
    reliably distinguished here between "already filled" (this test's stated scenario) and a
    transient API error, sell_option() now leaves the stale BUY tracked on ANY cancel failure
    rather than dropping it -- check_pending_option_fills() resolves it correctly either way on a
    later poll."""
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 1,
        }
        execution._pending_option_fills["order-opt-buy-stale"] = {
            "contract_symbol": "AAPL250117C00200000", "symbol": "AAPL", "action": "BUY", "qty": 2,
        }
    monkeypatch.setattr(execution.db, "record_options_trade_closed", lambda *a, **k: None)

    call_order = []
    submitted_requests = []

    class FakePosition:
        # The BUY's remaining qty landed by the time this is fetched -- the cancel below "loses the
        # race" for exactly that reason, so the true current qty is 2, not the earlier partial 1.
        qty = "2"
        current_price = "4.50"

    class FakeOrder:
        id = "order-opt-sell-1"

    class FakeTradingClient2:
        def cancel_order_by_id(self, order_id):
            call_order.append(("cancel", order_id))
            raise APIError("already filled")

        def get_open_position(self, contract_symbol):
            call_order.append(("get_open_position", contract_symbol))
            return FakePosition()

        def submit_order(self, req):
            submitted_requests.append(req)
            return FakeOrder()

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000", reason="take_profit")

    assert result["status"] == "submitted"
    assert call_order == [("cancel", "order-opt-buy-stale"), ("get_open_position", "AAPL250117C00200000")]
    assert submitted_requests[0].qty == 2
    with execution._state_lock:
        assert "order-opt-buy-stale" in execution._pending_option_fills
        assert execution._pending_option_fills["order-opt-sell-1"]["action"] == "SELL"


def test_sell_option_drops_matching_pending_option_fill_on_short_position(monkeypatch):
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
        execution._pending_option_fills["order-opt-stale"] = {"contract_symbol": "AAPL250117C00200000", "qty": 1}
        execution._pending_option_fills["order-opt-other"] = {"contract_symbol": "MSFT250117C00300000", "qty": 1}

    class FakePosition:
        qty = "-2"
        current_price = "4.50"

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            return FakePosition()

        def submit_order(self, req):
            raise AssertionError("must not submit an order for a short position")

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000")

    assert result["status"] == "skipped"
    with execution._state_lock:
        assert "order-opt-stale" not in execution._pending_option_fills
        assert "order-opt-other" in execution._pending_option_fills


def test_sell_option_drops_matching_pending_option_fill_on_position_not_found(monkeypatch):
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }
        execution._pending_option_fills["order-opt-stale"] = {"contract_symbol": "AAPL250117C00200000", "qty": 1}
        execution._pending_option_fills["order-opt-other"] = {"contract_symbol": "MSFT250117C00300000", "qty": 1}

    class FakeHttpError:
        class response:
            status_code = 404

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            raise APIError('{"code": 40410000, "message": "position does not exist"}', http_error=FakeHttpError())

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000")

    assert result["status"] == "skipped"
    with execution._state_lock:
        assert "order-opt-stale" not in execution._pending_option_fills
        assert "order-opt-other" in execution._pending_option_fills


def test_sell_option_preserves_tracking_on_transient_error_fetching_position(monkeypatch):
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            raise APIError('{"code": 50000000, "message": "internal server error"}')

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000")

    assert result["status"] == "error"
    with execution._state_lock:
        assert "AAPL250117C00200000" in execution._option_positions


def test_sell_option_preserves_tracking_when_submit_order_fails(monkeypatch):
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2025-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    class FakePosition:
        qty = "2"
        current_price = "4.50"

    class FakeTradingClient2:
        def get_open_position(self, contract_symbol):
            return FakePosition()

        def submit_order(self, req):
            raise APIError('{"code": 40310000, "message": "insufficient buying power"}')

    monkeypatch.setattr(execution, "trading_client", FakeTradingClient2())

    result = execution.sell_option("AAPL250117C00200000")

    assert result["status"] == "error"
    with execution._state_lock:
        assert "AAPL250117C00200000" in execution._option_positions


def test_check_option_stops_still_protects_tracked_positions_when_options_trading_disabled(monkeypatch):
    """Regression: options_trading.enabled=False must only block NEW entries, not strip protection
    from a position that's already open -- flipping the flag off as an emergency rollback must not
    leave an existing option position with zero stop-loss/take-profit coverage."""
    cfg = OmegaConf.create({"options_trading": {"enabled": False, "options_slP": 0.50, "options_tpP": 1.75, "dte_force_close": 3}})
    monkeypatch.setattr(execution, "load_config", lambda: cfg)
    monkeypatch.setattr(execution, "get_current_option_mid_price", lambda contract_symbol: 1.50)  # entry 3.20 * 0.50 = 1.60 -> 1.50 <= 1.60 triggers SL
    sell_calls = []
    monkeypatch.setattr(execution, "sell_option", lambda contract_symbol, reason: sell_calls.append((contract_symbol, reason)) or {"status": "submitted", "detail": "x", "order_id": "o1"})

    far_expiration = "2099-01-17"
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": far_expiration,
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    events = execution.check_option_stops()

    assert len(events) == 1
    assert events[0]["reason"] == "stop_loss"
    assert sell_calls == [("AAPL250117C00200000", "stop_loss")]


def test_check_option_stops_triggers_stop_loss(monkeypatch):
    cfg = OmegaConf.create({"options_trading": {"enabled": True, "options_slP": 0.50, "options_tpP": 1.75, "dte_force_close": 3}})
    monkeypatch.setattr(execution, "load_config", lambda: cfg)
    monkeypatch.setattr(execution, "get_current_option_mid_price", lambda contract_symbol: 1.50)  # entry 3.20 * 0.50 = 1.60 -> 1.50 <= 1.60 triggers SL
    sell_calls = []
    monkeypatch.setattr(execution, "sell_option", lambda contract_symbol, reason: sell_calls.append((contract_symbol, reason)) or {"status": "submitted", "detail": "x", "order_id": "o1"})

    far_expiration = "2099-01-17"
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": far_expiration,
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    events = execution.check_option_stops()

    assert len(events) == 1
    assert events[0]["reason"] == "stop_loss"
    assert sell_calls == [("AAPL250117C00200000", "stop_loss")]


def test_check_option_stops_skips_contract_with_malformed_expiration_without_poisoning_others(monkeypatch):
    cfg = OmegaConf.create({"options_trading": {"enabled": True, "options_slP": 0.50, "options_tpP": 1.75, "dte_force_close": 3}})
    monkeypatch.setattr(execution, "load_config", lambda: cfg)
    monkeypatch.setattr(execution, "get_current_option_mid_price", lambda contract_symbol: 1.50)
    sell_calls = []
    monkeypatch.setattr(execution, "sell_option", lambda contract_symbol, reason: sell_calls.append((contract_symbol, reason)) or {"status": "submitted", "detail": "x", "order_id": "o1"})

    with execution._state_lock:
        execution._option_positions["BAD250117C00200000"] = {
            "symbol": "BAD", "right": "call", "strike": 200.0, "expiration": "not-a-date",
            "delta": 0.45, "entry_premium": 3.20, "qty": 1,
        }
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2099-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    events = execution.check_option_stops()

    assert sell_calls == [("AAPL250117C00200000", "stop_loss")]
    assert len(events) == 1


def test_check_option_stops_skips_contract_when_quote_fetch_raises_keyerror_without_poisoning_others(monkeypatch):
    cfg = OmegaConf.create({"options_trading": {"enabled": True, "options_slP": 0.50, "options_tpP": 1.75, "dte_force_close": 3}})
    monkeypatch.setattr(execution, "load_config", lambda: cfg)

    def _mid(contract_symbol):
        if contract_symbol == "BAD250117C00200000":
            raise KeyError(contract_symbol)  # malformed Alpaca SDK quote response, not an APIError
        return 1.50

    monkeypatch.setattr(execution, "get_current_option_mid_price", _mid)
    sell_calls = []
    monkeypatch.setattr(execution, "sell_option", lambda contract_symbol, reason: sell_calls.append((contract_symbol, reason)) or {"status": "submitted", "detail": "x", "order_id": "o1"})

    with execution._state_lock:
        execution._option_positions["BAD250117C00200000"] = {
            "symbol": "BAD", "right": "call", "strike": 200.0, "expiration": "2099-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 1,
        }
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": "2099-01-17",
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    events = execution.check_option_stops()

    assert sell_calls == [("AAPL250117C00200000", "stop_loss")]
    assert len(events) == 1


def test_check_option_stops_force_closes_near_expiration(monkeypatch):
    cfg = OmegaConf.create({"options_trading": {"enabled": True, "options_slP": 0.50, "options_tpP": 1.75, "dte_force_close": 3}})
    monkeypatch.setattr(execution, "load_config", lambda: cfg)
    monkeypatch.setattr(execution, "get_current_option_mid_price", lambda contract_symbol: 3.20)  # flat P&L, would not otherwise trigger
    sell_calls = []
    monkeypatch.setattr(execution, "sell_option", lambda contract_symbol, reason: sell_calls.append((contract_symbol, reason)) or {"status": "submitted", "detail": "x", "order_id": "o1"})

    near_expiration = (datetime.now(pytz.timezone("US/Eastern")) + timedelta(days=1)).date().isoformat()
    with execution._state_lock:
        execution._option_positions["AAPL250117C00200000"] = {
            "symbol": "AAPL", "right": "call", "strike": 200.0, "expiration": near_expiration,
            "delta": 0.45, "entry_premium": 3.20, "qty": 2,
        }

    events = execution.check_option_stops()

    assert len(events) == 1
    assert events[0]["reason"] == "dte_force_close"
    assert sell_calls == [("AAPL250117C00200000", "dte_force_close")]
