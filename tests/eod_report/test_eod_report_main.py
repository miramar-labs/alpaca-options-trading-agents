from datetime import datetime, timedelta

import pytz

from src.common import eod
from src.eod_report import main


class FakeAccount:
    def __init__(self):
        self.equity = "1050.00"
        self.last_equity = "1000.00"
        self.cash = "500.00"
        self.buying_power = "2000.00"


class FakePosition:
    def __init__(
        self,
        symbol,
        qty,
        market_value,
        unrealized_plpc,
        avg_entry_price="0",
        unrealized_pl="0",
        current_price="0",
    ):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.unrealized_plpc = unrealized_plpc
        self.avg_entry_price = avg_entry_price
        self.unrealized_pl = unrealized_pl
        self.current_price = current_price


class FakeTradingClient:
    def __init__(self, activities=None):
        self._activities = activities or []

    def get_account(self):
        return FakeAccount()

    def get_all_positions(self):
        return [FakePosition("MGN", "3", "150.00", "0.05")]

    def get(self, path, data=None):
        return self._activities


class FakeResponse:
    def __init__(self, status_error=None):
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error


def _silence_slack(monkeypatch):
    calls = {}
    monkeypatch.setattr(main.slack, "notify_market_closed", lambda *a, **k: calls.setdefault("market_closed", (a, k)))
    monkeypatch.setattr(main.slack, "notify_eod_report", lambda *a, **k: calls.setdefault("eod_report", (a, k)))
    monkeypatch.setattr(main.slack, "notify_error", lambda *a, **k: calls.setdefault("error", (a, k)))
    return calls


def _today():
    return datetime.now(pytz.timezone("US/Eastern")).date()


def _close_at(hour: int, minute: int = 0):
    eastern = pytz.timezone("US/Eastern")
    return eastern.localize(datetime.combine(_today(), datetime.min.time()).replace(hour=hour, minute=minute))


def _set_now(monkeypatch, dt):
    monkeypatch.setattr(main, "_now_eastern", lambda: dt)


def _not_sent(monkeypatch):
    sent = []
    monkeypatch.setattr(main.db, "eod_report_already_sent", lambda report_date: False)
    monkeypatch.setattr(main.db, "record_eod_report_sent", lambda report_date: sent.append(report_date))
    return sent


def test_market_closed_posts_notification_and_skips_report(monkeypatch):
    """Regression: previously main() only logged locally on a closed market and returned -- a
    weekend/holiday CronJob run produced zero visible signal that anything happened at all."""
    monkeypatch.setattr(main, "get_stock_market_close", lambda day: None)
    sent = _not_sent(monkeypatch)
    calls = _silence_slack(monkeypatch)

    main.main()

    assert "market_closed" in calls
    assert "eod_report" not in calls
    assert sent == [_today()]


def test_open_market_sends_full_eod_report(monkeypatch):
    fake_client = FakeTradingClient(activities=[])
    monkeypatch.setattr(main, "trading_client", fake_client)
    monkeypatch.setattr(eod, "trading_client", fake_client)
    monkeypatch.setattr(main, "_trigger_pl_badges_workflow", lambda: None)
    close = _close_at(16)
    _set_now(monkeypatch, close + timedelta(minutes=30))
    monkeypatch.setattr(main, "get_stock_market_close", lambda day: close)
    sent = _not_sent(monkeypatch)
    calls = _silence_slack(monkeypatch)

    main.main()

    assert "eod_report" in calls
    assert "market_closed" not in calls
    assert sent == [_today()]
    args, _ = calls["eod_report"]
    _report_date, account_summary, fills, position_summaries = args
    assert account_summary["equity"] == 1050.0
    assert fills == []
    assert position_summaries == [
        {
            "symbol": "MGN",
            "qty": 3.0,
            "market_value": 150.0,
            "unrealized_plpc": 0.05,
            "unrealized_pl": 0.0,
            "avg_entry_price": 0.0,
            "current_price": 0.0,
        }
    ]


def test_open_market_skips_until_thirty_minutes_after_close(monkeypatch):
    close = _close_at(16)
    _set_now(monkeypatch, close + timedelta(minutes=29))
    monkeypatch.setattr(main, "get_stock_market_close", lambda day: close)
    sent = _not_sent(monkeypatch)
    calls = _silence_slack(monkeypatch)

    main.main()

    assert calls == {}
    assert sent == []


def test_eod_report_sends_at_early_close_plus_thirty(monkeypatch):
    fake_client = FakeTradingClient(activities=[])
    monkeypatch.setattr(main, "trading_client", fake_client)
    monkeypatch.setattr(eod, "trading_client", fake_client)
    monkeypatch.setattr(main, "_trigger_pl_badges_workflow", lambda: None)
    close = _close_at(13)
    _set_now(monkeypatch, close + timedelta(minutes=30))
    monkeypatch.setattr(main, "get_stock_market_close", lambda day: close)
    sent = _not_sent(monkeypatch)
    calls = _silence_slack(monkeypatch)

    main.main()

    assert "eod_report" in calls
    assert sent == [_today()]


def test_eod_report_only_sends_once_per_day(monkeypatch):
    monkeypatch.setattr(main.db, "eod_report_already_sent", lambda report_date: True)
    monkeypatch.setattr(
        main.db,
        "record_eod_report_sent",
        lambda report_date: (_ for _ in ()).throw(AssertionError("must not record twice")),
    )
    monkeypatch.setattr(
        main,
        "get_stock_market_close",
        lambda day: (_ for _ in ()).throw(AssertionError("must not fetch calendar after sent")),
    )
    calls = _silence_slack(monkeypatch)

    main.main()

    assert calls == {}


def test_alpaca_failure_notifies_error_and_reraises(monkeypatch):
    class FailingTradingClient(FakeTradingClient):
        def get_account(self):
            raise RuntimeError("alpaca unavailable")

    fake_client = FailingTradingClient()
    monkeypatch.setattr(main, "trading_client", fake_client)
    monkeypatch.setattr(
        main,
        "_trigger_pl_badges_workflow",
        lambda: (_ for _ in ()).throw(AssertionError("must not dispatch after failed EOD")),
    )
    close = _close_at(16)
    _set_now(monkeypatch, close + timedelta(minutes=30))
    monkeypatch.setattr(main, "get_stock_market_close", lambda day: close)
    _not_sent(monkeypatch)
    calls = _silence_slack(monkeypatch)

    try:
        main.main()
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert "error" in calls
    assert "eod_report" not in calls


def test_successful_eod_dispatches_pl_badges_workflow(monkeypatch):
    fake_client = FakeTradingClient(activities=[])
    monkeypatch.setattr(main, "trading_client", fake_client)
    monkeypatch.setattr(eod, "trading_client", fake_client)
    monkeypatch.setenv("GITHUB_WORKFLOW_TOKEN", "token-123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "miramar-labs-org/multi-agent-ai-trader")
    close = _close_at(16)
    _set_now(monkeypatch, close + timedelta(minutes=30))
    monkeypatch.setattr(main, "get_stock_market_close", lambda day: close)
    _not_sent(monkeypatch)
    _silence_slack(monkeypatch)
    posts = []
    monkeypatch.setattr(
        main.requests,
        "post",
        lambda *args, **kwargs: posts.append((args, kwargs)) or FakeResponse(),
    )

    main.main()

    assert len(posts) == 1
    args, kwargs = posts[0]
    assert args == (
        "https://api.github.com/repos/miramar-labs-org/multi-agent-ai-trader/actions/workflows/pl-badges.yaml/dispatches",
    )
    assert kwargs["headers"]["Authorization"] == "Bearer token-123"
    assert kwargs["json"] == {"ref": "main"}
    assert kwargs["timeout"] == 15


def test_pl_badges_workflow_dispatch_skips_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_WORKFLOW_TOKEN", raising=False)
    monkeypatch.setattr(
        main.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call GitHub without token")),
    )

    main._trigger_pl_badges_workflow()
