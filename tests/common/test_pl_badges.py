from datetime import date

from src.common import pl_badges


class FakeAccount:
    def __init__(self, equity, last_equity):
        self.equity = equity
        self.last_equity = last_equity


class FakeHistory:
    def __init__(self, base_value):
        self.base_value = base_value


class FakeTradingClient:
    def __init__(self, equity, last_equity, base_value):
        self._account = FakeAccount(equity, last_equity)
        self._history = FakeHistory(base_value)

    def get_account(self):
        return self._account

    def get_portfolio_history(self, request):
        return self._history


def test_fetch_pl_summary_computes_today_and_ytd_pl(monkeypatch):
    fake_client = FakeTradingClient(equity="1050.00", last_equity="1000.00", base_value="900.00")
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)

    summary = pl_badges.fetch_pl_summary(date(2026, 8, 13))

    assert summary == {"equity": 1050.0, "today_pl": 50.0, "ytd_pl": 150.0}


def test_fetch_pl_summary_handles_negative_pl(monkeypatch):
    fake_client = FakeTradingClient(equity="900.00", last_equity="1000.00", base_value="1200.00")
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)

    summary = pl_badges.fetch_pl_summary(date(2026, 8, 13))

    assert summary == {"equity": 900.0, "today_pl": -100.0, "ytd_pl": -300.0}


def test_fetch_pl_summary_falls_back_to_today_pl_when_base_value_is_none_and_no_history(monkeypatch):
    fake_client = FakeTradingClient(equity="999166.40", last_equity="1000000.00", base_value=None)
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)

    summary = pl_badges.fetch_pl_summary(date(2026, 8, 13))

    assert summary == {"equity": 999166.4, "today_pl": -833.6, "ytd_pl": -833.6}


def test_fetch_pl_summary_accumulates_ytd_from_persisted_history_when_base_value_is_none(monkeypatch):
    fake_client = FakeTradingClient(equity="999166.40", last_equity="1000000.00", base_value=None)
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)

    summary = pl_badges.fetch_pl_summary(date(2026, 8, 13), history_pl={"2026-08-05": -100.0, "2026-08-06": 50.0})

    assert summary["today_pl"] == -833.6
    assert summary["ytd_pl"] == -883.6


def test_fetch_pl_summary_excludes_todays_own_persisted_entry_from_the_ytd_sum(monkeypatch):
    """A second same-day run (e.g. EOD Report's backup dispatch) must not double-count today's
    own P&L using a stale value already persisted by an earlier run today."""
    fake_client = FakeTradingClient(equity="999166.40", last_equity="1000000.00", base_value=None)
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)
    today = date(2026, 8, 13)

    summary = pl_badges.fetch_pl_summary(today, history_pl={today.isoformat(): -999.0, "2026-01-02": 10.0})

    assert summary["today_pl"] == -833.6
    assert summary["ytd_pl"] == -823.6


def test_fetch_pl_summary_ignores_persisted_entries_from_a_prior_year(monkeypatch):
    fake_client = FakeTradingClient(equity="999166.40", last_equity="1000000.00", base_value=None)
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)

    summary = pl_badges.fetch_pl_summary(date(2026, 8, 13), history_pl={"2025-12-31": 5000.0})

    assert summary["today_pl"] == -833.6
    assert summary["ytd_pl"] == -833.6


def test_fetch_pl_summary_does_not_double_count_todays_entry_across_the_utc_midnight_rollover(monkeypatch):
    """Regression test for the production incident on 2026-08-13: a run at 20:25 ET (already
    00:25 UTC on 2026-08-14) previously computed `today_key` via a local `date.today()` call,
    which returned the UTC date (2026-08-14) instead of the Eastern trading date (2026-08-13)
    already used as the history key. That mismatch meant the "exclude today's own persisted
    entry" filter never matched, so today's P&L got summed in as a prior day *and* added again
    fresh -- YTD badge showed $239.12 instead of the correct $84.35. `fetch_pl_summary` now
    takes `today` as a caller-supplied Eastern date instead of resolving it locally, so this
    exclusion can no longer disagree with the history's own keys."""
    fake_client = FakeTradingClient(equity="1084.35", last_equity="929.58", base_value=None)
    monkeypatch.setattr(pl_badges, "trading_client", fake_client)
    today = date(2026, 8, 13)

    summary = pl_badges.fetch_pl_summary(
        today,
        history_pl={"2026-08-11": -27.37, "2026-08-12": -43.05, today.isoformat(): 154.77},
    )

    assert summary["today_pl"] == 154.77
    assert summary["ytd_pl"] == 84.35


def test_build_badge_payload_formats_positive_value_as_brightgreen():
    payload = pl_badges.build_badge_payload("Today's P/L", 243.359)

    assert payload == {
        "schemaVersion": 1,
        "label": "Today's P/L",
        "message": "+$243.36",
        "color": "brightgreen",
    }


def test_build_badge_payload_formats_negative_value_as_red():
    payload = pl_badges.build_badge_payload("YTD P/L", -1234.5)

    assert payload == {
        "schemaVersion": 1,
        "label": "YTD P/L",
        "message": "-$1,234.50",
        "color": "red",
    }


def test_build_badge_payload_treats_zero_as_up():
    payload = pl_badges.build_badge_payload("Today's P/L", 0.0)

    assert payload["message"] == "+$0.00"
    assert payload["color"] == "brightgreen"
