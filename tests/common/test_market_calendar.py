from datetime import date, time

from src.common import market_calendar


class FakeCalendarEntry:
    def __init__(self, open_, close):
        self.open = open_
        self.close = close


def test_get_stock_market_hours_returns_localized_open_and_close(monkeypatch):
    entry = FakeCalendarEntry(time(9, 30), time(16, 0))
    monkeypatch.setattr(market_calendar.trading_client, "get_calendar", lambda request: [entry])

    hours = market_calendar.get_stock_market_hours(date(2026, 8, 10))

    assert hours is not None
    open_dt, close_dt = hours
    assert (open_dt.hour, open_dt.minute) == (9, 30)
    assert (close_dt.hour, close_dt.minute) == (16, 0)
    assert str(open_dt.tzinfo) == "US/Eastern"
    assert str(close_dt.tzinfo) == "US/Eastern"


def test_get_stock_market_hours_returns_none_on_a_non_trading_day(monkeypatch):
    monkeypatch.setattr(market_calendar.trading_client, "get_calendar", lambda request: [])

    assert market_calendar.get_stock_market_hours(date(2026, 8, 8)) is None


def test_get_stock_market_close_still_returns_just_the_close(monkeypatch):
    entry = FakeCalendarEntry(time(9, 30), time(13, 0))
    monkeypatch.setattr(market_calendar.trading_client, "get_calendar", lambda request: [entry])

    close_dt = market_calendar.get_stock_market_close(date(2026, 8, 10))

    assert (close_dt.hour, close_dt.minute) == (13, 0)
