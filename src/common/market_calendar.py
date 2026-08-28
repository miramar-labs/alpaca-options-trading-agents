from datetime import date, datetime, time

from alpaca.trading.requests import GetCalendarRequest
import pytz

from src.common.alpaca_client import trading_client


def _to_eastern(day: date, value: datetime | time | str) -> datetime:
    eastern = pytz.timezone("US/Eastern")
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, time):
        dt = datetime.combine(day, value)
    else:
        dt = datetime.combine(day, time.fromisoformat(str(value)))

    if dt.tzinfo is None:
        return eastern.localize(dt)
    return dt.astimezone(eastern)


def is_stock_market_open(day: date) -> bool:
    return bool(trading_client.get_calendar(GetCalendarRequest(start=day, end=day)))


def get_stock_market_close(day: date) -> datetime | None:
    calendar = trading_client.get_calendar(GetCalendarRequest(start=day, end=day))
    if not calendar:
        return None
    return _to_eastern(day, calendar[0].close)


def get_stock_market_hours(day: date) -> tuple[datetime, datetime] | None:
    """Today's own (open, close) as localized US/Eastern datetimes, or None if `day` isn't a
    trading day (weekend/holiday). Used by power_scheduler to derive today's power-on window
    without needing a separate past/future calendar search."""
    calendar = trading_client.get_calendar(GetCalendarRequest(start=day, end=day))
    if not calendar:
        return None
    return _to_eastern(day, calendar[0].open), _to_eastern(day, calendar[0].close)
