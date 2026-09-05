from datetime import date, datetime, timedelta

import pytz
from alpaca.trading.requests import GetPortfolioHistoryRequest

from src.common.alpaca_client import trading_client

_EASTERN = pytz.timezone("US/Eastern")


def reconcile_settled_history(history_pl: dict[str, float], today: date, lookback_days: int = 10) -> dict[str, float]:
    """`account.equity` / `account.last_equity` can still reflect a stale intraday options mark
    for a while after the 4pm ET close -- observed drifting by $700-$1,300 on three separate
    trading days (1, 2, 3 Sep 2026; see commits 13fbf2e and bbbf0b5) before Alpaca's own ledger
    settled, with no reliable fixed delay after which it's always safe to read. Rather than chase
    that delay with cron timing, re-read Alpaca's own settled 1D portfolio-history bars for the
    trailing `lookback_days` on every run and overwrite any already-persisted entry that no
    longer matches -- self-healing by the next run regardless of when today's own (still
    provisional) number was captured. `today`'s own entry is never touched here since that day's
    bar isn't settled yet either -- only fetch_pl_summary's live equity/last_equity read is
    trusted for today, until it becomes "yesterday" on a later run.

    Portfolio-history 1D bars come back timestamped at UTC midnight; that's one calendar day past
    the actual (US/Eastern) trading date the bar covers whenever the venue is west of UTC, so the
    trading date is recovered by converting to US/Eastern rather than by reading the UTC date
    directly. `profit_loss[i]` is Alpaca's own day-over-day equity delta (equity[i] - equity[i-1]),
    matching exactly what `fetch_pl_summary`'s own today_pl computation measures."""
    start = today - timedelta(days=lookback_days)
    history = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(start=start, timeframe="1D"))
    timestamps = getattr(history, "timestamp", None) or []
    daily_pls = getattr(history, "profit_loss", None) or []

    reconciled = dict(history_pl)
    for ts, day_pl in zip(timestamps, daily_pls):
        if day_pl is None:
            continue
        day = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(_EASTERN).date()
        if day >= today:
            continue
        key = day.isoformat()
        settled_pl = round(day_pl, 2)
        if key not in reconciled or abs(reconciled[key] - settled_pl) >= 0.01:
            reconciled[key] = settled_pl
    return reconciled


def fetch_pl_summary(today: date, history_pl: dict[str, float] | None = None) -> dict:
    """Today's P&L is account.equity - account.last_equity -- the same day-boundary math
    execution.py's daily loss limit check already relies on. YTD P&L is equity - base_value from
    a portfolio-history request starting Jan 1 of the current year: confirmed against a live
    account that PortfolioHistory.profit_loss is a day-over-day delta series (profit_loss[i] ==
    equity[i] - equity[i-1]), not cumulative from base_value, so base_value (Alpaca's own equity
    snapshot as of the requested start date) is the only reliable YTD anchor -- when available.

    base_value comes back None for an account with no equity snapshot yet at the requested start
    date -- observed persistently on this paper account, not just as a one-day post-reset blip.
    In that case, fall back to summing `history_pl` (src/pl_badges/main.py's persisted
    {date_iso: today_pl} record of each prior day's own badge run) for the current year, plus
    today's own P&L -- this is what actually accumulates YTD when Alpaca's own anchor is
    unavailable, rather than repeatedly collapsing YTD down to just today's number. Today's own
    entry (if `history_pl` already has one from an earlier same-day run) is excluded from the sum
    so a second same-day dispatch can't double-count it.

    `today` must be the caller's US/Eastern trading-day date (see pl_badges/main.py's
    _now_eastern()), not computed locally via date.today() here -- doing that previously
    double-counted today's own persisted history entry whenever this ran after the system/UTC
    clock had already rolled to the next calendar day but it was still "today" in US/Eastern."""
    account = trading_client.get_account()
    equity = float(account.equity)
    today_pl = round(equity - float(account.last_equity), 2)

    year_start = date(today.year, 1, 1)
    history = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(start=year_start, timeframe="1D"))

    if history.base_value is not None:
        ytd_pl = round(equity - float(history.base_value), 2)
    else:
        today_key = today.isoformat()
        this_year = str(today.year)
        prior_days_total = sum(v for k, v in (history_pl or {}).items() if k != today_key and k.startswith(this_year))
        ytd_pl = round(prior_days_total + today_pl, 2)

    return {"equity": equity, "today_pl": today_pl, "ytd_pl": ytd_pl}


def _format_usd(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def build_badge_payload(label: str, value: float) -> dict:
    """Shields.io endpoint-badge schema (schemaVersion 1) -- https://shields.io/badges/endpoint-badge."""
    return {
        "schemaVersion": 1,
        "label": label,
        "message": _format_usd(value),
        "color": "brightgreen" if value >= 0 else "red",
    }
