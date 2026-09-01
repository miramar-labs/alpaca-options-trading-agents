import os
from datetime import datetime

import pytz
import requests

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.symbols import is_usd_crypto_symbol

log = get_logger("SLACK")

_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL2")

_ASSET_ICONS = {"stock": "🏛️", "crypto": "🪙", "option": "📜"}


def _asset_icon(symbol: str, asset_class: str | None) -> str:
    """Asset-class marker to prefix `symbol` with in a per-trade Slack line.

    `asset_class` is authoritative when given ("stock" | "crypto" | "option").
    When None -- the stock/crypto trade path, which never deals in options --
    stock-vs-crypto is inferred from the symbol. Inference falls back to "stock"
    for an unknown crypto base until symbols.py's first Alpaca refresh lands.
    """
    if asset_class is None:
        asset_class = "crypto" if is_usd_crypto_symbol(symbol) else "stock"
    return _ASSET_ICONS.get(asset_class, "")


def _timestamp() -> str:
    return datetime.now(pytz.timezone("US/Eastern")).strftime("%I:%M:%S %p %Z")


def _format_fill_time(iso_time: str) -> str:
    # Alpaca's activities API returns transaction_time as an ISO 8601 UTC string (e.g.
    # "2026-08-03T14:32:01.123456Z") -- fromisoformat needs "+00:00", not a bare "Z".
    dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    return dt.astimezone(pytz.timezone("US/Eastern")).strftime("%I:%M %p %Z")


def _escape_freetext(text: str) -> str:
    """LLM-authored reasoning/rationale/narrative routinely writes "~47" or "~$150" for
    "approximately" -- Slack's mrkdwn treats a *pair* of tildes as strikethrough delimiters, so
    any message with two or more of them (common: two indicator values in one paragraph) silently
    strikes through everything in between. Swap the character for the actual approx sign, which
    reads at least as correctly and carries no mrkdwn meaning to Slack."""
    return text.replace("~", "≈")


def _post(text: str) -> None:
    """Fire-and-forget POST to the Slack incoming webhook. Never raises -- a Slack
    outage must never affect a trading decision."""
    if not load_config().slack.enabled or not _WEBHOOK_URL:
        return
    text = _escape_freetext(text)
    try:
        resp = requests.post(_WEBHOOK_URL, json={"text": text}, timeout=10)
        if resp.status_code != 200:
            log(f"⚠️ non-200 from Slack webhook: {resp.status_code} {resp.text}")
    except requests.RequestException as exc:
        log(f"⚠️ Slack post failed: {exc}")


def notify_morning_report(
    report_date: str,
    account: dict,
    symbols: list[dict],
    *,
    stock_market_open: bool = True,
    crypto_enabled: bool = False,
    title: str = "Morning Market Report",
    emoji: str = "🌅",
) -> None:
    lines = [f"{emoji} *{title} — {report_date}*"]

    if not stock_market_open:
        note = "🔒 Stock market is closed today"
        if crypto_enabled:
            note += " — crypto trading continues 24/7"
        lines.append(note)

    lines.append(
        f"Equity ${account['equity']:,.2f} | Cash ${account['cash']:,.2f}"
        f" | Buying Power ${account['buying_power']:,.2f}"
    )

    if symbols:
        lines.append(f"\n*Today's picks ({len(symbols)}):*")
        lines += [f"• *{s['symbol']}* (${s['budget']:.0f}) — {s['rationale']}" for s in symbols]
    else:
        lines.append("\n*Today's picks:* none")

    _post("\n".join(lines))


def notify_dealer_signal(symbol: str, action: str, reasoning: str, *, asset_class: str | None = None) -> None:
    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "")
    icon = _asset_icon(symbol, asset_class)
    prefix = f"{icon} " if icon else ""
    _post(f"{emoji} *Dealer* — *{action}* {prefix}{symbol} _{_timestamp()}_\n> {reasoning}")


def notify_floor_broker_result(
    symbol: str,
    action: str,
    status: str,
    detail: str,
    *,
    asset_class: str | None = None,
    reason: str | None = None,
    fill_price: float | None = None,
    sl_price: float | None = None,
    tp_price: float | None = None,
) -> None:
    emoji = {"executed": "✅", "submitted": "📨", "skipped": "⚠️"}.get(status, "❌")
    icon = _asset_icon(symbol, asset_class)
    prefix = f"{icon} " if icon else ""
    lines = [f"{emoji} *Floor Broker* — {action} {prefix}{symbol}: `{status}` — {detail} _{_timestamp()}_"]

    details = []
    if reason is not None:
        details.append(f"reason: {reason}")
    if fill_price is not None:
        details.append(f"fill: ${fill_price:,.2f}")
    if sl_price is not None:
        details.append(f"SL: ${sl_price:,.2f}")
    if tp_price is not None:
        details.append(f"TP: ${tp_price:,.2f}")
    if details:
        lines.append("> " + " | ".join(details))

    _post("\n".join(lines))


def notify_buy_kill_switch(active: bool) -> None:
    if active:
        _post(f"🛑 *Floor Broker* — BUY kill switch ACTIVATED. New BUY orders blocked; SELL still allowed. _{_timestamp()}_")
    else:
        _post(f"✅ *Floor Broker* — BUY kill switch DEACTIVATED. BUY orders resumed. _{_timestamp()}_")


def notify_power_state(action: str, detail: str) -> None:
    if action == "powered_down":
        _post(f"🌙 *Power Scheduler* — dealer/floor-broker scaled to 0. {detail} _{_timestamp()}_")
    else:
        _post(f"☀️ *Power Scheduler* — dealer/floor-broker scaled to 1. {detail} _{_timestamp()}_")


def notify_error(component: str, text: str) -> None:
    _post(f"🚨 *ERROR [{component}]* {text} _{_timestamp()}_")


def notify_market_closed(component: str, report_date: str) -> None:
    _post(f"📅 *{component}* — {report_date} was not a trading day, no report to send. _{_timestamp()}_")


def notify_stock_market_closed(next_open: str) -> None:
    # No trailing _timestamp() here -- next_open is already a labeled clock time, and appending
    # a second one reads as two competing AM/PM times with no indication of which is which.
    _post(f"🔒 *Dealer* — stock market is closed. Next open: {next_open}")


def notify_eod_report(report_date: str, account: dict, fills: list[dict], positions: list[dict]) -> None:
    pl = account["equity"] - account["last_equity"]
    pl_pct = (pl / account["last_equity"] * 100) if account["last_equity"] else 0.0
    pl_emoji = "🟢" if pl >= 0 else "🔴"

    lines = [
        f"📊 *End of Day Report — {report_date}*",
        f"{pl_emoji} Equity ${account['equity']:,.2f} ({pl:+,.2f} / {pl_pct:+.2f}% today)"
        f" | Cash ${account['cash']:,.2f} | Buying Power ${account['buying_power']:,.2f}",
    ]

    if fills:
        lines.append(f"\n*Trades today ({len(fills)}):*")
        lines += [
            f"• {f['side'].upper()} {f['qty']:g} {f['symbol']} @ ${f['price']:,.2f} ({_format_fill_time(f['time'])})"
            for f in fills
        ]
    else:
        lines.append("\n*Trades today:* none")

    if positions:
        lines.append(f"\n*Open positions ({len(positions)}):*")
        lines += [
            f"• {p['symbol']}: {p['qty']:g} shares, ${p['market_value']:,.2f} ({p['unrealized_plpc'] * 100:+.2f}%)"
            for p in positions
        ]
    else:
        lines.append("\n*Open positions:* none")

    _post("\n".join(lines))


def notify_crypto_eod_report(report_date: str, fills: list[dict], positions: list[dict]) -> None:
    """Crypto trades 24/7 -- there's no market close to report against, so this rides along with
    the Analyst's morning report instead of the stock EOD's after-close CronJob, covering the
    prior full calendar day."""
    lines = [f"🪙 *Crypto EOD Report — {report_date}* (24/7 market, no close)"]

    if fills:
        lines.append(f"\n*Crypto trades ({len(fills)}):*")
        lines += [
            f"• {f['side'].upper()} {f['qty']:g} {f['symbol']} @ ${f['price']:,.2f} ({_format_fill_time(f['time'])})"
            for f in fills
        ]
    else:
        lines.append("\n*Crypto trades:* none")

    if positions:
        total_value = sum(p["market_value"] for p in positions)
        lines.append(f"\n*Crypto positions ({len(positions)}, ${total_value:,.2f} total):*")
        lines += [
            f"• {p['symbol']}: {p['qty']:g}, ${p['market_value']:,.2f} ({p['unrealized_plpc'] * 100:+.2f}%)"
            for p in positions
        ]
    else:
        lines.append("\n*Crypto positions:* none")

    _post("\n".join(lines))


def notify_analyst_explain(narrative: str, report_date: str) -> None:
    """Posts the /analyst-explain skill's synthesized narrative -- unlike every other notify_*
    function, this isn't called from any pipeline; the skill only calls it when the user
    explicitly asks to share the explanation, not on every invocation."""
    _post(f"📊 *Analyst Explain — {report_date}* _{_timestamp()}_\n{narrative}")
