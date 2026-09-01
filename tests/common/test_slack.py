import re
from types import SimpleNamespace

from src.common import slack

_TIMESTAMP_RE = r"\d{2}:\d{2}:\d{2} (AM|PM)"


class _FakeResponse:
    status_code = 200
    text = "ok"


def test_notify_morning_report_omits_closed_market_banner_when_market_is_open(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_morning_report("2026-08-04", _account(), [])

    assert "closed" not in posted["text"].lower()


def test_notify_morning_report_defaults_to_morning_title_and_sun_emoji(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_morning_report("2026-08-04", _account(), [])

    assert posted["text"].startswith("🌅 *Morning Market Report — 2026-08-04*")


def test_notify_morning_report_uses_overridden_title_and_emoji(monkeypatch):
    """Supports the Analyst's optional midday run, which posts a "Midday Update" using the same
    function rather than a duplicated "Morning Market Report" -- see src/analyst/graph.py."""
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_morning_report("2026-08-04", _account(), [], title="Midday Update", emoji="🕐")

    assert posted["text"].startswith("🕐 *Midday Update — 2026-08-04*")


def test_notify_morning_report_mentions_closed_market_and_crypto_continuing(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_morning_report("2026-08-04", _account(), [], stock_market_open=False, crypto_enabled=True)

    text = posted["text"]
    assert "stock market is closed" in text.lower()
    assert "crypto" in text.lower()
    assert "24/7" in text


def test_notify_morning_report_closed_market_omits_crypto_mention_when_crypto_disabled(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_morning_report("2026-08-04", _account(), [], stock_market_open=False, crypto_enabled=False)

    text = posted["text"]
    assert "stock market is closed" in text.lower()
    assert "crypto" not in text.lower()


def test_notify_dealer_signal_includes_timestamp(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_dealer_signal("MGN", "BUY", "strong momentum")

    assert "strong momentum" in posted["text"]
    assert re.search(_TIMESTAMP_RE, posted["text"])


def test_timestamp_omits_the_date():
    # Trading only happens within a single calendar day, so the date is redundant noise --
    # only the time of day is worth showing.
    assert not re.search(r"\d{4}-\d{2}-\d{2}", slack._timestamp())


def test_notify_floor_broker_result_includes_timestamp_and_opening_position_fields(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_floor_broker_result(
        "MGN",
        "BUY",
        "executed",
        "buy order submitted: order-123",
        reason="opening_position",
        fill_price=12.345,
        sl_price=11.5,
        tp_price=13.0,
    )

    text = posted["text"]
    assert re.search(_TIMESTAMP_RE, text)
    assert "reason: opening_position" in text
    assert "fill: $12.35" in text
    assert "SL: $11.50" in text
    assert "TP: $13.00" in text


def test_notify_floor_broker_result_uses_a_distinct_emoji_for_submitted(monkeypatch):
    """ROADMAP P0.14: a "submitted" order has not yet filled -- it must not render with the same
    ❌ used for "error", nor the ✅ used for a confirmed "executed" fill."""
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_floor_broker_result(
        "MGN",
        "BUY",
        "submitted",
        "buy order submitted: order-123",
        reason="opening_position",
        sl_price=9.8,
        tp_price=10.5,
    )

    text = posted["text"]
    assert text.startswith("📨")
    assert "❌" not in text
    assert "✅" not in text


def test_asset_icon_infers_stock_for_an_unknown_base():
    assert slack._asset_icon("ZZZZ", None) == "🏛️"


def test_asset_icon_infers_crypto_from_a_usd_pair():
    assert slack._asset_icon("BTC/USD", None) == "🪙"


def test_asset_icon_uses_an_explicit_class_over_inference():
    # symbol looks like an equity ticker, but the option path passes the underlying
    assert slack._asset_icon("AAPL", "option") == "📜"


def test_notify_dealer_signal_prefixes_the_stock_icon_for_an_equity_symbol(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_dealer_signal("MGN", "BUY", "strong momentum")

    assert "🏛️ MGN" in posted["text"]


def test_notify_dealer_signal_prefixes_the_crypto_icon_for_a_usd_pair(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_dealer_signal("BTC/USD", "BUY", "funding flipped positive")

    assert "🪙 BTC/USD" in posted["text"]


def test_notify_dealer_signal_uses_the_option_icon_when_asset_class_is_option(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_dealer_signal("AAPL", "BUY", "buying the 30-delta call", asset_class="option")

    assert "📜 AAPL" in posted["text"]
    assert "🏛️" not in posted["text"]


def test_notify_floor_broker_result_prefixes_the_asset_icon(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_floor_broker_result("MGN", "BUY", "executed", "buy order filled: order-1")
    assert "🏛️ MGN" in posted["text"]

    slack.notify_floor_broker_result("BTC/USD", "SELL", "executed", "synthetic take_profit @ 68000")
    assert "🪙 BTC/USD" in posted["text"]

    slack.notify_floor_broker_result(
        "AAPL", "BUY", "executed", "option buy order filled: order-2", asset_class="option"
    )
    assert "📜 AAPL" in posted["text"]


def test_notify_buy_kill_switch_activated_mentions_buy_blocked_and_sell_allowed(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_buy_kill_switch(True)

    assert "ACTIVATED" in posted["text"]
    assert "blocked" in posted["text"]
    assert "SELL" in posted["text"]


def test_notify_buy_kill_switch_deactivated_mentions_resumed(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_buy_kill_switch(False)

    assert "DEACTIVATED" in posted["text"]
    assert "resumed" in posted["text"]


def test_notify_power_state_powered_down_mentions_scaled_to_zero(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_power_state("powered_down", "2 crypto position(s) flattened first.")

    assert "scaled to 0" in posted["text"]
    assert "2 crypto position(s) flattened first." in posted["text"]


def test_notify_power_state_powered_up_mentions_scaled_to_one(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_power_state("powered_up", "")

    assert "scaled to 1" in posted["text"]


def test_notify_floor_broker_result_omits_extra_line_when_no_optional_fields_given(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_floor_broker_result("MGN", "SELL", "skipped", "no open position")

    assert "reason:" not in posted["text"]
    assert "fill:" not in posted["text"]
    assert "SL:" not in posted["text"]
    assert "TP:" not in posted["text"]


def test_notify_floor_broker_result_reports_take_profit_or_stop_loss_reason(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_floor_broker_result(
        "MGN",
        "SELL",
        "executed",
        "take_profit leg filled: order-456",
        reason="take_profit",
        fill_price=15.0,
    )

    assert "reason: take_profit" in posted["text"]
    assert "fill: $15.00" in posted["text"]
    assert "SL:" not in posted["text"]
    assert "TP:" not in posted["text"]


def _fill(symbol="MGN", side="buy", qty=1.0, price=10.0, time="2026-08-03T14:00:00Z"):
    return {"symbol": symbol, "side": side, "qty": qty, "price": price, "time": time}


def _account(equity=1050.0, last_equity=1000.0, cash=500.0, buying_power=2000.0):
    return {"equity": equity, "last_equity": last_equity, "cash": cash, "buying_power": buying_power}


def test_notify_eod_report_timestamps_each_fill_line(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_eod_report("2026-08-03", _account(), [_fill(time="2026-08-03T13:31:00Z")], [])

    assert re.search(r"\(\d{2}:\d{2} (AM|PM) \S+\)", posted["text"])


def test_notify_crypto_eod_report_timestamps_each_fill_line(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_crypto_eod_report("2026-08-03", [_fill(symbol="BTC/USD", time="2026-08-03T02:15:00Z")], [])

    assert re.search(r"\(\d{2}:\d{2} (AM|PM) \S+\)", posted["text"])


def test_format_fill_time_converts_utc_iso_string_to_eastern_clock_time():
    # 2026-08-03 is EDT (UTC-4): 14:00 UTC -> 10:00 AM EDT.
    assert slack._format_fill_time("2026-08-03T14:00:00Z") == "10:00 AM EDT"


def test_notify_error_includes_timestamp(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_error("Dealer", "boom")

    assert "boom" in posted["text"]
    assert re.search(_TIMESTAMP_RE, posted["text"])


def test_notify_market_closed_includes_timestamp(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_market_closed("EOD", "2026-08-01")

    assert "2026-08-01" in posted["text"]
    assert re.search(_TIMESTAMP_RE, posted["text"])


def test_notify_stock_market_closed_has_no_second_competing_timestamp(monkeypatch):
    # next_open is already a labeled clock time -- appending the generic "posted at" timestamp
    # produced two AM/PM times with no indication of which was which.
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_stock_market_closed("2026-08-04 09:30:00 ET")

    assert posted["text"] == "🔒 *Dealer* — stock market is closed. Next open: 2026-08-04 09:30:00 ET"


def test_notify_analyst_explain_includes_date_narrative_and_timestamp(monkeypatch):
    posted = {}
    monkeypatch.setattr(slack, "_post", lambda text: posted.update(text=text))

    slack.notify_analyst_explain("MGN was up 5% on earnings; Dealer bought, filled at $4.50.", "2026-08-04")

    text = posted["text"]
    assert "2026-08-04" in text
    assert "MGN was up 5% on earnings; Dealer bought, filled at $4.50." in text
    assert re.search(_TIMESTAMP_RE, text)


def test_escape_freetext_swaps_tilde_for_the_approx_sign():
    # Slack's mrkdwn treats a *pair* of tildes as strikethrough delimiters. LLM-authored
    # reasoning routinely writes "~47.7" for "approximately 47.7", and a paragraph with two such
    # values silently struck through everything between them.
    assert slack._escape_freetext("RSI (~47.7) vs SMA (~52.3)") == "RSI (≈47.7) vs SMA (≈52.3)"


def test_post_neutralizes_tildes_before_sending_to_slack(monkeypatch):
    monkeypatch.setattr(slack, "_WEBHOOK_URL", "http://example.invalid/hook")
    monkeypatch.setattr(slack, "load_config", lambda: SimpleNamespace(slack=SimpleNamespace(enabled=True)))
    sent = {}
    monkeypatch.setattr(
        slack.requests,
        "post",
        lambda url, json, timeout: sent.update(json) or _FakeResponse(),
    )

    slack._post("RSI (~47.7) vs SMA (~52.3)")

    assert "~" not in sent["text"]
    assert sent["text"] == "RSI (≈47.7) vs SMA (≈52.3)"


def test_post_does_not_send_when_slack_disabled(monkeypatch):
    monkeypatch.setattr(slack, "_WEBHOOK_URL", "http://example.invalid/hook")
    monkeypatch.setattr(slack, "load_config", lambda: SimpleNamespace(slack=SimpleNamespace(enabled=False)))
    called = []
    monkeypatch.setattr(slack.requests, "post", lambda *a, **k: called.append(1))

    slack._post("~ignored~")

    assert called == []
