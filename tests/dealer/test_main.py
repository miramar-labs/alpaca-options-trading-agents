from datetime import datetime, timezone

import pytz
from omegaconf import OmegaConf

from src.dealer import main as dealer_main
from src.dealer.main import market_is_open, should_process_entry


def _cfg(enable_stocks: bool, enable_crypto: bool):
    return OmegaConf.create(
        {"trading": {"stocks": {"enabled": enable_stocks}, "crypto": {"enabled": enable_crypto}}}
    )


def _trading_cfg(market_override=False, buffer=0):
    return OmegaConf.create({"trading": {"market_override": market_override, "buffer": buffer}})


class FakeClock:
    def __init__(self, is_open):
        self.is_open = is_open
        self.next_open = datetime(2026, 8, 4, 13, 30, tzinfo=timezone.utc)
        self.next_close = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)


class FakeTradingClient:
    def __init__(self, is_open):
        self._clock = FakeClock(is_open)

    def get_clock(self):
        return self._clock


def test_stock_entry_processed_only_when_stocks_enabled():
    entry = {"symbol": "MGN", "exchange": "stocks"}

    assert should_process_entry(entry, _cfg(enable_stocks=True, enable_crypto=True)) is True
    assert should_process_entry(entry, _cfg(enable_stocks=False, enable_crypto=True)) is False


def test_crypto_entry_processed_only_when_crypto_enabled():
    entry = {"symbol": "BTC/USD", "exchange": "binance"}

    assert should_process_entry(entry, _cfg(enable_stocks=True, enable_crypto=True)) is True
    assert should_process_entry(entry, _cfg(enable_stocks=True, enable_crypto=False)) is False


def test_market_closed_posts_slack_notice_on_first_check(monkeypatch):
    monkeypatch.setattr(dealer_main, "_last_market_open", None)
    monkeypatch.setattr(dealer_main, "trading_client", FakeTradingClient(is_open=False))
    posted = {}
    monkeypatch.setattr(dealer_main.slack, "notify_stock_market_closed", lambda next_open: posted.update(next_open=next_open))

    result = market_is_open(_trading_cfg(), log=lambda *a: None)

    assert result is False
    assert "next_open" in posted


def test_market_closed_does_not_repeat_notice_while_still_closed(monkeypatch):
    monkeypatch.setattr(dealer_main, "_last_market_open", False)
    monkeypatch.setattr(dealer_main, "trading_client", FakeTradingClient(is_open=False))
    posted = {}
    monkeypatch.setattr(dealer_main.slack, "notify_stock_market_closed", lambda next_open: posted.update(next_open=next_open))

    market_is_open(_trading_cfg(), log=lambda *a: None)

    assert posted == {}


def test_market_closed_notice_fires_again_after_reopening(monkeypatch):
    monkeypatch.setattr(dealer_main, "_last_market_open", True)
    monkeypatch.setattr(dealer_main, "trading_client", FakeTradingClient(is_open=False))
    posted = {}
    monkeypatch.setattr(dealer_main.slack, "notify_stock_market_closed", lambda next_open: posted.update(next_open=next_open))

    market_is_open(_trading_cfg(), log=lambda *a: None)

    assert "next_open" in posted


def test_market_open_does_not_post_closed_notice(monkeypatch):
    monkeypatch.setattr(dealer_main, "_last_market_open", False)
    monkeypatch.setattr(dealer_main, "trading_client", FakeTradingClient(is_open=True))
    eastern = pytz.timezone("US/Eastern")
    monkeypatch.setattr(dealer_main, "_now_et", lambda: eastern.localize(datetime(2026, 8, 3, 10, 0)))
    posted = {}
    monkeypatch.setattr(dealer_main.slack, "notify_stock_market_closed", lambda next_open: posted.update(next_open=next_open))

    result = market_is_open(_trading_cfg(buffer=0), log=lambda *a: None)

    assert result is True
    assert posted == {}


def test_refresh_symbol_bases_if_due_skips_within_the_interval(monkeypatch):
    monkeypatch.setattr(dealer_main.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(dealer_main, "_last_symbol_bases_refresh", 1000.0 - (dealer_main.symbols.REFRESH_INTERVAL_S - 1))
    calls = []
    monkeypatch.setattr(dealer_main.symbols, "refresh_known_usd_crypto_bases_from_alpaca", lambda: calls.append(1) or 11)

    dealer_main.refresh_symbol_bases_if_due()

    assert calls == []


def test_refresh_symbol_bases_if_due_refreshes_once_the_interval_has_elapsed(monkeypatch):
    monkeypatch.setattr(dealer_main.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(dealer_main, "_last_symbol_bases_refresh", 1000.0 - (dealer_main.symbols.REFRESH_INTERVAL_S + 1))
    calls = []
    monkeypatch.setattr(dealer_main.symbols, "refresh_known_usd_crypto_bases_from_alpaca", lambda: calls.append(1) or 11)

    dealer_main.refresh_symbol_bases_if_due()

    assert calls == [1]
    assert dealer_main._last_symbol_bases_refresh == 1000.0


def test_refresh_symbol_bases_if_due_survives_an_exception(monkeypatch):
    monkeypatch.setattr(dealer_main.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(dealer_main, "_last_symbol_bases_refresh", 1000.0 - (dealer_main.symbols.REFRESH_INTERVAL_S + 1))

    def _raise():
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(dealer_main.symbols, "refresh_known_usd_crypto_bases_from_alpaca", _raise)

    dealer_main.refresh_symbol_bases_if_due()  # must not raise

    assert dealer_main._last_symbol_bases_refresh == 1000.0


def test_market_override_does_not_post_closed_notice(monkeypatch):
    monkeypatch.setattr(dealer_main, "_last_market_open", False)
    posted = {}
    monkeypatch.setattr(dealer_main.slack, "notify_stock_market_closed", lambda next_open: posted.update(next_open=next_open))

    result = market_is_open(_trading_cfg(market_override=True), log=lambda *a: None)

    assert result is True
    assert posted == {}
