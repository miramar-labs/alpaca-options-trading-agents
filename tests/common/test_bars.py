from datetime import datetime, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from omegaconf import OmegaConf

from src.common import bars


class FakeBars:
    def __init__(self, df):
        self.df = df


def _bars_df(symbol: str = "MGN") -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[symbol], pd.date_range("2026-08-10", periods=3, freq="h", tz="UTC")],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "high": [10.7, 11.2, 11.5],
            "low": [9.9, 10.4, 10.8],
            "close": [10.5, 11.0, 11.2],
            "volume": [1000, 1200, 1400],
            "trade_count": [10, 11, 12],
        },
        index=index,
    )


def test_fetch_bars_stock_branch_uses_iex_feed(monkeypatch):
    captured = {}

    def _fake_get_stock_bars(request):
        captured["request"] = request
        return FakeBars(_bars_df("MGN"))

    monkeypatch.setattr(bars.stock_data_client, "get_stock_bars", _fake_get_stock_bars)

    result = bars.fetch_bars("MGN", "1h", datetime(2026, 8, 10, tzinfo=timezone.utc), datetime(2026, 8, 11, tzinfo=timezone.utc))

    assert captured["request"].feed == DataFeed.IEX
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert not isinstance(result.index, pd.MultiIndex)


def test_fetch_bars_crypto_branch_uses_crypto_client(monkeypatch):
    calls = []
    monkeypatch.setattr(bars.crypto_data_client, "get_crypto_bars", lambda request: calls.append(request) or FakeBars(_bars_df("BTC/USD")))

    result = bars.fetch_bars("BTC/USD", "1h", datetime(2026, 8, 10, tzinfo=timezone.utc), datetime(2026, 8, 11, tzinfo=timezone.utc))

    assert calls
    assert not result.empty


def test_fetch_bars_returns_empty_dataframe_on_client_exception(monkeypatch):
    def _raise(request):
        raise RuntimeError("alpaca unavailable")

    monkeypatch.setattr(bars.stock_data_client, "get_stock_bars", _raise)

    result = bars.fetch_bars("MGN", "1h", datetime(2026, 8, 10, tzinfo=timezone.utc), datetime(2026, 8, 11, tzinfo=timezone.utc))

    assert result.empty


def test_fetch_multi_timeframe_bars_skips_non_stock_exchange_without_api_call(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("crypto exchange entries must not fetch live Dealer OHLCV")

    monkeypatch.setattr(bars, "fetch_bars", _fail_if_called)
    cfg = OmegaConf.create({"ohlcv_enrichment": {"timeframes": ["5m"], "bar_count": 60}})

    result = bars.fetch_multi_timeframe_bars("BTC/USD", "binance", cfg)

    assert result == {}


def test_fetch_multi_timeframe_bars_fetches_configured_stock_timeframes(monkeypatch):
    calls = []

    def _fake_fetch(symbol, timeframe_key, start, end):
        calls.append((symbol, timeframe_key))
        return _bars_df(symbol)

    monkeypatch.setattr(bars, "fetch_bars", _fake_fetch)
    cfg = OmegaConf.create({"ohlcv_enrichment": {"timeframes": ["5m", "1h"], "bar_count": 2}})

    result = bars.fetch_multi_timeframe_bars("MGN", "stocks", cfg)

    assert calls == [("MGN", "5m"), ("MGN", "1h")]
    assert set(result) == {"5m", "1h"}
    assert all(len(df) == 2 for df in result.values())
