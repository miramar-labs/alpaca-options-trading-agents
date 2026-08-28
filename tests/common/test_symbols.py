import pytest

from src.common import symbols


@pytest.fixture(autouse=True)
def _reset_known_bases():
    # symbols._known_usd_crypto_bases is a module-level global -- each test must start from the
    # same bundled fallback rather than leaking whatever a previous test refreshed it to.
    original = symbols._known_usd_crypto_bases
    symbols._known_usd_crypto_bases = symbols._FALLBACK_USD_CRYPTO_BASES
    yield
    symbols._known_usd_crypto_bases = original


class FakeAsset:
    def __init__(self, symbol, tradable=True):
        self.symbol = symbol
        self.tradable = tradable


def test_canonical_crypto_symbol_passes_through_slash_form():
    assert symbols.canonical_crypto_symbol("btc/usd") == "BTC/USD"


def test_canonical_crypto_symbol_adds_slash_for_a_known_base():
    assert symbols.canonical_crypto_symbol("BTCUSD") == "BTC/USD"


def test_canonical_crypto_symbol_leaves_an_unrecognized_base_unconverted():
    # Before the first successful refresh from Alpaca, only the bundled fallback bases are known
    # -- an unlisted coin like XRP must not be guessed at from the suffix alone (see the PUSD
    # test below for why: that would also misroute a stock ticker ending in "USD").
    assert symbols.canonical_crypto_symbol("XRPUSD") == "XRPUSD"


def test_canonical_crypto_symbol_leaves_non_usd_pair_unconverted():
    assert symbols.canonical_crypto_symbol("SHIBUSDT") == "SHIBUSDT"


def test_canonical_crypto_symbol_leaves_stock_ticker_unconverted():
    assert symbols.canonical_crypto_symbol("AAPL") == "AAPL"


def test_canonical_crypto_symbol_leaves_a_usd_suffixed_stock_ticker_unconverted():
    # Regression guard: PUSD must never be treated as crypto just because it ends in "USD".
    assert symbols.canonical_crypto_symbol("PUSD") == "PUSD"


def test_is_usd_crypto_symbol_true_for_known_base():
    assert symbols.is_usd_crypto_symbol("BTCUSD") is True
    assert symbols.is_usd_crypto_symbol("BTC/USD") is True


def test_is_usd_crypto_symbol_false_for_non_usd_pair():
    assert symbols.is_usd_crypto_symbol("SHIB/USDT") is False


def test_is_usd_crypto_symbol_false_for_stock_ending_in_usd():
    assert symbols.is_usd_crypto_symbol("PUSD") is False


def test_alpaca_order_symbol_strips_slash():
    assert symbols.alpaca_order_symbol("XRP/USD") == "XRPUSD"
    assert symbols.alpaca_order_symbol("AAPL") == "AAPL"


def test_refresh_adds_a_newly_listed_base_from_alpaca(monkeypatch):
    # XRP isn't in the bundled fallback -- this is the regression the allowlist bug caused:
    # a coin Alpaca lists after the code was written stayed unrecognized forever.
    fake_trading_client = type(
        "FakeTradingClient",
        (),
        {"get_all_assets": lambda self, request: [FakeAsset("XRP/USD"), FakeAsset("BTC/USD")]},
    )()
    monkeypatch.setattr("src.common.alpaca_client.trading_client", fake_trading_client)

    count = symbols.refresh_known_usd_crypto_bases_from_alpaca()

    assert count == 2
    assert symbols.canonical_crypto_symbol("XRPUSD") == "XRP/USD"
    assert symbols.is_usd_crypto_symbol("XRPUSD") is True


def test_refresh_excludes_non_usd_and_non_tradable_assets(monkeypatch):
    fake_trading_client = type(
        "FakeTradingClient",
        (),
        {
            "get_all_assets": lambda self, request: [
                FakeAsset("SHIB/USDT"),
                FakeAsset("XRP/USD", tradable=False),
                FakeAsset("BTC/USD"),
            ]
        },
    )()
    monkeypatch.setattr("src.common.alpaca_client.trading_client", fake_trading_client)

    symbols.refresh_known_usd_crypto_bases_from_alpaca()

    assert symbols.known_usd_crypto_bases() == frozenset({"BTC"})


def test_refresh_keeps_last_known_good_set_on_failure(monkeypatch):
    def _raise(request):
        raise RuntimeError("alpaca unavailable")

    fake_trading_client = type("FakeTradingClient", (), {"get_all_assets": lambda self, request: _raise(request)})()
    monkeypatch.setattr("src.common.alpaca_client.trading_client", fake_trading_client)

    before = symbols.known_usd_crypto_bases()
    count = symbols.refresh_known_usd_crypto_bases_from_alpaca()

    assert symbols.known_usd_crypto_bases() == before
    assert count == len(before)


def test_refresh_keeps_last_known_good_set_when_alpaca_returns_no_usd_assets(monkeypatch):
    fake_trading_client = type(
        "FakeTradingClient", (), {"get_all_assets": lambda self, request: [FakeAsset("SHIB/USDT")]}
    )()
    monkeypatch.setattr("src.common.alpaca_client.trading_client", fake_trading_client)

    before = symbols.known_usd_crypto_bases()
    symbols.refresh_known_usd_crypto_bases_from_alpaca()

    assert symbols.known_usd_crypto_bases() == before
