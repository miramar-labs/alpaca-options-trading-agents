from alpaca.trading.enums import AssetClass
from omegaconf import OmegaConf

from src.common import portfolio_state


class FakePosition:
    def __init__(self, symbol, asset_class, market_value):
        self.symbol = symbol
        self.asset_class = asset_class
        self.market_value = market_value


class FakeTradingClient:
    def __init__(self, positions):
        self._positions = positions

    def get_all_positions(self):
        return self._positions


def _cfg(enable_stocks: bool, enable_crypto: bool):
    return OmegaConf.create(
        {
            "trading": {
                "stocks": {"enabled": enable_stocks},
                "crypto": {"enabled": enable_crypto},
                "crypto_taapi_exchange": "binance",
            }
        }
    )


def test_stock_position_skipped_when_stocks_disabled(monkeypatch):
    """Regression: an unmanaged equity position must not be merged in while enable_stocks=False,
    mirroring Dealer's own should_process_entry() gate."""
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("MGN", AssetClass.US_EQUITY, 100.0)])
    )

    result = portfolio_state.merge_held_positions({"symbols": []}, _cfg(enable_stocks=False, enable_crypto=True))

    assert result["symbols"] == []


def test_crypto_position_skipped_when_crypto_disabled(monkeypatch):
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("BTC/USD", AssetClass.CRYPTO, 10.0)])
    )

    result = portfolio_state.merge_held_positions({"symbols": []}, _cfg(enable_stocks=True, enable_crypto=False))

    assert result["symbols"] == []


def test_crypto_position_merged_with_configured_exchange_when_enabled(monkeypatch):
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("BTC/USD", AssetClass.CRYPTO, 10.0)])
    )

    result = portfolio_state.merge_held_positions({"symbols": []}, _cfg(enable_stocks=True, enable_crypto=True))

    assert result["symbols"] == [
        {
            "symbol": "BTC/USD",
            "exchange": "binance",
            "budget": 0.0,
            "held_value": 10.0,
            "is_held_only": True,
            "indicators": ["ALL"],
        }
    ]


def test_crypto_position_from_live_alpaca_shape_is_canonicalized_when_merged(monkeypatch):
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("BTCUSD", AssetClass.CRYPTO, 10.0)])
    )

    result = portfolio_state.merge_held_positions({"symbols": []}, _cfg(enable_stocks=True, enable_crypto=True))

    assert result["symbols"][0]["symbol"] == "BTC/USD"


def test_canonical_crypto_position_is_not_duplicated_by_live_alpaca_shape(monkeypatch):
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("BTCUSD", AssetClass.CRYPTO, 10.0)])
    )
    portfolio = {"symbols": [{"symbol": "BTC/USD", "exchange": "binance", "budget": 100, "indicators": ["ALL"]}]}

    result = portfolio_state.merge_held_positions(portfolio, _cfg(enable_stocks=True, enable_crypto=True))

    assert result["symbols"] == portfolio["symbols"]


def test_merged_position_budget_never_equals_market_value(monkeypatch):
    """Regression: a merged position's current market value must never flow through as new-BUY
    `budget` -- that would let a large held position silently re-authorize an equally large new
    BUY (or a shrunk one fall below Alpaca's crypto minimum notional). `held_value` carries the
    observed exposure instead; `budget` stays 0 for any merged (held-only) entry."""
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("MGN", AssetClass.US_EQUITY, 987.65)])
    )

    result = portfolio_state.merge_held_positions({"symbols": []}, _cfg(enable_stocks=True, enable_crypto=True))

    entry = result["symbols"][0]
    assert entry["budget"] == 0.0
    assert entry["held_value"] == 987.65
    assert entry["is_held_only"] is True


def test_analyst_picked_symbol_budget_is_left_untouched(monkeypatch):
    """A symbol already in the watchlist (Analyst's own pick, with its own real budget) must not
    be touched by the merge step at all -- the budget=0 rule only applies to entries the merge
    step itself synthesizes for previously-unmanaged positions."""
    monkeypatch.setattr(
        portfolio_state, "trading_client", FakeTradingClient([FakePosition("NVDA", AssetClass.US_EQUITY, 500.0)])
    )
    portfolio = {"symbols": [{"symbol": "NVDA", "exchange": "stocks", "budget": 5000, "indicators": ["ALL"]}]}

    result = portfolio_state.merge_held_positions(portfolio, _cfg(enable_stocks=True, enable_crypto=True))

    assert result["symbols"] == [{"symbol": "NVDA", "exchange": "stocks", "budget": 5000, "indicators": ["ALL"]}]
