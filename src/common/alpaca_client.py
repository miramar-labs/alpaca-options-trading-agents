import os

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest, OptionLatestQuoteRequest
from omegaconf import OmegaConf

from src.common.config import load_config
from src.common.symbols import canonical_crypto_symbol, is_usd_crypto_symbol

_DEFAULT_KEY_ENV = "ALPACA_PAPER_API_KEY"
_DEFAULT_SECRET_ENV = "ALPACA_PAPER_API_SECRET"


def live_account_env_names() -> tuple[str, str]:
    """Which env var names hold the live paper account's credentials, read from the live config's
    alpaca.live.key_env / alpaca.live.secret_env. Falls back to ALPACA_PAPER_API_KEY /
    ALPACA_PAPER_API_SECRET only when config loads but that section is absent -- e.g. a pod
    briefly running against the pre-unification alpaca.account1/account2 schema, where the default
    pair is exactly what account1 pointed at, so the migration is a no-op in either deploy order.

    A load_config() failure is deliberately NOT swallowed here. src.common.config already returns
    the last-known-good cached config on a transient GitHub blip and only raises on a genuine
    cold start with no config at all -- and in that state there is no safe default account to
    assume, since alpaca.live may have been switched to a non-default pair (e.g. the Alpaca
    competition's $100k Level-3 account). Silently routing the whole floor to ALPACA_PAPER_API_KEY
    would be worse than failing closed, so this propagates like every other config consumer.

    Every order -- stocks, crypto, options alike -- runs on this one account; there is no
    per-asset-class account split. This is what lets moving the whole floor to a different paper
    account be a config.yaml edit alone: config.yaml is polled fresh every 60s (see
    src.common.config.load_config), so a live pod picks up a new key_env/secret_env pointing at a
    different already-present env var pair with no rebuild/redeploy -- as long as that account's
    actual credentials were already added to the k8s Secret (and the pod restarted once to pick up
    the new env vars themselves; env var *values* are fixed at pod start, only which pair is
    *active* is live-switchable)."""
    cfg = load_config()
    key_env = OmegaConf.select(cfg, "alpaca.live.key_env", default=None)
    secret_env = OmegaConf.select(cfg, "alpaca.live.secret_env", default=None)
    return key_env or _DEFAULT_KEY_ENV, secret_env or _DEFAULT_SECRET_ENV


class _LazyAlpacaClient:
    """Wraps an Alpaca SDK client so it's built lazily -- from whichever env vars
    live_account_env_names() currently resolves to -- on first real use, instead of once at import
    time. Re-resolves (and rebuilds) on every access if the configured env var names have changed
    since the last build, which is what makes a config.yaml account switch take effect without a
    redeploy. Deferring construction out of import time also means importing this module never
    raises just because real Alpaca credentials aren't set (e.g. in a test environment) --
    construction only fails, same as before, the first time a caller actually uses the client."""

    def __init__(self, factory):
        self._factory = factory
        self._resolved_env_names = None
        self._client = None

    def _get_client(self):
        env_names = live_account_env_names()
        if self._client is None or env_names != self._resolved_env_names:
            key_env, secret_env = env_names
            self._client = self._factory(os.getenv(key_env), os.getenv(secret_env))
            self._resolved_env_names = env_names
        return self._client

    def __getattr__(self, name):
        return getattr(self._get_client(), name)


# One live paper account for the whole floor. trading_client places every order (stocks, crypto,
# options); the *_data_client wrappers read quotes. option_data_client is a separate wrapper only
# because it's a different Alpaca SDK class (OptionHistoricalDataClient), not a different account.
trading_client = _LazyAlpacaClient(lambda key, secret: TradingClient(key, secret, paper=True))
stock_data_client = _LazyAlpacaClient(StockHistoricalDataClient)
crypto_data_client = _LazyAlpacaClient(CryptoHistoricalDataClient)
option_data_client = _LazyAlpacaClient(OptionHistoricalDataClient)


def get_current_ask_price(symbol: str) -> float:
    if "/" in symbol or is_usd_crypto_symbol(symbol):
        symbol = canonical_crypto_symbol(symbol)
        quote = crypto_data_client.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=symbol))
    else:
        quote = stock_data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
    return quote[symbol].ask_price


def get_current_bid_price(symbol: str) -> float:
    if "/" in symbol or is_usd_crypto_symbol(symbol):
        symbol = canonical_crypto_symbol(symbol)
        quote = crypto_data_client.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=symbol))
    else:
        quote = stock_data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
    return quote[symbol].bid_price


def get_current_option_mid_price(contract_symbol: str) -> float:
    quote = option_data_client.get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol)
    )
    q = quote[contract_symbol]
    return (q.bid_price + q.ask_price) / 2


def get_current_option_ask_price(contract_symbol: str) -> float:
    quote = option_data_client.get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol)
    )
    return quote[contract_symbol].ask_price
