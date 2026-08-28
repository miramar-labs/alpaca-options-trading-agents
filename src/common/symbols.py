import threading

from src.common.logging import get_logger

log = get_logger("SYMBOLS")

USD_CRYPTO_SUFFIX = "USD"

# Shared refresh cadence for every caller of refresh_known_usd_crypto_bases_from_alpaca() --
# Alpaca's crypto listings change rarely, so this stays generous to avoid hammering the asset-list
# endpoint from every service that polls or loops more often than this.
REFRESH_INTERVAL_S = 3600

# Seed/fallback set used until refresh_known_usd_crypto_bases_from_alpaca() first succeeds, and
# again any time a later refresh fails -- classification must never block on, or go unavailable
# because of, a live network call (see that function's docstring for why the refresh itself is
# never called inline from here).
_FALLBACK_USD_CRYPTO_BASES = frozenset({
    "AAVE", "AVAX", "BCH", "BTC", "DOGE", "ETH", "LINK", "LTC", "SHIB", "SOL", "UNI",
})

_bases_lock = threading.Lock()
_known_usd_crypto_bases = _FALLBACK_USD_CRYPTO_BASES


def known_usd_crypto_bases() -> frozenset[str]:
    with _bases_lock:
        return _known_usd_crypto_bases


def refresh_known_usd_crypto_bases_from_alpaca() -> int:
    """Repopulates the known-USD-crypto-base set from Alpaca's own tradable crypto asset list, so
    canonical_crypto_symbol()/is_usd_crypto_symbol() stay correct as Alpaca lists new coins
    instead of silently mismanaging (or, if the base check were dropped instead, misrouting a
    stock ticker that happens to end in "USD") anything not on a hand-maintained list.

    Must be called periodically from each service's poll loop (mirrors src/common/config.py's
    refresh pattern) -- never inline from canonical_crypto_symbol()/is_usd_crypto_symbol(), since
    those run on hot paths (a quote lookup, a BUY/SELL gate, order reconciliation) that must never
    block on, or fail because of, a slow/unreachable Alpaca call. On failure, leaves the current
    (last-known-good, or the bundled fallback if this has never succeeded) set in place and
    returns its size. Imports trading_client lazily -- src.common.alpaca_client imports
    canonical_crypto_symbol/is_usd_crypto_symbol from this module at load time, so a top-level
    import here would be circular.
    """
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    from src.common.alpaca_client import trading_client

    global _known_usd_crypto_bases
    try:
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.CRYPTO, status=AssetStatus.ACTIVE)
        )
        bases = {
            asset.symbol.split("/", 1)[0].upper()
            for asset in assets
            if asset.tradable and asset.symbol.upper().endswith("/" + USD_CRYPTO_SUFFIX)
        }
    except Exception as exc:
        log(f"⚠️ crypto asset list refresh failed, keeping last-known-good ({len(known_usd_crypto_bases())} base(s)): {exc}")
        return len(known_usd_crypto_bases())

    if bases:
        with _bases_lock:
            _known_usd_crypto_bases = frozenset(bases)
    return len(known_usd_crypto_bases())


def canonical_crypto_symbol(symbol: str) -> str:
    """Return the app's canonical Alpaca USD crypto pair form, e.g. BTCUSD -> BTC/USD.

    Only inserts the "/" when the base is one of Alpaca's actual tradable USD-quoted crypto bases
    (known_usd_crypto_bases(), refreshed from Alpaca -- see refresh_known_usd_crypto_bases_from_
    alpaca()). This must stay conservative rather than matching on the "USD" suffix alone: a false
    positive here would misroute a real stock ticker that happens to end in "USD" (e.g. a
    hypothetical "PUSD") to the crypto quote/order path instead of just failing to canonicalize an
    unrecognized crypto one.
    """
    normalized = symbol.strip().upper()
    if "/" in normalized:
        return normalized
    if normalized.endswith(USD_CRYPTO_SUFFIX) and len(normalized) > len(USD_CRYPTO_SUFFIX):
        base = normalized[: -len(USD_CRYPTO_SUFFIX)]
        if base in known_usd_crypto_bases():
            return f"{base}/{USD_CRYPTO_SUFFIX}"
    return normalized


def alpaca_order_symbol(symbol: str) -> str:
    """Return Alpaca Trading API's position/order lookup form for any canonical app symbol."""
    return symbol.replace("/", "").upper()


def is_usd_crypto_symbol(symbol: str) -> bool:
    return canonical_crypto_symbol(symbol).endswith("/USD")
