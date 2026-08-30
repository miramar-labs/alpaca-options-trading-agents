import json
import os

from alpaca.trading.enums import AssetClass
from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException

from src.common.alpaca_client import trading_client
from src.common.symbols import canonical_crypto_symbol

NAMESPACE = os.getenv("POD_NAMESPACE", "multi-agent-ai-trader")
CONFIGMAP_NAME = "portfolio"
DATA_KEY = "portfolio.json"


def _load_k8s_config() -> None:
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()


def read_portfolio() -> dict:
    """Reads the `portfolio` ConfigMap fresh — no caching, matches the Dealer's poll cadence."""
    _load_k8s_config()
    v1 = client.CoreV1Api()
    cm = v1.read_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE)
    return json.loads(cm.data.get(DATA_KEY, "{}"))


def merge_held_positions(portfolio: dict, cfg) -> dict:
    """Adds any Alpaca position not already in the watchlist -- e.g. one opened before this app
    existed -- so Dealer keeps deciding BUY/HOLD/SELL on it instead of leaving it unmanaged
    forever. Stock positions are only merged in when `cfg.trading.stocks.enabled` is set, and
    crypto positions only when `cfg.trading.crypto.enabled` is set, matching Dealer's own
    exchange-filter gate."""
    symbols = portfolio.get("symbols", [])
    known = {entry["symbol"] for entry in symbols}

    for position in trading_client.get_all_positions():
        symbol = position.symbol
        if position.asset_class == AssetClass.CRYPTO:
            symbol = canonical_crypto_symbol(position.symbol)

        if symbol in known:
            continue

        if position.asset_class == AssetClass.US_EQUITY and cfg.trading.stocks.enabled:
            exchange = "stocks"
        elif position.asset_class == AssetClass.CRYPTO and cfg.trading.crypto.enabled:
            exchange = cfg.trading.crypto_taapi_exchange
        else:
            continue

        symbols.append(
            {
                "symbol": symbol,
                "exchange": exchange,
                # A merged position's current market value is observed exposure, not authorized
                # new-BUY capital -- passing it through as `budget` would let a large held
                # position silently re-authorize an equally large new BUY (and a shrunk one could
                # fall below Alpaca's crypto minimum notional). `held_value` carries the value for
                # SELL/HOLD context; `budget` stays 0 so a BUY on a held-only position is refused
                # by call_floor_broker rather than sized off it.
                "budget": 0.0,
                "held_value": float(position.market_value),
                "is_held_only": True,
                "indicators": ["ALL"],
            }
        )

    return {**portfolio, "symbols": symbols}


def write_portfolio(portfolio: dict) -> None:
    """Patches the `portfolio` ConfigMap. Called once per Analyst CronJob run."""
    _load_k8s_config()
    v1 = client.CoreV1Api()
    body = {"data": {DATA_KEY: json.dumps(portfolio, indent=2)}}
    try:
        v1.patch_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE, body)
    except ApiException as exc:
        if exc.status != 404:
            raise
        cm = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=CONFIGMAP_NAME, namespace=NAMESPACE),
            data={DATA_KEY: json.dumps(portfolio, indent=2)},
        )
        v1.create_namespaced_config_map(NAMESPACE, cm)
