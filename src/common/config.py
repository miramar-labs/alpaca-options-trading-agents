import time

import requests
import yaml
from omegaconf import OmegaConf

from src.common.logging import get_logger

log = get_logger("CONFIG")

_CONFIG_URL = "https://raw.githubusercontent.com/miramar-labs/alpaca-options-trading-agents/main/config.yaml"
_REFRESH_SECS = 60

_cache = {"value": None, "fetched_at": 0.0}


def load_config():
    """Fetches config.yaml fresh from GitHub (main branch) so every live service reflects a
    config change within _REFRESH_SECS of it being pushed, with no rebuild/redeploy. Caches
    briefly to avoid a network round-trip on every call site (a Dealer graph node invoked per
    poll cycle, a Floor Broker function invoked per HTTP request, Slack's _post() invoked many
    times per run). On fetch/parse failure: fall back to the last-known-good cached value with a
    warning if one exists (a transient GitHub blip must never crash a running trading system or
    block a notification) -- otherwise let the exception propagate, since nothing can run without
    config and there's no bundled fallback baked into the image."""
    now = time.monotonic()
    if _cache["value"] is not None and now - _cache["fetched_at"] < _REFRESH_SECS:
        return _cache["value"]

    try:
        resp = requests.get(_CONFIG_URL, timeout=10)
        resp.raise_for_status()
        cfg = OmegaConf.create(resp.text)
    except (requests.RequestException, yaml.YAMLError) as exc:
        if _cache["value"] is not None:
            age = int(now - _cache["fetched_at"])
            log(f"⚠️ config fetch failed, using last-known-good ({age}s old): {exc}")
            return _cache["value"]
        raise

    _cache["value"] = cfg
    _cache["fetched_at"] = now
    return cfg
