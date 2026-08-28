import pytest
import requests

from src.common import config


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture(autouse=True)
def _reset_cache_and_clock(monkeypatch):
    # config._cache is a module-level global shared across every call site (Dealer, Floor
    # Broker, Slack) -- each test must start from a clean slate, and control time.monotonic()
    # directly rather than sleeping, since the TTL window is _REFRESH_SECS (60) real seconds.
    monkeypatch.setitem(config._cache, "value", None)
    monkeypatch.setitem(config._cache, "fetched_at", 0.0)
    clock = {"now": 1000.0}
    monkeypatch.setattr(config.time, "monotonic", lambda: clock["now"])
    return clock


def test_load_config_fetches_and_parses_yaml_from_github(monkeypatch):
    monkeypatch.setattr(
        config.requests, "get", lambda url, timeout: FakeResponse(text="trading:\n  slP: 0.98\n")
    )

    cfg = config.load_config()

    assert cfg.trading.slP == 0.98


def test_load_config_reuses_cached_value_within_the_ttl_window(monkeypatch, _reset_cache_and_clock):
    calls = {"count": 0}

    def _fake_get(url, timeout):
        calls["count"] += 1
        return FakeResponse(text="trading:\n  slP: 0.98\n")

    monkeypatch.setattr(config.requests, "get", _fake_get)

    first = config.load_config()
    _reset_cache_and_clock["now"] += config._REFRESH_SECS - 1
    second = config.load_config()

    assert calls["count"] == 1
    assert second is first


def test_load_config_refetches_once_the_ttl_window_has_elapsed(monkeypatch, _reset_cache_and_clock):
    calls = {"count": 0}

    def _fake_get(url, timeout):
        calls["count"] += 1
        return FakeResponse(text=f"trading:\n  slP: {calls['count']}\n")

    monkeypatch.setattr(config.requests, "get", _fake_get)

    first = config.load_config()
    _reset_cache_and_clock["now"] += config._REFRESH_SECS + 1
    second = config.load_config()

    assert calls["count"] == 2
    assert first.trading.slP == 1
    assert second.trading.slP == 2


def test_load_config_falls_back_to_last_known_good_on_a_failed_refetch(monkeypatch, _reset_cache_and_clock):
    monkeypatch.setattr(config.requests, "get", lambda url, timeout: FakeResponse(text="trading:\n  slP: 0.98\n"))
    good = config.load_config()

    def _fail(url, timeout):
        raise requests.ConnectionError("network blip")

    monkeypatch.setattr(config.requests, "get", _fail)
    _reset_cache_and_clock["now"] += config._REFRESH_SECS + 1

    stale = config.load_config()

    assert stale is good


def test_load_config_falls_back_to_last_known_good_on_a_yaml_parse_error(monkeypatch, _reset_cache_and_clock):
    monkeypatch.setattr(config.requests, "get", lambda url, timeout: FakeResponse(text="trading:\n  slP: 0.98\n"))
    good = config.load_config()

    monkeypatch.setattr(config.requests, "get", lambda url, timeout: FakeResponse(text="not: valid: yaml: ["))
    _reset_cache_and_clock["now"] += config._REFRESH_SECS + 1

    stale = config.load_config()

    assert stale is good


def test_load_config_raises_when_no_cache_exists_and_the_fetch_fails(monkeypatch):
    def _fail(url, timeout):
        raise requests.ConnectionError("network blip")

    monkeypatch.setattr(config.requests, "get", _fail)

    with pytest.raises(requests.ConnectionError):
        config.load_config()


def test_load_config_raises_when_no_cache_exists_and_the_response_is_invalid_yaml(monkeypatch):
    monkeypatch.setattr(config.requests, "get", lambda url, timeout: FakeResponse(text="not: valid: yaml: ["))

    with pytest.raises(Exception):
        config.load_config()


def test_load_config_raises_on_a_non_200_response_with_no_cache(monkeypatch):
    monkeypatch.setattr(config.requests, "get", lambda url, timeout: FakeResponse(status_code=500))

    with pytest.raises(requests.HTTPError):
        config.load_config()
