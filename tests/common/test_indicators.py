import requests

from src.common import indicators

_INDICATORS_CFG = [
    {"name": "rsi", "properties": {"interval": "1h", "period": 14}},
    {"name": "macd", "properties": {"interval": "1h", "optInFastPeriod": 12}},
]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_fetch_indicators_bulk_formats_multiple_indicators_in_one_request(monkeypatch):
    payload = {
        "data": [
            {"id": "rsi", "result": {"value": 71.2}},
            {"id": "macd", "result": {"valueMACD": 1.1, "valueMACDSignal": 0.9, "valueMACDHist": 0.2}},
        ]
    }
    monkeypatch.setattr(indicators.requests, "post", lambda *a, **k: FakeResponse(payload=payload))

    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "MGN", "stocks", ["rsi", "macd"], log=lambda *a: None)

    assert "Relative Strength Index (RSI) for MGN is 71.2" in text
    assert "MACD 1.1, MACD Signal 0.9" in text


def test_fetch_indicators_bulk_returns_empty_string_when_no_indicators_match_config():
    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "MGN", "stocks", ["stochrsi"], log=lambda *a: None)

    assert text == ""


def test_fetch_indicators_bulk_skips_indicator_that_returned_errors(monkeypatch):
    payload = {
        "data": [
            {"id": "rsi", "errors": ["symbol not found"]},
            {"id": "macd", "result": {"valueMACD": 1.1, "valueMACDSignal": 0.9, "valueMACDHist": 0.2}},
        ]
    }
    monkeypatch.setattr(indicators.requests, "post", lambda *a, **k: FakeResponse(payload=payload))

    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "MGN", "stocks", ["rsi", "macd"], log=lambda *a: None)

    assert "RSI" not in text
    assert "MACD 1.1" in text


def test_fetch_indicators_bulk_skips_indicator_result_missing_expected_field(monkeypatch):
    """Regression for a live Analyst crash: TAAPI can return a result missing an expected field
    (e.g. insufficient historical bars for a thinly-traded symbol) without populating `errors` --
    this must skip just that one indicator, not raise and take down the whole run."""
    payload = {
        "data": [
            {"id": "macd", "result": {"valueMACD": 1.1}},  # missing valueMACDSignal/valueMACDHist
            {"id": "rsi", "result": {"value": 71.2}},
        ]
    }
    monkeypatch.setattr(indicators.requests, "post", lambda *a, **k: FakeResponse(payload=payload))

    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "MGN", "stocks", ["rsi", "macd"], log=lambda *a: None)

    assert "MACD" not in text
    assert "Relative Strength Index (RSI) for MGN is 71.2" in text


def test_fetch_indicators_bulk_logs_when_taapi_returns_no_data(monkeypatch):
    """Regression: TAAPI can return HTTP 200 with an empty `data` array (observed live for
    thinly-traded pairs like CRV/USD, WIF/USD, LDO/USD -- likely insufficient historical bars).
    This used to return "" with zero logging, making the failure invisible in the Dealer's logs
    even though the LLM was about to receive an empty indicators block."""
    monkeypatch.setattr(indicators.requests, "post", lambda *a, **k: FakeResponse(payload={"data": []}))
    logged = []

    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "CRV/USD", "binance", ["rsi", "macd"], log=logged.append)

    assert text == ""
    assert any("no data" in line and "CRV/USD" in line for line in logged)


def test_fetch_indicators_bulk_returns_empty_string_on_non_200_response(monkeypatch):
    monkeypatch.setattr(indicators.requests, "post", lambda *a, **k: FakeResponse(status_code=500, text="boom"))

    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "MGN", "stocks", ["rsi"], log=lambda *a: None)

    assert text == ""


def test_fetch_indicators_bulk_returns_empty_string_on_request_exception(monkeypatch):
    def raise_exc(*a, **k):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(indicators.requests, "post", raise_exc)

    text = indicators.fetch_indicators_bulk(_INDICATORS_CFG, "MGN", "stocks", ["rsi"], log=lambda *a: None)

    assert text == ""
