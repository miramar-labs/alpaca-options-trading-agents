import os

import requests

TAAPI_BULK_URL = "https://api.taapi.io/bulk"


def _fmt_rsi(result, symbol):
    return (
        [f"📈 rsi: {result['value']}"],
        f"The current Relative Strength Index (RSI) for {symbol} is {result['value']}",
    )


def _fmt_sma(result, symbol):
    return (
        [f"📈 sma: {result['value']}"],
        f"The current Simple Moving Average (SMA) for {symbol} is {result['value']}",
    )


def _fmt_ema(result, symbol):
    return (
        [f"📈 ema: {result['value']}"],
        f"The current Exponential Moving Average (EMA) for {symbol} is {result['value']}",
    )


def _fmt_macd(result, symbol):
    return (
        [
            f"📈 macd: {result['valueMACD']}",
            f"📈 macdsig: {result['valueMACDSignal']}",
            f"📈 macdhist: {result['valueMACDHist']}",
        ],
        f"The current Moving Average convergence Divergence (MACD) values for {symbol} are "
        f"MACD {result['valueMACD']}, MACD Signal {result['valueMACDSignal']}, MACD History{result['valueMACDHist']}",
    )


def _fmt_bbands(result, symbol):
    return (
        [
            f"📈 bbandupper: {result['valueUpperBand']}",
            f"📈 bbandmiddle: {result['valueMiddleBand']}",
            f"📈 bbandlower: {result['valueLowerBand']}",
        ],
        f"The current Bollinger Bands for {symbol} are: Lower {result['valueLowerBand']}, "
        f"LowMiddleer {result['valueMiddleBand']}, Upper {result['valueUpperBand']}",
    )


def _fmt_volume(result, symbol):
    return (
        [f"📈 vol: {result['value']}"],
        f"The current Volume for {symbol} is {result['value']}",
    )


def _fmt_vosc(result, symbol):
    return (
        [f"📈 vosc: {result['value']}"],
        f"The current Volume Oscillator for {symbol} is {result['value']}",
    )


def _fmt_vwap(result, symbol):
    return (
        [f"📈 vwap: {result['value']}"],
        f"The current Volume Weighted Average Price (VWAP) for {symbol} is {result['value']}",
    )


def _fmt_stochrsi(result, symbol):
    return (
        [
            f"📈 srsiFK: {result['valueFastK']}",
            f"📈 srsiFD: {result['valueFastD']}",
        ],
        f"The current Stochastic Relative Strength Index values for {symbol} are: "
        f"FastK {result['valueFastK']}, FastD {result['valueFastD']}",
    )


RESULT_FORMATTERS = {
    "rsi": _fmt_rsi,
    "sma": _fmt_sma,
    "ema": _fmt_ema,
    "macd": _fmt_macd,
    "bbands": _fmt_bbands,
    "volume": _fmt_volume,
    "vosc": _fmt_vosc,
    "vwap": _fmt_vwap,
    "stochrsi": _fmt_stochrsi,
}


def _build_constructs(indicators_cfg, symbol: str, exchange: str, names: list[str]) -> list[dict]:
    by_interval: dict[str, list[dict]] = {}
    for name in names:
        entry = next((i for i in indicators_cfg if i["name"] == name), None)
        if entry is None:
            continue
        props = dict(entry["properties"])
        interval = props.pop("interval", "1h")
        by_interval.setdefault(interval, []).append({"id": name, "indicator": name, **props})

    constructs = []
    for interval, indicators in by_interval.items():
        construct = {"symbol": symbol, "interval": interval, "indicators": indicators}
        if exchange == "stocks":
            construct["type"] = "stocks"
        else:
            construct["exchange"] = exchange
        constructs.append(construct)
    return constructs


def fetch_indicators_bulk(indicators_cfg, symbol: str, exchange: str, names: list[str], log) -> str:
    """Fetches every requested indicator for one symbol in a single TAAPI /bulk request instead
    of one GET per indicator. TAAPI's rate limit is per-15-second-window per plan (Free: 1
    request/15s, Basic: 5, Pro: 30, Expert: 75) -- firing up to 9 individual GETs per symbol
    back-to-back would blow through even the Pro plan's limit the moment two symbols overlap."""
    secret = os.getenv("TAAPI_API_KEY")
    constructs = _build_constructs(indicators_cfg, symbol, exchange, names)
    if not constructs:
        return ""

    body = {"secret": secret, "construct": constructs if len(constructs) > 1 else constructs[0]}

    try:
        response = requests.post(TAAPI_BULK_URL, json=body, timeout=15)
    except requests.RequestException as exc:
        log(f"⚠️ taapi bulk request failed for {symbol}: {exc}")
        return ""

    if response.status_code != 200:
        log(f"⚠️ taapi bulk error for {symbol}: {response.status_code} {response.text[:200]}")
        return ""

    data = response.json().get("data", [])
    if not data:
        log(f"⚠️ taapi bulk returned no data for {symbol} (exchange={exchange}) -- possibly insufficient historical bars")
        return ""

    lines = []
    for item in data:
        indicator_id = item.get("id")
        formatter = RESULT_FORMATTERS.get(indicator_id)
        if formatter is None:
            continue
        if item.get("errors"):
            log(f"⚠️ {indicator_id} error for {symbol}: {item['errors']}")
            continue
        try:
            log_lines, text_line = formatter(item["result"], symbol)
        except KeyError as exc:
            # TAAPI can return a result missing an expected field (e.g. insufficient historical
            # bars for a thinly-traded symbol to compute the indicator) without populating
            # `errors` -- the "errors" guard above doesn't catch this. Indicators are supplementary
            # LLM context, not a trading-money gate, so skip just this one indicator rather than
            # letting it crash the whole fetch_indicators node (and with it, the entire run).
            log(f"⚠️ {indicator_id} result for {symbol} missing expected field {exc} -- skipping")
            continue
        for line in log_lines:
            log(line)
        lines.append(text_line)

    return "\n".join(lines)
