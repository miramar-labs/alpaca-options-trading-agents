from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.common.alpaca_client import crypto_data_client, stock_data_client
from src.common.logging import get_logger

log = get_logger("BARS")

TIMEFRAMES = {
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "1h": TimeFrame.Hour,
    "1d": TimeFrame.Day,
}

TIMEFRAME_DELTAS = {
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}

MIN_LOOKBACKS = {
    "5m": timedelta(days=7),
    "1h": timedelta(days=30),
    "1d": timedelta(days=365),
}


def fetch_bars(symbol: str, timeframe_key: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV bars for stocks or crypto. Returns an empty DataFrame on market-data errors."""
    timeframe = TIMEFRAMES[timeframe_key]

    try:
        if "/" in symbol:
            request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
            bars = crypto_data_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = stock_data_client.get_stock_bars(request)
    except Exception as exc:
        log(f"⚠️ bar fetch failed for {symbol} {timeframe_key}: {exc}")
        return pd.DataFrame()

    df = bars.df
    if df.empty:
        log(f"⚠️ no bars returned for {symbol} {timeframe_key} between {start} and {end}")
        return df

    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[symbol]

    return df[["open", "high", "low", "close", "volume"]]


def fetch_multi_timeframe_bars(symbol: str, exchange: str, cfg) -> dict[str, pd.DataFrame]:
    """Fetch configured stock-only Dealer OHLCV windows. Crypto is intentionally a live no-op."""
    if exchange != "stocks":
        log(f"⏭️ OHLCV enrichment skipped for {symbol}: exchange={exchange}")
        return {}

    enrichment_cfg = cfg.ohlcv_enrichment
    bar_count = int(enrichment_cfg.get("bar_count", 60))
    now = datetime.now(timezone.utc)
    results: dict[str, pd.DataFrame] = {}

    for timeframe_key in enrichment_cfg.get("timeframes", []):
        if timeframe_key not in TIMEFRAMES:
            log(f"⚠️ unsupported OHLCV timeframe {timeframe_key} for {symbol}")
            continue

        # Calendar lookback is intentionally padded: equities have closed periods overnight,
        # weekends, and holidays, so exactly bar_count * interval often returns too few bars.
        lookback = max(TIMEFRAME_DELTAS[timeframe_key] * bar_count * 4, MIN_LOOKBACKS[timeframe_key])
        start = now - lookback
        df = fetch_bars(symbol, timeframe_key, start, now)
        if df.empty:
            continue
        results[timeframe_key] = df.tail(bar_count)

    return results
