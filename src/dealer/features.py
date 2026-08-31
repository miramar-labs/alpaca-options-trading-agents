import math

import pandas as pd


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _pct(value: float | int | None) -> float | None:
    value = _finite(value)
    return None if value is None else value * 100


def _atr(bars: pd.DataFrame, period: int) -> float | None:
    if len(bars) < period + 1:
        return None
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return _finite(true_range.tail(period).mean())


def compute_derived_features(bars: pd.DataFrame, cfg) -> dict:
    if bars is None or bars.empty or len(bars) < 2:
        return {}

    bars = bars.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(bars) < 2:
        return {}

    enrichment_cfg = cfg.ohlcv_enrichment
    realized_vol_window = int(enrichment_cfg.get("realized_vol_window", 20))
    atr_period = int(enrichment_cfg.get("atr_period", 14))
    distance_window = int(enrichment_cfg.get("distance_window", 20))

    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    latest_close = _finite(close.iloc[-1])
    first_close = _finite(close.iloc[0])
    if latest_close is None or first_close in (None, 0.0):
        return {}

    features: dict[str, float] = {
        "latest_close": latest_close,
        "window_return_pct": _pct((latest_close / first_close) - 1),
    }

    returns = close.pct_change().dropna()
    if len(returns) >= realized_vol_window:
        features["realized_vol_pct"] = _pct(returns.tail(realized_vol_window).std())

    atr = _atr(bars, atr_period)
    if atr is not None:
        features["atr"] = atr
        features["atr_pct"] = _pct(atr / latest_close)

    if len(volume) >= 2:
        avg_volume = _finite(volume.iloc[:-1].tail(realized_vol_window).mean())
        latest_volume = _finite(volume.iloc[-1])
        if avg_volume and latest_volume is not None:
            features["relative_volume"] = latest_volume / avg_volume

    if len(bars) >= distance_window:
        recent = bars.tail(distance_window)
        recent_high = _finite(recent["high"].max())
        recent_low = _finite(recent["low"].min())
        if recent_high:
            features["distance_high_pct"] = _pct((latest_close / recent_high) - 1)
        if recent_low:
            features["distance_low_pct"] = _pct((latest_close / recent_low) - 1)

    dollar_volume = close * volume
    total_volume = _finite(volume.sum())
    if total_volume:
        vwap = _finite(dollar_volume.sum() / total_volume)
        if vwap:
            features["vwap"] = vwap
            features["vwap_distance_pct"] = _pct((latest_close / vwap) - 1)

    ema20 = close.ewm(span=20, adjust=False).mean()
    if len(ema20) >= 20:
        last_ema20 = _finite(ema20.iloc[-1])
        prior_ema20 = _finite(ema20.iloc[-5] if len(ema20) >= 5 else ema20.iloc[0])
        if last_ema20:
            features["ema20"] = last_ema20
            features["ema20_distance_pct"] = _pct((latest_close / last_ema20) - 1)
        if last_ema20 and prior_ema20:
            features["ema20_slope_pct"] = _pct((last_ema20 / prior_ema20) - 1)

    ema50 = close.ewm(span=50, adjust=False).mean()
    if len(ema50) >= 50:
        last_ema50 = _finite(ema50.iloc[-1])
        if last_ema50:
            features["ema50"] = last_ema50
            features["ema50_distance_pct"] = _pct((latest_close / last_ema50) - 1)

    return {k: v for k, v in features.items() if _finite(v) is not None}


def format_features_text(features_by_timeframe: dict[str, dict], symbol: str) -> str:
    if not features_by_timeframe:
        return ""

    lines = [f"OHLCV-derived market structure for {symbol}:"]
    for timeframe, features in features_by_timeframe.items():
        if not features:
            continue
        lines.append(f"[{timeframe}]")
        for key, value in features.items():
            if value is None or not math.isfinite(float(value)):
                continue
            if key.endswith("_pct"):
                lines.append(f"- {key}: {float(value):.2f}%")
            elif key in {"latest_close", "atr", "vwap", "ema20", "ema50"}:
                lines.append(f"- {key}: {float(value):.4f}")
            else:
                lines.append(f"- {key}: {float(value):.2f}")

    return "\n".join(lines) if len(lines) > 1 else ""
