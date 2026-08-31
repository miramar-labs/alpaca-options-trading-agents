import math

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.dealer.features import compute_derived_features, format_features_text


def _cfg():
    return OmegaConf.create(
        {
            "ohlcv_enrichment": {
                "realized_vol_window": 5,
                "atr_period": 3,
                "distance_window": 5,
            }
        }
    )


def _bars(rows: int = 60) -> pd.DataFrame:
    closes = [100 + i for i in range(rows)]
    return pd.DataFrame(
        {
            "open": [c - 0.5 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000 + i * 10 for i in range(rows)],
        }
    )


def test_compute_derived_features_returns_expected_core_values():
    features = compute_derived_features(_bars(10), _cfg())

    assert features["latest_close"] == 109.0
    assert features["window_return_pct"] == pytest.approx(9.0)
    assert features["atr"] == 2.0
    assert features["atr_pct"] == pytest.approx(2.0 / 109.0 * 100)
    assert "relative_volume" in features
    assert "vwap_distance_pct" in features
    assert all(math.isfinite(value) for value in features.values())


def test_compute_derived_features_returns_empty_for_insufficient_rows():
    assert compute_derived_features(pd.DataFrame(), _cfg()) == {}
    assert compute_derived_features(_bars(1), _cfg()) == {}


def test_format_features_text_is_line_oriented_and_omits_empty_features():
    text = format_features_text({"5m": {"latest_close": 109.0, "window_return_pct": 9.0}, "1h": {}}, "MGN")

    assert "OHLCV-derived market structure for MGN" in text
    assert "[5m]" in text
    assert "- latest_close: 109.0000" in text
    assert "- window_return_pct: 9.00%" in text
    assert "[1h]" not in text


def test_format_features_text_returns_empty_for_empty_input():
    assert format_features_text({}, "MGN") == ""
