from datetime import datetime, timedelta

import pytz
from alpaca.trading.enums import AssetClass
from omegaconf import OmegaConf

from src.analyst import graph
from src.analyst.graph import crypto_eod_report, validate_selection
from src.common import eod


def _state(raw_candidates, picks):
    return {"raw_candidates": raw_candidates, "selection": {"symbols": picks}}


def _cfg(enable_crypto: bool):
    return OmegaConf.create({"trading": {"crypto": {"enabled": enable_crypto}}})


def test_discover_candidates_skips_stocks_when_market_closed(monkeypatch):
    """Crypto trades 24/7 and must still be discovered on a day the stock market is closed --
    only the stock screener branch is gated on stock_market_open."""
    monkeypatch.setattr(graph.sources, "fetch_screener_candidates", lambda n: [{"symbol": "MGN"}])
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [{"symbol": "BTC/USD"}])
    cfg = OmegaConf.create(
        {
            "trading": {"stocks": {"enabled": True}, "crypto": {"enabled": True}, "crypto_taapi_exchange": "binance"},
            "analyst": {"screener_top_n": 20},
            "earnings_blackout": {"enabled": False},
        }
    )
    state = {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": None,
        "stock_market_open": False,
    }

    result = graph.discover_candidates(state, cfg)

    symbols = [c["symbol"] for c in result["raw_candidates"]]
    assert symbols == ["BTC/USD"]


def test_discover_candidates_includes_stocks_when_market_open(monkeypatch):
    monkeypatch.setattr(graph.sources, "fetch_screener_candidates", lambda n, min_price, **kwargs: [{"symbol": "MGN"}])
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [])
    cfg = OmegaConf.create(
        {
            "trading": {"stocks": {"enabled": True}, "crypto": {"enabled": False}, "crypto_taapi_exchange": "binance"},
            "analyst": {"screener_top_n": 20, "min_price_usd": 1.0},
            "earnings_blackout": {"enabled": False},
        }
    )
    state = {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": None,
        "stock_market_open": True,
    }

    result = graph.discover_candidates(state, cfg)

    assert [c["symbol"] for c in result["raw_candidates"]] == ["MGN"]


def test_discover_candidates_drops_earnings_blackout_symbols(monkeypatch):
    monkeypatch.setattr(
        graph.sources,
        "fetch_screener_candidates",
        lambda n, min_price, **kwargs: [{"symbol": "MGN"}, {"symbol": "NVDA"}],
    )
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [])
    monkeypatch.setattr(graph.sources, "fetch_earnings_calendar", lambda symbols, days_before, days_after: {"NVDA"})
    cfg = OmegaConf.create(
        {
            "trading": {"stocks": {"enabled": True}, "crypto": {"enabled": False}, "crypto_taapi_exchange": "binance"},
            "analyst": {"screener_top_n": 20, "min_price_usd": 1.0},
            "earnings_blackout": {"enabled": True, "days_before": 2, "days_after": 1},
        }
    )
    state = {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": None,
        "stock_market_open": True,
    }

    result = graph.discover_candidates(state, cfg)

    assert [c["symbol"] for c in result["raw_candidates"]] == ["MGN"]


def test_discover_candidates_skips_earnings_filter_when_disabled_via_config(monkeypatch):
    monkeypatch.setattr(
        graph.sources,
        "fetch_screener_candidates",
        lambda n, min_price, **kwargs: [{"symbol": "MGN"}, {"symbol": "NVDA"}],
    )
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [])

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("fetch_earnings_calendar must not be called when earnings_blackout.enabled is False")

    monkeypatch.setattr(graph.sources, "fetch_earnings_calendar", _fail_if_called)
    cfg = OmegaConf.create(
        {
            "trading": {"stocks": {"enabled": True}, "crypto": {"enabled": False}, "crypto_taapi_exchange": "binance"},
            "analyst": {"screener_top_n": 20, "min_price_usd": 1.0},
            "earnings_blackout": {"enabled": False},
        }
    )
    state = {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": None,
        "stock_market_open": True,
    }

    result = graph.discover_candidates(state, cfg)

    assert {c["symbol"] for c in result["raw_candidates"]} == {"MGN", "NVDA"}


def _mix_cfg(
    large_cap_pct=0.4,
    crypto_pct=0.3,
    screener_pct=0.3,
    pool_size=10,
    enable_crypto=True,
    earnings_blackout_enabled=False,
    mix_enabled=True,
    large_cap_symbols=None,
):
    return OmegaConf.create(
        {
            "trading": {"stocks": {"enabled": True}, "crypto": {"enabled": enable_crypto}, "crypto_taapi_exchange": "binance"},
            "analyst": {
                "screener_top_n": 20,
                "min_price_usd": 1.0,
                "large_cap_symbols": large_cap_symbols if large_cap_symbols is not None else ["AAPL", "NVDA", "MSFT", "TSLA"],
                "candidate_mix": {
                    "enabled": mix_enabled,
                    "pool_size": pool_size,
                    "large_cap_pct": large_cap_pct,
                    "crypto_pct": crypto_pct,
                    "screener_pct": screener_pct,
                },
            },
            "earnings_blackout": {"enabled": earnings_blackout_enabled, "days_before": 2, "days_after": 1},
        }
    )


def _discover_state():
    return {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": None,
        "stock_market_open": True,
    }


def test_allocate_bucket_counts_splits_proportionally_and_sums_to_pool_size():
    counts = graph._allocate_bucket_counts(20, {"large_cap": 0.4, "crypto": 0.3, "screener": 0.3})

    assert counts == {"large_cap": 8, "crypto": 6, "screener": 6}


def test_allocate_bucket_counts_uses_largest_remainder_to_hit_the_exact_total():
    counts = graph._allocate_bucket_counts(7, {"large_cap": 0.4, "crypto": 0.3, "screener": 0.3})

    assert sum(counts.values()) == 7
    assert counts["large_cap"] == 3  # 0.4*7=2.8 -- largest fractional remainder gets the rounding bump


def test_allocate_bucket_counts_zero_weight_bucket_gets_zero_slots():
    counts = graph._allocate_bucket_counts(10, {"large_cap": 0.4, "crypto": 0.0, "screener": 0.3})

    assert counts["crypto"] == 0
    assert sum(counts.values()) == 10


def test_fill_buckets_with_backfill_takes_target_count_when_pools_have_enough_supply():
    pools = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1", "c2", "c3", "c4"]}

    selected = graph._fill_buckets_with_backfill(pools, {"a": 2, "b": 2, "c": 2})

    assert selected == {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]}


def test_fill_buckets_with_backfill_redistributes_shortfall_from_a_thin_bucket():
    pools = {"a": ["a1"], "b": ["b1", "b2"], "c": ["c1", "c2", "c3", "c4", "c5"]}

    selected = graph._fill_buckets_with_backfill(pools, {"a": 2, "b": 2, "c": 2})

    assert selected["a"] == ["a1"]
    assert sum(len(v) for v in selected.values()) == 6  # a's shortfall (1) backfilled from c's spare


def test_fill_buckets_with_backfill_shrinks_the_pool_when_total_supply_is_insufficient():
    pools = {"a": ["a1"], "b": [], "c": ["c1"]}

    selected = graph._fill_buckets_with_backfill(pools, {"a": 2, "b": 2, "c": 2})

    assert sum(len(v) for v in selected.values()) == 2


def test_discover_candidates_mix_composes_pool_by_configured_ratios(monkeypatch):
    monkeypatch.setattr(graph.sources, "fetch_large_cap_candidates", lambda symbols: [{"symbol": s} for s in symbols])
    monkeypatch.setattr(
        graph.sources,
        "fetch_screener_candidates",
        lambda n, min_price, **kwargs: [
            {"symbol": "W1", "change_pct": 5.0},
            {"symbol": "W2", "change_pct": 30.0},
            {"symbol": "W3", "change_pct": 10.0},
            {"symbol": "W4", "change_pct": 2.0},
            {"symbol": "W5", "change_pct": 20.0},
            {"symbol": "W6", "change_pct": 1.0},
        ],
    )
    monkeypatch.setattr(
        graph.sources,
        "fetch_crypto_candidates",
        lambda n: [
            {"symbol": "BTC/USD", "change_pct": 3.0},
            {"symbol": "ETH/USD", "change_pct": 15.0},
            {"symbol": "SOL/USD", "change_pct": 8.0},
            {"symbol": "DOGE/USD", "change_pct": 1.0},
        ],
    )

    result = graph.discover_candidates(_discover_state(), _mix_cfg(pool_size=10))

    by_market = {}
    for c in result["raw_candidates"]:
        by_market.setdefault(c["market"], []).append(c["symbol"])
    assert sorted(by_market["stocks"]) == sorted(["AAPL", "NVDA", "MSFT", "TSLA", "W2", "W5", "W3"])
    assert sorted(by_market["binance"]) == sorted(["ETH/USD", "SOL/USD", "BTC/USD"])
    assert len(result["raw_candidates"]) == 10


def test_discover_candidates_mix_excludes_large_cap_symbols_from_the_screener_bucket(monkeypatch):
    """A symbol on both the large-cap list and today's screener movers (e.g. NVDA having a
    genuinely huge day) belongs to the large-cap bucket -- it must not also be fetched/tagged as
    a screener candidate and counted twice."""
    monkeypatch.setattr(graph.sources, "fetch_large_cap_candidates", lambda symbols: [{"symbol": s} for s in symbols])
    monkeypatch.setattr(
        graph.sources,
        "fetch_screener_candidates",
        lambda n, min_price, **kwargs: [{"symbol": "NVDA", "change_pct": 50.0}, {"symbol": "W1", "change_pct": 5.0}],
    )
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [])

    result = graph.discover_candidates(
        _discover_state(), _mix_cfg(pool_size=6, large_cap_pct=0.5, crypto_pct=0.0, screener_pct=0.5, enable_crypto=False)
    )

    nvda_entries = [c for c in result["raw_candidates"] if c["symbol"] == "NVDA"]
    assert len(nvda_entries) == 1


def test_discover_candidates_mix_applies_earnings_blackout_to_stock_buckets(monkeypatch):
    monkeypatch.setattr(graph.sources, "fetch_large_cap_candidates", lambda symbols: [{"symbol": s} for s in symbols])
    monkeypatch.setattr(
        graph.sources, "fetch_screener_candidates", lambda n, min_price, **kwargs: [{"symbol": "W1", "change_pct": 5.0}]
    )
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [])
    monkeypatch.setattr(graph.sources, "fetch_earnings_calendar", lambda symbols, days_before, days_after: {"AAPL"})

    result = graph.discover_candidates(
        _discover_state(),
        _mix_cfg(pool_size=6, large_cap_pct=0.5, crypto_pct=0.0, screener_pct=0.5, enable_crypto=False, earnings_blackout_enabled=True),
    )

    assert "AAPL" not in {c["symbol"] for c in result["raw_candidates"]}


def test_discover_candidates_mix_redistributes_crypto_weight_when_crypto_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(graph.sources, "fetch_large_cap_candidates", lambda symbols: [{"symbol": s} for s in symbols])
    monkeypatch.setattr(
        graph.sources,
        "fetch_screener_candidates",
        lambda n, min_price, **kwargs: [{"symbol": f"W{i}", "change_pct": float(i)} for i in range(1, 10)],
    )
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: calls.append(1) or [])

    result = graph.discover_candidates(_discover_state(), _mix_cfg(pool_size=10, enable_crypto=False))

    assert calls == []
    assert len(result["raw_candidates"]) == 10  # crypto's share was redistributed, not left empty


def test_discover_candidates_mix_disabled_falls_back_to_legacy_behavior(monkeypatch):
    monkeypatch.setattr(graph.sources, "fetch_screener_candidates", lambda n, min_price, **kwargs: [{"symbol": "MGN"}])
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [])

    def _fail_if_called(symbols):
        raise AssertionError("fetch_large_cap_candidates must not be called when candidate_mix is disabled")

    monkeypatch.setattr(graph.sources, "fetch_large_cap_candidates", _fail_if_called)

    result = graph.discover_candidates(_discover_state(), _mix_cfg(mix_enabled=False))

    assert [c["symbol"] for c in result["raw_candidates"]] == ["MGN"]


def test_discover_candidates_mix_only_applies_when_stock_market_is_open(monkeypatch):
    """A market-closed day must still fall back to the existing crypto-only branch, unmixed -- the
    mix's whole purpose is fixing a stock-screener-ranking problem that doesn't exist when the
    stock branch isn't running at all."""

    def _fail_if_called(symbols):
        raise AssertionError("fetch_large_cap_candidates must not be called when the market is closed")

    monkeypatch.setattr(graph.sources, "fetch_large_cap_candidates", _fail_if_called)
    monkeypatch.setattr(graph.sources, "fetch_crypto_candidates", lambda n: [{"symbol": "BTC/USD"}])

    state = {**_discover_state(), "stock_market_open": False}
    result = graph.discover_candidates(state, _mix_cfg())

    assert [c["symbol"] for c in result["raw_candidates"]] == ["BTC/USD"]


class FakeAccountForPortfolio:
    def __init__(self):
        self.equity = "1000.00"
        self.cash = "500.00"
        self.buying_power = "2000.00"


class FakeAccountClient:
    def get_account(self):
        return FakeAccountForPortfolio()


def test_write_portfolio_passes_market_status_to_slack(monkeypatch):
    monkeypatch.setattr(graph, "_write_portfolio", lambda payload: None)
    monkeypatch.setattr(graph, "trading_client", FakeAccountClient())
    captured = {}
    monkeypatch.setattr(
        graph.slack, "notify_morning_report", lambda *a, **k: captured.update(args=a, kwargs=k)
    )
    state = {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": {"symbols": []},
        "stock_market_open": False,
    }
    cfg = OmegaConf.create({"trading": {"crypto": {"enabled": True}}})

    graph.write_portfolio(state, cfg)

    assert captured["kwargs"]["stock_market_open"] is False
    assert captured["kwargs"]["crypto_enabled"] is True
    assert captured["kwargs"]["title"] == "Morning Market Report"
    assert captured["kwargs"]["emoji"] == "🌅"


def test_write_portfolio_uses_midday_title_and_emoji_on_a_midday_run(monkeypatch):
    monkeypatch.setattr(graph, "_write_portfolio", lambda payload: None)
    monkeypatch.setattr(graph, "trading_client", FakeAccountClient())
    captured = {}
    monkeypatch.setattr(
        graph.slack, "notify_morning_report", lambda *a, **k: captured.update(args=a, kwargs=k)
    )
    state = {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "selection": {"symbols": []},
        "stock_market_open": True,
        "is_midday_run": True,
    }
    cfg = OmegaConf.create({"trading": {"crypto": {"enabled": True}}})

    graph.write_portfolio(state, cfg)

    assert captured["kwargs"]["title"] == "Midday Update"
    assert captured["kwargs"]["emoji"] == "🕐"


def _budget_cfg(max_total_budget_usd=1_000_000.0):
    return OmegaConf.create({"analyst": {"max_total_budget_usd": max_total_budget_usd}})


def test_validate_selection_overrides_llm_exchange_with_known_market():
    """The LLM sometimes echoes back the wrong exchange for a pick -- validate_selection must
    trust discover_candidates()'s own `market` tag, not the LLM's copy of the field."""
    state = _state(
        raw_candidates=[{"symbol": "BTC/USD", "market": "binance"}],
        picks=[{"symbol": "BTC/USD", "exchange": "stocks", "budget": 100.0}],
    )

    result = validate_selection(state, _budget_cfg())

    assert result["selection"]["symbols"] == [{"symbol": "BTC/USD", "exchange": "binance", "budget": 100.0}]


def test_validate_selection_drops_hallucinated_pick():
    """A pick whose symbol was never offered in raw_candidates is a hallucination and must be
    dropped rather than written to the portfolio with a guessed exchange."""
    state = _state(
        raw_candidates=[{"symbol": "MGN", "market": "stocks"}],
        picks=[
            {"symbol": "MGN", "exchange": "stocks", "budget": 100.0},
            {"symbol": "FAKE", "exchange": "stocks", "budget": 100.0},
        ],
    )

    result = validate_selection(state, _budget_cfg())

    assert [pick["symbol"] for pick in result["selection"]["symbols"]] == ["MGN"]


def test_validate_selection_keeps_picks_within_the_total_budget_cap():
    state = _state(
        raw_candidates=[{"symbol": "A", "market": "stocks"}, {"symbol": "B", "market": "stocks"}],
        picks=[
            {"symbol": "A", "exchange": "stocks", "budget": 5000.0},
            {"symbol": "B", "exchange": "stocks", "budget": 5000.0},
        ],
    )

    result = validate_selection(state, _budget_cfg(max_total_budget_usd=50000.0))

    assert [pick["symbol"] for pick in result["selection"]["symbols"]] == ["A", "B"]


def test_validate_selection_drops_trailing_picks_that_would_exceed_the_total_budget_cap():
    """Per-pick budget has no upper bound of its own (src/analyst/schema.py) -- a selection that
    sums past analyst.max_total_budget_usd must be trimmed, not written through as-is. Picks are
    considered in the LLM's own returned order; a later pick that doesn't fit is dropped even if
    an even-later, smaller pick would have fit."""
    state = _state(
        raw_candidates=[
            {"symbol": "A", "market": "stocks"},
            {"symbol": "B", "market": "stocks"},
            {"symbol": "C", "market": "stocks"},
        ],
        picks=[
            {"symbol": "A", "exchange": "stocks", "budget": 40000.0},
            {"symbol": "B", "exchange": "stocks", "budget": 20000.0},
            {"symbol": "C", "exchange": "stocks", "budget": 5000.0},
        ],
    )

    result = validate_selection(state, _budget_cfg(max_total_budget_usd=50000.0))

    assert [pick["symbol"] for pick in result["selection"]["symbols"]] == ["A", "C"]


class FakePosition:
    def __init__(
        self,
        symbol,
        qty,
        market_value,
        unrealized_plpc,
        asset_class,
        avg_entry_price="0",
        unrealized_pl="0",
        current_price="0",
    ):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.unrealized_plpc = unrealized_plpc
        self.asset_class = asset_class
        self.avg_entry_price = avg_entry_price
        self.unrealized_pl = unrealized_pl
        self.current_price = current_price


class FakeTradingClient:
    def __init__(self, positions, activities=None):
        self._positions = positions
        self._activities = activities or []

    def get_all_positions(self):
        return self._positions

    def get(self, path, data=None):
        return self._activities


def test_crypto_eod_report_skipped_when_crypto_disabled(monkeypatch):
    monkeypatch.setattr(graph, "trading_client", FakeTradingClient([]))
    posted = {}
    monkeypatch.setattr(graph.slack, "notify_crypto_eod_report", lambda *a, **k: posted.setdefault("called", True))

    crypto_eod_report({}, _cfg(enable_crypto=False))

    assert "called" not in posted


def test_crypto_eod_report_skipped_on_midday_run(monkeypatch):
    """A midday run already had its crypto EOD report posted this morning -- covers only the
    prior calendar day, so a second post on the same day would be a verbatim duplicate."""
    monkeypatch.setattr(graph, "trading_client", FakeTradingClient([]))
    posted = {}
    monkeypatch.setattr(graph.slack, "notify_crypto_eod_report", lambda *a, **k: posted.setdefault("called", True))

    crypto_eod_report({"is_midday_run": True}, _cfg(enable_crypto=True))

    assert "called" not in posted


def test_crypto_eod_report_posts_only_crypto_positions_and_fills_for_the_prior_day(monkeypatch):
    """Crypto trades 24/7 -- this rides along with the morning report and must cover the prior
    full ET calendar day, filtered to crypto-only positions/fills (equities are the stock EOD
    CronJob's job, not this one's)."""
    # Alpaca's live Position.symbol for crypto has no slash (e.g. "BTCUSD"), unlike fill/activity
    # records ("BTC/USD") -- filtering must key off asset_class, not symbol shape.
    positions = [
        FakePosition("MGN", "3", "150.00", "0.05", AssetClass.US_EQUITY),
        FakePosition("BTCUSD", "0.01", "600.00", "-0.02", AssetClass.CRYPTO),
    ]
    activities = [
        {"symbol": "MGN", "side": "buy", "qty": "1", "price": "100.00", "transaction_time": "2026-08-02T14:00:00Z"},
        {
            "symbol": "BTC/USD",
            "side": "sell",
            "qty": "0.005",
            "price": "60000.00",
            "transaction_time": "2026-08-02T15:00:00Z",
        },
    ]
    fake_client = FakeTradingClient(positions, activities)
    monkeypatch.setattr(graph, "trading_client", fake_client)
    monkeypatch.setattr(eod, "trading_client", fake_client)
    posted = {}
    monkeypatch.setattr(
        graph.slack,
        "notify_crypto_eod_report",
        lambda report_date, fills, positions: posted.update(report_date=report_date, fills=fills, positions=positions),
    )

    crypto_eod_report({}, _cfg(enable_crypto=True))

    expected_date = (datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=1)).date().isoformat()
    assert posted["report_date"] == expected_date
    assert [f["symbol"] for f in posted["fills"]] == ["BTC/USD"]
    assert [p["symbol"] for p in posted["positions"]] == ["BTCUSD"]


def _indicator_cfg(indicator_fetch_limit, enable_indicators=True, large_cap_symbols=None):
    return OmegaConf.create(
        {
            "analyst": {
                "indicator_fetch_limit": indicator_fetch_limit,
                "indicators": {"enabled": enable_indicators},
                "large_cap_symbols": large_cap_symbols or [],
            },
            "taapi": {"min_request_interval_secs": 15},
            "indicators": [],
        }
    )


def test_fetch_indicators_sorts_by_change_pct_and_respects_the_fetch_limit(monkeypatch):
    """Only the top `indicator_fetch_limit` candidates by |change_pct| get a real TAAPI call --
    fetching every screened candidate would blow past the 15s/request rate limit for candidates
    the LLM is unlikely to pick anyway. A missing change_pct (e.g. the crypto watchlist) must
    sort last, not crash the comparison."""
    candidates = [
        {"symbol": "A", "market": "stocks", "change_pct": 1.0},
        {"symbol": "B", "market": "stocks", "change_pct": -5.0},
        {"symbol": "C", "market": "stocks", "change_pct": 2.0},
        {"symbol": "D", "market": "stocks"},
    ]
    calls = []
    monkeypatch.setattr(
        graph,
        "fetch_indicators_bulk",
        lambda indicators_cfg, symbol, exchange, names, log: calls.append(symbol) or f"{symbol}-text",
    )
    sleeps = []
    monkeypatch.setattr(graph.time, "sleep", lambda s: sleeps.append(s))
    state = {"raw_candidates": candidates, "research_text": "", "indicator_text": "", "selection": None}

    result = graph.fetch_indicators(state, _indicator_cfg(indicator_fetch_limit=2))

    assert calls == ["B", "C"]
    assert sleeps == [15]
    assert "B-text" in result["indicator_text"]
    assert "C-text" in result["indicator_text"]
    assert "A-text" not in result["indicator_text"]
    assert "D-text" not in result["indicator_text"]


def test_fetch_indicators_guarantees_coverage_for_large_cap_symbols_outside_the_ranked_cutoff(monkeypatch):
    """AAPL has no change_pct (sorts last) and would normally fall outside indicator_fetch_limit
    -- being on the large-cap list must still get it a real indicator fetch, not just a
    raw_candidates entry with no numbers behind it."""
    candidates = [
        {"symbol": "A", "market": "stocks", "change_pct": 1.0},
        {"symbol": "B", "market": "stocks", "change_pct": -5.0},
        {"symbol": "AAPL", "market": "stocks"},
    ]
    calls = []
    monkeypatch.setattr(
        graph,
        "fetch_indicators_bulk",
        lambda indicators_cfg, symbol, exchange, names, log: calls.append(symbol) or f"{symbol}-text",
    )
    monkeypatch.setattr(graph.time, "sleep", lambda s: None)
    state = {"raw_candidates": candidates, "research_text": "", "indicator_text": "", "selection": None}

    result = graph.fetch_indicators(state, _indicator_cfg(indicator_fetch_limit=1, large_cap_symbols=["AAPL"]))

    assert calls == ["B", "AAPL"]
    assert "AAPL-text" in result["indicator_text"]


def test_fetch_indicators_large_cap_symbol_already_in_top_n_is_not_fetched_twice(monkeypatch):
    candidates = [
        {"symbol": "AAPL", "market": "stocks", "change_pct": 50.0},
        {"symbol": "B", "market": "stocks", "change_pct": 1.0},
    ]
    calls = []
    monkeypatch.setattr(
        graph,
        "fetch_indicators_bulk",
        lambda indicators_cfg, symbol, exchange, names, log: calls.append(symbol) or f"{symbol}-text",
    )
    monkeypatch.setattr(graph.time, "sleep", lambda s: None)
    state = {"raw_candidates": candidates, "research_text": "", "indicator_text": "", "selection": None}

    graph.fetch_indicators(state, _indicator_cfg(indicator_fetch_limit=1, large_cap_symbols=["AAPL"]))

    assert calls == ["AAPL"]


def test_fetch_indicators_empty_large_cap_list_matches_todays_behavior(monkeypatch):
    candidates = [
        {"symbol": "A", "market": "stocks", "change_pct": 1.0},
        {"symbol": "B", "market": "stocks", "change_pct": -5.0},
    ]
    calls = []
    monkeypatch.setattr(
        graph,
        "fetch_indicators_bulk",
        lambda indicators_cfg, symbol, exchange, names, log: calls.append(symbol) or f"{symbol}-text",
    )
    monkeypatch.setattr(graph.time, "sleep", lambda s: None)
    state = {"raw_candidates": candidates, "research_text": "", "indicator_text": "", "selection": None}

    graph.fetch_indicators(state, _indicator_cfg(indicator_fetch_limit=1))

    assert calls == ["B"]


def test_fetch_indicators_omits_candidates_with_no_indicator_data(monkeypatch):
    candidates = [{"symbol": "A", "market": "stocks", "change_pct": 1.0}]
    monkeypatch.setattr(graph, "fetch_indicators_bulk", lambda *a, **k: "")
    monkeypatch.setattr(graph.time, "sleep", lambda s: None)
    state = {"raw_candidates": candidates, "research_text": "", "indicator_text": "", "selection": None}

    result = graph.fetch_indicators(state, _indicator_cfg(indicator_fetch_limit=5))

    assert result["indicator_text"] == ""


def test_fetch_indicators_skipped_when_disabled_via_config(monkeypatch):
    """The enable_indicators feature gate must short-circuit before any TAAPI calls, not just
    filter the result -- fetch_indicators_bulk being called at all would still burn the rate limit."""
    candidates = [{"symbol": "A", "market": "stocks", "change_pct": 1.0}]
    calls = []
    monkeypatch.setattr(graph, "fetch_indicators_bulk", lambda *a, **k: calls.append(1) or "text")
    state = {"raw_candidates": candidates, "research_text": "", "indicator_text": "", "selection": None}

    result = graph.fetch_indicators(state, _indicator_cfg(indicator_fetch_limit=5, enable_indicators=False))

    assert result["indicator_text"] == ""
    assert calls == []


def test_fetch_research_includes_news_and_headlines_when_enabled(monkeypatch):
    monkeypatch.setattr(graph.sources, "fetch_news", lambda days: "MGN announces earnings beat")
    monkeypatch.setattr(graph.sources, "fetch_yahoo_rss_headlines", lambda url: "Market rallies on Fed news")
    cfg = OmegaConf.create({"analyst": {"news_days": 2, "yahoo_rss_url": "http://x", "news": {"enabled": True}}})
    state = {"raw_candidates": [], "research_text": "", "indicator_text": "", "selection": None}

    result = graph.fetch_research(state, cfg)

    assert "MGN announces earnings beat" in result["research_text"]
    assert "Market rallies on Fed news" in result["research_text"]


def test_fetch_research_skipped_when_disabled_via_config(monkeypatch):
    """The enable_news feature gate must short-circuit before any network calls, not just discard
    the result."""
    calls = []
    monkeypatch.setattr(graph.sources, "fetch_news", lambda days: calls.append(1) or "news")
    monkeypatch.setattr(graph.sources, "fetch_yahoo_rss_headlines", lambda url: calls.append(1) or "headlines")
    cfg = OmegaConf.create({"analyst": {"news_days": 2, "yahoo_rss_url": "http://x", "news": {"enabled": False}}})
    state = {"raw_candidates": [], "research_text": "", "indicator_text": "", "selection": None}

    result = graph.fetch_research(state, cfg)

    assert result["research_text"] == ""
    assert calls == []


class FakeSelection:
    def __init__(self, symbols):
        self._symbols = symbols

    def model_dump(self):
        return {"symbols": self._symbols}


class FakeLLM:
    def __init__(self, captured):
        self._captured = captured

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self._captured["messages"] = messages
        return FakeSelection([])


def test_llm_select_prompt_includes_indicator_text_and_research_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(graph, "ChatOpenAI", lambda **kwargs: FakeLLM(captured))
    cfg = OmegaConf.create(
        {
            "analyst": {
                "max_universe_size": 10,
                "default_budget": 5000,
                "indicator_fetch_limit": 15,
                "track_record_days": 5,
            },
            "llm": {"base_url": "http://x", "model": "m", "temperature": 0.1},
        }
    )
    state = {
        "raw_candidates": [{"symbol": "MGN", "market": "stocks"}],
        "research_text": "MGN announces earnings beat",
        "indicator_text": "MGN:\nThe current RSI for MGN is 71.2",
        "track_record_text": "",
        "pnl_text": "",
        "selection": None,
    }

    graph.llm_select(state, cfg)

    user_content = captured["messages"][1].content
    assert "The current RSI for MGN is 71.2" in user_content
    assert "MGN announces earnings beat" in user_content


def test_llm_select_prompt_includes_track_record_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(graph, "ChatOpenAI", lambda **kwargs: FakeLLM(captured))
    cfg = OmegaConf.create(
        {
            "analyst": {
                "max_universe_size": 10,
                "default_budget": 5000,
                "indicator_fetch_limit": 15,
                "track_record_days": 5,
            },
            "llm": {"base_url": "http://x", "model": "m", "temperature": 0.1},
        }
    )
    state = {
        "raw_candidates": [{"symbol": "MGN", "market": "stocks"}],
        "research_text": "",
        "indicator_text": "",
        "track_record_text": "- 2026-08-01 picked MGN (budget $100): momentum play",
        "pnl_text": "",
        "selection": None,
    }

    graph.llm_select(state, cfg)

    user_content = captured["messages"][1].content
    assert "- 2026-08-01 picked MGN (budget $100): momentum play" in user_content


def test_llm_select_prompt_includes_pnl_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(graph, "ChatOpenAI", lambda **kwargs: FakeLLM(captured))
    cfg = OmegaConf.create(
        {
            "analyst": {
                "max_universe_size": 10,
                "default_budget": 5000,
                "indicator_fetch_limit": 15,
                "track_record_days": 5,
            },
            "llm": {"base_url": "http://x", "model": "m", "temperature": 0.1},
        }
    )
    state = {
        "raw_candidates": [{"symbol": "MGN", "market": "stocks"}],
        "research_text": "",
        "indicator_text": "",
        "track_record_text": "",
        "pnl_text": "- MGN: qty 3, avg entry $45.00, current $50.00, unrealized +$15.00 (+11.11%)",
        "selection": None,
    }

    graph.llm_select(state, cfg)

    user_content = captured["messages"][1].content
    assert "- MGN: qty 3, avg entry $45.00, current $50.00, unrealized +$15.00 (+11.11%)" in user_content


def test_llm_select_sets_request_timeout_and_no_retries(monkeypatch):
    """The Analyst shares the DGX Ollama host with the Dealer; a hung structured-output call must
    fail on a per-request wall-clock ceiling and not retry into it -- same as the Dealer's
    llm_call (see the GB10 silent-hang incident)."""
    captured = {}

    def _capture_chat_openai(**kwargs):
        captured.update(kwargs)
        return FakeLLM({})

    monkeypatch.setattr(graph, "ChatOpenAI", _capture_chat_openai)
    cfg = OmegaConf.create(
        {
            "analyst": {
                "max_universe_size": 10,
                "default_budget": 5000,
                "indicator_fetch_limit": 15,
                "track_record_days": 5,
            },
            "llm": {"base_url": "http://x", "model": "m", "temperature": 0.1, "request_timeout_s": 90},
        }
    )
    state = {
        "raw_candidates": [{"symbol": "MGN", "market": "stocks"}],
        "research_text": "",
        "indicator_text": "",
        "track_record_text": "",
        "pnl_text": "",
        "selection": None,
    }

    graph.llm_select(state, cfg)

    assert captured["timeout"] == 90.0
    assert captured["max_retries"] == 0


def test_analyst_llm_timeout_defaults_to_120_when_unset():
    cfg = OmegaConf.create({"llm": {"base_url": "http://x", "model": "m", "temperature": 0.1}})
    assert graph._llm_timeout(cfg) == 120.0


def test_build_graph_does_not_capture_config_at_build_time(monkeypatch):
    """Every node must read config per-invocation (like the Dealer/Floor Broker hot paths), so a
    mid-run config change to analyst.news / analyst.indicators / etc. is honored rather than
    frozen at build_graph() time."""
    calls = []

    def _tracked_load_config():
        calls.append(1)
        return OmegaConf.create({})

    monkeypatch.setattr(graph, "load_config", _tracked_load_config)

    graph.build_graph()

    assert calls == []


def _track_record_cfg(track_record_days, enable_track_record=True):
    return OmegaConf.create(
        {"analyst": {"track_record_days": track_record_days, "track_record": {"enabled": enable_track_record}}}
    )


def test_fetch_track_record_skipped_when_disabled_via_config(monkeypatch):
    """The enable_track_record feature gate must short-circuit before any DB calls, not just
    discard the result -- mirrors the enable_news/enable_indicators skip tests."""
    calls = []
    monkeypatch.setattr(graph.db, "fetch_analyst_picks_since", lambda since: calls.append(1) or [])
    monkeypatch.setattr(graph.db, "fetch_dealer_decisions_since", lambda since: calls.append(1) or [])
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since: calls.append(1) or [])
    state = {"raw_candidates": [], "research_text": "", "indicator_text": "", "track_record_text": "", "selection": None}

    result = graph.fetch_track_record(state, _track_record_cfg(track_record_days=5, enable_track_record=False))

    assert result["track_record_text"] == ""
    assert calls == []


def test_fetch_track_record_returns_empty_when_no_prior_picks(monkeypatch):
    monkeypatch.setattr(graph.db, "fetch_analyst_picks_since", lambda since: [])
    calls = []
    monkeypatch.setattr(graph.db, "fetch_dealer_decisions_since", lambda since: calls.append(1) or [])
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since: calls.append(1) or [])
    state = {"raw_candidates": [], "research_text": "", "indicator_text": "", "track_record_text": "", "selection": None}

    result = graph.fetch_track_record(state, _track_record_cfg(track_record_days=5))

    assert result["track_record_text"] == ""
    assert calls == []


def test_fetch_track_record_includes_pick_history_when_enabled(monkeypatch):
    picks = [
        {
            "symbol": "MGN",
            "generated_at": datetime(2026, 8, 1, 9, 30),
            "budget": 100.0,
            "rationale": "momentum play",
        }
    ]
    decisions = [
        {
            "symbol": "MGN",
            "decided_at": datetime(2026, 8, 1, 10, 0),
            "action": "SELL",
            "reasoning": "hit stop loss",
        }
    ]
    events = [
        {
            "symbol": "MGN",
            "occurred_at": datetime(2026, 8, 1, 10, 1),
            "event_type": "sell_filled",
            "price": 4.5,
            "detail": "stop loss triggered",
        }
    ]
    monkeypatch.setattr(graph.db, "fetch_analyst_picks_since", lambda since: picks)
    monkeypatch.setattr(graph.db, "fetch_dealer_decisions_since", lambda since: decisions)
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since: events)
    state = {"raw_candidates": [], "research_text": "", "indicator_text": "", "track_record_text": "", "selection": None}

    result = graph.fetch_track_record(state, _track_record_cfg(track_record_days=5))

    assert "momentum play" in result["track_record_text"]
    assert "hit stop loss" in result["track_record_text"]
    assert "sell_filled" in result["track_record_text"]
    assert "stop loss triggered" in result["track_record_text"]


def _pnl_cfg(enable_position_pnl=True):
    return OmegaConf.create({"analyst": {"position_pnl": {"enabled": enable_position_pnl}}})


def _pnl_state():
    return {
        "raw_candidates": [],
        "research_text": "",
        "indicator_text": "",
        "track_record_text": "",
        "pnl_text": "",
        "selection": None,
    }


def test_fetch_position_pnl_skipped_when_disabled_via_config(monkeypatch):
    """The enable_position_pnl feature gate must short-circuit before any Alpaca call, not just
    discard the result -- mirrors the enable_news/enable_indicators/enable_track_record skip tests."""
    calls = []

    class TrackingClient:
        def get_all_positions(self):
            calls.append(1)
            return []

    monkeypatch.setattr(graph, "trading_client", TrackingClient())

    result = graph.fetch_position_pnl(_pnl_state(), _pnl_cfg(enable_position_pnl=False))

    assert result["pnl_text"] == ""
    assert calls == []


def test_fetch_position_pnl_returns_empty_when_no_open_positions(monkeypatch):
    monkeypatch.setattr(graph, "trading_client", FakeTradingClient([]))

    result = graph.fetch_position_pnl(_pnl_state(), _pnl_cfg())

    assert result["pnl_text"] == ""


def test_fetch_position_pnl_includes_positions_when_enabled(monkeypatch):
    positions = [
        FakePosition(
            "MGN", "3", "150.00", "0.1667", AssetClass.US_EQUITY,
            avg_entry_price="45.00", unrealized_pl="15.00", current_price="50.00",
        ),
        FakePosition(
            "BTCUSD", "0.01", "600.00", "-0.02", AssetClass.CRYPTO,
            avg_entry_price="61000.00", unrealized_pl="-10.00", current_price="60000.00",
        ),
    ]
    monkeypatch.setattr(graph, "trading_client", FakeTradingClient(positions))

    result = graph.fetch_position_pnl(_pnl_state(), _pnl_cfg())

    assert "MGN" in result["pnl_text"]
    assert "avg entry $45.00" in result["pnl_text"]
    assert "+$15.00" in result["pnl_text"]
    assert "BTCUSD" in result["pnl_text"]
    assert "-$10.00" in result["pnl_text"]


def test_fetch_position_pnl_returns_empty_when_alpaca_call_fails(monkeypatch):
    """Fails open, not closed -- this is supplementary context (same risk category as news/
    track-record), not a trading gate, so a transient Alpaca API error must degrade to an empty
    snapshot for this run rather than raising out of the node and failing the whole Analyst run."""

    class FailingClient:
        def get_all_positions(self):
            raise RuntimeError("Alpaca API unavailable")

    monkeypatch.setattr(graph, "trading_client", FailingClient())

    result = graph.fetch_position_pnl(_pnl_state(), _pnl_cfg())

    assert result["pnl_text"] == ""


def test_fetch_position_pnl_handles_none_unrealized_pl_and_current_price(monkeypatch):
    """unrealized_pl and current_price are Optional[str] on Alpaca's own Position model -- a
    missing value on one position must render as "n/a" rather than crashing the whole snapshot."""
    positions = [
        FakePosition(
            "MGN", "3", "150.00", "0.05", AssetClass.US_EQUITY,
            avg_entry_price="45.00", unrealized_pl=None, current_price=None,
        ),
    ]
    monkeypatch.setattr(graph, "trading_client", FakeTradingClient(positions))

    result = graph.fetch_position_pnl(_pnl_state(), _pnl_cfg())

    assert "MGN" in result["pnl_text"]
    assert "n/a" in result["pnl_text"]
