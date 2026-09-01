import asyncio
import json
from datetime import date, datetime, timedelta

import pytz
from omegaconf import OmegaConf

from src.dealer import graph


def _state(indicators_text: str) -> dict:
    return {
        "symbol": "CRV/USD",
        "exchange": "binance",
        "budget": 100.0,
        "indicator_names": ["ALL"],
        "indicators_text": indicators_text,
        "cycle_id": "cycle-1",
        "raw_bars": {},
        "ohlcv_features_text": "",
        "signal": None,
        "execution_result": None,
    }


def test_route_after_indicators_skips_llm_when_indicators_text_is_empty():
    assert graph._route_after_indicators(_state("")) == "skip_missing_indicators"


def test_route_after_indicators_skips_llm_when_indicators_text_is_whitespace_only():
    assert graph._route_after_indicators(_state("   \n")) == "skip_missing_indicators"


def test_route_after_indicators_calls_llm_when_indicators_text_present():
    assert graph._route_after_indicators(_state("rsi: 71.2")) == "llm_call"


def test_skip_missing_indicators_records_hold_without_calling_llm(monkeypatch):
    """Regression: CRV/USD, WIF/USD, LDO/USD intermittently got empty indicators_text back from
    TAAPI, and the old unconditional fetch_indicators -> llm_call edge let the LLM improvise a
    "please provide indicators" HOLD instead of a real trading decision. This node must record a
    diagnostic HOLD and skip the cycle without ever constructing an LLM client."""

    def _fail_if_called(**kwargs):
        raise AssertionError("LLM must not be called when indicators are missing")

    monkeypatch.setattr(graph, "ChatOpenAI", _fail_if_called)
    recorded = {}
    monkeypatch.setattr(graph.slack, "notify_dealer_signal", lambda *a, **k: recorded.setdefault("slack", a))
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: recorded.setdefault("db", (a, k)))

    result = graph.skip_missing_indicators(_state(""), cfg=None)

    assert result["signal"]["action"] == "HOLD"
    assert result["execution_result"] == {"status": "skipped", "detail": "missing_indicators"}
    assert recorded["slack"][1] == "HOLD"
    assert recorded["db"][0][1] == "HOLD"
    assert recorded["db"][1] == {"ohlcv_enrichment_active": False, "cycle_id": "cycle-1"}


def test_fetch_market_data_is_noop_when_disabled(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("disabled OHLCV enrichment must not fetch bars")

    monkeypatch.setattr(graph, "fetch_multi_timeframe_bars", _fail_if_called)
    cfg = OmegaConf.create({"ohlcv_enrichment": {"enabled": False}})

    result = graph.fetch_market_data(_state("rsi: 71.2"), cfg)

    assert result["raw_bars"] == {}
    assert result["ohlcv_features_text"] == ""


def test_fetch_market_data_populates_stock_features(monkeypatch):
    bars = {"5m": object()}
    monkeypatch.setattr(graph, "fetch_multi_timeframe_bars", lambda symbol, exchange, cfg: bars)
    monkeypatch.setattr(graph, "compute_derived_features", lambda df, cfg: {"latest_close": 10.0})
    monkeypatch.setattr(graph, "format_features_text", lambda features, symbol: "features text")
    cfg = OmegaConf.create({"ohlcv_enrichment": {"enabled": True}})

    result = graph.fetch_market_data({**_state("rsi: 71.2"), "exchange": "stocks"}, cfg)

    assert result["raw_bars"] == bars
    assert result["ohlcv_features_text"] == "features text"


def test_llm_call_includes_ohlcv_features_when_present(monkeypatch):
    captured = {}

    class FakeSignal:
        action = "HOLD"

        def model_dump(self):
            return {"action": "HOLD", "reasoning": "wait", "size_hint": 0.0, "confidence": 0.0}

    class FakeStructured:
        def invoke(self, messages):
            captured["user_prompt"] = messages[-1].content
            return FakeSignal()

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return FakeStructured()

    cfg = OmegaConf.create(
        {
            "llm": {"base_url": "http://llm.test/v1", "model": "test-model", "temperature": 0.0},
            "strategy": {"dealer_memory": {"enabled": False}},
        }
    )
    monkeypatch.setattr(graph, "ChatOpenAI", FakeChatOpenAI)

    graph.llm_call({**_state("rsi: 25"), "ohlcv_features_text": "5m return: 2%"}, cfg)

    assert "Additional OHLCV context" in captured["user_prompt"]
    assert "5m return: 2%" in captured["user_prompt"]


def test_llm_call_includes_recent_same_symbol_memory(monkeypatch):
    captured = {}

    class FakeSignal:
        action = "HOLD"

        def model_dump(self):
            return {"action": "HOLD", "reasoning": "wait", "size_hint": 0.0, "confidence": 0.0}

    class FakeStructured:
        def invoke(self, messages):
            captured["user_prompt"] = messages[-1].content
            return FakeSignal()

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema):
            return FakeStructured()

    cfg = OmegaConf.create(
        {
            "llm": {"base_url": "http://llm.test/v1", "model": "test-model", "temperature": 0.0},
            "strategy": {"dealer_memory": {"enabled": True}, "symbol_memory_days": 2, "symbol_memory_limit": 4},
        }
    )
    monkeypatch.setattr(graph, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        graph.db,
        "fetch_symbol_dealer_decisions_since",
        lambda symbol, since_date, limit: [
            {"decided_at": "t1", "action": "BUY", "size_hint": 0.5, "reasoning": "prior buy"}
        ],
    )
    monkeypatch.setattr(
        graph.db,
        "fetch_symbol_floor_broker_events_since",
        lambda symbol, since_date, limit: [
            {"occurred_at": "t2", "event_type": "fill", "detail": "stop_loss leg filled: o-1"}
        ],
    )

    result = graph.llm_call(_state("rsi: 25"), cfg)

    assert result["signal"]["action"] == "HOLD"
    assert "Recent same-symbol trading history" in captured["user_prompt"]
    assert "prior buy" in captured["user_prompt"]
    assert "stop_loss leg filled" in captured["user_prompt"]


def test_select_option_contract_is_noop_when_disabled(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("disabled options_trading must not call MCP tools")

    monkeypatch.setattr(graph, "_select_option_contract_async", _fail_if_called)
    cfg = OmegaConf.create({"options_trading": {"enabled": False}})
    state = {**_state("rsi: 71.2"), "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"}}

    result = graph.select_option_contract(state, cfg)

    assert result["option_pick"] is None


def test_select_option_contract_is_noop_on_hold(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("a HOLD signal must not trigger option contract search")

    monkeypatch.setattr(graph, "_select_option_contract_async", _fail_if_called)
    cfg = OmegaConf.create(
        {"options_trading": {"enabled": True}, "strategy": {"min_confidence": 0.6}}
    )
    state = {**_state("rsi: 71.2"), "signal": {"action": "HOLD", "confidence": 0.9, "reasoning": "r"}}

    result = graph.select_option_contract(state, cfg)

    assert result["option_pick"] is None
    assert result.get("option_skip") is None  # HOLD is routed away before this node; no skip reason


def test_select_option_contract_records_low_confidence_skip_reason(monkeypatch):
    """A sub-min_confidence BUY/SELL still reaches this node (only HOLD is routed away). The node
    must stash the real reason so call_floor_broker_option() can report low_confidence instead of a
    bare no_option_pick -- matching what the stock path records."""
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("a low-confidence signal must not trigger option contract search")

    monkeypatch.setattr(graph, "_select_option_contract_async", _fail_if_called)
    cfg = OmegaConf.create(
        {"options_trading": {"enabled": True}, "strategy": {"min_confidence": 0.6}}
    )
    state = {**_state("rsi: 71.2"), "signal": {"action": "BUY", "confidence": 0.4, "reasoning": "r"}}

    result = graph.select_option_contract(state, cfg)

    assert result["option_pick"] is None
    assert result["option_skip"]["reason"] == "low_confidence"
    assert result["option_skip"]["detail"] == "BUY confidence 0.40 below minimum 0.6"


def test_select_option_contract_returns_pick_dict(monkeypatch):
    async def _fake_select(state, cfg, signal):
        return graph.OptionContractPick(
            contract_symbol="AAPL250117C00200000",
            strike=200.0,
            expiration="2025-01-17",
            right="call",
            delta=0.45,
            premium=3.20,
            reasoning="within delta/DTE window with sufficient OI",
        )

    monkeypatch.setattr(graph, "_select_option_contract_async", _fake_select)
    cfg = OmegaConf.create(
        {"options_trading": {"enabled": True}, "strategy": {"min_confidence": 0.6}}
    )
    state = {**_state("rsi: 71.2"), "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"}}

    result = graph.select_option_contract(state, cfg)

    assert result["option_pick"]["contract_symbol"] == "AAPL250117C00200000"
    assert result["option_pick"]["right"] == "call"


def test_select_option_contract_async_passes_api_key_and_needs_no_openai_env(monkeypatch):
    """Regression: _select_option_contract_async built ChatOpenAI without api_key, so it fell back
    to demanding OPENAI_API_KEY from the env and every real contract selection died with "Missing
    credentials" before a single tool call. It must pass the same "not-needed" sentinel llm_call()
    uses -- the local model router does no auth -- and work with OPENAI_API_KEY absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return json.dumps(
                {"snapshots": {"AAPL250117C00200000": {"latestQuote": {"bp": 3.0, "ap": 3.4}, "greeks": {"delta": 0.45}}}}
            )

    async def _fake_get_options_tools():
        return [_Tool()]

    class _Bound:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return type("R", (), {"tool_calls": [{"name": "get_option_chain", "args": {}, "id": "c"}]})()
            return type("R", (), {"tool_calls": []})()

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol="AAPL250117C00200000",
                strike=200.0,
                expiration="2025-01-17",
                right="call",
                delta=0.45,
                premium=3.20,
                reasoning="within window",
            )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools):
            return _Bound()

        def with_structured_output(self, schema):
            return _Structured()

    monkeypatch.setattr(graph, "get_options_tools", _fake_get_options_tools)
    monkeypatch.setattr(graph, "ChatOpenAI", FakeChatOpenAI)

    cfg = OmegaConf.create(
        {
            "llm": {"base_url": "http://llm.test/v1", "model": "test-model", "temperature": 0.0},
            "strategy": {"min_confidence": 0.6},
            "options_trading": {
                "enabled": True,
                "dte_min": 14,
                "dte_max": 45,
                "target_delta_min": 0.30,
                "target_delta_max": 0.60,
                "min_open_interest": 100,
                "min_volume": 10,
            },
        }
    )
    state = {**_state("rsi: 71.2"), "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"}}

    pick = graph.select_option_contract(state, cfg)

    assert pick["option_pick"]["contract_symbol"] == "AAPL250117C00200000"
    assert captured.get("api_key")


def _async_options_cfg(**ot_overrides):
    ot = {
        "enabled": True,
        "dte_min": 14,
        "dte_max": 45,
        "target_delta_min": 0.30,
        "target_delta_max": 0.60,
        "min_open_interest": 100,
        "min_volume": 10,
    }
    ot.update(ot_overrides)
    return OmegaConf.create(
        {
            "llm": {"base_url": "http://llm.test/v1", "model": "m", "temperature": 0.0},
            "options_trading": ot,
        }
    )


def _async_options_state():
    return {
        **_state("rsi: 71.2"),
        "symbol": "AAPL",
        "exchange": "stocks",
        "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "breakout"},
    }


def _run_async_select(monkeypatch, tool, bound_cls, structured_cls, captured, cfg=None):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _fake_tools():
        return [tool] if tool is not None else []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def bind_tools(self, tools):
            return bound_cls()

        def with_structured_output(self, schema):
            return structured_cls()

    monkeypatch.setattr(graph, "get_options_tools", _fake_tools)
    monkeypatch.setattr(graph, "ChatOpenAI", FakeChatOpenAI)
    state = _async_options_state()
    return asyncio.run(
        graph._select_option_contract_async(state, cfg or _async_options_cfg(), state["signal"])
    )


def test_select_option_contract_async_compacts_results_and_steers_filtered_chain(monkeypatch):
    captured = {}
    big_chain = json.dumps(
        {
            "snapshots": {
                f"AAPL250620C{int((100 + i) * 1000):08d}": {
                    "latestQuote": {"bp": 1.0, "ap": 1.2},
                    "greeks": {"delta": round(0.01 * i, 2)},
                }
                for i in range(1, 120)
            }
        }
    )

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            captured["tool_args"] = args
            return big_chain

    class _Bound:
        def __init__(self):
            self._calls = 0

        def invoke(self, messages):
            captured["last_messages"] = messages
            self._calls += 1
            if self._calls == 1:
                return type(
                    "R",
                    (),
                    {"tool_calls": [{"name": "get_option_chain", "args": {"underlying_symbol": "AAPL"}, "id": "c1"}]},
                )()
            return type("R", (), {"tool_calls": []})()

    class _Structured:
        def invoke(self, messages):
            captured["final_messages"] = messages
            return graph.OptionContractPick(
                contract_symbol="AAPL250620C00145000",
                strike=145.0,
                expiration="2025-06-20",
                right="call",
                delta=0.45,
                premium=1.1,
                reasoning="mid-delta",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _Bound, _Structured, captured)

    assert pick.contract_symbol == "AAPL250620C00145000"
    human = captured["last_messages"][1].content
    assert "expiration_date_gte" in human
    assert "Days-to-expiration window: 14-45" not in human
    tool_msgs = [m for m in captured["final_messages"] if isinstance(m, graph.ToolMessage)]
    assert tool_msgs and len(str(tool_msgs[0].content)) < len(big_chain) // 2


def test_trim_history_noop_under_cap():
    msgs = [
        graph.SystemMessage(content="s"),
        graph.HumanMessage(content="h"),
        graph.ToolMessage(content="t", tool_call_id="x"),
    ]
    assert graph._trim_history(msgs, hard_cap=1000) == msgs


def test_trim_history_neutralizes_old_tool_messages_over_cap():
    msgs = [
        graph.SystemMessage(content="s"),
        graph.HumanMessage(content="h"),
        graph.ToolMessage(content="A" * 80000, tool_call_id="c1"),
        graph.ToolMessage(content="B" * 400, tool_call_id="c2"),
    ]
    out = graph._trim_history(msgs, hard_cap=1000)
    assert graph.estimate_tokens(out) <= 1000
    assert "budget" in str(out[2].content).lower()
    assert out[2].tool_call_id == "c1"
    assert str(out[3].content) == "B" * 400  # newest kept intact


def test_select_option_contract_async_stops_looping_at_token_budget(monkeypatch):
    captured = {}
    # each tool result compacts to ~6000 chars; 4 calls/round -> ~6k tokens/round,
    # so the ~12k-token budget trips after round 2 (well before the 6-round cap)
    blob = "Z" * 12000

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return blob

    class _Bound:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            captured["rounds"] = self.n
            return type(
                "R",
                (),
                {"tool_calls": [{"name": "get_option_chain", "args": {}, "id": f"c{self.n}-{k}"} for k in range(4)]},
            )()

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol="AAPL250620C00145000",
                strike=145.0,
                expiration="2025-06-20",
                right="call",
                delta=0.45,
                premium=1.1,
                reasoning="x",
            )

    _run_async_select(monkeypatch, _Tool(), _Bound, _Structured, captured)

    assert captured["rounds"] < graph._MAX_TOOL_CALL_ROUNDS  # broke early on the token budget


def _row(sym, delta, exp, bid=1.0, ask=1.2, right="call"):
    return {
        "symbol": sym,
        "strike": 100.0,
        "expiration": exp,
        "right": right,
        "delta": delta,
        "gamma": None,
        "theta": None,
        "vega": None,
        "iv": None,
        "bid": bid,
        "ask": ask,
        "oi": 500,
        "volume": 50,
    }


def test_fallback_pick_selects_closest_to_target_delta_passing_gates():
    today = date(2025, 6, 1)
    ok_exp = "2025-06-20"  # 19 DTE, inside 14-45
    rows = [
        _row("A", 0.35, ok_exp),
        _row("B", 0.45, ok_exp),  # closest to mid 0.45
        _row("C", 0.58, ok_exp),
        _row("D", 0.45, "2025-06-05"),  # 4 DTE -- outside window
        _row("E", 0.10, ok_exp),  # outside delta window
    ]
    cfg = OmegaConf.create(
        {"options_trading": {"dte_min": 14, "dte_max": 45, "target_delta_min": 0.30, "target_delta_max": 0.60}}
    )
    pick = graph._fallback_pick(rows, "call", cfg, delta_mid=0.45, today=today)
    assert pick.contract_symbol == "B"
    assert pick.premium == 1.1
    assert "fallback" in pick.reasoning.lower()


def test_fallback_pick_returns_none_when_nothing_qualifies():
    today = date(2025, 6, 1)
    cfg = OmegaConf.create(
        {"options_trading": {"dte_min": 14, "dte_max": 45, "target_delta_min": 0.30, "target_delta_max": 0.60}}
    )
    rows = [_row("A", 0.05, "2025-06-20"), _row("B", 0.45, "2025-06-20", bid=0, ask=0)]
    assert graph._fallback_pick(rows, "call", cfg, 0.45, today) is None


def test_select_option_contract_async_uses_fallback_when_structured_output_raises(monkeypatch):
    captured = {}
    ok_exp = (datetime.now(pytz.timezone("US/Eastern")).date() + timedelta(days=25)).isoformat()
    occ = f"AAPL{ok_exp.replace('-', '')[2:]}C00145000"
    chain = json.dumps({"snapshots": {occ: {"latestQuote": {"bp": 1.0, "ap": 1.2}, "greeks": {"delta": 0.46}}}})

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Bound:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return type("R", (), {"tool_calls": [{"name": "get_option_chain", "args": {}, "id": "c"}]})()
            return type("R", (), {"tool_calls": []})()

    class _Structured:
        def invoke(self, messages):
            raise ValueError("model returned 1 token, cannot parse OptionContractPick")

    pick = _run_async_select(monkeypatch, _Tool(), _Bound, _Structured, captured)

    assert pick is not None
    assert pick.right == "call"
    assert "fallback" in pick.reasoning.lower()


def test_select_option_contract_async_prefetches_chain_with_indicative_feed(monkeypatch):
    """The deterministic pre-fetch must run before the agent loop, filter the chain to the right /
    expiration window, and force feed=indicative -- so a model that never calls a tool and returns
    no structured pick still yields a fallback contract from the seeded rows."""
    captured = {}
    calls = []
    ok_exp = (datetime.now(pytz.timezone("US/Eastern")).date() + timedelta(days=25)).isoformat()
    occ = f"AAPL{ok_exp.replace('-', '')[2:]}C00145000"
    envelope = {
        "_alpaca_mcp_security": {"trust": "untrusted_tool_output"},
        "data": {"snapshots": {occ: {"latestQuote": {"bp": 1.0, "ap": 1.2}, "greeks": {"delta": 0.46}}}},
    }
    # the real adapter (>=0.3) hands back a list of content blocks wrapping the security envelope
    chain_blocks = [{"type": "text", "text": json.dumps(envelope)}]

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            calls.append(args)
            return chain_blocks

    class _Bound:
        def invoke(self, messages):
            return type("R", (), {"tool_calls": []})()  # model never calls a tool

    class _Structured:
        def invoke(self, messages):
            return None  # qwen3.6 empty-structured-output regression

    pick = _run_async_select(monkeypatch, _Tool(), _Bound, _Structured, captured)

    assert calls, "chain was never pre-fetched"
    seed = calls[0]
    assert seed["feed"] == "indicative"
    assert seed["type"] == "call"
    assert seed["expiration_date_gte"] and seed["expiration_date_lte"]
    assert pick is not None and pick.contract_symbol == occ
    assert "fallback" in pick.reasoning.lower()


def test_select_option_contract_async_forces_feed_on_llm_tool_calls(monkeypatch):
    """A get_option_latest_quote the model emits without a feed arg (or with feed=opra) must be
    rewritten to the configured feed before dispatch."""
    captured = {}
    seen_args = []

    class _Tool:
        name = "get_option_latest_quote"

        async def ainvoke(self, args):
            seen_args.append(dict(args))
            return json.dumps({"snapshots": {}})

    class _Bound:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return type(
                    "R", (),
                    {"tool_calls": [{"name": "get_option_latest_quote", "args": {"symbols": "X", "feed": "opra"}, "id": "c"}]},
                )()
            return type("R", (), {"tool_calls": []})()

    class _Structured:
        def invoke(self, messages):
            return None

    # no get_option_chain tool here, so the only ainvoke calls come from the loop
    _run_async_select(monkeypatch, _Tool(), _Bound, _Structured, captured)

    assert seen_args and all(a["feed"] == "indicative" for a in seen_args)


def _single_call_chain():
    """A one-contract chain whose sole row (a call, mid-delta, ~25 DTE, quoted) qualifies for the
    deterministic fallback -- so tests can assert the fallback engaged by checking the pick came back
    as that contract with fallback reasoning."""
    ok_exp = (datetime.now(pytz.timezone("US/Eastern")).date() + timedelta(days=25)).isoformat()
    occ = f"AAPL{ok_exp.replace('-', '')[2:]}C00145000"
    chain = json.dumps({"snapshots": {occ: {"latestQuote": {"bp": 1.0, "ap": 1.2}, "greeks": {"delta": 0.46}}}})
    return occ, chain


def _bound_one_tool_then_stop(chain_holder):
    class _Bound:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            chain_holder["rounds"] = self.n
            if self.n == 1:
                return type("R", (), {"tool_calls": [{"name": "get_option_chain", "args": {}, "id": "c"}]})()
            return type("R", (), {"tool_calls": []})()

    return _Bound


def test_select_option_contract_async_rejects_wrong_direction_structured_pick(monkeypatch):
    captured = {}
    occ, chain = _single_call_chain()

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Structured:
        def invoke(self, messages):
            # schema-valid but points the wrong way for a BUY signal
            return graph.OptionContractPick(
                contract_symbol=occ, strike=145.0, expiration="2025-06-20",
                right="put", delta=0.46, premium=1.1, reasoning="model said put",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _bound_one_tool_then_stop(captured), _Structured, captured)

    assert pick is not None
    assert pick.right == "call"  # fell back to the direction-safe deterministic pick
    assert pick.contract_symbol == occ
    assert "fallback" in pick.reasoning.lower()


def test_select_option_contract_async_rejects_unseen_structured_pick(monkeypatch):
    captured = {}
    occ, chain = _single_call_chain()

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol="AAPL991231C09999000", strike=9999.0, expiration="2099-12-31",
                right="call", delta=0.46, premium=1.1, reasoning="hallucinated OCC",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _bound_one_tool_then_stop(captured), _Structured, captured)

    assert pick is not None
    assert pick.contract_symbol == occ  # only contracts actually seen in a chain are eligible
    assert "fallback" in pick.reasoning.lower()


def test_select_option_contract_async_accepts_valid_structured_pick(monkeypatch):
    captured = {}
    occ, chain = _single_call_chain()

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol=occ, strike=145.0, expiration="2025-06-20",
                right="call", delta=0.46, premium=1.1, reasoning="right direction, real contract",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _bound_one_tool_then_stop(captured), _Structured, captured)

    ok_exp = (datetime.now(pytz.timezone("US/Eastern")).date() + timedelta(days=25)).isoformat()
    assert pick.contract_symbol == occ
    assert pick.reasoning == "right direction, real contract"  # the model's rationale is kept
    assert pick.strike == 145.0  # numeric fields are re-grounded in the observed chain row
    assert pick.expiration == ok_exp
    assert pick.delta == 0.46
    assert pick.premium == 1.1


def test_select_option_contract_async_reconciles_fabricated_metadata_from_observed_row(monkeypatch):
    """A schema-valid pick that names a real, seen contract but attaches a fabricated low premium
    and an out-of-window delta must be re-grounded in the observed chain row before it can drive
    Floor Broker sizing -- otherwise qty = risk_per_trade_usd // (premium * 100) is oversized."""
    captured = {}
    ok_exp = (datetime.now(pytz.timezone("US/Eastern")).date() + timedelta(days=25)).isoformat()
    occ, chain = _single_call_chain()

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol=occ, strike=999.0, expiration="2025-06-20",
                right="call", delta=0.99, premium=0.01, reasoning="cheap and juicy",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _bound_one_tool_then_stop(captured), _Structured, captured)

    assert pick.contract_symbol == occ
    assert pick.reasoning == "cheap and juicy"  # the model's rationale is preserved
    assert pick.strike == 145.0  # from the OCC symbol, not the model
    assert pick.expiration == ok_exp  # from the OCC symbol, not the model
    assert pick.delta == 0.46  # from the observed greeks
    assert pick.premium == 1.1  # mid of the observed quote, not the fabricated 0.01


def test_select_option_contract_async_falls_back_when_picked_row_has_no_quote(monkeypatch):
    """A contract seen only without a usable quote can't be reconciled, so the deterministic
    fallback (which requires a quote) picks a different, quoted contract instead."""
    captured = {}
    ok_exp = (datetime.now(pytz.timezone("US/Eastern")).date() + timedelta(days=25)).isoformat()
    ymd = ok_exp.replace("-", "")[2:]
    quoted = f"AAPL{ymd}C00145000"
    unquoted = f"AAPL{ymd}C00150000"
    chain = json.dumps(
        {
            "snapshots": {
                quoted: {"latestQuote": {"bp": 1.0, "ap": 1.2}, "greeks": {"delta": 0.46}},
                unquoted: {"greeks": {"delta": 0.44}},  # no latestQuote
            }
        }
    )

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol=unquoted, strike=150.0, expiration=ok_exp,
                right="call", delta=0.44, premium=0.9, reasoning="prefer the 150",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _bound_one_tool_then_stop(captured), _Structured, captured)

    assert pick.contract_symbol == quoted  # fell back to the quoted contract
    assert "fallback" in pick.reasoning.lower()


def test_reconcile_structured_pick_overwrites_numeric_fields_from_row():
    rows = [_row("AAPL250620C00145000", 0.52, "2025-06-20", bid=2.0, ask=2.4, right="call")]
    pick = graph.OptionContractPick(
        contract_symbol="AAPL250620C00145000", strike=1.0, expiration="2000-01-01",
        right="call", delta=0.01, premium=0.02, reasoning="keep me",
    )
    out = graph._reconcile_structured_pick(pick, "call", rows)
    assert out.strike == 100.0  # _row's strike
    assert out.expiration == "2025-06-20"
    assert out.delta == 0.52
    assert out.premium == 2.2  # mid of 2.0 / 2.4
    assert out.reasoning == "keep me"


def test_reconcile_structured_pick_rejects_direction_mismatch_in_observed_row():
    """The provenance check only compares the model's self-reported right; reconciliation also
    checks the right of the *matched* row, catching a call-labelled pick whose seen OCC symbol is
    actually a put."""
    put_occ = "AAPL250620P00145000"
    rows = [_row(put_occ, -0.40, "2025-06-20", right="put")]
    pick = graph.OptionContractPick(
        contract_symbol=put_occ, strike=145.0, expiration="2025-06-20",
        right="call", delta=0.40, premium=1.1, reasoning="mislabelled",
    )
    assert graph._reconcile_structured_pick(pick, "call", rows) is None


def test_select_option_contract_async_recovers_from_unknown_tool_call(monkeypatch):
    captured = {}
    occ, chain = _single_call_chain()

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            return chain

    class _Bound:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return type("R", (), {"tool_calls": [{"name": "nonexistent_tool", "args": {}, "id": "bad"}]})()
            if self.n == 2:
                return type("R", (), {"tool_calls": [{"name": "get_option_chain", "args": {}, "id": "c"}]})()
            return type("R", (), {"tool_calls": []})()

    class _Structured:
        def invoke(self, messages):
            return graph.OptionContractPick(
                contract_symbol=occ, strike=145.0, expiration="2025-06-20",
                right="call", delta=0.46, premium=1.1, reasoning="recovered",
            )

    pick = _run_async_select(monkeypatch, _Tool(), _Bound, _Structured, captured)

    assert pick is not None  # a hallucinated tool name no longer aborts the whole selector
    assert pick.contract_symbol == occ


def test_select_option_contract_async_recovers_from_failing_tool_call(monkeypatch):
    captured = {}

    class _Tool:
        name = "get_option_chain"

        async def ainvoke(self, args):
            raise RuntimeError("alpaca-mcp-server exploded")

    pick = _run_async_select(monkeypatch, _Tool(), _bound_one_tool_then_stop(captured), _raise_structured(), captured)

    assert pick is None  # no rows gathered -> fallback finds nothing, but nothing raised either


def _raise_structured():
    class _Structured:
        def invoke(self, messages):
            raise ValueError("no parse")

    return _Structured


def test_fallback_pick_enforces_configured_liquidity_when_fields_present():
    today = date(2025, 6, 1)
    ok_exp = "2025-06-20"
    thin = _row("THIN", 0.45, ok_exp)
    thin["oi"] = 5  # below min_open_interest 100
    deep = _row("DEEP", 0.50, ok_exp)
    cfg = OmegaConf.create(
        {
            "options_trading": {
                "dte_min": 14, "dte_max": 45, "target_delta_min": 0.30, "target_delta_max": 0.60,
                "min_open_interest": 100, "min_volume": 10,
            }
        }
    )
    pick = graph._fallback_pick([thin, deep], "call", cfg, delta_mid=0.45, today=today)
    assert pick.contract_symbol == "DEEP"  # THIN skipped despite being closer to target delta


def test_fallback_pick_keeps_rows_missing_liquidity_fields():
    today = date(2025, 6, 1)
    ok_exp = "2025-06-20"
    row = _row("NOLIQ", 0.45, ok_exp)
    row["oi"] = None
    row["volume"] = None
    cfg = OmegaConf.create(
        {
            "options_trading": {
                "dte_min": 14, "dte_max": 45, "target_delta_min": 0.30, "target_delta_max": 0.60,
                "min_open_interest": 100, "min_volume": 10,
            }
        }
    )
    pick = graph._fallback_pick([row], "call", cfg, delta_mid=0.45, today=today)
    assert pick.contract_symbol == "NOLIQ"  # Alpaca usually omits OI/volume; absence is not a fail


def test_llm_call_sets_request_timeout_and_no_retries(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema):
            class _S:
                def invoke(self, m):
                    return graph.Signal(
                        symbol="AAPL", action="HOLD", reasoning="r", size_hint=0.0, confidence=0.5
                    )

            return _S()

    monkeypatch.setattr(graph, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(graph, "_symbol_memory_text", lambda *a, **k: "")
    cfg = OmegaConf.create(
        {"llm": {"base_url": "http://llm.test/v1", "model": "m", "temperature": 0.0, "request_timeout_s": 90}}
    )

    graph.llm_call({**_state("rsi: 71"), "symbol": "AAPL"}, cfg)

    assert captured["timeout"] == 90.0
    assert captured["max_retries"] == 0


def test_llm_timeout_defaults_to_120_when_unset():
    cfg = OmegaConf.create({"llm": {"base_url": "x", "model": "m", "temperature": 0.0}})
    assert graph._llm_timeout(cfg) == 120.0


def _option_cfg(**overrides):
    base = {
        "floor_broker": {"base_url": "http://floor-broker.test:8000"},
        "macro_blackout": {"enabled": False, "dates": []},
        "strategy": {
            "risk_per_trade_usd": 100,
            "win_rate_throttle": {"enabled": False},
            "symbol_stop_cooldown": {"enabled": False},
        },
        "options_trading": {
            "dte_min": 14,
            "dte_max": 45,
            "target_delta_min": 0.30,
            "target_delta_max": 0.60,
        },
    }
    base["options_trading"].update(overrides)
    return OmegaConf.create(base)


def _far_expiration(days: int) -> str:
    return (datetime.now(pytz.timezone("US/Eastern")) + timedelta(days=days)).date().isoformat()


def test_call_floor_broker_option_skips_when_dte_out_of_range(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    cfg = _option_cfg()
    state = {
        **_state("rsi: 71.2"),
        "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"},
        "option_pick": {
            "contract_symbol": "AAPL250117C00200000",
            "strike": 200.0,
            "expiration": _far_expiration(2),
            "right": "call",
            "delta": 0.45,
            "premium": 3.20,
            "reasoning": "r",
        },
    }

    result = graph.call_floor_broker_option(state, cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "dte_out_of_range"


def test_call_floor_broker_option_skips_when_delta_out_of_range(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    cfg = _option_cfg()
    state = {
        **_state("rsi: 71.2"),
        "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"},
        "option_pick": {
            "contract_symbol": "AAPL250117C00200000",
            "strike": 200.0,
            "expiration": _far_expiration(20),
            "right": "call",
            "delta": 0.15,
            "premium": 3.20,
            "reasoning": "r",
        },
    }

    result = graph.call_floor_broker_option(state, cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "delta_out_of_range"


def test_call_floor_broker_option_skips_when_qty_would_be_zero(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    cfg = _option_cfg()
    state = {
        **_state("rsi: 71.2"),
        "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"},
        "option_pick": {
            "contract_symbol": "AAPL250117C00200000",
            "strike": 200.0,
            "expiration": _far_expiration(20),
            "right": "call",
            "delta": 0.45,
            "premium": 50.0,
            "reasoning": "r",
        },
    }

    result = graph.call_floor_broker_option(state, cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "qty_zero"


def test_call_floor_broker_option_posts_to_execute_option(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "submitted", "detail": "option buy order submitted: order-1"}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(graph.requests, "post", _fake_post)
    monkeypatch.setattr(graph, "_option_contract_already_exposed", lambda cfg, sym: False)
    cfg = _option_cfg()
    state = {
        **_state("rsi: 71.2"),
        "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"},
        "option_pick": {
            "contract_symbol": "AAPL250117C00200000",
            "strike": 200.0,
            "expiration": _far_expiration(20),
            "right": "call",
            "delta": 0.45,
            "premium": 0.50,
            "reasoning": "r",
        },
    }

    result = graph.call_floor_broker_option(state, cfg)

    assert captured["url"] == "http://floor-broker.test:8000/execute-option"
    assert captured["json"]["contract_symbol"] == "AAPL250117C00200000"
    assert captured["json"]["qty"] == 2  # floor(100 / (0.50 * 100)) == floor(2.0) == 2
    assert result["execution_result"]["status"] == "submitted"


def test_call_floor_broker_option_records_dealer_decision(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    recorded = {}
    monkeypatch.setattr(
        graph.db, "record_dealer_decision", lambda *a, **k: recorded.setdefault("call", (a, k))
    )
    monkeypatch.setattr(
        graph.slack, "notify_dealer_signal", lambda *a, **k: recorded.setdefault("slack", a)
    )

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "submitted", "detail": "option buy order submitted: order-1"}

    monkeypatch.setattr(graph.requests, "post", lambda url, json, timeout: FakeResponse())
    monkeypatch.setattr(graph, "_option_contract_already_exposed", lambda cfg, sym: False)
    cfg = _option_cfg()
    state = {
        **_state("rsi: 71.2"),
        "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "delta/DTE window", "size_hint": 0.5},
        "option_pick": {
            "contract_symbol": "AAPL250117C00200000",
            "strike": 200.0,
            "expiration": _far_expiration(20),
            "right": "call",
            "delta": 0.45,
            "premium": 0.50,
            "reasoning": "r",
        },
    }

    graph.call_floor_broker_option(state, cfg)

    args, kwargs = recorded["call"]
    assert args == ("CRV/USD", "BUY", "delta/DTE window", 0.5)
    assert kwargs == {"ohlcv_enrichment_active": False, "cycle_id": "cycle-1"}
    # Verify slack notification is called
    assert recorded["slack"] == ("CRV/USD", "BUY", "delta/DTE window")


def test_call_floor_broker_option_skips_quietly_when_contract_already_exposed(monkeypatch):
    """An already-held / in-flight contract re-triggers every poll cycle. The skip must land before
    the dealer-signal Slack line and the dealer_decision row -- only a single floor_broker_event
    'skip' row is written, for the audit trail."""
    events = []
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: events.append(a))
    monkeypatch.setattr(graph, "_option_contract_already_exposed", lambda cfg, sym: True)

    def _must_not_fire(*a, **k):
        raise AssertionError("an already-held contract must not announce a signal or POST an order")

    monkeypatch.setattr(graph.slack, "notify_dealer_signal", _must_not_fire)
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", _must_not_fire)
    monkeypatch.setattr(graph.db, "record_dealer_decision", _must_not_fire)
    monkeypatch.setattr(graph.requests, "post", _must_not_fire)

    result = graph.call_floor_broker_option(_option_pick_state(), _option_cfg())

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "duplicate_option_position"
    assert events == [
        ("CRV/USD", "skip", "AAPL250117C00200000 is already held or has a BUY order in flight")
    ]


def test_call_floor_broker_option_still_signals_when_selection_failed(monkeypatch):
    """A genuine failed selection (option_pick is None, no option_skip) skips the exposure check --
    which needs a contract symbol -- but must still emit the dealer signal so a failed pick is
    visible, and record one auditable floor_broker_event "skip" row (mirroring the low_confidence
    sibling branch and every other skip outcome in this node). No extra Floor Broker Slack line --
    the dealer signal above already announced the call."""
    signals, events = [], []
    monkeypatch.setattr(graph.slack, "notify_dealer_signal", lambda *a, **k: signals.append(a))
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: events.append(a))
    monkeypatch.setattr(graph, "_option_contract_already_exposed", _fail_if_exposure_checked)

    def _no_slack(*a, **k):
        raise AssertionError("a genuine no_option_pick skip must not post a Floor Broker line")

    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", _no_slack)

    state = {**_state("rsi: 71.2"), "signal": {"action": "BUY", "confidence": 0.9, "reasoning": "r"}, "option_pick": None}
    result = graph.call_floor_broker_option(state, _option_cfg())

    assert result["execution_result"]["reason"] == "no_option_pick"
    assert signals == [("CRV/USD", "BUY", "r")]
    assert events == [("CRV/USD", "skip", "no option contract was selected")]


def _fail_if_exposure_checked(cfg, sym):
    raise AssertionError("exposure check must be skipped when there is no option_pick")


def test_call_floor_broker_option_reports_low_confidence_skip(monkeypatch):
    """When select_option_contract() declined for low confidence, this node records a single
    auditable floor_broker_event "skip" row and a low_confidence execution_result -- mirroring the
    stock path's bookkeeping -- but emits no extra Floor Broker Slack line (the dealer signal above
    already announced the sub-threshold call)."""
    signals, events = [], []
    monkeypatch.setattr(graph.slack, "notify_dealer_signal", lambda *a, **k: signals.append(a))
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: events.append(a))
    monkeypatch.setattr(graph, "_option_contract_already_exposed", _fail_if_exposure_checked)

    def _no_slack(*a, **k):
        raise AssertionError("a low-confidence option skip must not post a Floor Broker line")

    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", _no_slack)

    state = {
        **_state("rsi: 71.2"),
        "signal": {"action": "BUY", "confidence": 0.4, "reasoning": "r"},
        "option_pick": None,
        "option_skip": {"reason": "low_confidence", "detail": "BUY confidence 0.40 below minimum 0.6"},
    }
    result = graph.call_floor_broker_option(state, _option_cfg())

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "low_confidence"
    assert result["execution_result"]["detail"] == "BUY confidence 0.40 below minimum 0.6"
    assert signals == [("CRV/USD", "BUY", "r")]
    assert events == [("CRV/USD", "skip", "BUY confidence 0.40 below minimum 0.6")]


def test_option_contract_already_exposed_returns_false_on_request_exception(monkeypatch, real_option_exposure):
    def _get_boom(url, timeout):
        raise graph.requests.RequestException("connection refused")

    monkeypatch.setattr(graph.requests, "get", _get_boom)

    assert graph._option_contract_already_exposed(_option_cfg(), "AAPL250117C00200000") is False


def test_option_contract_already_exposed_true_when_contract_in_exposure(monkeypatch, real_option_exposure):
    class _ExposureResp:
        status_code = 200

        def json(self):
            return {"contracts": ["AAPL250117C00200000", "MSFT250117P00300000"]}

    monkeypatch.setattr(graph.requests, "get", lambda url, timeout: _ExposureResp())

    assert graph._option_contract_already_exposed(_option_cfg(), "AAPL250117C00200000") is True
    assert graph._option_contract_already_exposed(_option_cfg(), "TSLA250117C00100000") is False


def _base_option_gate_cfg(**strategy_overrides):
    strategy = {
        "risk_per_trade_usd": 100,
        "win_rate_throttle": {"enabled": False},
        "symbol_stop_cooldown": {"enabled": False},
        "min_confidence": 0.6,
    }
    strategy.update(strategy_overrides)
    return OmegaConf.create(
        {
            "floor_broker": {"base_url": "http://floor-broker.test:8000"},
            "macro_blackout": {"enabled": False, "dates": []},
            "strategy": strategy,
            "options_trading": {
                "dte_min": 14,
                "dte_max": 45,
                "target_delta_min": 0.30,
                "target_delta_max": 0.60,
            },
            "analyst": {"track_record_days": 5},
        }
    )


def _option_pick_state(action: str = "BUY"):
    return {
        **_state("rsi: 71.2"),
        "signal": {"action": action, "confidence": 0.9, "reasoning": "r"},
        "option_pick": {
            "contract_symbol": "AAPL250117C00200000",
            "strike": 200.0,
            "expiration": _far_expiration(20),
            "right": "call",
            "delta": 0.45,
            "premium": 0.50,
            "reasoning": "r",
        },
    }


def test_call_floor_broker_option_skips_buy_during_macro_blackout(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called during a macro blackout")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    cfg = _base_option_gate_cfg()
    today = datetime.now(pytz.timezone("US/Eastern")).date().isoformat()
    cfg.macro_blackout.enabled = True
    cfg.macro_blackout.dates = [{"date": today, "label": "CPI release"}]

    result = graph.call_floor_broker_option(_option_pick_state(), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "macro_blackout"


def test_call_floor_broker_option_skips_buy_when_symbol_recently_stopped_out(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called while symbol cooldown is active")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    cfg = _base_option_gate_cfg(
        symbol_stop_cooldown={"enabled": True}, symbol_stop_cooldown_days=1, max_symbol_stop_losses=1
    )
    monkeypatch.setattr(
        graph.db,
        "fetch_symbol_floor_broker_events_since",
        lambda symbol, since_date, limit=100: [{"event_type": "fill", "detail": "stop_loss leg filled: order-1"}],
    )

    result = graph.call_floor_broker_option(_option_pick_state(), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "symbol_stop_cooldown"


def test_call_floor_broker_option_skips_buy_when_win_rate_below_minimum(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called while the win-rate throttle is active")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    cfg = _base_option_gate_cfg(
        win_rate_throttle={"enabled": True},
        win_rate_throttle_scope="global",
        min_win_rate=0.3,
        win_rate_min_sample=5,
    )
    events = [{"event_type": "fill", "detail": "stop_loss leg filled: order-1"}] * 4 + [
        {"event_type": "fill", "detail": "take_profit leg filled: order-1"}
    ]
    monkeypatch.setattr(graph.db, "fetch_floor_broker_events_since", lambda since_date: events)

    result = graph.call_floor_broker_option(_option_pick_state(), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "win_rate_throttle"


def test_call_floor_broker_option_skips_sell_during_macro_blackout(monkeypatch):
    """Regression: a SELL signal maps to buying a put (a new bearish entry, right = "put" in
    select_option_contract), never to closing an existing position -- it must be blocked by the same
    entry gates as a BUY/call pick."""
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called during a macro blackout")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    cfg = _base_option_gate_cfg()
    today = datetime.now(pytz.timezone("US/Eastern")).date().isoformat()
    cfg.macro_blackout.enabled = True
    cfg.macro_blackout.dates = [{"date": today, "label": "CPI release"}]

    result = graph.call_floor_broker_option(_option_pick_state(action="SELL"), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "macro_blackout"


def test_call_floor_broker_option_skips_when_no_authorized_budget(monkeypatch):
    """A held-only position carries budget=0 -- refuse the entry rather than forward it, mirroring
    call_floor_broker's no_authorized_budget guard. Applies to SELL too: unlike stocks, an option
    SELL is a new put entry, never a close."""
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called for a held-only (budget=0) position")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    cfg = _base_option_gate_cfg()
    state = {**_option_pick_state(action="SELL"), "budget": 0.0}

    result = graph.call_floor_broker_option(state, cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "no_authorized_budget"


def test_call_floor_broker_option_skips_when_risk_per_trade_usd_not_configured(monkeypatch):
    monkeypatch.setattr(graph.slack, "notify_floor_broker_result", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_floor_broker_event", lambda *a, **k: None)
    monkeypatch.setattr(graph.db, "record_dealer_decision", lambda *a, **k: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Floor Broker must not be called when risk_per_trade_usd is unset")

    monkeypatch.setattr(graph.requests, "post", _fail_if_called)

    cfg = _base_option_gate_cfg(risk_per_trade_usd=None)

    result = graph.call_floor_broker_option(_option_pick_state(), cfg)

    assert result["execution_result"]["status"] == "skipped"
    assert result["execution_result"]["reason"] == "risk_per_trade_usd_not_configured"


def test_route_after_llm_call_selects_option_branch_when_enabled():
    cfg = OmegaConf.create({"options_trading": {"enabled": True}})
    state = {"exchange": "stocks", "signal": {"action": "BUY"}}
    assert graph._route_after_llm_call(state, cfg) == "select_option_contract"


def test_route_after_llm_call_selects_stock_branch_when_disabled():
    cfg = OmegaConf.create({"options_trading": {"enabled": False}})
    state = {"exchange": "stocks", "signal": {"action": "BUY"}}
    assert graph._route_after_llm_call(state, cfg) == "call_floor_broker"


def test_route_after_llm_call_stays_on_stock_branch_for_crypto_even_when_enabled():
    """Regression: options_trading.enabled=true must never route a crypto symbol into option-contract
    selection, even though src/dealer/main.py's loop-level crypto.enabled gate makes this combination
    unreachable in the planned go-live config -- this is a defense-in-depth check on the routing
    function itself."""
    cfg = OmegaConf.create({"options_trading": {"enabled": True}})
    state = {"exchange": "binance", "signal": {"action": "BUY"}}
    assert graph._route_after_llm_call(state, cfg) == "call_floor_broker"


def test_route_after_llm_call_sends_hold_to_stock_branch_even_with_options_enabled():
    """A HOLD carries no contract to pick. Routing it through the option branch only produces a
    no-op in select_option_contract() and then an option-tagged Dealer Slack line for a non-trade.
    The stock path's HOLD skip semantics are identical regardless of asset class, so HOLD must go
    straight there."""
    cfg = OmegaConf.create({"options_trading": {"enabled": True}})
    state = {"exchange": "stocks", "signal": {"action": "HOLD"}}
    assert graph._route_after_llm_call(state, cfg) == "call_floor_broker"
