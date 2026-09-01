import asyncio
from datetime import datetime, timedelta
from typing import NotRequired, TypedDict

import pytz
import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.common import db, slack
from src.common.bars import fetch_multi_timeframe_bars
from src.common.config import load_config
from src.common.logging import get_logger
from src.common.indicators import fetch_indicators_bulk
from src.dealer.features import compute_derived_features, format_features_text
from src.dealer.mcp_options import get_options_tools
from src.dealer.option_chain import (
    compact_tool_result,
    ensure_option_feed,
    estimate_tokens,
    mcp_text,
    parse_option_chain,
)
from src.dealer.schema import OptionContractPick, Signal

log = get_logger("DEALER")

_MAX_TOOL_CALL_ROUNDS = 6
_OPTION_PROMPT_TOKEN_BUDGET = 12_000  # stop the tool-calling loop once history passes this
_OPTION_PROMPT_TOKEN_HARD_CAP = 24_000  # never invoke the LLM above this -- neutralize old tool msgs first


class DealerState(TypedDict):
    symbol: str
    exchange: str
    budget: float
    indicator_names: list[str]
    indicators_text: str
    cycle_id: str
    raw_bars: dict
    ohlcv_features_text: str
    signal: dict | None
    option_pick: dict | None
    # Set by select_option_contract() when it declines to pick a contract for a reason the next
    # node should report verbatim (currently {"reason": "low_confidence", "detail": ...}). Absent
    # for a genuine selection failure, which call_floor_broker_option() reports as no_option_pick.
    option_skip: NotRequired[dict | None]
    execution_result: dict | None


def fetch_indicators(state: DealerState, cfg) -> DealerState:
    names = state["indicator_names"]
    if names == ["ALL"]:
        names = [ind["name"] for ind in cfg.indicators]

    indicators_text = fetch_indicators_bulk(cfg.indicators, state["symbol"], state["exchange"], names, log)
    return {**state, "indicators_text": indicators_text}


def fetch_market_data(state: DealerState, cfg) -> DealerState:
    if not cfg.get("ohlcv_enrichment", {}).get("enabled", False):
        return {**state, "raw_bars": {}, "ohlcv_features_text": ""}

    raw_bars = fetch_multi_timeframe_bars(state["symbol"], state["exchange"], cfg)
    features_by_timeframe = {
        timeframe: features
        for timeframe, bars in raw_bars.items()
        if (features := compute_derived_features(bars, cfg))
    }
    features_text = format_features_text(features_by_timeframe, state["symbol"])
    if features_text:
        log(f"📊 OHLCV enrichment ready for {state['symbol']} ({', '.join(features_by_timeframe)})")
    else:
        log(f"⚠️ OHLCV enrichment produced no usable features for {state['symbol']}")
    return {**state, "raw_bars": raw_bars, "ohlcv_features_text": features_text}


def llm_call(state: DealerState, cfg) -> DealerState:
    llm = ChatOpenAI(
        base_url=cfg.llm.base_url,
        api_key="not-needed",
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        timeout=_llm_timeout(cfg),
        max_retries=0,
    ).with_structured_output(Signal)

    system_prompt = (
        "You are an expert technical trader in stocks. "
        "Based on the values of ALL of the indicators below, decide if you should BUY, SELL, or HOLD. "
        "size_hint must be a decimal fraction between 0.0 and 1.0 representing the portion of the "
        "symbol's budget to deploy on a BUY (e.g. 0.5 = half the budget, 1.0 = the full budget) -- "
        "never a dollar amount or share count. "
        "confidence must be a decimal fraction between 0.0 and 1.0 reflecting how strongly the "
        "indicators actually agree with and support this action -- reserve a high score for cases "
        "where multiple indicators clearly point the same direction, and score low when the "
        "reading is mixed, weak, or borderline."
    )
    memory_text = _symbol_memory_text(state["symbol"], cfg)
    user_prompt = f"Indicators for {state['symbol']}:\n{state['indicators_text']}"
    if state.get("ohlcv_features_text"):
        user_prompt += f"\n\nAdditional OHLCV context:\n{state['ohlcv_features_text']}"
    if memory_text:
        user_prompt += f"\n\nRecent same-symbol trading history:\n{memory_text}"

    log(f"🧠 thinking about {state['symbol']}...")
    signal: Signal = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    log(f"🧠 AI {cfg.llm.model} says: {signal.action} {state['symbol']}")
    return {**state, "signal": signal.model_dump()}


def _route_after_indicators(state: DealerState) -> str:
    return "llm_call" if state["indicators_text"].strip() else "skip_missing_indicators"


def skip_missing_indicators(state: DealerState, cfg) -> DealerState:
    """fetch_indicators_bulk can come back empty (TAAPI 200 with no data -- typically
    insufficient historical bars for a thinly-traded pair, see indicators.py). Skip the cycle
    outright instead of sending an empty indicators block to the LLM, which otherwise improvises
    a "please provide indicators" HOLD instead of a real trading decision."""
    reasoning = (
        f"no indicator data available for {state['symbol']} this cycle "
        f"(exchange={state['exchange']}) -- skipped without invoking the LLM"
    )
    log(f"⏭️  {reasoning}")
    slack.notify_dealer_signal(state["symbol"], "HOLD", reasoning)
    db.record_dealer_decision(
        state["symbol"],
        "HOLD",
        reasoning,
        None,
        ohlcv_enrichment_active=False,
        cycle_id=state.get("cycle_id"),
    )
    return {
        **state,
        "signal": {"action": "HOLD", "reasoning": reasoning, "size_hint": None, "confidence": 0.0},
        "execution_result": {"status": "skipped", "detail": "missing_indicators"},
    }


def _is_quad_witching_day(d) -> bool:
    """Third Friday of March/June/September/December -- the quarterly simultaneous expiration of
    stock options, index options, and index futures, historically one of the highest-volume,
    highest-volatility sessions of the year for the broad market. Computed directly rather than
    requiring a config.yaml entry since it's a fixed calendar rule, not an externally-published
    date like FOMC/CPI/NFP -- it can't drift or need quarterly upkeep."""
    if d.month not in (3, 6, 9, 12) or d.weekday() != 4:  # Friday
        return False
    return 15 <= d.day <= 21  # the only possible day-of-month range for a third Friday


def _macro_blackout_active(cfg) -> str | None:
    """Returns the matching event label if today (America/New_York calendar date) is either a
    hand-maintained macro_blackout.dates entry (FOMC, CPI, PPI, PCE, NFP, ISM PMI, GDP, Fed Chair
    testimony, or any other scheduled market-wide macro release) or an auto-computed quad
    witching day, else None. Whole-trading-day granularity, not hours-based -- see
    docs/architecture.md Risk controls for why."""
    if not cfg.macro_blackout.enabled:
        return None
    today = datetime.now(pytz.timezone("US/Eastern")).date()
    if _is_quad_witching_day(today):
        return "Quad witching (quarterly options/futures expiration)"
    today_str = today.isoformat()
    for entry in cfg.macro_blackout.dates:
        if entry["date"] == today_str:
            return entry.get("label", entry["date"])
    return None


def _classify_exit_event(event: dict) -> str | None:
    """Returns "win" for a take-profit exit, "loss" for a stop-loss exit, or None for any other
    floor_broker_events row (BUY opens, manual dealer-triggered SELLs, eod_flatten, errors,
    skips) -- those don't carry an unambiguous win/loss outcome without the original entry price,
    which floor_broker_events doesn't record. Covers both stock brackets (poll_bracket_fills
    records event_type="fill" with the leg reason embedded in `detail`) and crypto's synthetic
    stop/target (poll_bracket_fills records event_type="synthetic_take_profit"/
    "synthetic_stop_loss" directly, see src/floor_broker/main.py)."""
    event_type = event.get("event_type", "")
    if event_type == "synthetic_take_profit":
        return "win"
    if event_type == "synthetic_stop_loss":
        return "loss"
    if event_type == "fill":
        detail = event.get("detail") or ""
        if "take_profit leg filled" in detail:
            return "win"
        if "stop_loss leg filled" in detail:
            return "loss"
    return None


def _symbol_memory_text(symbol: str, cfg) -> str:
    """Returns compact recent same-symbol context for the Dealer prompt. This is advisory context
    only: DB failures fail open so a transient logging-store outage does not block decisions."""
    if not cfg.strategy.get("dealer_memory", {}).get("enabled", True):
        return ""
    days = cfg.strategy.get("symbol_memory_days", 2)
    limit = cfg.strategy.get("symbol_memory_limit", 8)
    since_date = (datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=days)).date()
    try:
        decisions = db.fetch_symbol_dealer_decisions_since(symbol, since_date, limit=limit)
        events = db.fetch_symbol_floor_broker_events_since(symbol, since_date, limit=limit)
    except Exception as exc:
        log(f"⚠️ same-symbol memory unavailable for {symbol}: {exc}")
        return ""

    lines = []
    for d in reversed(decisions):
        decided_at = d.get("decided_at", "")
        size_hint = d.get("size_hint")
        suffix = f", size_hint={size_hint}" if size_hint is not None else ""
        lines.append(f"dealer {decided_at}: {d.get('action')}{suffix}; {d.get('reasoning')}")
    for e in reversed(events):
        occurred_at = e.get("occurred_at", "")
        lines.append(f"floor {occurred_at}: {e.get('event_type')}; {e.get('detail')}")
    return "\n".join(lines[-limit:])


def _symbol_stop_cooldown_active(symbol: str, cfg) -> str | None:
    """Blocks repeated same-symbol BUYs after recent stop-outs. This is deterministic risk
    control, separate from the LLM's interpretation of recent history."""
    if not cfg.strategy.get("symbol_stop_cooldown", {}).get("enabled", True):
        return None
    days = cfg.strategy.get("symbol_stop_cooldown_days", 1)
    max_stops = cfg.strategy.get("max_symbol_stop_losses", 1)
    since_date = (datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=days)).date()
    try:
        events = db.fetch_symbol_floor_broker_events_since(symbol, since_date, limit=100)
    except Exception as exc:
        log(f"⚠️ stop cooldown check unavailable for {symbol}: {exc}")
        return None

    losses = sum(1 for event in events if _classify_exit_event(event) == "loss")
    if losses >= max_stops:
        return f"{losses} stop-loss exit(s) for {symbol} since {since_date}; cooldown blocks new BUYs"
    return None


def _win_rate_throttle_active(cfg, symbol: str | None = None) -> str | None:
    """Returns a Slack-friendly reason string if new BUYs should pause because the trailing
    automatic-exit win rate (take-profit vs stop-loss hits, both stock brackets and synthetic
    crypto stops) has fallen below strategy.min_win_rate -- a quantitative companion to
    fetch_track_record (src/analyst/graph.py), which only hands the LLM raw history and leaves
    interpretation to it. Requires at least strategy.win_rate_min_sample completed exits before
    evaluating at all, so a handful of early trades can't trip the throttle on noise. Discretionary
    dealer-triggered SELLs, eod_flatten, and BUY opens are excluded from the count -- see
    _classify_exit_event."""
    if not cfg.strategy.get("win_rate_throttle", {}).get("enabled", True):
        return None

    since_date = (datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=cfg.analyst.track_record_days)).date()
    scope = cfg.strategy.get("win_rate_throttle_scope", "global")
    if scope == "symbol" and symbol:
        events = db.fetch_symbol_floor_broker_events_since(symbol, since_date, limit=100)
        scope_label = f"{symbol} "
    else:
        events = db.fetch_floor_broker_events_since(since_date)
        scope_label = ""
    outcomes = [_classify_exit_event(e) for e in events]
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    total = wins + losses
    if total < cfg.strategy.win_rate_min_sample:
        return None

    win_rate = wins / total
    if win_rate < cfg.strategy.min_win_rate:
        return (
            f"{scope_label}trailing win rate {win_rate:.0%} ({wins}W/{losses}L over {total} exits) "
            f"below minimum {cfg.strategy.min_win_rate:.0%}"
        )
    return None


def call_floor_broker(state: DealerState, cfg) -> DealerState:
    signal = state["signal"]
    slack.notify_dealer_signal(state["symbol"], signal["action"], signal["reasoning"])
    db.record_dealer_decision(
        state["symbol"],
        signal["action"],
        signal["reasoning"],
        signal.get("size_hint"),
        ohlcv_enrichment_active=bool(state.get("ohlcv_features_text")),
        cycle_id=state.get("cycle_id"),
    )

    if signal["action"] == "HOLD":
        return {**state, "execution_result": {"status": "skipped", "detail": "HOLD"}}

    if signal["action"] == "BUY":
        blackout_label = _macro_blackout_active(cfg)
        if blackout_label:
            log(f"⏭️  BUY for {state['symbol']} skipped -- macro blackout ({blackout_label})")
            result = {
                "status": "skipped",
                "reason": "macro_blackout",
                "detail": f"new BUY entries paused for macro blackout: {blackout_label}",
            }
            slack.notify_floor_broker_result(state["symbol"], signal["action"], result["status"], result["detail"])
            db.record_floor_broker_event(state["symbol"], "skip", result["detail"])
            return {**state, "execution_result": result}

        cooldown_reason = _symbol_stop_cooldown_active(state["symbol"], cfg)
        if cooldown_reason:
            log(f"⏭️  BUY for {state['symbol']} skipped -- {cooldown_reason}")
            result = {
                "status": "skipped",
                "reason": "symbol_stop_cooldown",
                "detail": f"new BUY entry paused: {cooldown_reason}",
            }
            slack.notify_floor_broker_result(state["symbol"], signal["action"], result["status"], result["detail"])
            db.record_floor_broker_event(state["symbol"], "skip", result["detail"])
            return {**state, "execution_result": result}

        throttle_reason = _win_rate_throttle_active(cfg, state["symbol"])
        if throttle_reason:
            log(f"⏭️  BUY for {state['symbol']} skipped -- {throttle_reason}")
            result = {
                "status": "skipped",
                "reason": "win_rate_throttle",
                "detail": f"new BUY entries paused: {throttle_reason}",
            }
            slack.notify_floor_broker_result(state["symbol"], signal["action"], result["status"], result["detail"])
            db.record_floor_broker_event(state["symbol"], "skip", result["detail"])
            return {**state, "execution_result": result}

        confidence = signal.get("confidence", 1.0)
        if confidence < cfg.strategy.min_confidence:
            log(f"⏭️  BUY for {state['symbol']} skipped -- confidence {confidence:.2f} below minimum {cfg.strategy.min_confidence}")
            result = {
                "status": "skipped",
                "reason": "low_confidence",
                "detail": f"BUY confidence {confidence:.2f} below minimum {cfg.strategy.min_confidence}",
            }
            slack.notify_floor_broker_result(state["symbol"], signal["action"], result["status"], result["detail"])
            db.record_floor_broker_event(state["symbol"], "skip", result["detail"])
            return {**state, "execution_result": result}

    if signal["action"] == "BUY" and state["budget"] <= 0:
        # A held-only position (merge_held_positions()) carries budget=0 -- its market value is
        # observed exposure, not authorized new-BUY capital. Never forward a BUY for it.
        log(f"⚠️  no authorized BUY budget for {state['symbol']} (held position) -- skipping")
        result = {
            "status": "skipped",
            "reason": "no_authorized_budget",
            "detail": "held position has no authorized new-BUY budget",
        }
        slack.notify_floor_broker_result(state["symbol"], signal["action"], result["status"], result["detail"])
        db.record_floor_broker_event(state["symbol"], "skip", result["detail"])
        return {**state, "execution_result": result}

    budget = state["budget"]
    if signal["action"] == "BUY":
        size_hint = signal.get("size_hint", 1.0)
        budget = state["budget"] * size_hint
        if budget <= 0:
            # size_hint scaled the authorized budget to exactly $0 -- ExecuteRequest requires
            # budget > 0 (src/floor_broker/app.py), so this must be refused locally rather than
            # forwarded to a request that would 422. A budget scaled to a small-but-positive
            # amount is intentionally still forwarded -- execution.py's own minimum-notional/
            # insufficient-qty checks already handle that gracefully with their own reason codes.
            log(f"⚠️  size_hint={size_hint} scales BUY budget for {state['symbol']} to $0 -- skipping")
            result = {
                "status": "skipped",
                "reason": "size_hint_zero",
                "detail": f"size_hint {size_hint} scales authorized budget to $0",
            }
            slack.notify_floor_broker_result(state["symbol"], signal["action"], result["status"], result["detail"])
            db.record_floor_broker_event(state["symbol"], "skip", result["detail"])
            return {**state, "execution_result": result}

    payload = {
        "symbol": state["symbol"],
        "exchange": state["exchange"],
        "action": signal["action"],
        "budget": budget,
        "slP": cfg.trading.slP,
        "tpP": cfg.trading.tpP,
    }

    try:
        response = requests.post(f"{cfg.floor_broker.base_url}/execute", json=payload, timeout=30)
        if response.status_code != 200:
            log(f"💥 floor broker error: {response.status_code} {response.text}")
            slack.notify_floor_broker_result(state["symbol"], signal["action"], "error", response.text)
            db.record_floor_broker_event(state["symbol"], "error", response.text)
            return {**state, "execution_result": {"status": "error", "detail": response.text}}
        result = response.json()
        slack.notify_floor_broker_result(
            state["symbol"],
            signal["action"],
            result["status"],
            result["detail"],
            reason=result.get("reason"),
            fill_price=result.get("fill_price"),
            sl_price=result.get("sl_price"),
            tp_price=result.get("tp_price"),
        )
        db.record_floor_broker_event(
            state["symbol"],
            f"{signal['action'].lower()}_{result['status']}",
            result["detail"],
            price=result.get("fill_price"),
        )
        return {**state, "execution_result": result}
    except requests.RequestException as exc:
        log(f"💥 floor broker request failed: {exc}")
        slack.notify_floor_broker_result(state["symbol"], signal["action"], "error", str(exc))
        db.record_floor_broker_event(state["symbol"], "error", str(exc))
        return {**state, "execution_result": {"status": "error", "detail": str(exc)}}


def select_option_contract(state: DealerState, cfg) -> DealerState:
    if not cfg.get("options_trading", {}).get("enabled", False):
        return {**state, "option_pick": None}

    signal = state["signal"]
    if signal["action"] == "HOLD":
        # Defense in depth -- _route_after_llm_call() already sends HOLD to the stock path.
        return {**state, "option_pick": None}

    confidence = signal.get("confidence", 1.0)
    if confidence < cfg.strategy.min_confidence:
        # Stash the real reason: call_floor_broker_option() would otherwise report a bare
        # no_option_pick, losing the low-confidence explanation the stock path records.
        return {
            **state,
            "option_pick": None,
            "option_skip": {
                "reason": "low_confidence",
                "detail": (
                    f"{signal['action']} confidence {confidence:.2f} below minimum "
                    f"{cfg.strategy.min_confidence}"
                ),
            },
        }

    try:
        pick = asyncio.run(_select_option_contract_async(state, cfg, signal))
    except Exception as exc:
        log(f"💥 option contract selection failed for {state['symbol']}: {exc}")
        return {**state, "option_pick": None}

    return {**state, "option_pick": pick.model_dump() if pick else None}


def _llm_timeout(cfg) -> float:
    """Per-request wall-clock ceiling for every Ollama call. A hung generation fails fast
    instead of stacking behind the Dealer's 10-minute poll cycle."""
    return float(cfg.llm.get("request_timeout_s", 120))


def _trim_history(messages: list, hard_cap: int) -> list:
    """Keep every message (no orphaned tool results) but blank the content of the oldest large
    ToolMessages until estimate_tokens(messages) <= hard_cap. No-op when already under."""
    if estimate_tokens(messages) <= hard_cap:
        return messages
    trimmed = list(messages)
    for i, m in enumerate(trimmed[:-1]):  # never touch the most recent message
        if isinstance(m, ToolMessage) and len(str(m.content)) > 200:
            trimmed[i] = ToolMessage(
                content="[older tool result dropped to stay within the context budget]",
                tool_call_id=m.tool_call_id,
            )
            if estimate_tokens(trimmed) <= hard_cap:
                break
    return trimmed


def _fallback_pick(rows: list[dict], right: str, cfg, delta_mid: float, today) -> OptionContractPick | None:
    """Deterministic pick when the structured LLM call fails or is rejected: the contract whose
    |delta| is closest to the target-delta midpoint, among those matching right / delta window /
    DTE window with a usable quote. Configured min_open_interest / min_volume are enforced only for
    rows where Alpaca actually returned those fields (its option snapshots usually omit them). Keeps
    options trading working when qwen3.6 flakes; the Floor Broker's risk gates still run on the
    result."""
    ot = cfg.options_trading
    best = None
    for r in rows:
        if r.get("right") != right or r.get("delta") is None or not r.get("expiration"):
            continue
        d = abs(r["delta"])
        if not (ot.target_delta_min <= d <= ot.target_delta_max):
            continue
        try:
            dte = (datetime.strptime(r["expiration"], "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if not (ot.dte_min <= dte <= ot.dte_max):
            continue
        min_oi, min_vol = ot.get("min_open_interest"), ot.get("min_volume")
        oi, vol = r.get("oi"), r.get("volume")
        if min_oi is not None and oi is not None and oi < min_oi:
            continue
        if min_vol is not None and vol is not None and vol < min_vol:
            continue
        bid, ask = r.get("bid"), r.get("ask")
        if not bid or not ask or ask <= 0:
            continue
        mid = round((bid + ask) / 2, 2)
        if mid <= 0:
            continue
        key = (abs(d - delta_mid), ask - bid)
        if best is None or key < best[0]:
            best = (key, r, d, mid)
    if best is None:
        return None
    _, r, d, mid = best
    return OptionContractPick(
        contract_symbol=r["symbol"],
        strike=float(r["strike"]),
        expiration=r["expiration"],
        right=right,
        delta=d,
        premium=mid,
        reasoning="deterministic fallback: structured LLM pick unavailable",
    )


async def _select_option_contract_async(state: DealerState, cfg, signal: dict) -> OptionContractPick | None:
    tools = await get_options_tools()
    tools_by_name = {t.name: t for t in tools}

    # api_key is required by ChatOpenAI's client init but unused -- the local model router at
    # cfg.llm.base_url does no auth. Without it ChatOpenAI falls back to demanding OPENAI_API_KEY
    # from the env and every contract selection dies before a tool call (same "not-needed"
    # sentinel as llm_call()).
    llm = ChatOpenAI(
        base_url=cfg.llm.base_url,
        api_key="not-needed",
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        timeout=_llm_timeout(cfg),
        max_retries=0,
    )
    agent_llm = llm.bind_tools(tools)
    right = "call" if signal["action"] == "BUY" else "put"
    # opra (the alpaca-mcp default) is a paid real-time entitlement the paper account lacks -> 403
    # on every option-data call. "indicative" is the free feed. Overridable for the day an account
    # with an OPRA subscription is pointed at.
    data_feed = cfg.options_trading.get("data_feed", "indicative")

    et = pytz.timezone("US/Eastern")
    today = datetime.now(et).date()
    exp_gte = (today + timedelta(days=cfg.options_trading.dte_min)).isoformat()
    exp_lte = (today + timedelta(days=cfg.options_trading.dte_max)).isoformat()
    delta_mid = (cfg.options_trading.target_delta_min + cfg.options_trading.target_delta_max) / 2

    messages = [
        SystemMessage(
            content=(
                "You are an options contract selector for a paper-trading account. The option "
                f"chain for the target window has already been fetched (feed={data_feed}) and is "
                "included below -- pick exactly one contract from it that fits the stated "
                "constraints. If you need to look up anything further, use the provided Alpaca "
                f"tools and ALWAYS pass feed={data_feed!r} (the default 'opra' feed 403s on this "
                "account); set type to the desired right and keep limit at 50 or fewer."
            )
        ),
        HumanMessage(
            content=(
                f"Underlying: {state['symbol']}\n"
                f"Desired right: {right}\n"
                f"Expiration window: {exp_gte} to {exp_lte} "
                f"(pass as expiration_date_gte / expiration_date_lte to get_option_chain)\n"
                f"Target |delta| window: {cfg.options_trading.target_delta_min}"
                f"-{cfg.options_trading.target_delta_max}\n"
                f"Minimum open interest: {cfg.options_trading.min_open_interest}\n"
                f"Minimum volume: {cfg.options_trading.min_volume}\n"
                f"Dealer reasoning for the underlying signal: {signal['reasoning']}\n\n"
                "Inspect the pre-fetched chain below (call more tools only if needed), then "
                "respond with your final pick."
            )
        ),
    ]

    seen_rows: list[dict] = []

    # Deterministic chain pre-fetch. The local model (qwen3.6:35b-a3b) is unreliable at both
    # emitting the get_option_chain tool call and returning a structured pick -- see
    # _fallback_pick's docstring. Fetching the chain ourselves, once, with known-good filters
    # guarantees seen_rows is populated (so _fallback_pick can always run) regardless of what the
    # model does. The agent loop below still runs so a more capable model can refine, but it is no
    # longer load-bearing.
    chain_tool = tools_by_name.get("get_option_chain")
    if chain_tool is not None:
        seed_args = ensure_option_feed(
            "get_option_chain",
            {
                "underlying_symbol": state["symbol"],
                "type": right,
                "expiration_date_gte": exp_gte,
                "expiration_date_lte": exp_lte,
                "limit": 50,
            },
            data_feed,
        )
        try:
            seed_raw = mcp_text(await chain_tool.ainvoke(seed_args))
            seed_rows = parse_option_chain(seed_raw)
            seen_rows.extend(seed_rows)
            messages.append(
                HumanMessage(
                    content=(
                        f"Pre-fetched {right} option chain for {state['symbol']} "
                        f"({exp_gte} to {exp_lte}) -- {len(seed_rows)} contract(s):\n"
                        + compact_tool_result("get_option_chain", seed_raw, target_delta_mid=delta_mid)
                    )
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed pre-fetch must not abort the selector
            log(f"⚠️ option selection for {state['symbol']}: chain pre-fetch failed ({exc})")

    for _ in range(_MAX_TOOL_CALL_ROUNDS):
        messages = _trim_history(messages, _OPTION_PROMPT_TOKEN_HARD_CAP)
        response = agent_llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                log(f"⚠️ option selection for {state['symbol']}: LLM called unknown tool {call['name']!r}, skipping")
                messages.append(ToolMessage(content=f"error: unknown tool {call['name']!r}", tool_call_id=call["id"]))
                continue
            try:
                result = await tool.ainvoke(ensure_option_feed(call["name"], call["args"], data_feed))
            except Exception as exc:  # noqa: BLE001 - a bad tool call must not abort the whole selector
                log(f"⚠️ option selection for {state['symbol']}: tool {call['name']} failed ({exc}), skipping")
                messages.append(ToolMessage(content=f"error: tool {call['name']} failed: {exc}", tool_call_id=call["id"]))
                continue
            raw = mcp_text(result)
            seen_rows.extend(parse_option_chain(raw))
            messages.append(
                ToolMessage(
                    content=compact_tool_result(call["name"], raw, target_delta_mid=delta_mid),
                    tool_call_id=call["id"],
                )
            )
        if estimate_tokens(messages) > _OPTION_PROMPT_TOKEN_BUDGET:
            log(f"⚠️ option selection for {state['symbol']}: token budget reached, forcing final pick")
            break

    messages = _trim_history(messages, _OPTION_PROMPT_TOKEN_HARD_CAP)
    structured_llm = llm.with_structured_output(OptionContractPick)
    try:
        pick = structured_llm.invoke(messages)
        if pick:
            reject = _structured_pick_rejection(pick, right, seen_rows)
            if reject is None:
                reconciled = _reconcile_structured_pick(pick, right, seen_rows)
                if reconciled is not None:
                    return reconciled
                log(
                    f"⚠️ option selection for {state['symbol']}: structured pick "
                    f"{pick.contract_symbol} could not be reconciled against observed chain data; "
                    "using fallback"
                )
            else:
                log(f"⚠️ option selection for {state['symbol']}: {reject}; using fallback")
    except Exception as exc:  # noqa: BLE001 - any parse/transport failure falls back deterministically
        log(f"⚠️ option selection for {state['symbol']}: structured pick failed ({exc}); using fallback")
    return _fallback_pick(seen_rows, right, cfg, delta_mid, today)


def _structured_pick_rejection(pick: OptionContractPick, right: str, seen_rows: list[dict]) -> str | None:
    """Guards the LLM's structured pick before it reaches the Floor Broker. The model can return a
    schema-valid contract that points the wrong way (a put on a BUY signal, a call on a SELL) or an
    OCC symbol it never actually saw in a chain response. Either one is rejected here so the
    deterministic fallback -- which is direction-safe and only picks from observed rows -- runs
    instead. Returns a reason string when the pick must be rejected, or None when it is safe to use."""
    if pick.right != right:
        return f"structured pick right={pick.right!r} != intended {right!r}"
    if pick.contract_symbol not in {r.get("symbol") for r in seen_rows}:
        return f"structured pick {pick.contract_symbol} was not in any observed option chain"
    return None


def _reconcile_structured_pick(
    pick: OptionContractPick, right: str, seen_rows: list[dict]
) -> OptionContractPick | None:
    """Re-ground the LLM pick's numeric fields in the values actually observed for that contract in
    the option chain, so call_floor_broker_option()'s DTE / |delta| gates and its
    risk_per_trade_usd // (premium * 100) sizing never run on model-supplied numbers. A schema-valid
    pick can name a real, seen OCC symbol yet attach a fabricated low premium (-> oversized qty) or
    an in-window delta -- only the observed row is trusted here. Also rejects a pick whose matched
    row points the other way (a call-labelled pick on a seen put symbol), which
    _structured_pick_rejection -- comparing only the model's self-reported right -- would let
    through. Returns the reconciled pick (model rationale preserved), or None -> the deterministic
    fallback runs -- when the matched row lacks a usable direction, delta, or quote. Assumes
    _structured_pick_rejection has already passed, so a matching row exists."""
    row = next((r for r in seen_rows if r.get("symbol") == pick.contract_symbol), None)
    if row is None:  # defensive: _structured_pick_rejection should have guaranteed this
        return None
    if row.get("right") != right:
        return None
    delta, strike, expiration = row.get("delta"), row.get("strike"), row.get("expiration")
    bid, ask = row.get("bid"), row.get("ask")
    if delta is None or strike is None or not expiration or not bid or not ask or ask <= 0:
        return None
    mid = round((bid + ask) / 2, 2)
    if mid <= 0:
        return None
    return pick.model_copy(
        update={
            "strike": float(strike),
            "expiration": expiration,
            "right": right,
            "delta": abs(delta),
            "premium": mid,
        }
    )


def _route_after_llm_call(state: DealerState, cfg) -> str:
    # A HOLD has no contract to pick -- routing it through the option branch only no-ops in
    # select_option_contract() and then mislabels the Dealer Slack line as an option signal for a
    # non-trade. The stock path's HOLD skip is identical regardless of asset class, so send it there.
    if (
        state["exchange"] == "stocks"
        and cfg.get("options_trading", {}).get("enabled", False)
        and (state.get("signal") or {}).get("action") != "HOLD"
    ):
        return "select_option_contract"
    return "call_floor_broker"


def call_floor_broker_option(state: DealerState, cfg) -> DealerState:
    """Every option_pick here represents a brand-new position -- there is no "SELL means close"
    case for options via this graph node (the only exit path is check_option_stops()'s synthetic
    SL/TP/DTE-force-close, which calls sell_option() directly, outside this function). So unlike
    call_floor_broker()'s stock path, these risk gates are NOT limited to action == "BUY":
    select_option_contract() maps a bearish SELL signal to buying a put (right = "call" if
    action == "BUY" else "put"), and that put purchase is exactly as much a new entry as a call
    purchase is -- it must not bypass macro blackout / symbol-stop cooldown / win-rate throttle."""
    option_pick = state.get("option_pick")
    action = state["signal"]["action"]

    # Duplicate-exposure check runs first, before the dealer-signal Slack line and the
    # dealer_decision row. _fallback_pick is deterministic, so a contract already held (or with a
    # BUY in flight) gets re-picked identically every ~10-minute poll cycle for the life of the
    # position; announcing/recording that each time is pure noise. Skip it quietly -- a single
    # floor_broker_event keeps the skip auditable. A failed selection (option_pick is None) still
    # falls through to the dealer signal below.
    if option_pick and _option_contract_already_exposed(cfg, option_pick["contract_symbol"]):
        detail = f"{option_pick['contract_symbol']} is already held or has a BUY order in flight"
        db.record_floor_broker_event(state["symbol"], "skip", detail)
        return {
            **state,
            "execution_result": {
                "status": "skipped",
                "reason": "duplicate_option_position",
                "detail": detail,
            },
        }

    slack.notify_dealer_signal(
        state["symbol"], state["signal"]["action"], state["signal"]["reasoning"], asset_class="option"
    )
    db.record_dealer_decision(
        state["symbol"],
        state["signal"]["action"],
        state["signal"]["reasoning"],
        state["signal"].get("size_hint"),
        ohlcv_enrichment_active=bool(state.get("ohlcv_features_text")),
        cycle_id=state.get("cycle_id"),
    )

    if not option_pick:
        skip = state.get("option_skip")
        if skip and skip.get("reason") == "low_confidence":
            # Mirror the stock path's low-confidence bookkeeping: one auditable floor_broker_event
            # "skip" row and a matching execution_result reason. No extra Slack line -- the dealer
            # signal above already announced the sub-threshold call.
            db.record_floor_broker_event(state["symbol"], "skip", skip["detail"])
            return {
                **state,
                "execution_result": {
                    "status": "skipped",
                    "reason": "low_confidence",
                    "detail": skip["detail"],
                },
            }
        # Genuine selection failure (LLM produced nothing usable and _fallback_pick found no
        # contract, or _select_option_contract_async raised). Record one auditable floor_broker_event
        # "skip" row like the low_confidence branch above and every other skip outcome in this node --
        # otherwise this is the only skip that leaves no Floor Broker event trail. No extra Slack
        # line -- the dealer signal above already announced the call.
        detail = "no option contract was selected"
        db.record_floor_broker_event(state["symbol"], "skip", detail)
        return {**state, "execution_result": {"status": "skipped", "reason": "no_option_pick", "detail": detail}}

    blackout_label = _macro_blackout_active(cfg)
    if blackout_label:
        log(f"⏭️  option entry for {state['symbol']} skipped -- macro blackout ({blackout_label})")
        result = {
            "status": "skipped",
            "reason": "macro_blackout",
            "detail": f"new option entries paused for macro blackout: {blackout_label}",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    cooldown_reason = _symbol_stop_cooldown_active(state["symbol"], cfg)
    if cooldown_reason:
        log(f"⏭️  option entry for {state['symbol']} skipped -- {cooldown_reason}")
        result = {
            "status": "skipped",
            "reason": "symbol_stop_cooldown",
            "detail": f"new option entry paused: {cooldown_reason}",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    throttle_reason = _win_rate_throttle_active(cfg, state["symbol"])
    if throttle_reason:
        log(f"⏭️  option entry for {state['symbol']} skipped -- {throttle_reason}")
        result = {
            "status": "skipped",
            "reason": "win_rate_throttle",
            "detail": f"new option entries paused: {throttle_reason}",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    if state["budget"] <= 0:
        # A held-only position (merge_held_positions()) carries budget=0 -- see call_floor_broker's
        # identical guard. Options never merge held-only positions today (merge_held_positions only
        # reads stock/crypto positions), but every option_pick is a brand-new entry
        # regardless of BUY/SELL (unlike stocks, where SELL means close) -- so this guard applies
        # unconditionally, without the action == "BUY" restriction call_floor_broker uses.
        log(f"⚠️  no authorized budget for {state['symbol']} option entry -- skipping")
        result = {
            "status": "skipped",
            "reason": "no_authorized_budget",
            "detail": "no authorized budget for new option entry",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    expiration = datetime.strptime(option_pick["expiration"], "%Y-%m-%d").date()
    today = datetime.now(pytz.timezone("US/Eastern")).date()
    dte = (expiration - today).days
    if not (cfg.options_trading.dte_min <= dte <= cfg.options_trading.dte_max):
        result = {
            "status": "skipped",
            "reason": "dte_out_of_range",
            "detail": f"DTE {dte} outside [{cfg.options_trading.dte_min}, {cfg.options_trading.dte_max}]",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    delta = abs(option_pick["delta"])
    if not (cfg.options_trading.target_delta_min <= delta <= cfg.options_trading.target_delta_max):
        result = {
            "status": "skipped",
            "reason": "delta_out_of_range",
            "detail": f"delta {delta} outside [{cfg.options_trading.target_delta_min}, {cfg.options_trading.target_delta_max}]",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    # Option premiums run $300-900/contract, so the stock-sized strategy.risk_per_trade_usd (~$100,
    # tuned as "max $ lost on a stopped-out equity BUY") floors every option order to qty 0. Size
    # options off their own budget knob when set, falling back to the shared one. max_notional_usd
    # is still the hard ceiling re-checked at BUY time in the floor broker.
    option_risk_usd = cfg.options_trading.get("risk_per_trade_usd") or cfg.strategy.risk_per_trade_usd
    if not option_risk_usd:
        result = {
            "status": "skipped",
            "reason": "risk_per_trade_usd_not_configured",
            "detail": "neither options_trading.risk_per_trade_usd nor strategy.risk_per_trade_usd is configured",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    premium = option_pick["premium"]
    qty = int(option_risk_usd // (premium * 100)) if premium > 0 else 0
    if qty < 1:
        result = {
            "status": "skipped",
            "reason": "qty_zero",
            "detail": f"option_risk_usd={option_risk_usd} affords 0 contracts at premium ${premium}",
        }
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}

    payload = {
        "contract_symbol": option_pick["contract_symbol"],
        "side": "BUY",
        "qty": qty,
        "symbol": state["symbol"],
        "right": option_pick["right"],
        "strike": option_pick["strike"],
        "expiration": option_pick["expiration"],
        "delta": option_pick["delta"],
        "premium": premium,
        "reasoning": option_pick["reasoning"],
        "cycle_id": state.get("cycle_id"),
    }

    try:
        response = requests.post(f"{cfg.floor_broker.base_url}/execute-option", json=payload, timeout=30)
        if response.status_code != 200:
            log(f"💥 floor broker option error: {response.status_code} {response.text}")
            result = {"status": "error", "detail": response.text}
            graph_slack_and_record(state, action, result)
            return {**state, "execution_result": result}
        result = response.json()
        slack.notify_floor_broker_result(
            state["symbol"], action, result["status"], result["detail"],
            asset_class="option", reason=result.get("reason"),
        )
        db.record_floor_broker_event(state["symbol"], f"option_{result['status']}", result["detail"])
        return {**state, "execution_result": result}
    except requests.RequestException as exc:
        log(f"💥 floor broker option request failed: {exc}")
        result = {"status": "error", "detail": str(exc)}
        graph_slack_and_record(state, action, result)
        return {**state, "execution_result": result}


def _option_contract_already_exposed(cfg, contract_symbol: str) -> bool:
    """Best-effort pre-check against the Floor Broker's current option exposure so the Dealer skips
    a contract it already holds (or has a pending BUY for) before it announces the signal or spends
    an /execute-option round trip -- _fallback_pick is deterministic, so a slow-filling BUY
    otherwise gets re-picked and re-submitted identically every cycle. buy_option() enforces this
    authoritatively; any transient failure here just falls through to that."""
    try:
        resp = requests.get(f"{cfg.floor_broker.base_url}/option-exposure", timeout=10)
        if resp.status_code != 200:
            log(f"⚠️ option-exposure check for {contract_symbol} returned {resp.status_code}; proceeding")
            return False
        return contract_symbol in set(resp.json().get("contracts", []))
    except requests.RequestException as exc:
        log(f"⚠️ option-exposure check for {contract_symbol} failed ({exc}); proceeding")
        return False


def graph_slack_and_record(state: DealerState, action: str, result: dict) -> None:
    # Option-entry path only -- every call site is inside call_floor_broker_option().
    slack.notify_floor_broker_result(
        state["symbol"], action, result["status"], result["detail"],
        asset_class="option", reason=result.get("reason"),
    )
    db.record_floor_broker_event(state["symbol"], "skip" if result["status"] == "skipped" else result["status"], result["detail"])


def build_graph():
    # Each lambda calls load_config() fresh at invocation time (once per node per graph run,
    # i.e. once per Dealer poll cycle per symbol) rather than baking one cfg into the closure at
    # build time -- build_graph() itself only runs once per process, but a live config change
    # must be visible within load_config()'s refresh window without a Dealer restart.
    graph = StateGraph(DealerState)
    graph.add_node("fetch_indicators", lambda state: fetch_indicators(state, load_config()))
    graph.add_node("fetch_market_data", lambda state: fetch_market_data(state, load_config()))
    graph.add_node("llm_call", lambda state: llm_call(state, load_config()))
    graph.add_node("skip_missing_indicators", lambda state: skip_missing_indicators(state, load_config()))
    graph.add_node("select_option_contract", lambda state: select_option_contract(state, load_config()))
    graph.add_node("call_floor_broker", lambda state: call_floor_broker(state, load_config()))
    graph.add_node("call_floor_broker_option", lambda state: call_floor_broker_option(state, load_config()))

    graph.set_entry_point("fetch_indicators")
    graph.add_conditional_edges(
        "fetch_indicators",
        _route_after_indicators,
        {"llm_call": "fetch_market_data", "skip_missing_indicators": "skip_missing_indicators"},
    )
    graph.add_edge("fetch_market_data", "llm_call")
    graph.add_conditional_edges(
        "llm_call",
        lambda state: _route_after_llm_call(state, load_config()),
        {"call_floor_broker": "call_floor_broker", "select_option_contract": "select_option_contract"},
    )
    graph.add_edge("select_option_contract", "call_floor_broker_option")
    graph.add_edge("skip_missing_indicators", END)
    graph.add_edge("call_floor_broker", END)
    graph.add_edge("call_floor_broker_option", END)

    return graph.compile()
