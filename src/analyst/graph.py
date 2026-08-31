import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TypedDict

import pytz
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.analyst import sources
from src.analyst.schema import PortfolioSelection
from src.common import db, slack
from src.common.alpaca_client import trading_client
from src.common.config import load_config
from src.common.eod import fetch_fills, summarize_positions
from src.common.indicators import fetch_indicators_bulk
from src.common.logging import get_logger
from src.common.portfolio_state import write_portfolio as _write_portfolio

log = get_logger("ANALYST")

DEFAULT_INDICATORS = ["rsi", "macd", "vwap", "bbands", "sma", "ema"]


class AnalystState(TypedDict):
    raw_candidates: list
    research_text: str
    indicator_text: str
    track_record_text: str
    pnl_text: str
    selection: dict | None
    stock_market_open: bool
    is_midday_run: bool


def _by_abs_change_pct(candidate: dict) -> float:
    # A missing change_pct (e.g. a large-cap symbol with no mover data today) must sort last,
    # not crash the comparison.
    return abs(candidate["change_pct"]) if candidate.get("change_pct") is not None else -1


def _filter_earnings_blackout(candidates: list[dict], cfg) -> list[dict]:
    if not cfg.earnings_blackout.enabled or not candidates:
        return candidates
    blackout_symbols = sources.fetch_earnings_calendar(
        [c["symbol"] for c in candidates],
        cfg.earnings_blackout.days_before,
        cfg.earnings_blackout.days_after,
    )
    if blackout_symbols:
        log(f"📅 dropping {len(blackout_symbols)} candidate(s) in earnings blackout: {sorted(blackout_symbols)}")
    return [c for c in candidates if c["symbol"] not in blackout_symbols]


def _allocate_bucket_counts(pool_size: int, weights: dict[str, float]) -> dict[str, int]:
    """Splits `pool_size` slots across buckets proportional to `weights` (need not sum to 1.0 --
    normalized here), using largest-remainder rounding so the counts always sum to exactly
    `pool_size`."""
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return {bucket: 0 for bucket in weights}
    raw = {bucket: pool_size * weight / total_weight for bucket, weight in weights.items()}
    counts = {bucket: int(value) for bucket, value in raw.items()}
    remainder = pool_size - sum(counts.values())
    by_fraction_desc = sorted(weights, key=lambda bucket: raw[bucket] - counts[bucket], reverse=True)
    for bucket in by_fraction_desc[:remainder]:
        counts[bucket] += 1
    return counts


def _fill_buckets_with_backfill(pools: dict[str, list], target_counts: dict[str, int]) -> dict[str, list]:
    """Takes up to `target_counts[bucket]` candidates from each pool (in the pool's given order),
    then redistributes any shortfall -- a bucket without enough eligible candidates to hit its
    target, e.g. earnings blackout thinned the large-cap list -- to whichever other bucket(s)
    still have unused candidates, so a thin day in one bucket doesn't shrink the whole pool when
    another bucket has spare supply."""
    taken = {bucket: list(pool[: target_counts[bucket]]) for bucket, pool in pools.items()}
    shortfall = sum(max(0, target_counts[bucket] - len(pools[bucket])) for bucket in pools)
    if shortfall > 0:
        spare = {bucket: pools[bucket][target_counts[bucket]:] for bucket in pools}
        for bucket in sorted(spare, key=lambda b: len(spare[b]), reverse=True):
            if shortfall <= 0:
                break
            extra = spare[bucket][:shortfall]
            taken[bucket].extend(extra)
            shortfall -= len(extra)
    return taken


def _discover_mixed_candidates(cfg, mix_cfg) -> list[dict]:
    """Composes the stock+crypto candidate pool as a fixed percentage mix -- large-cap names,
    crypto, and today's screener movers/most-actives ("weird stocks") -- rather than letting the
    day's movers ranking alone decide what the LLM sees (mega-caps' small daily %-moves rarely
    compete with a thinly-traded microcap's 20-30%+ swing). See docs/architecture.md."""
    pool_size = mix_cfg.get("pool_size", 20)
    crypto_enabled = cfg.trading.crypto.enabled
    weights = {
        "large_cap": mix_cfg.get("large_cap_pct", 0.4),
        "crypto": mix_cfg.get("crypto_pct", 0.3) if crypto_enabled else 0.0,
        "screener": mix_cfg.get("screener_pct", 0.3),
    }
    target_counts = _allocate_bucket_counts(pool_size, weights)

    large_cap_pool = sources.fetch_large_cap_candidates(cfg.analyst.get("large_cap_symbols", []))
    large_cap_symbols = {c["symbol"] for c in large_cap_pool}

    screener_pool = sources.fetch_screener_candidates(
        cfg.analyst.screener_top_n,
        cfg.analyst.min_price_usd,
        max_abs_change_pct=cfg.analyst.get("max_abs_change_pct"),
        min_dollar_volume=cfg.analyst.get("min_dollar_volume_usd"),
        excluded_suffixes=cfg.analyst.get("excluded_symbol_suffixes", []),
    )
    # A symbol already in large_cap_symbols belongs to that bucket, not the "weird stocks" one.
    screener_pool = [c for c in screener_pool if c["symbol"] not in large_cap_symbols]

    # Earnings blackout applies to any stock candidate uniformly, same as today -- filtering here
    # (before allocation) lets a blackout-caused shortfall in one stock bucket get backfilled from
    # the other, rather than just shrinking the final pool.
    stock_pool = _filter_earnings_blackout(large_cap_pool + screener_pool, cfg)
    large_cap_pool = [c for c in stock_pool if c["symbol"] in large_cap_symbols]
    screener_pool = sorted((c for c in stock_pool if c["symbol"] not in large_cap_symbols), key=_by_abs_change_pct, reverse=True)

    crypto_pool = []
    if crypto_enabled:
        crypto_pool = sorted(sources.fetch_crypto_candidates(cfg.analyst.screener_top_n), key=_by_abs_change_pct, reverse=True)

    selected = _fill_buckets_with_backfill(
        {"large_cap": large_cap_pool, "crypto": crypto_pool, "screener": screener_pool}, target_counts
    )

    stock_candidates = selected["large_cap"] + selected["screener"]
    for c in stock_candidates:
        c["market"] = "stocks"
    crypto_candidates = selected["crypto"]
    for c in crypto_candidates:
        c["market"] = cfg.trading.crypto_taapi_exchange

    log(
        f"🎛️ candidate mix — target {target_counts}, got {len(selected['large_cap'])} large-cap, "
        f"{len(crypto_candidates)} crypto, {len(selected['screener'])} screener"
    )
    return stock_candidates + crypto_candidates


def discover_candidates(state: AnalystState, cfg) -> AnalystState:
    candidates = []
    mix_cfg = cfg.analyst.get("candidate_mix", {})
    mix_active = mix_cfg.get("enabled", False) and cfg.trading.stocks.enabled and state["stock_market_open"]

    if mix_active:
        candidates.extend(_discover_mixed_candidates(cfg, mix_cfg))
    else:
        if cfg.trading.stocks.enabled and state["stock_market_open"]:
            stock_candidates = sources.fetch_screener_candidates(
                cfg.analyst.screener_top_n,
                cfg.analyst.min_price_usd,
                max_abs_change_pct=cfg.analyst.get("max_abs_change_pct"),
                min_dollar_volume=cfg.analyst.get("min_dollar_volume_usd"),
                excluded_suffixes=cfg.analyst.get("excluded_symbol_suffixes", []),
            )
            stock_candidates = _filter_earnings_blackout(stock_candidates, cfg)
            for c in stock_candidates:
                c["market"] = "stocks"
            candidates.extend(stock_candidates)

        if cfg.trading.crypto.enabled:
            crypto_candidates = sources.fetch_crypto_candidates(cfg.analyst.screener_top_n)
            for c in crypto_candidates:
                c["market"] = cfg.trading.crypto_taapi_exchange
            candidates.extend(crypto_candidates)

    return {**state, "raw_candidates": candidates}


def fetch_research(state: AnalystState, cfg) -> AnalystState:
    if not cfg.analyst.news.enabled:
        log("⏭️ news feeds disabled via config — skipping")
        return {**state, "research_text": ""}
    news = sources.fetch_news(cfg.analyst.news_days)
    headlines = sources.fetch_yahoo_rss_headlines(cfg.analyst.yahoo_rss_url)
    research_text = f"{news}\n{headlines}"
    return {**state, "research_text": research_text}


def fetch_indicators(state: AnalystState, cfg) -> AnalystState:
    """Fetches real TAAPI indicator values for the top candidates by move size, so llm_select
    can reason about actual RSI/MACD/etc rather than news text alone. Bounded to
    `indicator_fetch_limit` candidates -- TAAPI's free-tier rate limit is 1 request/15s
    (cfg.taapi.min_request_interval_secs), and fetching every screened candidate would make the
    daily CronJob run take far longer than warranted for candidates the LLM is unlikely to pick."""
    if not cfg.analyst.indicators.enabled:
        log("⏭️ indicators disabled via config — skipping")
        return {**state, "indicator_text": ""}
    ranked = sorted(state["raw_candidates"], key=_by_abs_change_pct, reverse=True)
    top_candidates = ranked[: cfg.analyst.indicator_fetch_limit]
    top_symbols = {c["symbol"] for c in top_candidates}
    large_cap_symbols = set(cfg.analyst.get("large_cap_symbols", []))
    for c in ranked:
        if c["symbol"] in large_cap_symbols and c["symbol"] not in top_symbols:
            top_candidates.append(c)
            top_symbols.add(c["symbol"])

    lines = []
    for i, candidate in enumerate(top_candidates):
        if i > 0:
            time.sleep(cfg.taapi.min_request_interval_secs)
        text = fetch_indicators_bulk(
            cfg.indicators, candidate["symbol"], candidate["market"], DEFAULT_INDICATORS, log
        )
        if text:
            lines.append(f"{candidate['symbol']}:\n{text}")

    return {**state, "indicator_text": "\n".join(lines)}


def fetch_track_record(state: AnalystState, cfg) -> AnalystState:
    """Surfaces the Analyst's own recent pick history -- what it picked, why, what Dealer did
    about it, and how Floor Broker's execution went -- as a qualitative sequence the LLM can use
    to avoid blindly repeating a pattern that already lost. Deliberately stays qualitative rather
    than computing per-pick P&L here: a live, point-in-time unrealized-P&L snapshot of currently
    open positions now exists separately (see fetch_position_pnl below), but that's a read of
    current holdings, not attribution back to a specific past pick -- a symbol can be bought and
    sold more than once, so real per-pick/realized P&L still isn't computed anywhere live (only
    in the offline backtest simulator). Floor Broker's recorded price is still documented as
    informational only (skills/analyst-explain/SKILL.md) -- Alpaca's fills remain ground truth
    for that. This node runs before write_portfolio() records this run's own picks, so a symbol
    picked THIS run has no track record until the NEXT run -- but the query has no explicit upper
    date bound, so on a midday run this DOES surface the same day's earlier morning-run picks
    (and any Dealer/Floor Broker activity on them since) as track record, which is useful signal
    for the midday LLM pass, not a bug."""
    if not cfg.analyst.track_record.enabled:
        log("⏭️ track record disabled via config — skipping")
        return {**state, "track_record_text": ""}

    since_date = (
        datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=cfg.analyst.track_record_days)
    ).date()
    picks = db.fetch_analyst_picks_since(since_date)
    if not picks:
        return {**state, "track_record_text": ""}

    decisions_by_symbol = defaultdict(list)
    for d in db.fetch_dealer_decisions_since(since_date):
        decisions_by_symbol[d["symbol"]].append(d)

    events_by_symbol = defaultdict(list)
    for e in db.fetch_floor_broker_events_since(since_date):
        events_by_symbol[e["symbol"]].append(e)

    lines = []
    for pick in picks:
        symbol = pick["symbol"]
        lines.append(
            f"- {pick['generated_at'].date().isoformat()} picked {symbol} "
            f"(budget ${pick.get('budget')}): {pick.get('rationale')}"
        )
        for d in decisions_by_symbol.get(symbol, []):
            lines.append(
                f"    Dealer {d['decided_at'].date().isoformat()}: {d['action']} -- {d.get('reasoning')}"
            )
        for e in events_by_symbol.get(symbol, []):
            price_note = f" @ ${e['price']}" if e.get("price") is not None else ""
            lines.append(
                f"    Floor Broker {e['occurred_at'].date().isoformat()}: "
                f"{e['event_type']}{price_note} -- {e.get('detail')}"
            )

    return {**state, "track_record_text": "\n".join(lines)}


def fetch_position_pnl(state: AnalystState, cfg) -> AnalystState:
    """Live, point-in-time snapshot of Alpaca's own unrealized P&L for currently-open positions --
    deliberately NOT per-pick attribution: a symbol can be bought/sold more than once, this reads
    current holdings only, and nothing here is persisted or compared across days (that's
    fetch_track_record's job, kept qualitative on purpose). Includes both stocks and crypto --
    not redundant with crypto_eod_report's crypto-only Slack recap, since that posts after the
    fact while this feeds today's picking decision. Fails open like fetch_research/
    fetch_indicators: a transient Alpaca API error degrades to an empty snapshot for this run
    rather than failing the whole Analyst run, since this is supplementary context, not a trading
    gate."""
    if not cfg.analyst.position_pnl.enabled:
        log("⏭️ position P&L snapshot disabled via config — skipping")
        return {**state, "pnl_text": ""}

    try:
        positions = summarize_positions(trading_client.get_all_positions())
    except Exception as exc:
        log(f"⚠️ failed to fetch position P&L snapshot: {exc}")
        return {**state, "pnl_text": ""}

    if not positions:
        return {**state, "pnl_text": ""}

    lines = []
    for p in positions:
        if p["unrealized_pl"] is not None:
            sign = "+" if p["unrealized_pl"] >= 0 else "-"
            pl = f"{sign}${abs(p['unrealized_pl']):,.2f}"
        else:
            pl = "n/a"
        pct = f"{p['unrealized_plpc'] * 100:+.2f}%" if p.get("unrealized_plpc") is not None else "n/a"
        cur = f"${p['current_price']:,.2f}" if p["current_price"] is not None else "n/a"
        lines.append(
            f"- {p['symbol']}: qty {p['qty']:g}, avg entry ${p['avg_entry_price']:,.2f}, "
            f"current {cur}, unrealized {pl} ({pct})"
        )

    return {**state, "pnl_text": "\n".join(lines)}


def _llm_timeout(cfg) -> float:
    """Per-request wall-clock ceiling for the Analyst's Ollama call. It shares the DGX Ollama host
    with the Dealer, so a hung generation must fail fast rather than pin the CronJob pod (and an
    Ollama slot) indefinitely. Mirrors src/dealer/graph.py's _llm_timeout."""
    return float(cfg.llm.get("request_timeout_s", 120))


def llm_select(state: AnalystState, cfg) -> AnalystState:
    llm = ChatOpenAI(
        base_url=cfg.llm.base_url,
        api_key="not-needed",
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        timeout=_llm_timeout(cfg),
        max_retries=0,
    ).with_structured_output(PortfolioSelection)

    system_prompt = (
        "You are the Analyst agent on an automated trading floor. "
        f"Pick at most {cfg.analyst.max_universe_size} symbols worth trading today, "
        "drawn from the candidate list and informed by the research text. "
        "Where a candidate has technical indicator values provided, ground your reasoning in "
        "those actual values rather than news sentiment alone. "
        "Each candidate object includes a `market` field (\"stocks\", or a crypto exchange name "
        "for a 24/7-traded crypto pair) -- set your pick's exchange field to exactly that value. "
        f"Give each pick a budget in USD (default to {cfg.analyst.default_budget} "
        "unless you have a specific reason to size a position differently), an indicators list "
        f"(default {DEFAULT_INDICATORS} unless a symbol warrants different indicators), "
        "and a one-line rationale. "
        "You are also given your own recent track record below: past picks with your rationale "
        "at the time, what the Dealer ultimately decided to do about each, and how execution "
        "went. Use it to spot patterns -- if a rationale pattern has recently preceded a SELL, "
        "no_fill, or error outcome, don't blindly repeat it; the track record may be empty (no "
        "history yet, or the feature is disabled), in which case ignore it. "
        "You're also given a live snapshot of currently-open positions with Alpaca's own "
        "unrealized P&L for each -- a point-in-time read of current holdings, not per-pick "
        "attribution (a symbol can be bought and sold more than once, so a listed position may "
        "not correspond 1:1 to any specific past pick above). Use it to avoid compounding into "
        "an already-losing position, but don't over-interpret it as scored feedback on a "
        "specific rationale; it may be empty (no open positions, or the feature disabled)."
    )
    user_prompt = (
        f"Candidate symbols (from screener):\n{json.dumps(state['raw_candidates'])}\n\n"
        f"Technical indicators (top {cfg.analyst.indicator_fetch_limit} candidates by move size "
        f"only -- not every candidate has these):\n{state['indicator_text']}\n\n"
        f"Market research:\n{state['research_text']}\n\n"
        f"Your recent track record (last {cfg.analyst.track_record_days} days -- past picks, "
        f"what the Dealer decided, and how execution went):\n{state['track_record_text']}\n\n"
        f"Live snapshot of currently-open positions and unrealized P&L (Alpaca's own numbers, "
        f"point-in-time -- not tied to any specific past pick):\n{state['pnl_text']}"
    )

    log("🧠 asking LLM to select the tradeable universe")
    selection: PortfolioSelection = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {**state, "selection": selection.model_dump()}


def validate_selection(state: AnalystState, cfg) -> AnalystState:
    """The LLM's `exchange` field is regenerated text, not guaranteed to match the candidate it
    was actually given -- trust discover_candidates()'s own `market` tag instead, and drop any
    pick whose symbol isn't one we actually offered (a hallucination)."""
    market_by_symbol = {c["symbol"]: c["market"] for c in state["raw_candidates"]}

    validated = []
    for pick in state["selection"]["symbols"]:
        market = market_by_symbol.get(pick["symbol"])
        if market is None:
            log(f"⚠️ dropping hallucinated pick {pick['symbol']} -- not in candidate list")
            continue
        pick["exchange"] = market
        validated.append(pick)

    # Per-pick `budget` has no upper bound of its own (src/analyst/schema.py), and the system
    # prompt's per-pick default is only a suggestion the LLM can ignore -- nothing upstream stops
    # a selection from authorizing far more total new-BUY capital than intended. Cap the sum,
    # dropping trailing picks (in the LLM's own returned order, treated as its priority order)
    # once the running total would exceed the ceiling, rather than silently rewriting any pick's
    # budget.
    max_total_budget = cfg.analyst.max_total_budget_usd
    capped = []
    running_total = 0.0
    for pick in validated:
        if running_total + pick["budget"] > max_total_budget:
            log(
                f"⚠️ dropping pick {pick['symbol']} (budget {pick['budget']}) -- "
                f"would exceed analyst.max_total_budget_usd ({max_total_budget}), "
                f"running total {running_total}"
            )
            continue
        running_total += pick["budget"]
        capped.append(pick)

    return {**state, "selection": {**state["selection"], "symbols": capped}}


def write_portfolio(state: AnalystState, cfg) -> AnalystState:
    selection = state["selection"]
    generated_at = datetime.now(timezone.utc)
    payload = {
        "generated_at": generated_at.isoformat(),
        "symbols": selection["symbols"],
    }
    _write_portfolio(payload)
    log(f"✅ wrote portfolio with {len(payload['symbols'])} symbols")

    for pick in payload["symbols"]:
        db.record_analyst_pick(
            pick["symbol"], pick.get("exchange"), pick.get("budget"), pick.get("rationale"), generated_at
        )

    account = trading_client.get_account()
    account_summary = {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
    }
    report_date = datetime.now(pytz.timezone("US/Eastern")).date().isoformat()
    slack.notify_morning_report(
        report_date,
        account_summary,
        payload["symbols"],
        stock_market_open=state["stock_market_open"],
        crypto_enabled=cfg.trading.crypto.enabled,
        title="Midday Update" if state.get("is_midday_run") else "Morning Market Report",
        emoji="🕐" if state.get("is_midday_run") else "🌅",
    )
    return state


def crypto_eod_report(state: AnalystState, cfg) -> AnalystState:
    """Crypto trades 24/7, so it has no market close to hang an EOD report off of -- instead this
    rides along with the Analyst's morning run and covers the prior full ET calendar day, right
    after today's new picks go out in write_portfolio()'s notify_morning_report(). Skipped
    entirely on a midday run -- it already ran this morning and only ever reports on the prior
    calendar day, so a second run has nothing new to say."""
    if state.get("is_midday_run") or not cfg.trading.crypto.enabled:
        return state

    report_date = (datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=1)).date().isoformat()
    positions = summarize_positions(trading_client.get_all_positions(), only_crypto=True)
    fills = fetch_fills(report_date, only_crypto=True)
    slack.notify_crypto_eod_report(report_date, fills, positions)
    return state


def build_graph():
    # Each lambda calls load_config() fresh at invocation time (once per node per graph run) so a
    # config-only change (disabling analyst.news, analyst.indicators, analyst.track_record, ...) is
    # honored within load_config()'s refresh window, matching the Dealer / Floor Broker hot paths.
    graph = StateGraph(AnalystState)
    graph.add_node("discover_candidates", lambda state: discover_candidates(state, load_config()))
    graph.add_node("fetch_research", lambda state: fetch_research(state, load_config()))
    graph.add_node("fetch_indicators", lambda state: fetch_indicators(state, load_config()))
    graph.add_node("fetch_track_record", lambda state: fetch_track_record(state, load_config()))
    graph.add_node("fetch_position_pnl", lambda state: fetch_position_pnl(state, load_config()))
    graph.add_node("llm_select", lambda state: llm_select(state, load_config()))
    graph.add_node("validate_selection", lambda state: validate_selection(state, load_config()))
    graph.add_node("write_portfolio", lambda state: write_portfolio(state, load_config()))
    graph.add_node("crypto_eod_report", lambda state: crypto_eod_report(state, load_config()))

    graph.set_entry_point("discover_candidates")
    graph.add_edge("discover_candidates", "fetch_research")
    graph.add_edge("fetch_research", "fetch_indicators")
    graph.add_edge("fetch_indicators", "fetch_track_record")
    graph.add_edge("fetch_track_record", "fetch_position_pnl")
    graph.add_edge("fetch_position_pnl", "llm_select")
    graph.add_edge("llm_select", "validate_selection")
    graph.add_edge("validate_selection", "write_portfolio")
    graph.add_edge("write_portfolio", "crypto_eod_report")
    graph.add_edge("crypto_eod_report", END)

    return graph.compile()
