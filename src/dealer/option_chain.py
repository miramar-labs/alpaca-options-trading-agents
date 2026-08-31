"""Pure helpers for bounding the Dealer's option-selection MCP loop.

The Alpaca MCP `get_option_chain` / `get_option_snapshot` tools return the raw
Alpaca options-snapshot JSON, which can carry hundreds of contracts. Dumping that
straight into the LangChain message history (and re-sending it every tool-calling
round) is what hard-hung the DGX on 2026-08-27. `compact_tool_result` shrinks each
payload to the handful of fields the selector actually needs; `parse_option_chain`
exposes the same rows structurally for the deterministic fallback pick.

Both first peel the alpaca-mcp-server trust-boundary envelope (see
`_unwrap_security_envelope`) -- without that the real `snapshots` map stays hidden
under `data`, every contract row comes back empty, and the fallback pick starves.
"""

import json
import re

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
_CHAIN_TOOLS = {"get_option_chain", "get_option_snapshot"}

# alpaca-mcp-server >=0.x wraps every tool result in a trust-boundary envelope
# (alpaca_mcp_server.security.TrustBoundaryMiddleware): the real payload moves
# under `data`, alongside a `_alpaca_mcp_security` metadata block. langchain-mcp
# -adapters passes that joined text through verbatim, so the Dealer's option loop
# sees the wrapped JSON. Unwrap it before looking for `snapshots`.
_SECURITY_KEY = "_alpaca_mcp_security"
_DATA_KEY = "data"


def _unwrap_security_envelope(data):
    """Peel the alpaca-mcp-server trust-boundary envelope, if present. Non-enveloped
    payloads pass through untouched. The `data` payload can itself be a `{"text": "<json>"}`
    fallback block (non-structured tool output) -- parse that one level deeper too."""
    for _ in range(2):
        if not (isinstance(data, dict) and _SECURITY_KEY in data and _DATA_KEY in data):
            break
        data = data[_DATA_KEY]
        if isinstance(data, dict) and set(data) == {"text"} and isinstance(data["text"], str):
            try:
                data = json.loads(data["text"])
            except (TypeError, ValueError):
                break
    return data


def parse_occ_symbol(sym: str) -> tuple[str, str, str, float] | None:
    """`AAPL250117C00150000` -> `("AAPL", "2025-01-17", "call", 150.0)`; None if it doesn't match."""
    m = _OCC_RE.match(sym.strip())
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    expiration = f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}"
    right = "call" if cp == "C" else "put"
    return root, expiration, right, int(strike) / 1000.0


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_option_chain(raw: str) -> list[dict]:
    """Raw MCP option-snapshot JSON -> list of flat contract dicts. `[]` on any parse failure."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    data = _unwrap_security_envelope(data)
    if not isinstance(data, dict):
        return []
    snapshots = data.get("snapshots", data)
    if not isinstance(snapshots, dict):
        return []

    rows: list[dict] = []
    for sym, snap in snapshots.items():
        if not isinstance(snap, dict):
            continue
        occ = parse_occ_symbol(str(sym))
        greeks = snap.get("greeks") or {}
        quote = snap.get("latestQuote") or {}
        daily = snap.get("dailyBar") or snap.get("minuteBar") or {}
        rows.append(
            {
                "symbol": str(sym),
                "strike": occ[3] if occ else _num(snap.get("strikePrice")),
                "expiration": occ[1] if occ else snap.get("expirationDate"),
                "right": occ[2] if occ else None,
                "delta": _num(greeks.get("delta")),
                "gamma": _num(greeks.get("gamma")),
                "theta": _num(greeks.get("theta")),
                "vega": _num(greeks.get("vega")),
                "iv": _num(snap.get("impliedVolatility")),
                "bid": _num(quote.get("bp")),
                "ask": _num(quote.get("ap")),
                "oi": _num(snap.get("openInterest")),
                "volume": _num(daily.get("v")),
            }
        )
    return rows


def _truncate(raw: str, max_chars: int) -> str:
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n… [truncated]"


def _fmt(v, nd=2):
    return "" if v is None else f"{v:.{nd}f}"


def compact_tool_result(
    tool_name: str,
    raw: str,
    *,
    target_delta_mid: float = 0.45,
    max_contracts: int = 40,
    max_chars: int = 6000,
) -> str:
    """Shrink a raw MCP tool result to a short string safe to append to the message history."""
    if tool_name not in _CHAIN_TOOLS:
        return _truncate(raw, max_chars)
    rows = parse_option_chain(raw)
    if not rows:
        return _truncate(raw, max_chars)

    rows.sort(
        key=lambda r: abs(abs(r["delta"]) - target_delta_mid) if r["delta"] is not None else 9e9
    )
    kept = rows[:max_contracts]
    omitted = len(rows) - len(kept)

    lines = [
        f"{r['symbol']} K={_fmt(r['strike'])} exp={r['expiration']} {r['right'] or '?'} "
        f"d={_fmt(r['delta'])} g={_fmt(r['gamma'], 3)} th={_fmt(r['theta'], 3)} "
        f"v={_fmt(r['vega'], 3)} iv={_fmt(r['iv'], 3)} bid={_fmt(r['bid'])} ask={_fmt(r['ask'])} "
        f"oi={_fmt(r['oi'], 0)} vol={_fmt(r['volume'], 0)}"
        for r in kept
    ]
    if omitted:
        lines.append(f"… {omitted} more contracts omitted — narrow type/expiration_date/strike filters")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else _truncate(out, max_chars)


def estimate_tokens(messages: list) -> int:
    """Cheap upper-ish bound on prompt size: total content chars / 4. No tokenizer dependency."""
    return sum(len(str(getattr(m, "content", m))) for m in messages) // 4
