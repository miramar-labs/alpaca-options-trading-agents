import os
from datetime import date, datetime

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.common.logging import get_logger

log = get_logger("DB")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_picks (
    id SERIAL PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT,
    budget NUMERIC,
    rationale TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dealer_decisions (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    reasoning TEXT,
    size_hint NUMERIC,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE dealer_decisions
    ADD COLUMN IF NOT EXISTS ohlcv_enrichment_active BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE dealer_decisions
    ADD COLUMN IF NOT EXISTS cycle_id TEXT;

CREATE TABLE IF NOT EXISTS floor_broker_events (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT,
    qty NUMERIC,
    price NUMERIC,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS options_trades (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    contract_symbol TEXT NOT NULL,
    "right" TEXT NOT NULL,
    strike NUMERIC NOT NULL,
    expiration DATE NOT NULL,
    delta NUMERIC,
    entry_premium NUMERIC NOT NULL,
    qty INTEGER NOT NULL,
    reasoning TEXT,
    cycle_id TEXT,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ,
    exit_reason TEXT,
    exit_premium NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_options_trades_contract_symbol ON options_trades (contract_symbol);

-- Tracks the timestamp a currently-open stock/crypto position was (re)opened, keyed by symbol.
-- Populated from poll_pending_fills() (src/floor_broker/main.py) on a BUY fill, and cleared on a
-- SELL fill -- sell() always closes the entire current position, so there is no partial-lot case
-- to model. Used by check_eod_flatten()'s conditional mode to compute days-held for its
-- max_days_held_loss cap.
CREATE TABLE IF NOT EXISTS position_opens (
    symbol TEXT PRIMARY KEY,
    opened_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS eod_report_runs (
    report_date DATE PRIMARY KEY,
    sent_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dealer_decisions_symbol_date ON dealer_decisions (symbol, decided_at);
CREATE INDEX IF NOT EXISTS idx_dealer_decisions_cycle_id ON dealer_decisions (cycle_id);
CREATE INDEX IF NOT EXISTS idx_floor_broker_events_symbol_date ON floor_broker_events (symbol, occurred_at);
"""

_pool: ConnectionPool | None = None
_schema_ready = False


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        database_url = os.environ["DATABASE_URL"]
        _pool = ConnectionPool(database_url, min_size=1, max_size=5, open=True)
    return _pool


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _get_pool().connection() as conn:
        conn.execute(_SCHEMA)
    _schema_ready = True


def record_analyst_pick(
    symbol: str,
    exchange: str | None,
    budget: float | None,
    rationale: str | None,
    generated_at: datetime,
) -> None:
    """Fire-and-forget insert -- never raises, so a DB outage can't block the Analyst run."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO analyst_picks (generated_at, symbol, exchange, budget, rationale)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (generated_at, symbol, exchange, budget, rationale),
            )
    except Exception as exc:
        log(f"⚠️ record_analyst_pick failed: {exc}")


def record_dealer_decision(
    symbol: str,
    action: str,
    reasoning: str | None,
    size_hint: float | None,
    *,
    ohlcv_enrichment_active: bool = False,
    cycle_id: str | None = None,
) -> None:
    """Fire-and-forget insert -- never raises, so a DB outage can't block a Dealer decision."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO dealer_decisions (
                    symbol, action, reasoning, size_hint, ohlcv_enrichment_active, cycle_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (symbol, action, reasoning, size_hint, ohlcv_enrichment_active, cycle_id),
            )
    except Exception as exc:
        log(f"⚠️ record_dealer_decision failed: {exc}")


def record_floor_broker_event(
    symbol: str,
    event_type: str,
    detail: str | None,
    qty: float | None = None,
    price: float | None = None,
) -> None:
    """Fire-and-forget insert -- never raises, so a DB outage can't block order execution."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO floor_broker_events (symbol, event_type, detail, qty, price)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (symbol, event_type, detail, qty, price),
            )
    except Exception as exc:
        log(f"⚠️ record_floor_broker_event failed: {exc}")


def record_options_trade_opened(
    symbol: str,
    contract_symbol: str,
    right: str,
    strike: float,
    expiration: str,
    delta: float | None,
    entry_premium: float,
    qty: int,
    reasoning: str | None,
    cycle_id: str | None,
) -> None:
    """Fire-and-forget insert -- never raises, so a DB outage can't block option order submission."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO options_trades (
                    symbol, contract_symbol, "right", strike, expiration, delta, entry_premium, qty, reasoning, cycle_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (symbol, contract_symbol, right, strike, expiration, delta, entry_premium, qty, reasoning, cycle_id),
            )
    except Exception as exc:
        log(f"⚠️ record_options_trade_opened failed: {exc}")


def record_options_trade_closed(contract_symbol: str, exit_reason: str, exit_premium: float | None) -> None:
    """Fire-and-forget update -- never raises. Closes the most recent still-open row for this
    contract_symbol; a contract symbol is unique to one strike/expiration/right, so at most one
    open row can exist for it at a time."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE options_trades
                SET closed_at = now(), exit_reason = %s, exit_premium = %s
                WHERE contract_symbol = %s AND closed_at IS NULL
                """,
                (exit_reason, exit_premium, contract_symbol),
            )
    except Exception as exc:
        log(f"⚠️ record_options_trade_closed failed: {exc}")


def record_options_trade_updated(contract_symbol: str, entry_premium: float, qty: int) -> None:
    """Fire-and-forget update -- never raises. Updates the still-open row for this contract_symbol
    with the latest cumulative fill price/qty as a partially-filled option BUY order continues to
    fill; a contract symbol is unique to one strike/expiration/right, so at most one open row can
    exist for it at a time (same uniqueness assumption record_options_trade_closed relies on)."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE options_trades
                SET entry_premium = %s, qty = %s
                WHERE contract_symbol = %s AND closed_at IS NULL
                """,
                (entry_premium, qty, contract_symbol),
            )
    except Exception as exc:
        log(f"⚠️ record_options_trade_updated failed: {exc}")


def fetch_open_options_trades() -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM options_trades WHERE closed_at IS NULL ORDER BY opened_at")
            return cur.fetchall()


def record_position_opened(symbol: str) -> None:
    """Fire-and-forget upsert -- never raises. ON CONFLICT DO NOTHING means a BUY that adds to an
    already-open position never resets the clock; only a symbol's first BUY since it was last
    fully closed (or never tracked) starts a new opened_at."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO position_opens (symbol, opened_at)
                VALUES (%s, now())
                ON CONFLICT (symbol) DO NOTHING
                """,
                (symbol,),
            )
    except Exception as exc:
        log(f"⚠️ record_position_opened failed: {exc}")


def record_position_closed(symbol: str) -> None:
    """Fire-and-forget delete -- never raises. Called on a SELL fill, which always closes the
    entire current position (execution.sell() sells the full open qty)."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute("DELETE FROM position_opens WHERE symbol = %s", (symbol,))
    except Exception as exc:
        log(f"⚠️ record_position_closed failed: {exc}")


def eod_report_already_sent(report_date: date) -> bool:
    """Best-effort duplicate check. A DB outage must not prevent the Slack recap from posting."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT 1 FROM eod_report_runs WHERE report_date = %s", (report_date,))
                return cur.fetchone() is not None
    except Exception as exc:
        log(f"⚠️ eod_report_already_sent failed: {exc}")
        return False


def record_eod_report_sent(report_date: date) -> None:
    """Fire-and-forget marker -- never raises, so a DB outage can't block the EOD report."""
    try:
        _ensure_schema()
        with _get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO eod_report_runs (report_date)
                VALUES (%s)
                ON CONFLICT (report_date) DO NOTHING
                """,
                (report_date,),
            )
    except Exception as exc:
        log(f"⚠️ record_eod_report_sent failed: {exc}")


def fetch_position_opened_at(symbol: str) -> datetime | None:
    """Returns None if the symbol has no tracked open position -- e.g. a position that existed
    before this feature shipped and hasn't gone through a BUY fill since."""
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT opened_at FROM position_opens WHERE symbol = %s", (symbol,))
            row = cur.fetchone()
            return row["opened_at"] if row else None


def fetch_analyst_picks_for_date(for_date: date) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM analyst_picks WHERE generated_at::date = %s ORDER BY generated_at",
                (for_date,),
            )
            return cur.fetchall()


def fetch_dealer_decisions_for_date(for_date: date) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM dealer_decisions WHERE decided_at::date = %s ORDER BY decided_at",
                (for_date,),
            )
            return cur.fetchall()


def fetch_floor_broker_events_for_date(for_date: date) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM floor_broker_events WHERE occurred_at::date = %s ORDER BY occurred_at",
                (for_date,),
            )
            return cur.fetchall()


def fetch_analyst_picks_since(since_date: date) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM analyst_picks WHERE generated_at::date >= %s ORDER BY generated_at",
                (since_date,),
            )
            return cur.fetchall()


def fetch_dealer_decisions_since(since_date: date) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM dealer_decisions WHERE decided_at::date >= %s ORDER BY decided_at",
                (since_date,),
            )
            return cur.fetchall()


def fetch_floor_broker_events_since(since_date: date) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM floor_broker_events WHERE occurred_at::date >= %s ORDER BY occurred_at",
                (since_date,),
            )
            return cur.fetchall()


def fetch_symbol_dealer_decisions_since(symbol: str, since_date: date, limit: int = 20) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM dealer_decisions
                WHERE symbol = %s AND decided_at::date >= %s
                ORDER BY decided_at DESC
                LIMIT %s
                """,
                (symbol, since_date, limit),
            )
            return cur.fetchall()


def fetch_symbol_floor_broker_events_since(symbol: str, since_date: date, limit: int = 20) -> list[dict]:
    _ensure_schema()
    with _get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM floor_broker_events
                WHERE symbol = %s AND occurred_at::date >= %s
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (symbol, since_date, limit),
            )
            return cur.fetchall()
