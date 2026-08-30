from datetime import date, datetime

import pytest

from src.common import db


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.queries = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    def __init__(self, rows=None, raise_on_execute=None):
        self.executed = []
        self._rows = rows or []
        self._raise_on_execute = raise_on_execute
        self.last_cursor = None

    def execute(self, sql, params=None):
        if self._raise_on_execute:
            raise self._raise_on_execute
        self.executed.append((sql, params))

    def cursor(self, row_factory=None):
        self.last_cursor = FakeCursor(self._rows)
        return self.last_cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        return self._conn


def _patch_pool(monkeypatch, conn):
    monkeypatch.setattr(db, "_schema_ready", True)
    monkeypatch.setattr(db, "_get_pool", lambda: FakePool(conn))


def test_ensure_schema_runs_only_once(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_get_pool", lambda: FakePool(conn))

    db._ensure_schema()
    db._ensure_schema()

    assert len(conn.executed) == 1
    assert db._schema_ready is True


def test_record_dealer_decision_inserts_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_dealer_decision(
        "MGN",
        "BUY",
        "strong momentum",
        5.0,
        ohlcv_enrichment_active=True,
        cycle_id="cycle-1",
    )

    sql, params = conn.executed[0]
    assert "INSERT INTO dealer_decisions" in sql
    assert params == ("MGN", "BUY", "strong momentum", 5.0, True, "cycle-1")


def test_record_analyst_pick_inserts_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)
    generated_at = datetime(2026, 8, 4, 9, 30)

    db.record_analyst_pick("MGN", "NASDAQ", 100.0, "screener pick", generated_at)

    sql, params = conn.executed[0]
    assert "INSERT INTO analyst_picks" in sql
    assert params == (generated_at, "MGN", "NASDAQ", 100.0, "screener pick")


def test_record_floor_broker_event_inserts_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_floor_broker_event("MGN", "buy_submitted", "order abc123", qty=10, price=5.5)

    sql, params = conn.executed[0]
    assert "INSERT INTO floor_broker_events" in sql
    assert params == ("MGN", "buy_submitted", "order abc123", 10, 5.5)


def test_record_options_trade_opened_inserts_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_options_trade_opened(
        symbol="MGN",
        contract_symbol="MGN260116C00100000",
        right="call",
        strike=100.0,
        expiration="2026-01-16",
        delta=0.45,
        entry_premium=2.50,
        qty=1,
        reasoning="momentum breakout",
        cycle_id="cycle-1",
    )

    sql, params = conn.executed[0]
    assert "INSERT INTO options_trades" in sql
    assert params == (
        "MGN",
        "MGN260116C00100000",
        "call",
        100.0,
        "2026-01-16",
        0.45,
        2.50,
        1,
        "momentum breakout",
        "cycle-1",
    )


def test_record_options_trade_closed_updates_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_options_trade_closed("MGN260116C00100000", "take_profit", 4.00)

    sql, params = conn.executed[0]
    assert "UPDATE options_trades" in sql
    assert "WHERE contract_symbol = %s AND closed_at IS NULL" in sql
    assert params == ("take_profit", 4.00, "MGN260116C00100000")


def test_record_options_trade_updated_updates_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_options_trade_updated("MGN260116C00100000", 3.25, 2)

    sql, params = conn.executed[0]
    assert "UPDATE options_trades" in sql
    assert "WHERE contract_symbol = %s AND closed_at IS NULL" in sql
    assert params == (3.25, 2, "MGN260116C00100000")


def test_record_position_opened_inserts_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_position_opened("MGN")

    sql, params = conn.executed[0]
    assert "INSERT INTO position_opens" in sql
    assert "ON CONFLICT (symbol) DO NOTHING" in sql
    assert params == ("MGN",)


def test_record_position_closed_deletes_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)

    db.record_position_closed("MGN")

    sql, params = conn.executed[0]
    assert "DELETE FROM position_opens" in sql
    assert params == ("MGN",)


def test_record_eod_report_sent_inserts_expected_row(monkeypatch):
    conn = FakeConnection()
    _patch_pool(monkeypatch, conn)
    report_date = date(2026, 8, 6)

    db.record_eod_report_sent(report_date)

    sql, params = conn.executed[0]
    assert "INSERT INTO eod_report_runs" in sql
    assert "ON CONFLICT (report_date) DO NOTHING" in sql
    assert params == (report_date,)


def test_eod_report_already_sent_returns_true_when_row_exists(monkeypatch):
    conn = FakeConnection(rows=[{"?column?": 1}])
    _patch_pool(monkeypatch, conn)

    result = db.eod_report_already_sent(date(2026, 8, 6))

    assert result is True


def test_eod_report_already_sent_returns_false_when_no_row_exists(monkeypatch):
    conn = FakeConnection(rows=[])
    _patch_pool(monkeypatch, conn)

    result = db.eod_report_already_sent(date(2026, 8, 6))

    assert result is False


def test_eod_report_already_sent_fails_open_on_db_error(monkeypatch):
    monkeypatch.setattr(db, "_ensure_schema", lambda: (_ for _ in ()).throw(RuntimeError("db is down")))

    result = db.eod_report_already_sent(date(2026, 8, 6))

    assert result is False


def test_fetch_position_opened_at_returns_timestamp_when_tracked(monkeypatch):
    opened_at = datetime(2026, 8, 1, 9, 30)
    conn = FakeConnection(rows=[{"opened_at": opened_at}])
    _patch_pool(monkeypatch, conn)

    result = db.fetch_position_opened_at("MGN")

    assert result == opened_at


def test_fetch_position_opened_at_returns_none_when_untracked(monkeypatch):
    conn = FakeConnection(rows=[])
    _patch_pool(monkeypatch, conn)

    result = db.fetch_position_opened_at("MGN")

    assert result is None


@pytest.mark.parametrize(
    "record_fn,args",
    [
        (db.record_analyst_pick, ("MGN", "NASDAQ", 100.0, "rationale", datetime(2026, 8, 4))),
        (db.record_dealer_decision, ("MGN", "BUY", "reasoning", 5.0)),
        (db.record_floor_broker_event, ("MGN", "error", "boom")),
        (
            db.record_options_trade_opened,
            ("MGN", "MGN260116C00100000", "call", 100.0, "2026-01-16", 0.45, 2.50, 1, "reasoning", "cycle-1"),
        ),
        (db.record_options_trade_closed, ("MGN260116C00100000", "take_profit", 4.00)),
        (db.record_options_trade_updated, ("MGN260116C00100000", 3.25, 2)),
        (db.record_position_opened, ("MGN",)),
        (db.record_position_closed, ("MGN",)),
        (db.record_eod_report_sent, (date(2026, 8, 6),)),
    ],
)
def test_write_functions_swallow_exceptions(monkeypatch, record_fn, args):
    conn = FakeConnection(raise_on_execute=RuntimeError("db is down"))
    _patch_pool(monkeypatch, conn)

    record_fn(*args)  # must not raise


def test_fetch_dealer_decisions_for_date_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "action": "BUY", "reasoning": "momentum"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_dealer_decisions_for_date(date(2026, 8, 4))

    assert result == rows


def test_fetch_analyst_picks_for_date_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "rationale": "screener pick"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_analyst_picks_for_date(date(2026, 8, 4))

    assert result == rows


def test_fetch_floor_broker_events_for_date_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "event_type": "buy_submitted"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_floor_broker_events_for_date(date(2026, 8, 4))

    assert result == rows


def test_fetch_analyst_picks_since_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "rationale": "screener pick"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_analyst_picks_since(date(2026, 8, 1))

    assert result == rows


def test_fetch_dealer_decisions_since_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "action": "BUY", "reasoning": "momentum"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_dealer_decisions_since(date(2026, 8, 1))

    assert result == rows


def test_fetch_floor_broker_events_since_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "event_type": "buy_submitted"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_floor_broker_events_since(date(2026, 8, 1))

    assert result == rows


def test_fetch_symbol_dealer_decisions_since_filters_symbol_and_limits(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "action": "BUY"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_symbol_dealer_decisions_since("MGN", date(2026, 8, 1), limit=7)

    assert result == rows
    sql, params = conn.last_cursor.queries[-1]
    assert "WHERE symbol = %s" in sql
    assert params == ("MGN", date(2026, 8, 1), 7)


def test_fetch_symbol_floor_broker_events_since_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "event_type": "fill"}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_symbol_floor_broker_events_since("MGN", date(2026, 8, 1), limit=5)

    assert result == rows
    sql, params = conn.last_cursor.queries[-1]
    assert "WHERE symbol = %s" in sql
    assert params == ("MGN", date(2026, 8, 1), 5)


def test_fetch_open_options_trades_returns_list_of_dicts(monkeypatch):
    rows = [{"id": 1, "symbol": "MGN", "contract_symbol": "MGN260116C00100000", "closed_at": None}]
    conn = FakeConnection(rows=rows)
    _patch_pool(monkeypatch, conn)

    result = db.fetch_open_options_trades()

    assert result == rows
    sql, params = conn.last_cursor.queries[-1]
    assert "SELECT * FROM options_trades WHERE closed_at IS NULL" in sql
