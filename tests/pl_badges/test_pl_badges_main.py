import json
from datetime import datetime

import pytz

from src.pl_badges import main

# 20:25 ET on 2026-08-13 is already 00:25 UTC on 2026-08-14 -- exercises the EDT/UTC-rollover
# window a UTC-clocked `date.today()` used to mis-date (see regression test below).
_LATE_ET_INSTANT = pytz.timezone("US/Eastern").localize(datetime(2026, 8, 13, 20, 25))


def _use_tmp_badges_dir(monkeypatch, tmp_path):
    badges_dir = tmp_path / "badges"
    monkeypatch.setattr(main, "BADGES_DIR", badges_dir)
    monkeypatch.setattr(main, "HISTORY_FILE", badges_dir / "pl_history.json")
    return badges_dir


def test_market_closed_leaves_badges_dir_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_now_eastern", lambda: _LATE_ET_INSTANT)
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: False)
    badges_dir = _use_tmp_badges_dir(monkeypatch, tmp_path)

    main.main()

    assert not badges_dir.exists()


def test_open_market_writes_both_badge_files(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_now_eastern", lambda: _LATE_ET_INSTANT)
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    badges_dir = _use_tmp_badges_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "fetch_pl_summary", lambda today, history_pl: {"today_pl": 50.0, "ytd_pl": -100.0})

    main.main()

    today = json.loads((badges_dir / "today-pl.json").read_text())
    ytd = json.loads((badges_dir / "ytd-pl.json").read_text())
    assert today == {"schemaVersion": 1, "label": "Today's P/L", "message": "+$50.00", "color": "brightgreen"}
    assert ytd == {"schemaVersion": 1, "label": "YTD P/L", "message": "-$100.00", "color": "red"}


def test_open_market_persists_todays_pl_into_the_history_file(monkeypatch, tmp_path):
    badges_dir = _use_tmp_badges_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "_now_eastern", lambda: _LATE_ET_INSTANT)
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    monkeypatch.setattr(main, "fetch_pl_summary", lambda today, history_pl: {"today_pl": 50.0, "ytd_pl": -100.0})

    main.main()

    history = json.loads((badges_dir / "pl_history.json").read_text())
    assert history == {"2026-08-13": 50.0}


def test_open_market_merges_todays_pl_with_existing_history(monkeypatch, tmp_path):
    badges_dir = _use_tmp_badges_dir(monkeypatch, tmp_path)
    badges_dir.mkdir()
    (badges_dir / "pl_history.json").write_text(json.dumps({"2026-08-05": -100.0, "2026-08-06": 50.0}))
    monkeypatch.setattr(main, "_now_eastern", lambda: _LATE_ET_INSTANT)
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    received = {}

    def _fake_fetch(today, history_pl):
        received["history_pl"] = dict(history_pl)
        return {"today_pl": -20.0, "ytd_pl": -70.0}

    monkeypatch.setattr(main, "fetch_pl_summary", _fake_fetch)

    main.main()

    assert received["history_pl"] == {"2026-08-05": -100.0, "2026-08-06": 50.0}
    history = json.loads((badges_dir / "pl_history.json").read_text())
    assert history == {"2026-08-05": -100.0, "2026-08-06": 50.0, "2026-08-13": -20.0}


def test_history_dates_by_eastern_trading_day_not_utc_rollover(monkeypatch, tmp_path):
    """Regression test: a run at 20:25 ET (already 00:25 UTC the next day) must file today's
    P/L under the Eastern trading date (2026-08-13), not the UTC date (2026-08-14) --
    this is exactly what happened in production before switching off `date.today()`."""
    badges_dir = _use_tmp_badges_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "_now_eastern", lambda: _LATE_ET_INSTANT)
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    monkeypatch.setattr(main, "fetch_pl_summary", lambda today, history_pl: {"today_pl": 154.77, "ytd_pl": 84.35})

    main.main()

    history = json.loads((badges_dir / "pl_history.json").read_text())
    assert "2026-08-13" in history
    assert "2026-08-14" not in history
