from omegaconf import OmegaConf

from src.analyst import main


class FakeGraph:
    def __init__(self, captured):
        self._captured = captured

    def invoke(self, state, config=None):
        self._captured["state"] = state
        self._captured["config"] = config
        return {"selection": {"symbols": []}}


def _cfg(enable_midday_run=False):
    return OmegaConf.create({"analyst": {"midday_run": {"enabled": enable_midday_run}}})


def test_main_threads_stock_market_open_true_into_initial_state(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph(captured))
    monkeypatch.setattr(main, "load_config", lambda: _cfg())
    monkeypatch.setattr(main.langsmith, "configure", lambda cfg: None)

    main.main()

    assert captured["state"]["stock_market_open"] is True


def test_main_threads_stock_market_open_false_into_initial_state_on_a_closed_day(monkeypatch):
    """Regression: Analyst must not skip its whole run on a closed stock market day -- crypto
    still trades 24/7 -- so the closed status is threaded into the graph state rather than
    short-circuiting main() the way EOD Report does."""
    captured = {}
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: False)
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph(captured))
    monkeypatch.setattr(main, "load_config", lambda: _cfg())
    monkeypatch.setattr(main.langsmith, "configure", lambda cfg: None)

    main.main()

    assert captured["state"]["stock_market_open"] is False


def test_main_exits_before_building_the_graph_when_midday_run_disabled_via_config(monkeypatch):
    """The midday CronJob always fires on schedule -- the feature gate must short-circuit main()
    entirely (before build_graph()/graph.invoke() are even called) when
    analyst.midday_run.enabled is false, not just skip a downstream step."""
    monkeypatch.setenv("ANALYST_RUN_LABEL", "midday")
    monkeypatch.setattr(main, "load_config", lambda: _cfg(enable_midday_run=False))
    monkeypatch.setattr(main.langsmith, "configure", lambda cfg: None)

    def _raise():
        raise AssertionError("build_graph() must not be called when the midday run is disabled")

    monkeypatch.setattr(main, "build_graph", _raise)

    main.main()


def test_main_runs_and_tags_midday_when_midday_run_enabled_via_config(monkeypatch):
    captured = {}
    monkeypatch.setenv("ANALYST_RUN_LABEL", "midday")
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph(captured))
    monkeypatch.setattr(main, "load_config", lambda: _cfg(enable_midday_run=True))
    monkeypatch.setattr(main.langsmith, "configure", lambda cfg: None)

    main.main()

    assert captured["state"]["is_midday_run"] is True
    assert captured["config"]["tags"] == ["analyst", "midday"]


def test_main_threads_is_midday_run_false_and_tags_plain_on_the_morning_run(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "is_stock_market_open", lambda day: True)
    monkeypatch.setattr(main, "build_graph", lambda: FakeGraph(captured))
    monkeypatch.setattr(main, "load_config", lambda: _cfg())
    monkeypatch.setattr(main.langsmith, "configure", lambda cfg: None)

    main.main()

    assert captured["state"]["is_midday_run"] is False
    assert captured["config"]["tags"] == ["analyst"]
