import pytest

from src.dealer import graph

_REAL_EXPOSURE_CHECK = graph._option_contract_already_exposed


@pytest.fixture(autouse=True)
def _stub_option_exposure(request, monkeypatch):
    """call_floor_broker_option() consults the Floor Broker's /option-exposure endpoint as its very
    first gate (before the dealer-signal Slack line). Default every dealer test to "not exposed" so
    the unrelated gate tests don't attempt a real HTTP call. Tests that drive the dedup path re-stub
    graph._option_contract_already_exposed themselves; tests of the helper itself request the
    `real_option_exposure` fixture to opt back out."""
    if "real_option_exposure" in request.fixturenames:
        return
    monkeypatch.setattr(graph, "_option_contract_already_exposed", lambda cfg, sym: False)


@pytest.fixture
def real_option_exposure(monkeypatch):
    monkeypatch.setattr(graph, "_option_contract_already_exposed", _REAL_EXPOSURE_CHECK)
