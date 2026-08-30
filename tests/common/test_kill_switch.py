from kubernetes.client.exceptions import ApiException

from src.common import kill_switch


class FakeConfigMap:
    def __init__(self, data):
        self.data = data


class FakeCoreV1Api:
    def __init__(self, result):
        self._result = result

    def read_namespaced_config_map(self, name, namespace):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _use(monkeypatch, result):
    monkeypatch.setattr(kill_switch, "_load_k8s_config", lambda: None)
    monkeypatch.setattr(kill_switch.client, "CoreV1Api", lambda: FakeCoreV1Api(result))


def test_active_when_data_key_is_true(monkeypatch):
    _use(monkeypatch, FakeConfigMap({"active": "true"}))

    assert kill_switch.buy_kill_switch_active() is True


def test_inactive_when_data_key_is_false(monkeypatch):
    _use(monkeypatch, FakeConfigMap({"active": "false"}))

    assert kill_switch.buy_kill_switch_active() is False


def test_value_comparison_is_case_and_whitespace_insensitive(monkeypatch):
    _use(monkeypatch, FakeConfigMap({"active": " TRUE "}))

    assert kill_switch.buy_kill_switch_active() is True


def test_inactive_when_data_key_is_absent(monkeypatch):
    _use(monkeypatch, FakeConfigMap({}))

    assert kill_switch.buy_kill_switch_active() is False


def test_inactive_when_configmap_not_yet_seeded(monkeypatch):
    """A missing ConfigMap (404) means the deploy workflow's seed step hasn't run -- a setup
    gap, not a deliberate activation -- so this fails open (inactive) rather than blocking every
    BUY."""
    not_found = ApiException(status=404)
    _use(monkeypatch, not_found)

    assert kill_switch.buy_kill_switch_active() is False


def test_reraises_on_non_404_api_error(monkeypatch):
    """Any other k8s API failure (e.g. RBAC misconfigured, apiserver unreachable) must propagate
    rather than silently failing open -- unlike a 404, that's not the expected/seeded case."""
    forbidden = ApiException(status=403)
    _use(monkeypatch, forbidden)

    try:
        kill_switch.buy_kill_switch_active()
        raise AssertionError("expected ApiException to propagate")
    except ApiException as exc:
        assert exc.status == 403
