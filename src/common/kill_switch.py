import os

from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException

NAMESPACE = os.getenv("POD_NAMESPACE", "multi-agent-ai-trader")
CONFIGMAP_NAME = "buy-kill-switch"
DATA_KEY = "active"


def _load_k8s_config() -> None:
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()


def buy_kill_switch_active() -> bool:
    """Reads the `buy-kill-switch` ConfigMap fresh on every call (ROADMAP P0.5) -- no caching, so
    an operator's kubectl patch takes effect on the very next /execute BUY request. The deploy
    workflow always seeds this ConfigMap once (mirroring the `portfolio` ConfigMap), so a missing
    ConfigMap means it was never seeded rather than deliberately activated -- treated as inactive
    (fail open) rather than blocking all BUYs on a setup gap."""
    _load_k8s_config()
    v1 = client.CoreV1Api()
    try:
        cm = v1.read_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE)
    except ApiException as exc:
        if exc.status == 404:
            return False
        raise
    return (cm.data or {}).get(DATA_KEY, "false").strip().lower() == "true"
