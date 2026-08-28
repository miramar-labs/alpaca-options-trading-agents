import sys


def get_logger(prefix: str):
    """Emoji fprint()-style logger, ported from gpt-trader.py minus the app.log file half —
    k8s convention is stdout only, captured by `kubectl logs`."""

    def log(text: str) -> None:
        print(f"[{prefix}] {text}", flush=True, file=sys.stdout)

    return log
