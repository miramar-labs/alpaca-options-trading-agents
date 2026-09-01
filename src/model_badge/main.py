import json
from pathlib import Path

import yaml

from src.common.logging import get_logger

log = get_logger("MODEL-BADGE")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.yaml"
BADGES_DIR = REPO_ROOT / "badges"


def main():
    config = yaml.safe_load(CONFIG_FILE.read_text())
    model = config["llm"]["model"]

    BADGES_DIR.mkdir(exist_ok=True)
    payload = {"schemaVersion": 1, "label": "Dealer LLM", "message": model, "color": "blue"}
    (BADGES_DIR / "model.json").write_text(json.dumps(payload))

    log(f"✅ model badge written — {model}")


if __name__ == "__main__":
    main()
