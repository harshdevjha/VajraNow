import os
from pathlib import Path

import yaml

CONFIG_PATH = Path(
    os.getenv("VAJRANOW_CONFIG", Path(__file__).resolve().parents[1] / "config" / "prototype.yaml")
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
