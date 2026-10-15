from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    base = config_path.parent
    for key, value in config.get("paths", {}).items():
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        config["paths"][key] = str(candidate)
    return config, config_path


def select_device(requested: str | None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

