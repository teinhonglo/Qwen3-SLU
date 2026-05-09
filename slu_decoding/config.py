import json
import os


def load_dexperts_config(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DExperts config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a JSON object")
    return cfg
