from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Turn a path from the config into an absolute path."""
    return PROJECT_ROOT / rel_path