"""Configuration loading.

`studio.config.json` holds the shared defaults and is committed.
`studio.config.local.json` holds machine-specific paths and is git-ignored.
The local file is deep-merged over the defaults.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "studio.config.json"
LOCAL_FILE = ROOT / "studio.config.local.json"


def _deep_merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load(default_file=DEFAULT_FILE, local_file=LOCAL_FILE):
    cfg = json.loads(Path(default_file).read_text(encoding="utf-8"))
    if Path(local_file).exists():
        cfg = _deep_merge(cfg, json.loads(Path(local_file).read_text(encoding="utf-8")))
    return cfg


def resolve(path_str):
    """Resolve a config path relative to the project root."""
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else ROOT / p
