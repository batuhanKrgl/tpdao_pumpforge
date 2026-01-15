import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict:
    data = _load_json_file(path)
    includes = data.pop("__include__", [])
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            candidate = (path.parent / include).resolve()
            if candidate.exists():
                include_path = candidate
            else:
                include_path = (Path.cwd() / include).resolve()
        merged = _deep_merge(merged, load_json(include_path))
    merged = _deep_merge(merged, data)
    return merged
