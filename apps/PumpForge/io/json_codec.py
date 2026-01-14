import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return _load_json_with_includes(path, visited=set())


def _load_json_with_includes(path: Path, visited: set[Path]) -> Dict[str, Any]:
    path = path.resolve()
    if path in visited:
        raise ValueError(f"Circular __include__ detected for {path}")
    visited.add(path)

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    includes = payload.pop("__include__", [])
    merged: Dict[str, Any] = {}
    for include_path in _normalize_includes(path, includes):
        included_payload = _load_json_with_includes(include_path, visited)
        merged = merge_dicts(merged, included_payload)

    merged = merge_dicts(merged, payload)
    visited.remove(path)
    return merged


def _normalize_includes(base_path: Path, includes: Iterable[str]) -> list[Path]:
    resolved = []
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = (base_path.parent / include_path).resolve()
        resolved.append(include_path)
    return resolved


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "__dict__"):
        return to_jsonable(obj.__dict__)
    return obj


def merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
