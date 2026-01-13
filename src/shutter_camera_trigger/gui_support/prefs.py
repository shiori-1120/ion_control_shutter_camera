from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_repo_relative_path(module_file: str, relative: Path) -> Path:
    """Resolve a repo-relative path (best-effort).

    If resolution fails, returns the input relative path.
    """

    try:
        root = Path(module_file).resolve().parents[2]
        return root / relative
    except Exception:
        return relative


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
