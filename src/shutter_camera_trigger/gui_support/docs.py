from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import subprocess


def open_usage_doc(app: Any, *, source_file: str) -> None:
    try:
        doc_path = Path(source_file).resolve().parents[2] / "docs" / "shutter_gui_usage.md"
        if not doc_path.exists():
            return
        if os.name == "nt":
            subprocess.Popen(["cmd", "/c", "start", str(doc_path)])
        else:
            subprocess.Popen(["xdg-open", str(doc_path)])
    except Exception:
        try:
            app.status_var.set("Failed to open usage doc")
        except Exception:
            pass