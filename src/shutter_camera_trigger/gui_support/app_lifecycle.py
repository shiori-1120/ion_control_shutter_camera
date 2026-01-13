from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import tkinter.font as tkfont
from tkinter import ttk

from .camera_prefs import load_camera_trigger_prefs, save_camera_trigger_prefs


def apply_default_fonts(app: Any, *, size: int) -> None:
    """Increase default Tk/ttk font sizes for readability."""
    if size <= 0:
        return

    for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkTooltipFont", "TkFixedFont"):
        try:
            f = tkfont.nametofont(name)
            f.configure(size=int(size))
        except Exception:
            pass

    try:
        style = ttk.Style(app)
        style.configure(".", font=tkfont.nametofont("TkDefaultFont"))
    except Exception:
        pass


def load_camera_prefs(app: Any, *, prefs_path: Path) -> None:
    load_camera_trigger_prefs(app, prefs_path=prefs_path)


def save_camera_prefs(app: Any, *, prefs_path: Path) -> None:
    save_camera_trigger_prefs(app, prefs_path=prefs_path)


def on_close(
    app: Any,
    *,
    prefs_path: Path,
    stop_sweep_cb: Callable[[], None],
    disconnect_daq_cb: Callable[[], None],
    disconnect_fg_cb: Callable[[], None],
) -> None:
    try:
        save_camera_prefs(app, prefs_path=prefs_path)
    except Exception:
        pass
    try:
        stop_sweep_cb()
    except Exception:
        pass
    try:
        disconnect_daq_cb()
    except Exception:
        pass
    try:
        disconnect_fg_cb()
    except Exception:
        pass
    app.destroy()
