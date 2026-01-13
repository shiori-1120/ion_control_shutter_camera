from __future__ import annotations

from typing import Any

from .prefs import read_json, write_json


def load_camera_trigger_prefs(app: Any, *, prefs_path) -> None:
    data = read_json(prefs_path)
    if not isinstance(data, dict):
        return

    trig = data.get("camera_trigger")
    if isinstance(trig, dict):
        for key, var in (
            ("source", app.camera_trigger_source_var),
            ("connector", app.camera_trigger_connector_var),
            ("polarity", app.camera_trigger_polarity_var),
            ("active", app.camera_trigger_active_var),
            ("mode", app.camera_trigger_mode_var),
        ):
            try:
                v = trig.get(key)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    var.set(s)
            except Exception:
                pass

        try:
            app.camera_verbose_var.set(bool(trig.get("verbose") or False))
        except Exception:
            pass

    sub = data.get("camera_subarray")
    if isinstance(sub, dict):
        try:
            app.camera_subarray_enable_var.set(bool(sub.get("enabled") or False))
        except Exception:
            pass
        for key, var in (
            ("x", app.camera_sub_x_var),
            ("y", app.camera_sub_y_var),
            ("width", app.camera_sub_w_var),
            ("height", app.camera_sub_h_var),
        ):
            try:
                v = sub.get(key)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    var.set(s)
            except Exception:
                pass


def save_camera_trigger_prefs(app: Any, *, prefs_path) -> None:
    trig = {
        "source": (app.camera_trigger_source_var.get() or "").strip(),
        "connector": (app.camera_trigger_connector_var.get() or "").strip(),
        "polarity": (app.camera_trigger_polarity_var.get() or "").strip(),
        "active": (app.camera_trigger_active_var.get() or "").strip(),
        "mode": (app.camera_trigger_mode_var.get() or "").strip(),
        "verbose": bool(app.camera_verbose_var.get()),
    }

    sub = {
        "enabled": bool(app.camera_subarray_enable_var.get()),
        "x": (app.camera_sub_x_var.get() or "").strip(),
        "y": (app.camera_sub_y_var.get() or "").strip(),
        "width": (app.camera_sub_w_var.get() or "").strip(),
        "height": (app.camera_sub_h_var.get() or "").strip(),
    }
    write_json(prefs_path, {"camera_trigger": trig, "camera_subarray": sub})
