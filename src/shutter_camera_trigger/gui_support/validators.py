from __future__ import annotations

from typing import Any


def parse_exposure_s(app: Any) -> float:
    """Return camera exposure in seconds parsed from UI (ms input)."""
    try:
        s = (app.camera_exposure_ms_var.get() or "").strip()
        if not s:
            raise ValueError("Exposure (ms) is empty")
        v = float(s)
    except Exception as e:
        raise ValueError(f"Invalid exposure (ms): {s!r}") from e
    if v <= 0:
        raise ValueError("Exposure (ms) must be > 0")
    return float(v) / 1000.0


def parse_fg_amp_vpp(app: Any, *, max_mvpp: float) -> float:
    try:
        s = (app.fg_amp_mvpp_var.get() or "").strip()
        if not s:
            raise ValueError("FG amplitude is empty")
        mvpp = float(s)
    except Exception as e:
        raise ValueError(f"Invalid FG amp (mVpp): {s!r}") from e
    if mvpp <= 0:
        raise ValueError("FG amp (mVpp) must be > 0")
    if mvpp > float(max_mvpp):
        raise ValueError(f"FG amp (mVpp) must be <= {max_mvpp}")
    return float(mvpp) / 1000.0


def parse_camera_subarray(app: Any) -> tuple[int, int, int, int] | None:
    """Return subarray as ROI tuple (xw,yw,xs,ys) or None if disabled."""
    try:
        enabled = bool(getattr(app, "camera_subarray_enable_var").get())
    except Exception:
        enabled = False
    if not enabled:
        return None

    def _get_int(var_name: str, label: str) -> int:
        s = (getattr(app, var_name).get() or "").strip()
        if not s:
            raise ValueError(f"{label} is empty")
        try:
            v = int(float(s))
        except Exception as e:
            raise ValueError(f"Invalid {label}: {s!r}") from e
        return int(v)

    x = _get_int("camera_sub_x_var", "Subarray X")
    y = _get_int("camera_sub_y_var", "Subarray Y")
    w = _get_int("camera_sub_w_var", "Subarray Width")
    h = _get_int("camera_sub_h_var", "Subarray Height")
    if w <= 0 or h <= 0:
        raise ValueError("Subarray width/height must be > 0")
    if x < 0 or y < 0:
        raise ValueError("Subarray X/Y must be >= 0")
    return (int(w), int(h), int(x), int(y))


def apply_subarray_to_cam_cfg(app: Any, cfg: dict[str, Any]) -> None:
    sub = parse_camera_subarray(app)
    if sub is None:
        return
    cfg["subarray"] = [int(sub[0]), int(sub[1]), int(sub[2]), int(sub[3])]


def parse_camera_trigger_cfg(app: Any) -> dict[str, Any]:
    delay_s_raw = (app.camera_trigger_delay_s_var.get() or "").strip()
    delay_s: float | None = None
    if delay_s_raw:
        try:
            delay_s = float(delay_s_raw)
        except Exception:
            raise ValueError(f"Invalid trigger delay (s): {delay_s_raw!r}")

    cfg: dict[str, Any] = {
        "source": (app.camera_trigger_source_var.get() or "EXTERNAL").strip().upper() or "EXTERNAL",
        "connector": (app.camera_trigger_connector_var.get() or "BNC").strip().upper() or "BNC",
        "polarity": (app.camera_trigger_polarity_var.get() or "POSITIVE").strip().upper() or "POSITIVE",
        "active": (app.camera_trigger_active_var.get() or "EDGE").strip().upper() or "EDGE",
        "mode": (app.camera_trigger_mode_var.get() or "NORMAL").strip().upper() or "NORMAL",
    }
    if delay_s is not None:
        cfg["delay_s"] = float(delay_s)
    return cfg


def parse_exposure_s_safe(app: Any, *, default_s: float = 0.1) -> float:
    try:
        return parse_exposure_s(app)
    except Exception:
        return float(default_s)


def parse_fg_amp_vpp_safe(app: Any, *, max_mvpp: float, default_vpp: float) -> float:
    try:
        return parse_fg_amp_vpp(app, max_mvpp=float(max_mvpp))
    except Exception:
        return float(default_vpp)
