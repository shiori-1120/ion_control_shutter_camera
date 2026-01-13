from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.device_registry import (
    CameraConfig,
    DaqConfig,
    DeviceRegistry,
    FgConfig,
    IoPaths,
    SubarrayConfig,
    SweepDefaults,
    TriggerConfig,
    UiFlags,
    load_device_registry,
    save_device_registry,
)


def _set_var(app: Any, name: str, value: object) -> None:
    var = getattr(app, name, None)
    if var is None:
        return
    try:
        var.set(value)
    except Exception:
        pass


def _get_var(app: Any, name: str, default: str) -> str:
    var = getattr(app, name, None)
    if var is None:
        return default
    try:
        value = var.get()
        return str(value)
    except Exception:
        return default


def _get_bool(app: Any, name: str, default: bool) -> bool:
    var = getattr(app, name, None)
    if var is None:
        return default
    try:
        return bool(var.get())
    except Exception:
        return default


def _get_float(app: Any, name: str, default: float) -> float:
    raw = _get_var(app, name, str(default)).strip()
    try:
        return float(raw)
    except Exception:
        return float(default)


def _get_int(app: Any, name: str, default: int) -> int:
    raw = _get_var(app, name, str(default)).strip()
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def apply_device_registry_to_ui(app: Any, registry: DeviceRegistry) -> None:
    _set_var(app, "device_var", registry.daq.device)
    _set_var(app, "device_mode_var", registry.daq.mode)
    _set_var(app, "fg_resource_var", registry.fg.visa_resource)
    _set_var(app, "fg_amp_mvpp_var", str(registry.fg.amp_mvpp))
    _set_var(app, "sw_no_fg", registry.fg.no_fg)
    _set_var(app, "camera_mode_top_var", registry.camera.mode)
    _set_var(app, "camera_exposure_ms_var", str(registry.camera.exposure_ms))
    _set_var(app, "camera_verbose_var", registry.camera.verbose)
    _set_var(app, "camera_subarray_enable_var", registry.camera.subarray.enabled)
    _set_var(app, "camera_sub_x_var", str(registry.camera.subarray.x))
    _set_var(app, "camera_sub_y_var", str(registry.camera.subarray.y))
    _set_var(app, "camera_sub_w_var", str(registry.camera.subarray.width))
    _set_var(app, "camera_sub_h_var", str(registry.camera.subarray.height))
    _set_var(app, "camera_trigger_source_var", registry.camera.trigger.source)
    _set_var(app, "camera_trigger_connector_var", registry.camera.trigger.connector)
    _set_var(app, "camera_trigger_polarity_var", registry.camera.trigger.polarity)
    _set_var(app, "camera_trigger_active_var", registry.camera.trigger.active)
    _set_var(app, "camera_trigger_mode_var", registry.camera.trigger.mode)
    _set_var(app, "sw_seq_path", registry.sequence_json_path)
    _set_var(app, "sw_n_target", str(registry.sweep_defaults.n_target))
    _set_var(app, "sw_max_attempt", str(registry.sweep_defaults.max_attempt))
    _set_var(app, "sw_settle_s", str(registry.sweep_defaults.settle_s))
    _set_var(app, "sw_update_interval", str(registry.sweep_defaults.update_interval))
    try:
        app.camera_verbose_additional_only = bool(registry.ui.camera_verbose_additional_only)
    except Exception:
        pass
    try:
        app.show_debug_fields = bool(registry.ui.show_debug_fields)
    except Exception:
        pass
    try:
        app.output_root = Path(registry.io_paths.output_root)
    except Exception:
        pass


def build_device_registry_from_ui(app: Any) -> DeviceRegistry:
    existing = None
    try:
        path = getattr(app, "_device_registry_path", None)
        if path:
            existing = load_device_registry(Path(path))
    except Exception:
        existing = None
    trigger = TriggerConfig(
        source="EXTERNAL",
        connector="BNC",
        polarity="POSITIVE",
        active="EDGE",
        mode="NORMAL",
    )
    subarray = SubarrayConfig(
        enabled=_get_bool(app, "camera_subarray_enable_var", False),
        x=_get_int(app, "camera_sub_x_var", 0),
        y=_get_int(app, "camera_sub_y_var", 0),
        width=_get_int(app, "camera_sub_w_var", 0),
        height=_get_int(app, "camera_sub_h_var", 0),
    )
    camera = CameraConfig(
        mode=_get_var(app, "camera_mode_top_var", "dry"),
        exposure_ms=_get_float(app, "camera_exposure_ms_var", 100.0),
        subarray=subarray,
        verbose=_get_bool(app, "camera_verbose_var", False),
        trigger=trigger,
    )
    fg = FgConfig(
        visa_resource=_get_var(app, "fg_resource_var", ""),
        amp_mvpp=_get_float(app, "fg_amp_mvpp_var", 790.0),
        wave="SIN",
        offset_v=0.0,
        start_hz=1000.0,
        stop_hz=10000.0,
        time_s=1.0,
        no_fg=_get_bool(app, "sw_no_fg", True),
    )
    sweep_defaults = SweepDefaults(
        n_target=_get_int(app, "sw_n_target", 50),
        max_attempt=_get_int(app, "sw_max_attempt", 100),
        settle_s=_get_float(app, "sw_settle_s", 0.02),
        update_interval=_get_float(app, "sw_update_interval", 1.0),
    )
    defaults = existing or DeviceRegistry(
        version="1.0",
        daq=DaqConfig(device="Dev1", mode="dry"),
        camera=camera,
        fg=fg,
        sweep_defaults=sweep_defaults,
        sequence_json_path=_get_var(app, "sw_seq_path", ""),
        io_paths=IoPaths(logs_root="logs", output_root="data/output"),
        ui=UiFlags(show_debug_fields=True, camera_verbose_additional_only=True),
    )
    registry = DeviceRegistry(
        version=defaults.version,
        daq=DaqConfig(
            device=_get_var(app, "device_var", defaults.daq.device),
            mode=_get_var(app, "device_mode_var", defaults.daq.mode),
        ),
        camera=camera,
        fg=fg,
        sweep_defaults=sweep_defaults,
        sequence_json_path=_get_var(app, "sw_seq_path", defaults.sequence_json_path),
        io_paths=defaults.io_paths,
        ui=defaults.ui,
    )
    return registry


def load_device_registry_ui(app: Any, path: Path) -> None:
    try:
        registry = load_device_registry(path)
    except Exception:
        return
    apply_device_registry_to_ui(app, registry)


def save_device_registry_ui(app: Any, path: Path) -> None:
    registry = build_device_registry_from_ui(app)
    save_device_registry(path, registry)
