from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal


@dataclass
class DaqConfig:
    device: str
    mode: Literal["real", "dry"]


@dataclass
class SubarrayConfig:
    enabled: bool
    x: int
    y: int
    width: int
    height: int


@dataclass
class TriggerConfig:
    source: Literal["EXTERNAL"]
    connector: Literal["BNC"]
    polarity: Literal["POSITIVE"]
    active: Literal["EDGE"]
    mode: Literal["NORMAL"]


@dataclass
class CameraConfig:
    mode: Literal["real", "dry"]
    exposure_ms: float
    subarray: SubarrayConfig
    verbose: bool
    trigger: TriggerConfig


@dataclass
class FgConfig:
    visa_resource: str
    amp_mvpp: float
    wave: Literal["SIN"]
    offset_v: float
    start_hz: float
    stop_hz: float
    time_s: float
    no_fg: bool


@dataclass
class SweepDefaults:
    n_target: int
    max_attempt: int
    settle_s: float
    update_interval: float


@dataclass
class IoPaths:
    logs_root: str
    output_root: str


@dataclass
class UiFlags:
    show_debug_fields: bool
    camera_verbose_additional_only: bool


@dataclass
class DeviceRegistry:
    version: str
    daq: DaqConfig
    camera: CameraConfig
    fg: FgConfig
    sweep_defaults: SweepDefaults
    sequence_json_path: str
    io_paths: IoPaths
    ui: UiFlags

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceRegistry":
        return cls(
            version=str(data.get("version") or "1.0"),
            daq=DaqConfig(**data["daq"]),
            camera=CameraConfig(
                mode=data["camera"]["mode"],
                exposure_ms=float(data["camera"]["exposure_ms"]),
                subarray=SubarrayConfig(**data["camera"]["subarray"]),
                verbose=bool(data["camera"]["verbose"]),
                trigger=TriggerConfig(**data["camera"]["trigger"]),
            ),
            fg=FgConfig(**data["fg"]),
            sweep_defaults=SweepDefaults(**data["sweep_defaults"]),
            sequence_json_path=str(data["sequence_json_path"]),
            io_paths=IoPaths(**data["io_paths"]),
            ui=UiFlags(**data["ui"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "daq": self.daq.__dict__,
            "camera": {
                "mode": self.camera.mode,
                "exposure_ms": self.camera.exposure_ms,
                "subarray": self.camera.subarray.__dict__,
                "verbose": self.camera.verbose,
                "trigger": self.camera.trigger.__dict__,
            },
            "fg": self.fg.__dict__,
            "sweep_defaults": self.sweep_defaults.__dict__,
            "sequence_json_path": self.sequence_json_path,
            "io_paths": self.io_paths.__dict__,
            "ui": self.ui.__dict__,
        }


def validate_device_registry(registry: DeviceRegistry) -> None:
    if registry.daq.mode not in ("real", "dry"):
        raise ValueError(f"Invalid DAQ mode: {registry.daq.mode!r}")
    if registry.camera.mode not in ("real", "dry"):
        raise ValueError(f"Invalid camera mode: {registry.camera.mode!r}")
    if registry.camera.exposure_ms <= 0:
        raise ValueError("exposure_ms must be > 0")
    sub = registry.camera.subarray
    if sub.enabled and (sub.width <= 0 or sub.height <= 0):
        raise ValueError("subarray width/height must be > 0 when enabled")
    trig = registry.camera.trigger
    if (trig.source, trig.connector, trig.polarity, trig.active, trig.mode) != (
        "EXTERNAL",
        "BNC",
        "POSITIVE",
        "EDGE",
        "NORMAL",
    ):
        raise ValueError("trigger config must be fixed to EXTERNAL/BNC/POSITIVE/EDGE/NORMAL")
    if registry.fg.wave != "SIN":
        raise ValueError("fg.wave must be 'SIN'")
    if registry.sweep_defaults.n_target <= 0:
        raise ValueError("sweep_defaults.n_target must be > 0")


def load_device_registry(path: Path) -> DeviceRegistry:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    registry = DeviceRegistry.from_dict(data)
    validate_device_registry(registry)
    return registry


def save_device_registry(path: Path, registry: DeviceRegistry) -> None:
    validate_device_registry(registry)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(registry.to_dict(), indent=2), encoding="utf-8")


def resolve_output_root(
    *,
    path: Path = Path("config") / "device_registry.json",
    default: str = "data/output",
) -> Path:
    try:
        registry = load_device_registry(path)
        output_root = registry.io_paths.output_root
        if output_root:
            return Path(output_root)
    except Exception:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                output_root = data.get("io_paths", {}).get("output_root", "")
                if output_root:
                    return Path(output_root)
        except Exception:
            pass
    return Path(default)
