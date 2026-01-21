from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .worker_utils import to_uint8_image


def load_dry_samples(dry_dir: str | None) -> list[tuple[Any, bool, str]]:
    dry_samples: list[tuple[Any, bool, str]] = []
    if not dry_dir:
        return dry_samples
    try:
        import numpy as np

        try:
            from PIL import Image  # type: ignore
        except Exception:
            Image = None  # type: ignore

        base = Path(dry_dir)
        if not (base.exists() and base.is_dir()):
            return dry_samples

        def load_img(p: Path) -> Any:
            if p.suffix.lower() == ".npy":
                return np.load(p)
            if Image is None:
                raise RuntimeError("Pillow not available to load image")
            return np.asarray(Image.open(p).convert("L"))

        def try_load(stem: str, is_bright: bool) -> None:
            for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "npy"):
                for candidate in base.glob(f"{stem}*.{ext}"):
                    try:
                        arr = load_img(candidate)
                        dry_samples.append((arr, is_bright, candidate.name))
                        return
                    except Exception:
                        continue

        try_load("bright", True)
        try_load("roi_test", True)
        try_load("dark", False)
    except Exception:
        return []
    return dry_samples


def handle_dry_command(
    *,
    name: str,
    cmd: dict[str, Any],
    dry_samples: list[tuple[Any, bool, str]],
    subarray_t: tuple[int, int, int, int] | None,
    roi_t: tuple[int, int, int, int] | None,
    bg_roi_t: tuple[int, int, int, int] | None,
    tau_on: float | None,
    tau_off: float | None,
    prev_state: bool | None,
) -> tuple[dict[str, Any], bool | None]:
    if name == "get_frame":
        # Force a specific sample if requested (useful for ROI check).
        try:
            prefer = cmd.get("prefer_sample")
            if isinstance(prefer, str) and prefer.strip():
                p = Path(prefer)
                if p.exists() and p.is_file():
                    import numpy as _np

                    if p.suffix.lower() == ".npy":
                        arr = _np.load(p)
                    else:
                        try:
                            from PIL import Image  # type: ignore

                            arr = _np.asarray(Image.open(p).convert("L"))
                        except Exception:
                            arr = _np.load(p)
                    frame = to_uint8_image(arr)
                    frame = _np.asarray(frame)
                    if subarray_t is not None:
                        from .lib.image_ops import crop_roi

                        frame = crop_roi(frame, subarray_t)
                    resp = {
                        "ok": True,
                        "event": "frame",
                        "frame": frame,
                        "bright": True,
                        "S_norm": float(_np.mean(frame)) if frame.size else 0.0,
                        "tau_on": None,
                        "tau_off": None,
                        "sample": str(p),
                    }
                    if cmd.get("tag") is not None:
                        resp["tag"] = cmd.get("tag")
                    return resp, prev_state
        except Exception:
            pass

        if dry_samples:
            import numpy as _np

            prefer = cmd.get("prefer_sample")
            pick = None
            if isinstance(prefer, str) and prefer.strip():
                prefer_path = Path(prefer)
                prefer_name = prefer_path.name.lower()
                prefer_stem = prefer_path.stem.lower()
                for a, b, n in dry_samples:
                    if not isinstance(n, str):
                        continue
                    n_l = n.lower()
                    if n_l == prefer_name:
                        pick = (a, b, n)
                        break
                    try:
                        if Path(n_l).stem == prefer_stem:
                            pick = (a, b, n)
                            break
                    except Exception:
                        continue
            if pick is None:
                pick = random.choice(dry_samples)

            arr, bright_label, sample_name = pick
            frame = to_uint8_image(arr)
            frame = _np.asarray(frame)
            if subarray_t is not None:
                from .lib.image_ops import crop_roi

                frame = crop_roi(frame, subarray_t)
            resp = {
                "ok": True,
                "event": "frame",
                "frame": frame,
                "bright": bool(bright_label),
                "S_norm": float(_np.mean(frame)) if frame.size else 0.0,
                "tau_on": None,
                "tau_off": None,
                "sample": sample_name,
            }
            if cmd.get("tag") is not None:
                resp["tag"] = cmd.get("tag")
            return resp, prev_state

        import numpy as _np

        is_bright = (random.random() < 0.5)
        base = 180.0 if is_bright else 40.0
        noise = _np.random.normal(loc=0.0, scale=18.0, size=(256, 256))
        frame_f = base + noise
        frame = _np.asarray(_np.clip(_np.rint(frame_f), 0, 255), dtype=_np.uint8)
        if subarray_t is not None:
            from .lib.image_ops import crop_roi

            frame = crop_roi(frame, subarray_t)
        resp = {
            "ok": True,
            "event": "frame",
            "frame": frame,
            "bright": is_bright,
            "S_norm": float(_np.mean(frame)),
            "tau_on": None,
            "tau_off": None,
        }
        if cmd.get("tag") is not None:
            resp["tag"] = cmd.get("tag")
        return resp, prev_state

    if dry_samples:
        import numpy as np
        from .lib.image_ops import crop_roi

        arr, bright_label, name = random.choice(dry_samples)
        label_bright = bool(bright_label)
        try:
            frame = np.asarray(to_uint8_image(arr))
            if subarray_t is not None:
                frame = crop_roi(frame, subarray_t)
            if roi_t is not None:
                xw, yw, xs, ys = map(int, roi_t)
                crop = crop_roi(frame, (xw, yw, xs, ys))
                s_norm = float(np.mean(crop)) if getattr(crop, "size", 0) else float(np.mean(frame))
            else:
                s_norm = float(np.mean(frame)) if frame.size else float(random.gauss(120.0, 30.0))
        except Exception:
            s_norm = float(random.gauss(120.0, 30.0))

        bright: bool
        try:
            if (tau_on is not None) and (tau_off is not None):
                if abs(float(tau_on) - float(tau_off)) < 1e-12:
                    bright = bool(float(s_norm) > float(tau_on))
                    prev_state = bool(bright)
                else:
                    if prev_state is None:
                        prev_state = bool(float(s_norm) > float(tau_on))
                    if prev_state:
                        prev_state = bool(float(s_norm) > float(tau_off))
                    else:
                        prev_state = bool(float(s_norm) > float(tau_on))
                    bright = bool(prev_state)
            else:
                bright = bool(label_bright)
        except Exception:
            bright = bool(label_bright)
        resp = {
            "ok": True,
            "event": "state",
            "bright": bool(bright),
            "label_bright": bool(label_bright),
            "S_norm": s_norm,
            "tau_on": float(tau_on) if tau_on is not None else None,
            "tau_off": float(tau_off) if tau_off is not None else None,
            "sample": name,
        }
        if cmd.get("tag") is not None:
            resp["tag"] = cmd.get("tag")
        return resp, prev_state

    s_norm = float(random.gauss(150.0, 25.0))
    if random.random() < 0.5:
        s_norm = float(random.gauss(50.0, 15.0))
    s_norm = float(max(0.0, min(255.0, s_norm)))
    bright = bool(s_norm > 100.0)
    resp = {
        "ok": True,
        "event": "state",
        "bright": bright,
        "label_bright": None,
        "S_norm": s_norm,
        "tau_on": None,
        "tau_off": None,
        "sample": None,
    }
    if cmd.get("tag") is not None:
        resp["tag"] = cmd.get("tag")
    return resp, prev_state
