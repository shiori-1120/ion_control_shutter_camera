from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable


def setup_worker_logging(cfg: dict[str, Any]) -> tuple[
    Callable[[str], None],
    Callable[[str], None],
    Callable[[], None],
    Callable[[], None],
    bool,
]:
    import logging
    import sys

    log_path = cfg.get("log_path")
    run_id = str(cfg.get("run_id") or "")
    cam_verbose = bool(cfg.get("verbose") or cfg.get("camera_verbose") or False)

    _log_file: Any | None = None
    log_handlers = []
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_fmt)
    log_handlers.append(stream_handler)
    log_file_path = cfg.get("log_path", None)
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(log_fmt)
        log_handlers.append(file_handler)
    logging.basicConfig(
        level=logging.DEBUG if cam_verbose else logging.INFO,
        handlers=log_handlers,
    )

    def log(msg: str) -> None:
        nonlocal _log_file
        if not log_path:
            return
        try:
            if _log_file is None:
                p = Path(str(log_path))
                p.parent.mkdir(parents=True, exist_ok=True)
                _log_file = open(p, "a", encoding="utf-8")
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[{ts}]"
            if run_id:
                prefix = f"{prefix} {run_id}"
            _log_file.write(f"{prefix} {msg}\n")
            _log_file.flush()
        except Exception:
            pass

    def log_debug(msg: str) -> None:
        if cam_verbose:
            log(msg)

    def log_worker_env() -> None:
        try:
            log("===== Camera Worker Environment Diagnostics =====")
            log(f"os.getcwd(): {os.getcwd()}")
            log(f"sys.executable: {getattr(__import__('sys'), 'executable', '')}")
            log(f"sys.argv: {getattr(__import__('sys'), 'argv', '')}")
            log(f"sys.path: {getattr(__import__('sys'), 'path', '')}")
            log(f"os.environ['PATH']: {os.environ.get('PATH', '')}")
            log(f"os.environ['PYTHONPATH']: {os.environ.get('PYTHONPATH', '')}")
            dcam_dll = "dcamapi4.dll"
            found = False
            for p in os.environ.get("PATH", "").split(os.pathsep):
                dll_path = os.path.join(p, dcam_dll)
                if os.path.exists(dll_path):
                    log(f"Found {dcam_dll} at: {dll_path}")
                    found = True
            if not found:
                log(f"{dcam_dll} NOT found in PATH.")
            log("===============================================")
        except Exception as e:
            log(f"log_worker_env error: {e}")

    def close_log() -> None:
        nonlocal _log_file
        try:
            if _log_file is not None:
                _log_file.close()
        except Exception:
            pass
        _log_file = None

    return log, log_debug, log_worker_env, close_log, cam_verbose
