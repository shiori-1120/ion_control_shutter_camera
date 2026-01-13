from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def read_last_worker_pids(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_last_worker_pids(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _get_cmdline_for_pid(pid: int) -> str:
    if pid <= 0 or getattr(os, "name", "") != "nt":
        return ""
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if($p){{$p.CommandLine}}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _taskkill_pid(pid: int, *, force: bool = False) -> bool:
    if pid <= 0 or getattr(os, "name", "") != "nt":
        return False
    try:
        args = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            args.append("/F")
        r = subprocess.run(args, capture_output=True, text=True, timeout=4)
        return r.returncode == 0
    except Exception:
        return False


def cleanup_stale_workers(path: Path) -> None:
    """Try to release camera locks left by crashed runs (best-effort)."""
    try:
        data = read_last_worker_pids(path)
        pids: list[int] = []
        for k in ("cam_pid", "daq_pid"):
            try:
                v = int(data.get(k, 0) or 0)
            except Exception:
                v = 0
            if v > 0:
                pids.append(v)

        if not pids:
            return

        for pid in pids:
            cmdline = (_get_cmdline_for_pid(pid) or "").lower()
            looks_like_ours = any(m in cmdline for m in ("ion_state_worker", "daq_worker", "shutter_gui"))
            if not looks_like_ours:
                continue

            if not _taskkill_pid(pid, force=False):
                _taskkill_pid(pid, force=True)

        write_last_worker_pids(path, {})
    except Exception:
        pass