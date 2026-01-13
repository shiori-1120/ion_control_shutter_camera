from __future__ import annotations

from typing import Any


def format_worker_failure(resp: Any, *, label: str, log_path: str | None = None) -> str:
    """Format worker failure dicts into a human-friendly message."""

    msg = ""
    if isinstance(resp, dict):
        event = str(resp.get("event") or "").strip()
        err = resp.get("error")
        if err is None:
            err = resp.get("msg") or resp.get("message") or resp
        msg = str(err)
        if event:
            msg = f"{label}: {msg} (event={event})"
        else:
            msg = f"{label}: {msg}"

        if "NOCAMERA" in msg or "No camera detected" in msg:
            msg += (
                "\n\nDCAM がカメラを検出できていません。\n"
                "- カメラの電源/接続(USB/CameraLink等)\n"
                "- Hamamatsu/DCAM-API ドライバの導入\n"
                "- 他アプリがカメラを掴んでいないか\n"
                "を確認してください。カメラ無しPCで試す場合は Camera mode を dry にしてください。"
            )
    else:
        msg = f"{label}: {resp}"

    if log_path:
        try:
            lp = str(log_path).strip()
        except Exception:
            lp = ""
        if lp:
            msg += f"\n\nLog: {lp}"
    return msg
