from __future__ import annotations

import time
from typing import Any, Callable


def acquire_frame_with_ttl(
    *,
    send_get_frame: Callable[[float, str | None], None],
    run_ttl: Callable[[], None],
    wait_resp: Callable[[float, str], dict[str, Any]],
    max_attempt: int,
    frame_timeout_s: float,
    resp_timeout_s: float,
    prefer_sample_path: str | None = None,
    sleep_s: float = 0.05,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Acquire a single frame by issuing get_frame, then TTL, then waiting for response."""
    last_err: Any | None = None
    last_resp: dict[str, Any] | None = None
    for attempt in range(int(max_attempt)):
        try:
            if log_cb is not None:
                log_cb(f"frame attempt {attempt + 1}/{int(max_attempt)}")
            send_get_frame(float(frame_timeout_s), prefer_sample_path)
            run_ttl()
            resp = wait_resp(float(resp_timeout_s), "Camera frame")
        except Exception as e:
            last_err = str(e)
            if log_cb is not None:
                log_cb(f"frame attempt error: {e}")
            time.sleep(float(sleep_s))
            continue
        if isinstance(resp, dict) and resp.get("ok"):
            return {"ok": True, "resp": resp}
        last_resp = resp if isinstance(resp, dict) else None
        last_err = (resp.get("error") if isinstance(resp, dict) else f"invalid response: {resp!r}")
        time.sleep(float(sleep_s))
    return {"ok": False, "resp": last_resp, "error": last_err}
