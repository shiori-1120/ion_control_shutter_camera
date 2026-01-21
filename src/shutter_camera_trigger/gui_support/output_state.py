from __future__ import annotations

from typing import Any


def set_output_state(app: Any, value: int | None) -> None:
    try:
        app._last_do_value = None if value is None else int(value)
    except Exception:
        pass

    lamps = getattr(app, "_output_lamps", None)
    if not lamps:
        return

    on_color = getattr(app, "_lamp_on_color", "#4caf50")
    off_color = getattr(app, "_lamp_off_color", "#444444")
    for mask, lamp in lamps:
        is_on = bool(value is not None and (int(value) & int(mask)))
        try:
            lamp.configure(bg=on_color if is_on else off_color)
        except Exception:
            pass
