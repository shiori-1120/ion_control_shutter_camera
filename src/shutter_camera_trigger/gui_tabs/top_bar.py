from __future__ import annotations

import tkinter as tk
from typing import Any, Callable
from tkinter import messagebox, ttk

from ..gui_support.validators import parse_camera_subarray
from ..hardware import CameraQueueDevice

from ..gui_support.output_state import set_output_state

def build_top_bar(
    app: Any,
    *,
    parent: tk.Misc | None = None,
    connect_cb: Callable[[], None],
    disconnect_cb: Callable[[], None],
    fg_connect_cb: Callable[[], None],
    fg_disconnect_cb: Callable[[], None],
    pick_seq_json_cb: Callable[[], None],
    output_defs: list[tuple[str, int]] | None = None,
) -> None:
    top = ttk.Frame(parent or app, padding=10)
    top.pack(side=tk.TOP, fill=tk.X)

    daq = ttk.LabelFrame(top, text="DAQ")
    daq.grid(row=0, column=0, sticky=tk.W + tk.E, pady=(0, 6))

    ttk.Label(daq, text="Device").grid(row=0, column=0, sticky=tk.W)
    ttk.Entry(daq, textvariable=app.device_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

    ttk.Label(daq, text="Mode").grid(row=0, column=2, sticky=tk.W)
    ttk.Combobox(daq, textvariable=app.device_mode_var, values=["real", "dry"], width=6, state="readonly").grid(
        row=0, column=3, sticky=tk.W, padx=5
    )

    ttk.Label(daq, text="AO width").grid(row=0, column=4, sticky=tk.W)
    ttk.Entry(daq, textvariable=app.width_var, width=10).grid(
        row=0, column=5, sticky=tk.W, padx=5
    )
    ttk.Label(daq, text="ms").grid(row=0, column=6, sticky=tk.W)

    app.connect_btn = ttk.Button(daq, text="Connect", command=connect_cb)
    app.connect_btn.grid(row=0, column=7, padx=5)
    app.disconnect_btn = ttk.Button(daq, text="Disconnect", command=disconnect_cb, state=tk.DISABLED)
    app.disconnect_btn.grid(row=0, column=8)

    fg = ttk.LabelFrame(top, text="FG")
    fg.grid(row=1, column=0, sticky=tk.W + tk.E, pady=(0, 6))

    ttk.Label(fg, text="VISA").grid(row=0, column=0, sticky=tk.W)
    ttk.Entry(fg, textvariable=app.fg_resource_var, width=32).grid(row=0, column=1, sticky=tk.W, padx=5)
    app.fg_connect_btn = ttk.Button(fg, text="Connect", command=fg_connect_cb)
    app.fg_connect_btn.grid(row=0, column=2, padx=5)
    app.fg_disconnect_btn = ttk.Button(fg, text="Disconnect", command=fg_disconnect_cb, state=tk.DISABLED)
    app.fg_disconnect_btn.grid(row=0, column=3, padx=5)
    ttk.Label(fg, text="Amp").grid(row=0, column=4, sticky=tk.W)
    ttk.Entry(fg, textvariable=app.fg_amp_mvpp_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)
    ttk.Label(fg, text="mVpp").grid(row=0, column=6, sticky=tk.W)
    ttk.Checkbutton(fg, text="No FG", variable=app.sw_no_fg).grid(row=0, column=7, sticky=tk.W, padx=(8, 0))

    seq = ttk.LabelFrame(top, text="Sequence")
    seq.grid(row=2, column=0, sticky=tk.W + tk.E, pady=(0, 6))
    ttk.Label(seq, text="JSON path").grid(row=0, column=0, sticky=tk.W)
    ttk.Entry(seq, textvariable=app.sw_seq_path, width=48).grid(row=0, column=1, sticky=tk.W, padx=5)
    ttk.Button(seq, text="...", width=3, command=pick_seq_json_cb).grid(row=0, column=2)

    cam = ttk.LabelFrame(top, text="Camera")
    cam.grid(row=3, column=0, sticky=tk.W + tk.E, pady=(0, 6))
    ttk.Label(cam, text="Mode").grid(row=0, column=0, sticky=tk.W)
    ttk.Combobox(cam, textvariable=app.camera_mode_top_var, values=["dry", "real"], width=6, state="readonly").grid(
        row=0, column=1, sticky=tk.W, padx=5
    )
    ttk.Label(cam, text="Exposure").grid(row=0, column=2, sticky=tk.W)
    ttk.Entry(cam, textvariable=app.camera_exposure_ms_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
    ttk.Label(cam, text="ms").grid(row=0, column=4, sticky=tk.W)

    trig = ttk.LabelFrame(top, text="Trigger")
    trig.grid(row=4, column=0, sticky=tk.W + tk.E, pady=(0, 6))
    ttk.Label(trig, text="Source").grid(row=0, column=0, sticky=tk.W)
    ttk.Combobox(
        trig,
        textvariable=app.camera_trigger_source_var,
        values=["EXTERNAL", "INTERNAL"],
        width=9,
        state="disabled",
    ).grid(row=0, column=1, sticky=tk.W, padx=5)
    ttk.Label(trig, text="Conn").grid(row=0, column=2, sticky=tk.W)
    ttk.Combobox(
        trig,
        textvariable=app.camera_trigger_connector_var,
        values=["BNC", "MULTI", "INTERFACE"],
        width=9,
        state="disabled",
    ).grid(row=0, column=3, sticky=tk.W, padx=5)
    ttk.Label(trig, text="Pol").grid(row=0, column=4, sticky=tk.W)
    ttk.Combobox(
        trig,
        textvariable=app.camera_trigger_polarity_var,
        values=["POSITIVE", "NEGATIVE"],
        width=9,
        state="disabled",
    ).grid(row=0, column=5, sticky=tk.W, padx=5)
    ttk.Label(trig, text="Act").grid(row=0, column=6, sticky=tk.W)
    ttk.Combobox(
        trig,
        textvariable=app.camera_trigger_active_var,
        values=["EDGE", "LEVEL"],
        width=7,
        state="disabled",
    ).grid(row=0, column=7, sticky=tk.W, padx=5)
    ttk.Label(trig, text="Mode").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(
        trig,
        textvariable=app.camera_trigger_mode_var,
        values=["NORMAL", "START"],
        width=9,
        state="disabled",
    ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=(6, 0))
    verbose_label = "Verbose (extra)" if getattr(app, "camera_verbose_additional_only", False) else "Verbose"
    ttk.Checkbutton(trig, text=verbose_label, variable=app.camera_verbose_var).grid(
        row=1, column=2, sticky=tk.W, padx=5, pady=(6, 0)
    )

    sub = ttk.LabelFrame(top, text="Subarray")
    sub.grid(row=5, column=0, sticky=tk.W + tk.E, pady=(0, 6))

    ttk.Checkbutton(sub, text="Enable", variable=app.camera_subarray_enable_var).grid(
        row=0, column=0, sticky=tk.W, padx=6, pady=4
    )
    ttk.Label(sub, text="X").grid(row=0, column=1, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_x_var, width=8).grid(row=0, column=2, sticky=tk.W, padx=(2, 10))
    ttk.Label(sub, text="Y").grid(row=0, column=3, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_y_var, width=8).grid(row=0, column=4, sticky=tk.W, padx=(2, 10))
    ttk.Label(sub, text="W").grid(row=0, column=5, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_w_var, width=8).grid(row=0, column=6, sticky=tk.W, padx=(2, 10))
    ttk.Label(sub, text="H").grid(row=0, column=7, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_h_var, width=8).grid(row=0, column=8, sticky=tk.W, padx=(2, 10))

    def _apply_subarray() -> None:
        try:
            sub_t = parse_camera_subarray(app)
        except Exception as e:
            messagebox.showerror("Subarray", f"Invalid subarray: {e}")
            return
        cam_cmd_q = None
        cam_resp_q = None
        proc = getattr(app, "_cam_worker_proc", None)
        if proc is not None and getattr(proc, "is_alive", lambda: False)():
            cam_cmd_q = getattr(app, "_cam_worker_cmd_q", None)
            cam_resp_q = getattr(app, "_cam_worker_resp_q", None)
        if not (cam_cmd_q and cam_resp_q):
            state = getattr(app, "_sweep_state", None)
            cam_cmd_q = state.queues.get("cam_cmd") if getattr(state, "queues", None) else None
            cam_resp_q = state.queues.get("cam_resp") if getattr(state, "queues", None) else None
        if not (cam_cmd_q and cam_resp_q):
            messagebox.showwarning("Subarray", "Camera worker is not running. Start camera check or sweep first.")
            return
        try:
            CameraQueueDevice(cmd_q=cam_cmd_q).set_subarray(list(sub_t) if sub_t else None)
            resp = cam_resp_q.get(timeout=5)
            if not isinstance(resp, dict) or not resp.get("ok"):
                raise RuntimeError(resp.get("error", "Camera subarray update failed"))
        except Exception as e:
            messagebox.showerror("Subarray", f"Failed to apply subarray: {e}")
            return
        app.status_var.set("Subarray applied")

    ttk.Button(sub, text="Apply", command=_apply_subarray).grid(row=0, column=9, sticky=tk.W, padx=6)

    try:
        sub.grid_columnconfigure(10, weight=1)
    except Exception:
        pass

    app.status_var = tk.StringVar(value="Disconnected")
    outputs = ttk.LabelFrame(top, text="Outputs")
    outputs.grid(row=6, column=0, sticky=tk.W + tk.E, pady=(0, 6))
    app._output_lamps = []
    app._lamp_on_color = "#4caf50"
    app._lamp_off_color = "#444444"
    if output_defs:
        for idx, (label, mask) in enumerate(output_defs):
            lamp = tk.Label(outputs, text=" ", width=2, relief=tk.SUNKEN, bg=app._lamp_off_color)
            lamp.grid(row=0, column=idx * 2, sticky=tk.W, padx=(6 if idx == 0 else 2, 2), pady=4)
            ttk.Label(outputs, text=label).grid(row=0, column=idx * 2 + 1, sticky=tk.W, padx=(0, 8))
            app._output_lamps.append((int(mask), lamp))
    set_output_state(app, None)

    ttk.Label(top, textvariable=app.status_var).grid(row=7, column=0, sticky=tk.W, pady=(4, 0))

    top.grid_columnconfigure(0, weight=1)
