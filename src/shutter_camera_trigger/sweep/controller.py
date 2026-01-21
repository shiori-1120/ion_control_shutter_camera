from __future__ import annotations

from typing import Any

from .model import SweepDeps, SweepEvents, SweepInput, SweepIO, SweepPhase, SweepState

_MSG_BUSY = "Sweep is busy. Stop the current run first."
_MSG_NEED_ROI = "Run '1) ROI check' first."
_MSG_NEED_THRESH = "Run '1) ROI check' and '2) Threshold' first."
from .workflow import prepare_session, roi_check, start_sweep, stop_sweep, threshold_check, override_threshold


class SweepController:
    def __init__(self, *, events: SweepEvents, io: SweepIO, deps: SweepDeps) -> None:
        self._events = events
        self._io = io
        self._deps = deps

    def prepare_session(self, state: SweepState, inputs: SweepInput) -> bool:
        if state.phase in {SweepPhase.RUNNING, SweepPhase.STOPPING, SweepPhase.ERROR}:
            return self._reject_phase(
                _MSG_BUSY,
            )
        return prepare_session(
            state=state,
            inputs=inputs,
            events=self._events,
            io=self._io,
            deps=self._deps,
        )

    def roi_check(self, state: SweepState, *, fig: Any, canvas: Any) -> None:
        if not self._require_phase(
            state,
            allowed={SweepPhase.PREPARED, SweepPhase.ROI_DONE, SweepPhase.THRESHOLD_DONE},
            message=_MSG_NEED_ROI,
        ):
            return
        roi_check(
            state=state,
            fig=fig,
            canvas=canvas,
            events=self._events,
            io=self._io,
            deps=self._deps,
        )

    def threshold_check(self, state: SweepState, *, fig: Any, canvas: Any) -> None:
        if not self._require_phase(
            state,
            allowed={SweepPhase.PREPARED, SweepPhase.ROI_DONE, SweepPhase.THRESHOLD_DONE},
            message=_MSG_NEED_ROI,
        ):
            return
        threshold_check(
            state=state,
            fig=fig,
            canvas=canvas,
            events=self._events,
            io=self._io,
            deps=self._deps,
        )

    def override_threshold(self, state: SweepState, *, fig: Any, canvas: Any, tau: float, apply: bool) -> None:
        if not self._require_phase(
            state,
            allowed={SweepPhase.PREPARED, SweepPhase.ROI_DONE, SweepPhase.THRESHOLD_DONE},
            message=_MSG_NEED_ROI,
        ):
            return
        override_threshold(
            state=state,
            fig=fig,
            canvas=canvas,
            tau=float(tau),
            apply=bool(apply),
            events=self._events,
            io=self._io,
            deps=self._deps,
        )

    def start_sweep(
        self,
        state: SweepState,
        *,
        fig: Any,
        canvas: Any,
        fg_connected: bool,
        fg_handle: Any | None,
        fallback_fg_amp_vpp: float,
    ) -> None:
        if state.phase is SweepPhase.RUNNING:
            return
        if not self._require_phase(
            state,
            allowed={SweepPhase.THRESHOLD_DONE},
            message=_MSG_NEED_THRESH,
        ):
            return
        start_sweep(
            state=state,
            fig=fig,
            canvas=canvas,
            fg_connected=fg_connected,
            fg_handle=fg_handle,
            fallback_fg_amp_vpp=fallback_fg_amp_vpp,
            events=self._events,
            io=self._io,
            deps=self._deps,
        )

    def stop_sweep(self, state: SweepState, *, clean_only: bool = False, fig: Any | None = None) -> None:
        if state.phase is SweepPhase.IDLE:
            return
        stop_sweep(
            state=state,
            events=self._events,
            io=self._io,
            deps=self._deps,
            clean_only=clean_only,
            fig=fig,
        )

    def _reject_phase(self, message: str) -> bool:
        self._events.on_error("Sweep", message)
        self._io.set_last_error_cb("Sweep", message, None)
        self._io.refresh_buttons()
        return False

    def _require_phase(self, state: SweepState, *, allowed: set[SweepPhase], message: str) -> bool:
        if state.phase in allowed:
            return True
        return self._reject_phase(message)
