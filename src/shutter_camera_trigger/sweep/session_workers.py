from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any

from ..workers.camera_worker_process import start_camera_worker_process
from ..workers.daq_worker_process import start_daq_worker_process


@dataclass(frozen=True)
class SweepWorkers:
    daq_proc: Process
    daq_cmd_q: Any
    daq_resp_q: Any
    cam_proc: Process
    cam_cmd_q: Any
    cam_resp_q: Any


def create_sweep_workers(
    *,
    device: str,
    daq_mode: str,
    cam_cfg: dict,
    daq_log_path: str | None = None,
    run_id: str | None = None,
) -> SweepWorkers:
    """Create DAQ+Camera workers and queues for a sweep session.

    This mirrors the GUI's startup ordering expectations:
    - DAQ process is created but NOT started here.
    - Camera process is created but NOT started here.

    The caller can start DAQ, wait for ready with UI pumping, then start camera.
    """

    daq_proc, daq_cmd_q, daq_resp_q = start_daq_worker_process(
        device=device,
        mode=daq_mode,
        start=False,
        wait_ready=False,
        log_path=daq_log_path,
        run_id=run_id,
    )

    cam_cmd_q: Queue = Queue()
    cam_resp_q: Queue = Queue()
    cam_proc, _, _ = start_camera_worker_process(cfg=cam_cfg, start=False, cmd_q=cam_cmd_q, resp_q=cam_resp_q)

    return SweepWorkers(
        daq_proc=daq_proc,
        daq_cmd_q=daq_cmd_q,
        daq_resp_q=daq_resp_q,
        cam_proc=cam_proc,
        cam_cmd_q=cam_cmd_q,
        cam_resp_q=cam_resp_q,
    )
