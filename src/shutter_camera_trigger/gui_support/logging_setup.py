from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from logging.handlers import QueueHandler
from pathlib import Path
import queue
from typing import Optional


@dataclass(frozen=True)
class LogContext:
    """Holds logging config for the GUI process."""

    run_id: str
    log_dir: Path
    logger: logging.Logger
    gui_queue: Optional[queue.Queue]


class _RunIdFilter(logging.Filter):
    """Inject a run_id attribute into each LogRecord."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self._run_id
        return True


def _build_log_dir(*, logs_root: str, run_id: str) -> Path:
    """Create and return a daily log directory."""

    day = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(logs_root) / day
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _create_run_id() -> str:
    """Create a run_id string for log correlation."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def init_app_logging(*, logs_root: str = "logs", run_id: str | None = None) -> LogContext:
    """Initialize app logging with a file handler and optional GUI queue."""

    rid = run_id or _create_run_id()
    log_dir = _build_log_dir(logs_root=logs_root, run_id=rid)
    log_path = log_dir / "app.log"

    logger = logging.getLogger("shutter.app")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(run_id)s %(levelname)s %(name)s %(message)s")
    run_id_filter = _RunIdFilter(rid)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(run_id_filter)
    logger.handlers.clear()
    logger.addHandler(file_handler)

    gui_queue: queue.Queue | None = queue.Queue()
    queue_handler = QueueHandler(gui_queue)
    queue_handler.addFilter(run_id_filter)
    queue_handler.setLevel(logging.DEBUG)
    logger.addHandler(queue_handler)

    return LogContext(run_id=rid, log_dir=log_dir, logger=logger, gui_queue=gui_queue)


def get_file_logger(*, name: str, log_dir: Path, run_id: str, filename: str) -> logging.Logger:
    """Create or reset a logger that writes to a dedicated file."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(run_id)s %(levelname)s %(name)s %(message)s")
    run_id_filter = _RunIdFilter(run_id)

    path = log_dir / filename
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(fmt)
    handler.addFilter(run_id_filter)

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
