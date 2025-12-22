import os
import datetime
import logging


def make_run_folder(base_dir: str = "data/output/shutter") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logger(run_dir: str, name: str = "shutter") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_event(logger: logging.Logger, event: str, **kv):
    if kv:
        extras = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"{event} {extras}")
    else:
        logger.info(event)
