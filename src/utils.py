import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int) -> None:
    """Set seeds across libraries for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Not all TF builds support deterministic ops.
            pass
    except Exception:
        pass


def get_logger(name: str = "benchmark", log_file: str | Path | None = None) -> logging.Logger:
    """Configure a consistent logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger


@contextmanager
def track_time() -> float:
    """Context manager to measure elapsed time."""
    start = time.time()
    yield lambda: time.time() - start
