"""
Centralised reproducibility helper.

Every entry point (``main.py``, the smoke run, the test suite) calls
:func:`set_global_seeds` exactly once so that NumPy, the Python ``random``
module, and PyTorch all draw from the same fixed seed.  Keeping the seed
logic in one place avoids the common bug where only ``torch`` is seeded while
NumPy-based shuffling or sampling stays non-deterministic.
"""

import os
import random

import numpy as np

SEED: int = 42


def set_global_seeds(seed: int = SEED) -> None:
    """
    Seed every random number generator used in the project.

    Args:
        seed: The integer seed shared by all RNGs. Defaults to :data:`SEED`.

    Notes:
        PyTorch is imported lazily so that lightweight consumers (the metric
        and feature unit tests) can call this function without having the
        heavy deep-learning stack installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
