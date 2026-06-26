"""Tests for the centralised reproducibility helper."""

import numpy as np

from seed import SEED, set_global_seeds


def test_set_global_seeds_makes_numpy_deterministic():
    set_global_seeds(SEED)
    first = np.random.rand(5)
    set_global_seeds(SEED)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_set_global_seeds_accepts_custom_seed():
    set_global_seeds(0)
    a = np.random.rand(3)
    set_global_seeds(0)
    b = np.random.rand(3)
    assert np.array_equal(a, b)
