import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from microjax.utils import (
    first_nonzero,
    last_nonzero,
    first_zero,
    min_zero_avoiding,
    max_zero_avoiding,
    mean_zero_avoiding,
    sparse_argsort,
    match_points,
    trapz_zero_avoiding,
)


def test_first_last_and_zero_helpers():
    x = jnp.array([0.0, 0.0, 3.0, 0.0, 5.0])
    assert int(first_nonzero(x)) == 2
    assert int(last_nonzero(x)) == 4
    assert int(first_zero(x)) == 0


def test_min_max_zero_avoiding():
    x = jnp.array([0.0, -2.0, 0.0, 1.0, 3.0])
    assert float(min_zero_avoiding(x)) == -2.0
    assert float(max_zero_avoiding(x)) == 3.0

    # all zeros -> mean_zero_avoiding returns 0.0
    x0 = jnp.zeros(5)
    assert float(mean_zero_avoiding(x0)) == 0.0

    # mixture with zeros -> ignore zeros in mean
    x1 = jnp.array([0.0, 2.0, 0.0, 4.0])
    assert float(mean_zero_avoiding(x1)) == 3.0


def test_sparse_argsort():
    a = jnp.array([0.0, 2.0, 0.0, -1.0, 3.0])
    # nan-safe argsort should order by non-zero values only
    idx = list(map(int, sparse_argsort(a)))
    # Extract non-zero portion in sorted order
    non_zero_sorted = [a[i] for i in idx if float(a[i]) != 0.0]
    assert non_zero_sorted == [-1.0, 2.0, 3.0]


def test_match_points_simple():
    # a and b are close but shuffled
    a = jnp.array([0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j])
    b = jnp.array([1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j])
    perm = list(map(int, match_points(a, b)))
    # perm should map a[i] to closest element index in b
    mapped = b[jnp.array(perm)]
    assert jnp.allclose(mapped, a)


def test_trapz_zero_avoiding_tail_removal():
    # Integrate a simple linear function and remove the last trapezoid explicitly
    x = jnp.linspace(0.0, 1.0, 6)  # 0,0.2,...,1.0
    y = 2.0 * x  # integral over [0,1] is 1.0

    # Full integral
    full_I = trapz_zero_avoiding(y, x, tail_idx=len(x) - 2)  # should drop last slice
    # Compute expected: integral over [0,0.8]
    # For linear y=2x, integral is x^2 from 0 to 0.8 -> 0.64
    assert jnp.isclose(full_I, 0.64, atol=1e-12)
