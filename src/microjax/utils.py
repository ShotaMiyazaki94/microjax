# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Utility helpers shared across microJAX modules."""

__all__ = [
    "match_points",
    "first_nonzero",
    "last_nonzero",
    "first_zero",
    "min_zero_avoiding",
    "max_zero_avoiding",
    "mean_zero_avoiding",
    "sparse_argsort",
    "trapz_zero_avoiding",
]
import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray

def match_points(a: Array, b: Array) -> Array:
    """Greedy nearest-neighbour matching between two point sets.

    The algorithm iterates over ``a`` and, for each element, picks the closest
    unused element of ``b``.  It does **not** solve the optimal assignment
    problem but provides a fast, deterministic ordering that is sufficient for
    tracking lens images across sampling steps.

    Parameters
    ----------
    a : Array
        One-dimensional complex array of source points.
    b : Array
        One-dimensional complex array of candidate points to match against.

    Returns
    -------
    Array
        Indices that permute ``b`` into the greedy match order of ``a``.
    """
    # First guess
    vals = jnp.argsort(jnp.abs(b - a[:, None]), axis=1)
    idcs = []
    for i, idx in enumerate(vals[:, 0]):
        # If index is duplicate choose the next best solution
        mask = ~jnp.isin(vals[i], jnp.array(idcs), assume_unique=True)
        idx = vals[i, first_nonzero(mask)]
        idcs.append(idx)

    return jnp.array(idcs)


def first_nonzero(x: Array, axis: int = 0) -> Array:
    """Return the index of the first non-zero entry along ``axis``.

    Parameters
    ----------
    x : Array
        Input array.
    axis : int, optional
        Axis along which to search. Defaults to 0.

    Returns
    -------
    Array
        Index of the first non-zero entry. If no entry is non-zero the result is
        ``0``.
    """
    return jnp.argmax(x != 0.0, axis=axis)


def last_nonzero(x: Array, axis: int = 0) -> Array:
    """Return the index of the last non-zero entry along ``axis``.

    Parameters
    ----------
    x : Array
        Input array.
    axis : int, optional
        Axis along which to search. Defaults to 0.

    Returns
    -------
    Array
        Index of the last non-zero entry. If all entries are zero the result is
        ``0``.
    """
    return lax.cond(
        jnp.any(x, axis=axis),  # if any non-zero
        lambda: (x.shape[axis] - 1)
        - jnp.argmax(jnp.flip(x, axis=axis) != 0, axis=axis),
        lambda: 0,
    )


def first_zero(x: Array, axis: int = 0) -> Array:
    """Return the index of the first zero entry along ``axis``.

    Parameters
    ----------
    x : Array
        Input array.
    axis : int, optional
        Axis along which to search. Defaults to 0.

    Returns
    -------
    Array
        Index of the first zero entry. If no entry is zero the result is ``0``.
    """
    return jnp.argmax(x == 0.0, axis=axis)


def min_zero_avoiding(x: Array) -> Array:
    """Minimum value of ``x`` while skipping zeros.

    Parameters
    ----------
    x : Array
        One-dimensional input array.

    Returns
    -------
    Array
        Minimum non-zero value if present, otherwise ``0``.
    """
    x = jnp.sort(x)
    min_x = jnp.min(x)
    cond = min_x == 0.0
    return jnp.where(cond, x[(x != 0).argmax(axis=0)], min_x)


def max_zero_avoiding(x: Array) -> Array:
    """Maximum value of ``x`` while skipping zeros.

    Parameters
    ----------
    x : Array
        One-dimensional input array.

    Returns
    -------
    Array
        Maximum non-zero value if present, otherwise ``0``.
    """
    x = jnp.sort(x)
    max_x = jnp.max(x)
    cond = max_x == 0.0
    return jnp.where(cond, -min_zero_avoiding(jnp.abs(x)), max_x)


def mean_zero_avoiding(x: Array) -> Array:
    """Mean of non-zero entries in ``x``.

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    Array
        Mean over the non-zero entries or ``0`` if all entries are zero.
    """
    mask = x == 0.0
    return jnp.where(jnp.all(mask), 0.0, jnp.nanmean(jnp.where(mask, jnp.nan, x)))


def sparse_argsort(a: Array) -> Array:
    """Argsort that treats zeros as missing values.

    Parameters
    ----------
    a : Array
        Input array.

    Returns
    -------
    Array
        Indices that sort ``a`` after replacing zeros with ``NaN``.
    """
    return jnp.where(a != 0, a, jnp.nan).argsort()


def trapz_zero_avoiding(y: Array, x: Array, tail_idx: int) -> Array:
    """Trapezoidal integration omitting the final segment beyond ``tail_idx``.

    Equivalent to ``jax.scipy.integrate.trapezoid(y[:tail_idx+1], x[:tail_idx+1])``
    but avoids the last segment if ``tail_idx`` does not point to the final
    sample.

    Parameters
    ----------
    y : Array
        Function samples.
    x : Array
        Sample positions.
    tail_idx : int
        Index of the last reliable sample to include in the integral.

    Returns
    -------
    Array
        Trapezoidal integral up to ``tail_idx``.
    """
    I = jax.scipy.integrate.trapezoid(y, x=x)
    #I = jnp.trapz(y, x=x)
    xt, yt = x[tail_idx], y[tail_idx]
    xtp1, ytp1 = x[tail_idx + 1], y[tail_idx + 1]
    return lax.cond(
        tail_idx == len(x) - 1,
        lambda: I,
        lambda: I - 0.5 * ((yt + ytp1) * (xtp1 - xt)),
    )
