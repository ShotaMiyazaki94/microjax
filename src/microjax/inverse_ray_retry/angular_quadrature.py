"""Resolution-free quadrature inside boundary-root angular intervals."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .angular import (
    ANGULAR_OK,
    ANGULAR_ROOT_FAILURE,
    AngularIntervals,
)
from .quadrature_rules import G15_W_ON_GK31, GK31_W, GK31_X


Array = jnp.ndarray


class AngularIntegral(NamedTuple):
    """Profile integral, absolute error estimate, and angular status bits."""

    value: Array
    error: Array
    status: Array


def _integrate_interval(
    integrand: Callable[[Array], Array], lower: Array, upper: Array
) -> tuple[Array, Array, Array]:
    """Apply embedded G15/K31 after a two-sided sine-squared map."""

    dtype = jnp.asarray(lower).dtype
    x = jnp.asarray(GK31_X, dtype=dtype)
    angle = 0.25 * jnp.pi * (x + 1.0)
    width = upper - lower
    theta = lower + width * jnp.sin(angle) ** 2
    jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)
    values = jax.vmap(integrand)(theta)
    kronrod = jnp.sum(jnp.asarray(GK31_W, dtype=dtype) * jacobian * values)
    gauss = jnp.sum(
        jnp.asarray(G15_W_ON_GK31, dtype=dtype) * jacobian * values
    )
    finite = jnp.isfinite(kronrod) & jnp.isfinite(gauss)
    return kronrod, jnp.abs(kronrod - gauss), finite


def _integrate_interval_subdivided(
    integrand: Callable[[Array], Array],
    lower: Array,
    upper: Array,
    subdivisions: int,
) -> tuple[Array, Array, Array]:
    """Integrate one inside arc on a small fixed number of child arcs."""

    if subdivisions == 1:
        return _integrate_interval(integrand, lower, upper)
    edges = jnp.linspace(lower, upper, subdivisions + 1)
    value, error, finite = jax.vmap(
        lambda lo, hi: _integrate_interval(integrand, lo, hi)
    )(edges[:-1], edges[1:])
    return jnp.sum(value), jnp.sum(error), jnp.all(finite)


def integrate_angular_profile(
    integrand: Callable[[Array], Array],
    intervals: AngularIntervals,
    *,
    endpoint_value_bound: Array = 0.0,
    subdivisions: int = 1,
) -> AngularIntegral:
    """Integrate a profile over all fixed-shape inside angular intervals.

    The sine-squared map cancels the square-root limb behaviour at both ends of
    every interval.  G15/K31 disagreement estimates the remaining profile
    quadrature error.  ``endpoint_value_bound`` converts the boundary-angle
    uncertainty reported by the root solver into an integral error bound.
    """

    if subdivisions not in (1, 2, 4):
        raise ValueError("subdivisions must be one of 1, 2, or 4")

    active = jnp.arange(intervals.intervals.shape[0]) < intervals.n_intervals

    def integrate_nonempty(bounds):
        # Duplicate the first physical interval into inactive static slots.
        # Evaluating their padded ``[0, 0]`` bounds can hit the square-root
        # singularity of a limb profile and poison reverse mode through a
        # zero cotangent. The duplicated results are still masked below.
        safe_bounds = jnp.where(active[:, None], bounds, bounds[0])
        values, errors, finite = jax.vmap(
            lambda pair: _integrate_interval_subdivided(
                integrand, pair[0], pair[1], subdivisions
            )
        )(safe_bounds)
        value = jnp.sum(jnp.where(active, values, 0.0))
        embedded_error = jnp.sum(jnp.where(active, errors, 0.0))
        error = (
            embedded_error
            + jnp.abs(endpoint_value_bound) * intervals.error
        )
        all_finite = (
            jnp.all(jnp.where(active, finite, True)) & jnp.isfinite(error)
        )
        status = jnp.bitwise_or(
            intervals.status,
            jnp.where(
                all_finite,
                jnp.int32(ANGULAR_OK),
                jnp.int32(ANGULAR_ROOT_FAILURE),
            ),
        )
        return AngularIntegral(value, error, status)

    def integrate_empty(_):
        dtype = intervals.intervals.dtype
        error = jnp.abs(endpoint_value_bound) * intervals.error
        status = jnp.bitwise_or(
            intervals.status,
            jnp.where(
                jnp.isfinite(error),
                jnp.int32(ANGULAR_OK),
                jnp.int32(ANGULAR_ROOT_FAILURE),
            ),
        )
        return AngularIntegral(jnp.asarray(0.0, dtype=dtype), error, status)

    return jax.lax.cond(
        intervals.n_intervals > 0,
        integrate_nonempty,
        integrate_empty,
        intervals.intervals,
    )
