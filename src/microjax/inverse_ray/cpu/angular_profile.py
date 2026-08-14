"""Angular brightness-residual quadrature for the CPU linear-LD kernel."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ..roots.angular import ANGULAR_OK, ANGULAR_ROOT_FAILURE, AngularIntervals

Array = jnp.ndarray

_JACOBI4_X = (
    -0.8090169943749475,
    -0.3090169943749474,
    0.3090169943749474,
    0.8090169943749475,
)
_JACOBI4_W = (
    0.21707871342270593,
    0.5683194499747423,
    0.5683194499747423,
    0.21707871342270593,
)
_JACOBI8_X = (
    -0.9396926207859084,
    -0.7660444431189781,
    -0.5,
    -0.17364817766693033,
    0.17364817766693033,
    0.5,
    0.7660444431189781,
    0.9396926207859084,
)
_JACOBI8_W = (
    0.0408329477091071,
    0.14422560079567256,
    0.2617993877991494,
    0.33854022709351916,
    0.33854022709351916,
    0.2617993877991494,
    0.14422560079567256,
    0.0408329477091071,
)
_LEGENDRE8_X = (
    -0.9602898564975362,
    -0.7966664774136267,
    -0.525532409916329,
    -0.18343464249564984,
    0.18343464249564984,
    0.525532409916329,
    0.7966664774136267,
    0.9602898564975362,
)
_LEGENDRE8_W = (
    0.10122853629037652,
    0.22238103445337445,
    0.31370664587788716,
    0.3626837833783618,
    0.3626837833783618,
    0.31370664587788716,
    0.22238103445337445,
    0.10122853629037652,
)
_LEGENDRE16_X = (
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878316,
    -0.7554044083550031,
    -0.6178762444026438,
    -0.45801677765722737,
    -0.2816035507792589,
    -0.09501250983763744,
    0.09501250983763744,
    0.2816035507792589,
    0.45801677765722737,
    0.6178762444026438,
    0.7554044083550031,
    0.8656312023878316,
    0.9445750230732326,
    0.9894009349916499,
)
_LEGENDRE16_W = (
    0.027152459411754055,
    0.06225352393864761,
    0.095158511682493,
    0.12462897125553395,
    0.14959598881657665,
    0.16915651939500262,
    0.1826034150449236,
    0.18945061045506859,
    0.18945061045506859,
    0.1826034150449236,
    0.16915651939500262,
    0.14959598881657665,
    0.12462897125553395,
    0.095158511682493,
    0.06225352393864761,
    0.027152459411754055,
)


class ResidualIntegral(NamedTuple):
    """Angular residual value, embedded error, and status."""

    value: Array
    error: Array
    status: Array


def _weighted_rule(
    mu: Callable[[Array], Array],
    lower: Array,
    upper: Array,
    nodes,
    weights,
) -> Array:
    dtype = lower.dtype
    x = jnp.asarray(nodes, dtype=dtype)
    weight = jnp.asarray(weights, dtype=dtype)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    theta = midpoint + half_width * x
    values = jax.vmap(mu)(theta)
    jacobi_weight = jnp.sqrt(
        jnp.maximum((1.0 - x) * (1.0 + x), jnp.finfo(dtype).tiny)
    )
    smooth = values / jacobi_weight
    return half_width * jnp.sum(weight * smooth)


def _legendre_rule(
    mu: Callable[[Array], Array],
    lower: Array,
    upper: Array,
    nodes,
    weights,
) -> Array:
    dtype = lower.dtype
    x = jnp.asarray(nodes, dtype=dtype)
    weight = jnp.asarray(weights, dtype=dtype)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    theta = midpoint + half_width * x
    return half_width * jnp.sum(weight * jax.vmap(mu)(theta))


def _integrate_interval(
    mu: Callable[[Array], Array], lower: Array, upper: Array
) -> tuple[Array, Array, Array]:
    width = upper - lower
    full_ring = width >= 2.0 * jnp.pi - 64.0 * jnp.finfo(lower.dtype).eps

    def integrate_full(_):
        coarse = _legendre_rule(mu, lower, upper, _LEGENDRE8_X, _LEGENDRE8_W)
        fine = _legendre_rule(mu, lower, upper, _LEGENDRE16_X, _LEGENDRE16_W)
        return fine, jnp.abs(fine - coarse), jnp.isfinite(fine)

    def integrate_bounded(_):
        coarse = _weighted_rule(mu, lower, upper, _JACOBI4_X, _JACOBI4_W)
        fine = _weighted_rule(mu, lower, upper, _JACOBI8_X, _JACOBI8_W)
        return fine, jnp.abs(fine - coarse), jnp.isfinite(fine)

    return jax.lax.cond(full_ring, integrate_full, integrate_bounded, operand=None)


def integrate_mu_residual(
    mu: Callable[[Array], Array], intervals: AngularIntervals
) -> ResidualIntegral:
    """Integrate ``sqrt(1-u**2)`` over fixed-shape inside arcs.

    Root-bounded arcs use Gauss-Jacobi rules with weight
    ``sqrt(1-x**2)``.  A completely inside ring has no limb endpoints and
    therefore uses ordinary Gauss-Legendre rules.
    """

    active = jnp.arange(intervals.intervals.shape[0]) < intervals.n_intervals

    def nonempty(bounds):
        safe = jnp.where(active[:, None], bounds, bounds[0])
        values, errors, finite = jax.vmap(
            lambda pair: _integrate_interval(mu, pair[0], pair[1])
        )(safe)
        value = jnp.sum(jnp.where(active, values, 0.0))
        error = jnp.sum(jnp.where(active, errors, 0.0))
        all_finite = jnp.all(jnp.where(active, finite, True))
        status = jnp.bitwise_or(
            intervals.status,
            jnp.where(
                all_finite, jnp.int32(ANGULAR_OK), jnp.int32(ANGULAR_ROOT_FAILURE)
            ),
        )
        return ResidualIntegral(value, error, status)

    def empty(_):
        dtype = intervals.intervals.dtype
        return ResidualIntegral(
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            intervals.status,
        )

    return jax.lax.cond(
        intervals.n_intervals > 0,
        nonempty,
        empty,
        intervals.intervals,
    )


__all__ = ["ResidualIntegral", "integrate_mu_residual"]
