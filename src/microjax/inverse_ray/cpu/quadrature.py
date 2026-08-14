"""Small fixed radial quadrature rules for CPU ICRS."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


def _symmetric(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values[:-1], values[::-1]))


_GK15_X_POSITIVE_DESC = np.asarray(
    [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.20778495500789847,
        0.0,
    ]
)
_GK15_W_POSITIVE_DESC = np.asarray(
    [
        0.022935322010529224,
        0.06309209262997855,
        0.10479001032225018,
        0.14065325971552592,
        0.1690047266392679,
        0.19035057806478542,
        0.20443294007529889,
        0.20948214108472782,
    ]
)
_G7_W_POSITIVE_DESC = np.asarray(
    [
        0.0,
        0.1294849661688697,
        0.0,
        0.27970539148927664,
        0.0,
        0.38183005050511894,
        0.0,
        0.4179591836734694,
    ]
)

GK15_X = np.concatenate((-_GK15_X_POSITIVE_DESC[:-1], _GK15_X_POSITIVE_DESC[::-1]))
GK15_W = _symmetric(_GK15_W_POSITIVE_DESC)
G7_W_ON_GK15 = _symmetric(_G7_W_POSITIVE_DESC)

GL11_X, GL11_W = np.polynomial.legendre.leggauss(11)
_GL11_EMBEDDED_INDICES = np.asarray([0, 2, 4, 5, 6, 8, 10])
_GL11_EMBEDDED_POSITIVE = GL11_X[[10, 8, 6]]
_GL11_EMBEDDED_SYSTEM = np.asarray(
    [
        [
            2.0 * _GL11_EMBEDDED_POSITIVE[0] ** degree,
            2.0 * _GL11_EMBEDDED_POSITIVE[1] ** degree,
            2.0 * _GL11_EMBEDDED_POSITIVE[2] ** degree,
            0.0,
        ]
        for degree in (2, 4, 6)
    ]
)
_GL11_EMBEDDED_SYSTEM = np.concatenate(
    (np.asarray([[2.0, 2.0, 2.0, 1.0]]), _GL11_EMBEDDED_SYSTEM),
    axis=0,
)
_GL11_EMBEDDED_MOMENTS = np.asarray(
    [2.0, 2.0 / 3.0, 2.0 / 5.0, 2.0 / 7.0]
)
_GL11_EMBEDDED_SYMMETRIC_WEIGHTS = np.linalg.solve(
    _GL11_EMBEDDED_SYSTEM, _GL11_EMBEDDED_MOMENTS
)
G7_W_ON_GL11 = np.zeros(11)
G7_W_ON_GL11[_GL11_EMBEDDED_INDICES] = np.asarray(
    [
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[0],
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[1],
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[2],
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[3],
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[2],
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[1],
        _GL11_EMBEDDED_SYMMETRIC_WEIGHTS[0],
    ]
)

GL21_X, GL21_W = np.polynomial.legendre.leggauss(21)
_GL21_EMBEDDED_INDICES = np.arange(0, 21, 2)
_GL21_EMBEDDED_SYSTEM = np.vstack(
    [GL21_X[_GL21_EMBEDDED_INDICES] ** degree for degree in range(11)]
)
_GL21_EMBEDDED_MOMENTS = np.asarray(
    [0.0 if degree % 2 else 2.0 / (degree + 1) for degree in range(11)]
)
G11_W_ON_GL21 = np.zeros(21)
G11_W_ON_GL21[_GL21_EMBEDDED_INDICES] = np.linalg.solve(
    _GL21_EMBEDDED_SYSTEM,
    _GL21_EMBEDDED_MOMENTS,
)

# A high-order linear-radius rule for cells crossed by a traced image limb.
# The odd-indexed GL31 nodes form a symmetric positive 15-node interpolatory
# rule, giving an embedded difference without evaluating another radius.
GL31_X, GL31_W = np.polynomial.legendre.leggauss(31)
_GL31_EMBEDDED_INDICES = np.arange(1, 31, 2)
_GL31_EMBEDDED_SYSTEM = np.vstack(
    [GL31_X[_GL31_EMBEDDED_INDICES] ** degree for degree in range(15)]
)
_GL31_EMBEDDED_MOMENTS = np.asarray(
    [0.0 if degree % 2 else 2.0 / (degree + 1) for degree in range(15)]
)
G15_W_ON_GL31 = np.zeros(31)
G15_W_ON_GL31[_GL31_EMBEDDED_INDICES] = np.linalg.solve(
    _GL31_EMBEDDED_SYSTEM,
    _GL31_EMBEDDED_MOMENTS,
)


class GkIntegral(NamedTuple):
    """Integral, conservative local error estimate, and combined status."""

    value: Array
    error: Array
    status: Array
    n_nodes: Array


def integrate_gk15(
    integrand,
    intervals: Array,
    active: Array,
    *,
    subdivisions: int = 1,
) -> GkIntegral:
    """Integrate a scalar fixed-shape integrand over active radial cells."""

    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    fractions = jnp.linspace(
        0.0,
        1.0,
        subdivisions + 1,
        dtype=intervals.dtype,
    )
    cell_lower = (
        intervals[:, 0, None]
        + (intervals[:, 1] - intervals[:, 0])[:, None] * fractions[:-1]
    )
    cell_upper = (
        intervals[:, 0, None]
        + (intervals[:, 1] - intervals[:, 0])[:, None] * fractions[1:]
    )
    intervals = jnp.stack((cell_lower, cell_upper), axis=-1).reshape((-1, 2))
    active = jnp.broadcast_to(active[:, None], (active.size, subdivisions)).reshape(-1)

    dtype = intervals.dtype
    nodes = jnp.asarray(GK15_X, dtype=dtype)
    kronrod_weights = jnp.asarray(GK15_W, dtype=dtype)
    gauss_weights = jnp.asarray(G7_W_ON_GK15, dtype=dtype)

    safe_lower = jnp.where(active, intervals[:, 0], 1.0)
    safe_upper = jnp.where(active, intervals[:, 1], 1.0)
    angle = 0.25 * jnp.pi * (nodes + 1.0)
    width = safe_upper - safe_lower
    radii = safe_lower[:, None] + width[:, None] * jnp.sin(angle)[None, :] ** 2
    jacobian = 0.25 * jnp.pi * width[:, None] * jnp.sin(2.0 * angle)[None, :]

    values, errors, statuses = jax.vmap(jax.vmap(integrand))(radii)
    kronrod_combined = jacobian * kronrod_weights[None, :]
    gauss_combined = jacobian * gauss_weights[None, :]
    kronrod = jnp.sum(values * kronrod_combined, axis=1)
    gauss = jnp.sum(values * gauss_combined, axis=1)
    propagated = jnp.sum(errors * jnp.abs(kronrod_combined), axis=1)
    cell_error = jnp.abs(kronrod - gauss) + propagated

    value = jnp.sum(jnp.where(active, kronrod, 0.0))
    error = jnp.sum(jnp.where(active, cell_error, 0.0))
    active_status = jnp.where(active[:, None], statuses, jnp.int32(0))
    status = jnp.bitwise_or.reduce(active_status.reshape(-1))
    n_nodes = jnp.sum(active, dtype=jnp.int32) * nodes.size
    return GkIntegral(value, error, status, n_nodes)


__all__ = [
    "G7_W_ON_GK15",
    "G7_W_ON_GL11",
    "G11_W_ON_GL21",
    "G15_W_ON_GL31",
    "GK15_W",
    "GK15_X",
    "GL11_W",
    "GL11_X",
    "GL21_W",
    "GL21_X",
    "GL31_W",
    "GL31_X",
    "GkIntegral",
    "integrate_gk15",
]
