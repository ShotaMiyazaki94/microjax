"""Fixed composite rules for event-partitioned inverse-ray integrals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .quadrature import GkIntegral


def _three_panel_sin2_arrays(order):
    """Three Gauss panels; the fixed order must be a positive multiple of 3."""

    nodes, weights = np.polynomial.legendre.leggauss(order // 3)
    edges = np.linspace(0, 1, 4)
    widths = np.diff(edges)
    angle = (np.pi / 2) * (
        edges[:-1, None] + widths[:, None] * (nodes[None, :] + 1) / 2
    )
    angle_weights = np.pi / 4 * widths[:, None] * weights[None, :]
    return (np.sin(angle).reshape(-1) ** 2,
            (angle_weights * np.sin(2 * angle)).reshape(-1))


def _three_panel_sin2_rule(order, dtype):
    return tuple(jnp.asarray(value, dtype=dtype)
                 for value in _three_panel_sin2_arrays(order))


def _three_panel_linear_rule(order, dtype):
    """Three linear panels; the fixed order must be a positive multiple of 3."""

    nodes, weights = np.polynomial.legendre.leggauss(order // 3)
    unit_nodes = np.concatenate([(i + .5 * (nodes + 1)) / 3 for i in range(3)])
    unit_weights = np.tile(weights / 6, 3)
    return (jnp.asarray(unit_nodes, dtype=dtype),
            jnp.asarray(unit_weights, dtype=dtype))


# Embed a four-node interpolatory rule in each seven-node Gauss panel.
# Its difference is an error indicator, not a certificate of support coverage.
_RADIAL21_X, _RADIAL21_W = _three_panel_sin2_arrays(21)
_GL7_X, _GL7_W = np.polynomial.legendre.leggauss(7)
_COARSE_INDICES = np.arange(0, 7, 2)
_COARSE_W = np.zeros(7)
_COARSE_W[_COARSE_INDICES] = np.linalg.solve(
    _GL7_X[_COARSE_INDICES][None, :] ** np.arange(4)[:, None],
    np.asarray([2., 0., 2. / 3., 0.]),
)
_EDGES = np.linspace(0, 1, 4)
_WIDTHS = np.diff(_EDGES)
_ANGLES = (np.pi / 2) * (
    _EDGES[:-1, None] + _WIDTHS[:, None] * (_GL7_X[None, :] + 1) / 2
)
_RADIAL21_COARSE = (np.pi / 4 * _WIDTHS[:, None] * _COARSE_W[None, :]
                    * np.sin(2 * _ANGLES)).reshape(-1)


def _integrate_radial21(integrand, intervals, active, *, subdivisions=1):
    """Evaluate the same 21-point composite rule on every radial cell."""

    fractions = jnp.linspace(0, 1, subdivisions + 1, dtype=intervals.dtype)
    lower = intervals[:, 0, None]
    width = intervals[:, 1, None] - lower
    start = (lower + width * fractions[:-1]).reshape(-1)
    end = (lower + width * fractions[1:]).reshape(-1)
    active = jnp.broadcast_to(active[:, None], (active.size, subdivisions)).reshape(-1)
    start = jnp.where(active, start, 1.)
    end = jnp.where(active, end, 1.)
    nodes, weights, coarse = [jnp.asarray(value, dtype=intervals.dtype)
                              for value in (_RADIAL21_X, _RADIAL21_W, _RADIAL21_COARSE)]
    width = end - start
    radii = start[:, None] + width[:, None] * nodes[None, :]
    values, errors, statuses = jax.vmap(jax.vmap(integrand))(radii)
    fine_weight = width[:, None] * weights[None, :]
    coarse_weight = width[:, None] * coarse[None, :]
    fine = jnp.sum(values * fine_weight, axis=1)
    low = jnp.sum(values * coarse_weight, axis=1)
    error = jnp.abs(fine - low) + jnp.sum(jnp.abs(fine_weight) * errors, axis=1)
    return GkIntegral(
        jnp.sum(jnp.where(active, fine, 0)),
        jnp.sum(jnp.where(active, error, 0)),
        jnp.bitwise_or.reduce(jnp.where(active[:, None], statuses, jnp.int32(0)).reshape(-1)),
        jnp.int32(21) * jnp.sum(active, dtype=jnp.int32),
    )
