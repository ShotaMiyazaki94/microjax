"""Internal Kepler-equation solvers shared by trajectory submodules."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def solve_kepler_newton(
    M: float | Array,
    ecc: float | Array,
    *,
    n_iter: int = 10,
    init: str = "m_plus_esinm",
) -> Array:
    """Solve ``M = E - e sin(E)`` using fixed-count Newton iterations.

    Parameters
    ----------
    M : float or Array
        Mean anomaly.
    ecc : float or Array
        Eccentricity.
    n_iter : int, optional
        Number of Newton updates. Fixed iteration count keeps JIT/autodiff
        behavior predictable.
    init : {"m_plus_esinm", "parallax_empirical"}, optional
        Initial guess recipe:
        - ``"m_plus_esinm"``: ``E0 = M + e sin(M)``
        - ``"parallax_empirical"``: ``E0 = M + sign(sin(M))*0.85*e``

    Returns
    -------
    Array
        Eccentric anomaly ``E`` with broadcasted shape of ``M`` and ``ecc``.
    """
    M = jnp.asarray(M)
    ecc = jnp.asarray(ecc, dtype=M.dtype)

    if init == "m_plus_esinm":
        E = M + ecc * jnp.sin(M)
    elif init == "parallax_empirical":
        E = M + jnp.sign(jnp.sin(M)) * 0.85 * ecc
    else:
        raise ValueError(
            "Unsupported init strategy. Use 'm_plus_esinm' or 'parallax_empirical'."
        )

    for _ in range(n_iter):
        f = E - ecc * jnp.sin(E) - M
        fp = 1.0 - ecc * jnp.cos(E)
        E = E - f / fp

    return E
