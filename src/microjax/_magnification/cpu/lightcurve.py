"""CPU trajectory evaluation: multipoles followed by compact scalar solves."""

import jax.numpy as jnp
from jax import jit, lax, tree_util, vmap

from .one_shot import mag_cpu_one_shot
from .lens import binary_lens, triple_lens, source_polynomial
from .result import CpuMagnificationResult, INVALID_ROOTS, NONFINITE
from .support import polish_images, physical_image_mask
from ..selection import _binary_prefilter, _caustics_proximity_test
from microjax.lens_geometry import triple_lens_geometry
from microjax.multipole import mag_hexadecapole
from microjax.poly_solver import poly_roots

Array = jnp.ndarray

# A calibrated shortcut policy, not a full-solve error bound.
_CPU_MULTIPOLE_GATE = 1.0e-3


@jit
def mag_binary_cpu_lightcurve(w_points, rho, *, s, q, u1=0.0):
    """Full finite-source calculation, useful for isolating solver regressions."""
    return lax.map(
        lambda source: mag_cpu_one_shot(source, rho, lens=binary_lens(s, q), u1=u1),
        jnp.asarray(w_points),
    )


@jit
def mag_cpu_hybrid_lightcurve(
    w_points,
    multipole,
    accepted,
    correction,
    trigger_scale,
    rho,
    *,
    lens,
    u1=0.0,
):
    """Solve only sources rejected by the shared geometric multipole gate."""
    w_points = jnp.asarray(w_points)
    accepted = (
        accepted
        & jnp.isfinite(trigger_scale)
        & jnp.isfinite(correction)
        & (correction <= _CPU_MULTIPOLE_GATE * trigger_scale)
    )
    result = CpuMagnificationResult(
        multipole,
        jnp.zeros(w_points.shape, dtype=jnp.int32),
    )
    if w_points.shape[0] == 0:
        return result

    count = jnp.sum(~accepted, dtype=jnp.int32)
    indices = jnp.nonzero(~accepted, size=w_points.shape[0], fill_value=0)[0]

    def evaluate(i, values):
        index = indices[i]
        solved = mag_cpu_one_shot(w_points[index], rho, lens=lens, u1=u1)
        return tree_util.tree_map(
            lambda array, value: array.at[index].set(value),
            values,
            solved,
        )

    return lax.fori_loop(jnp.int32(0), count, evaluate, result)


@jit
def mag_binary(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
):
    """Fuse the CPU multipole prefilter and selected one-shot scheduler."""

    # Full-solve triggering is deliberately profile-independent: uniform and
    # limb-darkened sources use the same geometric multipole selector.  A
    # formerly relaxed uniform-only gate admitted non-convergent
    # hexadecapole series for source discs intersecting a close-binary
    # caustic, while the identical LD geometry correctly entered full ICRS.
    multipole, accepted, multipole_error, multipole_trigger_scale = _binary_prefilter(
        w_points,
        rho,
        u1,
        s,
        q,
    )
    return mag_cpu_hybrid_lightcurve(
        w_points,
        multipole,
        accepted,
        multipole_error,
        multipole_trigger_scale,
        rho,
        lens=binary_lens(s, q),
        u1=u1,
    )


@jit
def mag_triple(w_points, rho, *, s, q, q3, r3, psi, u1=0.0):
    """Use CPU image roots, a profile-independent gate, and the shared integrator."""
    w_points = jnp.asarray(w_points)
    if w_points.shape[0] == 0:
        return CpuMagnificationResult(jnp.zeros_like(w_points.real), jnp.zeros(w_points.shape, dtype=jnp.int32))
    lens = triple_lens(s, q, q3, r3, psi)
    geometry = triple_lens_geometry(s, q, q3, r3, psi)
    distances = jnp.abs(lens.positions[:, None] - lens.positions[None, :])
    valid_lens = (
        jnp.all(jnp.isfinite(lens.positions))
        & jnp.all(jnp.isfinite(lens.masses) & (lens.masses > 0))
        & jnp.all((distances > 0) | jnp.eye(3, dtype=bool))
        & (s > 0)
        & (rho > 0)
    )

    def evaluate(_):
        coefficients, image_shift = vmap(lambda w: source_polynomial(w, lens))(w_points)
        images = poly_roots(coefficients) + image_shift[:, None]
        images = vmap(lambda z, w: polish_images(z, w, lens))(images, w_points)
        mask = vmap(lambda z, w: physical_image_mask(z, w, lens))(images, w_points)
        # Multipole utilities use the internal midpoint frame and image-first axes.
        z = (images - geometry.shifted).T
        z_mask = mask.T
        parameters = dict(a=geometry.a, e1=geometry.e1, e2=geometry.e2, r3=r3, psi=psi, r3_complex=geometry.r3_complex)
        values, corrections = mag_hexadecapole(
            z[..., None],
            z_mask[..., None],
            rho,
            nlenses=3,
            u1=jnp.stack((u1, jnp.zeros_like(u1))),
            **parameters,
        )
        accepted = _caustics_proximity_test(
            w_points - geometry.shifted,
            z,
            z_mask,
            rho,
            corrections[..., 1],
            nlenses=3,
            **parameters,
        )
        count = jnp.sum(mask, axis=-1)
        accepted = accepted & (count >= 4) & (count % 2 == 0)
        return mag_cpu_hybrid_lightcurve(
            w_points,
            values[..., 0],
            accepted,
            corrections[..., 1],
            jnp.maximum(jnp.abs(values[..., 1]), 1.0),
            rho,
            lens=lens,
            u1=u1,
        )

    def invalid(_):
        return CpuMagnificationResult(
            jnp.full(w_points.shape, jnp.nan, dtype=w_points.real.dtype),
            jnp.full(w_points.shape, INVALID_ROOTS | NONFINITE, dtype=jnp.int32),
        )

    return lax.cond(valid_lens, evaluate, invalid, operand=None)
