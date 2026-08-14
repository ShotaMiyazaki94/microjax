"""Fixed-shape two-stage source-limb trace for CPU polar ICRS experiments."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from microjax.poly_solver import poly_roots

from ..geometry.lens import binary_geometry
from .coefficients import binary_quintic_coefficients
from .support import (
    _physical_image_mask,
    _polish_binary_images,
    trace_binary_source_limb,
    tracked_limb_neighbors,
)

Array = jnp.ndarray


class RefinedLimbTrace(NamedTuple):
    """A 32+32 trace and the non-uniform phases used to construct it."""

    image_limb: Array
    physical_mask: Array
    phases: Array
    inserted_fraction: Array
    interval_score: Array


def _image_phase_derivative(
    image: Array,
    phase: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array]:
    """Return ``dz/dphi`` and the real lens-Jacobian determinant."""

    lens = binary_geometry(s, q)
    image_midpoint = image - lens.shifted
    shear = (
        lens.e1 / (jnp.conjugate(image_midpoint) - lens.a) ** 2
        + (1.0 - lens.e1) / (jnp.conjugate(image_midpoint) + lens.a) ** 2
    )
    shear_abs = jnp.abs(shear)
    determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
    source_derivative = 1.0j * rho * jnp.exp(1.0j * phase)
    safe = jnp.abs(determinant) > jnp.sqrt(jnp.finfo(image.real.dtype).eps)
    derivative = (
        source_derivative - shear * jnp.conjugate(source_derivative)
    ) / jnp.where(safe, determinant, 1.0)
    return jnp.where(safe, derivative, 0.0 + 0.0j), determinant


def plan_refined_limb_phases(
    coarse_limb: Array,
    coarse_mask: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array]:
    """Place one diagnostic sample inside every coarse source-phase interval.

    Coverage is never traded for importance sampling: every one of the 32
    coarse intervals receives one extra point.  Its position is moved toward
    a bracketed radial turn, or otherwise toward the endpoint with the worse
    Jacobian conditioning on the branch with the largest polar motion.
    """

    n_coarse = coarse_limb.shape[1]
    dtype = coarse_limb.real.dtype
    step = 2.0 * jnp.pi / n_coarse
    phases = step * jnp.arange(n_coarse, dtype=dtype)
    next_phases = phases + step
    _, following, _, following_mask = tracked_limb_neighbors(
        coarse_limb, coarse_mask
    )

    derivative, determinant = _image_phase_derivative(
        coarse_limb,
        phases[None, :],
        rho,
        s=s,
        q=q,
    )
    following_derivative, following_determinant = _image_phase_derivative(
        following,
        next_phases[None, :],
        rho,
        s=s,
        q=q,
    )
    radius = jnp.abs(coarse_limb)
    following_radius = jnp.abs(following)
    radial_derivative = jnp.real(jnp.conjugate(coarse_limb) * derivative) / jnp.maximum(
        radius, jnp.finfo(dtype).tiny
    )
    following_radial_derivative = jnp.real(
        jnp.conjugate(following) * following_derivative
    ) / jnp.maximum(following_radius, jnp.finfo(dtype).tiny)

    connected = coarse_mask & following_mask
    angular_motion = jnp.abs(
        jnp.angle(following * jnp.conjugate(coarse_limb))
    )
    radial_motion = jnp.abs(following_radius - radius) / jnp.maximum(
        rho, jnp.finfo(dtype).tiny
    )
    conditioning = 1.0 / jnp.sqrt(
        jnp.maximum(
            jnp.abs(determinant * following_determinant),
            jnp.sqrt(jnp.finfo(dtype).eps),
        )
    )
    branch_score = jnp.where(
        connected,
        angular_motion * conditioning + 0.25 * radial_motion,
        -jnp.inf,
    )
    selected_branch = jnp.argmax(jax.lax.stop_gradient(branch_score), axis=0)
    slot = jnp.arange(n_coarse, dtype=jnp.int32)

    selected_j0 = jnp.abs(determinant[selected_branch, slot])
    selected_j1 = jnp.abs(following_determinant[selected_branch, slot])
    has_selected_branch = jnp.any(connected, axis=0)
    turning = (
        connected
        & jnp.isfinite(radial_derivative)
        & jnp.isfinite(following_radial_derivative)
        & (radial_derivative * following_radial_derivative <= 0.0)
    )
    turning_score = jnp.where(turning, branch_score, -jnp.inf)
    turning_branch = jnp.argmax(jax.lax.stop_gradient(turning_score), axis=0)
    has_turning = jnp.any(turning, axis=0)
    turn_d0 = radial_derivative[turning_branch, slot]
    turn_d1 = following_radial_derivative[turning_branch, slot]
    turn_fraction = jnp.abs(turn_d0) / jnp.maximum(
        jnp.abs(turn_d0) + jnp.abs(turn_d1),
        jnp.finfo(dtype).tiny,
    )
    # Linear interpolation in sqrt(|det J|) biases the point toward the less
    # conditioned endpoint without allowing either half of the interval to
    # become arbitrarily large.
    determinant_fraction = jnp.sqrt(selected_j0) / jnp.maximum(
        jnp.sqrt(selected_j0) + jnp.sqrt(selected_j1),
        jnp.finfo(dtype).tiny,
    )
    mask_transition = jnp.any(coarse_mask != following_mask, axis=0)
    fraction = jnp.where(has_turning, turn_fraction, determinant_fraction)
    fraction = jnp.where(mask_transition | ~has_selected_branch, 0.5, fraction)
    # Retain a global-resolution floor: neither half of a coarse interval may
    # exceed 80% of its width.  The 0.2--0.8 window was faster and at least as
    # fail-closed as the tested 0.35--0.65 alternative in the dense audit.
    fraction = jax.lax.stop_gradient(jnp.clip(fraction, 0.2, 0.8))
    inserted = phases + fraction * step
    refined = jnp.stack((phases, inserted), axis=1).reshape(-1)
    interval_score = jnp.max(branch_score, axis=0)
    interval_score = jnp.where(has_selected_branch, interval_score, 0.0)
    return refined, fraction, jax.lax.stop_gradient(interval_score)


def trace_binary_source_limb_two_stage(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_coarse: int = 32,
) -> RefinedLimbTrace:
    """Trace a fixed 32+32-style non-uniform source limb.

    Each inserted point is continued from the coarse root immediately to its
    left.  The coarse roots are retained, so this performs exactly
    ``2 * n_coarse`` quintic solves rather than re-solving a 64-point trace.
    """

    if n_coarse <= 0:
        raise ValueError("n_coarse must be positive")
    w_center = jnp.asarray(w_center)
    dtype = w_center.real.dtype
    coarse_phases = (
        2.0 * jnp.pi * jnp.arange(n_coarse, dtype=dtype) / n_coarse
    )
    coarse_limb, coarse_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_coarse,
        phases=coarse_phases,
    )
    refined_phases, inserted_fraction, interval_score = plan_refined_limb_phases(
        coarse_limb,
        coarse_mask,
        jnp.asarray(rho, dtype=dtype),
        s=jnp.asarray(s, dtype=dtype),
        q=jnp.asarray(q, dtype=dtype),
    )
    inserted_phases = refined_phases.reshape(n_coarse, 2)[:, 1]
    inserted_source = w_center + jnp.asarray(rho, dtype=dtype) * jnp.exp(
        1.0j * inserted_phases
    )
    quintics = binary_quintic_coefficients(inserted_source, s=s, q=q)
    coarse_planet = jnp.moveaxis(coarse_limb, 0, 1) - quintics.image_shift

    def continue_one(coefficients, initial):
        return poly_roots(
            coefficients[None, :],
            custom_init=True,
            roots_init=initial[None, :],
        )[0]

    inserted_planet = jax.vmap(continue_one)(
        quintics.coefficients,
        coarse_planet,
    )
    inserted_com = inserted_planet + quintics.image_shift
    polished_inserted = jax.vmap(
        lambda images, source: _polish_binary_images(images, source, s, q)
    )(inserted_com, inserted_source)
    inserted_mask = jax.vmap(
        lambda images, source: _physical_image_mask(images, source, s, q)
    )(polished_inserted, inserted_source)
    # Keep the inserted coordinates consistent with the mask computed from
    # the polished roots.  Returning the pre-polish quintic roots here would
    # reintroduce the same trace/support mismatch that the main limb solver
    # avoids.
    inserted_limb = jnp.moveaxis(polished_inserted, 0, 1)
    inserted_mask = jnp.moveaxis(inserted_mask, 0, 1)
    image_limb = jnp.stack((coarse_limb, inserted_limb), axis=2).reshape(
        coarse_limb.shape[0], 2 * n_coarse
    )
    physical_mask = jnp.stack((coarse_mask, inserted_mask), axis=2).reshape(
        coarse_mask.shape[0], 2 * n_coarse
    )
    return RefinedLimbTrace(
        image_limb,
        physical_mask,
        refined_phases,
        inserted_fraction,
        interval_score,
    )


__all__ = [
    "RefinedLimbTrace",
    "plan_refined_limb_phases",
    "trace_binary_source_limb_two_stage",
]
