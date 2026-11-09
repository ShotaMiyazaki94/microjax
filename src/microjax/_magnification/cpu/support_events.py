"""Fixed Newton solves for projection tangencies and critical crossings."""

from __future__ import annotations
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
from microjax.poly_solver import poly_roots
from .lens import lens_map, lens_shear, source_polynomial
from .support import physical_image_mask, polish_images

Array = jnp.ndarray


class _CriticalEventImages(NamedTuple):
    images: Array
    physical_mask: Array
    critical_images: Array
    critical_valid: Array


def _event_equations(
    xy: Array,
    w_center: Array,
    rho: Array,
    lens,
    *,
    critical: bool,
    axis: complex | Array,
) -> Array:
    """Combine the source boundary with a fold or projection tangency."""

    z = xy[0] + 1j * xy[1]
    mapped = lens_map(z, lens) - w_center
    shear = lens_shear(z, lens)
    level = (jnp.real(mapped * jnp.conj(mapped)) - rho * rho) / (rho * rho)
    # This choice is static: both fixed solves run for every Cartesian source.
    if critical:
        condition = 1 - jnp.real(shear * jnp.conj(shear))
    else:
        direction = 1j * axis
        derivative = direction + shear * jnp.conj(direction)
        condition = 2 * jnp.real(jnp.conj(mapped) * derivative) / rho
    return jnp.stack([level, condition])


@partial(jax.jit, static_argnames=["critical"])
def _refine_support_events(
    seeds: Array,
    w_center: Array,
    rho: Array,
    lens,
    *,
    critical: bool,
    axis: complex | Array = 1 + 0j,
) -> tuple[Array, Array, Array]:
    """Use one fixed Newton schedule with a nearest-pole trust radius."""

    def equations(xy: Array) -> Array:
        return _event_equations(xy, w_center, rho, lens, critical=critical, axis=axis)

    jacobian = jax.jacfwd(equations)
    xy = jnp.stack([seeds.real, seeds.imag], axis=-1)

    def step(_: int, xy: Array) -> Array:
        f = jax.vmap(equations)(xy)
        j = jax.vmap(jacobian)(xy)
        det = j[:, 0, 0] * j[:, 1, 1] - j[:, 1, 0] * j[:, 0, 1]
        delta = jnp.stack(
            [
                (j[:, 1, 1] * f[:, 0] - j[:, 0, 1] * f[:, 1]) / det,
                (-j[:, 1, 0] * f[:, 0] + j[:, 0, 0] * f[:, 1]) / det,
            ],
            axis=-1,
        )
        z = xy[:, 0] + 1j * xy[:, 1]
        radius = 0.25 * jnp.min(jnp.abs(z[:, None] - lens.positions), axis=-1)
        length = jnp.linalg.norm(delta, axis=-1)
        delta = delta * jnp.minimum(1.0, radius / jnp.maximum(length, jnp.finfo(xy.dtype).tiny))[:, None]
        delta = jnp.where(jnp.isfinite(delta), delta, 0.0)
        return xy - delta

    xy = jax.lax.fori_loop(0, 16, step, xy)
    residual = jax.vmap(equations)(xy)
    z = xy[:, 0] + 1j * xy[:, 1]
    valid = jnp.all(jnp.abs(residual) < 1e-6, axis=-1) & jnp.all(jnp.isfinite(xy), axis=-1)
    return z, valid, residual


def _critical_event_image_seeds(
    image_limb: Array,
    physical_mask: Array,
    w_center: Array,
    rho: Array,
    lens,
) -> _CriticalEventImages:
    """Probe one phase between two continuously resolved fold crossings."""

    n_limb = image_limb.shape[1]
    phases = 2 * jnp.pi * jnp.arange(n_limb) / n_limb
    mapped = lens_map(image_limb, lens)
    residual = jnp.abs(mapped - (w_center + rho * jnp.exp(1j * phases))[None, :])
    ghost = jnp.where(~physical_mask, residual, jnp.inf)
    column = jnp.argmin(jnp.min(ghost, axis=0))
    candidates = jnp.argsort(ghost[:, column])[:2]
    seeds = image_limb[candidates, column]
    critical, valid, _ = _refine_support_events(seeds, w_center, rho, lens, critical=True)
    delta = lens_map(critical, lens) - w_center
    directions = delta / jnp.maximum(jnp.abs(delta), jnp.finfo(rho.dtype).tiny)
    separation = jnp.abs(jnp.angle(directions[0] * jnp.conj(directions[1])))
    resolved = (
        jnp.all(valid) & (separation > 64 * jnp.finfo(rho.dtype).eps) & jnp.all(jnp.isfinite(ghost[candidates, column]))
    )
    phase = jnp.angle(jnp.sum(directions))
    # An unresolved event still takes the same fixed source solve, but cannot
    # add an endpoint. This also covers a trace with no rejected image pair.
    phase = jnp.where(resolved, phase, 0.0)
    source = w_center + rho * jnp.exp(1j * phase)
    coefficients, image_shift = source_polynomial(source, lens)
    images = (
        poly_roots(
            coefficients[None, :],
            custom_init=True,
            roots_init=(image_limb[:, column] - image_shift)[None, :],
        )[0]
        + image_shift
    )
    images = polish_images(images, source, lens)
    physical = physical_image_mask(images, source, lens)
    return _CriticalEventImages(images, resolved & physical, critical, valid)
