"""Angular support and implicit tangency derivatives for the CPU solver."""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .lens import lens_map, lens_shear

Array = jnp.ndarray
ANGULAR_MOMENT_TOPOLOGY = 1 << 17
ANGULAR_MOMENT_EXHAUSTED = 1 << 18
ANGULAR_MOMENT_SUPPORT = 1 << 19


class AngularSupport(NamedTuple):
    """Angular cells and the diagnostics required to certify them."""

    cells: Array
    active: Array
    topology_uncertain: Array
    minimum_ghost_residual: Array
    limb_topology: Array
    tangencies_valid: Array


def _stable_radial_level_value(
    angles: Array,
    radii: Array,
    w_center: Array,
    rho: Array,
    *,
    lens,
) -> Array:
    """Evaluate ``|N| - rho |D|`` without squared-level-set cancellation."""

    trailing = max(0, radii.ndim - angles.ndim)
    unit = jnp.exp(1j * jnp.reshape(angles, angles.shape + (1,) * trailing))
    image = radii * unit
    factors = jnp.conj(image[..., None] - lens.positions)
    denominator = jnp.prod(factors, axis=-1)
    numerator = (image - w_center) * denominator
    for index in range(lens.positions.shape[0]):
        other = jnp.concatenate((factors[..., :index], factors[..., index + 1 :]), axis=-1)
        numerator = numerator - lens.masses[index] * jnp.prod(other, axis=-1)
    return jnp.abs(numerator) - rho * jnp.abs(denominator)


def _trace_sector_union_mask(
    cells: Array,
    physical_limb: Array,
    physical_mask: Array,
    following_limb: Array,
    outgoing_edge_active: Array,
    *,
    conservative_guard: Array | bool = False,
) -> Array:
    """Return polar sectors swept by the tracked image-limb polylines.

    This is the branch-aware version of the original GPU-v1 support builder:
    image-limb coordinates alone determine which angular sectors can contain
    an image.  In particular, support construction performs no independent
    radial all-root solve at cell midpoints.  The extrema partition keeps the
    square-root onsets at quadrature endpoints, while consecutive tracked limb
    samples identify the occupied side of that partition.

    A physical sample also retains its containing sector, covering a fold image
    represented by only one sample.  ``conservative_guard`` retains the full
    one-dimensional partition for a trace-proven hidden-loop or crossing case.
    """

    nonempty = cells[:, 1] > cells[:, 0]

    def trace_support(_):
        midpoint = 0.5 * (cells[:, 0] + cells[:, 1])
        sample_angles = jnp.mod(jnp.angle(physical_limb), 2.0 * jnp.pi).reshape(-1)
        sample_mask = physical_mask.reshape(-1) & jnp.isfinite(sample_angles)
        half_width = 0.5 * (cells[:, 1] - cells[:, 0])
        circular_distance = jnp.abs(jnp.angle(jnp.exp(1.0j * (sample_angles[None, :] - midpoint[:, None]))))
        endpoint_tolerance = 64.0 * jnp.finfo(cells.dtype).eps
        contains_limb_sample = jnp.any(
            sample_mask[None, :] & (circular_distance <= half_width[:, None] + endpoint_tolerance),
            axis=1,
        )

        start = jnp.mod(jnp.angle(physical_limb), 2.0 * jnp.pi)
        step = jnp.angle(following_limb * jnp.conjugate(physical_limb))
        connected = outgoing_edge_active & jnp.isfinite(start) & jnp.isfinite(step)
        relative = jnp.angle(jnp.exp(1.0j * (midpoint[:, None, None] - start[None, :, :])))
        tolerance = 64.0 * jnp.finfo(cells.dtype).eps
        swept_positive = (
            (step[None, :, :] >= 0.0) & (relative >= -tolerance) & (relative <= step[None, :, :] + tolerance)
        )
        swept_negative = (step[None, :, :] < 0.0) & (relative <= tolerance) & (relative >= step[None, :, :] - tolerance)
        swept = jnp.any(
            connected[None, :, :] & (swept_positive | swept_negative),
            axis=(1, 2),
        )
        return nonempty & (swept | contains_limb_sample)

    # A trace-proven hidden loop or 3-to-5 cut is the one case where a
    # physical-limb-only sector union is not sufficient: the occupied sector
    # can be absent from the boundary sample altogether.  Keep the existing
    # one-dimensional partition as a conservative guard only for that
    # structural witness.  This does not create a product support or use any
    # lens/source parameter band.
    conservative_guard = jnp.asarray(conservative_guard)
    return jax.lax.cond(
        conservative_guard,
        lambda _: nonempty,
        lambda _: trace_support(None),
        operand=None,
    )


def _angular_support_cells_from_trace(
    w_center: Array,
    rho: Array,
    *,
    lens,
    physical_limb: Array,
    physical_mask: Array,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    neighbors: tuple[Array, Array, Array, Array],
    edge_activity: tuple[Array, Array],
) -> AngularSupport:
    """Build polar support from an already traced source boundary.

    Cartesian and polar moments are two coordinate charts of the same image
    region. Accepting a shared trace here avoids a second boundary-root solve
    when a poorly conditioned Cartesian projection needs an independent
    angular certificate.
    """

    previous_limb, following_limb, previous_mask, following_mask = neighbors
    incoming_edge_active, outgoing_edge_active = edge_activity
    previous_step = jnp.angle(physical_limb * jnp.conjugate(previous_limb))
    following_step = jnp.angle(following_limb * jnp.conjugate(physical_limb))
    sampled_sign_change = previous_mask & following_mask & (previous_step * following_step <= 0.0)
    previous_endpoint = physical_mask & ~previous_mask
    following_endpoint = physical_mask & ~following_mask
    physical_endpoint = previous_endpoint | following_endpoint
    candidate_mask = physical_mask & (sampled_sign_change | physical_endpoint)

    center = jnp.angle(physical_limb)
    minus = center - previous_step
    plus = center + following_step
    curvature = minus - 2.0 * center + plus
    safe_curvature = jnp.where(
        jnp.abs(curvature) > 64.0 * jnp.finfo(center.dtype).eps,
        curvature,
        1.0,
    )
    raw_offset = 0.5 * (minus - plus) / safe_curvature
    offset = jnp.clip(raw_offset, -1.0, 1.0)
    fitted = center + 0.5 * (plus - minus) * offset + 0.5 * curvature * offset**2
    fitted = jnp.where(
        incoming_edge_active & outgoing_edge_active,
        fitted,
        center,
    )
    fitted = jnp.mod(fitted, 2.0 * jnp.pi)

    maximum_extrema = 4 * physical_limb.shape[0] + 4
    flat_mask = candidate_mask.reshape(-1)
    # Every sampled turn and physical branch endpoint is refined. Selected
    # indices come from this mask; there is no separate brightness policy.
    # ``trace_binary_source_limb`` carries every quintic root forward as the
    # next solve's initial value. Opposite signed angular increments on one
    # continuously tracked branch therefore bracket a real support extremum,
    # even when the 2-D tangency Newton system is singular. Keep this as an
    # independent support certificate, guarded against root-slot jumps.
    previous_jump = jnp.abs(physical_limb - previous_limb)
    following_jump = jnp.abs(following_limb - physical_limb)
    image_scale = 1.0 + jnp.abs(physical_limb)
    branch_bracketed = (
        candidate_mask
        & ~physical_endpoint
        & incoming_edge_active
        & outgoing_edge_active
        & jnp.isfinite(previous_limb)
        & jnp.isfinite(physical_limb)
        & jnp.isfinite(following_limb)
        & (jnp.abs(previous_step) <= 0.5)
        & (jnp.abs(following_step) <= 0.5)
        & (previous_jump <= 0.5 * image_scale)
        & (following_jump <= 0.5 * image_scale)
    ).reshape(-1)
    indices = jnp.nonzero(flat_mask, size=maximum_extrema, fill_value=-1)[0]
    safe_indices = jnp.maximum(indices, 0)
    selected = fitted.reshape(-1)[safe_indices]
    selected = jnp.where(indices >= 0, selected, 2.0 * jnp.pi)
    selected_images = physical_limb.reshape(-1)[safe_indices]
    sampled_angles = jnp.mod(center, 2.0 * jnp.pi).reshape(-1)[safe_indices]
    sampled_angles = jnp.where(indices >= 0, sampled_angles, 2.0 * jnp.pi)
    bracketed = branch_bracketed[safe_indices] & (indices >= 0)

    selected, tangencies_valid = _refine_angular_tangencies(
        w_center,
        rho,
        lens=lens,
        initial_images=selected_images,
        initial_angles=sampled_angles,
        fallback_angles=selected,
        active=indices >= 0,
        required=indices >= 0,
        bracketed=bracketed,
    )
    n_extrema = jnp.sum(flat_mask, dtype=jnp.int32)
    topology_uncertain = topology_uncertain | (n_extrema > maximum_extrema)
    trace_finite = jnp.all(jnp.isfinite(physical_limb))
    topology_uncertain = topology_uncertain | ~trace_finite | (n_extrema == 0)
    # A malformed trace must not be downgraded to a soft topology warning:
    # otherwise the fallback full-circle cell could agree at two quadrature
    # orders and be certified as a zero/incorrect support.  Reuse the existing
    # support-validity channel so every angular caller fails closed.
    trace_support_valid = trace_finite & (n_extrema > 0)
    tangencies_valid = jnp.where(
        trace_support_valid,
        tangencies_valid,
        jnp.zeros_like(tangencies_valid),
    )
    endpoints = jnp.sort(
        jnp.concatenate(
            (
                jnp.asarray([0.0, 2.0 * jnp.pi], dtype=w_center.real.dtype),
                selected,
            )
        )
    )
    cells = jnp.stack((endpoints[:-1], endpoints[1:]), axis=-1)
    physical_counts = jnp.sum(physical_mask, axis=0, dtype=jnp.int32)
    ghost_ratio = minimum_ghost_residual / jnp.maximum(rho, jnp.finfo(cells.dtype).tiny)
    ghost_guard_valid = jnp.isfinite(ghost_ratio) & (ghost_ratio < 1.5)
    spacing_guard_valid = jnp.max(jnp.abs(jnp.angle(following_limb * jnp.conjugate(physical_limb)))) >= 5.0e-4
    crossing_guard = (
        limb_topology & (jnp.max(physical_counts) > jnp.min(physical_counts)) & ghost_guard_valid & spacing_guard_valid
    )
    active = _trace_sector_union_mask(
        cells,
        physical_limb,
        physical_mask,
        following_limb,
        outgoing_edge_active,
        conservative_guard=crossing_guard,
    )
    return AngularSupport(
        cells,
        active,
        topology_uncertain,
        minimum_ghost_residual,
        limb_topology,
        tangencies_valid,
    )


@jax.custom_jvp
def _implicit_angular_tangency_angles(angles, radii, converged, w_center, rho, lens):
    """Keep the refined angles and differentiate only converged tangencies."""

    return angles


@_implicit_angular_tangency_angles.defjvp
def _implicit_angular_tangency_angles_jvp(primals, tangents):
    angles, radii, converged, w_center, rho, lens = primals
    _, _, _, dw, drho, dlens = tangents
    # Failed Newton lanes retain their historical zero endpoint derivative.
    # Benign coordinates prevent undefined auxiliary tangents in those lanes.
    radius = jnp.where(converged, radii, 0.417)
    angle = jnp.where(converged, angles, 0.263)
    image = radius * jnp.exp(1.0j * angle)

    def level(w_center, rho, lens):
        offset = lens_map(image, lens) - w_center
        return jnp.real(offset * jnp.conj(offset)) - rho * rho

    shear = lens_shear(image, lens)
    offset = lens_map(image, lens) - w_center
    image_motion = 1.0j * image - 1.0j * shear * jnp.conj(image)
    angular_derivative = 2 * jnp.real(jnp.conj(offset) * image_motion)
    _, parameter_derivative = jax.jvp(
        level,
        (w_center, rho, lens),
        (dw, drho, dlens),
    )
    # H(r, theta, p)=0 and H_r=0 imply dtheta=-H_p/H_theta. No
    # differentiation of Newton iterates or 2x2 implicit solve is needed.
    # A genuinely singular accepted tangency is not regularized by epsilon.
    denominator = jnp.where(converged, angular_derivative, 1.0)
    derivative = jnp.where(converged, -parameter_derivative / denominator, 0.0)
    return angles, derivative


def _refine_angular_tangencies(
    w_center: Array,
    rho: Array,
    *,
    lens,
    initial_images: Array,
    initial_angles: Array,
    fallback_angles: Array,
    active: Array,
    required: Array,
    bracketed: Array,
) -> tuple[Array, Array]:
    """Project traced extrema onto exact radial level-set tangencies.

    An angular Gauss rule converges rapidly only when its cells terminate at
    the square-root singularities of the radial moment.  A sampled source limb
    locates those singularities, but even a dense trace leaves a phase error
    large enough for several Gauss orders to miss the same narrow fold.  The
    exact endpoints satisfy ``H(r, theta) = dH/dr = 0``; six damped Newton
    steps remove that trace-spacing error without densifying the support.

    Implicit derivatives of converged tangencies retain the movement of
    the finite quadrature nodes. Failed solves instead differentiate the
    fitted trace endpoint that is actually returned.
    At singular tangencies or discrete support/allocation changes, forward
    derivatives are not an accuracy certificate.
    """

    real_dtype = w_center.real.dtype
    rho_scale = jnp.maximum(rho**2, jnp.finfo(real_dtype).tiny)

    def refine_one(
        image,
        initial_angle,
        fallback_angle,
        is_bracketed,
    ):
        initial_state = jnp.stack((jnp.abs(image), initial_angle))

        def level_set(radius, angle):
            image_point = radius * jnp.exp(1.0j * angle)
            mapped = lens_map(image_point, lens)
            return (jnp.abs(mapped - w_center) ** 2 - rho**2) / rho_scale

        radial_derivative = jax.grad(level_set, argnums=0)

        def equations(state):
            radius, angle = state
            return jnp.stack(
                (
                    level_set(radius, angle),
                    radial_derivative(radius, angle),
                )
            )

        equation_jacobian = jax.jacfwd(equations)

        def newton_step(_, state):
            residual = equations(state)
            jacobian = equation_jacobian(state)
            determinant = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
            safe_determinant = jnp.where(
                jnp.abs(determinant) > 128.0 * jnp.finfo(real_dtype).eps,
                determinant,
                1.0,
            )
            radial_step = (jacobian[1, 1] * residual[0] - jacobian[0, 1] * residual[1]) / safe_determinant
            angular_step = (-jacobian[1, 0] * residual[0] + jacobian[0, 0] * residual[1]) / safe_determinant
            radial_limit = 0.25 * jnp.maximum(jnp.abs(state[0]), 1.0e-3)
            step = jnp.stack(
                (
                    jnp.clip(radial_step, -radial_limit, radial_limit),
                    jnp.clip(angular_step, -0.25, 0.25),
                )
            )
            candidate = state - step
            candidate_valid = (
                jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(residual))
                & (jnp.abs(determinant) > 128.0 * jnp.finfo(real_dtype).eps)
                & (candidate[0] > 0.0)
            )
            return jnp.where(candidate_valid, candidate, state)

        refined = jax.lax.fori_loop(0, 6, newton_step, initial_state)
        final_residual = equations(refined)
        angular_displacement = jnp.abs(jnp.angle(jnp.exp(1.0j * (refined[1] - initial_angle))))
        converged = (
            jnp.all(jnp.isfinite(final_residual))
            & (jnp.max(jnp.abs(final_residual)) <= 1.0e-7)
            & (angular_displacement <= 0.35)
            & (refined[0] > 0.0)
        )
        angle = jnp.where(converged, refined[1], fallback_angle)
        return (jnp.mod(angle, 2.0 * jnp.pi), converged | is_bracketed, refined[0], converged)

    # Compact extrema before Newton. A vectorized 24-slot solve made every
    # polar point pay for padding and branch endpoints; the sequential CPU
    # chart needs exact tangencies only at true, continuously tracked extrema.
    required = required & active
    n_required = jnp.sum(required, dtype=jnp.int32)
    indices = jnp.nonzero(
        required,
        size=required.size,
        fill_value=0,
    )[0]
    initial_refined = jnp.mod(fallback_angles, 2.0 * jnp.pi)
    initial_valid = ~required

    def refine_selected(compact_index, state):
        refined_angles, valid, radii, convergence = state
        index = indices[compact_index]
        angle, is_valid, radius, converged = refine_one(
            initial_images[index],
            initial_angles[index],
            fallback_angles[index],
            bracketed[index],
        )
        return (
            refined_angles.at[index].set(angle),
            valid.at[index].set(is_valid),
            radii.at[index].set(radius),
            convergence.at[index].set(converged),
        )

    refined, valid, radii, converged = jax.lax.fori_loop(
        jnp.int32(0),
        n_required,
        refine_selected,
        (initial_refined, initial_valid, jnp.abs(initial_images), jnp.zeros_like(active)),
    )
    refined = jax.lax.stop_gradient(refined)
    refined = _implicit_angular_tangency_angles(
        refined,
        jax.lax.stop_gradient(radii),
        converged,
        w_center,
        rho,
        lens,
    )
    # A failed solve returns the fitted trace angle. Freezing that angle
    # drops a real endpoint dependency and disagrees with finite differences.
    refined = jnp.where(converged, refined, initial_refined)
    return refined, valid
