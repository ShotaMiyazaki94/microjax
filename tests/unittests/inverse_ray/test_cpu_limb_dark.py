import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_limb_dark import (
    mag_limb_dark_angular_moment_refined,
)
from microjax.inverse_ray.cpu.cartesian_limb_dark import (
    _bernstein_negative_intervals,
    _mag_limb_dark_cartesian_impl,
    _strip_profile_moments,
    mag_limb_dark_cartesian_adaptive,
)
from microjax.inverse_ray.cpu.cartesian_moment import (
    binary_line_level_set_coefficients,
)
from microjax.inverse_ray.cpu.limb_dark import (
    _mag_limb_dark_cpu_fixed_impl,
    mag_limb_dark_cpu,
    mag_limb_dark_cpu_fixed,
)
from microjax.inverse_ray.cpu.one_shot import (
    mag_limb_dark_cpu_one_shot,
    mag_uniform_cpu_one_shot,
)
from microjax.inverse_ray.cpu.support import (
    polar_occupancy_atlas_from_limb,
    polar_product_regions_from_limb,
    radial_support_from_limb,
    trace_binary_source_limb,
)
from microjax.inverse_ray.cpu.uniform import mag_uniform_cpu_fixed
from microjax.inverse_ray.integrators.limb_dark import mag_limb_dark_boundary


@jax.jit
def _limb_dark_fixed_case(source, rho, separation, mass_ratio, u1):
    return mag_limb_dark_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        u1=u1,
        n_limb=128,
        radial_splits=4,
        return_info=True,
    )


@jax.jit
def _limb_dark_zero_case(source, rho, separation, mass_ratio):
    return mag_limb_dark_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        u1=0.0,
        n_limb=32,
        radial_splits=1,
    )


@jax.jit
def _uniform_zero_case(source, rho, separation, mass_ratio):
    return mag_uniform_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        n_limb=32,
        radial_splits=1,
    )


@jax.jit
def _limb_dark_boundary_case(source, rho, separation, mass_ratio, u1):
    return mag_limb_dark_boundary(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        u1=u1,
        Nlimb=64,
        angular_atol=1e-7,
        relative_tolerance=1e-6,
        robust_roots=True,
        certify_topology=True,
        return_info=True,
    )


def test_cpu_zero_limb_darkening_reduces_to_uniform():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    limb = _limb_dark_zero_case(source, 1e-2, 1.0, 0.3)
    uniform = _uniform_zero_case(source, 1e-2, 1.0, 0.3)
    assert np.isclose(float(limb), float(uniform), rtol=1e-12, atol=1e-12)


def test_cartesian_limb_dark_integrates_noncontiguous_active_cells():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    cells = jnp.asarray(
        [[-2.0, -0.5], [-0.5, 0.5], [0.5, 2.0]],
        dtype=jnp.float64,
    )
    hole_active = jnp.asarray([True, False, True])
    packed_cells = jnp.asarray(
        [[-2.0, -0.5], [0.5, 2.0], [0.0, 0.0]],
        dtype=jnp.float64,
    )
    packed_active = jnp.asarray([True, True, False])
    diagnostics = (
        jnp.asarray(False),
        jnp.asarray(1.0),
        jnp.asarray(False),
    )
    common = dict(
        w_center=source,
        rho=jnp.asarray(0.05),
        s=jnp.asarray(1.0),
        q=jnp.asarray(0.3),
        u1=jnp.asarray(0.5),
        n_slice=4,
        n_profile=4,
        n_limb=8,
        axis=jnp.asarray(1.0 + 0.0j),
        root_mode="companion",
        return_info=True,
    )

    with_hole = _mag_limb_dark_cartesian_impl(
        support=(cells, hole_active, *diagnostics),
        **common,
    )
    packed = _mag_limb_dark_cartesian_impl(
        support=(packed_cells, packed_active, *diagnostics),
        **common,
    )

    np.testing.assert_allclose(
        np.asarray(with_hole.magnification),
        np.asarray(packed.magnification),
        rtol=0.0,
        atol=0.0,
    )
    assert int(with_hole.invalid_root_count) == int(packed.invalid_root_count)


def test_radial_support_uses_explicit_cut_edges_as_segment_endpoints():
    radii = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    image_limb = jnp.zeros((5, 4), dtype=jnp.complex128).at[0].set(radii)
    physical_mask = jnp.zeros((5, 4), dtype=bool).at[0].set(True)
    previous = jnp.roll(image_limb, 1, axis=1)
    following = jnp.roll(image_limb, -1, axis=1)
    previous_mask = jnp.roll(physical_mask, 1, axis=1)
    following_mask = jnp.roll(physical_mask, -1, axis=1)
    incoming = physical_mask & previous_mask
    outgoing = physical_mask & following_mask
    outgoing = outgoing.at[0, 1].set(False)
    incoming = incoming.at[0, 2].set(False)

    support = radial_support_from_limb(
        image_limb,
        physical_mask,
        neighbors=(previous, following, previous_mask, following_mask),
        edge_activity=(incoming, outgoing),
        split_turning_points=True,
    )

    finite_boundaries = np.asarray(support.intervals)[
        np.asarray(support.active)
    ].reshape(-1)
    assert np.any(finite_boundaries == 2.0)
    assert np.any(finite_boundaries == 3.0)


def test_polar_occupancy_atlas_covers_each_tracked_limb_branch():
    source = jnp.asarray(0.0114913320114742 + 4.643338212823392e-6j)
    image_limb, image_mask = trace_binary_source_limb(
        source,
        3.0e-5,
        s=1.0,
        q=1.0e-6,
        n_limb=64,
    )
    atlas = polar_occupancy_atlas_from_limb(image_limb, image_mask)
    radius = jnp.abs(image_limb)
    angle = jnp.mod(jnp.angle(image_limb), 2.0 * jnp.pi)
    angular_span = atlas.angular_bounds[:, 1] - atlas.angular_bounds[:, 0]
    angular_offset = jnp.mod(
        angle - atlas.angular_bounds[:, 0, None],
        2.0 * jnp.pi,
    )
    covered = (
        (radius >= atlas.radial_bounds[:, 0, None])
        & (radius <= atlas.radial_bounds[:, 1, None])
        & (angular_offset <= angular_span[:, None])
    )
    np.testing.assert_array_equal(
        np.asarray(atlas.active),
        np.asarray(jnp.any(image_mask, axis=1)),
    )
    assert bool(jnp.all(jnp.where(image_mask, covered, True)))


def test_optional_polar_occupancy_filter_preserves_radial_first_value():
    source = jnp.asarray(0.0114913320114742 + 4.643338212823392e-6j)
    rho = jnp.asarray(3.0e-5)
    image_limb, image_mask = trace_binary_source_limb(
        source,
        rho,
        s=1.0,
        q=1.0e-6,
        n_limb=64,
    )
    support = radial_support_from_limb(
        image_limb,
        image_mask,
        split_turning_points=True,
    )
    atlas = polar_occupancy_atlas_from_limb(image_limb, image_mask)
    common = dict(
        s=1.0,
        q=1.0e-6,
        u1=0.5,
        n_limb=64,
        radial_splits=1,
        include_ghost_support=False,
        support=support,
    )
    baseline, _ = _mag_limb_dark_cpu_fixed_impl(source, rho, **common)
    filtered, _ = _mag_limb_dark_cpu_fixed_impl(
        source,
        rho,
        occupancy_atlas=atlas,
        **common,
    )
    np.testing.assert_allclose(
        np.asarray(filtered.magnification),
        np.asarray(baseline.magnification),
        rtol=1.0e-6,
    )


def test_polar_product_regions_are_disjoint_and_cover_traced_limb():
    source = jnp.asarray(0.0114913320114742 + 4.643338212823392e-6j)
    image_limb, image_mask = trace_binary_source_limb(
        source,
        3.0e-5,
        s=1.0,
        q=1.0e-6,
        n_limb=64,
    )
    regions = polar_product_regions_from_limb(image_limb, image_mask)
    winding_regions = polar_product_regions_from_limb(
        image_limb,
        image_mask,
        include_enclosed=True,
    )
    partitioned_regions = polar_product_regions_from_limb(
        image_limb,
        image_mask,
        capacity=64,
        include_enclosed=True,
        partition_supports=True,
    )
    active = np.asarray(regions.active)
    radial = np.asarray(regions.radial_bounds)[active]
    angular = np.asarray(regions.angular_bounds)[active]

    assert not bool(regions.overflow)
    assert not bool(winding_regions.overflow)
    assert not bool(partitioned_regions.overflow)
    assert int(jnp.sum(winding_regions.active)) >= int(jnp.sum(regions.active))
    assert radial.shape[0] > 0
    radial_overlap = np.minimum(radial[:, None, 1], radial[None, :, 1]) - np.maximum(
        radial[:, None, 0], radial[None, :, 0]
    )
    angular_overlap = np.minimum(
        angular[:, None, 1], angular[None, :, 1]
    ) - np.maximum(angular[:, None, 0], angular[None, :, 0])
    positive_area_overlap = (radial_overlap > 0.0) & (angular_overlap > 0.0)
    np.fill_diagonal(positive_area_overlap, False)
    assert not np.any(positive_area_overlap)

    radius = np.abs(np.asarray(image_limb)).reshape(-1)
    angle = np.mod(np.angle(np.asarray(image_limb)), 2.0 * np.pi).reshape(-1)
    valid = np.asarray(image_mask).reshape(-1)
    covered = np.any(
        (radius[None, :] >= radial[:, 0, None])
        & (radius[None, :] <= radial[:, 1, None])
        & (angle[None, :] >= angular[:, 0, None])
        & (angle[None, :] <= angular[:, 1, None]),
        axis=0,
    )
    assert np.all(covered[valid])


@pytest.mark.slow
def test_one_shot_geometry_route_is_shared_by_brightness_profiles():
    sources = jnp.asarray([0.05 + 0.02j], dtype=jnp.complex128)
    common = dict(rho=5.0e-3, s=1.0, q=1.0e-3, return_state=True)
    solve_uniform = jax.jit(
        jax.vmap(lambda source: mag_uniform_cpu_one_shot(source, **common))
    )
    solve_limb_dark = jax.jit(
        jax.vmap(
            lambda source: mag_limb_dark_cpu_one_shot(
                source,
                u1=0.5,
                **common,
            )
        )
    )
    (uniform, uniform_state) = solve_uniform(sources)
    (limb_dark, limb_dark_state) = solve_limb_dark(sources)
    np.testing.assert_array_equal(
        np.asarray(uniform.stage),
        np.asarray(limb_dark.stage),
    )
    for uniform_value, limb_dark_value in zip(uniform_state, limb_dark_state):
        np.testing.assert_allclose(
            np.asarray(uniform_value),
            np.asarray(limb_dark_value),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "w,rho,s,q,u1",
    [
        (0.1 + 0.2j, 1e-2, 1.0, 0.3, 0.5),
        (-0.35 + 0.17j, 3e-3, 0.7, 1e-2, 0.7),
        (0.6 - 0.2j, 5e-3, 1.6, 0.1, 0.3),
    ],
)
def test_cpu_limb_dark_fixed_matches_existing_boundary(w, rho, s, q, u1):
    w = jnp.asarray(w, dtype=jnp.complex128)
    actual = _limb_dark_fixed_case(w, rho, s, q, u1)
    expected = _limb_dark_boundary_case(w, rho, s, q, u1)
    assert int(actual.status) == 0
    assert np.isclose(
        float(actual.magnification),
        float(expected.magnification),
        rtol=2e-5,
        atol=1e-8,
    )


@pytest.mark.slow
def test_cpu_limb_dark_hierarchy_supports_jit():
    solve = jax.jit(
        lambda source: mag_limb_dark_cpu(
            source,
            1e-2,
            s=1.0,
            q=0.3,
            u1=0.5,
            rtol=1e-4,
        )
    )
    value = solve(jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128))
    assert np.isfinite(float(value))


@pytest.mark.slow
def test_angular_profile_moment_resolves_caustic_residual():
    result = mag_limb_dark_angular_moment_refined(
        -0.10794172585680531 - 0.1079417258568053j,
        0.005,
        s=0.85,
        q=0.03,
        u1=0.5,
        rtol=1e-3,
        return_info=True,
    )
    assert int(result.status) == 0
    assert np.isclose(float(result.magnification), 11.216178805288225, rtol=1e-5)


def test_bernstein_cartesian_profile_matches_companion_intervals():
    source = jnp.asarray(-0.05 * jnp.exp(0.25j * jnp.pi), dtype=jnp.complex128)

    def solve(root_mode):
        return jax.jit(
            lambda value: _mag_limb_dark_cartesian_impl(
                value,
                0.005,
                s=0.85,
                q=0.03,
                u1=0.5,
                n_slice=8,
                n_profile=8,
                n_limb=64,
                axis=1.0j * value / jnp.abs(value),
                root_mode=root_mode,
                return_info=True,
            )
        )(source)

    companion = solve("companion")
    bernstein = solve("bernstein")
    assert int(bernstein.invalid_root_count) == 0
    assert int(bernstein.status) == 0
    assert np.isclose(
        float(bernstein.magnification),
        float(companion.magnification),
        rtol=2.0e-5,
    )


def test_limb_dark_certified_strip_matches_physical_boundary_width():
    source = jnp.asarray(
        -0.8333103523401397 + 0.0033066776182536003j,
        dtype=jnp.complex128,
    )
    abscissa = jnp.asarray([-0.0007507018212049516], dtype=jnp.float64)
    rho = jnp.asarray(3.0e-5, dtype=jnp.float64)
    s = jnp.asarray(2.0 / 3.0, dtype=jnp.float64)
    q = jnp.asarray(1.778279410038923e-6, dtype=jnp.float64)
    axis = jnp.asarray(1.0j, dtype=jnp.complex128)
    coefficients = jax.vmap(
        lambda value: binary_line_level_set_coefficients(
            value * axis,
            1.0j * axis,
            source,
            rho,
            s=s,
            q=q,
        )
    )(abscissa)
    uniform, profile, invalid = _strip_profile_moments(
        abscissa,
        coefficients,
        source,
        rho,
        s=s,
        q=q,
        axis=axis,
        n_profile=4,
        root_mode="companion_certified",
        ordinate_bound=jnp.asarray(2.0, dtype=jnp.float64),
    )
    # Independent 90-decimal roots of the unexpanded lens equation are
    # -0.66674883644788201465798 and -0.66660563818295982752967, treating the
    # binary64 inputs as exact. The former reference was one platform's
    # companion result. This ill-conditioned thin strip loses several digits
    # in binary64; test its physical width to 1e-8, not bitwise solver parity.
    np.testing.assert_allclose(uniform, [0.00014319826492218712831], rtol=1e-8, atol=0)
    assert np.all(np.isfinite(np.asarray(profile)))
    assert np.all(np.asarray(profile) >= 0)
    assert np.all(np.asarray(profile) <= np.asarray(uniform))
    assert int(invalid[0]) == 0


def test_bernstein_isolation_rejects_unresolved_close_root_pair():
    roots = np.asarray([-0.8, -0.4, 0.125, 0.125001, 0.5, 0.9])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)
    _, _, _, invalid = jax.jit(
        lambda values: _bernstein_negative_intervals(
            values,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
    )(coefficients)
    assert int(invalid[0]) > 0


@pytest.mark.slow
def test_cartesian_limb_dark_uses_lazy_companion_refinement():
    source = jnp.asarray(
        -0.01167894083040845 - 0.011678940830408447j,
        dtype=jnp.complex128,
    )
    solve = jax.jit(
        lambda value: mag_limb_dark_cartesian_adaptive(
            value,
            0.005,
            s=0.85,
            q=0.03,
            u1=0.5,
            rtol=1.0e-3,
            return_info=True,
        )
    )
    result = solve(source)
    assert int(result.stage) == 2
    assert int(result.status) == 0
    assert np.isclose(float(result.magnification), 80.5470205519234, rtol=2.0e-4)


@pytest.mark.slow
def test_cartesian_limb_dark_external_cross_certificate_is_fail_closed():
    source = jnp.asarray(-0.05 * jnp.exp(0.25j * jnp.pi), dtype=jnp.complex128)

    def solve(external):
        return jax.jit(
            lambda value, estimate: mag_limb_dark_cartesian_adaptive(
                value,
                0.005,
                s=0.85,
                q=0.03,
                u1=0.5,
                rtol=1.0e-3,
                external_magnification=estimate,
                return_info=True,
            )
        )(source, external)

    baseline = mag_limb_dark_cartesian_adaptive(
        source,
        0.005,
        s=0.85,
        q=0.03,
        u1=0.5,
        rtol=1.0e-3,
        return_info=True,
    )
    accepted = solve(baseline.magnification)
    rejected = solve(2.0 * baseline.magnification)
    assert int(accepted.status) == 0
    assert int(accepted.n_slices) < int(baseline.n_slices)
    accepted_relative_difference = abs(
        float(accepted.magnification - baseline.magnification)
    ) / max(abs(float(baseline.magnification)), 1.0)
    assert accepted_relative_difference <= 0.05 * 1.0e-3
    assert int(rejected.status) == int(baseline.status)
    assert int(rejected.n_slices) == int(baseline.n_slices)
    rejected_relative_difference = abs(
        float(rejected.magnification - baseline.magnification)
    ) / max(abs(float(baseline.magnification)), 1.0)
    assert rejected_relative_difference <= 0.10 * 1.0e-3
    assert not np.isclose(
        float(rejected.magnification),
        float(2.0 * baseline.magnification),
    )
