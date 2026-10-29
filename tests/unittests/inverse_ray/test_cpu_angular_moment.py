import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_moment import (
    ANGULAR_MOMENT_SUPPORT,
    AngularSupport,
    _angular_support_cells,
    _ray_intervals,
    _ray_intervals_from_roots,
    _radial_image_bound,
    _stable_radial_level_value,
    _trace_sector_union_mask,
    _split_angular_support_at_angle,
    _split_overlapping_angular_support,
    _uniform_result_from_support,
    binary_radial_level_set_coefficients,
    mag_uniform_angular_moment_compact,
    mag_uniform_angular_moment_fixed,
    mag_uniform_angular_moment_refined,
)
from microjax.inverse_ray.geometry.lens import binary_geometry
from microjax.inverse_ray.roots.level_set import binary_level_set

jax.config.update("jax_enable_x64", True)

def test_overlapping_support_split_reuses_inactive_cells_without_moving_edges():
    cells = jnp.asarray(
        [[0.0, 1.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=jnp.float64,
    )
    support = AngularSupport(
        cells=cells,
        active=jnp.asarray([True, True, False, False]),
        topology_uncertain=jnp.asarray(False),
        minimum_ghost_residual=jnp.asarray(1.0),
        limb_topology=jnp.asarray(False),
        tangencies_valid=jnp.asarray([True]),
    )
    angles = jnp.asarray([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 1.5]])
    physical_limb = jnp.exp(1.0j * angles)
    physical_mask = jnp.ones_like(angles, dtype=bool)

    split = _split_overlapping_angular_support(
        support, physical_limb, physical_mask, parts=3
    )

    assert int(jnp.sum(split.active)) == 4
    split_edges = np.sort(np.asarray(split.cells[split.active]).reshape(-1))
    np.testing.assert_allclose(split_edges, [0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1, 2])
    np.testing.assert_allclose(
        jnp.sum(split.cells[split.active, 1] - split.cells[split.active, 0]),
        2.0,
    )


def test_trace_sector_union_does_not_sweep_across_inactive_edge():
    cells = jnp.asarray(
        [[0.0, 0.2], [0.2, 0.6], [0.6, 0.8]],
        dtype=jnp.float64,
    )
    angles = jnp.asarray([[0.1, 0.7]], dtype=jnp.float64)
    physical_limb = jnp.exp(1.0j * angles)
    physical_mask = jnp.ones_like(angles, dtype=bool)
    following_limb = jnp.roll(physical_limb, -1, axis=1)
    outgoing_edge_active = jnp.zeros_like(physical_mask)

    active = _trace_sector_union_mask(
        cells,
        physical_limb,
        physical_mask,
        following_limb,
        outgoing_edge_active,
        support_uncertain=jnp.asarray(False),
    )

    np.testing.assert_array_equal(np.asarray(active), [True, False, True])


def test_angular_support_exposes_lossless_support_1d_contract():
    support = AngularSupport(
        cells=jnp.asarray([[0.0, 1.0], [1.0, 2.0], [0.0, 0.0]]),
        active=jnp.asarray([True, True, False]),
        topology_uncertain=jnp.asarray(False),
        minimum_ghost_residual=jnp.asarray(1.0),
        limb_topology=jnp.asarray(False),
        tangencies_valid=jnp.asarray([True, True]),
    )
    contract = support.as_support_1d()
    np.testing.assert_array_equal(np.asarray(contract.cells), np.asarray(support.cells))
    np.testing.assert_array_equal(np.asarray(contract.active), np.asarray(support.active))
    np.testing.assert_array_equal(np.asarray(contract.core), np.asarray(support.active))
    assert contract.payload is support


def test_structural_angle_split_inserts_exact_boundary_once():
    support = AngularSupport(
        cells=jnp.asarray([[0.0, 1.0], [1.0, 2.0], [0.0, 0.0]]),
        active=jnp.asarray([True, True, False]),
        topology_uncertain=jnp.asarray(True),
        minimum_ghost_residual=jnp.asarray(1.0),
        limb_topology=jnp.asarray(False),
        tangencies_valid=jnp.asarray([True]),
    )

    split, applied = _split_angular_support_at_angle(
        support, jnp.asarray(0.25), enabled=True
    )

    assert bool(applied)
    np.testing.assert_allclose(
        np.sort(np.asarray(split.cells[split.active]).reshape(-1)),
        [0.0, 0.25, 0.25, 1.0, 1.0, 2.0],
    )


def test_structural_angle_split_revives_both_halves_of_inactive_guard_cell():
    """A rejected-root angle must not lose half of its conservative sector."""

    support = AngularSupport(
        cells=jnp.asarray(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [0.0, 0.0]],
            dtype=jnp.float64,
        ),
        active=jnp.asarray([True, False, False, False]),
        topology_uncertain=jnp.asarray(True),
        minimum_ghost_residual=jnp.asarray(1.0),
        limb_topology=jnp.asarray(False),
        tangencies_valid=jnp.asarray([True]),
    )

    split, applied = _split_angular_support_at_angle(
        support, jnp.asarray(1.5), enabled=True
    )

    assert bool(applied)
    active_cells = np.asarray(split.cells[split.active])
    np.testing.assert_allclose(
        np.sort(active_cells.reshape(-1)),
        [0.0, 1.0, 1.0, 1.5, 1.5, 2.0],
    )


def test_radial_coefficients_match_factored_level_set():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    theta = jnp.asarray(0.73)
    coefficients = binary_radial_level_set_coefficients(
        theta, source, 1e-2, s=1.0, q=0.3
    )
    lens = binary_geometry(1.0, 0.3)
    radii = jnp.asarray([0.0, 0.2, 0.7, 1.4])
    polynomial = jax.vmap(lambda radius: jnp.polyval(coefficients, radius))(radii)
    factored = jax.vmap(
        lambda radius: binary_level_set(
            radius * jnp.exp(1j * theta),
            source - lens.shifted,
            1e-2,
            lens.shifted,
            a=lens.a,
            e1=lens.e1,
        )
    )(radii)
    ratio = factored / polynomial
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10, atol=1e-12)


def test_fixed_ea_radial_intervals_match_real_companion():
    roots = np.asarray([0.12, 0.31, 0.74, 1.05, 1.42, 1.87])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)
    companion = jax.jit(
        lambda values: _ray_intervals(values, root_mode="companion")
    )(coefficients)
    for mode in ("ea_fixed20", "ea_fixed24", "ea_fixed28"):
        ea = jax.jit(
            lambda values, selected_mode=mode: _ray_intervals(
                values,
                root_mode=selected_mode,
                ordinate_bound=jnp.asarray(2.5),
            )
        )(coefficients)
        for expected, actual in zip(companion, ea):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)
        assert int(ea[3][0]) == 0


def test_fixed_ea_fails_back_to_companion_for_nearly_multiple_positive_roots():
    # At this x64-limit polynomial a tiny residual is insufficient to resolve
    # the two positive roots.  The EA guard must preserve the established
    # companion interval rather than silently changing the occupied area.
    source = jnp.asarray(
        -2.037662902008118e-6 + 1.6520468408939938e-6j,
        dtype=jnp.complex128,
    )
    rho = jnp.asarray(3.1332996978614516e-7, dtype=jnp.float64)
    coefficients = binary_radial_level_set_coefficients(
        0.0017368781264312616,
        source,
        rho,
        s=0.9984852932118772,
        q=2.6711612549712744e-10,
    )[None, :]
    bound = _radial_image_bound(
        source,
        rho,
        s=0.9984852932118772,
        q=2.6711612549712744e-10,
    )
    companion = jax.jit(
        lambda values: _ray_intervals(values, root_mode="companion")
    )(coefficients)
    ea = jax.jit(
        lambda values: _ray_intervals(
            values,
            root_mode="ea_fixed20",
            ordinate_bound=bound,
        )
    )(coefficients)
    for expected, actual in zip(companion[:3], ea[:3]):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)
    assert int(ea[3][0]) > int(companion[3][0])


def test_stable_level_value_rejects_near_cancelled_interval():
    source = jnp.asarray(
        -2.037662902008118e-6 + 1.6520468408939938e-6j,
        dtype=jnp.complex128,
    )
    rho = jnp.asarray(3.1332996978614516e-7, dtype=jnp.float64)
    s = jnp.asarray(0.9984852932118772, dtype=jnp.float64)
    q = jnp.asarray(2.6711612549712744e-10, dtype=jnp.float64)
    angles = jnp.asarray([0.0017368781264312616], dtype=jnp.float64)
    coefficients = jax.vmap(
        lambda angle: binary_radial_level_set_coefficients(
            angle,
            source,
            rho,
            s=s,
            q=q,
        )
    )(angles)
    stable = _ray_intervals(
        coefficients,
        root_mode="companion",
        stable_context=(angles, source, rho, s, q),
    )

    # The squared level-set polynomial is almost completely cancelled near
    # this close root pair. Its final sign depends on the JAX/LAPACK version,
    # while the unsquared level value remains safely positive.
    radius = jnp.asarray([[0.9999979]], dtype=jnp.float64)
    squared_level = jnp.polyval(coefficients[0], radius[0, 0])
    stable_level = _stable_radial_level_value(
        angles,
        radius,
        source,
        rho,
        s=s,
        q=q,
    )
    assert abs(float(squared_level)) < 1.0e-14
    assert float(stable_level[0, 0]) > 1.0e-10
    assert not bool(jnp.any(stable[2]))
    assert int(stable[3][0]) == 0


def test_stable_level_does_not_create_area_from_unresolved_complex_pair(monkeypatch):
    # Supply the root classification rather than depending on LAPACK to put
    # a nearly multiple pair on a particular side of the real-root threshold.
    # The tiny negative mapped level must not manufacture a whole [0, 1] cell
    # when no positive roots were accepted and the polynomial is cancelling.
    from microjax.inverse_ray.cpu import angular_moment

    candidates = np.asarray([0.5 + 4e-7j, 0.5 - 4e-7j, -1.0, -2.0, -3.0, -4.0])
    polynomial = np.poly(candidates).real
    polynomial /= np.max(np.abs(polynomial))
    assert abs(np.polyval(polynomial, 0.5)) < 1e-12
    coefficients = jnp.asarray(polynomial[None, :], dtype=jnp.float64)
    roots = jnp.asarray(candidates[None, :], dtype=jnp.complex128)
    monkeypatch.setattr(
        angular_moment,
        "_stable_radial_level_value",
        lambda _angles, points, _source, _rho, **_lens: jnp.full_like(points, -jnp.finfo(points.dtype).eps),
    )
    lower, upper, inside, invalid = _ray_intervals_from_roots(
        coefficients,
        roots,
        stable_context=(jnp.asarray([0.0]), jnp.asarray(0.0j), jnp.asarray(1.0), 1.0, 0.1),
    )
    assert float(jnp.sum(jnp.where(inside, upper - lower, 0.0))) == 0.0
    assert float(lower[0, 0]) == 0.0
    assert float(upper[0, 0]) == 1.0
    assert int(invalid[0]) == 0


def test_marginal_companion_ray_does_not_manufacture_a_broad_interval():
    # This physical ray has a very narrow occupied interval. Different
    # eigensolvers may resolve its candidates as real or as a conjugate pair;
    # neither classification may turn the entire radial domain into image area.
    source = jnp.asarray(
        -1.4999573888723472 + 0.010965537139530405j,
        dtype=jnp.complex128,
    )
    rho = jnp.asarray(3.0e-5, dtype=jnp.float64)
    angle = jnp.asarray(6.279239655504644, dtype=jnp.float64)
    coefficients = binary_radial_level_set_coefficients(
        angle, source, rho, s=0.5, q=1.0e-5
    )[None, :]
    lower, upper, inside, invalid = _ray_intervals(
        coefficients,
        root_mode="companion",
        stable_context=(jnp.asarray([angle]), source, rho, 0.5, 1.0e-5),
    )
    # Independent 90-decimal solutions of the unexpanded lens equation, with
    # the binary64 inputs above treated as exact. This checks containment, not
    # completeness of the companion solver's nearly multiple root candidates.
    physical_lower = 0.49998892961983111624
    physical_upper = 0.50000614649415968412
    active = np.asarray(inside & (upper > lower))
    assert np.all(np.asarray(lower)[active] >= physical_lower - 1e-12)
    assert np.all(np.asarray(upper)[active] <= physical_upper + 1e-12)
    assert int(invalid[0]) == 0


def test_fixed_ea_polar_area_matches_companion():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    support = _angular_support_cells(source, 1e-2, s=1.0, q=0.3, n_limb=64)
    common = dict(
        w_center=source,
        rho=jnp.asarray(1e-2),
        s=jnp.asarray(1.0),
        q=jnp.asarray(0.3),
        n_theta=16,
        cells=support.cells,
        active=support.active,
        topology_uncertain=support.topology_uncertain,
        minimum_ghost_residual=support.minimum_ghost_residual,
        limb_topology=support.limb_topology,
        tangencies_valid=support.tangencies_valid,
    )
    companion = _uniform_result_from_support(**common, root_mode="companion")
    ea = _uniform_result_from_support(**common, root_mode="ea_fixed28")
    np.testing.assert_allclose(
        float(ea.magnification),
        float(companion.magnification),
        rtol=2e-10,
        atol=2e-12,
    )
    assert int(ea.invalid_root_count) == int(companion.invalid_root_count)


@pytest.mark.fast
def test_angular_moment_returns_finite_magnification():
    value = jax.jit(
        lambda source: mag_uniform_angular_moment_fixed(
            source, 1e-2, s=1.0, q=0.3, n_theta=16
        )
    )(jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128))
    assert np.isfinite(float(value))
    assert float(value) > 1.0


def test_refined_angular_moment_resolves_buried_planetary_caustic():
    result = mag_uniform_angular_moment_refined(
        -0.0010946356833467202 - 0.006172611801159194j,
        0.00988422630886768,
        s=0.5992640188539046,
        q=0.0005041898087262488,
        rtol=1e-3,
        return_info=True,
    )
    assert int(result.status) == 0
    assert np.isclose(float(result.magnification), 180.0580995708625, rtol=1e-4)


def test_compact_polar_chart_fails_closed_when_tangencies_are_unverified():
    source = jnp.asarray(-0.005297926262718188 + 0.00023223841465465215j)
    rho = jnp.asarray(0.0036652167671266146)
    support = _angular_support_cells(
        source,
        rho,
        s=0.6753608953491668,
        q=0.007935217541601874,
        n_limb=64,
    )
    unverified = support._replace(
        tangencies_valid=jnp.zeros_like(support.tangencies_valid)
    )
    result = mag_uniform_angular_moment_compact(
        source,
        rho,
        s=0.6753608953491668,
        q=0.007935217541601874,
        rtol=1e-3,
        _support=unverified,
        return_info=True,
    )
    assert int(result.status) & ANGULAR_MOMENT_SUPPORT


def test_polar_support_uses_an_isolated_root_branch_as_a_tangency_seed():
    source = jnp.asarray(0.3667626068246604 - 0.03417414721542691j)
    rho = jnp.asarray(0.001)
    support = _angular_support_cells(source, rho, s=1.2, q=0.001, n_limb=64)
    result = mag_uniform_angular_moment_refined(
        source,
        rho,
        s=1.2,
        q=0.001,
        rtol=1e-3,
        _support=support,
        return_info=True,
    )
    assert int(result.status) == 0
    assert np.isclose(float(result.magnification), 6.577297831349562, rtol=1e-4)


def test_trace_sector_union_skips_only_empty_atomic_cells():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    rho = jnp.asarray(1.0e-2)
    support = _angular_support_cells(source, rho, s=1.0, q=0.3, n_limb=64)
    partition_active = support.cells[:, 1] > support.cells[:, 0]

    common = dict(
        w_center=source,
        rho=rho,
        s=jnp.asarray(1.0),
        q=jnp.asarray(0.3),
        n_theta=32,
        cells=support.cells,
        topology_uncertain=support.topology_uncertain,
        minimum_ghost_residual=support.minimum_ghost_residual,
        limb_topology=support.limb_topology,
        tangencies_valid=support.tangencies_valid,
    )
    union = _uniform_result_from_support(active=support.active, **common)
    full_circle = _uniform_result_from_support(active=partition_active, **common)

    assert int(jnp.sum(support.active)) < int(jnp.sum(partition_active))
    assert int(union.n_theta) < int(full_circle.n_theta)
    np.testing.assert_allclose(
        union.magnification,
        full_circle.magnification,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_trace_sector_union_keeps_full_partition_for_trace_proven_guard():
    support = _angular_support_cells(
        jnp.asarray(0.3667626068246604 - 0.03417414721542691j),
        jnp.asarray(1.0e-3),
        s=1.2,
        q=1.0e-3,
        n_limb=64,
    )
    partition_active = support.cells[:, 1] > support.cells[:, 0]

    assert bool(support.topology_uncertain)
    # This short-lived fold is exactly the trace-proven crossing case in which
    # a physical-limb-only sweep can omit the newborn image sector.  The
    # conservative guard therefore retains the complete 1-D partition.
    np.testing.assert_array_equal(support.active, partition_active)
