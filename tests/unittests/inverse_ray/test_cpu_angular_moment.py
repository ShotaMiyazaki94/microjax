import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_moment import (
    ANGULAR_MOMENT_SUPPORT,
    AngularSupport,
    _angular_support_cells,
    _ray_intervals,
    _radial_image_bound,
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


def test_stable_level_value_rejects_false_close_companion_interval():
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
    ordinary = _ray_intervals(coefficients, root_mode="companion")
    stable = _ray_intervals(
        coefficients,
        root_mode="companion",
        stable_context=(angles, source, rho, s, q),
    )
    assert bool(jnp.any(ordinary[2]))
    assert not bool(jnp.any(stable[2]))
    assert int(stable[3][0]) == 0


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


def test_endpoint_preserving_union_skips_only_empty_atomic_cells():
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


def test_endpoint_preserving_union_keeps_all_cells_when_support_is_uncertain():
    support = _angular_support_cells(
        jnp.asarray(0.3667626068246604 - 0.03417414721542691j),
        jnp.asarray(1.0e-3),
        s=1.2,
        q=1.0e-3,
        n_limb=64,
    )
    partition_active = support.cells[:, 1] > support.cells[:, 0]

    assert bool(support.topology_uncertain)
    np.testing.assert_array_equal(support.active, partition_active)
