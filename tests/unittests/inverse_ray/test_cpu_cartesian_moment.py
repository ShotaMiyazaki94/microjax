import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.cartesian_moment import (
    _binary_line_factored_level,
    _bernstein_strip_widths,
    _certified_companion_intervals,
    _companion_strip_widths,
    _mapped_sign_companion_intervals,
    _strip_widths,
    binary_line_level_set_coefficients,
    mag_uniform_cartesian_cross_moment,
    mag_uniform_cartesian_moment_fixed,
)
from microjax.inverse_ray.cpu.certified_intervals import _merge_intervals
from microjax.inverse_ray.geometry.lens import binary_geometry
from microjax.inverse_ray.roots.level_set import binary_level_set

jax.config.update("jax_enable_x64", True)

def test_line_coefficients_match_factored_level_set():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    offset = jnp.asarray(0.37 - 0.11j, dtype=jnp.complex128)
    direction = jnp.exp(0.73j)
    coefficients = binary_line_level_set_coefficients(
        offset, direction, source, 1e-2, s=1.0, q=0.3
    )
    lens = binary_geometry(1.0, 0.3)
    parameters = jnp.asarray([-0.7, -0.2, 0.0, 0.5, 1.1])
    polynomial = jax.vmap(lambda value: jnp.polyval(coefficients, value))(parameters)
    factored = jax.vmap(
        lambda value: binary_level_set(
            offset + value * direction,
            source - lens.shifted,
            1e-2,
            lens.shifted,
            a=lens.a,
            e1=lens.e1,
        )
    )(parameters)
    ratio = factored / polynomial
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10, atol=1e-12)


def test_cartesian_factored_level_matches_direct_lens_map():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    rho = jnp.asarray(1.0e-2, dtype=jnp.float64)
    axis = jnp.exp(0.37j)
    abscissa = jnp.asarray(0.23)
    ordinate = jnp.asarray([-0.8, -0.1, 0.4, 1.2])
    lens = binary_geometry(1.0, 0.3)
    image = abscissa * axis + ordinate * (1.0j * axis)
    centered = image - lens.shifted
    conjugate = jnp.conjugate(centered)
    mapped = (
        centered
        - lens.e1 / (conjugate - lens.a)
        - (1.0 - lens.e1) / (conjugate + lens.a)
        + lens.shifted
    )
    mapped_radius = jnp.abs(mapped - source)
    direct = (mapped_radius - rho) / (mapped_radius + rho)
    factored = _binary_line_factored_level(
        abscissa,
        ordinate,
        source,
        rho,
        s=jnp.asarray(1.0),
        q=jnp.asarray(0.3),
        axis=axis,
    )
    np.testing.assert_allclose(factored, direct, rtol=2e-13, atol=2e-13)


def test_cartesian_factored_certificate_repairs_ulp_unstable_pair():
    """One source-coordinate ULP must not invent a different image width."""

    source_x = -0.8333103523401397
    source_y = 0.0033066776182536003
    source_y_ulp = np.nextafter(source_y, np.inf)
    sources = (
        jnp.asarray(source_x + 1.0j * source_y, dtype=jnp.complex128),
        jnp.asarray(source_x + 1.0j * source_y_ulp, dtype=jnp.complex128),
    )
    abscissae = (-0.0007507018212049516, -0.0007507018212083754)
    rho = jnp.asarray(3.0e-5, dtype=jnp.float64)
    s = jnp.asarray(2.0 / 3.0, dtype=jnp.float64)
    q = jnp.asarray(1.778279410038923e-6, dtype=jnp.float64)
    axis = jnp.asarray(1.0j, dtype=jnp.complex128)
    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(
        jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a)
    )

    old_widths = []
    certified_widths = []
    for source, abscissa in zip(sources, abscissae):
        x = jnp.asarray([abscissa], dtype=jnp.float64)
        coefficients = jax.vmap(
            lambda value: binary_line_level_set_coefficients(
                value * axis,
                1.0j * axis,
                source,
                rho,
                s=s,
                q=q,
            )
        )(x)
        old, _ = _companion_strip_widths(coefficients)
        radial_offset = jnp.abs(source) + rho - lens_radius
        bound = lens_radius + 0.5 * (
            radial_offset + jnp.sqrt(radial_offset**2 + 4.0)
        )
        lower, upper, active, invalid = _certified_companion_intervals(
            coefficients,
            x,
            source,
            rho,
            s=s,
            q=q,
            axis=axis,
            ordinate_bound=bound,
        )
        old_widths.append(float(old[0]))
        certified_widths.append(
            float(jnp.sum(jnp.where(active, upper - lower, 0.0)))
        )
        assert int(invalid[0]) == 0

    assert not np.isclose(old_widths[0], old_widths[1], rtol=0.1)
    np.testing.assert_allclose(
        certified_widths[0], certified_widths[1], rtol=2e-9, atol=0.0
    )


def test_cartesian_mapped_sign_certificate_rejects_outside_pair():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    rho = jnp.asarray(1.0e-2, dtype=jnp.float64)
    s = jnp.asarray(1.0, dtype=jnp.float64)
    q = jnp.asarray(0.3, dtype=jnp.float64)
    axis = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
    abscissa = jnp.asarray([0.23], dtype=jnp.float64)
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
    lower, upper, active, invalid = _mapped_sign_companion_intervals(
        coefficients,
        abscissa,
        source,
        rho,
        s=s,
        q=q,
        axis=axis,
    )
    midpoint = 0.5 * (lower + upper)
    level = _binary_line_factored_level(
        abscissa[:, None],
        midpoint,
        source,
        rho,
        s=s,
        q=q,
        axis=axis,
    )
    assert bool(jnp.all(jnp.where(active, level < 0.0, True)))
    assert int(invalid[0]) == 0


def test_fixed_shape_interval_merge_ignores_inactive_values_per_batch_row():
    lower = jnp.asarray(
        [
            [-0.8, 10.0, -0.2, 20.0],
            [-1.2, -0.5, 30.0, 40.0],
        ]
    )
    upper = jnp.asarray(
        [
            [-0.4, 11.0, 0.1, 21.0],
            [-0.9, -0.1, 31.0, 41.0],
        ]
    )
    active = jnp.asarray(
        [[True, False, True, False], [True, True, False, False]]
    )
    merged_lower, merged_upper, merged_active, count, overflow = (
        _merge_intervals(lower, upper, active)
    )
    np.testing.assert_allclose(
        merged_lower,
        [[-0.8, -0.2, 0.0], [-1.2, -0.5, 0.0]],
    )
    np.testing.assert_allclose(
        merged_upper,
        [[-0.4, 0.1, 0.0], [-0.9, -0.1, 0.0]],
    )
    np.testing.assert_array_equal(
        merged_active, [[True, True, False], [True, True, False]]
    )
    np.testing.assert_array_equal(count, [2, 2])
    np.testing.assert_array_equal(overflow, [False, False])


@pytest.mark.fast
def test_cartesian_moment_returns_finite_magnification():
    result = jax.jit(
        lambda source: mag_uniform_cartesian_moment_fixed(
            source,
            1e-2,
            s=1.0,
            q=0.3,
            n_slice=8,
            n_limb=128,
            return_info=True,
        )
    )(jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128))
    assert np.isfinite(float(result.magnification))
    assert float(result.magnification) > 1.0


def test_cartesian_moment_is_projection_invariant_at_high_order():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    evaluate = jax.jit(
        lambda axis: mag_uniform_cartesian_moment_fixed(
            source,
            1e-2,
            s=1.0,
            q=0.3,
            n_slice=12,
            n_limb=128,
            axis=axis,
        )
    )
    horizontal = evaluate(jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128))
    vertical = evaluate(jnp.asarray(0.0 + 1.0j, dtype=jnp.complex128))
    np.testing.assert_allclose(horizontal, vertical, rtol=2e-7, atol=1e-9)


def test_orthogonal_cartesian_cross_certifies_regular_source():
    result = jax.jit(
        lambda source: mag_uniform_cartesian_cross_moment(
            source,
            1e-2,
            s=1.0,
            q=0.3,
            rtol=1e-3,
            return_info=True,
        )
    )(jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128))
    assert np.isfinite(float(result.magnification))
    assert int(result.invalid_root_count) == 0
    assert int(result.status) == 0


def test_bernstein_strip_widths_resolve_a_narrow_image_pair():
    roots = np.asarray([-1.2, -0.8, -0.10001, -0.1, 0.4, 1.1])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)
    widths, invalid = jax.jit(
        lambda values: _bernstein_strip_widths(
                values,
                ordinate_bound=jnp.asarray(2.0),
                max_depth=20,
                capacity=12,
        )
    )(coefficients)
    expected = np.sum(roots[1::2] - roots[::2])
    np.testing.assert_allclose(np.asarray(widths), expected, rtol=0.0, atol=2e-8)
    assert int(invalid[0]) == 0


def test_bernstein_lookup_compaction_matches_top_k_compaction():
    roots = np.asarray([-1.2, -0.8, -0.35, -0.1, 0.4, 1.1])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)

    def evaluate(mode, source_radius):
        return _strip_widths(
            coefficients,
            continuation=mode,
            ordinate_bound=jnp.asarray(2.0),
            source_radius=jnp.asarray(source_radius),
        )

    for source_radius in (0.05, 1.0e-4, 1.0e-5):
        reference, reference_invalid = evaluate("bernstein_lean", source_radius)
        lookup, lookup_invalid = evaluate("bernstein_lookup", source_radius)
        np.testing.assert_allclose(
            np.asarray(lookup), np.asarray(reference), rtol=0, atol=1e-13
        )
        np.testing.assert_array_equal(np.asarray(lookup_invalid), reference_invalid)


def test_independent_ea_strip_widths_match_known_negative_intervals():
    roots = np.asarray([-1.2, -0.8, -0.35, -0.1, 0.4, 1.1])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)
    expected = np.sum(roots[1::2] - roots[::2])
    for mode in (
        "ea_independent",
        "ea_scaled",
        "ea_fixed24",
        "ea_fixed26",
        "ea_fixed28",
        "ea_fixed32",
    ):
        widths, invalid = jax.jit(
            lambda values: _strip_widths(
                values,
                continuation=mode,
                ordinate_bound=jnp.asarray(2.0),
            )
        )(coefficients)
        np.testing.assert_allclose(
            np.asarray(widths), expected, rtol=0.0, atol=1e-10
        )
        assert int(invalid[0]) == 0


def test_bernstein_child_clipping_avoids_transient_sextic_overflow():
    """Six root cells must not become a fictitious twelve-cell overflow."""

    roots = np.asarray([-1.3, -0.9, -0.45, -0.1, 0.35, 1.15])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)
    widths, invalid = jax.jit(
        lambda values: _bernstein_strip_widths(
            values,
            ordinate_bound=jnp.asarray(2.0),
            max_depth=12,
            capacity=8,
            clip_children=True,
        )
    )(coefficients)
    expected = np.sum(roots[1::2] - roots[::2])
    np.testing.assert_allclose(np.asarray(widths), expected, rtol=0.0, atol=2e-8)
    assert int(invalid[0]) == 0


def test_root_free_bernstein_cartesian_cross_gradient_matches_vbbl():
    evaluate = jax.jit(
        lambda coordinate: mag_uniform_cartesian_cross_moment(
            coordinate[0] + 1.0j * coordinate[1],
            1e-2,
            s=1.0,
            q=0.3,
            n_limb=192,
            continuation="bernstein_adaptive",
        )
    )
    coordinate = jnp.asarray([0.1, 0.2], dtype=jnp.float64)
    automatic = np.asarray(jax.jacfwd(evaluate)(coordinate))
    # VBBinaryLensing Tol=RelTol=1e-10 with central step 1e-6.
    reference = np.asarray([-0.30967502, -1.92947507])
    np.testing.assert_allclose(automatic, reference, rtol=2e-3, atol=1e-6)
