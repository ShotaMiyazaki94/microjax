import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.cartesian_moment import (
    _bernstein_strip_widths,
    _strip_widths,
    binary_line_level_set_coefficients,
    mag_uniform_cartesian_cross_moment,
    mag_uniform_cartesian_moment_fixed,
)
from microjax.inverse_ray.geometry.lens import binary_geometry
from microjax.inverse_ray.roots.level_set import binary_level_set

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.fast


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
