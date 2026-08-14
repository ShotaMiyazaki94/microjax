import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.radial_continuation import (
    _angular_measure,
    _fourier,
    _initial_theta_roots_from_fourier,
    _initial_theta_roots_tangent_from_fourier,
    mag_uniform_radial_continuation_cpu,
)
from microjax.inverse_ray.cpu.polar_atlas import (
    build_polar_branch_atlas,
    polar_atlas_angular_measure,
    refine_polar_atlas_tangencies,
)
from microjax.inverse_ray.cpu.quadrature import G7_W_ON_GL11, GL11_W, GL11_X
from microjax.inverse_ray.cpu.simple_polar_support import build_simple_polar_topology
from microjax.inverse_ray.cpu.refined_limb import trace_binary_source_limb_two_stage
from microjax.inverse_ray.cpu.support import trace_binary_source_limb
from microjax.inverse_ray.cpu.support import _physical_image_mask


def test_gl11_embedded_g7_moments_and_weights():
    assert np.all(GL11_W > 0.0)
    assert np.all(G7_W_ON_GL11 >= 0.0)
    for degree in range(8):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        np.testing.assert_allclose(
            np.sum(G7_W_ON_GL11 * GL11_X**degree),
            expected,
            rtol=0.0,
            atol=5e-15,
        )


def test_two_stage_limb_trace_returns_polished_inserted_roots():
    source = jnp.asarray(0.1 + 0.1j, dtype=jnp.complex128)
    rho = jnp.asarray(0.05, dtype=jnp.float64)
    trace = trace_binary_source_limb_two_stage(
        source, rho, s=0.85, q=0.03, n_coarse=32
    )
    phases = trace.phases
    sources = source + rho * jnp.exp(1.0j * phases)
    recomputed = jax.vmap(
        lambda image, limb_source: _physical_image_mask(
            image, limb_source, 0.85, 0.03
        ),
        in_axes=(1, 0),
    )(trace.image_limb, sources).T
    np.testing.assert_array_equal(np.asarray(trace.physical_mask), np.asarray(recomputed))


def test_parallel_complex_root_continuation_matches_independent_rings():
    source = jnp.asarray(0.1 + 0.1j, dtype=jnp.complex128)
    solve_continued = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            0.05,
            s=0.85,
            q=0.03,
        )
    )
    solve_independent = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            0.05,
            s=0.85,
            q=0.03,
            independent_roots=True,
        )
    )

    continued = solve_continued()
    independent = solve_independent()

    assert int(continued.status) == 0
    assert int(independent.status) == 0
    assert int(continued.n_full_root_solves) < int(independent.n_full_root_solves)
    np.testing.assert_allclose(
        float(continued.magnification),
        float(independent.magnification),
        rtol=2e-11,
        atol=2e-11,
    )


def test_parallel_complex_root_continuation_is_jittable_over_sources():
    sources = jnp.asarray([0.1 + 0.1j, 0.02 - 0.03j], dtype=jnp.complex128)
    solve = jax.jit(
        lambda values: jax.lax.map(
            lambda source: mag_uniform_radial_continuation_cpu(
                source,
                0.05,
                s=0.85,
                q=0.03,
            ),
            values,
        )
    )

    result = solve(sources)

    assert np.all(np.isfinite(np.asarray(result.magnification)))
    assert np.all(np.asarray(result.status) == 0)


def test_polar_branch_atlas_seeds_exact_angular_boundaries():
    source = jnp.asarray(0.1 + 0.1j, dtype=jnp.complex128)
    rho = jnp.asarray(0.05, dtype=jnp.float64)
    s = jnp.asarray(0.85, dtype=jnp.float64)
    q = jnp.asarray(0.03, dtype=jnp.float64)
    image_limb, physical_mask = trace_binary_source_limb(
        source, rho, s=s, q=q, n_limb=64
    )
    atlas = refine_polar_atlas_tangencies(
        build_polar_branch_atlas(image_limb, physical_mask),
        source,
        rho,
        s=s,
        q=q,
    )

    for radius in jnp.asarray([0.77, 0.93, 1.06, 1.08]):
        fourier = _fourier(source, rho, radius, s=s, q=q)
        atlas_measure = polar_atlas_angular_measure(
            atlas,
            fourier.coefficients,
            fourier.padding,
            fourier.degenerate,
            radius,
        )
        independent = _initial_theta_roots_from_fourier(
            fourier.coefficients,
            fourier.padding,
            fourier.degenerate,
            radius,
            use_companion=True,
        )
        assert int(atlas_measure.status) == 0
        assert int(atlas_measure.n_crossings) == 2
        np.testing.assert_allclose(
            float(atlas_measure.measure),
            float(_angular_measure(fourier.coefficients, independent)),
            rtol=0.0,
            atol=3e-13,
        )


def test_rotated_tangent_sextic_matches_complex_unit_circle_solver():
    source = jnp.asarray(0.1 + 0.1j, dtype=jnp.complex128)
    rho = jnp.asarray(0.05, dtype=jnp.float64)
    s = jnp.asarray(0.85, dtype=jnp.float64)
    q = jnp.asarray(0.03, dtype=jnp.float64)

    for radius in jnp.asarray([0.77, 0.93, 1.06, 1.08]):
        fourier = _fourier(source, rho, radius, s=s, q=q)
        complex_roots = _initial_theta_roots_from_fourier(
            fourier.coefficients,
            fourier.padding,
            fourier.degenerate,
            radius,
            use_companion=True,
        )
        tangent_roots = _initial_theta_roots_tangent_from_fourier(
            fourier.coefficients,
            fourier.padding,
            fourier.degenerate,
            radius,
        )
        assert int(complex_roots.status) == 0
        assert int(tangent_roots.status) == 0
        assert int(jnp.sum(complex_roots.active)) == int(
            jnp.sum(tangent_roots.active)
        )
        np.testing.assert_allclose(
            float(_angular_measure(fourier.coefficients, tangent_roots)),
            float(_angular_measure(fourier.coefficients, complex_roots)),
            rtol=0.0,
            atol=3e-12,
        )


def test_polar_branch_atlas_low_q_central_source_is_certified():
    # Saved VBML quick-matrix reference.  The endpoint-transformed radial cells
    # exercise the turning caps that a linear limb projection leaves empty.
    source = jnp.asarray(
        0.0008047670945470802 - 0.0008851167380534286j,
        dtype=jnp.complex128,
    )
    solve = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            3.0e-4,
            s=0.8,
            q=1.0e-6,
            use_polar_atlas=True,
        )
    )
    result = solve()

    assert int(result.status) == 0
    np.testing.assert_allclose(
        float(result.magnification),
        841.9140473239304,
        rtol=3e-9,
        atol=0.0,
    )


def test_polar_branch_atlas_supports_forward_mode_ad():
    parameters = jnp.asarray(
        [
            0.0008047670945470802,
            -0.0008851167380534286,
            3.0e-4,
            0.8,
            1.0e-6,
        ],
        dtype=jnp.float64,
    )

    def value(values):
        result = mag_uniform_radial_continuation_cpu(
            values[0] + 1.0j * values[1],
            values[2],
            s=values[3],
            q=values[4],
            use_polar_atlas=True,
        )
        return result.magnification

    gradient = jax.jit(jax.jacfwd(value))(parameters)

    assert np.all(np.isfinite(np.asarray(gradient)))


def test_simple_polar_support_keeps_filled_annulus_between_root_tracks():
    # The source encloses the primary lens.  Inner and outer annulus limbs live
    # in different algebraic root slots, so a branchwise radial union leaves a
    # false gap around r=1.  The global radial envelope must keep that filled
    # interval.
    source = jnp.asarray(
        -0.0007977780221762626 + 0.0011211518928967016j,
        dtype=jnp.complex128,
    )
    rho = jnp.asarray(3.0e-3, dtype=jnp.float64)
    s = jnp.asarray(0.8, dtype=jnp.float64)
    q = jnp.asarray(1.0e-4, dtype=jnp.float64)
    image_limb, physical_mask = trace_binary_source_limb(
        source, rho, s=s, q=q, n_limb=64
    )
    topology = build_simple_polar_topology(
        image_limb,
        physical_mask,
        source,
        rho,
        s=s,
        q=q,
    )
    intervals = np.asarray(topology.intervals[: topology.n_intervals])

    assert int(topology.status) == 0
    assert np.any((intervals[:, 0] <= 1.0) & (1.0 <= intervals[:, 1]))

    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            rho,
            s=s,
            q=q,
            use_simple_support=True,
        )
    )()
    assert np.isfinite(float(result.magnification))
    np.testing.assert_allclose(
        float(result.magnification),
        629.7387079602801,
        rtol=5.0e-4,
        atol=0.0,
    )


def test_simple_polar_support_rejects_underresolved_close_contacts():
    # A close 2->4->2 angular-root transition lies above the requested 1e-3
    # accuracy here; the fixed embedded-error safety factor must prevent a
    # false certificate.
    source = jnp.asarray(
        0.6002215991122256 - 0.0026938346526468645j,
        dtype=jnp.complex128,
    )
    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            3.0e-3,
            s=1.25,
            q=1.0e-2,
            rtol=1.0e-3,
            use_simple_support=True,
        )
    )()

    assert np.isfinite(float(result.magnification))
    assert int(result.status) != 0


@pytest.mark.fast
def test_simple_polar_support_low_q_central_source_meets_roman_accuracy():
    source = jnp.asarray(
        0.0008047670945470802 - 0.0008851167380534286j,
        dtype=jnp.complex128,
    )
    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            3.0e-4,
            s=0.8,
            q=1.0e-6,
            use_simple_support=True,
        )
    )()

    assert np.isfinite(float(result.magnification))
    np.testing.assert_allclose(
        float(result.magnification),
        841.9140473239304,
        rtol=5.0e-4,
        atol=0.0,
    )


def test_simple_polar_support_keeps_narrow_three_image_radial_support():
    # A non-caustic three-image source can have a radially narrow planetary
    # image whose extrema are not visible as derivative sign changes after the
    # algebraic root slots permute.  Per-branch limb minima/maxima must remain
    # support boundaries even though the final theta roots are solved afresh.
    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            jnp.asarray(
                0.000563603062767322 + 0.013401516702481188j,
                dtype=jnp.complex128,
            ),
            3.0e-4,
            s=1.0,
            q=1.0e-4,
            rtol=1.0e-3,
            use_simple_support=True,
        )
    )()

    assert int(result.status) == 0
    np.testing.assert_allclose(
        float(result.magnification),
        75.0516005525488,
        rtol=2.0e-4,
        atol=0.0,
    )


def test_simple_polar_support_supports_forward_mode_ad():
    parameters = jnp.asarray(
        [
            0.0008047670945470802,
            -0.0008851167380534286,
            3.0e-4,
            0.8,
            1.0e-6,
        ],
        dtype=jnp.float64,
    )

    def value(values):
        return mag_uniform_radial_continuation_cpu(
            values[0] + 1.0j * values[1],
            values[2],
            s=values[3],
            q=values[4],
            use_simple_support=True,
        ).magnification

    gradient = jax.jit(jax.jacfwd(value))(parameters)

    assert np.all(np.isfinite(np.asarray(gradient)))


def test_two_stage_limb_keeps_global_coverage_and_reuses_coarse_points():
    trace = jax.jit(
        lambda: trace_binary_source_limb_two_stage(
            jnp.asarray(0.1 + 0.1j, dtype=jnp.complex128),
            0.05,
            s=0.85,
            q=0.03,
            n_coarse=32,
        )
    )()

    phases = np.asarray(trace.phases)
    coarse = 2.0 * np.pi * np.arange(32) / 32
    np.testing.assert_allclose(phases[::2], coarse, rtol=0.0, atol=2e-15)
    assert np.all(np.diff(phases) > 0.0)
    assert np.all(np.asarray(trace.inserted_fraction) >= 0.2)
    assert np.all(np.asarray(trace.inserted_fraction) <= 0.8)
    assert trace.image_limb.shape == (5, 64)
    assert trace.physical_mask.shape == (5, 64)


def test_two_stage_limb_improves_narrow_planetary_support_without_rescue():
    source = jnp.asarray(
        0.000563603062767322 + 0.013401516702481188j,
        dtype=jnp.complex128,
    )
    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            source,
            3.0e-4,
            s=1.0,
            q=1.0e-4,
            rtol=1.0e-3,
            n_limb=64,
            use_simple_support=True,
            use_refined_limb=True,
        )
    )()

    assert int(result.status) == 0
    assert int(result.n_cells) == 7
    np.testing.assert_allclose(
        float(result.magnification),
        75.0516005525488,
        rtol=2.0e-7,
        atol=0.0,
    )


def test_two_stage_limb_supports_forward_mode_ad():
    parameters = jnp.asarray(
        [0.000563603062767322, 0.013401516702481188, 3.0e-4, 1.0, 1.0e-4],
        dtype=jnp.float64,
    )

    def value(values):
        return mag_uniform_radial_continuation_cpu(
            values[0] + 1.0j * values[1],
            values[2],
            s=values[3],
            q=values[4],
            n_limb=64,
            use_simple_support=True,
            use_refined_limb=True,
        ).magnification

    gradient = jax.jit(jax.jacfwd(value))(parameters)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_two_stage_limb_fails_closed_when_critical_motion_is_unresolved():
    # VBML and the independent accelerator ICRS agree at 4122.217 to 2e-7,
    # while a uniform support trace co-converges 1.84e-3 high.  The second-stage
    # point approaches the nearly tangent central cusp and exposes a local
    # first-order image displacement larger than the image scale.
    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            jnp.asarray(
                -4.050791549847268e-05 - 0.00030033660530534936j,
                dtype=jnp.complex128,
            ),
            3.0e-4,
            s=0.8,
            q=1.0e-6,
            n_limb=64,
            use_simple_support=True,
            use_refined_limb=True,
        )
    )()

    assert np.isfinite(float(result.magnification))
    assert int(result.status) != 0


def test_two_stage_motion_certificate_accepts_complete_buried_caustic_support():
    # A caustic reference point is inside this source, but the source limb is
    # sufficiently far from the critical image that the radial support is
    # resolved.  The general motion certificate must not repeat the old
    # blanket low-q buried-caustic rejection.
    result = jax.jit(
        lambda: mag_uniform_radial_continuation_cpu(
            jnp.asarray(
                9.84050041854608e-05 - 0.0001082302615975933j,
                dtype=jnp.complex128,
            ),
            3.0e-4,
            s=0.8,
            q=1.0e-6,
            n_limb=64,
            use_simple_support=True,
            use_refined_limb=True,
        )
    )()

    assert np.isfinite(float(result.magnification))
    assert int(result.status) == 0
