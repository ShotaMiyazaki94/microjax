import jax
import jax.numpy as jnp
import numpy as np

from microjax.inverse_ray.roots.angular import (
    ANGULAR_OK,
    angular_measure_triple_roots,
    evaluate_fourier,
)
from microjax.inverse_ray.roots.level_set import (
    triple_level_set,
    triple_level_set_fourier,
)
from microjax.lens_geometry import triple_lens_geometry
from microjax.point_source import lens_eq


def _setup():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    center_cm = jnp.asarray(-0.1 + 0.2j)
    center_midpoint = center_cm - geometry.shifted
    return params, geometry, center_cm, center_midpoint


def _sampled_fourier_coefficients(radius, center_midpoint, rho, geometry, chart_center):
    """Reproduce the removed 20-point FFT construction as a test oracle."""

    angles = 2.0 * jnp.pi * jnp.arange(20, dtype=jnp.asarray(radius).dtype) / 20
    samples = triple_level_set(
        chart_center + radius * jnp.exp(1j * angles),
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    coefficients = jnp.fft.fft(samples)[:5] / samples.size
    scale = jnp.abs(coefficients[0]) + 2.0 * jnp.sum(jnp.abs(coefficients[1:]))
    return coefficients / scale


def test_triple_denominator_cleared_level_set_matches_lens_equation():
    params, geometry, _, center_midpoint = _setup()
    rho = 0.03
    z_cm = jnp.asarray([0.2 + 0.3j, -0.7 + 0.1j, 0.4 - 0.5j, 1.2 + 0.6j])
    z_midpoint = z_cm - geometry.shifted
    cleared = triple_level_set(
        z_cm,
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    mapped = lens_eq(
        z_midpoint,
        nlenses=3,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3=params["r3"],
        psi=params["psi"],
    )
    rational = jnp.abs(mapped - center_midpoint) ** 2 - rho**2
    zbar = jnp.conjugate(z_midpoint)
    denominator = (zbar - geometry.a) * (zbar + geometry.a) * (zbar - jnp.conjugate(geometry.r3_complex))
    expected = rational * jnp.abs(denominator) ** 2
    assert np.allclose(np.asarray(cleared), np.asarray(expected), rtol=2e-13, atol=2e-15)


def test_triple_degree_four_fourier_level_set_matches_direct_evaluation():
    _, geometry, _, center_midpoint = _setup()
    radius = jnp.asarray(0.8)
    rho = 0.03
    fourier = triple_level_set_fourier(
        radius,
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    angles = jnp.asarray([0.07, 0.43, 1.2, 2.9, 4.8, 6.1])
    direct = triple_level_set(
        radius * jnp.exp(1j * angles),
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    reconstructed = evaluate_fourier(fourier.coefficients, angles)
    raw_scale = jnp.abs(fourier.coefficients[0]) + 2.0 * jnp.sum(jnp.abs(fourier.coefficients[1:]))
    # Coefficients are normalized to unit L1 Fourier scale. Recover the direct
    # samples through a matching sampled value instead of relying on internals.
    reference_angle = jnp.asarray(0.31)
    direct_reference = triple_level_set(
        radius * jnp.exp(1j * reference_angle),
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    reconstructed_reference = evaluate_fourier(fourier.coefficients, reference_angle)
    scale = direct_reference / reconstructed_reference
    assert np.isclose(float(raw_scale), 1.0, rtol=0.0, atol=2e-15)
    assert float(fourier.padding) < 1e-12
    assert not bool(fourier.degenerate)
    assert np.allclose(
        np.asarray(reconstructed * scale),
        np.asarray(direct),
        rtol=2e-12,
        atol=2e-13,
    )


def test_analytic_triple_fourier_coefficients_match_removed_fft_path():
    _, geometry, _, center_midpoint = _setup()
    cases = (
        (0.8, 0.03, 0.0 + 0.0j),
        (0.013, 0.03, 1.09 + 0.046j),
        (1.4, 1.0e-4, -0.3 + 0.7j),
    )
    for radius, rho, chart_center in cases:
        analytic = triple_level_set_fourier(
            radius,
            center_midpoint,
            rho,
            geometry.shifted,
            a=geometry.a,
            e1=geometry.e1,
            e2=geometry.e2,
            r3_complex=geometry.r3_complex,
            chart_center=chart_center,
        )
        sampled = _sampled_fourier_coefficients(radius, center_midpoint, rho, geometry, chart_center)
        assert np.allclose(np.asarray(analytic.coefficients), np.asarray(sampled), rtol=3e-13, atol=3e-14)


def test_triple_fourier_level_set_supports_an_image_local_chart():
    _, geometry, _, center_midpoint = _setup()
    radius = jnp.asarray(0.013)
    chart_center = jnp.asarray(1.09 + 0.046j)
    rho = 0.03
    fourier = triple_level_set_fourier(
        radius,
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
        chart_center=chart_center,
    )
    angles = jnp.asarray([0.07, 0.43, 1.2, 2.9, 4.8, 6.1])
    direct = triple_level_set(
        chart_center + radius * jnp.exp(1j * angles),
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    reconstructed = evaluate_fourier(fourier.coefficients, angles)
    reference_angle = jnp.asarray(0.31)
    direct_reference = triple_level_set(
        chart_center + radius * jnp.exp(1j * reference_angle),
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    scale = direct_reference / evaluate_fourier(fourier.coefficients, reference_angle)
    assert not bool(fourier.degenerate)
    assert np.allclose(np.asarray(reconstructed * scale), np.asarray(direct), rtol=2e-11, atol=2e-13)


def test_triple_degree_eight_roots_match_dense_angular_oracle():
    _, geometry, _, center_midpoint = _setup()
    rho = 0.03
    radius = jnp.asarray(0.8)
    result = angular_measure_triple_roots(
        radius,
        0.0,
        2.0 * jnp.pi,
        center_midpoint,
        rho,
        geometry.shifted,
        1e-12,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 200_001)
    direct = triple_level_set(
        radius * jnp.exp(1j * angles),
        center_midpoint,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    dense_measure = jnp.mean(direct[:-1] <= 0.0) * (2.0 * jnp.pi)
    assert int(result.status) == ANGULAR_OK
    assert float(result.error) < 1e-10
    assert np.isclose(float(result.measure), float(dense_measure), rtol=0.0, atol=4e-5)


def test_analytic_coefficients_preserve_near_tangent_root_pair():
    geometry = triple_lens_geometry(1.1, 0.1, 0.01, abs(0.3 + 1.2j), np.angle(0.3 + 1.2j))
    source = 0.43685353 + 0.67619414j - geometry.shifted
    result = angular_measure_triple_roots(
        1.5123912766442602,
        0.0,
        2.0 * jnp.pi,
        source,
        0.01,
        geometry.shifted,
        1e-12,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    )
    assert int(result.status) == ANGULAR_OK


def test_triple_degree_eight_root_gradient_matches_forward_and_reverse():
    source_cm = jnp.asarray(-0.1 + 0.2j)

    def measure(q3):
        geometry = triple_lens_geometry(0.9, 0.3, q3, 0.4, 0.7)
        result = angular_measure_triple_roots(
            0.8,
            0.0,
            2.0 * jnp.pi,
            source_cm - geometry.shifted,
            0.03,
            geometry.shifted,
            1e-12,
            a=geometry.a,
            e1=geometry.e1,
            e2=geometry.e2,
            r3_complex=geometry.r3_complex,
        )
        return result.measure

    q3 = jnp.asarray(0.2)
    value = measure(q3)
    jvp_value, jvp_tangent = jax.jvp(measure, (q3,), (jnp.ones_like(q3),))
    forward = jax.jacfwd(measure)(q3)
    reverse = jax.grad(measure)(q3)
    step = 1e-4
    finite_difference = (
        measure(q3 - 2.0 * step) - 8.0 * measure(q3 - step) + 8.0 * measure(q3 + step) - measure(q3 + 2.0 * step)
    ) / (12.0 * step)
    assert float(jvp_value) == float(value)
    assert np.isclose(float(jvp_tangent), float(forward), rtol=1e-12, atol=1e-12)
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=1e-10, atol=1e-11)
    assert np.isclose(float(forward), float(finite_difference), rtol=1e-8, atol=1e-10)
