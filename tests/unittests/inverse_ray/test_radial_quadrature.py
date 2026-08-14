import jax
import jax.numpy as jnp
import numpy as np

from microjax.inverse_ray.geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_OK,
    RADIAL_TOLERANCE,
)
from microjax.inverse_ray.quadrature.radial import (
    RadialIntegrand,
    _refine_cell_uniform,
    adaptive_radial_integral,
    fixed_radial_integral,
)


def _padded_intervals(*active):
    intervals = np.zeros((8, 2), dtype=np.float64)
    intervals[: len(active)] = np.asarray(active)
    return jnp.asarray(intervals)


def test_radial_quadrature_resolves_square_root_endpoints():
    intervals = _padded_intervals((0.0, 1.0))

    def integrand(radius):
        value = jnp.sqrt(jnp.maximum(radius * (1.0 - radius), 0.0))
        return RadialIntegrand(value, jnp.asarray(0.0), jnp.int32(RADIAL_OK))

    result = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-12),
        chunk_size=4,
    )

    assert int(result.status) == RADIAL_OK
    assert np.isclose(float(result.value), np.pi / 8.0, rtol=0.0, atol=1e-12)
    assert float(result.error) <= 1e-12


def test_radial_quadrature_ignores_inactive_slots_and_is_differentiable():
    intervals = _padded_intervals((0.0, 1.0), (100.0, 101.0))

    def integral(scale):
        def integrand(radius):
            return RadialIntegrand(
                scale * radius,
                jnp.asarray(0.0),
                jnp.int32(RADIAL_OK),
            )

        return adaptive_radial_integral(
            integrand,
            intervals,
            jnp.int32(1),
            jnp.asarray(1e-12),
            chunk_size=4,
        ).value

    assert np.isclose(float(integral(2.0)), 1.0, rtol=0.0, atol=1e-13)
    assert np.isclose(float(jax.grad(integral)(2.0)), 0.5, rtol=0.0, atol=1e-13)


def test_fixed_radial_quadrature_carries_interval_parameters():
    intervals = _padded_intervals((0.0, 1.0), (0.0, 1.0))

    def integral(parameters):
        def integrand(radius, scale):
            return RadialIntegrand(
                scale * radius,
                jnp.asarray(0.0),
                jnp.int32(RADIAL_OK),
            )

        return fixed_radial_integral(
            integrand,
            intervals,
            jnp.int32(2),
            jnp.asarray(1e-12),
            chunk_size=4,
            subdivisions=1,
            single_cell_order=47,
            interval_parameters=parameters,
        ).value

    parameters = jnp.asarray([2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isclose(float(integral(parameters)), 2.5, atol=1e-13)
    jacobian = jax.jacfwd(integral)(parameters)
    assert np.allclose(np.asarray(jacobian[:2]), 0.5, atol=1e-13)
    assert np.all(np.asarray(jacobian[2:]) == 0.0)


def test_fixed_g19_fast_rule_integrates_smooth_cell_once():
    intervals = _padded_intervals((0.0, 1.0))

    def integrand(radius):
        return RadialIntegrand(radius**2, jnp.asarray(0.0), jnp.int32(RADIAL_OK))

    result = fixed_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-12),
        chunk_size=4,
        subdivisions=1,
        single_cell_order=19,
    )

    assert int(result.status) == RADIAL_OK
    assert np.isclose(float(result.value), 1.0 / 3.0, atol=1e-13)


def test_radial_quadrature_propagates_topology_status():
    intervals = _padded_intervals((0.0, 1.0))

    def integrand(radius):
        return RadialIntegrand(radius, jnp.asarray(0.0), jnp.int32(RADIAL_OK))

    result = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-12),
        initial_status=jnp.int32(RADIAL_CAPACITY),
        chunk_size=4,
    )

    assert int(result.status) == RADIAL_CAPACITY


def test_inactive_radial_cell_does_not_poison_reverse_mode():
    intervals = jnp.asarray([[1.0, 2.0], [0.0, 0.0]])

    def integral(scale):
        def integrand(radius):
            return RadialIntegrand(
                jnp.sqrt(scale * radius),
                jnp.asarray(0.0),
                jnp.int32(RADIAL_OK),
            )

        return adaptive_radial_integral(
            integrand,
            intervals,
            jnp.int32(1),
            jnp.asarray(1e-8),
            chunk_size=2,
        ).value

    forward = jax.jacfwd(integral)(jnp.asarray(1.0))
    reverse = jax.grad(integral)(jnp.asarray(1.0))
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=0.0, atol=1e-13)


def test_radial_quadrature_accepts_relative_error_budget():
    intervals = _padded_intervals((0.0, 1.0))

    def integrand(radius):
        del radius
        return RadialIntegrand(
            jnp.asarray(10.0),
            jnp.asarray(5e-4),
            jnp.int32(RADIAL_OK),
        )

    absolute_only = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-6),
        chunk_size=4,
    )
    relative = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-6),
        relative_tolerance=jnp.asarray(1e-4),
        chunk_size=4,
    )

    assert int(absolute_only.status) == RADIAL_TOLERANCE
    assert int(relative.status) == RADIAL_OK
    assert float(relative.error) <= 1e-6 + 1e-4 * abs(float(relative.value))


def test_radial_quadrature_refines_beyond_one_bisection():
    intervals = _padded_intervals((0.0, 1.0))

    def integrand(radius):
        return RadialIntegrand(
            jnp.sin(100.0 * radius),
            jnp.asarray(0.0, dtype=radius.dtype),
            jnp.int32(RADIAL_OK),
        )

    two_way = _refine_cell_uniform(
        integrand,
        jnp.asarray(0.0, dtype=intervals.dtype),
        jnp.asarray(1.0, dtype=intervals.dtype),
        2,
    )
    result = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-6, dtype=intervals.dtype),
        chunk_size=4,
    )

    assert float(two_way.error) > 1e-6
    assert int(result.status) == RADIAL_OK
    assert float(result.error) <= 1e-6
    assert np.isclose(
        float(result.value),
        (1.0 - np.cos(100.0)) / 100.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_radial_quadrature_supports_bounded_sixteen_way_retry():
    intervals = _padded_intervals((0.0, 1.0))

    def integrand(radius):
        return RadialIntegrand(
            jnp.sin(160.0 * radius),
            jnp.asarray(0.0, dtype=radius.dtype),
            jnp.int32(RADIAL_OK),
        )

    eight_way = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-6, dtype=intervals.dtype),
        chunk_size=4,
        max_subdivisions=8,
    )
    sixteen_way = adaptive_radial_integral(
        integrand,
        intervals,
        jnp.int32(1),
        jnp.asarray(1e-6, dtype=intervals.dtype),
        chunk_size=4,
        max_subdivisions=16,
    )

    assert int(eight_way.status) == RADIAL_TOLERANCE
    assert int(sixteen_way.status) == RADIAL_OK
    assert float(sixteen_way.error) <= 1e-6
    assert np.isclose(
        float(sixteen_way.value),
        (1.0 - np.cos(160.0)) / 160.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_fixed_radial_quadrature_uses_one_fine_pass_and_masks_padding():
    intervals = _padded_intervals((0.0, 1.0))

    def integral(scale):
        def integrand(radius):
            return RadialIntegrand(
                scale * radius**2,
                jnp.asarray(0.0, dtype=radius.dtype),
                jnp.int32(RADIAL_OK),
            )

        return fixed_radial_integral(
            integrand,
            intervals,
            jnp.int32(1),
            jnp.asarray(1e-12, dtype=intervals.dtype),
            chunk_size=4,
            subdivisions=12,
        )

    result = integral(jnp.asarray(3.0))
    derivative = jax.grad(lambda scale: integral(scale).value)(
        jnp.asarray(3.0)
    )
    assert int(result.status) == RADIAL_OK
    assert np.isclose(float(result.value), 1.0, rtol=0.0, atol=1e-13)
    assert np.isclose(float(derivative), 1.0 / 3.0, rtol=0.0, atol=1e-13)
