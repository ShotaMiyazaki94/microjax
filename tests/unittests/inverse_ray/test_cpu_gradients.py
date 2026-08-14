import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray import mag_binary
from microjax.inverse_ray.cpu.angular_moment import (
    mag_uniform_angular_moment_refined,
)
from microjax.inverse_ray.cpu.cartesian_limb_dark import (
    _mag_limb_dark_cartesian_impl,
)
from microjax.inverse_ray.cpu.limb_dark import mag_limb_dark_cpu
from microjax.inverse_ray.cpu.uniform import mag_uniform_cpu, mag_uniform_cpu_fixed


def assert_grad_matches_finite_difference(function, point, steps, rtol=2e-5):
    function = jax.jit(function)
    point = jnp.asarray(point, dtype=jnp.float64)
    steps = np.asarray(steps)
    automatic = np.asarray(jax.jit(jax.jacfwd(function))(point))
    basis = jnp.eye(point.size, dtype=point.dtype)
    finite = np.asarray(
        [
            (
                float(function(point + basis[index] * step))
                - float(function(point - basis[index] * step))
            )
            / (2.0 * step)
            for index, step in enumerate(steps)
        ]
    )
    assert np.all(np.isfinite(automatic))
    np.testing.assert_allclose(automatic, finite, rtol=rtol, atol=1e-6)


@pytest.mark.fast
def test_uniform_fixed_gradients_match_finite_difference():
    def magnification(parameters):
        return mag_uniform_cpu_fixed(
            parameters[0] + 1.0j * parameters[1],
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            n_limb=32,
            radial_splits=1,
        )

    assert_grad_matches_finite_difference(
        magnification,
        [0.1, 0.2, 1e-2, 1.0, 0.3],
        [1e-5, 1e-5, 1e-6, 1e-5, 1e-5],
    )


@pytest.mark.slow
def test_uniform_adaptive_gradients_match_finite_difference():
    def magnification(parameters):
        return mag_uniform_cpu(
            parameters[0] + 1.0j * parameters[1],
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            rtol=1e-3,
        )

    assert_grad_matches_finite_difference(
        magnification,
        [0.1, 0.2, 1e-2, 1.0, 0.3],
        [1e-5, 1e-5, 1e-6, 1e-5, 1e-5],
    )


@pytest.mark.slow
def test_uniform_caustic_topology_gradient_matches_finite_difference():
    source_imag = 0.7403391538113067

    def magnification(source_real):
        return mag_uniform_cpu(
            source_real[0] + 1.0j * source_imag,
            0.003,
            s=0.8,
            q=0.3,
            rtol=1e-3,
        )

    assert_grad_matches_finite_difference(
        magnification,
        [-0.1845738496739989],
        [1e-6],
    )


@pytest.mark.slow
def test_limb_dark_adaptive_gradients_match_finite_difference():
    def magnification(parameters):
        return mag_limb_dark_cpu(
            parameters[0] + 1.0j * parameters[1],
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            u1=parameters[5],
            rtol=1e-3,
        )

    assert_grad_matches_finite_difference(
        magnification,
        [0.1, 0.2, 1e-2, 1.0, 0.3, 0.5],
        [1e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-5],
    )


@pytest.mark.slow
def test_cpu_lightcurve_source_gradients_match_finite_difference():
    source_imag = jnp.asarray([0.18, 0.22], dtype=jnp.float64)

    def summed_magnification(source_real):
        result = mag_binary(
            source_real + 1.0j * source_imag,
            1e-2,
            s=1.0,
            q=0.3,
            backend="cpu",
            return_info=True,
        )
        return jnp.sum(result.magnification)

    assert_grad_matches_finite_difference(
        summed_magnification,
        [0.08, 0.12],
        [1e-5, 1e-5],
    )


@pytest.mark.slow
def test_public_small_source_gradient_matches_vbbl_finite_difference():
    def magnification(source_coordinates):
        source = source_coordinates[0] + 1.0j * source_coordinates[1]
        result = mag_binary(
            jnp.asarray([source]),
            1.0e-4,
            s=0.85,
            q=0.03,
            backend="cpu",
            return_info=True,
        )
        return result.magnification[0]

    point = jnp.asarray(
        [-0.20349669628741984, -0.20349669628741981], dtype=jnp.float64
    )
    automatic = np.asarray(jax.jit(jax.jacfwd(magnification))(point))
    # VBBinaryLensing Tol=RelTol=1e-10, central step 1e-6.
    reference = np.asarray([2.81215508, 24.66633285])
    np.testing.assert_allclose(automatic, reference, rtol=2.0e-3, atol=1e-6)


@pytest.mark.slow
def test_public_cartesian_limb_dark_gradient_matches_companion_reference():
    def magnification(source_coordinates):
        source = source_coordinates[0] + 1.0j * source_coordinates[1]
        result = mag_binary(
            jnp.asarray([source]),
            0.005,
            s=0.85,
            q=0.03,
            u1=0.5,
            backend="cpu",
            return_info=True,
        )
        return result.magnification[0]

    def companion_magnification(source_coordinates):
        source = source_coordinates[0] + 1.0j * source_coordinates[1]
        return _mag_limb_dark_cartesian_impl(
            source,
            0.005,
            s=0.85,
            q=0.03,
            u1=0.5,
            n_slice=8,
            n_profile=8,
            n_limb=64,
            axis=1.0j * source / jnp.abs(source),
            root_mode="companion",
        )

    point = jnp.asarray(
        [-0.03535533905932738, -0.035355339059327376], dtype=jnp.float64
    )
    automatic = np.asarray(jax.jit(jax.jacfwd(magnification))(point))
    reference = np.asarray(jax.jit(jax.jacfwd(companion_magnification))(point))
    np.testing.assert_allclose(
        automatic,
        reference,
        rtol=5e-5,
        atol=1e-6,
    )


@pytest.mark.slow
def test_public_small_q_limb_dark_forward_jacobian_is_finite():
    """Inactive LD quadrature lanes must not inject NaNs into forward AD."""

    times = jnp.linspace(-5.0, 5.0, 1000, dtype=jnp.float64)
    alpha = jnp.deg2rad(jnp.asarray(50.0, dtype=jnp.float64))
    tau = times / 10.0
    trajectory = (
        -0.05 * jnp.sin(alpha)
        + tau * jnp.cos(alpha)
        + 1.0j * (0.05 * jnp.cos(alpha) + tau * jnp.sin(alpha))
    )
    # These samples exercise the Cartesian and polar LD stages respectively
    # on the q=1e-6 example trajectory.
    base_sources = trajectory[jnp.asarray([472, 449])]

    def magnification(parameters):
        source_shift = parameters[0] + 1.0j * parameters[1]
        return mag_binary(
            base_sources + source_shift,
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            u1=0.5,
            backend="cpu",
            return_info=True,
        ).magnification

    parameters = jnp.asarray([0.0, 0.0, 0.005, 1.0, 1.0e-6])
    values = np.asarray(jax.jit(magnification)(parameters))
    forward = np.asarray(jax.jit(jax.jacfwd(magnification))(parameters))

    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(forward))


@pytest.mark.slow
def test_public_central_polar_forward_jacobian_is_finite():
    base_sources = jnp.asarray(
        [
            -7.071067811865475e-7 + 7.071067811865476e-7j,
            1.0e-4j,
            -2.77163859753386e-4 + 1.1480502970952697e-4j,
        ],
        dtype=jnp.complex128,
    )

    def magnification(parameters):
        source_shift = parameters[0] + 1.0j * parameters[1]
        return mag_binary(
            base_sources + source_shift,
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            backend="cpu",
            return_info=True,
        ).magnification

    parameters = jnp.asarray([0.0, 0.0, 1.0e-3, 0.9, 1.0e-4])
    values = np.asarray(jax.jit(magnification)(parameters))
    forward = np.asarray(jax.jit(jax.jacfwd(magnification))(parameters))

    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(forward))


@pytest.mark.slow
def test_public_central_contact_forward_jacobian_is_finite():
    rho = 5.0e-3
    angle = 2.5 * 2.0 * np.pi / 8.0
    source = 0.9 * rho * np.exp(1.0j * angle)

    def magnification(parameters):
        shifted_source = source + parameters[0] + 1.0j * parameters[1]
        return mag_binary(
            jnp.asarray([shifted_source]),
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            backend="cpu",
            return_info=True,
        ).magnification[0]

    parameters = jnp.asarray([0.0, 0.0, rho, 0.9, 1.0e-8])
    value = np.asarray(jax.jit(magnification)(parameters))
    forward = np.asarray(jax.jit(jax.jacfwd(magnification))(parameters))

    assert np.isfinite(value)
    assert np.all(np.isfinite(forward))


@pytest.mark.slow
def test_public_cpu_path_gradients_match_vbbl_finite_difference():
    def magnification(parameters):
        result = mag_binary(
            jnp.asarray([parameters[0] + 1.0j * parameters[1]]),
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            backend="cpu",
            return_info=True,
        )
        return result.magnification[0]

    point = jnp.asarray(
        [
            -0.3421551581838465,
            0.19397867617641237,
            0.0048408299514288945,
            0.8917065758514715,
            0.028320791269324543,
        ],
        dtype=jnp.float64,
    )
    automatic = np.asarray(jax.jit(jax.jacfwd(magnification))(point))
    # VBBinaryLensing Tol=RelTol=1e-10 with central steps
    # (2e-7, 2e-7, 1e-7, 3e-7, 5e-8).
    reference = np.asarray(
        [-6.64195328, 112.58030501, 30.67879960, 87.58013554, -446.69389094]
    )
    np.testing.assert_allclose(automatic, reference, rtol=5e-4, atol=1e-5)


@pytest.mark.slow
def test_refined_angular_path_gradients_match_finite_difference():
    def magnification(parameters):
        return mag_uniform_angular_moment_refined(
            parameters[0] + 1.0j * parameters[1],
            parameters[2],
            s=parameters[3],
            q=parameters[4],
            rtol=1e-3,
        )

    assert_grad_matches_finite_difference(
        magnification,
        [
            -0.0010946356833467202,
            -0.006172611801159194,
            0.00988422630886768,
            0.5992640188539046,
            0.0005041898087262488,
        ],
        [1e-7, 1e-7, 1e-7, 1e-6, 1e-7],
        rtol=5e-4,
    )
