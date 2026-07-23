import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.roots.angular import ANGULAR_OK
from microjax.inverse_ray.extended_source import (
    mag_limb_dark_boundary,
    mag_radial_profile_boundary,
    mag_uniform_triple_boundary,
)
from microjax.inverse_ray.integrators.triple import _compact_edge_intensity, _hard_value_soft_jvp
from microjax.inverse_ray.geometry.topology import _polish_sampled_turning_radii
from microjax.point_source import mag_point_source

PARAMS = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}


def test_triple_soft_edge_bridge_keeps_hard_primal_and_uses_soft_tangent():
    def bridged(x):
        return _hard_value_soft_jvp(2.0 * x, x**2)

    value = bridged(jnp.asarray(3.0))
    primal, tangent = jax.jvp(bridged, (jnp.asarray(3.0),), (jnp.asarray(1.0),))
    assert float(value) == 6.0
    assert float(primal) == 6.0
    assert float(tangent) == 6.0


def test_triple_compact_edge_profile_has_unit_flux_and_zero_limb():
    radius = jnp.linspace(0.0, 1.0, 20_001)
    intensity = _compact_edge_intensity(radius)
    flux = 2.0 * jnp.trapezoid(radius * intensity, radius)
    assert np.isclose(float(flux), 1.0, rtol=0.0, atol=2e-8)
    assert float(intensity[-1]) == 0.0


def test_sampled_radial_turning_point_is_polished_without_new_samples():
    sample_index = jnp.arange(5, dtype=jnp.float64)
    radii = ((sample_index - 2.3) ** 2 + 1.0)[None, :]
    turning = jnp.asarray([[False, False, True, False, False]])
    polished = _polish_sampled_turning_radii(radii, turning, jnp.asarray(2.0))
    assert np.isclose(float(polished[0, 2]), 1.0, rtol=0.0, atol=1e-14)
    assert np.array_equal(np.asarray(polished[0, [0, 1, 3, 4]]), np.asarray(radii[0, [0, 1, 3, 4]]))


@pytest.mark.parametrize("name", ["nlenses", "r_resolution", "bins_r", "bins_th", "margin_th"])
def test_triple_boundary_rejects_removed_dense_arguments(name):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mag_uniform_triple_boundary(0.8 + 0.4j, 1e-2, **PARAMS, **{name: 3})


def test_triple_uniform_boundary_is_clean_and_reaches_small_source_limit():
    source = jnp.asarray(1.5 - 0.7j)
    rho = 1e-2
    result = mag_uniform_triple_boundary(
        source,
        rho,
        Nlimb=100,
        angular_atol=1e-4,
        return_info=True,
        **PARAMS,
    )
    point = mag_point_source(source, nlenses=3, **PARAMS)
    assert int(result.status) == ANGULAR_OK
    assert float(result.estimated_error) <= 1e-4
    assert np.isclose(float(result.magnification), float(point), rtol=3e-5, atol=0.0)


def test_triple_uniform_boundary_reverse_matches_forward_mode():
    def magnification(q3):
        return mag_uniform_triple_boundary(
            0.8 + 0.4j,
            1e-2,
            s=0.9,
            q=0.3,
            q3=q3,
            r3=0.4,
            psi=0.7,
            Nlimb=30,
            angular_atol=1e-3,
        )

    q3 = jnp.asarray(0.2)
    value = magnification(q3)
    primal, _ = jax.jvp(magnification, (q3,), (jnp.ones_like(q3),))
    forward = jax.jacfwd(magnification)(q3)
    reverse = jax.grad(magnification)(q3)
    assert float(primal) == float(value)
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=1e-6, atol=1e-8)


def test_triple_uniform_boundary_does_not_differentiate_padded_support():
    def magnification(margin_r):
        return mag_uniform_triple_boundary(
            0.8 + 0.4j,
            1e-2,
            margin_r=margin_r,
            Nlimb=30,
            angular_atol=1e-3,
            **PARAMS,
        )

    derivative = jax.jacfwd(magnification)(jnp.asarray(0.5))
    assert float(derivative) == 0.0


def test_triple_generic_radial_profile_reduces_to_uniform_source():
    common = dict(
        w_center=0.8 + 0.4j,
        rho=1e-2,
        Nlimb=60,
        angular_atol=1e-3,
        return_info=True,
        **PARAMS,
    )
    uniform = mag_uniform_triple_boundary(**common)
    profile = mag_radial_profile_boundary(
        radial_intensity=lambda distance: jnp.ones_like(distance),
        intensity_flux=jnp.pi,
        nlenses=3,
        **common,
    )
    limb_zero = mag_limb_dark_boundary(nlenses=3, u1=0.0, **common)
    assert int(profile.status) == ANGULAR_OK
    assert np.isclose(
        float(profile.magnification),
        float(uniform.magnification),
        rtol=0.0,
        atol=2e-10,
    )
    assert np.isclose(
        float(limb_zero.magnification),
        float(uniform.magnification),
        rtol=0.0,
        atol=2e-10,
    )


def test_triple_linear_limb_profile_is_finite_and_status_clean():
    result = mag_limb_dark_boundary(
        0.8 + 0.4j,
        1e-2,
        nlenses=3,
        u1=0.5,
        Nlimb=60,
        angular_atol=1e-3,
        return_info=True,
        **PARAMS,
    )
    assert int(result.status) == ANGULAR_OK
    assert np.isfinite(float(result.magnification))
    assert float(result.estimated_error) <= 1e-3


@pytest.mark.parametrize("u1", [0.0, 0.5])
def test_triple_profile_supports_one_pass_fixed_radial_integration(u1):
    result = mag_limb_dark_boundary(
        0.8 + 0.4j,
        1e-2,
        nlenses=3,
        u1=u1,
        Nlimb=40,
        angular_atol=1e-3,
        max_radial_subdivisions=1,
        radial_strategy="fixed",
        certify_topology=False,
        return_info=True,
        **PARAMS,
    )
    assert np.isfinite(float(result.magnification))
    assert int(result.status) == ANGULAR_OK
