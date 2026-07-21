import numpy as np
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from microjax.point_source import (
    lens_eq,
    lens_eq_det_jac,
    _images_point_source,
    _images_point_source_sequential,
    mag_point_source,
    critical_and_caustic_curves,
)
from microjax.lens_geometry import triple_lens_geometry

import pytest


def test_binary_images_satisfy_lens_equation_and_mask_true():
    params = {"s": 1.2, "q": 0.3}
    a = 0.5 * params["s"]
    e1 = params["q"] / (1.0 + params["q"]) 
    w = 0.37 - 0.19j

    # Compute images in mid-point frame, lens_eq expects 'a' and 'e1'
    z, mask = _images_point_source(w, nlenses=2, a=a, e1=e1)
    res = np.array(lens_eq(np.array(z), nlenses=2, a=a, e1=e1) - np.array(w))
    ok = np.array(mask)
    assert ok.any()
    assert np.all(np.abs(res[ok]) < 1e-6)


def test_binary_magnification_matches_sum_of_inverse_det():
    params = {"s": 1.2, "q": 0.3}
    w = 0.37 - 0.19j
    # mag_point_source takes mid-point coord in COM internally; we mirror its logic here
    a = 0.5 * params["s"]
    e1 = params["q"] / (1.0 + params["q"]) 
    x_cm = a * (1.0 - params["q"]) / (1.0 + params["q"]) 
    w_mid = w - x_cm

    z, mask = _images_point_source(w_mid, nlenses=2, a=a, e1=e1)
    det = lens_eq_det_jac(z, nlenses=2, a=a, e1=e1)
    A_expected = (1.0 / jnp.abs(det)) * mask
    A_expected = np.sum(np.array(A_expected))

    A_num = float(np.array(mag_point_source(jnp.array(w), nlenses=2, **params)))
    assert np.isclose(A_num, A_expected, rtol=1e-10, atol=1e-10)

def test_critical_caustic_mapping_binary():
    params = {"s": 1.1, "q": 0.5}
    a = 0.5 * params["s"]
    e1 = params["q"] / (1.0 + params["q"]) 
    x_cm = a * (1.0 - params["q"]) / (1.0 + params["q"]) 
    z_cr, z_ca = critical_and_caustic_curves(npts=64, nlenses=2, **params)
    # Undo the COM shift to test mapping property in mid-point coords
    z_cr_mid = z_cr - x_cm
    z_ca_mid = z_ca - x_cm
    mapped = lens_eq(z_cr_mid, nlenses=2, a=a, e1=e1)
    assert np.allclose(np.array(mapped), np.array(z_ca_mid), rtol=1e-6, atol=1e-6)


def test_triple_images_shape_and_mask_reasonable():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.3}
    a = 0.5 * params["s"]
    e1 = params["q"] / (1.0 + params["q"] + params["q3"]) 
    e2 = 1.0 / (1.0 + params["q"] + params["q3"]) 
    w = -0.1 + 0.2j
    z, m = _images_point_source(w, nlenses=3, a=a, r3=params["r3"], psi=params["psi"], e1=e1, e2=e2)
    # For a triple lens, up to 10 images are possible; check non-zero mask entries
    assert np.array(z).ndim == 1
    assert np.sum(np.array(m)) >= 1


def test_triple_magnification_finite_far_field():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.3}
    w = 100.0 + 0.0j
    A = float(np.array(mag_point_source(jnp.array(w), nlenses=3, **params)))
    # Should approach 1 for very large |w|
    assert np.isclose(A, 1.0, rtol=1e-8, atol=1e-8)


def test_triple_geometry_keeps_binary_com_public_and_total_com_internal():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    expected_center_of_mass = (
        -geometry.a * geometry.e2
        + geometry.a * geometry.e1
        + geometry.r3_complex * geometry.e3
    )
    expected_binary_shift = (
        geometry.a * (1.0 - params["q"]) / (1.0 + params["q"])
    )
    assert np.isclose(complex(geometry.shifted), expected_binary_shift)
    assert np.isclose(
        complex(geometry.total_shifted), -complex(expected_center_of_mass)
    )
    assert not np.isclose(float(geometry.total_shifted.imag), 0.0)


def test_triple_public_shift_is_independent_of_third_lens():
    s, q = 0.9, 0.3
    geometry = triple_lens_geometry(s, q, 0.8, 0.4, 0.7)
    expected = 0.5 * s * (1.0 - q) / (1.0 + q)
    assert np.isclose(complex(geometry.shifted), expected, atol=1e-15)


def test_triple_magnification_matches_midpoint_frame_inverse_det():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    w_binary_com = jnp.asarray(-0.1 + 0.2j)
    w_midpoint = w_binary_com - geometry.shifted
    lens_params = {
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
        "r3": params["r3"],
        "psi": params["psi"],
    }
    images, mask = _images_point_source(
        w_midpoint, nlenses=3, **lens_params
    )
    det = lens_eq_det_jac(images, nlenses=3, **lens_params)
    expected = jnp.sum(jnp.where(mask, 1.0 / jnp.abs(det), 0.0))
    actual = mag_point_source(w_binary_com, nlenses=3, **params)
    assert np.isclose(float(actual), float(expected), rtol=1e-11, atol=1e-12)


def test_triple_critical_caustic_mapping_uses_binary_center_of_mass():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    z_cr, z_ca = critical_and_caustic_curves(
        npts=64, nlenses=3, **params
    )
    lens_params = {
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
        "r3": params["r3"],
        "psi": params["psi"],
    }
    z_cr_midpoint = z_cr - geometry.shifted
    z_ca_midpoint = z_ca - geometry.shifted
    mapped = lens_eq(z_cr_midpoint, nlenses=3, **lens_params)
    assert np.allclose(
        np.asarray(mapped), np.asarray(z_ca_midpoint), rtol=1e-8, atol=1e-8
    )


def test_triple_binary_frame_gradient_matches_both_ad_modes():
    source = jnp.asarray(-0.1 + 0.2j)

    def magnification(q3):
        return mag_point_source(
            source,
            nlenses=3,
            s=0.9,
            q=0.3,
            q3=q3,
            r3=0.4,
            psi=0.7,
        )

    q3 = jnp.asarray(0.2)
    forward = jax.jacfwd(magnification)(q3)
    reverse = jax.grad(magnification)(q3)
    step = 1e-4
    finite_difference = (
        magnification(q3 - 2.0 * step)
        - 8.0 * magnification(q3 - step)
        + 8.0 * magnification(q3 + step)
        - magnification(q3 + 2.0 * step)
    ) / (12.0 * step)
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=1e-10, atol=1e-9)
    assert np.isclose(
        float(forward), float(finite_difference), rtol=1e-8, atol=1e-7
    )


def test_invalid_nlenses_raises():
    w = 0.0 + 0.0j
    try:
        _ = mag_point_source(w, nlenses=4)
        raised = False
    except ValueError:
        raised = True
    assert raised


@pytest.mark.parametrize("s", [0.1, 0.5, 2.0, 10.0])
@pytest.mark.parametrize("q", [1e-6, 1e-3, 0.1, 1.0])
def test_binary_far_field_magnification_over_param_grid(s, q):
    params = {"s": s, "q": q}
    # Large |w| should give A ~ 1
    w = jnp.array(100.0 + 0.0j)
    A = float(np.array(mag_point_source(w, nlenses=2, **params)))
    assert np.isclose(A, 1.0, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("s", [0.1, 0.5, 2.0, 10.0])
@pytest.mark.parametrize("q", [1e-6, 1e-3, 0.1, 1.0])
def test_binary_mag_matches_det_over_param_grid(s, q):
    # Check A = sum 1/|det J| for valid images across parameter space
    params = {"s": s, "q": q}
    a = 0.5 * s
    e1 = q / (1.0 + q)
    x_cm = a * (1.0 - q) / (1.0 + q)
    w = 0.37 - 0.19j
    w_mid = w - x_cm

    z, mask = _images_point_source(w_mid, nlenses=2, a=a, e1=e1)
    det = lens_eq_det_jac(z, nlenses=2, a=a, e1=e1)
    A_expected = np.sum(np.array((1.0 / jnp.abs(det)) * mask))

    A_num = float(np.array(mag_point_source(jnp.array(w), nlenses=2, **params)))
    # Allow a modest tolerance due to numerical differences across wide params
    assert np.isfinite(A_num)
    assert np.isclose(A_num, A_expected, rtol=1e-8, atol=1e-10)
