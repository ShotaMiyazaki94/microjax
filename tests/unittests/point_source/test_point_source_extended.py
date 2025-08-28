import numpy as np
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
