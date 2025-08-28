import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from microjax.point_source import (
    mag_point_source,
    _images_point_source,
    critical_and_caustic_curves,
    lens_eq,
)


def analytic_A_single(u):
    u = np.abs(u)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def test_single_lens_magnification_basic_properties():
    # Basic properties: monotone decreasing in |w| and tends to 1 as |w|->inf
    ws = jnp.array([0.1 + 0j, 1.0 + 0j, 3.0 + 0j, 100.0 + 0j])
    A_num = np.array(mag_point_source(ws, nlenses=1))
    # monotone decreasing
    assert np.all(np.diff(A_num) < 0)
    # asymptotic to 1
    assert np.isclose(A_num[-1], 1.0, rtol=5e-4, atol=5e-4)


def test_single_lens_shape_scalar_and_vector():
    w_scalar = 0.5 + 0j
    w_vec = jnp.array([0.5 + 0j, 1.5 + 0j])
    A_s = mag_point_source(w_scalar, nlenses=1)
    A_v = mag_point_source(w_vec, nlenses=1)
    assert np.array(A_s).ndim == 0
    assert np.array(A_v).shape == (2,)


def test_images_satisfy_lens_eq_single():
    w = 0.7 + 0.3j
    z, mask = _images_point_source(w, nlenses=1)
    # Verify masked images satisfy lens eq: y = x - 1/conj(x)
    residual = np.array(lens_eq(np.array(z), nlenses=1) - np.array(w))
    ok = np.array(mask)
    assert residual[ok].size >= 1
    assert np.all(np.abs(residual[ok]) < 1e-9)


def test_binary_far_field_magnification_approx_one():
    # Far from lenses magnification -> 1
    params = {"s": 1.2, "q": 0.3}
    w = jnp.array([50.0 + 0j, 100.0 + 0j])
    A = np.array(mag_point_source(w, nlenses=2, **params))
    assert np.allclose(A, 1.0, rtol=1e-6, atol=1e-6)


def test_critical_caustic_curves_single_shapes():
    n = 32
    z_cr, z_ca = critical_and_caustic_curves(npts=n, nlenses=1)
    z_cr = np.array(z_cr)
    z_ca = np.array(z_ca)
    assert z_cr.shape == (n,)
    assert z_ca.shape == (n,)
    # Unit circle for critical curve; caustic at origin
    assert np.allclose(np.abs(z_cr), 1.0, rtol=1e-12, atol=1e-12)
    assert np.allclose(z_ca, 0.0 + 0.0j, atol=1e-12)
