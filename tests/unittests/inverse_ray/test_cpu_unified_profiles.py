"""Uniform is the zero-coefficient member of the shared CPU brightness law."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray import mag_binary
from microjax.inverse_ray.cpu.lightcurve import mag_binary_cpu_one_shot_lightcurve


@pytest.mark.slow
@pytest.mark.parametrize('public', [False, True])
def test_uniform_and_ld_share_geometry_work_and_affine_flux(public):
    # This source formerly selected the separate uniform radial kernel.
    source = jnp.asarray([-0.1742947428830735-0.0009439342481879419j])
    rho, s, q = 5.751765563428599e-5, 0.9166666666666666, 1e-6
    coefficients = np.asarray([0., .25, .5, 1.])
    results = []
    for u1 in coefficients:
        if public:
            result = mag_binary(source, rho, s=s, q=q, u1=float(u1), backend='cpu', return_info=True)
        else:
            result = mag_binary_cpu_one_shot_lightcurve(source, rho, s=s, q=q, u1=float(u1))
        results.append(jax.device_get(result))
    assert int(results[0].tier[0]) in (5, 8)
    for result in results[1:]:
        np.testing.assert_array_equal(result.tier, results[0].tier)
        np.testing.assert_array_equal(result.status, results[0].status)
        np.testing.assert_array_equal(result.n_limb, results[0].n_limb)
        np.testing.assert_array_equal(result.n_radial_nodes, results[0].n_radial_nodes)
    magnification = np.asarray([r.magnification[0] for r in results])
    flux = magnification*(1-coefficients/3)
    expected = flux[0]*(1-coefficients)+flux[-1]*coefficients
    np.testing.assert_allclose(flux, expected, rtol=2e-11, atol=0)


@pytest.mark.parametrize('source,rho,s,q,reference', [
    (0.5884177083962848+0.006969559388922229j, .005477225575051662,
     1.3333333333333333, 3.1622776601683795e-5, 1.8618390896235484),
    (0.8465177256186989-.3124763697006164j, .2720446295173432,
     1.5, .007196856730011514, 1.4174782671260222),
    (0.668646800987191-.5921567190284301j, .5215789772578485,
     1.4166666666666665, .019306977288832496, 1.4567448532418397),
])
def test_shared_uniform_kernel_resolves_support_missed_by_the_56_point_trace(source, rho, s, q, reference):
    # Independent uniform VBMicrolensing Tol=RelTol=1e-7. Quadrature refinement
    # alone did not repair the missing angular support in these configurations.
    result = mag_binary_cpu_one_shot_lightcurve(jnp.asarray([source]), rho, s=s, q=q, u1=0.)
    np.testing.assert_allclose(result.magnification, [reference], rtol=1e-4, atol=0)


def test_shared_uniform_kernel_keeps_the_thin_central_image_contribution():
    # A full-grid regression missed by the initial screen: 80 trace samples
    # lose 0.143% of the uniform area. More outer quadrature does not recover it.
    result = mag_binary_cpu_one_shot_lightcurve(
        jnp.asarray([.0010338458121469764+.00011373998388780149j]),
        3e-5, s=.5833333333333334, q=.0006105402296585327, u1=0.,
    )
    # VBMicrolensing Tol=RelTol=1e-9; also stable at 1e-7 and 1e-8.
    np.testing.assert_allclose(result.magnification, [821.3838029661517], rtol=1e-3, atol=0)
