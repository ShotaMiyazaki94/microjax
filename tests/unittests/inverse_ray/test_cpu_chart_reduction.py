"""Physical root and moving fallback contracts required by two LD charts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_moment import _refine_angular_tangencies


@pytest.mark.fast
def test_failed_tangency_preserves_the_returned_fitted_endpoint_derivative():
    # A bracketed trace may supply the returned endpoint even when Newton
    # fails. Its parameter dependence must survive that fallback.
    def endpoint(p, implicit):
        fallback = jnp.asarray([p[0] + 2*p[1]])
        return _refine_angular_tangencies(
            jnp.asarray(.1+.2j), jnp.asarray(.01),
            s=jnp.asarray(1.), q=jnp.asarray(.3),
            initial_images=jnp.asarray([jnp.nan+0j]),
            initial_angles=jnp.asarray([.4]), fallback_angles=fallback,
            active=jnp.asarray([True]), required=jnp.asarray([True]),
            bracketed=jnp.asarray([True]), implicit_derivatives=implicit,
        )

    p = jnp.asarray([.2, .1])
    value, valid = jax.jit(lambda p: endpoint(p, True))(p)
    stopped, _ = jax.jit(lambda p: endpoint(p, False))(p)
    np.testing.assert_array_equal(value, stopped)
    np.testing.assert_array_equal(valid, [True])
    derivative = jax.jit(jax.jacfwd(lambda p: endpoint(p, True)[0]))(p)
    np.testing.assert_allclose(derivative, [[1., 2.]], rtol=0, atol=0)
    np.testing.assert_array_equal(jax.jacfwd(lambda p: endpoint(p, False)[0])(p), 0)


@pytest.mark.fast
def test_physical_line_correction_keeps_complex_pairs_and_high_precision_width():
    from microjax.inverse_ray.cpu.line_roots import (
        binary_line_level_set_roots, physical_binary_line_level_set_roots,
    )
    source = jnp.asarray(-1.4998598061310287 + .003443021045947857j)
    options = dict(s=.5, q=1e-6)
    offset, direction, rho = -.0004415497608313467j, -1.+0j, .00021142727844446993
    polynomial = jax.jit(lambda w: binary_line_level_set_roots(offset, direction, w, rho, **options))(source)
    physical = jax.jit(lambda w: physical_binary_line_level_set_roots(offset, direction, w, rho, **options))(source)
    mask = np.asarray(polynomial.imag == 0)
    np.testing.assert_array_equal(physical.imag == 0, mask)
    np.testing.assert_array_equal(np.asarray(physical)[~mask], np.asarray(polynomial)[~mask])
    roots = np.sort(np.asarray(physical.real)[mask])
    assert len(roots) == 2
    # Independently computed 100-digit physical roots (not a solver fixture).
    expected = -.49985138312807610138 - (-.50019028574250084110)
    np.testing.assert_allclose(roots[1]-roots[0], expected, rtol=1e-8, atol=0)


@pytest.mark.fast
def test_physical_line_root_jvp_and_second_derivative_match_a_regular_line():
    from microjax.inverse_ray.cpu.line_roots import physical_binary_line_level_set_roots
    from microjax.inverse_ray.geometry.lens import binary_geometry

    rho, s, q = .08, 1.3, .3
    lens = binary_geometry(s, q)
    image = .35+.6j
    zbar = jnp.conj(image)-lens.shifted
    source = image-lens.e1/(zbar-lens.a)-(1-lens.e1)/(zbar+lens.a)

    def roots(x):
        return physical_binary_line_level_set_roots(jnp.asarray([.35+0j]), 1j, x+1j*source.imag, rho, s=s, q=q)[0]

    x = source.real
    value, first, second = jax.jit(lambda x: (roots(x), jax.jacfwd(roots)(x), jax.jacfwd(jax.jacfwd(roots))(x)))(x)
    mask = np.asarray(value.imag == 0)
    assert mask.any()
    h = 2e-6
    fd_first = (roots(x+h)-roots(x-h))/(2*h)
    fd_second = (jax.jacfwd(roots)(x+h)-jax.jacfwd(roots)(x-h))/(2*h)
    np.testing.assert_allclose(np.asarray(first)[mask], np.asarray(fd_first)[mask], rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(second)[mask], np.asarray(fd_second)[mask], rtol=2e-4, atol=1e-6)
