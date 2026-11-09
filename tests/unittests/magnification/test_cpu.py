"""The CPU public contract, critical support cases, and forward derivatives."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax import make_mag_binary, make_binary_lightcurve

pytestmark = pytest.mark.fast


def test_public_result_preserves_best_effort_values(monkeypatch):
    from microjax._magnification import lightcurve

    expected = SimpleNamespace(
        magnification=jnp.asarray([4.0, 5.0, jnp.nan]),
        status=jnp.asarray([0, 1, 1], dtype=jnp.int32),
    )
    monkeypatch.setattr(lightcurve, "_mag_binary_cpu_impl", lambda *args, **kwargs: expected)
    sources = jnp.asarray([0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j])
    mag = make_mag_binary(backend="cpu")
    mag_with_info = make_mag_binary(backend="cpu", return_info=True)
    ordinary = mag(sources, 0.01, s=1.0, q=0.3)
    diagnostic = mag_with_info(sources, 0.01, s=1.0, q=0.3)
    np.testing.assert_array_equal(ordinary, expected.magnification)
    assert diagnostic is expected


def test_empty_input_keeps_the_public_result_shape():
    mag_binary = make_mag_binary(backend="cpu", return_info=True)
    result = mag_binary(jnp.asarray([], dtype=jnp.complex128), 0.01, s=1.0, q=0.3)
    assert result._fields == ("magnification", "status")
    assert result.magnification.shape == result.status.shape == (0,)


def test_values_retain_critical_image_support():
    mag_binary = make_mag_binary(backend="cpu", return_info=True)
    # Independent frozen VBM/VBBL values. These cover smooth values, hidden
    # angular support, a thin central image, and a narrow Cartesian image pair.
    cases = (
        (0.1 + 0.2j, 0.01, 1.0, 0.3, 0.0, 4.5028889540801815, 5e-7),
        (0.1 + 0.2j, 0.01, 1.0, 0.3, 0.5, 4.502713625245832, 5e-7),
        (
            0.5884177083962848 + 0.006969559388922229j,
            0.005477225575051662,
            1.3333333333333333,
            3.1622776601683795e-5,
            0.0,
            1.8618390896235484,
            1e-4,
        ),
        (
            0.0010338458121469764 + 0.00011373998388780149j,
            3e-5,
            0.5833333333333334,
            0.0006105402296585327,
            0.0,
            821.3838029661517,
            1e-3,
        ),
        (-0.22122504688671182 - 0.18562985524981634j, 0.03, 0.9, 0.01, 0.5, 4.055193392644604, 1e-3),
        (0.5823420163986178 + 0.0001720705839026j, 3e-5, 1.3333333333333333, 1e-6, 0.5, 1.7985531538150774, 1e-4),
        # The former eight-node LD rule missed this by 0.165%. Fresh VBM
        # Tol=RelTol=1e-8 and 1e-9 agree within 6.1e-10 relative.
        (
            0.5174784122812909 + 0.0004756466207490966j,
            0.0014900498023478466,
            1.2916666666666665,
            1e-6,
            0.5,
            2.791818587815443,
            1e-3,
        ),
        # Saved VBBL cases from the retired local-chart solver: an enclosing
        # annulus, narrow fold pairs, and transient planetary images.
        (-2.845269680283363e-05 + 8.040213267373542e-05j, 1e-4, 1.6, 1e-5, 0.0, 15873.646704428846, 1e-4),
        (0.025872115099088773 - 0.00012660115914060716j, 1e-5, 1.0, 1e-5, 0.0, 144.17177551337485, 1e-4),
        (-7.794776723440755e-05 - 7.416540530774523e-05j, 1e-5, 1.6, 1e-4, 0.0, 20696.29552460964, 1e-4),
        (0.9824943188437842 + 0.00880220969350141j, 1e-5, 1.6, 1e-3, 0.0, 49.87308550714601, 1e-4),
        (-0.728614562384981 - 0.007200699676102258j, 1e-5, 0.7, 1e-5, 0.0, 7.055904959651395, 1e-4),
        (0.9788824508896057 + 0.001942618692247761j, 1e-4, 1.6, 1e-4, 0.0, 9.551087158506649, 1e-4),
    )
    for source, rho, s, q, u1, reference, tolerance in cases:
        result = mag_binary(jnp.asarray([source]), rho, s=s, q=q, u1=u1)
        assert result._fields == ("magnification", "status")
        np.testing.assert_allclose(result.magnification, [reference], rtol=tolerance, atol=0)

    # One compiled model spans uniform and LD, with both full and multipole
    # sources. Normalized flux is affine in the brightness coefficient.
    traces = []

    @jax.jit
    def model(coefficient):
        traces.append(None)
        return mag_binary(
            jnp.asarray([-0.22122504688671182 - 0.18562985524981634j, 2.0 + 1.0j]),
            0.03,
            s=0.9,
            q=0.01,
            u1=coefficient,
        )

    coefficients = (0.0, 0.5, 1.0)
    results = [model(coefficient) for coefficient in coefficients]
    assert len(traces) == 1
    flux = [result.magnification * (1 - coefficient / 3) for result, coefficient in zip(results, coefficients)]
    np.testing.assert_allclose(flux[1], (flux[0] + flux[2]) / 2, rtol=2e-12, atol=0)
    for result in results[1:]:
        np.testing.assert_array_equal(result.status, results[0].status)
    _, derivative = jax.jvp(lambda a: model(a).magnification * (1 - a / 3), (0.0,), (1.0,))
    np.testing.assert_allclose(derivative, flux[2] - flux[0], rtol=2e-10, atol=2e-12)


def test_public_forward_jacobian_matches_independent_reference():
    from microjax._magnification.cpu.one_shot import mag_binary_cpu_one_shot

    x, y = -0.3421551581838465, 0.19397867617641237
    parameters = jnp.asarray([-x, 1.0, y, 0.028320791269324543, 0.8917065758514715, 0.0, 0.0048408299514288945])
    lightcurve = make_binary_lightcurve(backend="cpu")
    # VBBinaryLensing Tol=RelTol=1e-10, centered steps
    # (2e-7, 2e-7, 1e-7, 3e-7, 5e-8), independent of the JAX graph.
    dx, dy, drho, ds, dq = [-6.64195328, 112.58030501, 30.67879960, 87.58013554, -446.69389094]
    # At time zero and alpha=0, w=x+iy. Convert the independent position
    # derivatives analytically to the seven event-parameter derivatives.
    expected = [-dx, -x * dx, dy, dq, ds, -y * dx + x * dy, drho]
    traces = []

    def model(p, times):
        traces.append(None)
        return lightcurve(p, times)

    derivative = jax.jit(jax.jacfwd(model, argnums=0))
    actual = derivative(parameters, jnp.asarray([0.0]))
    # Shift the time origin: the same compiled function must use the new times.
    shifted = derivative(parameters.at[0].add(0.25), jnp.asarray([0.25]))
    assert len(traces) == 1
    np.testing.assert_allclose(actual, [expected], rtol=5e-4, atol=1e-5)
    np.testing.assert_allclose(shifted, actual, rtol=1e-9, atol=1e-9)

    # Force a full theta solve instead of taking the public multipole shortcut.
    # Its moving-endpoint rho derivative exposed a former outer-rule failure;
    # one directional check is sufficient to keep that integration path live.
    theta = jax.jit(
        lambda rho: mag_binary_cpu_one_shot(
            jnp.asarray(0.1 + 0.2j),
            rho,
            s=1.0,
            q=0.3,
            u1=0.5,
        ).magnification
    )
    rho = jnp.asarray(0.01)
    _, derivative = jax.jit(lambda r: jax.jvp(theta, (r,), (jnp.ones_like(r),)))(rho)
    step = 1e-5
    finite_difference = (theta(rho + step) - theta(rho - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-3, atol=1e-6)


def test_root_correction_and_moving_support_keep_their_derivatives():
    from microjax._magnification.cpu.line_roots import (
        binary_line_level_set_roots,
        physical_binary_line_level_set_roots,
    )
    from microjax._magnification.cpu.angular_moment import _refine_angular_tangencies
    from microjax._magnification.cpu.lens import binary_lens

    source = jnp.asarray(-1.4998598061310287 + 0.003443021045947857j)
    arguments = (-0.0004415497608313467j, -1.0 + 0j, source, 0.00021142727844446993)
    polynomial = binary_line_level_set_roots(*arguments, s=0.5, q=1e-6)
    physical = physical_binary_line_level_set_roots(*arguments, s=0.5, q=1e-6)
    real = np.asarray(polynomial.imag == 0)
    np.testing.assert_array_equal(physical.imag == 0, real)
    np.testing.assert_array_equal(np.asarray(physical)[~real], np.asarray(polynomial)[~real])
    roots = np.sort(np.asarray(physical.real)[real])
    assert roots.size == 2
    # Independent 100-digit physical roots protect this cancellation-sensitive width.
    width = -0.49985138312807610138 - (-0.50019028574250084110)
    np.testing.assert_allclose(roots[1] - roots[0], width, rtol=1e-8, atol=0)

    def endpoint(p):
        value, _ = _refine_angular_tangencies(
            jnp.asarray(0.1 + 0.2j),
            jnp.asarray(0.01),
            lens=binary_lens(jnp.asarray(1.0), jnp.asarray(0.3)),
            initial_images=jnp.asarray([jnp.nan + 0j]),
            initial_angles=jnp.asarray([0.4]),
            fallback_angles=jnp.asarray([p[0] + 2 * p[1]]),
            active=jnp.asarray([True]),
            required=jnp.asarray([True]),
            bracketed=jnp.asarray([True]),
        )
        return value

    np.testing.assert_allclose(jax.jit(jax.jacfwd(endpoint))(jnp.asarray([0.2, 0.1])), [[1.0, 2.0]], rtol=0, atol=0)
