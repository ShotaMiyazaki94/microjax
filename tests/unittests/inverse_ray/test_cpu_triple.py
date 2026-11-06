"""Independent critical triple-lens values and the ten-parameter forward Jacobian."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray import mag_triple

pytestmark = pytest.mark.fast
REFERENCES = json.loads((Path(__file__).resolve().parents[3] / "tools" / "cpu_triple_cases.json").read_text())


def test_triple_values_and_shared_brightness_geometry():
    traces = []

    @jax.jit
    def evaluate(source, p, coefficient):
        traces.append(None)
        rho, s, q, q3, r3, psi = p
        return mag_triple(source, rho, s=s, q=q, q3=q3, r3=r3, psi=psi, u1=coefficient, backend="cpu", return_info=True)

    # Independent VBM values cover two central-caustic cases, the third-body
    # fold, comparable masses, and small/wide and tiny third-lens images. The reference
    # tolerance was tightened independently before freezing these inputs.
    for case in REFERENCES["cases"]:
        source = jnp.asarray([complex(*case["source"])])
        parameters = jnp.asarray([case[key] for key in ("rho", "s", "q", "q3", "r3", "psi")])
        result = evaluate(source, parameters, jnp.asarray(0.5, dtype=jnp.float64))
        assert result._fields == ("magnification", "status")
        np.testing.assert_allclose(result.magnification, [case["magnification"]], rtol=1e-3, atol=0)

    # Uniform is the zero coefficient of the same compiled model. Check the
    # affine normalized flux at one full-solve point and one distant source.
    case = REFERENCES["cases"][0]
    parameters = jnp.asarray([case[key] for key in ("rho", "s", "q", "q3", "r3", "psi")])
    for source in (complex(*case["source"]), 5.0 + 3.0j):
        coefficients = (0.0, 0.5, 1.0)
        results = [
            evaluate(jnp.asarray([source]), parameters, jnp.asarray(coefficient, dtype=jnp.float64))
            for coefficient in coefficients
        ]
        flux = [result.magnification * (1 - coefficient / 3) for result, coefficient in zip(results, coefficients)]
        np.testing.assert_allclose(flux[1], (flux[0] + flux[2]) / 2, rtol=2e-12, atol=0)
        for result in results[1:]:
            np.testing.assert_array_equal(result.status, results[0].status)
    assert len(traces) == 1

    # Zero masses and coincident positions are lower-order lens models, not
    # valid degree-ten triples. Reject them rather than miscounting images.
    for invalid in (parameters.at[3].set(0.0), parameters.at[4].set(parameters[1] / 2).at[5].set(0.0)):
        result = evaluate(jnp.asarray([complex(*case["source"])]), invalid, jnp.asarray(0.5, dtype=jnp.float64))
        assert np.isnan(result.magnification).all()
        assert (result.status != 0).all()


def test_triple_forward_jacobian_matches_independent_reference():
    reference = REFERENCES["forward_jacobian"]

    def model(p, time):
        t0, t_e, u0, q, s, alpha, rho, q3, r3, psi = p
        tau = (time - t0) / t_e
        source = (-u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)) + 1j * (u0 * jnp.cos(alpha) + tau * jnp.sin(alpha))
        return mag_triple(jnp.asarray([source]), rho, s=s, q=q, q3=q3, r3=r3, psi=psi, u1=0.5, backend="cpu")[0]

    # VBM Tol=RelTol=1e-9, centered finite differences, independent of the
    # JAX graph. Halving the stored steps changed all ten columns by <8e-8
    # relative. This position requires the full finite-source calculation.
    evaluate = jax.jit(jax.jacfwd(model))
    parameters = jnp.asarray(reference["parameters"])
    actual = evaluate(parameters, reference["observation_time"])
    np.testing.assert_allclose(actual, reference["derivatives"], rtol=1e-3, atol=1e-6)

    # Reuse the compiled Jacobian at central and third-body caustic points.
    # Their selected columns have independent references; fixture notes retain
    # the radial-integration oracle's incomplete half-step check for r3.
    for case in REFERENCES["critical_forward_jacobians"]:
        actual = np.asarray(evaluate(parameters, case["observation_time"]))
        np.testing.assert_allclose(actual[case["parameter_indices"]], case["derivatives"], rtol=1e-3, atol=1e-6)
