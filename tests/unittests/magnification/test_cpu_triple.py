"""Independent critical triple-lens values and the ten-parameter forward Jacobian."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax import make_mag_triple, make_triple_lightcurve

pytestmark = pytest.mark.fast
REFERENCES = json.loads((Path(__file__).resolve().parents[3] / "tools" / "cpu_triple_cases.json").read_text())


def test_triple_values_and_shared_brightness_geometry():
    mag_triple = make_mag_triple(backend="cpu", return_info=True)
    traces = []

    @jax.jit
    def evaluate(source, p, coefficient):
        traces.append(None)
        rho, s, q, q3, r3, psi = p
        return mag_triple(source, rho, s=s, q=q, q3=q3, r3=r3, psi=psi, u1=coefficient)

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

    times = [reference["observation_time"]] + [
        case["observation_time"] for case in REFERENCES["critical_forward_jacobians"]
    ]
    lightcurve = make_triple_lightcurve()
    parameters = jnp.asarray(reference["parameters"])

    # VBM Tol=RelTol=1e-9, centered finite differences, independent of the
    # JAX graph. Halving the stored steps changed all ten columns by <8e-8
    # relative. This position requires the full finite-source calculation.
    derivative = jax.jit(jax.jacfwd(lightcurve, argnums=0))
    actual = np.asarray(derivative(parameters, jnp.asarray(times), u1=0.5))
    np.testing.assert_allclose(actual[0], reference["derivatives"], rtol=1e-3, atol=1e-6)

    # Central and third-body caustics have independent selected-column references.
    # Fixture notes retain the radial oracle's incomplete half-step check for r3.
    for row, case in zip(actual[1:], REFERENCES["critical_forward_jacobians"]):
        np.testing.assert_allclose(row[case["parameter_indices"]], case["derivatives"], rtol=1e-3, atol=1e-6)
