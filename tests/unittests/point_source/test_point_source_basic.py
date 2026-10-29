import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from microjax.multipole import mag_hexadecapole
from microjax.point_source import (
    _images_point_source,
    _images_point_source_sequential,
    critical_and_caustic_curves,
    lens_eq,
    lens_eq_det_jac,
    mag_point_source,
)


def test_binary_far_field_magnification_approx_one():
    sources = jnp.asarray([50.0 + 0j, 100.0 + 0j])
    magnification = mag_point_source(sources, nlenses=2, s=1.2, q=0.3)
    np.testing.assert_allclose(magnification, 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("nlenses", [0, 1, 4])
@pytest.mark.parametrize(
    "operation",
    [
        lambda n: lens_eq(jnp.asarray(0.7 + 0.3j), nlenses=n),
        lambda n: lens_eq_det_jac(jnp.asarray(0.7 + 0.3j), nlenses=n),
        lambda n: _images_point_source(jnp.asarray(0.7 + 0.3j), nlenses=n),
        lambda n: _images_point_source_sequential(jnp.asarray([0.7 + 0.3j]), nlenses=n),
        lambda n: critical_and_caustic_curves(npts=8, nlenses=n),
        lambda n: mag_point_source(jnp.asarray(0.7 + 0.3j), nlenses=n),
        lambda n: mag_hexadecapole(jnp.asarray([0.7 + 0.3j]), jnp.asarray([True]), 0.01, nlenses=n),
    ],
    ids=["lens-equation", "jacobian", "images", "sequential-images", "caustics", "magnification", "multipole"],
)
def test_unsupported_lens_counts_are_rejected_before_model_parameters(operation, nlenses):
    # The removed single-lens API must not silently fall through to a binary
    # model or fail later with a missing-parameter KeyError.
    with pytest.raises(ValueError, match="nlenses.*2 or 3"):
        operation(nlenses)


def test_finite_source_single_lens_package_is_not_available():
    with pytest.raises(ModuleNotFoundError, match=r"microjax\.fspl"):
        importlib.import_module("microjax.fspl")
