import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.coeffs import _poly_coeffs_binary
from microjax.inverse_ray.cpu.coefficients import binary_quintic_coefficients
from microjax.inverse_ray.cpu.support import (
    _physical_image_mask,
    trace_binary_source_limb,
)
from microjax.point_source import lens_eq
from microjax.poly_solver import poly_roots


pytestmark = pytest.mark.fast


def _lens_residuals(images_com, w_com, s, q):
    a = 0.5 * s
    e1 = q / (1.0 + q)
    midpoint_to_com = a * (1.0 - q) / (1.0 + q)
    return jnp.abs(
        lens_eq(
            images_com - midpoint_to_com,
            nlenses=2,
            a=a,
            e1=e1,
        )
        - (w_com - midpoint_to_com)
    )


def _solve_stable(w, s, q):
    quintic = binary_quintic_coefficients(w, s=s, q=q)
    roots = poly_roots(quintic.coefficients[None, :])[0]
    return roots + quintic.image_shift


@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("q", [1.0, 1e-3, 1e-6, 1e-9])
def test_q_aware_coefficients_recover_three_physical_images(s, q):
    w = jnp.asarray(-1.49981640625 + 0.0035j, dtype=jnp.complex128)
    quintic = binary_quintic_coefficients(w, s=s, q=q)
    images = _solve_stable(w, s, q)
    residuals = _lens_residuals(images, w, s, q)

    assert np.all(np.isfinite(np.asarray(quintic.coefficients)))
    assert int(jnp.sum(residuals < 1e-6)) >= 3


@pytest.mark.parametrize(
    "s,q,w",
    [
        (0.5, 0.3, -0.2 + 0.1j),
        (1.0, 1.0, 0.17 - 0.31j),
        (1.7, 1e-3, 0.4 + 0.2j),
    ],
)
def test_q_aware_and_existing_coefficients_have_same_physical_images(s, q, w):
    w = jnp.asarray(w, dtype=jnp.complex128)
    stable_images = _solve_stable(w, s, q)

    a = 0.5 * s
    e1 = q / (1.0 + q)
    midpoint_to_com = a * (1.0 - q) / (1.0 + q)
    existing_coefficients = _poly_coeffs_binary(w - midpoint_to_com, a, e1)
    existing_images = poly_roots(existing_coefficients[None, :])[0]
    existing_images = existing_images + midpoint_to_com

    stable_residuals = np.asarray(_lens_residuals(stable_images, w, s, q))
    existing_residuals = np.asarray(_lens_residuals(existing_images, w, s, q))
    stable_physical = np.sort_complex(
        np.asarray(stable_images)[stable_residuals < 1e-7]
    )
    existing_physical = np.sort_complex(
        np.asarray(existing_images)[existing_residuals < 1e-7]
    )

    assert stable_physical.shape == existing_physical.shape
    np.testing.assert_allclose(
        stable_physical, existing_physical, rtol=1e-10, atol=1e-10
    )


def test_q_aware_coefficients_support_jit_and_jvp():
    w = jnp.asarray(-0.2 + 0.1j, dtype=jnp.complex128)

    @jax.jit
    def coefficients(mass_ratio):
        return binary_quintic_coefficients(w, s=0.8, q=mass_ratio).coefficients

    values, tangent = jax.jvp(
        coefficients,
        (jnp.asarray(1e-5),),
        (jnp.asarray(1e-6),),
    )
    assert values.shape == (6,)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(tangent)))


def test_limb_trace_returns_the_roots_used_for_its_physical_mask():
    """Support coordinates and the physical-root certificate share one trace."""

    source = jnp.asarray(-0.1473754964687426 + 0.07700093692907822j)
    images, physical_mask = trace_binary_source_limb(
        source,
        3.0e-3,
        s=1.0,
        q=1.0e-2,
        n_limb=64,
    )
    phases = 2.0 * jnp.pi * jnp.arange(images.shape[1]) / images.shape[1]
    sources = source + 3.0e-3 * jnp.exp(1.0j * phases)
    recomputed = jax.vmap(
        lambda image, limb_source: _physical_image_mask(
            image, limb_source, 1.0, 1.0e-2
        ),
        in_axes=(1, 0),
    )(images, sources).T
    np.testing.assert_array_equal(np.asarray(physical_mask), np.asarray(recomputed))
