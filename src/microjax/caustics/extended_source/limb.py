# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Utilities for sampling point-source images along the source limb."""

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, random

from ...utils import match_points
from ...point_source import (
    lens_eq_det_jac,
    _images_point_source,
    _images_point_source_sequential,
)


def _permute_images(z, z_mask, z_parity):
    """Reorder image indices so neighbouring limb samples follow the same image.

    The raw output of ``_images_point_source`` for successive source-limb
    evaluations contains images in arbitrary order.  For contour construction we
    need consistent indexing such that, for each limb angle, image ``i`` is the
    continuation of image ``i`` from the previous angle.  This helper walks
    around the limb once using ``match_points`` to greedily match each new batch
    of images with the final ordering from the previous sample.

    Parameters
    ----------
    z : array_like
        Complex array of shape ``(n_images, n_samples)`` with image positions.
    z_mask : array_like
        Boolean array indicating whether each entry represents a real image.
    z_parity : array_like
        Array with the parity (sign of the Jacobian) for each image sample.

    Returns
    -------
    tuple of array_like
        Reordered ``(z, z_mask, z_parity)`` arrays such that each column traces
        a continuous image sequence around the source limb.
    """
    xs = jnp.stack([z, z_mask, z_parity])

    def apply_match_points(carry, xs):
        z, z_mask, z_parity = xs
        idcs = match_points(carry, z)
        return z[idcs], jnp.stack([z[idcs], z_mask[idcs], z_parity[idcs]])

    init = xs[0, :, 0]
    _, xs = lax.scan(apply_match_points, init, jnp.moveaxis(xs, -1, 0))
    z, z_mask, z_parity = jnp.moveaxis(xs, 1, 0)

    return z.T, z_mask.real.astype(bool).T, z_parity.T


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts",
        "niter",
    ),
)
def _images_of_source_limb(
    w0,
    rho,
    nlenses=2,
    npts=300,
    niter=10,
    **params,
):
    """Sample point-source images along the limb of a circular source.

    The routine follows the prescription of the ICRS method: start with a
    coarse sampling of the source circumference, reuse previously found image
    roots when moving along the limb, and adaptively insert additional limb
    angles where consecutive samples are far apart.  A tiny random perturbation
    is added whenever duplicate roots appear to avoid numerical degeneracies in
    downstream processing.

    Parameters
    ----------
    w0 : complex
        Source-centre position in the complex plane (already shifted to the
        coordinate frame expected by the lens equation helpers).
    rho : float
        Angular radius of the source in Einstein units.
    nlenses : int, optional
        Lens multiplicity (1–3).  Determines the degree of the polynomial
        solved by ``_images_point_source`` and the expected number of images.
    npts : int, optional
        Target number of limb samples once refinement is complete.
    niter : int, optional
        Number of refinement passes when inserting midpoints between wide image
        separations.
    **params : dict
        Additional lens parameters forwarded to the point-source solver.

    Returns
    -------
    tuple of array_like
        ``(z, z_mask, z_parity)`` arrays describing the images, their validity
        masks, and parity for every limb sample after permutation.
    """
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)

    def fn(theta, z_init):
        # Add a small perturbation to avoid duplicate convergence for nearby points
        u1 = random.uniform(key1, shape=z_init.shape, minval=-1e-6, maxval=1e-6)
        u2 = random.uniform(key2, shape=z_init.shape, minval=-1e-6, maxval=1e-6)
        z_init = z_init + u1 + u2 * 1j

        z, z_mask = _images_point_source(
            rho * jnp.exp(1j * theta) + w0,
            nlenses=nlenses,
            z_init=z_init.T,
            custom_init=True,
            **params,
        )
        det = lens_eq_det_jac(z, nlenses=nlenses, **params)
        z_parity = jnp.sign(det)
        return z, z_mask, z_parity

    # Initial sampling on the source limb
    npts_init = int(0.5 * npts)
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
    z, z_mask = _images_point_source_sequential(
        rho * jnp.exp(1j * theta) + w0, nlenses=nlenses, **params
    )
    z_parity = jnp.sign(lens_eq_det_jac(z, nlenses=nlenses, **params))

    # Refine sampling by adding npts_init additional points a fraction 1/niter at a time
    npts_additional = int(0.5 * npts)
    n = int(npts_additional / niter)

    for _ in range(niter):
        delta_z = jnp.abs(z[:, 1:] - z[:, :-1])
        delta_z = jnp.where(
            jnp.logical_or(z_mask[:, 1:], z_mask[:, :-1]),
            delta_z,
            jnp.zeros_like(delta_z.real),
        )
        delta_z_max = jnp.max(delta_z, axis=0)
        idcs_theta = jnp.argsort(delta_z_max)[::-1][:n]

        theta_new = 0.5 * (theta[idcs_theta] + theta[idcs_theta + 1])
        z_new, z_mask_new, z_parity_new = fn(theta_new, z[:, idcs_theta])

        theta = jnp.insert(theta, idcs_theta + 1, theta_new, axis=0)
        z = jnp.insert(z, idcs_theta + 1, z_new, axis=1)
        z_mask = jnp.insert(z_mask, idcs_theta + 1, z_mask_new, axis=1)
        z_parity = jnp.insert(z_parity, idcs_theta + 1, z_parity_new, axis=1)

    # De-duplicate identical images by adding a tiny perturbation
    z_flat = z.reshape(-1)
    _, ix = jnp.unique(z_flat, return_index=True, size=len(z_flat))
    mask_dup = jnp.full(z_flat.shape, True)
    mask_dup = mask_dup.at[ix].set(False).reshape(z.shape)

    z = jnp.where(
        mask_dup,
        z + random.uniform(key, shape=z.shape, minval=-1e-9, maxval=1e-9),
        z,
    )

    z, z_mask, z_parity = _permute_images(z, z_mask, z_parity)

    return z, z_mask, z_parity
