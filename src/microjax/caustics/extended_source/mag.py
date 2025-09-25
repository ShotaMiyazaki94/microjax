# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""High-level finite-source magnification evaluation.

This module combines the lower-level helpers to deliver the public
``mag_extended_source`` routine.  The implementation follows the standard
image-centred ray-shooting pipeline: shift the coordinate frame to the centre
of mass, trace images along the source limb, assemble closed contours (handling
caustic crossings when necessary), and evaluate the relevant surface integrals
for uniform or limb-darkened intensity profiles.
"""

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap

from ...utils import last_nonzero

from ..integrate import _integrate_unif, _integrate_ld

from .limb import _images_of_source_limb
from .segments import _get_segments
from .contours import _contours_from_closed_segments, _contours_from_open_segments


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts_limb",
        "limb_darkening",
        "npts_ld",
    ),
)
def mag_extended_source(
    w0,
    rho,
    nlenses=2,
    npts_limb=150,
    limb_darkening=False,
    u1=0.0,
    npts_ld=100,
    **params,
):
    """Compute the magnification of a finite-sized source for up to 3 lenses.

    Parameters
    ----------
    w0 : complex
        Source-centre position in the complex plane.
    rho : float
        Source radius in Einstein units.
    nlenses : int, optional
        Lens multiplicity (1–3).  Controls the point-source solver and contour
        assembly heuristics.
    npts_limb : int, optional
        Base number of sampling points along the source limb before adaptive
        refinement.
    limb_darkening : bool, optional
        If ``True`` evaluate surface integrals using the linear limb-darkening
        profile with coefficient ``u1``.
    u1 : float, optional
        Linear limb-darkening coefficient.
    npts_ld : int, optional
        Number of quadrature points for the Dominik (1998) P/Q integrals when
        limb darkening is enabled.
    **params : dict
        Additional lens parameters (e.g. ``s``, ``q``, ``q3``) forwarded to the
        point-source solver.

    Returns
    -------
    float
        Total magnification for the extended source at ``w0``.
    """
    if nlenses == 1:
        _params = {}
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"a": a, "e1": e1}
        x_cm = a * (1 - q) / (1 + q)
        w0 -= x_cm
    elif nlenses == 3:
        s, q, q3 = params["s"], params["q"], params["q3"]
        r3, psi = params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = (1 - q3) / (1.0 + q + q3)
        _params = {"a": a, "r3": r3, "psi": psi, "e1": e1, "e2": e2}
        x_cm = a * (1 - q) / (1 + q)
        w0 -= x_cm
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    z, z_mask, z_parity = _images_of_source_limb(
        w0,
        rho,
        nlenses=nlenses,
        npts=npts_limb,
        **_params,
    )

    if limb_darkening:
        integrate = lambda contour, tidx: _integrate_ld(
            contour,
            tidx,
            w0,
            rho,
            u1=u1,
            nlenses=nlenses,
            npts=npts_ld,
            **_params,
        )
    else:
        integrate = lambda contour, tidx: _integrate_unif(contour, tidx)

    if nlenses == 1:
        contours, contours_p = _contours_from_closed_segments(
            jnp.moveaxis(jnp.stack([z, z_parity]), 0, 1)
        )
        tail_idcs = jnp.array([z.shape[1] - 1, z.shape[1] - 1])
        I = vmap(integrate)(contours, tail_idcs)
        return jnp.abs(jnp.sum(I * contours_p)) / (np.pi * rho**2)

    elif (nlenses == 2) or (nlenses == 3):
        max_nr_of_contours = 3
        segments_closed, segments_open, all_closed = _get_segments(
            z, z_mask, z_parity, nlenses=nlenses
        )

        contours1, contours_p1 = _contours_from_closed_segments(segments_closed)
        tail_idcs = jnp.repeat(contours1.shape[1] - 1, contours1.shape[0])
        I1 = vmap(integrate)(contours1, tail_idcs)
        mags1 = I1 * contours_p1

        branch1 = lambda _: jnp.zeros(max_nr_of_contours)

        def branch2(segments):
            contours, contours_p = _contours_from_open_segments(
                segments, max_nr_of_contours=max_nr_of_contours
            )
            tail_idcs = vmap(last_nonzero)(contours.real)
            I = vmap(integrate)(contours, tail_idcs)
            return I * contours_p

        mags2 = lax.cond(all_closed, branch1, branch2, segments_open)
        mag = jnp.abs(mags1.sum() + mags2.sum()) / (np.pi * rho**2)

        return mag

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")
