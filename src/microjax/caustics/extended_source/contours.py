# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Helpers to convert image segments into closed contours."""

import jax.numpy as jnp
from jax import vmap

from ...utils import last_nonzero

from .segments import _merge_open_segments


def _contours_from_closed_segments(segments):
    """Convert closed segments into explicit contours with parity labels.

    Parameters
    ----------
    segments : array_like
        Array of shape ``(n_segments, 2, n_points)`` where the first row holds
        complex coordinates and the second row contains parity values.

    Returns
    -------
    tuple
        ``(contours, contours_p)`` where ``contours`` appends the starting point
        to close the polygon explicitly and ``contours_p`` stores the parity per
        contour.
    """
    contours_p = segments[:, 1, 0].real

    contours = segments[:, 0]
    contours = jnp.hstack([contours, contours[:, 0][:, None]])

    return contours, contours_p


def _contours_from_open_segments(
    segments,
    max_nr_of_contours=3,
    max_nr_of_segments_in_contour=20,
):
    """Merge open segments and expose the resulting contours and parity.

    Parameters
    ----------
    segments : array_like
        Collection of open segment candidates (shape ``(n_segments, 2, n_points)``).
    max_nr_of_contours : int, optional
        Maximum number of contours to produce.
    max_nr_of_segments_in_contour : int, optional
        Maximal number of merge steps per contour.

    Returns
    -------
    tuple
        ``(contours, contours_p)`` similar to :func:`_contours_from_closed_segments`.
    """

    segments_merged = _merge_open_segments(
        segments,
        max_nr_of_contours=max_nr_of_contours,
        max_nr_of_segments_in_contour=max_nr_of_segments_in_contour,
    )

    contours_p = segments_merged[:, 1, 0].real
    contours = segments_merged[:, 0]
    tail_idcs = vmap(last_nonzero)(contours.real)
    contours = vmap(lambda idx, c: c.at[idx + 1].set(c[0]))(tail_idcs, contours)

    return contours, contours_p
