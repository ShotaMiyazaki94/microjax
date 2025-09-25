# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Segment construction and merging utilities for extended-source contours."""

import jax.numpy as jnp
from jax import lax, vmap

from ...utils import (
    first_nonzero,
    first_zero,
    last_nonzero,
    sparse_argsort,
)


def _split_segment(segment, n_parts=5):
    """Split a raw image track into sub-segments with uniform properties.

    Each row produced by ``_images_of_source_limb`` can contain jumps where the
    solver switches between real/ghost images or between parity branches.  For
    contour stitching we divide such rows whenever:

    * the mask toggles between real and complex images,
    * the parity changes sign, or
    * consecutive points are separated by more than 0.1 Einstein radii.

    Parameters
    ----------
    segment : array_like
        Array with shape ``(3, n_points)`` holding positions, parity, and mask
        for a single image track.
    n_parts : int, optional
        Maximum number of sub-segments returned; unused slots are filled with
        zeros.

    Returns
    -------
    array_like
        Stack of up to ``n_parts`` segments with incompatible regions nulled
        out, ready for further filtering.
    """
    z, z_parity, z_mask = segment
    npts = len(z)

    z_diff = z[1:] - z[:-1]
    diff_dist_mask = z_diff.real**2 + z_diff.imag**2 > 0.1**2
    diff_parity = z_parity[1:].real - z_parity[:-1].real
    diff_mask = z_mask[1:].real - z_mask[:-1].real
    changepoints = (diff_dist_mask != False) | (diff_parity != 0) | (diff_mask != False)

    changepoints = jnp.concatenate([
        jnp.array([jnp.where(z_mask[0] == 0, False, True)]),
        changepoints,
        jnp.array([jnp.where(z_mask[-1] == 0, False, True)]),
    ])
    diff_mask = jnp.concatenate([
        jnp.array([jnp.where(z_mask[0] == 0, 0.0, 1.0)]),
        diff_mask,
        jnp.array([jnp.where(z_mask[-1] == 0, 0.0, -1.0)]),
    ])

    idcs_start = jnp.where(
        changepoints & (diff_mask >= 0),
        changepoints,
        jnp.zeros_like(changepoints),
    )
    idcs_end = jnp.where(
        changepoints & (diff_mask <= 0),
        changepoints,
        jnp.zeros_like(changepoints),
    )

    idcs_start = jnp.argwhere(idcs_start, size=2 * n_parts)[:, 0]
    idcs_end = jnp.argwhere(idcs_end, size=2 * n_parts)[:, 0]

    n = jnp.arange(npts)

    def mask_region(_, xs):
        l, r = xs
        mask = jnp.where(
            (l == 0.0) & (r == 0.0),
            jnp.zeros(npts),
            (n >= l) & (n < r),
        )
        return 0, mask

    _, masks = lax.scan(mask_region, 0, (idcs_start, idcs_end))
    segments_split = vmap(lambda mask: segment * mask)(masks)
    return segments_split


def _process_segments(segments, nr_of_segments=20):
    """Canonicalise open segments before attempting to merge them.

    This helper splits every candidate using :func:`_split_segment`, removes
    degenerate pieces with fewer than two valid points, sorts non-empty entries
    to the front, and rotates each track so the head is positioned at index 0.

    Parameters
    ----------
    segments : array_like
        Array of shape ``(n_segments, 3, n_points)`` containing raw open
        segments.
    nr_of_segments : int, optional
        Upper bound on the number of segments kept after padding/truncation.

    Returns
    -------
    array_like
        Cleaned array with shape ``(nr_of_segments, 3, n_points)`` suitable for
        the merging routine.
    """
    segments = jnp.concatenate(vmap(_split_segment)(segments))

    mask = jnp.sum((segments[:, 0] != 0 + 0j).astype(int), axis=1) < 2
    segments = segments * (~mask[:, None, None])

    sorted_idcs = jnp.argsort(jnp.any(jnp.abs(segments[:, 0, :]) > 0.0, axis=1))[::-1]
    segments = segments[sorted_idcs]
    segments = segments[:nr_of_segments, :, :]

    head_idcs = vmap(first_nonzero)(jnp.abs(segments[:, 0, :]))

    segments = vmap(lambda segment, head_idx: jnp.roll(segment, -head_idx, axis=-1))(
        segments,
        head_idcs,
    )

    return segments


def _get_segments(z, z_mask, z_parity, nlenses=2):
    """Separate image tracks into closed contours and open segments.

    Parameters
    ----------
    z : array_like
        Complex array of shape ``(n_images, n_samples)`` with image positions.
    z_mask : array_like
        Boolean array indicating which entries correspond to physical images.
    z_parity : array_like
        Parity values (sign of the Jacobian) for each image sample.
    nlenses : int, optional
        Lens multiplicity, used to derive heuristic limits on the number of
        segments.

    Returns
    -------
    tuple
        ``(segments_closed, segments_open, all_closed)`` where closed segments
        already form loops and open segments still need merging.
    """
    n_images = nlenses**2 + 1
    nr_of_segments = 3 * n_images

    z = z * z_mask
    z_parity = z_parity * z_mask
    segments = jnp.stack([z, z_parity, z_mask])
    segments = jnp.moveaxis(segments, 0, 1)

    mask_closed = (jnp.abs((z[:, 0] - z[:, -1])) < 1e-5) & jnp.all(z_mask, axis=1)

    segments_closed = segments * mask_closed[:, None, None]
    segments_open = segments * (~mask_closed[:, None, None])

    all_closed = jnp.all(mask_closed)
    segments_open = lax.cond(
        all_closed,
        lambda s: jnp.pad(s, ((0, nr_of_segments - s.shape[0]), (0, 0), (0, 0))),
        lambda s: _process_segments(s, nr_of_segments=nr_of_segments),
        segments_open,
    )

    segments_closed = segments_closed[:, :2, :]
    segments_open = segments_open[:, :2, :]

    return segments_closed, segments_open, all_closed


def _concatenate_segments(segment_first, segment_second):
    """Concatenate two segment buffers while respecting their logical length."""
    segment_first_length = first_zero(jnp.abs(segment_first[0]))
    return segment_first + jnp.roll(segment_second, segment_first_length, axis=-1)


def _get_segment_length(segment, tail_idx):
    """Compute the chord-length of a segment up to (and including) its tail."""
    diff = jnp.diff(segment[0])
    diff = diff.at[tail_idx].set(0.0)
    return jnp.abs(diff).sum()


def _connection_condition(
    seg1, seg2, tidx1, tidx2, ctype, min_dist=1e-05, max_dist=1e-01, max_ang=60.0
):
    """Check whether two segments can be stitched according to the heuristic.

    Four connection types are considered (tail/head combinations).  A merge is
    permitted when parity matches the expectation for that type and the
    resulting polygon would be geometrically smooth—i.e. endpoints are
    sufficiently close, the turn angle does not exceed ``max_ang``, and the
    join reduces the gap between segment endpoints.

    Returns
    -------
    bool
        ``True`` if the candidate pair satisfies all connection criteria.
    """

    def get_segment_head(seg, t):
        x = seg[0]
        cond = (jnp.abs(x[1] - x[0]) > 1e-5) | (t <= 1)
        line = lax.cond(
            cond,
            lambda: x[:2],
            lambda: x[1:3],
        )
        return line[::-1]

    def get_segment_tail(seg, t):
        x = seg[0]
        cond = (jnp.abs(x[t] - x[t - 1]) > 1e-5) | (t <= 1)
        line = lax.cond(
            cond,
            lambda: lax.dynamic_slice(x, (t - 1,), (2,)),
            lambda: lax.dynamic_slice(x, (t - 2,), (2,)),
        )
        return line

    same_parity = seg1[1, 0].real * seg2[1, 0].real > 0.0
    conds_parity = jnp.stack(
        [
            same_parity,
            same_parity,
            ~same_parity,
            ~same_parity,
        ]
    )
    cond_parity = conds_parity[ctype]

    line1, line2 = lax.switch(
        ctype,
        [
            lambda s1, s2, t1, t2: (get_segment_tail(s1, t1), get_segment_head(s2, t2)),
            lambda s1, s2, t1, t2: (get_segment_head(s1, t1), get_segment_tail(s2, t2)),
            lambda s1, s2, t1, t2: (get_segment_head(s1, t1), get_segment_head(s2, t2)),
            lambda s1, s2, t1, t2: (get_segment_tail(s1, t1), get_segment_tail(s2, t2)),
        ],
        seg1,
        seg2,
        tidx1,
        tidx2,
    )

    dist = jnp.abs(line1[1] - line2[1])
    cond1 = dist < max_dist

    vec1 = (line1[1] - line1[0]) / jnp.abs(line1[1] - line1[0])
    vec2 = (line2[1] - line2[0]) / jnp.abs(line2[1] - line2[0])
    alpha = jnp.arccos(jnp.real(vec1) * jnp.real(vec2) + jnp.imag(vec1) * jnp.imag(vec2))
    cond2 = (180.0 - jnp.rad2deg(alpha)) < max_ang

    cond3 = jnp.abs(line1[1] - line2[1]) < jnp.abs(line1[0] - line2[0])

    cond_geom = jnp.logical_or(cond1 & cond2 & cond3, dist < min_dist)

    return cond_parity & cond_geom


def _merge_two_segments(seg1, seg2, tidx1, tidx2, ctype):
    """Merge two segments according to the requested connection type."""

    def hh_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1]
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg2 = seg2.at[1].set(-1 * seg2[1])
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1

    def tt_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1]
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg2 = seg2.at[1].set(-1 * seg2[1])
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1

    def th_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1

    def ht_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1

    seg_merged, tidx_merged = lax.switch(
        ctype,
        [th_connection, ht_connection, hh_connection, tt_connection],
        seg1,
        seg2,
        tidx1,
        tidx2,
    )
    return seg_merged, tidx_merged


def _merge_open_segments(
    segments,
    max_nr_of_contours=3,
    max_nr_of_segments_in_contour=10,
):
    """Sequentially merge open segments into closed contours.

    The algorithm iteratively selects the shortest remaining segment as the
    active contour, searches for nearby candidates under all four connection
    types, evaluates :func:`_connection_condition`, and merges the best
    candidate when a valid match exists.  The process repeats until the active
    segment closes or no suitable partner remains, after which we proceed to
    the next contour.

    Parameters
    ----------
    segments : array_like
        Array of shape ``(n_segments, 2, n_points)`` with open segments.
    max_nr_of_contours : int, optional
        Maximum number of closed contours to assemble.
    max_nr_of_segments_in_contour : int, optional
        Limit on the number of merge attempts per contour to avoid infinite
        loops in pathological cases.

    Returns
    -------
    array_like
        Stack of merged contour buffers, each occupying one row.
    """

    def merge_with_another_segment(seg_active, tidx_active, segments, tidcs):
        # Compute all T-H, H-T, H-H, T-T distances between the active segment
        # and candidate segments
        dist_th = jnp.abs(seg_active[0, tidx_active] - segments[:, 0, 0])
        dist_ht = vmap(lambda seg, tidx: jnp.abs(seg_active[0, 0] - seg[0, tidx]))(
            segments, tidcs
        )
        dist_hh = jnp.abs(seg_active[0, 0] - segments[:, 0, 0])
        dist_tt = vmap(
            lambda seg, tidx: jnp.abs(seg_active[0, tidx_active] - seg[0, tidx])
        )(segments, tidcs)

        distances = jnp.stack([dist_th, dist_ht, dist_hh, dist_tt])

        ctype1, idx1 = jnp.unravel_index(jnp.argsort(distances.reshape(-1))[0], distances.shape)
        ctype2, idx2 = jnp.unravel_index(jnp.argsort(distances.reshape(-1))[1], distances.shape)
        ctype3, idx3 = jnp.unravel_index(jnp.argsort(distances.reshape(-1))[2], distances.shape)
        ctype4, idx4 = jnp.unravel_index(jnp.argsort(distances.reshape(-1))[3], distances.shape)

        success1 = _connection_condition(
            seg_active, segments[idx1], tidx_active, tidcs[idx1], ctype1
        )
        success2 = _connection_condition(
            seg_active, segments[idx2], tidx_active, tidcs[idx2], ctype2
        )
        success3 = _connection_condition(
            seg_active, segments[idx3], tidx_active, tidcs[idx3], ctype3
        )
        success4 = _connection_condition(
            seg_active, segments[idx4], tidx_active, tidcs[idx4], ctype4
        )

        def branch1(segments, tidcs):
            idx_best = first_nonzero(
                jnp.array([success1, success2, success3, success4]).astype(float)
            )
            seg_best = jnp.stack(
                [segments[idx1], segments[idx2], segments[idx3], segments[idx4]]
            )[idx_best]
            tidx_best = jnp.stack([tidcs[idx1], tidcs[idx2], tidcs[idx3], tidcs[idx4]])[
                idx_best
            ]
            ctype = jnp.stack([ctype1, ctype2, ctype3, ctype4])[idx_best]

            seg_active_new, tidx_active_new = _merge_two_segments(
                seg_active,
                seg_best,
                tidx_active,
                tidx_best,
                ctype,
            )

            idx_seg = jnp.array([idx1, idx2, idx3, idx4])[idx_best]
            segments = segments.at[idx_seg].set(jnp.zeros_like(segments[idx_seg]))
            return seg_active_new, tidx_active_new, segments, tidcs

        def branch2(segments, tidcs):
            return seg_active, tidx_active, segments, tidcs

        return lax.cond(
            jnp.any(jnp.array([success1, success2, success3, success4])),
            branch1,
            branch2,
            segments,
            tidcs,
        )

    def body_fn(carry, _):
        seg_active, tidx_active, segments, tidcs = carry

        stopping_criterion = ~jnp.any(segments[:, 0, 0])
        seg_active, tidx_active, segments, tidcs = lax.cond(
            stopping_criterion,
            lambda: (seg_active, tidx_active, segments, tidcs),
            lambda: merge_with_another_segment(seg_active, tidx_active, segments, tidcs),
        )

        return (
            seg_active,
            tidx_active,
            segments,
            tidcs,
        ), 0.0

    tail_idcs = vmap(last_nonzero)(segments[:, 0, :].real)

    segments = jnp.pad(
        segments, ((0, 0), (0, 0), (0, 3 * segments.shape[-1])), constant_values=0.0
    )

    segments_merged_list = []

    for _ in range(max_nr_of_contours):
        segment_lengths = vmap(_get_segment_length)(segments, tail_idcs)
        _idcs = sparse_argsort(segment_lengths)
        segments, tail_idcs = segments[_idcs], tail_idcs[_idcs]

        idcs = jnp.arange(0, max_nr_of_segments_in_contour)
        init = (segments[0], tail_idcs[0], segments[1:], tail_idcs[1:])
        carry, _ = lax.scan(body_fn, init, idcs)
        seg_merged, tidx_merged, segments, tail_idcs = carry
        segments_merged_list.append(seg_merged)
        max_nr_of_segments_in_contour -= 2

    return jnp.stack(segments_merged_list)
