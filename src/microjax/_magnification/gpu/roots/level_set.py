"""Construct exact Fourier source-boundary level sets on polar rings.

For binary and triple lenses, the source-limb membership test can be written as
a finite real trigonometric polynomial on every image-centred polar ring. This
module constructs those polynomials without evaluating the singular rational
lens equation at the lens positions.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray

BINARY_FOURIER_DEGREE = 3
TRIPLE_FOURIER_DEGREE = 4


class FourierLevelSet(NamedTuple):
    """Normalized positive-frequency coefficients and their error padding."""

    coefficients: Array
    padding: Array
    degenerate: Array


def binary_level_set(
    z_cm: Array,
    w_center_shifted: complex,
    rho: float,
    shifted: float,
    *,
    a: float,
    e1: float,
) -> Array:
    """Evaluate the finite binary-lens source membership level set.

    ``z_cm`` is expressed in the centre-of-mass image-plane frame used by the
    inverse-ray grid.  The binary lens equation itself uses the midpoint frame,
    hence ``z = z_cm - shifted``.  The returned value has the same sign as
    ``abs(lens_eq(z) - w_center_shifted)**2 - rho**2`` away from the lenses,
    but remains finite at both lens positions.
    """

    z = z_cm - shifted
    zbar = jnp.conjugate(z)
    d_plus = zbar - a
    d_minus = zbar + a
    denominator = d_plus * d_minus
    numerator = (z - w_center_shifted) * denominator - e1 * d_minus - (1.0 - e1) * d_plus
    return jnp.real(numerator * jnp.conjugate(numerator) - rho**2 * denominator * jnp.conjugate(denominator))


def binary_level_set_fourier(
    r: float,
    w_center_shifted: complex,
    rho: float,
    shifted: float,
    *,
    a: float,
    e1: float,
    chart_center: complex = 0.0 + 0.0j,
) -> FourierLevelSet:
    """Construct the exact degree-three Fourier representation of ``H``.

    The numerator and denominator Laurent coefficients are correlated
    directly. This avoids a sampled FFT and its cancellation error for small
    sources while retaining a fixed algebraic shape set by the binary-lens
    degree rather than by an accuracy-resolution knob.
    """

    coefficients = binary_level_set_fourier_raw(
        r,
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
        chart_center=chart_center,
    )
    r = jnp.asarray(r)
    raw_scale = jnp.abs(coefficients[0]) + 2.0 * jnp.sum(jnp.abs(coefficients[1:]))
    tiny = jnp.finfo(r.dtype).tiny
    scale = jnp.maximum(raw_scale, tiny)
    coefficients = coefficients / scale
    padding = 64.0 * jnp.finfo(r.dtype).eps
    return FourierLevelSet(coefficients, padding, raw_scale == 0.0)


def binary_level_set_fourier_raw(
    r: float,
    w_center_shifted: complex,
    rho: float,
    shifted: float,
    *,
    a: float,
    e1: float,
    chart_center: complex = 0.0 + 0.0j,
) -> Array:
    """Return the unnormalised positive Fourier modes of the binary level set.

    Unlike :func:`binary_level_set_fourier`, every returned mode is a degree-at-
    most-six polynomial in ``r``.  Keeping that polynomial structure is useful
    for interval proofs in radial Bernstein form; normalising each ring would
    destroy it.  The represented real function is

    ``c[0] + 2 * real(sum(c[k] * exp(1j*k*theta), k=1..3))``.
    """

    # Write the denominator and numerator as Laurent polynomials in
    # ``u = exp(i theta)``. Their squared moduli then give the four
    # non-negative Fourier modes by a four-term correlation.  Constructing the
    # coefficients directly avoids an FFT of values formed by subtracting two
    # nearly equal positive quantities, which becomes relevant for very small
    # sources and planetary mass ratios.
    r = jnp.asarray(r)
    shifted = jnp.asarray(shifted, dtype=r.dtype)
    rho = jnp.asarray(rho, dtype=r.dtype)
    complex_dtype = jnp.result_type(w_center_shifted, 1j * r)
    chart_center = jnp.asarray(chart_center, dtype=complex_dtype)
    midpoint_offset = chart_center - shifted
    conjugate_offset = jnp.conjugate(midpoint_offset)
    source_offset = midpoint_offset - jnp.asarray(w_center_shifted, dtype=complex_dtype)
    # Form the lens factors before multiplying them.  Near the low-mass lens,
    # ``conjugate_offset**2 - a**2`` subtracts two O(a**2) values to recover
    # an O(sqrt(q)) planetary scale.  The factored form retains that scale
    # directly, matching the q-aware construction used by the limb quintic.
    plus_offset = conjugate_offset - a
    minus_offset = conjugate_offset + a
    denominator = jnp.asarray(
        [
            r**2,
            r * (plus_offset + minus_offset),
            plus_offset * minus_offset,
        ],
        dtype=complex_dtype,
    )
    deflection = jnp.asarray(
        [
            r,
            e1 * minus_offset + (1.0 - e1) * plus_offset,
        ],
        dtype=complex_dtype,
    )
    numerator = jnp.asarray(
        [
            source_offset * denominator[0],
            source_offset * denominator[1]
            + r * denominator[0]
            - deflection[0],
            source_offset * denominator[2]
            + r * denominator[1]
            - deflection[1],
            r * denominator[2],
        ],
        dtype=complex_dtype,
    )

    def positive_mode(values, mode):
        paired = values[: values.size - mode]
        return jnp.sum(values[mode:] * jnp.conjugate(paired))

    numerator_modes = jnp.stack([positive_mode(numerator, mode) for mode in range(4)])
    denominator_modes = jnp.stack([positive_mode(denominator, mode) if mode < 3 else 0.0j for mode in range(4)])
    return numerator_modes - rho**2 * denominator_modes


def triple_level_set(
    z_cm: Array,
    w_center_shifted: complex,
    rho: float,
    shifted: complex,
    *,
    a: float,
    e1: float,
    e2: float,
    r3_complex: complex,
) -> Array:
    """Evaluate the denominator-cleared triple-lens source level set.

    The public polar ring ``z_cm`` is in the first-two-lens centre-of-mass
    frame, while ``w_center_shifted`` and the lens positions use their midpoint
    frame. The mass fractions at ``+a``, ``-a``, and ``r3_complex`` are
    ``e1``, ``e2``, and ``1-e1-e2`` respectively.
    """

    z = z_cm - shifted
    zbar = jnp.conjugate(z)
    d_plus = zbar - a
    d_minus = zbar + a
    d_third = zbar - jnp.conjugate(r3_complex)
    denominator = d_plus * d_minus * d_third
    e3 = 1.0 - e1 - e2
    numerator = (
        (z - w_center_shifted) * denominator - e1 * d_minus * d_third - e2 * d_plus * d_third - e3 * d_plus * d_minus
    )
    return jnp.real(numerator * jnp.conjugate(numerator) - rho**2 * denominator * jnp.conjugate(denominator))


def triple_level_set_fourier(
    r: float,
    w_center_shifted: complex,
    rho: float,
    shifted: complex,
    *,
    a: float,
    e1: float,
    e2: float,
    r3_complex: complex,
    chart_center: complex = 0.0 + 0.0j,
) -> FourierLevelSet:
    """Construct the exact degree-four Fourier representation of ``H``.

    Write the three denominator factors as polynomials in ``u**-1``, where
    ``u = exp(i theta)``. The denominator then has four coefficients and the
    lens-equation numerator has five, spanning powers ``u**-3`` through
    ``u**1``. Their short autocorrelations give all five non-negative Fourier
    modes directly, avoiding angular sampling and an FFT at every radial node.
    """

    r = jnp.asarray(r)
    rho = jnp.asarray(rho, dtype=r.dtype)
    complex_dtype = jnp.result_type(w_center_shifted, 1j * r)
    midpoint_offset = jnp.asarray(chart_center, dtype=complex_dtype) - jnp.asarray(shifted, dtype=complex_dtype)
    conjugate_offset = jnp.conjugate(midpoint_offset)
    source_offset = midpoint_offset - jnp.asarray(w_center_shifted, dtype=complex_dtype)
    third_offset = conjugate_offset - jnp.conjugate(jnp.asarray(r3_complex, dtype=complex_dtype))
    plus_offset = conjugate_offset - a
    minus_offset = conjugate_offset + a

    pair_plus_third = jnp.asarray(
        [r**2, r * (plus_offset + third_offset), plus_offset * third_offset], dtype=complex_dtype
    )
    pair_minus_third = jnp.asarray(
        [r**2, r * (minus_offset + third_offset), minus_offset * third_offset], dtype=complex_dtype
    )
    pair_plus_minus = jnp.asarray(
        [r**2, r * (plus_offset + minus_offset), plus_offset * minus_offset], dtype=complex_dtype
    )
    denominator = jnp.asarray(
        [
            r**3,
            r**2 * (plus_offset + minus_offset + third_offset),
            r * (plus_offset * minus_offset + plus_offset * third_offset + minus_offset * third_offset),
            plus_offset * minus_offset * third_offset,
        ],
        dtype=complex_dtype,
    )
    e3 = 1.0 - e1 - e2
    deflection_numerator = e1 * pair_minus_third + e2 * pair_plus_third + e3 * pair_plus_minus
    numerator = jnp.asarray(
        [
            source_offset * denominator[0],
            source_offset * denominator[1] + r * denominator[0] - deflection_numerator[0],
            source_offset * denominator[2] + r * denominator[1] - deflection_numerator[1],
            source_offset * denominator[3] + r * denominator[2] - deflection_numerator[2],
            r * denominator[3],
        ],
        dtype=complex_dtype,
    )

    def positive_mode(values, mode):
        return jnp.sum(values[mode:] * jnp.conjugate(values[: values.size - mode]))

    numerator_modes = jnp.stack([positive_mode(numerator, mode) for mode in range(TRIPLE_FOURIER_DEGREE + 1)])
    denominator_modes = jnp.stack(
        [
            positive_mode(denominator, mode) if mode < denominator.size else 0.0j
            for mode in range(TRIPLE_FOURIER_DEGREE + 1)
        ]
    )
    coefficients = numerator_modes - rho**2 * denominator_modes
    raw_scale = jnp.abs(coefficients[0]) + 2.0 * jnp.sum(jnp.abs(coefficients[1:]))
    scale = jnp.maximum(raw_scale, jnp.finfo(r.dtype).tiny)
    # The triple construction has more fixed multiply-add stages than the
    # binary correlation. Two binary-sized roundoff allowances also cover the
    # discarded-mode floor measured by the former 20-point FFT construction.
    padding = 128.0 * jnp.finfo(r.dtype).eps
    return FourierLevelSet(coefficients / scale, padding, raw_scale == 0.0)
