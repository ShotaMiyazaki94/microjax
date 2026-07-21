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
BINARY_FOURIER_SAMPLES = 16
TRIPLE_FOURIER_DEGREE = 4
TRIPLE_FOURIER_SAMPLES = 20


class FourierLevelSet(NamedTuple):
    """Normalized positive-frequency coefficients and their error padding."""

    coefficients: Array
    padding: Array
    degenerate: Array


def _normalized_fourier_level_set(samples: Array, degree: int) -> FourierLevelSet:
    """Normalize exact-bandwidth samples and bound discarded roundoff."""

    sample_count = samples.shape[0]
    spectrum = jnp.fft.fft(samples) / sample_count
    coefficients = spectrum[: degree + 1]
    raw_scale = jnp.abs(coefficients[0]) + 2.0 * jnp.sum(jnp.abs(coefficients[1:]))
    tiny = jnp.finfo(samples.dtype).tiny
    scale = jnp.maximum(raw_scale, tiny)
    coefficients = coefficients / scale
    high_modes = spectrum[degree + 1 : sample_count - degree]
    roundoff = 64.0 * jnp.finfo(samples.dtype).eps
    padding = jnp.sum(jnp.abs(high_modes)) / scale + roundoff
    return FourierLevelSet(coefficients, padding, raw_scale == 0.0)


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
    denominator = jnp.asarray(
        [
            r**2,
            2.0 * r * conjugate_offset,
            conjugate_offset**2 - a**2,
        ],
        dtype=complex_dtype,
    )
    numerator = jnp.asarray(
        [
            source_offset * r**2,
            r**3 + 2.0 * source_offset * r * conjugate_offset - r,
            source_offset * (conjugate_offset**2 - a**2)
            + 2.0 * r**2 * conjugate_offset
            - conjugate_offset
            + a * (1.0 - 2.0 * e1),
            r * (conjugate_offset**2 - a**2),
        ],
        dtype=complex_dtype,
    )

    def positive_mode(values, mode):
        paired = values[: values.size - mode]
        return jnp.sum(values[mode:] * jnp.conjugate(paired))

    numerator_modes = jnp.stack([positive_mode(numerator, mode) for mode in range(4)])
    denominator_modes = jnp.stack([positive_mode(denominator, mode) if mode < 3 else 0.0j for mode in range(4)])
    coefficients = numerator_modes - rho**2 * denominator_modes
    raw_scale = jnp.abs(coefficients[0]) + 2.0 * jnp.sum(jnp.abs(coefficients[1:]))
    tiny = jnp.finfo(r.dtype).tiny
    scale = jnp.maximum(raw_scale, tiny)
    coefficients = coefficients / scale
    padding = 64.0 * jnp.finfo(r.dtype).eps
    return FourierLevelSet(coefficients, padding, raw_scale == 0.0)


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
    """Recover the exact degree-four Fourier representation on one ring."""

    real_dtype = jnp.asarray(r).dtype
    angles = 2.0 * jnp.pi * jnp.arange(TRIPLE_FOURIER_SAMPLES, dtype=real_dtype) / TRIPLE_FOURIER_SAMPLES
    complex_dtype = jnp.result_type(w_center_shifted, 1j * r)
    center = jnp.asarray(chart_center, dtype=complex_dtype)
    z_cm = center + r * jnp.exp(1j * angles)
    samples = triple_level_set(
        z_cm,
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
        e2=e2,
        r3_complex=r3_complex,
    )
    return _normalized_fourier_level_set(samples, TRIPLE_FOURIER_DEGREE)
