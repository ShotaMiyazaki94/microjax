"""Stability of benchmark summaries under stratified configuration resampling.

These percentile intervals describe the empirical resampling distribution. A
summary of the entire fixed reference grid has no finite-grid sampling error:
its intervals describe resampling stability, not uncertainty of that census.
They do not cover selection bias, reference error, or timing-repeat noise.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np


def _weighted_quantiles(sorted_values, sorted_configurations, multiplicities, quantile):
    """Type-7 quantiles of explicitly repeated clusters without expanding rows."""
    cumulative = np.cumsum(multiplicities[:, sorted_configurations], axis=1, dtype=np.int64)
    totals = cumulative[:, -1]
    positions = np.maximum(totals - 1, 0) * quantile
    lower_rank = np.floor(positions).astype(np.int64)
    upper_rank = np.ceil(positions).astype(np.int64)
    lower = sorted_values[np.argmax(cumulative > lower_rank[:, None], axis=1)]
    upper = sorted_values[np.argmax(cumulative > upper_rank[:, None], axis=1)]
    fraction = positions - lower_rank
    # Match NumPy's linear/type-7 interpolation, including its stable upper-half form.
    difference = upper - lower
    estimates = np.where(fraction >= 0.5, upper - difference * (1 - fraction), lower + difference * fraction)
    return np.where(totals > 0, estimates, np.nan)


def estimate(
    values,
    config_ids,
    strata,
    quantile=0.5,
    log=False,
    min_configs=20,
    min_points=0,
    repeats=4000,
    seed=20261106,
    frame_config_ids=None,
    frame_strata=None,
):
    """Estimate a pooled quantile and stratified configuration-bootstrap intervals.

    Each configuration must belong to one stratum. Within each stratum, each
    bootstrap replicate draws its original number of configurations with
    replacement. A selected configuration carries all its input rows, including
    repeated copies when selected more than once. Accuracy therefore remains a
    pooled point quantile; callers pass one row per configuration for speed or
    runtime. ``log=True`` computes the quantile in log10 space and returns its
    estimate and intervals in the original units.

    For a regional estimate, pass the aligned, unfiltered ``frame_config_ids``
    and ``frame_strata``. The frame may repeat configurations, but must contain
    every input configuration with the same stratum. Bootstrap draws then
    include configurations with no rows in the region, allowing its membership
    and point count to vary. Without a frame, only input configurations are
    resampled. For matched timing comparisons the frame should contain only
    matched configurations, preserving that conditional comparison.
    Strata with no contributing input configurations are skipped: all of their
    draws have zero weight in this statistic. Every frame configuration within
    a contributing stratum remains eligible for resampling.

    Quantiles use NumPy's linear/type-7 definition. Interval endpoints are the
    2.5/97.5 and 15.8655/84.1345 percentiles of bootstrap estimates (95% and
    68.2689% nominal levels). Standard errors are sample standard deviations of those
    estimates, not the dispersion of the original observations. The log10
    standard error is omitted whenever any input is zero.

    The fixed seed and PCG64 generator make results reproducible for the same
    inputs and NumPy implementation. Chunked multiplicity-weighted order
    statistics avoid materializing thousands of repeated point arrays.
    Insufficient samples retain their point estimate but return null intervals
    and standard errors. Empty input returns a null point estimate as well.
    Draws containing no regional rows are excluded and counted; intervals are
    suppressed if fewer than 99% of draws contain regional rows. When no
    bootstrap is run, ``n_valid_resamples`` is zero and its fraction is null.
    """
    values = np.asarray(values, dtype=float)
    config_ids = np.asarray(config_ids)
    strata = np.asarray(strata)
    if values.ndim != 1 or config_ids.shape != values.shape or strata.shape != values.shape:
        raise ValueError("values, config_ids and strata must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("benchmark values must be finite and nonnegative")
    if log and np.any(values <= 0):
        raise ValueError("log-space estimates require strictly positive values")
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    for name, value, minimum in (
        ("min_configs", min_configs, 1),
        ("min_points", min_points, 0),
        ("repeats", repeats, 2),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    configurations, first, configuration = np.unique(config_ids, return_index=True, return_inverse=True)
    configuration_strata = strata[first]
    if np.any(strata != configuration_strata[configuration]):
        raise ValueError("each configuration must belong to exactly one stratum")
    if (frame_config_ids is None) != (frame_strata is None):
        raise ValueError("frame_config_ids and frame_strata must be supplied together")
    if frame_config_ids is None:
        frame_configurations, frame_configuration_strata = configurations, configuration_strata
    else:
        frame_config_ids, frame_strata = np.asarray(frame_config_ids), np.asarray(frame_strata)
        if frame_config_ids.ndim != 1 or frame_strata.shape != frame_config_ids.shape:
            raise ValueError("frame_config_ids and frame_strata must be aligned one-dimensional arrays")
        frame_configurations, frame_first, frame_inverse = np.unique(
            frame_config_ids, return_index=True, return_inverse=True
        )
        frame_configuration_strata = frame_strata[frame_first]
        if np.any(frame_strata != frame_configuration_strata[frame_inverse]):
            raise ValueError("each frame configuration must belong to exactly one stratum")
        positions = np.searchsorted(frame_configurations, configurations)
        if np.any(positions >= frame_configurations.size) or not np.array_equal(
            frame_configurations[positions], configurations
        ):
            raise ValueError("every input configuration must belong to the sampling frame")
        if np.any(configuration_strata != frame_configuration_strata[positions]):
            raise ValueError("input and frame strata must agree for each configuration")
        configuration = positions[configuration]
    transformed = np.log10(values) if log else values
    central = np.quantile(transformed, quantile) if values.size else None
    result = {
        "n_points": int(values.size),
        "n_configurations": int(first.size),
        "estimate": None if central is None else float(10**central if log else central),
        "standard_error": None,
        "log10_standard_error": None,
        "ci68": None,
        "ci95": None,
        "n_valid_resamples": 0,
        "valid_resample_fraction": None,
        "eligible": bool(first.size >= min_configs and values.size >= min_points),
    }
    if not result["eligible"]:
        return result

    # Zero-contribution strata cannot change this regional statistic. Keep all
    # configurations within the remaining strata, including zero-contribution
    # configurations, so regional membership can still vary across draws.
    contributing = np.isin(frame_configuration_strata, configuration_strata)
    configuration = np.cumsum(contributing)[configuration] - 1
    frame_configurations = frame_configurations[contributing]
    frame_configuration_strata = frame_configuration_strata[contributing]
    groups = [
        np.flatnonzero(frame_configuration_strata == stratum) for stratum in np.unique(frame_configuration_strata)
    ]
    order = np.argsort(transformed, kind="stable")
    ordered_values, ordered_configuration = transformed[order], configuration[order]
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty(repeats)
    # Bound the largest row-weight/cumulative array to approximately one million entries.
    chunk = min(128, max(1, 1_000_000 // values.size))
    for start in range(0, repeats, chunk):
        size = min(chunk, repeats - start)
        counts = np.zeros((size, frame_configurations.size), dtype=np.int32)
        rows = np.arange(size)[:, None]
        for group in groups:
            draws = rng.choice(group, size=(size, group.size), replace=True)
            np.add.at(counts, (rows, draws), 1)
        bootstrap[start : start + size] = _weighted_quantiles(ordered_values, ordered_configuration, counts, quantile)
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    result.update(n_valid_resamples=int(bootstrap.size), valid_resample_fraction=float(bootstrap.size / repeats))
    if bootstrap.size < 2 or bootstrap.size / repeats < 0.99:
        result["eligible"] = False
        return result
    samples = 10**bootstrap if log else bootstrap
    one_sigma_tail = (1 - erf(1 / sqrt(2))) / 2
    low95, low68, high68, high95 = np.quantile(samples, [0.025, one_sigma_tail, 1 - one_sigma_tail, 0.975])
    result.update(
        standard_error=float(np.std(samples, ddof=1)),
        log10_standard_error=(
            float(np.std(bootstrap if log else np.log10(samples), ddof=1)) if np.all(values > 0) else None
        ),
        ci68=[float(low68), float(high68)],
        ci95=[float(low95), float(high95)],
    )
    return result
