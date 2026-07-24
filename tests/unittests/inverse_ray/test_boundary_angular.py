import json
from dataclasses import fields
from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from microjax.inverse_ray.roots.angular import (
    ANGULAR_OK,
    ANGULAR_ROOT_FAILURE,
    _reciprocal_pair_rescue,
    angular_intervals_binary_roots,
    angular_measure_binary_roots,
    evaluate_fourier,
)
from microjax.inverse_ray.geometry.mapping import distance_from_source
from microjax.inverse_ray.geometry.lens import binary_geometry
from microjax.inverse_ray.extended_source import (
    mag_limb_dark_boundary,
    mag_radial_profile_boundary,
    mag_uniform_boundary,
)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.inverse_ray.config import BinaryMagConfig
from microjax.inverse_ray_dense.lightcurve import mag_binary_dense
from microjax.inverse_ray_retry.extended_source import mag_uniform_local_boundary
from microjax.inverse_ray_retry.lightcurve import mag_binary_safe
from microjax.inverse_ray.roots.level_set import (
    binary_level_set,
    binary_level_set_fourier,
)
from microjax.inverse_ray.geometry.limb import calc_source_limb
from microjax.inverse_ray.geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_RETRY_BREAKPOINT_CAPACITY,
    RADIAL_RETRY_INTERVAL_CAPACITY,
    RADIAL_TOLERANCE,
    RADIAL_TOPOLOGY,
    define_radial_topology,
)
from microjax.inverse_ray_retry.radial import build_local_image_charts
from microjax.inverse_ray.integrators.charts import _planetary_mixed_topology
from microjax.point_source import _images_point_source

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "inverse_ray"


def _load_fixture(name):
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _binary_setup():
    s = 1.0
    q = 1e-2
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center = 0.03 + 0.01j
    rho = 5e-3
    return s, q, a, e1, shifted, w_center, w_center - shifted, rho


def _topology_stress_setup():
    q = 1e-2
    s = 1.6
    rho = 1e-2
    w_center = 0.9456698327281224 + 0.042379329770381564j
    # VBBinaryLensing, Tol=1e-10.  This source encloses a caustic and exposed
    # an omitted radial image component in the legacy 2D histogram regions.
    vbbl = 2.8674590148757604
    return s, q, w_center, rho, vbbl


def _transient_pair_setup():
    q = 1.0
    s = 1.6
    rho = 1e-2
    w_center = 0.09137100699070808 + 0.11558584957491835j
    vbbl = 5.7098774931299685
    return s, q, w_center, rho, vbbl


def test_binary_level_set_matches_rational_membership_and_is_finite_at_lenses():
    _, _, a, e1, shifted, _, w_shifted, rho = _binary_setup()
    theta = jnp.linspace(-jnp.pi, jnp.pi, 4096, endpoint=False)
    radii = jnp.linspace(0.2, 1.8, theta.size)
    z_cm = radii * jnp.exp(1j * theta)

    level = binary_level_set(z_cm, w_shifted, rho, shifted, a=a, e1=e1)
    distances = distance_from_source(
        radii,
        theta,
        w_shifted,
        shifted,
        nlenses=2,
        a=a,
        e1=e1,
    )
    assert np.array_equal(np.asarray(level <= 0.0), np.asarray(distances <= rho))

    lens_positions_cm = jnp.asarray([shifted - a, shifted + a])
    at_lenses = binary_level_set(lens_positions_cm, w_shifted, rho, shifted, a=a, e1=e1)
    assert np.all(np.isfinite(np.asarray(at_lenses)))
    assert np.all(np.asarray(at_lenses) > 0.0)


def test_analytic_binary_fourier_coefficients_match_small_planet_level_set():
    q = 1e-4
    s = 1.6
    rho = 1e-4
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center_shifted = (0.945 + 0.003j) - shifted
    radius = 1.5997
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 257, endpoint=False)
    direct = binary_level_set(
        radius * jnp.exp(1j * theta),
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
    )
    fourier = binary_level_set_fourier(
        radius,
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
    )
    reconstructed = evaluate_fourier(fourier.coefficients, theta)
    scale = jnp.vdot(reconstructed, direct) / jnp.vdot(reconstructed, reconstructed)

    assert np.max(np.abs(np.asarray(direct - scale * reconstructed))) <= (1e-13 * np.max(np.abs(np.asarray(direct))))
    assert float(fourier.padding) == 64.0 * np.finfo(np.float64).eps


def test_analytic_binary_fourier_coefficients_match_local_image_chart():
    q = 1e-4
    s = 1.6
    rho = 1e-4
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center_shifted = (0.9788824508896057 + 0.001942618692247761j) - shifted
    chart_center = 1.59873 - 2.1e-4j
    radius = 1.7e-4
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 257, endpoint=False)
    direct = binary_level_set(
        chart_center + radius * jnp.exp(1j * theta),
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
    )
    fourier = binary_level_set_fourier(
        radius,
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
        chart_center=chart_center,
    )
    reconstructed = evaluate_fourier(fourier.coefficients, theta)
    scale = jnp.vdot(reconstructed, direct) / jnp.vdot(reconstructed, reconstructed)

    assert np.max(np.abs(np.asarray(direct - scale * reconstructed))) <= (2e-13 * np.max(np.abs(np.asarray(direct))))


def test_local_image_charts_reject_an_empty_limb_root_set():
    charts = build_local_image_charts(
        jnp.zeros((5, 8), dtype=jnp.complex128),
        jnp.zeros((5, 8), dtype=bool),
        1e-4,
        margin_r=1.0,
        shifted=0.8,
        a=0.8,
        e1=1e-4 / (1.0 + 1e-4),
    )

    assert int(charts.status) & RADIAL_CAPACITY
    assert not np.any(np.asarray(charts.active))


@pytest.mark.slow
def test_local_image_rescue_handles_caustic_enclosing_annulus():
    point = -2.845269680283363e-05 + 8.040213267373542e-05j
    reference = 15873.646704428846
    result = mag_uniform_local_boundary(
        point,
        1e-4,
        s=1.6,
        q=1e-5,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        return_info=True,
    )

    assert int(result.status) == 0
    assert abs(float(result.magnification) - reference) / reference < 1e-4


def test_nonannular_fold_pair_uses_a_guaranteed_interior_chart_origin():
    point = -7.794776723440755e-05 - 7.416540530774523e-05j
    rho = 1e-5
    q = 1e-4
    s = 1.6
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    image_limb, mask_limb = calc_source_limb(
        point,
        rho,
        500,
        nlenses=2,
        q=q,
        s=s,
        a=a,
        e1=e1,
    )
    interior_images, interior_mask = _images_point_source(point - shifted, nlenses=2, a=a, e1=e1)
    interior_images = interior_images + shifted
    charts = build_local_image_charts(
        image_limb,
        mask_limb,
        rho,
        margin_r=1.0,
        shifted=shifted,
        a=a,
        e1=e1,
        interior_images=interior_images,
        interior_mask=interior_mask,
        prefer_interior_anchor=True,
    )

    # Slots 1 and 4 are the two boundary arcs of one transient fold pair.
    # Their two source-centre images belong to the physical image, but their
    # arithmetic mean lies in the gap and maps outside the source.  The chart
    # must therefore use one guaranteed-interior image, not that mean.
    label = int(charts.branch_labels[1])
    assert label == int(charts.branch_labels[4])
    center = charts.centers[label]
    assert binary_level_set(center, point - shifted, rho, shifted, a=a, e1=e1) <= 0.0
    assert np.min(np.abs(np.asarray(interior_images)[np.asarray(interior_mask)] - complex(center))) < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize(
    "point,q,s,reference",
    [
        (
            0.025872115099088773 - 0.00012660115914060716j,
            1e-5,
            1.0,
            144.17177551337485,
        ),
        (
            -7.794776723440755e-05 - 7.416540530774523e-05j,
            1e-4,
            1.6,
            20696.29552460964,
        ),
    ],
)
def test_local_nested_phase_closes_strict_rho1e5_audit_points(point, q, s, reference):
    result = mag_uniform_local_boundary(
        point,
        1e-5,
        s=s,
        q=q,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        prefer_interior_anchor=True,
        return_info=True,
    )

    tolerance = 1e-5 + 1e-4 * abs(reference)
    assert int(result.status) == 0
    assert float(result.estimated_error) <= tolerance
    assert abs(float(result.magnification) - reference) <= tolerance


@pytest.mark.slow
def test_safe_lightcurve_compacts_mean_local_failure_into_interior_anchor_retry():
    point = -7.794776723440755e-05 - 7.416540530774523e-05j
    reference = 20696.29552460964
    common = dict(
        s=1.6,
        q=1e-4,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
    )
    mean_local = mag_uniform_local_boundary(point, 1e-5, return_info=True, **common)
    interior_local = mag_uniform_local_boundary(
        point,
        1e-5,
        prefer_interior_anchor=True,
        return_info=True,
        **common,
    )
    highlevel = mag_binary_safe(
        jnp.asarray([point]),
        1e-5,
        MAX_FULL_CALLS=1,
        chunk_size=1,
        **common,
    )[0]

    assert int(mean_local.status) != 0
    assert int(interior_local.status) == 0
    assert np.isfinite(float(highlevel))
    assert abs(float(highlevel) - reference) <= (1e-5 + 1e-4 * abs(reference))


@pytest.mark.slow
def test_local_nested_support_closes_between_sample_fold_minimum():
    point = 0.9824943188437842 + 0.00880220969350141j
    reference = 49.87308550714601
    result = mag_uniform_local_boundary(
        point,
        1e-5,
        s=1.6,
        q=1e-3,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        return_info=True,
    )
    actual_error = abs(float(result.magnification) - reference)
    tolerance = 1e-5 + 1e-4 * abs(reference)

    assert int(result.status) == 0
    # The nested difference is an empirical estimator, not a formal bound on
    # the independent-reference error.  Both must satisfy the public budget;
    # one is not required to dominate the other point by point.
    assert float(result.estimated_error) <= tolerance
    assert actual_error <= tolerance


@pytest.mark.slow
def test_local_nested_chart_closes_transient_planetary_component():
    point = -0.728614562384981 - 0.007200699676102258j
    reference = 7.055904959651395
    result = mag_uniform_local_boundary(
        point,
        1e-5,
        s=0.7,
        q=1e-5,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        return_info=True,
    )

    assert int(result.status) == 0
    assert abs(float(result.magnification) - reference) <= (1e-5 + 1e-4 * abs(reference))


def test_radial_topology_reports_small_capacity_and_large_kernel_closes():
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 256)
    image_limb = jnp.zeros((5, angles.size), dtype=jnp.complex128)
    image_limb = image_limb.at[0].set((1.0 + 0.01 * jnp.cos(angles)) * jnp.exp(1j * angles))
    mask_limb = jnp.zeros(image_limb.shape, dtype=bool)
    mask_limb = mask_limb.at[0, ::2].set(True)
    topology = define_radial_topology(
        image_limb,
        mask_limb,
        1e-4,
        margin_r=1.0,
        track_roots=False,
        sampled_turning_points=False,
    )
    retry = define_radial_topology(
        image_limb,
        mask_limb,
        1e-4,
        margin_r=1.0,
        track_roots=False,
        sampled_turning_points=False,
        breakpoint_capacity=RADIAL_RETRY_BREAKPOINT_CAPACITY,
        interval_capacity=RADIAL_RETRY_INTERVAL_CAPACITY,
    )

    assert int(topology.n_candidates_raw) > 64
    assert int(topology.status) == RADIAL_CAPACITY
    assert int(topology.n_intervals) <= 3
    assert int(retry.n_candidates_raw) == int(topology.n_candidates_raw)
    assert int(retry.n_intervals_raw) > 64
    assert int(retry.status) == 0


def test_root_angular_measure_matches_dense_ring_without_resolution_parameter():
    _, _, a, e1, shifted, _, w_shifted, rho = _binary_setup()
    r = 1.0056271741900544
    roots_result = angular_measure_binary_roots(
        r,
        0.0,
        2.0 * jnp.pi,
        w_shifted,
        rho,
        shifted,
        1e-9,
        a=a,
        e1=e1,
    )
    roots_result_tiny_local_budget = angular_measure_binary_roots(
        r,
        0.0,
        2.0 * jnp.pi,
        w_shifted,
        rho,
        shifted,
        1e-30,
        a=a,
        e1=e1,
    )
    intervals = angular_intervals_binary_roots(
        r,
        0.0,
        2.0 * jnp.pi,
        w_shifted,
        rho,
        shifted,
        1e-9,
        a=a,
        e1=e1,
    )

    n_dense = 65536
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_dense, endpoint=False)
    distances = distance_from_source(
        r,
        theta,
        w_shifted,
        shifted,
        nlenses=2,
        a=a,
        e1=e1,
    )
    dense_measure = jnp.mean(distances <= rho) * 2.0 * jnp.pi
    dense_cell_width = 2.0 * np.pi / n_dense

    assert int(roots_result.status) == ANGULAR_OK
    assert int(roots_result_tiny_local_budget.status) == ANGULAR_OK
    active = np.arange(intervals.intervals.shape[0]) < int(intervals.n_intervals)
    interval_measure = np.sum(np.diff(np.asarray(intervals.intervals), axis=1).ravel()[active])
    assert np.isclose(interval_measure, float(roots_result.measure), atol=1e-14)
    assert abs(float(roots_result.measure - dense_measure)) < dense_cell_width


def test_small_source_off_unit_roots_do_not_create_false_fatal_status():
    # At this empty ring two non-physical reciprocal roots sit near, but not on,
    # the unit circle and converge slowly under the fixed EA iteration budget.
    # Their residual must not invalidate a ring with no physical boundary.  The
    # independent 16-angle sign check also finds no crossing here.
    q = 0.05
    s = 1.0
    rho = 1e-4
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center = -0.002654304734183799 - 0.0026543047341837985j
    result = angular_intervals_binary_roots(
        0.8020109290208768,
        0.0,
        2.0 * jnp.pi,
        w_center - shifted,
        rho,
        shifted,
        64.0 * jnp.finfo(jnp.float64).eps,
        a=a,
        e1=e1,
    )

    assert int(result.status) == ANGULAR_OK
    assert int(result.n_intervals) == 0
    assert float(result.error) == 0.0


def test_near_unit_reciprocal_pair_is_rescued_atomically():
    eps = jnp.finfo(jnp.float64).eps
    unit_tolerance = 2048.0 * jnp.sqrt(eps)
    theta = 0.3
    inner_radius = 1.0 - 0.99999 * unit_tolerance
    roots = jnp.asarray(
        [
            inner_radius * jnp.exp(1j * theta),
            (1.0 / inner_radius) * jnp.exp(1j * theta),
        ]
    )
    unit_error = jnp.abs(jnp.abs(roots) - 1.0)
    unit_candidate = unit_error <= unit_tolerance

    rescued = _reciprocal_pair_rescue(
        roots,
        jnp.asarray([True, True]),
        unit_error,
        jnp.asarray([True, True]),
        unit_candidate,
        unit_tolerance,
    )

    assert np.array_equal(np.asarray(unit_candidate), np.asarray([True, False]))
    assert np.array_equal(np.asarray(rescued), np.asarray([True, True]))


def test_ill_conditioned_reciprocal_pair_uses_unit_band_scaled_tolerance():
    eps = jnp.finfo(jnp.float64).eps
    unit_tolerance = 2048.0 * jnp.sqrt(eps)
    theta = 0.3
    outer_radius = 1.0 + 0.99 * unit_tolerance
    inner_radius = 1.0 - 1.04 * unit_tolerance
    roots = jnp.asarray(
        [
            outer_radius * jnp.exp(1j * theta),
            inner_radius * jnp.exp(1j * (theta + 2e-6)),
        ]
    )
    unit_error = jnp.abs(jnp.abs(roots) - 1.0)
    unit_candidate = unit_error <= unit_tolerance

    rescued = _reciprocal_pair_rescue(
        roots,
        jnp.asarray([True, True]),
        unit_error,
        jnp.asarray([True, True]),
        unit_candidate,
        unit_tolerance,
    )

    assert np.array_equal(np.asarray(unit_candidate), np.asarray([True, False]))
    assert np.array_equal(np.asarray(rescued), np.asarray([True, True]))


def test_fixed_binary_roots_resolve_roman_recovery_boundary_crossings():
    # This exact GK31 radial node occurs in a Roman-model recovery case.  The
    # former 20-step fixed EA solve found only one of the two boundary roots,
    # so the public one-pass light curve returned NaN.
    s = 1.025189817009389
    q = 9.840799897016288e-6
    rho = 0.0018453285205730348
    w_center = 0.06940317682632598 - 0.0013061043233880167j
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)

    result = angular_intervals_binary_roots(
        1.0360819265012022,
        0.0,
        2.0 * jnp.pi,
        w_center - shifted,
        rho,
        shifted,
        64.0 * jnp.finfo(jnp.float64).eps,
        a=a,
        e1=e1,
        robust_roots=False,
        chart_center=0.0 + 0.0j,
    )

    assert int(result.status) == ANGULAR_OK
    assert int(result.n_intervals) == 2
    assert np.all(np.isfinite(np.asarray(result.intervals)))
    assert np.all(np.diff(np.asarray(result.intervals[:2]), axis=1) > 0.0)


def test_fixed_binary_roots_ignore_nearby_non_crossing_reciprocal_pair():
    # At this low-q GK31 node, a near-tangent reciprocal pair straddles the
    # unit-circle threshold on GPU.  It is a same-sign contact candidate, not
    # an inside interval, so pairwise validation must remain finite and empty.
    s = 1.0319953783486504
    q = 9.968586649656417e-6
    rho = 0.0018320985728076894
    w_center = 0.07005290915518367 + 0.00010278068291661585j
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)

    result = angular_intervals_binary_roots(
        1.0347532636407577,
        0.0,
        2.0 * jnp.pi,
        w_center - shifted,
        rho,
        shifted,
        64.0 * jnp.finfo(jnp.float64).eps,
        a=a,
        e1=e1,
        robust_roots=False,
        chart_center=0.0 + 0.0j,
    )

    assert int(result.status) == ANGULAR_OK
    assert int(result.n_intervals) == 0
    assert float(result.error) == 0.0


def test_planetary_chart_filters_roundoff_radial_turning_points():
    point = 0.0722579255149055 - 0.0024438627408989305j
    rho = 0.0017685892259106202
    s = 1.0484985960884143
    q = 9.962312735135355e-6
    lens = binary_geometry(s, q)
    image_limb, mask_limb = calc_source_limb(
        point,
        rho,
        999,
        nlenses=2,
        s=s,
        q=q,
    )
    origin_inside = binary_level_set(
        0.0 + 0.0j,
        point - lens.shifted,
        rho,
        lens.shifted,
        a=lens.a,
        e1=lens.e1,
    ) <= 0.0
    topology, _ = _planetary_mixed_topology(
        image_limb,
        mask_limb,
        rho,
        margin_r=0.5,
        lens=lens,
        w_center_shifted=point - lens.shifted,
        origin_inside=origin_inside,
        jacobian_radial_margin=True,
    )

    # The unfiltered local radius has more than 50 machine-scale zig-zags.
    # Only global extrema and physical host turning points remain.
    assert int(topology.status) == ANGULAR_OK
    assert int(topology.n_candidates_raw) <= 16
    assert int(topology.n_intervals_raw) <= 16


@pytest.mark.slow
def test_uniform_boundary_is_finite_at_rho_1e4_regression_point():
    result = mag_uniform_boundary(
        -0.002654304734183799 - 0.0026543047341837985j,
        1e-4,
        s=1.0,
        q=0.05,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        return_info=True,
    )
    reference = 30.60068676769095

    assert int(result.status) == ANGULAR_OK
    assert np.isfinite(float(result.magnification))
    assert np.isclose(float(result.magnification), reference, rtol=1e-4, atol=0.0)


@pytest.mark.slow
def test_safe_lightcurve_rescues_small_planet_boundary_failure():
    # VBBinaryLensing 3.7.0, Tol=RelTol=1e-12.  In the global COM polar frame
    # this tiny planetary image produces poorly conditioned angular roots and
    # a fatal boundary status.  The image-local fixed-chart pass is finite.
    point = 0.9788824508896057 + 0.001942618692247761j
    reference = 9.551087158506649
    common = dict(
        s=1.6,
        q=1e-4,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
    )
    global_boundary = mag_uniform_boundary(
        point,
        1e-4,
        max_radial_subdivisions=8,
        robust_roots=True,
        return_info=True,
        **common,
    )
    local_boundary = mag_uniform_local_boundary(
        point,
        1e-4,
        max_radial_subdivisions=8,
        return_info=True,
        **common,
    )
    lightcurve = mag_binary_safe(
        jnp.asarray([point]),
        1e-4,
        MAX_FULL_CALLS=1,
        chunk_size=1,
        **common,
    )
    single_pass = mag_binary(
        jnp.asarray([point]),
        1e-4,
        s=common["s"],
        q=common["q"],
        config=BinaryMagConfig(n_limb=common["Nlimb"]),
    )

    assert int(global_boundary.status) != ANGULAR_OK
    assert int(local_boundary.status) == ANGULAR_OK
    assert np.isclose(float(local_boundary.magnification), reference, rtol=1.5e-4)
    assert np.isfinite(float(single_pass[0]))
    assert np.isclose(float(single_pass[0]), reference, rtol=1e-3)
    assert np.isfinite(float(lightcurve[0]))
    assert np.isclose(float(lightcurve[0]), reference, rtol=1.5e-4)


@pytest.mark.slow
def test_uniform_boundary_matches_vbbl_reference_and_reports_error():
    s, q, _, _, _, w_center, _, rho = _binary_setup()
    common = dict(
        s=s,
        q=q,
        Nlimb=500,
        margin_r=0.5,
    )
    boundary = mag_uniform_boundary(
        w_center,
        rho,
        angular_atol=1e-5,
        return_info=True,
        **common,
    )
    vbbl = 26.504733989309987

    assert int(boundary.status) == ANGULAR_OK
    assert float(boundary.estimated_error) <= 1e-5
    assert np.isclose(float(boundary.magnification), vbbl, rtol=0.0, atol=4e-4)


def test_binary_public_path_handles_difficult_resonant_caustic_point():
    # VBBinaryLensing 3.7.0, Tol=RelTol=1e-12.  This is a deliberately
    # difficult point from the public comparison trajectory.
    point = -0.22880106808663814 - 0.2288010680866381j
    vbbl = 5.611936739107956
    lightcurve = mag_binary(
        jnp.asarray([point]),
        0.03,
        s=1.0,
        q=0.05,
        config=BinaryMagConfig(n_limb=500),
    )

    assert np.isfinite(float(lightcurve[0]))
    # This is a breakage guard for the fixed-work public solver, not an
    # accuracy guarantee.
    assert np.isclose(float(lightcurve[0]), vbbl, rtol=5e-3, atol=0.0)


def test_binary_public_path_avoids_false_planetary_radial_overflow():
    point = 0.0722579255149055 - 0.0024438627408989305j
    vbbl = 13.34459941133492
    lightcurve = mag_binary(
        jnp.asarray([point]),
        0.0017685892259106202,
        s=1.0484985960884143,
        q=9.962312735135355e-6,
        config=BinaryMagConfig(n_limb=500),
    )

    assert np.isfinite(float(lightcurve[0]))
    assert np.isclose(float(lightcurve[0]), vbbl, rtol=5e-3, atol=0.0)


def test_binary_public_path_handles_vmapped_dynamic_ea_boundary_pair():
    point = 0.07009147785397735 - 0.0012258946844154843j
    vbbl = 15.377807025845549

    evaluate = jax.jit(
        lambda value, radius, separation, mass_ratio: mag_binary(
            value[None],
            radius,
            s=separation,
            q=mass_ratio,
            config=BinaryMagConfig(n_limb=500),
        )[0]
    )
    magnification = jax.block_until_ready(
        evaluate(
            jnp.asarray(point),
            jnp.asarray(0.0017747191379164267),
            jnp.asarray(1.0484523576725895),
            jnp.asarray(1.0548626990703671e-5),
        )
    )

    assert np.isfinite(float(magnification))
    assert np.isclose(float(magnification), vbbl, rtol=5e-3, atol=0.0)


@pytest.mark.slow
def test_binary_public_path_difficult_point_has_consistent_derivatives():
    def public_magnification(real):
        return mag_binary(
            jnp.asarray([real - 0.2288010680866381j]),
            0.03,
            s=1.0,
            q=0.05,
            config=BinaryMagConfig(n_limb=500),
        )[0]

    real = jnp.asarray(-0.22880106808663814)
    forward = jax.jacfwd(public_magnification)(real)
    reverse = jax.grad(public_magnification)(real)
    assert np.isfinite(float(forward))
    assert np.isfinite(float(reverse))
    assert np.isclose(float(forward), float(reverse), rtol=1e-10, atol=1e-10)


@pytest.mark.slow
def test_deep_global_sixteen_way_retry_closes_saved_tolerance_failures():
    cases = _load_fixture("binary_highlevel_stress_failures.json")["cases"][:2]
    points = jnp.asarray([complex(case["source_x"], case["source_y"]) for case in cases])
    separations = jnp.asarray([case["s"] for case in cases])
    mass_ratios = jnp.asarray([case["q"] for case in cases])
    reference = np.asarray([case["vbbl"] for case in cases])

    def evaluate(point, separation, mass_ratio, subdivisions):
        return mag_uniform_boundary(
            point,
            1e-2,
            s=separation,
            q=mass_ratio,
            Nlimb=500,
            margin_r=1.0,
            angular_atol=1e-5,
            relative_tolerance=1e-4,
            robust_roots=True,
            deep_topology_sampling=True,
            max_radial_subdivisions=subdivisions,
            return_info=True,
        )

    eight_way = jax.jit(jax.vmap(lambda point, s, q: evaluate(point, s, q, 8)))(points, separations, mass_ratios)
    sixteen_way = jax.jit(jax.vmap(lambda point, s, q: evaluate(point, s, q, 16)))(points, separations, mass_ratios)
    eight_way, sixteen_way = jax.block_until_ready((eight_way, sixteen_way))

    assert np.all(np.asarray(eight_way.status) & RADIAL_TOLERANCE)
    assert np.all(np.asarray(sixteen_way.status) == ANGULAR_OK)
    actual = np.asarray(sixteen_way.magnification)
    target = 1e-5 + 1e-4 * np.abs(reference)
    assert np.all(np.abs(actual - reference) <= target)


@pytest.mark.slow
def test_binary_public_path_handles_saved_stress_point():
    case = _load_fixture("binary_highlevel_stress_failures.json")["cases"][0]
    point = jnp.asarray(complex(case["source_x"], case["source_y"]))
    single_pass = mag_binary(
        jnp.asarray([point]),
        case["rho"],
        s=case["s"],
        q=case["q"],
        config=BinaryMagConfig(n_limb=500),
    )[0]
    reference = float(case["vbbl"])

    assert np.isfinite(float(single_pass))
    assert np.isclose(float(single_pass), reference, rtol=5e-3)


@pytest.mark.slow
def test_dynamic_separation_preserves_near_tangent_root_pair():
    cases = _load_fixture("binary_highlevel_stress_failures.json")["cases"]
    case = next(item for item in cases if item["id"] == "case_10_point_18")
    point = jnp.asarray(complex(case["source_x"], case["source_y"]))

    # Keep separation traced. Capturing s=1.6 as a closure constant changes
    # GPU lowering and hid the original one-sided tangency-root rejection.
    evaluate = jax.jit(
        lambda value, separation: mag_uniform_boundary(
            value,
            case["rho"],
            s=separation,
            q=case["q"],
            Nlimb=500,
            margin_r=1.0,
            angular_atol=1e-5,
            relative_tolerance=1e-4,
            max_radial_subdivisions=16,
            return_info=True,
        )
    )
    result = jax.block_until_ready(evaluate(point, case["s"]))
    reference = float(case["vbbl"])
    tolerance = 1e-5 + 1e-4 * abs(reference)

    assert int(result.status) == ANGULAR_OK
    assert float(result.estimated_error) <= tolerance
    assert abs(float(result.magnification) - reference) <= tolerance


@pytest.mark.slow
def test_rho1e5_p0_fixtures_close_after_robust_error_retry():
    cases = _load_fixture("binary_p0_rho1e5_failures.json")["cases"]
    points = jnp.asarray([complex(case["source_x"], case["source_y"]) for case in cases])
    mass_ratios = jnp.asarray([case["q"] for case in cases])
    reference = np.asarray([case["vbbl"] for case in cases])

    def evaluate(point, mass_ratio, robust_roots, max_subdivisions):
        return mag_uniform_boundary(
            point,
            1e-5,
            s=1.0,
            q=mass_ratio,
            Nlimb=500,
            margin_r=1.0,
            angular_atol=1e-5,
            relative_tolerance=1e-4,
            robust_roots=robust_roots,
            max_radial_subdivisions=max_subdivisions,
            return_info=True,
        )

    shallow = jax.jit(jax.vmap(lambda point, q: evaluate(point, q, False, 2)))(points, mass_ratios)
    robust = jax.jit(jax.vmap(lambda point, q: evaluate(point, q, True, 8)))(points, mass_ratios)
    shallow, robust = jax.block_until_ready((shallow, robust))

    shallow_status = np.asarray(shallow.status)
    robust_status = np.asarray(robust.status)
    values = np.asarray(robust.magnification)
    estimated_error = np.asarray(robust.estimated_error)
    tolerance = 1e-5 + 1e-4 * np.abs(reference)

    # These are the original failure mechanism: the conservative EA32 pass is
    # rejected by radial error accounting, then the bounded EA40/8-way retry
    # closes with the correlated-roundoff estimate. This is not merely a finite
    # output assertion.
    assert np.all((shallow_status & RADIAL_TOLERANCE) != 0)
    assert np.array_equal(robust_status, np.zeros(6, dtype=np.int32))
    assert np.all(estimated_error <= tolerance)
    assert np.all(np.abs(values - reference) <= tolerance)


@pytest.mark.slow
def test_limb_boundary_resolves_profile_quadrature_near_peak():
    # Independent VBBL uniform-disk layer cake: 128 Gauss radii, each evaluated
    # in an isolated process with Tol=1e-10. The former angular G7/K15 rule made
    # the propagated error exceed the production tolerance at this point.
    result = mag_limb_dark_boundary(
        -0.02282702071398015 - 0.022827020713980146j,
        0.03,
        s=1.0,
        q=0.05,
        u1=0.7,
        Nlimb=500,
        margin_r=1.0,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        return_info=True,
    )
    reference = 39.506154721110065
    tolerance = 1e-5 + 1e-4 * abs(float(result.magnification))

    assert int(result.status) == ANGULAR_OK
    assert float(result.estimated_error) <= tolerance
    assert abs(float(result.magnification) - reference) <= float(result.estimated_error)
    assert np.isclose(float(result.magnification), reference, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize(
    "solver",
    [mag_uniform_boundary, mag_limb_dark_boundary],
)
def test_boundary_api_rejects_dense_grid_arguments(solver):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        solver(
            0.03 + 0.01j,
            5e-3,
            s=1.0,
            q=1e-2,
            r_resolution=500,
        )


@pytest.mark.slow
def test_limb_dark_boundary_reduces_to_uniform_and_matches_vbbl():
    s, q, _, _, _, w_center, _, rho = _binary_setup()
    common = dict(
        s=s,
        q=q,
        Nlimb=500,
        angular_atol=1e-5,
        return_info=True,
    )
    uniform = mag_uniform_boundary(w_center, rho, **common)
    limb_zero = mag_limb_dark_boundary(w_center, rho, u1=0.0, **common)
    limb = mag_limb_dark_boundary(w_center, rho, u1=0.5, **common)
    # VBBinaryLensing 3.7.0 in a state-isolated process with ``a1=0.5``,
    # ``minannuli=100``, and absolute accuracy 1e-8.
    vbbl = 26.499860262906463

    assert int(limb_zero.status) == ANGULAR_OK
    assert int(limb.status) == ANGULAR_OK
    assert np.isclose(
        float(limb_zero.magnification),
        float(uniform.magnification),
        rtol=0.0,
        atol=1e-10,
    )
    assert float(limb.estimated_error) <= 1e-5
    assert np.isclose(float(limb.magnification), vbbl, rtol=0.0, atol=3e-6)


@pytest.mark.slow
def test_limb_dark_boundary_reverse_matches_forward_mode():
    s, q, _, _, _, w_center, _, rho = _binary_setup()

    def magnification(mass_ratio):
        return mag_limb_dark_boundary(
            w_center,
            rho,
            s=s,
            q=mass_ratio,
            u1=0.5,
            Nlimb=500,
            angular_atol=1e-5,
        )

    mass_ratio = jnp.asarray(q)
    forward = jax.jacfwd(magnification)(mass_ratio)
    reverse = jax.grad(magnification)(mass_ratio)
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=1e-7, atol=1e-6)


@pytest.mark.slow
def test_generic_radial_profile_reduces_to_uniform_disk():
    s, q, _, _, _, w_center, _, rho = _binary_setup()
    common = dict(
        s=s,
        q=q,
        Nlimb=500,
        angular_atol=1e-5,
        return_info=True,
    )
    uniform = mag_uniform_boundary(w_center, rho, **common)
    generic = mag_radial_profile_boundary(
        w_center,
        rho,
        lambda distance_over_rho: jnp.ones_like(distance_over_rho),
        jnp.pi,
        **common,
    )

    assert int(generic.status) == ANGULAR_OK
    assert np.isclose(
        float(generic.magnification),
        float(uniform.magnification),
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.slow
def test_generic_radial_profile_rejects_nonpositive_flux():
    s, q, _, _, _, w_center, _, rho = _binary_setup()
    result = mag_radial_profile_boundary(
        w_center,
        rho,
        lambda distance_over_rho: jnp.ones_like(distance_over_rho),
        0.0,
        s=s,
        q=q,
        Nlimb=60,
        return_info=True,
    )

    assert int(result.status) == ANGULAR_ROOT_FAILURE
    assert np.isnan(float(result.magnification))
    assert np.isnan(float(result.estimated_error))


@pytest.mark.slow
def test_parallel_and_sequential_image_regions_agree():
    s, q, _, _, _, w_center, _, rho = _binary_setup()
    common = dict(
        s=s,
        q=q,
        Nlimb=60,
        margin_r=0.5,
        angular_atol=1e-5,
        return_info=True,
    )
    sequential = mag_uniform_boundary(w_center, rho, parallel_regions=False, **common)
    parallel = mag_uniform_boundary(w_center, rho, parallel_regions=True, **common)

    assert int(sequential.status) == int(parallel.status)
    assert np.isclose(
        float(sequential.magnification),
        float(parallel.magnification),
        rtol=0.0,
        atol=1e-10,
    )
    assert np.isclose(
        float(sequential.estimated_error),
        float(parallel.estimated_error),
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.slow
def test_uniform_boundary_ignores_unused_fixed_topology_slots():
    result = mag_uniform_boundary(
        -0.031860785514265286 + 0.022521213663533896j,
        5e-3,
        s=1.0,
        q=1e-2,
        Nlimb=500,
        margin_r=0.5,
        angular_atol=3e-5,
        return_info=True,
    )

    assert int(result.status) == ANGULAR_OK
    assert float(result.estimated_error) <= 3e-5
    assert np.isfinite(float(result.magnification))


def test_limb_root_tracking_removes_false_radial_extrema():
    q, s, rho = 1e-2, 1.0, 5e-3
    point = -0.08428327024111676 + 0.008474651215741182j
    image_limb, mask_limb = calc_source_limb(point, rho, 500, nlenses=2, s=s, q=q)
    untracked = define_radial_topology(image_limb, mask_limb, rho, margin_r=0.5, track_roots=False)
    tracked = define_radial_topology(image_limb, mask_limb, rho, margin_r=0.5, track_roots=True)

    assert int(tracked.n_candidates_raw) <= 16
    assert int(tracked.n_candidates_raw) * 4 < int(untracked.n_candidates_raw)
    assert int(untracked.n_candidates_raw) > 64
    assert int(tracked.status) == 0
    assert int(untracked.status) == RADIAL_CAPACITY
    assert int(untracked.n_intervals_raw) <= 16


def test_generic_lens_margin_matches_binary_compatibility_shorthand():
    q, s, rho = 1e-2, 1.0, 5e-3
    point = -0.08428327024111676 + 0.008474651215741182j
    image_limb, mask_limb = calc_source_limb(point, rho, 100, nlenses=2, s=s, q=q)
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    shorthand = define_radial_topology(
        image_limb,
        mask_limb,
        rho,
        binary_margin_parameters=(shifted, a, e1),
    )
    generic = define_radial_topology(
        image_limb,
        mask_limb,
        rho,
        lens_margin_parameters=(
            shifted,
            jnp.asarray([a, -a]),
            jnp.asarray([e1, 1.0 - e1]),
        ),
    )
    assert np.array_equal(np.asarray(generic.status), np.asarray(shorthand.status))
    assert np.array_equal(np.asarray(generic.n_intervals), np.asarray(shorthand.n_intervals))
    assert np.allclose(
        np.asarray(generic.intervals),
        np.asarray(shorthand.intervals),
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.slow
def test_jacobian_margin_recovers_sharp_image_support_between_limb_samples():
    point = -0.6757280840456696 + 0.6174860296825364j
    reference = 61.92633478114333
    common = dict(
        rho=1e-4,
        q=0.1,
        s=0.7,
        Nlimb=500,
        robust_roots=False,
        certify_topology=False,
        return_info=True,
    )
    fixed_margin = mag_uniform_boundary(point, jacobian_radial_margin=False, **common)
    jacobian_margin = mag_uniform_boundary(point, jacobian_radial_margin=True, **common)

    fixed_error = jnp.abs(fixed_margin.magnification - reference)
    jacobian_error = jnp.abs(jacobian_margin.magnification - reference)
    assert jacobian_error < 1e-5
    assert jacobian_error < 1e-3 * fixed_error


@pytest.mark.slow
def test_radial_topology_regression_matches_vbbl_reference():
    s, q, w_center, rho, vbbl = _topology_stress_setup()
    result = mag_uniform_boundary(
        w_center,
        rho,
        s=s,
        q=q,
        Nlimb=500,
        margin_r=0.5,
        angular_atol=1e-5,
        return_info=True,
    )

    assert int(result.status) == ANGULAR_OK
    assert np.isclose(float(result.magnification), vbbl, rtol=0.0, atol=1e-4)


@pytest.mark.slow
def test_nested_radial_phase_disagreement_fails_closed():
    s, q, w_center, rho, vbbl = _topology_stress_setup()
    common = dict(
        s=s,
        q=q,
        Nlimb=3,
        margin_r=0.5,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        max_radial_subdivisions=2,
        robust_roots=True,
    )
    diagnostic = mag_uniform_boundary(w_center, rho, return_info=True, **common)
    rejected = mag_uniform_boundary(w_center, rho, **common)
    closed = mag_uniform_boundary(
        w_center,
        rho,
        s=s,
        q=q,
        Nlimb=8,
        margin_r=0.5,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        max_radial_subdivisions=2,
        robust_roots=True,
        return_info=True,
    )

    assert int(diagnostic.status) & RADIAL_TOPOLOGY
    assert not np.isfinite(float(rejected))
    assert int(closed.status) == 0
    assert np.isclose(float(closed.magnification), vbbl, rtol=0.0, atol=1e-4)


@pytest.mark.slow
def test_transient_fold_pair_integration_matches_reference():
    s, q, w_center, rho, vbbl = _transient_pair_setup()
    result = mag_uniform_boundary(
        w_center,
        rho,
        s=s,
        q=q,
        Nlimb=500,
        angular_atol=1e-5,
        return_info=True,
    )
    assert int(result.status) == ANGULAR_OK
    assert np.isclose(float(result.magnification), vbbl, rtol=0.0, atol=5e-6)


def test_binary_public_config_exposes_only_topology_sampling():
    assert [field.name for field in fields(BinaryMagConfig)] == ["n_limb"]


@pytest.mark.slow
def test_binary_fast_path_retains_tolerance_warning_without_dense_retry():
    s, q, _, _, _, w_center, _, rho = _binary_setup()
    w_points = jnp.asarray([w_center])
    config = BinaryMagConfig(n_limb=40)
    failed_boundary = mag_uniform_boundary(
        w_center,
        rho,
        angular_atol=0.0,
        s=s,
        q=q,
        Nlimb=40,
        margin_r=0.5,
        relative_tolerance=0.0,
    )
    relative_boundary = mag_uniform_boundary(
        w_center,
        rho,
        angular_atol=0.0,
        relative_tolerance=1e-4,
        s=s,
        q=q,
        Nlimb=40,
        margin_r=0.5,
    )
    boundary = mag_binary(w_points, rho, s=s, q=q, config=config)
    dense = mag_binary_dense(
        w_points,
        rho,
        s=s,
        q=q,
        r_resolution=30,
        th_resolution=120,
        Nlimb=40,
        bins_r=20,
        bins_th=40,
        margin_r=0.5,
        margin_th=0.5,
        MAX_FULL_CALLS=1,
        chunk_size=1,
    )

    assert not np.isfinite(float(failed_boundary))
    assert np.isfinite(float(relative_boundary))
    assert np.isfinite(float(boundary[0]))
    assert np.isclose(float(boundary[0]), float(relative_boundary), rtol=1e-3)
    assert np.isfinite(float(dense[0]))
