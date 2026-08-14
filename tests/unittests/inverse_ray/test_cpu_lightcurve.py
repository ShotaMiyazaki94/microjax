import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray import mag_binary
from microjax.inverse_ray.lightcurve import _binary_prefilter
from microjax.inverse_ray.cpu import mag_binary_cpu_lightcurve


@pytest.mark.fast
def test_public_cpu_backend_matches_vbbl_uniform_reference():
    sources = jnp.asarray([0.1 + 0.2j, 0.6 - 0.2j], dtype=jnp.complex128)
    result = mag_binary(
        sources,
        1e-2,
        s=1.0,
        q=0.3,
        backend="cpu",
        return_info=True,
    )
    # VBBinaryLensing BinaryMag2 with Tol=RelTol=1e-8.
    expected = np.asarray([4.5028889540801815, 1.7962876762356366])
    np.testing.assert_allclose(np.asarray(result.magnification), expected, rtol=5e-7, atol=0.0)
    np.testing.assert_array_equal(np.asarray(result.status), 0)


def test_public_binary_has_no_user_accuracy_tolerance_argument():
    assert "rtol" not in inspect.signature(mag_binary).parameters


@pytest.mark.fast
def test_public_cpu_backend_matches_vbbl_limb_dark_reference():
    sources = jnp.asarray([0.1 + 0.2j], dtype=jnp.complex128)
    result = mag_binary(
        sources,
        1e-2,
        s=1.0,
        q=0.3,
        u1=0.5,
        backend="cpu",
        return_info=True,
    )
    # VBBinaryLensing BinaryMag2 with a1=0.5 and Tol=RelTol=1e-8.
    expected = 4.502713625245832
    assert np.isclose(float(result.magnification[0]), expected, rtol=5e-7)
    assert int(result.status[0]) == 0


def test_public_cpu_backend_is_the_one_shot_alias():
    u1 = 0.5
    sources = jnp.asarray([0.1 + 0.2j, 0.6 - 0.2j], dtype=jnp.complex128)
    default = mag_binary(
        sources,
        1e-2,
        s=1.0,
        q=0.3,
        u1=u1,
        backend="cpu",
        return_info=True,
    )
    alias = mag_binary(
        sources,
        1e-2,
        s=1.0,
        q=0.3,
        u1=u1,
        backend="cpu-one-shot",
        return_info=True,
    )
    for field in (
        "magnification",
        "estimated_error",
        "tier",
        "n_limb",
        "n_radial_nodes",
        "status",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(default, field)), np.asarray(getattr(alias, field)))


@pytest.mark.parametrize(
    ("u1", "expected"),
    (
        (0.0, 4.5028889540801815),
        (0.5, 4.502713625245832),
    ),
)
@pytest.mark.slow
def test_direct_cpu_lightcurve_uses_the_fixed_chart_scheduler(u1, expected):
    result = mag_binary_cpu_lightcurve(
        jnp.asarray([0.1 + 0.2j], dtype=jnp.complex128),
        1e-2,
        s=1.0,
        q=0.3,
        u1=u1,
        rtol=1e-3,
    )
    assert int(result.status[0]) == 0
    assert np.isclose(float(result.magnification[0]), expected, rtol=2e-4)


def test_public_cpu_backend_refines_buried_planetary_caustic_with_radial_chart():
    result = mag_binary(
        jnp.asarray([-0.0010946356833467202 - 0.006172611801159194j]),
        0.00988422630886768,
        s=0.5992640188539046,
        q=0.0005041898087262488,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) >= 5
    assert np.isnan(float(result.estimated_error[0]))
    assert np.isclose(float(result.magnification[0]), 180.0580995708625, rtol=2e-4)


def test_public_cpu_backend_cartesian_limb_dark_resolves_caustic_residual():
    result = mag_binary(
        jnp.asarray([-0.10794172585680531 - 0.1079417258568053j]),
        0.005,
        s=0.85,
        q=0.03,
        u1=0.5,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) >= 5
    # VBBinaryLensing BinaryMag2 with a1=0.5 and Tol=RelTol=1e-8.
    assert np.isclose(float(result.magnification[0]), 11.216178805288225, rtol=5e-5)


@pytest.mark.slow
def test_public_cpu_backend_rotated_cartesian_resolves_false_angular_convergence():
    result = mag_binary(
        jnp.asarray([0.36569748065577623 - 0.03401942639008235j]),
        0.001,
        s=1.2,
        q=0.001,
        backend="cpu-adaptive",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 4.702229917827288, rtol=1e-3)


def test_public_cpu_one_shot_routes_a_distant_three_to_five_image_fold_to_polar():
    source = jnp.asarray([-0.29120329365498104 - 0.291203293654981j])
    result = mag_binary(
        source,
        1.0e-2,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) == 8
    assert np.isclose(float(result.magnification[0]), 4.6192814, rtol=1.0e-4)


def test_public_cpu_fragmented_fold_selects_a_complete_cartesian_projection():
    """A minimum-cell half-step axis must expose the narrow fold lobe."""

    result = mag_binary(
        jnp.asarray([-0.042822783044830975 - 0.04282278304483097j]),
        5.0e-2,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )

    # VBBinaryLensing BinaryMag2 with Tol=RelTol=1e-7.  The old five-axis
    # route selected a projection that omitted a narrow lobe and certified a
    # value lower by 8.0e-3 relative.
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) >= 5
    assert np.isclose(
        float(result.magnification[0]), 21.769986213395345, rtol=2.0e-4
    )


@pytest.mark.fast
def test_public_cpu_one_shot_rejects_low_q_buried_central_caustic():
    """Nested polar rules must not certify the same incomplete image support."""

    result = mag_binary(
        jnp.asarray([-0.00010742178999379606 + 0.00027830204890296644j]),
        3.0e-4,
        s=0.8,
        q=1.0e-6,
        backend="cpu",
        return_info=True,
    )

    # VBBinaryLensing converges to 23315.11524 for Tol=1e-7 through 1e-10;
    # the incomplete three-image polar support is about 4300.  The one-shot
    # contract exposes this case without adding a retry to the JAX graph.
    assert int(result.status[0]) != 0


def test_public_cpu_resolves_buried_contact_at_critical_image_angle():
    """A full angular chart resolves a near-limb buried caustic locally."""

    source = jnp.asarray([-0.22122504688671182 - 0.18562985524981634j])
    uniform = mag_binary(
        source,
        3.0e-2,
        s=0.9,
        q=1.0e-2,
        backend="cpu",
        return_info=True,
    )
    limb_dark = mag_binary(
        source,
        3.0e-2,
        s=0.9,
        q=1.0e-2,
        u1=0.5,
        backend="cpu",
        return_info=True,
    )

    assert int(uniform.status[0]) == 0
    assert int(limb_dark.status[0]) == 0
    assert int(uniform.tier[0]) == 7
    assert int(limb_dark.tier[0]) == 7
    np.testing.assert_allclose(uniform.magnification, [4.061743963071618], rtol=5e-5)
    np.testing.assert_allclose(limb_dark.magnification, [4.055193392644604], rtol=5e-5)


def test_public_cpu_low_q_fold_falls_back_from_incomplete_real_root_set():
    """A marginal real companion solve must retain the complete fold support."""

    result = mag_binary(
        jnp.asarray([-5.5411597499865775e-5 - 1.377747406402e-4j]),
        3.0e-4,
        s=1.25,
        q=1.0e-6,
        backend="cpu",
        return_info=True,
    )

    # The polished-real ray solver alone loses a positive root at this dense
    # planetary fold.  Its residual guard must select the established complex
    # companion solve, which agrees with the VBML reference below.
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) == 7
    assert np.isclose(float(result.magnification[0]), 6238.1923308948235, rtol=2e-7)


@pytest.mark.parametrize(
    ("source", "rho", "s", "q"),
    (
        (0.44705482558656645 + 0.0962464714933328j, 3.0e-3, 1.25, 1.0e-2),
        (-0.45551582456655687 + 0.015003246996856474j, 3.0e-4, 0.8, 1.0e-4),
    ),
)
def test_public_cpu_one_shot_rejects_unresolved_distant_fold(
    source, rho, s, q
):
    result = mag_binary(
        jnp.asarray([source]),
        rho,
        s=s,
        q=q,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) != 0


def test_public_cpu_rejects_twinkle_close_planetary_support_gap():
    """A large close-planet caustic must not certify incomplete polar support."""

    result = mag_binary(
        jnp.asarray(
            [
                2.0 + 2.0j,
                -1.4995854860387174 + 0.03538610753742722j,
            ]
        ),
        5.0e-5,
        s=0.5,
        q=1.0e-4,
        backend="cpu",
        return_info=True,
    )

    # VBBL converges to 4.695890824 from 1e-8 to 1e-9 while the two polar
    # estimates co-converge near 4.6691.  Reject the shared support omission.
    assert int(result.status[1]) != 0


def test_public_cpu_accepts_complete_low_q_radial_chart_away_from_contact():
    """A conservative low-q warning must not reject complete radial support."""

    result = mag_binary(
        jnp.asarray([9.84050041854608e-05 - 0.0001082302615975933j]),
        3.0e-4,
        s=0.8,
        q=1.0e-6,
        backend="cpu",
        return_info=True,
    )

    assert int(result.tier[0]) >= 5
    assert int(result.status[0]) == 0
    assert np.isclose(float(result.magnification[0]), 6250.794430174132, rtol=5.0e-5)


def test_public_cpu_angle_first_resolves_former_cartesian_topology_omission():
    result = mag_binary(
        jnp.asarray([-0.1473754964687426 + 0.07700093692907822j]),
        3.0e-3,
        s=1.0,
        q=1.0e-2,
        backend="cpu",
        return_info=True,
    )

    assert int(result.status[0]) == 0
    assert int(result.tier[0]) == 7
    assert np.isclose(float(result.magnification[0]), 13.049857875268737, rtol=1e-4)


def test_public_cpu_rejects_distant_cartesian_topology_omission():
    result = mag_binary(
        jnp.asarray([0.447812800537068 - 0.0029996436395797036j]),
        3.0e-3,
        s=1.25,
        q=1.0e-6,
        backend="cpu",
        return_info=True,
    )

    assert int(result.status[0]) != 0


def test_public_cpu_one_shot_ld_uses_shared_radial_geometry_route():
    sources = jnp.asarray(
        [
            -0.08495097 - 0.02345444j,
            -0.06500461 + 0.00031671j,
        ]
    )
    result = mag_binary(
        sources,
        5.0e-3,
        s=1.0,
        q=1.0e-3,
        u1=0.5,
        backend="cpu",
        return_info=True,
    )
    # VBBinaryLensing BinaryMag2 with a1=0.5 and Tol=RelTol=1e-8.
    expected = np.asarray([13.288252113263258, 8.927697151222718])
    np.testing.assert_array_equal(np.asarray(result.status), 0)
    np.testing.assert_array_equal(np.asarray(result.tier), np.asarray([7, 7]))
    np.testing.assert_allclose(np.asarray(result.magnification), expected, rtol=1.0e-4)


def test_public_cpu_planetary_guard_uses_the_internal_lens_frame():
    # This source overlaps a wide planetary caustic.  The old guard used
    # ``-1 / s`` instead of ``a - 1 / s`` in the symmetric internal frame and
    # incorrectly accepted the multipole value with 1.15e-3 relative error.
    result = mag_binary(
        jnp.asarray([1.119987237 + 0.057820520j]),
        0.04981945,
        s=1.672178637,
        q=0.00013581767,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 1.2624449858, rtol=1e-3)


def test_public_cpu_chart_consensus_resolves_correlated_projection_error():
    # The source-normal strip projections have a correlated 2.20e-3 error
    # here.  Independent chart arbitration resolves it.
    result = mag_binary(
        jnp.asarray([0.01663364300088473 + 0.01663364300088473j]),
        0.05,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 32.680254589161514, rtol=1e-4)


def test_public_cpu_source_limb_support_resolves_planetary_topology():
    result = mag_binary(
        jnp.asarray([-0.005297926262718188 + 0.00023223841465465215j]),
        0.0036652167671266146,
        s=0.6753608953491668,
        q=0.007935217541601874,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) >= 5
    assert np.isclose(float(result.magnification[0]), 235.7462655075391, rtol=1e-4)


@pytest.mark.slow
def test_public_cpu_small_q_resonant_caustic_uses_high_order_polar_pair():
    times = jnp.linspace(-5.0, 5.0, 1000, dtype=jnp.float64)
    alpha = jnp.deg2rad(jnp.asarray(50.0, dtype=jnp.float64))
    tau = times / 10.0
    trajectory = -0.001 * jnp.sin(alpha) + tau * jnp.cos(alpha) + 1.0j * (0.001 * jnp.cos(alpha) + tau * jnp.sin(alpha))
    sources = trajectory[jnp.asarray([495, 500, 503, 504])]
    result = mag_binary(
        sources,
        0.005,
        s=1.0,
        q=1.0e-6,
        backend="cpu-adaptive",
        return_info=True,
    )

    # VBBinaryLensing BinaryMag2 with Tol=RelTol=1e-10.  The low-order polar
    # pairs either converge non-monotonically or retain a fitted tangency at
    # these central-caustic samples; the lazy 24/32 pair resolves both cases.
    expected = np.asarray(
        [
            290.8775438935654,
            394.9879477943679,
            339.9575932430476,
            290.88842162049804,
        ]
    )
    np.testing.assert_array_equal(np.asarray(result.status), 0)
    np.testing.assert_array_equal(np.asarray(result.tier), 8)
    np.testing.assert_allclose(np.asarray(result.magnification), expected, rtol=1e-4, atol=0.0)


def test_cpu_one_shot_small_q_is_certified_without_retry():
    """The lens-state route selects fixed angle-first GL24/32 up front."""

    times = jnp.linspace(-5.0, 5.0, 1000, dtype=jnp.float64)
    alpha = jnp.deg2rad(jnp.asarray(50.0, dtype=jnp.float64))
    tau = times / 10.0
    trajectory = -0.001 * jnp.sin(alpha) + tau * jnp.cos(alpha) + 1.0j * (0.001 * jnp.cos(alpha) + tau * jnp.sin(alpha))
    sources = trajectory[jnp.asarray([500])]
    result = mag_binary(
        sources,
        0.005,
        s=1.0,
        q=1.0e-6,
        backend="cpu-one-shot",
        return_info=True,
    )
    expected = np.asarray([394.9879477943679])
    np.testing.assert_array_equal(np.asarray(result.status), 0)
    np.testing.assert_array_equal(np.asarray(result.tier), 7)
    np.testing.assert_allclose(np.asarray(result.magnification), expected, rtol=1.0e-4, atol=0.0)


@pytest.mark.slow
def test_cpu_one_shot_small_rho_uses_conditioned_polar_chart():
    """The fixed-r chart avoids the small-rho angle-first conditioning loss."""

    result = mag_binary(
        jnp.asarray([0.0004701465366526 + 0.0447735510389778j]),
        1.0e-4,
        s=1.0,
        q=1.0e-3,
        backend="cpu-one-shot",
        return_info=True,
    )

    assert int(result.tier[0]) == 7
    assert int(result.status[0]) == 0
    assert np.isclose(float(result.magnification[0]), 36.64035974390989, rtol=1e-4)

    parameters = jnp.asarray(
        [0.0004701465366526, 0.0447735510389778, 1.0e-4, 1.0, 1.0e-3]
    )

    def value(values):
        return mag_binary(
            jnp.asarray([values[0] + 1.0j * values[1]]),
            values[2],
            s=values[3],
            q=values[4],
            backend="cpu-one-shot",
            return_info=True,
        ).magnification

    forward = jax.jit(jax.jacfwd(value))(parameters)
    assert np.all(np.isfinite(np.asarray(forward)))


@pytest.mark.parametrize(
    "source,rho,s,q,expected,expected_tier",
    [
        (
            0.5823420163986178 + 0.0001720705839026j,
            3.0e-5,
            1.3333333333333333,
            1.0e-6,
            1.7985531538150774,
            5,
        ),
        (
            0.6477242270705221 + 0.000650846241964j,
            1.701901334924e-4,
            1.375,
            1.0e-6,
            4.414241874999235,
            6,
        ),
        (
            -0.9750533454782816 - 0.004635711979391j,
            3.0e-5,
            0.625,
            3.162277660168379e-6,
            4.448959955719384,
            6,
        ),
    ],
    ids=("cartesian", "cartesian-high", "former-polar-high"),
)
def test_cpu_one_shot_small_source_radial_projection_resolves_narrow_image_pair(
    source,
    rho,
    s,
    q,
    expected,
    expected_tier,
):
    """The source-radial chart must not lose the thin planetary image pair."""

    result = mag_binary(
        jnp.asarray([source]),
        rho,
        s=s,
        q=q,
        u1=0.5,
        backend="cpu-one-shot",
        return_info=True,
    )

    assert int(result.tier[0]) == expected_tier
    assert np.isclose(float(result.magnification[0]), expected, rtol=1.0e-4)


@pytest.mark.slow
def test_cpu_one_shot_simple_polar_small_q_supports_forward_ad():
    """The production source-limb polar route has finite forward derivatives."""

    parameters = jnp.asarray(
        [-0.0030387583062423104, 0.004441826568286664, 0.005, 1.0, 1.0e-6],
        dtype=jnp.float64,
    )

    def value(values):
        return mag_binary(
            jnp.asarray([values[0] + 1.0j * values[1]]),
            values[2],
            s=values[3],
            q=values[4],
            backend="cpu-one-shot",
            return_info=True,
        ).magnification

    result = mag_binary(
        jnp.asarray([parameters[0] + 1.0j * parameters[1]]),
        parameters[2],
        s=parameters[3],
        q=parameters[4],
        backend="cpu-one-shot",
        return_info=True,
    )
    forward = jax.jit(jax.jacfwd(value))(parameters)

    np.testing.assert_array_equal(np.asarray(result.status), 0)
    np.testing.assert_array_equal(np.asarray(result.tier), 7)
    assert np.all(np.isfinite(np.asarray(forward)))


def test_cpu_one_shot_planetary_topology_selects_simple_radial_before_integration():
    cases = (
        (
            -0.0010946356833467202 - 0.006172611801159194j,
            0.00988422630886768,
            0.5992640188539046,
            0.0005041898087262488,
            180.0580995708625,
        ),
        (
            -0.005297926262718188 + 0.00023223841465465215j,
            0.0036652167671266146,
            0.6753608953491668,
            0.007935217541601874,
            235.7462655075391,
        ),
    )
    for source, rho, s, q, expected in cases:
        result = mag_binary(
            jnp.asarray([source]),
            rho,
            s=s,
            q=q,
            backend="cpu-one-shot",
            return_info=True,
        )
        assert int(result.status[0]) == 0
        assert int(result.tier[0]) == 7
        assert np.isclose(float(result.magnification[0]), expected, rtol=2.0e-4)


@pytest.mark.slow
def test_public_cpu_central_planetary_images_use_the_faster_polar_chart():
    # When u < rho these images wrap most of the Einstein ring.  Several
    # Cartesian projections share a roughly 3.4e-3 support bias, whereas the
    # early polar chart is both faster and independently accurate.
    sources = jnp.asarray(
        [
            -7.071067811865475e-7 + 7.071067811865476e-7j,
            1.0e-4j,
            -2.77163859753386e-4 + 1.1480502970952697e-4j,
        ],
        dtype=jnp.complex128,
    )
    result = mag_binary(
        sources,
        1.0e-3,
        s=0.9,
        q=1.0e-4,
        backend="cpu-adaptive",
        return_info=True,
    )

    # VBBinaryLensing BinaryMag2 with Tol=RelTol=1e-9.
    expected = np.asarray([1982.2002989135124, 1976.9052521431684, 1935.5497859585093])
    np.testing.assert_array_equal(np.asarray(result.status), 0)
    np.testing.assert_array_equal(np.asarray(result.tier), 8)
    np.testing.assert_allclose(np.asarray(result.magnification), expected, rtol=2e-5, atol=0.0)


@pytest.mark.slow
def test_public_cpu_central_contact_band_skips_low_order_coconvergence():
    rho = 5.0e-3
    angle = 2.5 * 2.0 * np.pi / 8.0
    source = jnp.asarray([0.9 * rho * np.exp(1.0j * angle)])
    result = mag_binary(
        source,
        rho,
        s=0.9,
        q=1.0e-8,
        backend="cpu-adaptive",
        return_info=True,
    )

    # The shared-trace 12/16 polar pair co-converges 1.77e-3 below this
    # VBBinaryLensing Tol=RelTol=1e-10 value.  The contact band must go
    # directly to the lazy 24/32 pair instead.
    expected = 298.3720571411986
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) == 8
    assert np.isclose(float(result.magnification[0]), expected, rtol=2e-5)


@pytest.mark.slow
def test_public_cpu_fold_transition_requires_the_polar_chart():
    # The source crosses a planetary fold. Source-normal and lens-axis strips
    # agree within 0.20 times the old adaptive threshold while sharing a
    # 1.56e-3 area bias. A detected
    # 3-to-5 image transition must therefore bypass Cartesian consensus.
    result = mag_binary(
        jnp.asarray([-0.0005541615362192219 + 0.0004943152078276775j]),
        0.001,
        s=0.8,
        q=0.001,
        backend="cpu-adaptive",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) == 8
    assert np.isclose(float(result.magnification[0]), 1388.0542954476145, rtol=1e-4)


def test_public_cpu_short_lived_fold_branch_refines_its_support_trace():
    # At 64 source-limb samples one tracked fold image is physical at only a
    # single sample.  All polar quadrature orders then agree on the same
    # 2.10e-3-biased support.  Branch continuity detects that short lifetime
    # and exposes its hidden radial turn without rebuilding a denser trace.
    result = mag_binary(
        jnp.asarray([0.3667626068246604 - 0.03417414721542691j]),
        0.001,
        s=1.2,
        q=0.001,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 6.577297831349562, rtol=1e-4)


def test_public_cpu_stress_fold_branch_refines_its_hidden_tangency():
    # Re-tracing this fold at 96 samples is non-monotonic and moves the polar
    # value 1.28e-3 away from VBBL.  The one/two-sample branch lifetime instead
    # identifies a missing radial tangency directly on the 64-point trace.
    result = mag_binary(
        jnp.asarray([0.023784858167783307 + 0.23798295822016458j]),
        0.01,
        s=1.0,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 5.291371511487747, rtol=1e-3)


@pytest.mark.slow
def test_public_cpu_support_scout_does_not_undersample_a_wide_planetary_fold():
    # A 48-point source-limb scout makes the polar rules co-converge 4.08e-3
    # low here.  The production 64-point support threshold must be retained.
    result = mag_binary(
        jnp.asarray([0.35720275544848207 - 0.0369925802126353j]),
        0.01,
        s=1.2,
        q=0.001,
        backend="cpu-adaptive",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert np.isclose(float(result.magnification[0]), 3.004124947551677, rtol=1e-3)


def test_public_cpu_trace_support_resolves_previous_radial_pair_disagreement():
    # Trace-derived branch extrema now isolate the narrow radial support.  The
    # fixed pair agrees and certifies the same value without a rescue pass.
    result = mag_binary(
        jnp.asarray([0.0003539072978911648 * (1.0 + 1.0j)]),
        0.02,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) >= 5
    assert np.isclose(float(result.magnification[0]), 67.10120530670953, rtol=2e-4)


def test_public_cpu_prefilter_rejects_nonconvergent_hexadecapole_series():
    # With rho=5e-3 the local hexadecapole correction is only 2.38e-2, but the
    # series is outside its convergence region and underestimates VBBL by 13%.
    # The false-image/cusp guards must send this point to Cartesian ICRS.
    result = mag_binary(
        jnp.asarray([-0.10369483828211133 - 0.10369483828211132j]),
        0.005,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    # VBBinaryLensing BinaryMag2 with Tol=RelTol=1e-8.
    assert np.isclose(float(result.magnification[0]), 6.121655841538924, rtol=1e-4)


def test_public_cpu_small_source_uses_deeper_bernstein_isolation():
    # A depth-15 Cartesian floor misses a narrow image pair at rho=1e-4;
    # increasing limb support does not change the correlated projection error.
    result = mag_binary(
        jnp.asarray([-0.15465748917843908 - 0.15465748917843905j]),
        1.0e-4,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 6.635910797171055, rtol=1e-3)


def test_public_cpu_external_multipole_certificate_is_fail_closed():
    # The primary projection and multipole agree at 2.6e-4 but lie on the same
    # side of the reference.  The stricter external gate must evaluate the
    # independent rotated projection instead of returning the primary alone.
    result = mag_binary(
        jnp.asarray([0.005308609468367472 + 0.005308609468367471j]),
        0.01,
        s=0.85,
        q=0.03,
        backend="cpu",
        return_info=True,
    )
    assert int(result.status[0]) == 0
    assert int(result.tier[0]) != -1
    assert np.isclose(float(result.magnification[0]), 46.69088811202847, rtol=1e-3)


def test_public_cpu_prefilter_uses_the_fixed_multipole_gate():
    # These adjacent samples straddle the resonant caustic in the standard
    # seven-parameter Jacobian example.  The relaxed geometric prefilter used
    # to accept the hexadecapole estimates with 16% and 43% relative error.
    sources = jnp.asarray(
        [
            -0.08884188377486801 + 0.049694748524505034j,
            -0.08690965649124113 + 0.05199748733147592j,
        ]
    )
    result = mag_binary(
        sources,
        0.01,
        s=1.0,
        q=0.01,
        backend="cpu",
        return_info=True,
    )
    expected = np.asarray([7.254842583970952, 10.699711838720459])
    np.testing.assert_array_equal(np.asarray(result.status), 0)
    assert np.all(np.asarray(result.tier) != -1)
    np.testing.assert_allclose(np.asarray(result.magnification), expected, rtol=3e-4)


@pytest.mark.slow
def test_public_cpu_uniform_and_ld_share_the_full_solve_trigger():
    # These three source discs intersect the close-binary caustic.  The old
    # uniform-only relaxed selector accepted a non-convergent hexadecapole
    # series with 5--39% errors, while the LD selector sent the same geometry
    # to tier 7.  Both brightness profiles must now trigger full ICRS.
    sources = jnp.asarray(
        [
            -0.04715889214396113 - 0.039571009004726904j,
            -0.046392080889587774 - 0.03892757796399964j,
            -0.045625269635214426 - 0.03828414692327237j,
        ]
    )
    uniform = mag_binary(
        sources,
        0.003,
        s=0.9,
        q=0.01,
        backend="cpu",
        return_info=True,
    )
    limb_dark = mag_binary(
        sources,
        0.003,
        s=0.9,
        q=0.01,
        u1=0.5,
        backend="cpu",
        return_info=True,
    )

    np.testing.assert_array_equal(np.asarray(uniform.status), 0)
    np.testing.assert_array_equal(np.asarray(limb_dark.status), 0)
    uniform_fast = np.asarray(uniform.tier) == -1
    limb_dark_fast = np.asarray(limb_dark.tier) == -1
    np.testing.assert_array_equal(uniform_fast, limb_dark_fast)
    np.testing.assert_array_equal(uniform_fast, False)
    np.testing.assert_allclose(
        np.asarray(uniform.magnification),
        np.asarray([16.706036191630403, 13.928119339794385, 11.028569228555225]),
        rtol=3.0e-4,
    )


def test_public_cpu_prefilter_keeps_nearby_ghost_pair_out_of_multipole_path():
    # This source lies just outside the old four-radius false-image guard.  Its
    # small local hexadecapole correction is misleading: the series is not
    # converged and underestimates the finite-source value by 10%.  A five-
    # radius clearance sends both brightness profiles through full ICRS.
    source = jnp.asarray([-0.04869251465270779 - 0.040857871086181406j])
    uniform = mag_binary(
        source,
        0.002,
        s=0.9,
        q=0.01,
        backend="cpu",
        return_info=True,
    )
    limb_dark = mag_binary(
        source,
        0.002,
        s=0.9,
        q=0.01,
        u1=0.5,
        backend="cpu",
        return_info=True,
    )

    np.testing.assert_array_equal(np.asarray(uniform.status), 0)
    np.testing.assert_array_equal(np.asarray(limb_dark.status), 0)
    np.testing.assert_array_equal(np.asarray(uniform.tier) == -1, False)
    np.testing.assert_array_equal(np.asarray(limb_dark.tier) == -1, False)
    np.testing.assert_allclose(
        np.asarray(uniform.magnification),
        np.asarray([10.993295309705255]),
        rtol=1.0e-4,
    )
    np.testing.assert_allclose(
        np.asarray(limb_dark.magnification),
        np.asarray([10.648826518131237]),
        rtol=1.0e-4,
    )


def test_binary_prefilter_trigger_diagnostics_are_profile_independent():
    sources = jnp.asarray(
        [
            -0.04715889214396113 - 0.039571009004726904j,
            0.2 + 0.1j,
            0.8 - 0.3j,
        ]
    )
    uniform = _binary_prefilter(sources, 0.003, 0.0, 0.9, 0.01)
    limb_dark = _binary_prefilter(sources, 0.003, 0.5, 0.9, 0.01)

    np.testing.assert_array_equal(np.asarray(uniform[1]), np.asarray(limb_dark[1]))
    np.testing.assert_array_equal(np.asarray(uniform[2]), np.asarray(limb_dark[2]))
    np.testing.assert_array_equal(np.asarray(uniform[3]), np.asarray(limb_dark[3]))


@pytest.mark.slow
def test_public_cpu_adaptive_compatibility_path_uses_fixed_gate():
    result = mag_binary(
        jnp.asarray([-0.005632199639136862 - 0.012486743125903076j]),
        0.00988422630886768,
        s=0.5992640188539046,
        q=0.0005041898087262488,
        u1=0.308139371305733,
        backend="cpu-adaptive",
        return_info=True,
    )
    assert np.isfinite(float(result.magnification[0]))
    assert int(result.status[0]) == 0
    assert np.isclose(float(result.magnification[0]), 79.80322397033538, rtol=1e-3)


def test_public_binary_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        mag_binary(
            jnp.asarray([0.1 + 0.2j]),
            1e-2,
            s=1.0,
            q=0.3,
            backend="unknown",
        )
