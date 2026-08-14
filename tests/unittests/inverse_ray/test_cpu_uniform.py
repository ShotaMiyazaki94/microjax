import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.support import (
    build_radial_support,
    tracked_limb_neighbors,
)
from microjax.inverse_ray.cpu.uniform import (
    CPU_TIER_EXHAUSTED,
    mag_uniform_cpu,
    mag_uniform_cpu_fixed,
)
from microjax.inverse_ray.integrators.uniform import mag_uniform_boundary


@jax.jit
def _uniform_fixed_case(source, rho, separation, mass_ratio):
    return mag_uniform_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        n_limb=32,
        return_info=True,
    )


@jax.jit
def _uniform_hierarchy_case(source, rho, separation, mass_ratio):
    return mag_uniform_cpu(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        rtol=1e-6,
        return_info=True,
    )


@jax.jit
def _uniform_default_case(source, rho, separation, mass_ratio):
    return mag_uniform_cpu(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        rtol=1e-3,
        return_info=True,
    )


@jax.jit
def _uniform_boundary_case(source, rho, separation, mass_ratio):
    return mag_uniform_boundary(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        Nlimb=64,
        angular_atol=1e-8,
        relative_tolerance=1e-7,
        robust_roots=True,
        certify_topology=True,
        return_info=True,
    )


@pytest.mark.parametrize(
    "w,rho,s,q",
    [
        (0.1 + 0.2j, 1e-2, 1.0, 0.3),
        (-0.35 + 0.17j, 3e-3, 0.7, 1e-2),
        (0.6 - 0.2j, 5e-3, 1.6, 0.1),
    ],
)
def test_cpu_uniform_fixed_matches_certified_boundary(w, rho, s, q):
    w = jnp.asarray(w, dtype=jnp.complex128)
    actual = _uniform_fixed_case(w, rho, s, q)
    expected = _uniform_boundary_case(w, rho, s, q)

    assert int(actual.status) == 0
    assert np.isfinite(float(actual.magnification))
    assert np.isclose(
        float(actual.magnification),
        float(expected.magnification),
        rtol=2e-4,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "w,rho,s,q",
    [
        (0.1 + 0.2j, 1e-2, 1.0, 0.3),
        (-0.35 + 0.17j, 3e-3, 0.7, 1e-2),
        (0.6 - 0.2j, 5e-3, 1.6, 0.1),
    ],
)
def test_cpu_uniform_hierarchy_reaches_high_accuracy(w, rho, s, q):
    w = jnp.asarray(w, dtype=jnp.complex128)
    actual = _uniform_hierarchy_case(w, rho, s, q)
    expected = _uniform_boundary_case(w, rho, s, q)
    assert int(actual.tier) == 2
    assert int(actual.status) == CPU_TIER_EXHAUSTED
    assert np.isclose(
        float(actual.magnification),
        float(expected.magnification),
        rtol=5e-6,
        atol=1e-8,
    )


def test_cpu_radial_support_is_disjoint_and_nonempty():
    support = build_radial_support(
        jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128),
        1e-2,
        s=1.0,
        q=0.3,
        n_limb=32,
    )
    intervals = np.asarray(support.intervals)[np.asarray(support.active)]
    assert intervals.shape[0] > 0
    assert np.all(intervals[:, 1] > intervals[:, 0])
    assert np.all(intervals[1:, 0] >= intervals[:-1, 1])


def test_tracked_limb_neighbors_close_a_permuted_root_loop():
    first = jnp.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=jnp.complex128)
    last = jnp.asarray([10.1, 30.1, 0.1, 40.1, 20.1], dtype=jnp.complex128)
    limb = jnp.stack((first, first + 0.05j, last), axis=1)
    mask = jnp.asarray(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
            [False, False, True],
            [True, True, False],
        ]
    )
    previous, following, previous_mask, following_mask = tracked_limb_neighbors(
        limb, mask
    )
    np.testing.assert_allclose(np.asarray(previous[:, 0].real), np.arange(5) * 10 + 0.1)
    np.testing.assert_allclose(
        np.asarray(following[:, -1].real),
        np.asarray([10.0, 30.0, 0.0, 40.0, 20.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(previous_mask[:, 0]),
        np.asarray([True, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        np.asarray(following_mask[:, -1]),
        np.asarray([False, False, True, True, True]),
    )


def test_cpu_uniform_fixed_supports_jit():
    solve = jax.jit(
        lambda source: mag_uniform_cpu_fixed(
            source,
            1e-2,
            s=1.0,
            q=0.3,
            n_limb=32,
        )
    )
    value = solve(jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128))
    assert np.isfinite(float(value))


def test_cpu_uniform_exposed_caustic_uses_ghost_support():
    sources = jnp.asarray(
        [
            -0.1902368579544647 + 0.7383570479131436j,
            -0.1845738496739989 + 0.7403391538113067j,
        ],
        dtype=jnp.complex128,
    )
    references = np.asarray([7.632499052150089, 2.3027897176499414])
    solve = jax.jit(
        lambda source: mag_uniform_cpu(
            source,
            0.003,
            s=0.8,
            q=0.3,
            rtol=1e-4,
            return_info=True,
        )
    )
    results = [solve(source) for source in sources]
    values = np.asarray([float(result.magnification) for result in results])

    assert all(int(result.tier) == 2 for result in results)
    assert all(int(result.status) == 0 for result in results)
    np.testing.assert_allclose(values, references, rtol=3e-6)


def test_cpu_uniform_rejects_calibrated_marginal_false_accept():
    """A stratified VBBL-sweep regression must remain explicitly exhausted."""

    result = _uniform_default_case(
        -0.182686139104828 + 0.7410001087528486j,
        1e-2,
        separation=0.8,
        mass_ratio=0.3,
    )
    assert int(result.status) & CPU_TIER_EXHAUSTED


def test_cpu_uniform_rejects_buried_planetary_caustic_false_accept():
    """Consistent radial tiers cannot certify a wholly enclosed caustic."""

    result = _uniform_default_case(
        -0.0010946356833467202 - 0.006172611801159194j,
        0.00988422630886768,
        separation=0.5992640188539046,
        mass_ratio=0.0005041898087262488,
    )
    assert int(result.status) & CPU_TIER_EXHAUSTED
