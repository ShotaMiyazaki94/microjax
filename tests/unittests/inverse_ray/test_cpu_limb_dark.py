import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_limb_dark import (
    mag_limb_dark_angular_moment_refined,
)
from microjax.inverse_ray.cpu.cartesian_limb_dark import (
    _bernstein_negative_intervals,
    _mag_limb_dark_cartesian_impl,
    mag_limb_dark_cartesian_adaptive,
)
from microjax.inverse_ray.cpu.limb_dark import (
    mag_limb_dark_cpu,
    mag_limb_dark_cpu_fixed,
)
from microjax.inverse_ray.cpu.one_shot import (
    mag_limb_dark_cpu_one_shot,
    mag_uniform_cpu_one_shot,
)
from microjax.inverse_ray.cpu.uniform import mag_uniform_cpu_fixed
from microjax.inverse_ray.integrators.limb_dark import mag_limb_dark_boundary


@jax.jit
def _limb_dark_fixed_case(source, rho, separation, mass_ratio, u1):
    return mag_limb_dark_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        u1=u1,
        n_limb=128,
        radial_splits=4,
        return_info=True,
    )


@jax.jit
def _limb_dark_zero_case(source, rho, separation, mass_ratio):
    return mag_limb_dark_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        u1=0.0,
        n_limb=32,
        radial_splits=1,
    )


@jax.jit
def _uniform_zero_case(source, rho, separation, mass_ratio):
    return mag_uniform_cpu_fixed(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        n_limb=32,
        radial_splits=1,
    )


@jax.jit
def _limb_dark_boundary_case(source, rho, separation, mass_ratio, u1):
    return mag_limb_dark_boundary(
        source,
        rho,
        s=separation,
        q=mass_ratio,
        u1=u1,
        Nlimb=64,
        angular_atol=1e-7,
        relative_tolerance=1e-6,
        robust_roots=True,
        certify_topology=True,
        return_info=True,
    )


def test_cpu_zero_limb_darkening_reduces_to_uniform():
    source = jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128)
    limb = _limb_dark_zero_case(source, 1e-2, 1.0, 0.3)
    uniform = _uniform_zero_case(source, 1e-2, 1.0, 0.3)
    assert np.isclose(float(limb), float(uniform), rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_one_shot_geometry_route_is_shared_by_brightness_profiles():
    sources = jnp.asarray([0.05 + 0.02j], dtype=jnp.complex128)
    common = dict(rho=5.0e-3, s=1.0, q=1.0e-3, return_state=True)
    solve_uniform = jax.jit(
        jax.vmap(lambda source: mag_uniform_cpu_one_shot(source, **common))
    )
    solve_limb_dark = jax.jit(
        jax.vmap(
            lambda source: mag_limb_dark_cpu_one_shot(
                source,
                u1=0.5,
                **common,
            )
        )
    )
    (uniform, uniform_state) = solve_uniform(sources)
    (limb_dark, limb_dark_state) = solve_limb_dark(sources)
    np.testing.assert_array_equal(
        np.asarray(uniform.stage),
        np.asarray(limb_dark.stage),
    )
    for uniform_value, limb_dark_value in zip(uniform_state, limb_dark_state):
        np.testing.assert_allclose(
            np.asarray(uniform_value),
            np.asarray(limb_dark_value),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "w,rho,s,q,u1",
    [
        (0.1 + 0.2j, 1e-2, 1.0, 0.3, 0.5),
        (-0.35 + 0.17j, 3e-3, 0.7, 1e-2, 0.7),
        (0.6 - 0.2j, 5e-3, 1.6, 0.1, 0.3),
    ],
)
def test_cpu_limb_dark_fixed_matches_existing_boundary(w, rho, s, q, u1):
    w = jnp.asarray(w, dtype=jnp.complex128)
    actual = _limb_dark_fixed_case(w, rho, s, q, u1)
    expected = _limb_dark_boundary_case(w, rho, s, q, u1)
    assert int(actual.status) == 0
    assert np.isclose(
        float(actual.magnification),
        float(expected.magnification),
        rtol=2e-5,
        atol=1e-8,
    )


@pytest.mark.slow
def test_cpu_limb_dark_hierarchy_supports_jit():
    solve = jax.jit(
        lambda source: mag_limb_dark_cpu(
            source,
            1e-2,
            s=1.0,
            q=0.3,
            u1=0.5,
            rtol=1e-4,
        )
    )
    value = solve(jnp.asarray(0.1 + 0.2j, dtype=jnp.complex128))
    assert np.isfinite(float(value))


@pytest.mark.slow
def test_angular_profile_moment_resolves_caustic_residual():
    result = mag_limb_dark_angular_moment_refined(
        -0.10794172585680531 - 0.1079417258568053j,
        0.005,
        s=0.85,
        q=0.03,
        u1=0.5,
        rtol=1e-3,
        return_info=True,
    )
    assert int(result.status) == 0
    assert np.isclose(float(result.magnification), 11.216178805288225, rtol=1e-5)


def test_bernstein_cartesian_profile_matches_companion_intervals():
    source = jnp.asarray(-0.05 * jnp.exp(0.25j * jnp.pi), dtype=jnp.complex128)

    def solve(root_mode):
        return jax.jit(
            lambda value: _mag_limb_dark_cartesian_impl(
                value,
                0.005,
                s=0.85,
                q=0.03,
                u1=0.5,
                n_slice=8,
                n_profile=8,
                n_limb=64,
                axis=1.0j * value / jnp.abs(value),
                root_mode=root_mode,
                return_info=True,
            )
        )(source)

    companion = solve("companion")
    bernstein = solve("bernstein")
    assert int(bernstein.invalid_root_count) == 0
    assert int(bernstein.status) == 0
    assert np.isclose(
        float(bernstein.magnification),
        float(companion.magnification),
        rtol=2.0e-5,
    )


def test_bernstein_isolation_rejects_unresolved_close_root_pair():
    roots = np.asarray([-0.8, -0.4, 0.125, 0.125001, 0.5, 0.9])
    coefficients = jnp.asarray(np.poly(roots)[None, :], dtype=jnp.float64)
    _, _, _, invalid = jax.jit(
        lambda values: _bernstein_negative_intervals(
            values,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
    )(coefficients)
    assert int(invalid[0]) > 0


@pytest.mark.slow
def test_cartesian_limb_dark_uses_lazy_companion_refinement():
    source = jnp.asarray(
        -0.01167894083040845 - 0.011678940830408447j,
        dtype=jnp.complex128,
    )
    solve = jax.jit(
        lambda value: mag_limb_dark_cartesian_adaptive(
            value,
            0.005,
            s=0.85,
            q=0.03,
            u1=0.5,
            rtol=1.0e-3,
            return_info=True,
        )
    )
    result = solve(source)
    assert int(result.stage) == 2
    assert int(result.status) == 0
    assert np.isclose(float(result.magnification), 80.5470205519234, rtol=2.0e-4)


@pytest.mark.slow
def test_cartesian_limb_dark_external_cross_certificate_is_fail_closed():
    source = jnp.asarray(-0.05 * jnp.exp(0.25j * jnp.pi), dtype=jnp.complex128)

    def solve(external):
        return jax.jit(
            lambda value, estimate: mag_limb_dark_cartesian_adaptive(
                value,
                0.005,
                s=0.85,
                q=0.03,
                u1=0.5,
                rtol=1.0e-3,
                external_magnification=estimate,
                return_info=True,
            )
        )(source, external)

    baseline = mag_limb_dark_cartesian_adaptive(
        source,
        0.005,
        s=0.85,
        q=0.03,
        u1=0.5,
        rtol=1.0e-3,
        return_info=True,
    )
    accepted = solve(baseline.magnification)
    rejected = solve(2.0 * baseline.magnification)
    assert int(accepted.status) == 0
    assert int(accepted.n_slices) < int(baseline.n_slices)
    accepted_relative_difference = abs(
        float(accepted.magnification - baseline.magnification)
    ) / max(abs(float(baseline.magnification)), 1.0)
    assert accepted_relative_difference <= 0.05 * 1.0e-3
    assert int(rejected.status) == int(baseline.status)
    assert int(rejected.n_slices) == int(baseline.n_slices)
    rejected_relative_difference = abs(
        float(rejected.magnification - baseline.magnification)
    ) / max(abs(float(baseline.magnification)), 1.0)
    assert rejected_relative_difference <= 0.10 * 1.0e-3
    assert not np.isclose(
        float(rejected.magnification),
        float(2.0 * baseline.magnification),
    )
