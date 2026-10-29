"""Analytic accuracy, cache reuse, measured work, stopping and JVP contracts."""

from math import erf, pi, sqrt

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from microjax.inverse_ray.cpu.projection_moments import (  # noqa: E402
    integrate_adaptive_moments,
    _GK11_X,
    _GK11_W,
    _GL5_X,
    _GL5_W,
    _E3_W,
    _sum_status,
)


def polynomial_profile(x):
    values = jnp.stack([1 + x, 0.2 + x * x], axis=-1)
    return values, jnp.zeros_like(values), jnp.zeros(x.shape, dtype=jnp.int32)


def test_positive_nested_rules_are_exact_through_their_polynomial_orders():
    np.testing.assert_array_equal(_GK11_X[1::2], _GL5_X)
    assert np.all(_GK11_W > 0) and np.all(np.abs(_GK11_X) < 1)
    for nodes, weights, order in [(_GL5_X, _GL5_W, 9), (_GK11_X, _GK11_W, 17)]:
        for degree in range(order + 1):
            exact = 2 / (degree + 1) if degree % 2 == 0 else 0
            np.testing.assert_allclose(weights @ nodes**degree, exact, rtol=0, atol=5e-14)


@pytest.mark.parametrize("options", [{"budget": 15}, {"rtol": 0}, {"rtol": -1}, {"rtol": np.nan}, {"rtol": np.inf}])
def test_invalid_static_controls_are_rejected(options):
    with pytest.raises(ValueError):
        integrate_adaptive_moments(polynomial_profile, jnp.asarray([[0.0, 1.0]]), jnp.asarray([True]), **options)


def test_zero_capacity_support_returns_without_calling_profile():
    def forbidden(x):
        pytest.fail("zero-capacity support evaluated")

    result = jax.jit(lambda: integrate_adaptive_moments(forbidden, jnp.empty((0, 2)), jnp.empty(0, dtype=bool)))()
    np.testing.assert_array_equal(result.value, [0.0, 0.0])
    assert int(result.n_nodes) == 0 and int(result.status) == 0
    assert bool(result.converged) and not bool(result.budget_exhausted)


def test_empty_support_does_not_evaluate_invalid_padding():
    def forbidden(x):
        jax.debug.callback(lambda _: pytest.fail("inactive support evaluated"), x)
        return polynomial_profile(x)

    result = jax.jit(
        lambda: integrate_adaptive_moments(forbidden, jnp.full((3, 2), jnp.nan), jnp.zeros(3, dtype=bool))
    )()
    jax.block_until_ready(result)
    jax.effects_barrier()
    np.testing.assert_array_equal(result.value, [0.0, 0.0])
    assert int(result.n_nodes) == 0 and bool(result.converged)
    assert not bool(result.budget_exhausted)


@pytest.mark.parametrize("rtol", [1e-4, 3e-5])
def test_analytic_integrals_with_inactive_padding(rtol):
    a, b = 0.2, 1.3
    result = jax.jit(
        lambda: integrate_adaptive_moments(
            polynomial_profile,
            jnp.asarray([[np.nan, np.nan], [a, b], [5.0, 20.0]]),
            jnp.asarray([False, True, False]),
            rtol=rtol,
        )
    )()
    exact = [(b - a) + (b * b - a * a) / 2, 0.2 * (b - a) + (b**3 - a**3) / 3]
    np.testing.assert_allclose(result.value, exact, rtol=2e-7, atol=1e-10)
    assert bool(result.converged) and int(result.consecutive_passes) >= 2
    assert 16 <= int(result.n_nodes) <= 84
    assert np.all(np.asarray(result.error) <= rtol * np.abs(np.asarray(result.value)))


def test_zero_profile_still_requires_one_upgrade_and_two_allocations():
    def zero(x):
        return jnp.zeros((x.size, 2)), jnp.zeros((x.size, 2)), jnp.zeros(x.shape, dtype=jnp.int32)

    result = jax.jit(lambda: integrate_adaptive_moments(zero, jnp.asarray([[0.0, 1.0]]), jnp.asarray([True])))()
    assert int(result.n_nodes) == 16
    assert int(result.n_actions) == int(result.n_p_upgrades) == 1
    assert int(result.n_h_splits) == 0
    assert int(result.consecutive_passes) == 2 and bool(result.converged)


def test_six_new_rays_preserve_cached_values_error_and_status_exactly_once():
    samples = []

    def profile(x):
        jax.debug.callback(lambda xx: samples.append(np.asarray(xx)), x)
        values = jnp.stack([jnp.exp(x), jnp.exp(2 * x)], axis=-1)
        return values, 1e-8 * jnp.ones_like(values), jnp.full(x.shape, 2, dtype=jnp.int32)

    result = jax.jit(
        lambda: integrate_adaptive_moments(
            profile, jnp.asarray([[0.0, 1.0]]), jnp.asarray([True]), budget=16, status_reducer=_sum_status
        )
    )()
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert sorted(map(len, samples)) == [6, 10]
    assert len(np.unique(np.concatenate(samples))) == int(result.n_nodes) == 16
    assert int(result.status) == 32  # Eleven live rays plus five; no duplicate cache count.
    assert bool(result.budget_exhausted) and not bool(result.converged)

    def direct(lo, hi, nodes, weights, coarse_weights=None):
        angle = 0.5 * pi * (lo + 0.5 * (hi - lo) * (nodes + 1))
        x = np.sin(angle) ** 2
        jac = 0.25 * pi * (hi - lo) * np.sin(2 * angle)
        y = np.stack([np.exp(x), np.exp(2 * x)], axis=-1)
        value = np.sum((weights * jac)[:, None] * y, axis=0)
        propagated = np.sum(weights * np.abs(jac)) * np.full(2, 1e-8)
        coarse = None if coarse_weights is None else np.sum((coarse_weights * jac)[:, None] * y, axis=0)
        return value, propagated, coarse

    old = [direct(lo, hi, _GL5_X, _GL5_W, _E3_W) for lo, hi in [(0.0, 0.5), (0.5, 1.0)]]
    old_errors = [abs(value - coarse) + err for value, err, coarse in old]
    chosen = int(np.argmax([e.sum() for e in old_errors]))
    lo, hi = [(0.0, 0.5), (0.5, 1.0)][chosen]
    new_value, new_error, _ = direct(lo, hi, _GK11_X, _GK11_W)
    exact_value = old[1 - chosen][0] + new_value
    exact_error = old_errors[1 - chosen] + abs(new_value - old[chosen][0]) + new_error
    np.testing.assert_allclose(result.value, exact_value, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(result.error, exact_error, rtol=1e-9, atol=3e-15)


def test_each_moment_and_propagated_error_control_stopping():
    def uncertain_tiny_second(x):
        values = jnp.broadcast_to(jnp.asarray([1.0, 1e-12]), (x.size, 2))
        error = jnp.broadcast_to(jnp.asarray([0.0, 1e-15]), values.shape)
        return values, error, jnp.zeros(x.shape, dtype=jnp.int32)

    result = jax.jit(
        lambda: integrate_adaptive_moments(uncertain_tiny_second, jnp.asarray([[0.0, 1.0]]), jnp.asarray([True]))
    )()
    assert not bool(result.converged) and bool(result.budget_exhausted)
    assert 74 <= int(result.n_nodes) <= 84
    assert float(result.error[1]) > 1e-4 * float(result.value[1])
    assert int(result.status) == 0


def test_relative_stopping_has_no_absolute_scale_floor():
    def solve(scale):
        def scaled(x):
            values, error, status = polynomial_profile(x)
            return scale * values, scale * error, status

        return integrate_adaptive_moments(scaled, jnp.asarray([[0.2, 1.3]]), jnp.asarray([True]))

    run = jax.jit(solve)
    ordinary = run(jnp.asarray(1.0))
    for scale in [1e-40, 1e40]:
        result = run(jnp.asarray(scale))
        np.testing.assert_allclose(result.value / scale, ordinary.value, rtol=2e-13, atol=0)
        assert int(result.n_nodes) == int(ordinary.n_nodes)
        assert bool(result.converged) == bool(ordinary.converged)


def test_refinement_work_matches_real_profile_calls_for_localized_structure():
    samples = []

    def profile(x):
        jax.debug.callback(lambda xx: samples.append(np.asarray(xx)), x)
        values = jnp.stack([jnp.exp(-40 * (x - 0.4) ** 2), jnp.exp(-160 * (x - 2.7) ** 2)], axis=-1)
        return values, jnp.zeros_like(values), jnp.zeros(x.shape, dtype=jnp.int32)

    result = jax.jit(
        lambda: integrate_adaptive_moments(
            profile, jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), jnp.asarray([True, True, True]), rtol=3e-5
        )
    )()
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert sum(map(len, samples)) == int(result.n_nodes)
    assert all(len(x) in (6, 10) for x in samples)
    assert int(result.n_nodes) <= 84 * 3
    assert int(result.n_nodes) == 30 + 6 * int(result.n_p_upgrades) + 10 * int(result.n_h_splits)
    assert int(result.n_h_splits) > 0 and int(result.n_p_upgrades) > 0

    def exact(rate, center):
        return sqrt(pi / rate) / 2 * (erf(sqrt(rate) * (3 - center)) + erf(sqrt(rate) * center))

    np.testing.assert_allclose(result.value, [exact(40, 0.4), exact(160, 2.7)], rtol=3e-5, atol=1e-10)


def test_existing_root_status_is_preserved_and_prevents_successful_stopping():
    def flagged(x):
        values, errors, _ = polynomial_profile(x)
        return values, errors, jnp.full(x.shape, 4, dtype=jnp.int32)

    result = jax.jit(lambda: integrate_adaptive_moments(flagged, jnp.asarray([[0.2, 1.3]]), jnp.asarray([True])))()
    assert int(result.status) == 4
    assert not bool(result.converged) and bool(result.budget_exhausted)


def test_h_split_replaces_parent_status_without_recounting_discarded_rays():
    def flagged_symmetric(x):
        values = jnp.stack([jnp.exp((x - 0.5) ** 2), jnp.exp(2 * (x - 0.5) ** 2)], axis=-1)
        return values, jnp.zeros_like(values), jnp.full(x.shape, 2, dtype=jnp.int32)

    result = jax.jit(
        lambda: integrate_adaptive_moments(
            flagged_symmetric,
            jnp.asarray([[0.0, 1.0]]),
            jnp.asarray([True]),
            budget=32,
            status_reducer=_sum_status,
        )
    )()
    assert int(result.n_p_upgrades) == 2 and int(result.n_h_splits) == 1
    assert int(result.n_nodes) == 32
    # The discarded GK11 parent had eleven rays. Its two new GL5 children
    # have ten, while the other original GK11 panel still has eleven.
    assert int(result.status) == 2 * (10 + 11)
    assert bool(result.budget_exhausted) and not bool(result.converged)


def test_dynamic_active_cell_counts_obey_the_shared_work_cap():
    cells = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    solve = jax.jit(lambda active: integrate_adaptive_moments(polynomial_profile, cells, active))
    for count in range(5):
        result = solve(jnp.arange(4) < count)
        assert int(result.n_nodes) <= 84 * count
        np.testing.assert_allclose(
            result.value, [count + count**2 / 2, 0.2 * count + count**3 / 3], rtol=2e-7, atol=1e-9
        )


def test_moving_endpoints_forward_jacobian_and_hessian_retain_cached_dependencies():
    def moments(x):
        value = jnp.stack([x, x * x], axis=-1)
        return value, jnp.zeros_like(value), jnp.zeros(x.shape, dtype=jnp.int32)

    def integral(bounds):
        return integrate_adaptive_moments(moments, bounds[None, :], jnp.asarray([True]), rtol=3e-5).value

    a, b = 0.2, 1.3
    value, first, second = jax.jit(
        lambda bounds: (integral(bounds), jax.jacfwd(integral)(bounds), jax.jacfwd(jax.jacfwd(integral))(bounds))
    )(jnp.asarray([a, b]))
    np.testing.assert_allclose(value, [(b * b - a * a) / 2, (b**3 - a**3) / 3], rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(first, [[-a, b], [-a * a, b * b]], rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(second, [[[-1.0, 0.0], [0.0, 1.0]], [[-2 * a, 0.0], [0.0, 2 * b]]], rtol=2e-7, atol=1e-8)


@pytest.mark.slow
def test_public_cpu_uses_cached_moments_in_both_ld_charts(monkeypatch):
    from microjax.inverse_ray import lightcurve
    from microjax.inverse_ray.cpu import one_shot

    observed = []
    cached = one_shot._integrate_ld_moments

    def cached_with_work(profile, cells, active, **options):
        result = cached(profile, cells, active, **options)
        jax.debug.callback(
            lambda cells, nodes, p, h: observed.append(("cached", int(cells), int(nodes), int(p), int(h))),
            jnp.sum(active),
            result.n_nodes,
            result.n_p_upgrades,
            result.n_h_splits,
        )
        return result

    # Exercise both charts and a former radial case through the actual public
    # scheduler. All LD full solves now share the cached moment allocator.
    cases = [
        (-1.505891145751099 - 0.0010483906073588275j, 0.005477225575051662, 0.5, 8),
        (-1.3045835407076574 - 0.003035878947864619j, 3e-5, 0.5416666666666666, 5),
        (-0.1742947428830735 - 0.0009439342481879419j, 5.751765563428599e-5, 0.9166666666666666, 5),
    ]
    jax.clear_caches()
    try:
        with monkeypatch.context() as selected:
            selected.setattr(one_shot, "_integrate_ld_moments", cached_with_work)
            for source, rho, s, tier in cases:
                observed.clear()
                result = lightcurve.mag_binary(
                    jnp.asarray([source]),
                    rho,
                    s=s,
                    q=1e-6,
                    u1=0.5,
                    backend="cpu",
                    return_info=True,
                )
                jax.block_until_ready(result)
                jax.effects_barrier()
                assert int(result.tier[0]) == tier
                assert np.isfinite(float(result.magnification[0]))
                assert len(observed) == 1
                tag, cells, nodes, p, h = observed[0]
                assert tag == "cached" and p >= 1
                assert nodes == 10 * cells + 6 * p + 10 * h
                assert nodes <= 84 * cells
    finally:
        # Clear JIT closures that captured instrumented callbacks before the
        # monkeypatch was restored, so later tests see the normal graph.
        jax.clear_caches()
