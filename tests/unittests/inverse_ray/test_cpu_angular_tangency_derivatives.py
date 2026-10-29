"""Primary angular tangency motion, independent of the quadrature rule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_moment import (
    _implicit_angular_tangency_angles,
    _refine_angular_tangencies,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.fast
@pytest.mark.parametrize("exchange", [False, True])
def test_primary_tangency_motion_matches_resolved_endpoint_finite_differences(exchange):
    # Ten coarse limb samples, two per image, for a smooth five-image source.
    # Fixed seeds isolate the endpoint equation from limb tracking and from
    # the quadrature allocation. Each Newton solve must actually converge.
    seeds = jnp.asarray([
        1.082143212340037-0.2759866029330049j,
        1.0990893969281754-0.24842755026026914j,
        0.3899890426374073+0.9669039823586657j,
        0.34752456443655644+0.982937802847542j,
        0.4666311674596826-0.07100334734405354j,
        0.46648522582902846-0.06384951771582899j,
        -0.8357211907827058-0.45745505254410546j,
        -0.8048701607838504-0.49741961477441465j,
        0.5987472906774401-0.6012250540273858j,
        0.6323127760579488-0.5831711548895274j,
    ])
    point = jnp.asarray([.1, .2, .01, 1., .3])
    if exchange:
        seeds = -seeds
        point = point.at[:2].multiply(-1).at[4].set(1 / point[4])
    angles = jnp.mod(jnp.angle(seeds), 2*jnp.pi)
    active = jnp.ones(seeds.shape, dtype=bool)

    def refine(parameters, implicit):
        x, y, rho, s, q = parameters
        return _refine_angular_tangencies(
            x+1j*y, rho, s=s, q=q, initial_images=seeds,
            initial_angles=angles, fallback_angles=angles,
            active=active, required=active, bracketed=~active,
            implicit_derivatives=implicit,
        )

    evaluate = jax.jit(lambda p: refine(p, True))
    refined, valid = evaluate(point)
    assert np.asarray(valid).all()
    stopped, stopped_valid = jax.jit(lambda p: refine(p, False))(point)
    np.testing.assert_array_equal(refined, stopped)
    np.testing.assert_array_equal(valid, stopped_valid)
    np.testing.assert_array_equal(
        jax.jit(jax.jacfwd(lambda p: refine(p, False)[0]))(point), 0,
    )

    jacobian = np.asarray(jax.jit(jax.jacfwd(lambda p: refine(p, True)[0]))(point))
    assert np.isfinite(jacobian).all()
    steps = np.asarray([point[2]]*4+[point[4]])*1e-5
    finite_difference = []
    for i, step in enumerate(steps):
        upper, upper_valid = evaluate(point.at[i].add(step))
        lower, lower_valid = evaluate(point.at[i].add(-step))
        assert np.asarray(upper_valid & lower_valid).all()
        finite_difference.append(np.asarray(upper-lower)/(2*step))
    np.testing.assert_allclose(
        jacobian, np.stack(finite_difference, axis=-1),
        rtol=2e-5, atol=2e-7,
    )


@pytest.mark.fast
def test_unconverged_primary_tangencies_have_zero_motion():
    # The auxiliary Newton coordinates may be undefined in rejected lanes;
    # those lanes must retain the historical zero endpoint derivative.
    angles = jnp.asarray([.3, jnp.nan])
    radii = jnp.asarray([1., jnp.nan])
    accepted = jnp.zeros(2, dtype=bool)
    _, tangent = jax.jvp(
        lambda a, r, w: _implicit_angular_tangency_angles(
            a, r, accepted, w, jnp.asarray(.01), jnp.asarray(1.), jnp.asarray(.3),
        ),
        (angles, radii, jnp.asarray(.1+.2j)),
        (jnp.full_like(angles, jnp.nan), jnp.full_like(radii, jnp.nan),
         jnp.asarray(1.+1j)),
    )
    np.testing.assert_array_equal(tangent, 0)
