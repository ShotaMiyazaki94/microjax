"""Physical coverage and implicit-event regressions for angular cells."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_projection_support import (
    _angular_support_cells_from_events, _critical_event_positions,
)
from microjax.inverse_ray.cpu.angular_limb_dark import _physical_ray_intervals
from microjax.inverse_ray.cpu.support_events import _event_lens_map
from microjax.inverse_ray.cpu.one_shot import _trace_state
from microjax.inverse_ray.cpu.support_events import (
    _critical_event_image_seeds, _event_equations, _refine_support_events,
)

jax.config.update('jax_enable_x64', True)


@jax.jit
def _geometry(w, rho, s, q):
    trace, state = _trace_state(w, rho, s=s, q=q, n_limb=56)
    support = _angular_support_cells_from_events(
        w, rho, s=s, q=q,
        physical_limb=trace.image_limb, physical_mask=trace.physical_mask,
        topology_uncertain=state.topology_uncertain,
        minimum_ghost_residual=state.ghost_residual_ratio*rho,
        limb_topology=state.limb_topology,
        neighbors=(trace.previous_limb, trace.following_limb,
                   trace.previous_mask, trace.following_mask),
        edge_activity=(trace.incoming_edge_active, trace.outgoing_edge_active),
    )
    return trace, support


@pytest.mark.fast
@pytest.mark.parametrize('source', [
    -5.822987259370673e-5 + 3.22130157927591e-5j,
    7.975153506638603e-6 - 8.339375025276927e-5j,
])
def test_near_origin_image_arcs_keep_their_physical_trace_coverage(source):
    """Unreliable new tangency solves must not erase an observed image arc."""

    trace, support = _geometry(*map(jnp.asarray, (
        source, 3e-5, 2/3, 5.623413251903491e-5,
    )))
    cells = np.asarray(support.cells)[np.asarray(support.active)]
    physical = np.asarray(trace.image_limb)[np.asarray(trace.physical_mask)]
    angles = np.mod(np.angle(physical), 2*np.pi)
    covered = np.any(
        (angles[:, None] >= cells[None, :, 0]-1e-12)
        & (angles[:, None] <= cells[None, :, 1]+1e-12), axis=1,
    )
    assert covered.all()
    assert np.any(physical.real < -.5)
    assert support.cells.shape == (49, 2)
    assert np.all(np.diff(cells, axis=1) <= np.pi/2 + 1e-12)


@pytest.mark.fast
def test_critical_cut_jvp_satisfies_the_physical_event_equations():
    """Differentiate the event equations, excluding rejected Newton lanes."""

    point = jnp.asarray([.8178052469807563, -.030664159824672937,
                         .020133555236026012, 1.5, .0006105402296585327])
    w, rho, s, q = point[0]+1j*point[1], *point[2:]
    trace, _ = _geometry(w, rho, s, q)
    probe = _critical_event_image_seeds(
        trace.image_limb, trace.physical_mask, w, rho, s, q,
    )
    seeds = jax.lax.stop_gradient(probe.critical_images)

    def roots(parameters):
        x, y, rho, s, q = parameters
        z, valid, _ = _refine_support_events(
            seeds, x+1j*y, rho, s, q, critical=True,
        )
        return _critical_event_positions(z, valid, x+1j*y, rho, s, q)

    def residual(parameters):
        x, y, rho, s, q = parameters
        z = roots(parameters)
        xy = jnp.stack([z.real, z.imag], axis=-1)
        return jax.vmap(lambda position: _event_equations(
            position, x+1j*y, rho, s, q, critical=True, axis=1+0j,
        ))(xy)

    jacobian = np.asarray(jax.jit(jax.jacfwd(roots))(point))
    assert np.isfinite(jacobian).all()
    np.testing.assert_allclose(np.asarray(residual(point)), 0, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jax.jacfwd(residual))(point)), 0, atol=1e-7,
    )
    evaluate = jax.jit(roots)
    steps = np.asarray([rho, rho, rho, rho, q])*1e-5
    finite_difference = np.stack([
        (np.asarray(evaluate(point.at[i].add(step)))
         - np.asarray(evaluate(point.at[i].add(-step)))) / (2*step)
        for i, step in enumerate(steps)
    ], axis=-1)
    np.testing.assert_allclose(
        jacobian, finite_difference, rtol=2e-5, atol=2e-7,
    )

    # An invalid event has no cut motion even if its upstream Newton tangent
    # is undefined. The accepted event still has its physical derivative.
    position = jnp.concatenate([seeds, jnp.asarray([jnp.nan+1j*jnp.nan])])
    valid = jnp.asarray([True, True, False])
    _, tangent = jax.jvp(
        lambda z: _critical_event_positions(z, valid, w, rho, s, q),
        (position,), (jnp.full_like(position, jnp.nan),),
    )
    np.testing.assert_array_equal(np.asarray(tangent), 0)


@pytest.mark.fast
@pytest.mark.parametrize('exchange',[False,True])
def test_root_slot_endpoints_do_not_omit_a_physically_occupied_angular_sector(exchange):
    w=0.6661323040911742+0.4848190652006763j
    rho=0.2720446295173432
    s=1.3333333333333333
    q=0.1389495494373136
    angle=6.03
    if exchange:
        w,q,angle=-w,1/q,(angle+np.pi)%(2*np.pi)
    w,rho,s,q=map(jnp.asarray,(w,rho,s,q))
    trace,support=_geometry(w,rho,s,q)
    lower,upper,inside,_=_physical_ray_intervals(jnp.asarray([angle]),w,rho,s=s,q=q)
    selected=np.asarray(inside)[0]
    assert selected.any()
    # Verify occupancy by the physical lens map as well as the sextic roots.
    images=.5*(lower[0]+upper[0])*jnp.exp(1j*angle)
    distance=np.abs(np.asarray(_event_lens_map(images,s,q)-w))
    assert np.any(selected&(distance<float(rho)))
    cells=np.asarray(support.cells)[np.asarray(support.active)]
    assert np.any((cells[:,0]<angle)&(angle<cells[:,1]))
