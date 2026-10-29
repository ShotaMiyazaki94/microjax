"""Continuous support events for components absent from the sampled trace."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.cartesian_moment import (
    _cartesian_support_cells_from_trace,
)
from microjax.inverse_ray.cpu.support import trace_binary_source_limb_record
from microjax.inverse_ray.cpu.support_events import (
    _cartesian_support_cells_with_events,
    _critical_event_image_seeds,
)

jax.config.update("jax_enable_x64", True)


def _trace_and_supports(source, rho, s, q, n_limb):
    source, rho, s, q = map(jnp.asarray, (source, rho, s, q))
    trace = jax.jit(lambda w: trace_binary_source_limb_record(
        w, rho, s=s, q=q, n_limb=n_limb
    ))(source)
    common = dict(
        w_center=source, rho=rho, s=s, q=q, axis=1 + 0j,
        refine_extrema=False, topology_uncertain=jnp.asarray(False),
        minimum_ghost_residual=jnp.asarray(jnp.inf),
        limb_topology=jnp.asarray(False),
        neighbors=(trace.previous_limb, trace.following_limb,
                   trace.previous_mask, trace.following_mask),
        edge_activity=(trace.incoming_edge_active, trace.outgoing_edge_active),
    )
    old = jax.jit(lambda z, mask: _cartesian_support_cells_from_trace(
        z, mask, **common
    ))(trace.image_limb, trace.physical_mask)
    new = jax.jit(lambda z, mask: _cartesian_support_cells_with_events(
        z, mask, **common
    ))(trace.image_limb, trace.physical_mask)
    event = jax.jit(lambda z, mask: _critical_event_image_seeds(
        z, mask, source, rho, s, q
    ))(trace.image_limb, trace.physical_mask)
    return trace, old, new, event


@pytest.mark.fast
def test_fold_pair_born_and_lost_between_trace_phases_gets_support():
    """The 56-point trace has three images everywhere and omits a real pair."""

    source = 0.07208739252567238 - 0.0038756551613259704j
    rho, s, q = 3e-5, 0.7916666666666666, 0.011787686347935866
    trace, old, new, event = _trace_and_supports(source, rho, s, q, 56)
    np.testing.assert_array_equal(np.asarray(trace.image_counts), 3)
    assert np.asarray(event.critical_valid).all()
    assert np.asarray(event.physical_mask).sum() == 5

    # Evaluate the physical lens equation independently of the event helper.
    def mapped(images):
        primary = -s * q / (1 + q)
        secondary = s / (1 + q)
        return images - 1 / (1 + q) / (np.conj(images) - primary) - (
            q / (1 + q) / (np.conj(images) - secondary)
        )

    crossings = mapped(np.asarray(event.critical_images)) - source
    np.testing.assert_allclose(np.abs(crossings), rho, rtol=1e-8, atol=0)
    phases = np.mod(np.angle(crossings), 2 * np.pi)
    bins = np.floor(phases / (2 * np.pi / 56)).astype(int)
    assert bins[0] == bins[1]
    assert phases.min() < float(event.source_phase) < phases.max()
    probe_source = source + rho * np.exp(1j * float(event.source_phase))
    np.testing.assert_allclose(mapped(np.asarray(event.images)), probe_source,
                               rtol=0, atol=1e-10)

    old_cells = np.asarray(old[0])[np.asarray(old[1])]
    new_cells = np.asarray(new[0])[np.asarray(new[1])]
    old_cuts = np.unique(old_cells.ravel())
    new_cuts = np.unique(new_cells.ravel())
    assert new[0].shape == (21, 2)
    assert np.isin(old_cuts, new_cuts).all()
    assert old_cuts.max() < 1.0
    # Projection endpoints of the missed component, independently checked
    # against a dense image trace and the continuous tangency equations.
    np.testing.assert_allclose(
        new_cuts[new_cuts > 1.0],
        [1.037548030664129, 1.037677676410301],
        rtol=0, atol=3e-10,
    )


@pytest.mark.fast
def test_closed_five_image_trace_adds_no_event_endpoints():
    """A trace without rejected roots cannot enable a fabricated event pair."""

    trace, old, new, event = _trace_and_supports(
        -0.03250042182889892 - 0.0002573669068225559j,
        0.0004053600464421104, 1.375, 0.03162277660168379, 64,
    )
    assert np.asarray(trace.physical_mask).all()
    assert not np.asarray(event.physical_mask).any()
    np.testing.assert_array_equal(
        np.asarray(new[0])[np.asarray(new[1])],
        np.asarray(old[0])[np.asarray(old[1])],
    )
    for before, after in zip(old[2:], new[2:]):
        np.testing.assert_array_equal(before, after)
