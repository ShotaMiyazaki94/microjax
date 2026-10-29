"""Physical regressions for continuous Cartesian projection cells."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.line_roots import binary_line_level_set_roots
from microjax.inverse_ray.cpu.angular_moment import ANGULAR_MOMENT_TOPOLOGY
from microjax.inverse_ray.cpu.cartesian_limb_dark import _mag_limb_dark_cartesian_impl
from microjax.inverse_ray.cpu.one_shot import _fixed_cartesian_result
from microjax.inverse_ray.cpu.projection_support import (
    _cartesian_support_cells_from_events,
    _observed_critical_events,
)
from microjax.inverse_ray.cpu.support import trace_binary_source_limb_record

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["n_limb", "maximum_extrema"])
def _geometry(source, rho, s, q, *, n_limb=56, maximum_extrema=20):
    trace = trace_binary_source_limb_record(source, rho, s=s, q=q, n_limb=n_limb)
    neighbors = (trace.previous_limb, trace.following_limb,
                 trace.previous_mask, trace.following_mask)
    support = _cartesian_support_cells_from_events(
        trace.image_limb, trace.physical_mask,
        w_center=source, rho=rho, s=s, q=q, axis=1 + 0j,
        topology_uncertain=jnp.asarray(False),
        minimum_ghost_residual=jnp.asarray(jnp.inf),
        limb_topology=jnp.asarray(False), neighbors=neighbors,
        maximum_extrema=maximum_extrema,
    )
    return trace, support


def _parameters(source, rho, s, q):
    return tuple(jnp.asarray(value) for value in (source, rho, s, q))


def _lens_map_and_shear(images, s, q):
    """An independent centre-of-mass lens equation for physical assertions."""
    primary = -s * q / (1 + q)
    secondary = s / (1 + q)
    zp = np.conj(images) - primary
    zs = np.conj(images) - secondary
    mapped = images - 1 / (1 + q) / zp - q / (1 + q) / zs
    shear = 1 / (1 + q) / zp**2 + q / (1 + q) / zs**2
    return mapped, shear


@pytest.mark.fast
def test_unobserved_fold_component_has_its_true_projection_bounds():
    """The complete pair lies between two of the 56 source phases."""
    args = _parameters(0.07208739252567238 - 0.0038756551613259704j,
                       3e-5, 0.7916666666666666, 0.011787686347935866)
    trace, support = _geometry(*args)
    np.testing.assert_array_equal(np.asarray(trace.image_counts), 3)
    assert np.asarray(trace.image_limb).real[np.asarray(trace.physical_mask)].max() < 1
    cells = np.asarray(support[0])[np.asarray(support[1])]
    component = cells[cells[:, 0] > 1]
    assert support[0].shape == (21, 2)
    np.testing.assert_allclose(
        [component[:, 0].min(), component[:, 1].max()],
        [1.037548030664129, 1.037677676410301], rtol=0, atol=3e-10,
    )


@pytest.mark.fast
def test_four_critical_crossings_partition_the_interior_profile():
    """Critical crossings are needed even when they are not projection turns."""
    source = -0.470039364740036 - 0.005415503314978648j
    rho, s, q = (0.00021142727844446993, 0.7916666666666666,
                 1.1787686347935867e-5)
    args = _parameters(source, rho, s, q)
    trace, support = _geometry(*args)
    counts = np.asarray(trace.image_counts)
    assert np.count_nonzero(counts != np.roll(counts, 1)) == 4
    roots, valid, _ = _observed_critical_events(
        trace.image_limb, trace.physical_mask, *args,
        neighbors=(trace.previous_limb, trace.following_limb,
                   trace.previous_mask, trace.following_mask),
    )
    roots = np.asarray(roots)[np.asarray(valid)]
    assert np.unique(np.round(roots.real, 10)).size == 4
    mapped, shear = _lens_map_and_shear(roots, s, q)
    np.testing.assert_allclose(abs(mapped - source), rho, rtol=1e-7, atol=0)
    np.testing.assert_allclose(abs(shear), 1, rtol=0, atol=1e-6)
    transverse = 2 * np.real(np.conj(mapped - source) * 1j * (1 - shear)) / rho
    assert np.min(abs(transverse)) > 1e-5
    endpoints = np.asarray(support[0])[np.asarray(support[1])].ravel()
    assert np.max(np.min(abs(roots.real[:, None] - endpoints[None, :]), axis=1)) < 1e-10


@pytest.mark.fast
def test_physical_trace_evidence_overrides_an_empty_midpoint():
    """The exchanged lens has an unresolved turn inside a witnessed cell."""
    args = _parameters(1.1310941327957198 - 0.002738526166409757j,
                       3e-5, 0.5833333333333334, 1e6)
    trace, support = _geometry(*args)
    cells, active = map(np.asarray, support[:2])
    finite = np.isfinite(cells[:, 1])
    indices = np.flatnonzero(finite)
    mids = jnp.asarray(cells[finite, 0] + .5 * np.diff(cells[finite], axis=1)[:, 0])
    source, rho, s, q = args
    roots = np.asarray(jax.jit(jax.vmap(lambda x: binary_line_level_set_roots(
        x + 0j, 1j, source, rho, s=s, q=q
    )))(mids))
    midpoint_empty = np.sum((roots.imag == 0) & np.isfinite(roots.real), axis=1) < 2
    projected = np.asarray(trace.image_limb).real[np.asarray(trace.physical_mask)]
    witnessed = np.any((projected[None, :] > cells[finite, 0, None])
                       & (projected[None, :] < cells[finite, 1, None]), axis=1)
    protected = indices[midpoint_empty & witnessed]
    assert protected.size > 0
    assert active[protected].all()


@pytest.mark.fast
def test_insufficient_cell_capacity_reaches_the_structural_status():
    """A finite partial integral must retain the known support truncation."""
    args = _parameters(0.07208739252567238 - 0.0038756551613259704j,
                       3e-5, 0.7916666666666666, 0.011787686347935866)
    _, complete = _geometry(*args)
    _, limited = _geometry(*args, maximum_extrema=2)
    assert not bool(complete[2])
    assert bool(limited[2])
    assert limited[0].shape == (3, 2)
    source, rho, s, q = args
    result = _mag_limb_dark_cartesian_impl(
        source, rho, s=s, q=q, u1=.5, axis=1 + 0j,
        n_slice=9, n_profile=8, n_limb=56, support=limited, return_info=True,
    )
    public_result = _fixed_cartesian_result(result, stage=1)
    assert np.isfinite(float(public_result.magnification))
    assert int(public_result.status) & ANGULAR_MOMENT_TOPOLOGY
    assert int(public_result.stage) == 1
