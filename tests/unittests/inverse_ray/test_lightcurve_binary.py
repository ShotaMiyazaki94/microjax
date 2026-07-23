import jax
import numpy as np
import jax.numpy as jnp
import pytest
from tests.utils.gpu import has_cuda

from microjax.inverse_ray.lightcurve import (
    _tiled_vmap_active_scalar,
    _select_full_points,
    mag_binary,
)
from microjax.inverse_ray.config import BinaryMagConfig
from microjax.inverse_ray.extended_source import mag_limb_dark_boundary, mag_uniform_boundary
from microjax.inverse_ray_dense.lightcurve import mag_binary_dense
from microjax.multipole import mag_hexadecapole
from microjax.point_source import _images_point_source


def make_trajectory(u0, tE, t0, alpha, n=100, span=3.0):
    t = t0 + jnp.linspace(-span * tE, span * tE, n)
    tau = (t - t0) / tE
    y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    return t, w_points


def test_chunked_active_map_skips_fully_inactive_chunks():
    data = jnp.arange(8, dtype=jnp.float64).astype(jnp.complex128)

    evaluate = jax.jit(
        lambda values, n_active: _tiled_vmap_active_scalar(
            lambda value: value.real + 1.0,
            values,
            n_active,
            tile_size=4,
        )
    )
    result = evaluate(data, jnp.int32(3))

    # The partially active first chunk is deliberately evaluated in full.  The
    # wholly inactive second chunk must be skipped and returned as zeros.
    assert np.array_equal(
        np.asarray(result),
        np.asarray([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_full_point_compaction_pads_with_the_first_rejected_point():
    points = jnp.asarray([10.0 + 1.0j, 20.0 + 2.0j, 30.0 + 3.0j])
    accepted = jnp.asarray([True, False, True])

    compact, indices, n_active = _select_full_points(points, accepted)

    assert int(n_active) == 1
    assert np.array_equal(np.asarray(indices), np.asarray([1, 3, 3]))
    # Inactive SIMD lanes execute in a partially filled GPU chunk.  They must
    # duplicate the actual rejected configuration, not unrelated point zero.
    assert np.array_equal(np.asarray(compact), np.asarray([points[1], points[1], points[1]]))


def _uniform_single_pass(point, rho, s, q, use_local_chart):
    return mag_uniform_boundary(
        point,
        rho,
        s=s,
        q=q,
        Nlimb=500,
        margin_r=0.5,
        angular_atol=1e-5,
        relative_tolerance=1e-4,
        max_radial_subdivisions=1,
        fixed_radial_order=31,
        robust_roots=False,
        certify_topology=False,
        radial_strategy="fixed",
        radial_chunk_size=8,
        return_info=True,
        _planetary_local_chart=use_local_chart,
    ).magnification


@pytest.mark.gpu
def test_planetary_chart_gate_is_value_and_forward_continuous_at_threshold():
    if not has_cuda():
        pytest.skip("CUDA GPU not available")
    point = 0.07166622782296088 + 0.07166622782296087j
    q = jnp.asarray(1e-2)
    local = lambda mass_ratio: _uniform_single_pass(point, 3e-3, 1.0, mass_ratio, True)
    global_ = lambda mass_ratio: _uniform_single_pass(point, 3e-3, 1.0, mass_ratio, False)
    local_value, local_forward = jax.jvp(local, (q,), (jnp.ones_like(q),))
    global_value, global_forward = jax.jvp(global_, (q,), (jnp.ones_like(q),))

    assert np.isclose(float(local_value), float(global_value), rtol=1e-9, atol=1e-10)
    assert np.isclose(float(local_forward), float(global_forward), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_small_q_public_path_matches_local_chart_value_and_forward_q():
    if not has_cuda():
        pytest.skip("CUDA GPU not available")
    point = 0.9699918738276432 - 0.001572032916854858j
    rho, s = 1e-4, 1.6
    q = jnp.asarray(1e-4)
    config = BinaryMagConfig(n_limb=500)
    public = lambda mass_ratio: mag_binary(
        jnp.asarray([point]), rho, s=s, q=mass_ratio, config=config
    )[0]
    local = lambda mass_ratio: _uniform_single_pass(point, rho, s, mass_ratio, True)
    public_value, public_forward = jax.jvp(public, (q,), (jnp.ones_like(q),))
    local_value, local_forward = jax.jvp(local, (q,), (jnp.ones_like(q),))

    assert np.isclose(float(public_value), float(local_value), rtol=5e-7, atol=1e-8)
    assert np.isclose(float(public_forward), float(local_forward), rtol=1e-3, atol=1e-3)


def test_partial_source_chunk_has_finite_reverse_gradient():
    data = jnp.asarray([1.0, 4.0, 9.0], dtype=jnp.complex128)

    def objective(scale):
        values = _tiled_vmap_active_scalar(
            lambda value: jnp.sqrt(scale * value.real),
            data,
            jnp.int32(3),
            tile_size=2,
        )
        return jnp.sum(values)

    forward = jax.jacfwd(objective)(jnp.asarray(1.0))
    reverse = jax.grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=0.0, atol=1e-14)


def test_far_field_uses_multipole_matches_internal():
    # Choose far from caustics: large |w| and tiny q
    s, q, rho = 1.2, 1e-3, 1e-3
    a = 0.5 * s
    e1 = q / (1.0 + q)
    x_cm = a * (1.0 - q) / (1.0 + q)
    # Trajectory points far away
    _, w = make_trajectory(u0=5.0, tE=10.0, t0=0.0, alpha=jnp.deg2rad(30), n=16, span=1.0)
    w_shift = w - x_cm
    z, z_mask = _images_point_source(w_shift, nlenses=2, a=a, e1=e1)
    mu_multi, _ = mag_hexadecapole(z, z_mask, rho, nlenses=2, a=a, e1=e1, s=s, q=q)

    mags = mag_binary(
        w,
        rho,
        s=s,
        q=q,
        config=BinaryMagConfig(n_limb=500),
    )
    # All far-field points pass the multipole gate.
    assert np.allclose(np.array(mags), np.array(mu_multi), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "name,value",
    [
        ("th_resolution", 120),
        ("grid_fp32", True),
        ("full_method", "dense"),
    ],
)
def test_boundary_api_rejects_dense_backend_arguments(name, value):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mag_binary(
            jnp.asarray([4.0 + 0.2j]),
            5e-3,
            s=1.0,
            q=1e-2,
            **{name: value},
        )


def test_dense_api_rejects_boundary_backend_arguments():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mag_binary_dense(
            jnp.asarray([4.0 + 0.2j]),
            5e-3,
            s=1.0,
            q=1e-2,
            angular_atol=1e-5,
        )


def test_boundary_lightcurve_supports_linear_limb_darkening():
    s, q, rho, u1 = 1.0, 1e-2, 5e-3, 0.5
    point = 0.03 + 0.01j
    expected = mag_limb_dark_boundary(
        point,
        rho,
        s=s,
        q=q,
        u1=u1,
        Nlimb=500,
        angular_atol=1e-5,
    )
    result = mag_binary(
        jnp.asarray([point]),
        rho,
        s=s,
        q=q,
        u1=u1,
        config=BinaryMagConfig(n_limb=500),
    )

    assert np.isfinite(float(expected))
    assert np.allclose(np.asarray(result), np.asarray([expected]), atol=1e-10)


@pytest.mark.parametrize("u1", [0.0, 0.5])
def test_binary_grid_fp32_matches_baseline_on_small_case(u1):
    s, q, rho = 1.0, 1e-2, 5e-3
    _, w = make_trajectory(
        u0=0.03,
        tE=20.0,
        t0=0.0,
        alpha=jnp.deg2rad(15.0),
        n=6,
        span=0.15,
    )

    common = dict(
        s=s,
        q=q,
        u1=u1,
        r_resolution=60,
        th_resolution=60,
        Nlimb=60,
        bins_r=20,
        bins_th=40,
        margin_r=0.5,
        margin_th=0.5,
        MAX_FULL_CALLS=6,
        chunk_size=3,
    )

    mags_base = mag_binary_dense(w, rho, grid_fp32=False, **common)
    mags_mixed = mag_binary_dense(w, rho, grid_fp32=True, **common)

    assert np.all(np.isfinite(np.array(mags_mixed)))
    assert np.allclose(np.array(mags_mixed), np.array(mags_base), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "s,q,u0,tE,rho,alpha",
    [
        (1.0, 1e-2, 0.05, 20.0, 5e-3, np.deg2rad(15.0)),
    ],
)
@pytest.mark.gpu
def test_binary_lightcurve_matches_vbbl(s, q, u0, tE, rho, alpha):
    if not has_cuda():
        pytest.skip("CUDA GPU not available")
    VB = pytest.importorskip("VBBinaryLensing")
    VBBL = VB.VBBinaryLensing()
    VBBL.a1 = 0.0
    VBBL.RelTol = 1e-5

    t0 = 0.0
    npts = 100
    t, w = make_trajectory(u0=u0, tE=tE, t0=t0, alpha=alpha, n=npts, span=1.0)
    params_vb = [jnp.log(s), jnp.log(q), u0, alpha - jnp.pi, jnp.log(rho), jnp.log(tE), t0]
    mag_vb, _, _ = jnp.array(VBBL.BinaryLightCurve(params_vb, t))

    mags = mag_binary(
        w,
        rho,
        s=s,
        q=q,
        config=BinaryMagConfig(n_limb=500),
    )

    diff = np.array(mags) - np.array(mag_vb)
    # Allow a modest tolerance due to discretization and algorithmic differences
    assert np.max(np.abs(diff)) < 5e-3
