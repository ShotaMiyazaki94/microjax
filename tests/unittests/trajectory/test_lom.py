import numpy as np
import jax.numpy as jnp

from microjax.trajectory.lom import (
    circular_orbital_motion_3d,
    circular_orbital_motion_3d_jit,
    circular_orbital_motion_3d_state,
    circular_orbital_motion_3d_state_jit,
    elliptic_orbital_motion_3d,
    elliptic_orbital_motion_3d_jit,
    elliptic_orbital_motion_3d_state,
    elliptic_orbital_motion_3d_state_jit,
    linear_orbital_motion,
    linear_orbital_motion_jit,
    linear_orbital_motion_state,
    linear_orbital_motion_state_jit,
    to_rotating_lens_frame,
)


def _vb_orbital_reference(t, s0, alpha0, w1, w2, w3, tref):
    """Numpy reference mirroring VBBinaryLensing::BinaryLightCurveOrbital."""
    t = np.asarray(t, dtype=np.float64)

    w13 = np.sqrt(w1 * w1 + w3 * w3)
    w123 = np.sqrt(w13 * w13 + w2 * w2)
    if w13 > 1e-8:
        w3_eff = w3 if w3 > 1e-8 else 1e-8
        w_orb = w3_eff * w123 / w13
        inc = np.arccos(np.clip((w2 * w3_eff) / (w13 * w123), -1.0, 1.0))
        phi0 = np.arctan2(-w1 * w123, w3_eff * w13)
    else:
        w_orb = w2
        inc = 0.0
        phi0 = 0.0

    c_phi0 = np.cos(phi0)
    s_phi0 = np.sin(phi0)
    c_inc = np.cos(inc)
    s_inc = np.sin(inc)
    den0 = np.sqrt(c_phi0 * c_phi0 + c_inc * c_inc * s_phi0 * s_phi0)
    s_true = s0 / den0

    c_Om = (c_phi0 * np.cos(alpha0) + c_inc * np.sin(alpha0) * s_phi0) / den0
    s_Om = (c_phi0 * np.sin(alpha0) - c_inc * np.cos(alpha0) * s_phi0) / den0

    phi = w_orb * (t - tref) + phi0
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    den = np.sqrt(c_phi * c_phi + c_inc * c_inc * s_phi * s_phi)

    s_t = s_true * den
    sin_alpha_t = (c_phi * s_Om + c_inc * s_phi * c_Om) / den
    cos_alpha_t = (c_phi * c_Om - c_inc * s_phi * s_Om) / den
    alpha_t = np.arctan2(sin_alpha_t, cos_alpha_t)
    sz_t = s_true * s_inc * s_phi
    return s_t, alpha_t, sz_t


def test_linear_orbital_motion_matches_formula():
    t = jnp.array([98.0, 100.0, 104.0], dtype=jnp.float64)
    s0 = 1.2
    alpha0 = 0.3
    ds_dt = -2.0e-3
    dalpha_dt = 5.0e-4
    tref = 100.0

    s_t, alpha_t = linear_orbital_motion(
        t, s0=s0, alpha0=alpha0, ds_dt=ds_dt, dalpha_dt=dalpha_dt, tref=tref
    )

    dt = np.array(t) - tref
    np.testing.assert_allclose(np.array(s_t), s0 + ds_dt * dt, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.array(alpha_t), alpha0 + dalpha_dt * dt, rtol=0.0, atol=1e-15
    )


def test_linear_orbital_motion_jit_consistency():
    t = jnp.linspace(0.0, 10.0, 17)
    args = dict(s0=0.95, alpha0=1.1, ds_dt=1.0e-3, dalpha_dt=-2.5e-3, tref=4.0)
    s_ref, alpha_ref = linear_orbital_motion(t, **args)
    s_jit, alpha_jit = linear_orbital_motion_jit(t, **args)

    np.testing.assert_allclose(np.array(s_jit), np.array(s_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.array(alpha_jit), np.array(alpha_ref), rtol=0.0, atol=1e-15
    )


def test_to_rotating_lens_frame_rotation():
    w = jnp.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=jnp.complex128)
    alpha = jnp.pi / 2.0
    w_lens = to_rotating_lens_frame(w, alpha)

    expected = np.array([0.0 - 1.0j, 1.0 + 0.0j], dtype=np.complex128)
    np.testing.assert_allclose(np.array(w_lens), expected, rtol=0.0, atol=1e-15)


def test_linear_orbital_motion_state_reduces_to_fixed_rotation():
    t = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
    w = jnp.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128)
    w_lens, s_t, alpha_t = linear_orbital_motion_state(
        t=t,
        w=w,
        s0=1.0,
        alpha0=jnp.pi / 4.0,
        ds_dt=0.0,
        dalpha_dt=0.0,
        tref=0.0,
    )

    np.testing.assert_allclose(np.array(s_t), np.ones(3), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.array(alpha_t), np.full(3, np.pi / 4.0), rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(w_lens),
        np.exp(-1j * np.pi / 4.0) * np.ones(3, dtype=np.complex128),
        rtol=0.0,
        atol=1e-15,
    )


def test_linear_orbital_motion_state_jit_consistency():
    t = jnp.linspace(-2.0, 3.0, 8)
    w = jnp.exp(1j * t)
    args = dict(s0=0.9, alpha0=0.2, ds_dt=1.0e-3, dalpha_dt=-2.0e-2, tref=0.5)

    w_ref, s_ref, a_ref = linear_orbital_motion_state(t=t, w=w, **args)
    w_jit, s_jit, a_jit = linear_orbital_motion_state_jit(t=t, w=w, **args)

    np.testing.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(s_jit), np.array(s_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(a_jit), np.array(a_ref), rtol=0.0, atol=1e-15)


def test_circular_orbital_motion_3d_matches_vb_reference_formula():
    t = jnp.linspace(2450000.0 - 20.0, 2450000.0 + 20.0, 17, dtype=jnp.float64)
    args = dict(
        s0=1.15,
        alpha0=0.4,
        w1=1.0e-2,
        w2=-2.5e-2,
        w3=3.0e-2,
        tref=2450000.0,
    )

    s_t, alpha_t, sz_t = circular_orbital_motion_3d(t, **args)
    s_ref, alpha_ref, sz_ref = _vb_orbital_reference(np.array(t), **args)

    np.testing.assert_allclose(np.array(s_t), s_ref, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.sin(np.array(alpha_t)), np.sin(alpha_ref), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.cos(np.array(alpha_t)), np.cos(alpha_ref), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.array(sz_t), sz_ref, rtol=0.0, atol=1e-12)


def test_circular_orbital_motion_3d_matches_vb_reference_negative_w3_guard():
    t = jnp.linspace(-5.0, 5.0, 9, dtype=jnp.float64)
    args = dict(s0=1.1, alpha0=-0.7, w1=-3.0e-2, w2=1.0e-2, w3=-2.0e-2, tref=0.0)

    s_t, alpha_t, sz_t = circular_orbital_motion_3d(t, **args)
    s_ref, alpha_ref, sz_ref = _vb_orbital_reference(np.array(t), **args)

    np.testing.assert_allclose(np.array(s_t), s_ref, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.sin(np.array(alpha_t)), np.sin(alpha_ref), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.cos(np.array(alpha_t)), np.cos(alpha_ref), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.array(sz_t), sz_ref, rtol=2e-10, atol=1e-6)


def test_circular_orbital_motion_3d_reduces_to_linear_angle_only():
    t = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float64)
    w = jnp.exp(1j * t)
    s0 = 0.95
    alpha0 = 0.2
    w2 = -3.0e-2

    w_lens, s_t, alpha_t, sz_t = circular_orbital_motion_3d_state(
        t=t, w=w, s0=s0, alpha0=alpha0, w1=0.0, w2=w2, w3=0.0, tref=0.0
    )

    np.testing.assert_allclose(np.array(s_t), np.full(t.shape, s0), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(alpha_t), alpha0 + w2 * np.array(t), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(sz_t), np.zeros(t.shape), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.array(w_lens),
        np.array(to_rotating_lens_frame(w, alpha_t)),
        rtol=0.0,
        atol=1e-15,
    )


def test_circular_orbital_motion_3d_jit_consistency():
    t = jnp.linspace(0.0, 10.0, 21, dtype=jnp.float64)
    args = dict(s0=1.3, alpha0=-0.3, w1=2.0e-2, w2=1.0e-2, w3=4.0e-2, tref=4.0)

    s_ref, a_ref, sz_ref = circular_orbital_motion_3d(t, **args)
    s_jit, a_jit, sz_jit = circular_orbital_motion_3d_jit(t, **args)

    np.testing.assert_allclose(np.array(s_jit), np.array(s_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(a_jit), np.array(a_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(sz_jit), np.array(sz_ref), rtol=0.0, atol=1e-15)


def test_circular_orbital_motion_3d_state_jit_consistency():
    t = jnp.linspace(-3.0, 3.0, 13, dtype=jnp.float64)
    w = (0.1 * t) + 1j * (0.2 - 0.05 * t)
    args = dict(s0=1.0, alpha0=0.1, w1=1.0e-2, w2=-2.0e-2, w3=2.5e-2, tref=0.7)

    w_ref, s_ref, a_ref, sz_ref = circular_orbital_motion_3d_state(t=t, w=w, **args)
    w_jit, s_jit, a_jit, sz_jit = circular_orbital_motion_3d_state_jit(t=t, w=w, **args)

    np.testing.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(s_jit), np.array(s_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(a_jit), np.array(a_ref), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.array(sz_jit), np.array(sz_ref), rtol=0.0, atol=1e-15)


def test_elliptic_orbital_motion_3d_jit_consistency():
    t = jnp.linspace(-10.0, 10.0, 31, dtype=jnp.float64)
    args = dict(
        s0=1.15,
        alpha0=0.2,
        w1=1.0e-2,
        w2=-1.3e-2,
        w3=8.0e-3,
        szs=0.25,
        ar=0.9,
        tref=5000.0,
        kepler_newton_iter=12,
    )

    s_ref, a_ref, sz_ref = elliptic_orbital_motion_3d(t, **args)
    s_jit, a_jit, sz_jit = elliptic_orbital_motion_3d_jit(t, **args)

    np.testing.assert_allclose(np.array(s_jit), np.array(s_ref), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(a_jit), np.array(a_ref), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(sz_jit), np.array(sz_ref), rtol=0.0, atol=1e-14)


def test_elliptic_orbital_motion_3d_state_rotation_consistency():
    t = jnp.linspace(-4.0, 4.0, 17, dtype=jnp.float64)
    w = (0.03 * t + 0.1) + 1j * (0.15 - 0.02 * t)
    args = dict(
        s0=0.95,
        alpha0=-0.5,
        w1=7.0e-3,
        w2=1.2e-2,
        w3=-4.0e-3,
        szs=-0.1,
        ar=0.8,
        tref=100.0,
        kepler_newton_iter=10,
    )

    w_ref, s_ref, a_ref, sz_ref = elliptic_orbital_motion_3d_state(t=t, w=w, **args)
    w_jit, s_jit, a_jit, sz_jit = elliptic_orbital_motion_3d_state_jit(t=t, w=w, **args)

    np.testing.assert_allclose(np.array(w_ref), np.array(to_rotating_lens_frame(w, a_ref)), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(s_jit), np.array(s_ref), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(a_jit), np.array(a_ref), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.array(sz_jit), np.array(sz_ref), rtol=0.0, atol=1e-14)
