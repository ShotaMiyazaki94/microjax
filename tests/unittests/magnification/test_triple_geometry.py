import jax.numpy as jnp
import numpy as np

from microjax._magnification.gpu.lightcurve import _triple_prefilter
from microjax._magnification.gpu.geometry.limb import calc_source_limb
from microjax.lens_geometry import triple_lens_geometry
from microjax.multipole import mag_hexadecapole
from microjax.point_source import _images_point_source, lens_eq, mag_point_source


def test_triple_source_limb_round_trips_between_coordinate_frames():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    center = jnp.asarray(-0.1 + 0.2j)
    rho = 1e-3
    n_limb = 32
    images_binary_com, mask = calc_source_limb(
        center, rho, n_limb, nlenses=3, **params
    )
    images_midpoint = images_binary_com - geometry.shifted
    mapped_midpoint = lens_eq(
        images_midpoint,
        nlenses=3,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3=params["r3"],
        psi=params["psi"],
    )
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_limb)
    source_midpoint = (
        center + rho * jnp.exp(1j * angles) - geometry.shifted
    )
    residual = jnp.abs(mapped_midpoint - source_midpoint[None, :])
    assert np.any(np.asarray(mask))
    assert np.all(np.asarray(residual)[np.asarray(mask)] < 1e-6)


def test_triple_limb_polish_preserves_physical_images_near_third_lens():
    params = {
        "s": 1.1,
        "q": 0.1,
        "q3": 0.01,
        "r3": np.abs(0.3 + 1.2j),
        "psi": np.angle(0.3 + 1.2j),
    }
    geometry = triple_lens_geometry(**params)
    alpha = np.deg2rad(50.0)
    time = 8.5
    center = (-0.1 * np.sin(alpha) + time / 10.0 * np.cos(alpha)) + 1j * (
        0.1 * np.cos(alpha) + time / 10.0 * np.sin(alpha)
    )
    n_limb = 80
    images, mask = calc_source_limb(center, 0.01, n_limb, nlenses=3, **params)
    source = center + 0.01 * jnp.exp(1j * jnp.linspace(0.0, 2.0 * jnp.pi, n_limb))
    raw_images, raw_mask = _images_point_source(
        source - geometry.shifted,
        nlenses=3,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        **params,
    )
    residual = jnp.abs(
        lens_eq(
            images - geometry.shifted,
            nlenses=3,
            a=geometry.a,
            e1=geometry.e1,
            e2=geometry.e2,
            r3=params["r3"],
            psi=params["psi"],
        )
        - (source - geometry.shifted)[None, :]
    )

    assert raw_images.shape == images.shape
    assert np.array_equal(np.asarray(mask), np.asarray(raw_mask))
    assert np.all(np.asarray(mask).sum(axis=0) >= 4)
    assert np.all(np.asarray(residual)[np.asarray(mask)] < 1e-6)


def test_triple_lightcurve_multipole_baseline_uses_binary_center_of_mass():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    points = jnp.asarray([-0.1 + 0.2j, 0.3 - 0.1j])
    rho = 1e-3
    lens_params = {
        **params,
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
    }
    images, mask = _images_point_source(
        points - geometry.shifted, nlenses=3, **lens_params
    )
    expected, _ = mag_hexadecapole(
        images, mask, rho, nlenses=3, u1=0.0, **lens_params
    )
    actual, _ = _triple_prefilter(points, rho, u1=0.0, **params)
    assert np.allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12
    )


def test_triple_multipole_point_source_limit_keeps_third_lens_azimuth():
    params = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}
    geometry = triple_lens_geometry(**params)
    points = jnp.asarray([0.8 + 0.4j, 1.5 - 0.7j, -1.2 + 0.9j])
    lens_params = {
        **params,
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
    }
    images, mask = _images_point_source(
        points - geometry.shifted, nlenses=3, **lens_params
    )
    multipole, _ = mag_hexadecapole(
        images,
        mask,
        1e-10,
        nlenses=3,
        **lens_params,
    )
    point_source = jnp.asarray(
        [mag_point_source(point, nlenses=3, **params) for point in points]
    )
    assert np.allclose(
        np.asarray(multipole),
        np.asarray(point_source),
        rtol=1e-11,
        atol=1e-12,
    )
