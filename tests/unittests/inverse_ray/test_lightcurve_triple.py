import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray import TripleMagConfig, mag_triple
from microjax.inverse_ray.config import DEFAULT_TRIPLE_CONFIG
from microjax.inverse_ray.extended_source import mag_limb_dark_boundary, mag_uniform_triple_boundary
from microjax.inverse_ray.geometry.limb import calc_source_limb
from microjax.inverse_ray.integrators.charts import _triple_compact_mixed_topology
from microjax.inverse_ray.roots.level_set import triple_level_set
from microjax.lens_geometry import triple_lens_geometry
from tests.utils.gpu import has_cuda


PARAMS = {"s": 0.9, "q": 0.3, "q3": 0.2, "r3": 0.4, "psi": 0.7}


def test_triple_config_has_the_one_pass_defaults():
    assert DEFAULT_TRIPLE_CONFIG == TripleMagConfig()
    assert DEFAULT_TRIPLE_CONFIG.n_limb == 500
    assert DEFAULT_TRIPLE_CONFIG.margin_r == 0.5


@pytest.mark.parametrize(
    "name,value",
    [
        ("r_resolution", 100),
        ("bins_th", 40),
        ("MAX_FULL_CALLS", 2),
        ("chunk_size", 2),
    ],
)
def test_triple_public_api_rejects_dense_arguments(name, value):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mag_triple(
            jnp.asarray([0.8 + 0.4j]),
            1e-2,
            **PARAMS,
            **{name: value},
        )


@pytest.mark.gpu
@pytest.mark.parametrize("u1", [0.0, 0.5])
def test_triple_public_path_is_finite_on_gpu(u1):
    if not has_cuda():
        pytest.skip("CUDA GPU not available")
    # The first point fails the multipole gate and exercises the boundary
    # kernel; the second remains on the fast multipole path.
    points = jnp.asarray([0.2 + 0.1j, 1.5 - 0.7j])
    result = mag_triple(
        points,
        1e-2,
        u1=u1,
        config=TripleMagConfig(n_limb=40, angular_atol=1e-3),
        **PARAMS,
    )
    assert result.shape == points.shape
    assert np.all(np.isfinite(np.asarray(result)))
    common = dict(
        w_center=points[0],
        rho=1e-2,
        Nlimb=40,
        angular_atol=1e-3,
        relative_tolerance=1e-4,
        max_radial_subdivisions=1,
        radial_strategy="fixed",
        radial_chunk_size=8,
        return_info=True,
        **PARAMS,
    )
    if u1 == 0.0:
        expected = mag_uniform_triple_boundary(fixed_radial_order=31, _compact_local_chart=True, **common)
    else:
        expected = mag_limb_dark_boundary(
            nlenses=3,
            u1=u1,
            certify_topology=False,
            _compact_local_chart=True,
            **common,
        )
    assert np.isclose(float(result[0]), float(expected.magnification), rtol=0.0, atol=1e-12)


@pytest.mark.gpu
def test_triple_public_path_has_finite_forward_q3_derivative_on_gpu():
    if not has_cuda():
        pytest.skip("CUDA GPU not available")
    point = jnp.asarray([0.2 + 0.1j])
    config = TripleMagConfig(n_limb=40, angular_atol=1e-3)
    fixed_params = {name: value for name, value in PARAMS.items() if name != "q3"}

    def evaluate(q3):
        return mag_triple(point, 1e-2, q3=q3, config=config, **fixed_params)[0]

    value, derivative = jax.jvp(evaluate, (jnp.asarray(PARAMS["q3"]),), (jnp.asarray(1.0),))
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


@pytest.mark.gpu
def test_triple_small_angle_images_use_local_charts_with_consistent_value_and_forward_ad():
    if not has_cuda():
        pytest.skip("CUDA GPU not available")
    params = {"s": 1.1, "q": 0.1, "q3": 0.03, "r3": abs(0.3 + 1.2j), "psi": np.angle(0.3 + 1.2j)}
    source = jnp.asarray(-0.39799824915516746 - 0.3187434605908351j)
    rho = 0.02
    geometry = triple_lens_geometry(**params)
    lens_params = {
        **params,
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
    }
    image_limb, mask_limb = calc_source_limb(source, rho, 80, nlenses=3, **lens_params)
    source_shifted = source - geometry.shifted
    origin_inside = triple_level_set(
        0.0 + 0.0j,
        source_shifted,
        rho,
        geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
    ) <= 0.0
    margin_parameters = (
        geometry.shifted,
        jnp.asarray([geometry.a, -geometry.a, geometry.r3_complex]),
        jnp.asarray([geometry.e1, geometry.e2, geometry.e3]),
    )
    charted = _triple_compact_mixed_topology(
        image_limb,
        mask_limb,
        rho,
        margin_r=0.5,
        w_center_shifted=source_shifted,
        origin_inside=origin_inside,
        shifted=geometry.shifted,
        a=geometry.a,
        e1=geometry.e1,
        e2=geometry.e2,
        r3_complex=geometry.r3_complex,
        lens_margin_parameters=margin_parameters,
    )
    assert int(jnp.sum(charted.chart_active)) == 2

    common = dict(
        w_center=source,
        rho=rho,
        s=params["s"],
        q=params["q"],
        r3=params["r3"],
        psi=params["psi"],
        Nlimb=80,
        angular_atol=1e-5,
        max_radial_subdivisions=1,
        radial_strategy="fixed",
        fixed_radial_order=31,
        radial_chunk_size=8,
    )

    def evaluate(q3, use_local):
        return mag_uniform_triple_boundary(q3=q3, _compact_local_chart=use_local, **common)

    q3 = jnp.asarray(params["q3"])
    global_value, global_forward = jax.jvp(lambda value: evaluate(value, False), (q3,), (jnp.ones_like(q3),))
    local_value, local_forward = jax.jvp(lambda value: evaluate(value, True), (q3,), (jnp.ones_like(q3),))
    assert np.isclose(float(local_value), float(global_value), rtol=1e-6, atol=1e-9)
    assert np.isfinite(float(local_forward))
    assert np.isclose(float(local_forward), float(global_forward), rtol=1e-5, atol=1e-6)
