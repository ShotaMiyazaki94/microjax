import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray import TripleMagConfig, mag_triple
from microjax.inverse_ray.config import DEFAULT_TRIPLE_CONFIG
from microjax.inverse_ray.extended_source import mag_limb_dark_boundary, mag_uniform_triple_boundary
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
        expected = mag_uniform_triple_boundary(fixed_radial_order=31, **common)
    else:
        expected = mag_limb_dark_boundary(nlenses=3, u1=u1, certify_topology=False, **common)
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
