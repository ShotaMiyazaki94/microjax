"""Configuration is bound once; calls expose physical parameters only."""

import jax.numpy as jnp
import numpy as np
import pytest

import microjax as mj
from microjax._magnification import lightcurve


@pytest.mark.parametrize("lens", ["binary", "triple"])
def test_factory_binds_solver_and_settings(monkeypatch, lens):
    make_mag = getattr(mj, f"make_mag_{lens}")
    config = getattr(mj, f"{lens.title()}MagConfig")(source_tile_size=16)
    calls = []

    def accelerator(w, rho, **parameters):
        calls.append(parameters)
        return w.real + rho

    def cpu(w, rho, **parameters):
        return mj.CpuMagnificationResult(w.real + rho, jnp.zeros(w.shape, dtype=jnp.int32))

    monkeypatch.setattr(lightcurve, f"_mag_{lens}_single_pass_impl", accelerator)
    monkeypatch.setattr(lightcurve, f"_mag_{lens}_cpu_impl", cpu)
    mag_gpu = make_mag(backend="gpu", config=config)
    mag_cpu = make_mag(backend="cpu")
    mag_info = make_mag(backend="cpu", return_info=True)

    # After creation the functions keep their solvers, rather than dispatching
    # through a mutable module setting at each evaluation.
    monkeypatch.setattr(lightcurve, f"_mag_{lens}_single_pass_impl", None)
    monkeypatch.setattr(lightcurve, f"_mag_{lens}_cpu_impl", None)
    w = jnp.asarray([0.1 + 0.2j])
    parameters = dict(s=1.0, q=0.1, u1=0.5)
    if lens == "triple":
        parameters.update(q3=0.01, r3=1.2, psi=0.7)
    np.testing.assert_array_equal(mag_gpu(w, 0.01, **parameters), mag_cpu(w, 0.01, **parameters))
    np.testing.assert_array_equal(mag_info(w, 0.01, **parameters).magnification, w.real + 0.01)
    assert calls == [{**parameters, "config": config}]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mag_cpu(w, 0.01, **parameters, backend="gpu")


def test_configuration_errors_are_reported_at_creation():
    with pytest.raises(ValueError, match="backend must be"):
        mj.make_binary_lightcurve(backend="unknown")
    with pytest.raises(ValueError, match="return_info"):
        mj.make_mag_binary(return_info=True)
    with pytest.raises(ValueError, match="config applies only"):
        mj.make_mag_triple(backend="cpu", config=mj.TripleMagConfig())
