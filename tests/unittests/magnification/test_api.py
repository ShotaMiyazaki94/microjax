"""Configuration is bound once; calls expose physical parameters only."""

import jax.numpy as jnp
import numpy as np
import pytest

import microjax as mj
from microjax._magnification.cpu import lightcurve as cpu_lightcurve
from microjax._magnification.gpu import lightcurve as gpu_lightcurve


@pytest.mark.parametrize("lens", ["binary", "triple"])
def test_factory_binds_solver_and_settings(monkeypatch, lens):
    make_mag = getattr(mj, f"make_mag_{lens}")
    config = getattr(mj, f"{lens.title()}MagConfig")(source_tile_size=16)
    calls = []

    def gpu(w, rho, **parameters):
        calls.append(parameters)
        return w.real + 2 * rho

    def cpu(w, rho, **parameters):
        return mj.CpuMagnificationResult(w.real + rho, jnp.zeros(w.shape, dtype=jnp.int32))

    monkeypatch.setattr(gpu_lightcurve, f"mag_{lens}", gpu)
    monkeypatch.setattr(cpu_lightcurve, f"mag_{lens}", cpu)
    mag_gpu = make_mag(backend="gpu", config=config)
    mag_cpu = make_mag(backend="cpu")
    mag_default = make_mag()
    mag_info = make_mag(return_info=True)
    make_curve = getattr(mj, f"make_{lens}_lightcurve")
    curve_default = make_curve()
    curve_gpu = make_curve(backend="gpu", config=config)

    # After creation the functions keep their solvers, rather than dispatching
    # through a mutable module setting at each evaluation.
    monkeypatch.setattr(gpu_lightcurve, f"mag_{lens}", None)
    monkeypatch.setattr(cpu_lightcurve, f"mag_{lens}", None)
    w = jnp.asarray([0.1 + 0.2j])
    parameters = dict(s=1.0, q=0.1, u1=0.5)
    if lens == "triple":
        parameters.update(q3=0.01, r3=1.2, psi=0.7)
    np.testing.assert_array_equal(mag_gpu(w, 0.01, **parameters), w.real + 0.02)
    np.testing.assert_array_equal(mag_default(w, 0.01, **parameters), mag_cpu(w, 0.01, **parameters))
    np.testing.assert_array_equal(mag_info(w, 0.01, **parameters).magnification, w.real + 0.01)
    assert calls == [{**parameters, "config": config}]

    event = [0.0, 1.0, 0.2, parameters["q"], parameters["s"], 0.0, 0.01]
    if lens == "triple":
        event += [parameters["q3"], parameters["r3"], parameters["psi"]]
    times = jnp.asarray([0.1])
    np.testing.assert_array_equal(curve_default(jnp.asarray(event), times), times + 0.01)
    np.testing.assert_array_equal(curve_gpu(jnp.asarray(event), times), times + 0.02)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mag_cpu(w, 0.01, **parameters, backend="gpu")


def test_configuration_errors_are_reported_at_creation():
    with pytest.raises(ValueError, match="backend must be"):
        mj.make_binary_lightcurve(backend="unknown")
    with pytest.raises(ValueError, match="return_info"):
        mj.make_mag_binary(backend="gpu", return_info=True)
    with pytest.raises(ValueError, match="config applies only"):
        mj.make_mag_triple(backend="cpu", config=mj.TripleMagConfig())
