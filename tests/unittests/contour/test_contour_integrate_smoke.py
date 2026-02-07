import numpy as _np
import pytest


def test_contour_integrate_smoke():
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    from jax import config
    import jax.numpy as jnp

    config.update("jax_enable_x64", True)

    try:
        from microjax.contour import IntegratorOptions, integrate

        trajectory = jnp.array(_np.array([0.1 + 0.05j]))
        result = integrate(
            source={"trajectory": trajectory, "rho": 0.05},
            lens={"s": 1.2, "q": 0.1},
            options=IntegratorOptions(
                tol=1e-2,
                retol=1e-3,
                default_strategy=(20, 20),
                analytic=True,
                return_info=False,
                limb_darkening_coeff=0.5,
                n_annuli=1,
            ),
        )

        assert result.mu.shape == trajectory.shape
        assert jnp.all(jnp.isfinite(result.mu))
    except RuntimeError as e:
        if "Unable to initialize backend" in str(e):
            pytest.skip("Skipping on non-CPU JAX backend environment")
        raise
