import os

# Force JAX to use CPU to avoid GPU plugin initialization in CI
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    from jax import config
    config.update("jax_enable_x64", True)
except Exception:
    pass

