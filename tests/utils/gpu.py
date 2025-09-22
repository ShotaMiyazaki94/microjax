import os

try:
    import jax
except Exception:  # pragma: no cover
    jax = None


def gpu_tests_enabled() -> bool:
    return os.environ.get("MICROJAX_GPU_TESTS", "0") in {"1", "true", "True"}


def has_cuda() -> bool:
    if jax is None:
        return False
    try:
        return len(jax.devices("cuda")) > 0
    except Exception:
        return False


## Deprecated: A100-specific checks have been removed. Any CUDA GPU is acceptable.
