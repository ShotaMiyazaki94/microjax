try:
    import jax
except Exception:  # pragma: no cover
    jax = None


def has_cuda() -> bool:
    if jax is None:
        return False
    for backend in ("gpu", "cuda"):
        try:
            return len(jax.devices(backend)) > 0
        except Exception:
            pass
    return False
