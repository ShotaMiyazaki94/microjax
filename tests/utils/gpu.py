try:
    import jax
except Exception:  # pragma: no cover
    jax = None


def has_cuda() -> bool:
    if jax is None:
        return False
    try:
        return len(jax.devices("cuda")) > 0
    except Exception:
        return False
