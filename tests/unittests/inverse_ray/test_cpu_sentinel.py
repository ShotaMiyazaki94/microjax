import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.sentinel import (
    binary_caustic_reference_points,
    hidden_caustic_candidate,
)

def test_binary_reference_points_are_mapped_from_critical_curve():
    references = binary_caustic_reference_points(s=1.0, q=0.1)
    assert references.shape == (4,)
    assert np.all(np.isfinite(np.asarray(references)))


def test_hidden_caustic_requires_reference_inside_and_no_limb_transition():
    references = binary_caustic_reference_points(s=0.8, q=1e-3)
    source = references[0]
    assert bool(
        hidden_caustic_candidate(
            source,
            1e-6,
            s=0.8,
            q=1e-3,
            limb_transition=False,
        )
    )
    assert not bool(
        hidden_caustic_candidate(
            source,
            1e-6,
            s=0.8,
            q=1e-3,
            limb_transition=True,
        )
    )


@pytest.mark.fast
def test_hidden_caustic_sentinel_supports_jit():
    evaluate = jax.jit(
        lambda source: hidden_caustic_candidate(
            source,
            1e-3,
            s=1.2,
            q=0.01,
            limb_transition=False,
        )
    )
    value = evaluate(jnp.asarray(10.0 + 10.0j, dtype=jnp.complex128))
    assert not bool(value)
