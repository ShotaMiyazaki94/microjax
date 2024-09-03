import jax
import jax.numpy as jnp
from jax import lax

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

@jax.jit
def merge_intervals_jax(arr, offset=1.0):
    intervals = jnp.stack([arr - offset, arr + offset], axis=1)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        current_interval = carry

        start_max = jnp.maximum(current_interval[0], next_interval[0])
        start_min = jnp.minimum(current_interval[0], next_interval[0])
        end_max = jnp.maximum(current_interval[1], next_interval[1])
        end_min = jnp.minimum(current_interval[1], next_interval[1])

        overlap_exists = start_max <= end_min

        # merge interval if overlap_exists is True
        updated_current_interval = jnp.where(
            overlap_exists,
            jnp.array([start_min, end_max]),
            next_interval
        )

        return updated_current_interval, updated_current_interval

    _, merged_intervals = jax.lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask




import time
bins = 10**jnp.arange(-7, 0.5, 0.5)
print(bins)

for b in bins:
    arr = jnp.arange(-1, 1, b)
    start = time.time()
    result, _ = merge_intervals_jax(arr, offset=1e-5)
    result.block_until_ready()
    end = time.time()
    print(len(arr),end - start,)