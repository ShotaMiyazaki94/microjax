import jax 
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves

@partial(jit, static_argnames=("nlenses", "NBIN", "Nlimb"))
def mag_inverse_ray(w_center, rho, NBIN=10, Nlimb=1000, margin=1.0, nlenses=2, **_params):

    q, s = _params["q"], _params["s"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
    w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q)  # half-axis coordinate
    image, mask  = _images_point_source(w_limb_shift, a=a, e1=e1)  # half-axis coordinate
    image_limb   = image + 0.5*s*(1 - q)/(1 + q)  # center-of-mass coordinate 
    image_limb = image_limb.ravel()
    mask = mask.ravel()

    
    dr  = rho / NBIN 
    image_start = jnp.where(mask, image_limb, 0).ravel()
    r_is = jnp.sqrt(image_start.real**2 + image_start.imag**2)
    r_, r_mask = merge_intervals(r_is, offset=margin*rho)

    r_ = jnp.concatenate([jnp.arange(x1, x2, dr) for x1, x2 in r_[r_mask]])

    return 0.0

@jax.jit
def merge_intervals(arr, offset=1.0):
    intervals = jnp.stack([jnp.maximum(arr - offset, 0), arr + offset], axis=1)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        current_interval = carry
        start_max = jnp.maximum(current_interval[0], next_interval[0])
        start_min = jnp.minimum(current_interval[0], next_interval[0])
        end_max = jnp.maximum(current_interval[1], next_interval[1])
        end_min = jnp.minimum(current_interval[1], next_interval[1])
        overlap_exists = start_max <= end_min

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

@jax.jit
def merge_intervals_circ(arr, offset=1.0):
    arr_start  = jnp.clip(arr - offset, 0, 2*jnp.pi)
    arr_end    = jnp.clip(arr + offset, 0, 2*jnp.pi)
    intervals = jnp.stack([arr_start, arr_end], axis=1)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        current_interval = carry
        start_max = jnp.maximum(current_interval[0], next_interval[0])
        start_min = jnp.minimum(current_interval[0], next_interval[0])
        end_max = jnp.maximum(current_interval[1], next_interval[1])
        end_min = jnp.minimum(current_interval[1], next_interval[1])
        overlap_exists = start_max <= end_min

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