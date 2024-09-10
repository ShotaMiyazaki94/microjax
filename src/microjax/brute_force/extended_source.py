import jax 
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves

@partial(jit, static_argnames=("nlenses", "NBIN", "Nlimb"))
def mag_inverse_ray(w_center, rho, NBIN=10, Nlimb=1000, margin=1.0, nlenses=2, **_params):
    q, s  = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    _params = {"q": q, "s": s, "a": a, "e1": e1}

    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
    w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
    image_limb, mask_image_limb = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
    image_limb = image_limb + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate

    # construct r-range!
    image_start = image_limb[mask_image_limb].ravel()
    r_is = jnp.sqrt(image_start.real**2 + image_start.imag**2)
    r_, r_mask = merge_intervals(r_is, offset= margin*rho)
    r_ = jnp.concatenate([jnp.arange(x1, x2, rho/NBIN) for x1, x2 in r_[r_mask]])
    # construct theta-range!
    th_is = jnp.arctan2(image_start.imag, image_start.real) 
    th_, th_mask = merge_intervals(th_is, offset=margin*rho)
    th_ = jnp.concatenate([jnp.arange(x1, x2, rho/NBIN) for x1, x2 in th_[th_mask]])
    # construct search grid!
    r_grid, th_grid = jnp.meshgrid(r_, th_) 
    x_grid = r_grid.ravel()*jnp.cos(th_grid.ravel())
    y_grid = r_grid.ravel()*jnp.sin(th_grid.ravel())
    image_mesh = x_grid.ravel() + 1j * y_grid.ravel()

    source_mesh = lens_eq(image_mesh - 0.5*s*(1 - q)/(1 + q), **_params) 
    source_mask = jnp.abs(source_mesh - w_center + 0.5*s*(1 - q)/(1 + q)) < rho
    source_mask_2 = jnp.abs(image_mesh - w_center) < rho
    image_area  = r_grid.ravel() * source_mask.astype(float)
    source_area = r_grid.ravel() * source_mask_2.astype(float)

    magnification = image_area / source_area

    return magnification

@jit
def merge_intervals(arr, margin=1.0):
    intervals = jnp.stack([arr - margin, arr + margin], axis=1)
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

    _, merged_intervals = lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask