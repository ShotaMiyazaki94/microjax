import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
from microjax.inverse_ray.merge_area import calc_source_limb, calculate_overlap_and_range 
import jax.numpy as jnp
from jax import jit, vmap

import jax.numpy as jnp
from jax import jit, vmap

def _compute_in_mask(r_limb, th_limb, r_use, th_use):
    M = r_use.shape[0]  
    K = th_use.shape[0]  
    N = r_limb.shape[0]
    r_limb_expanded = r_limb.reshape(1, 1, N)
    th_limb_expanded = th_limb.reshape(1, 1, N)
    r_use_min = r_use[:, 0].reshape(M, 1, 1)
    r_use_max = r_use[:, 1].reshape(M, 1, 1)
    th_use_min = th_use[:, 0].reshape(1, K, 1)
    th_use_max = th_use[:, 1].reshape(1, K, 1)

    r_condition = (r_limb_expanded > r_use_min) & (r_limb_expanded < r_use_max)  # shape: (M, 1, N)
    th_condition = (th_limb_expanded > th_use_min) & (th_limb_expanded < th_use_max)  # shape: (1, K, N)
    combined_condition = r_condition & th_condition  # shape: (M, K, N)

    # condition for all the combination
    in_mask = jnp.any(combined_condition, axis=2)  # shape: (M, K)
    return in_mask

@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def mag_simple2(w_center, rho, r_resolution=200, th_resolution=200, Nlimb=1000, 
                offset_r = 2.0, offset_th = 5.0, 
                fac_r = 1.0, fac_th = 1.0, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    # if merging is correct, 5 is good for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    in_mask = _compute_in_mask(r_limb.ravel(), th_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    # binary-lens should have less than 5 images.
    r_vmap   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:5]
    th_vmap  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:5]

    r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

    def compute_for_range(r_range, th_range):
        in_mask = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) &
                          (th_limb > th_range[0]) & (th_limb < th_range[1]))
        def compute_if_in():
            dr = fac_r * (r_range[1] - r_range[0]) / r_resolution
            dth = fac_th * (th_range[1] - th_range[0]) / th_resolution
            r_values  = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
            th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]
            def process_r(r0):
                z_th = r0 * (jnp.cos(th_values) + 1j * jnp.sin(th_values))
                image_mesh = lens_eq(z_th - shifted, **_params)
                distances = jnp.abs(image_mesh - w_center_shifted)
                in_source = (distances - rho < 0.0).astype(float)
                in0, in1 = in_source[:-1], in_source[1:]
                th0, th1 = th_values[:-1], th_values[1:]
                d0, d1   = distances[:-1], distances[1:]
                segment_inside = (in0 == 1) & (in1 == 1)
                segment_in2out = (in0 == 1) & (in1 == 0)
                segment_out2in = (in0 == 0) & (in1 == 1)
                frac = jnp.clip((rho - d0) / (d1 - d0), 0.0, 1.0)
                area_inside    = r0 * dth * segment_inside
                area_crossing  = r0 * dth * (segment_in2out * frac + segment_out2in * (1.0 - frac))
                return jnp.sum(area_inside + area_crossing)
            total_area = dr * jnp.sum(vmap(process_r)(r_values))
            return total_area
        return jnp.where(in_mask, compute_if_in(), 0.0)
    compute_vmap = vmap(compute_for_range, in_axes=(0, 0))
    image_areas = compute_vmap(r_vmap, th_vmap)
    magnification = jnp.sum(image_areas) / rho**2 / jnp.pi 
    return magnification

@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7))
def mag_simple_lax(w_center, rho, resolution=200, Nlimb=100, offset_r=1.0, offset_th=5.0, GRID_RATIO=1, num_fine=10, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)

    th_resolution = resolution * GRID_RATIO
    r_grid_norm = jnp.linspace(0, 1, resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)
    def compute_for_range(r_range, th_range):
        in_mask = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) &
                      (th_limb > th_range[0]) & (th_limb < th_range[1]))
        dr = (r_range[1] - r_range[0]) / resolution
        dth = (th_range[1] - th_range[0]) / (resolution * GRID_RATIO)
        r_values  = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
        th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]

        cos_th = jnp.cos(th_values)
        sin_th = jnp.sin(th_values)

        def process_r(r0):
            z_th = r0 * (cos_th + 1j * sin_th)
            image_mesh = lens_eq(z_th - shifted, **_params)
            distances = jnp.abs(image_mesh - w_center_shifted)
            in_source = (distances - rho < 0.0).astype(float)
            in0, in1 = in_source[:-1], in_source[1:]
            d0, d1   = distances[:-1], distances[1:]
            segment_inside = (in0 == 1) & (in1 == 1)
            segment_in2out = (in0 == 1) & (in1 == 0)
            segment_out2in = (in0 == 0) & (in1 == 1)
            frac = jnp.clip((rho - d0) / (d1 - d0), 0.0, 1.0)
            area_inside    = r0 * dth * segment_inside
            area_crossing  = r0 * dth * (segment_in2out * frac + segment_out2in * (1.0 - frac))
            return jnp.sum(area_inside + area_crossing)

        total_area = dr * jnp.sum(vmap(process_r)(r_values))
        return lax.select(in_mask, total_area, 0.0)
    compute_vmap = vmap(vmap(compute_for_range, in_axes=(None, 0)), in_axes=(0, None))
    image_areas = compute_vmap(r_use, th_use)
    magnification = jnp.sum(image_areas) / (rho**2 * jnp.pi)
    return magnification


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def mag_simple(w_center, rho, r_resolution=200, th_resolution=200, Nlimb=200, offset_r = 2.0, offset_th = 5.0, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    # if merging is correct, 5 is good for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)

    r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

    def compute_for_range(r_range, th_range):
        in_mask = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) &
                          (th_limb > th_range[0]) & (th_limb < th_range[1]))
        def compute_if_in():
            dr = (r_range[1] - r_range[0]) / r_resolution
            dth = (th_range[1] - th_range[0]) / th_resolution
            r_values  = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
            th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]
            def process_r(r0):
                z_th = r0 * (jnp.cos(th_values) + 1j * jnp.sin(th_values))
                image_mesh = lens_eq(z_th - shifted, **_params)
                distances = jnp.abs(image_mesh - w_center_shifted)
                in_source = (distances - rho < 0.0).astype(float)
                in0, in1 = in_source[:-1], in_source[1:]
                th0, th1 = th_values[:-1], th_values[1:]
                d0, d1   = distances[:-1], distances[1:]
                segment_inside = (in0 == 1) & (in1 == 1)
                segment_in2out = (in0 == 1) & (in1 == 0)
                segment_out2in = (in0 == 0) & (in1 == 1)
                frac = jnp.clip((rho - d0) / (d1 - d0), 0.0, 1.0)
                area_inside    = r0 * dth * segment_inside
                area_crossing  = r0 * dth * (segment_in2out * frac + segment_out2in * (1.0 - frac))
                return jnp.sum(area_inside + area_crossing)
            total_area = dr * jnp.sum(vmap(process_r)(r_values))
            return total_area
        return jnp.where(in_mask, compute_if_in(), 0.0)
    compute_vmap = vmap(vmap(compute_for_range, in_axes=(None, 0)), in_axes=(0, None))
    image_areas = compute_vmap(r_use, th_use)
    magnification = jnp.sum(image_areas) / rho**2 / jnp.pi 
    return magnification

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    q = 0.1
    s = 1.0
    alpha = jnp.deg2rad(30) # angle between lens axis and source trajectory
    tE = 30 # einstein radius crossing time
    t0 = 0.0 # time of peak magnification
    u0 = 0.1 # impact parameter
    rho = 5e-4

    num_points = 1000
    t  =  jnp.linspace(-5, 7.5, num_points)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    test_params = {"q": q, "s": s}  # Lens parameters

    from microjax.caustics.extended_source import mag_extended_source
    import MulensModel as mm
    def mag_vbbl(w0, rho, u1=0., accuracy=1e-05):
        a  = 0.5 * s
        e1 = 1.0 / (1.0 + q)
        e2 = 1.0 - e1  
        bl = mm.BinaryLens(e1, e2, 2*a)
        return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)
    #magn  = lambda w: mag_binary(w, rho, resolution=100, GRID_RATIO=1, **test_params)
    #magn  = lambda w: mag_extended_source(w, rho, **test_params, npts_limb = 100)
    magn  = lambda w: mag_simple2(w, rho, r_resolution=200, th_resolution=1000, **test_params, Nlimb=1000)
    #magn  = lambda w: mag_simple(w, rho, r_resolution=100, th_resolution=100, **test_params, Nlimb=100)
    magn2  = lambda w0: jnp.array([mag_vbbl(w, rho) for w in w0])
    #magn2 =  jit(vmap(magn2, in_axes=(0,)))
    magn =  jit(vmap(magn, in_axes=(0,)))

    start = time.time()
    magnifications = magn(w_points)
    magnifications.block_until_ready() 
    end = time.time()
    print("computation time: %.3f sec per points"%((end - start)/num_points))
    magnifications2 = magn2(w_points) 
    # Print out the result
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from microjax.point_source import critical_and_caustic_curves, mag_point_source
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import seaborn as sns
    #sns.set_theme(font="Arial", style="ticks")

    mags_poi = mag_point_source(w_points, s=s, q=q)
    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

    fig, ax_ = plt.subplots(2,1,figsize=(8,6), sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[4,1]))
    ax  = ax_[0]
    ax1 = ax_[1]
    ax_in = inset_axes(ax,
        width="60%", height="60%", 
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0.35, 0.35, 0.6, 0.6)
    )
    ax_in.set_aspect(1)
    ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")
    for cc in caustic_curves:
        ax_in.plot(cc.real, cc.imag, color='black', lw=0.7)
    circles = [plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, ec="k", zorder=1) 
               for xi, yi in zip(w_points.real, w_points.imag)
               ]
    c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.5)
    ax_in.add_collection(c)
    ax_in.set_aspect(1)
    ax_in.set(xlim=(-1., 1.2), ylim=(-0.8, 1.))

    ax.plot(t, magnifications, ".")
    ax.plot(t, magnifications2)
    #ax.plot(t, mags_poi, ls="--")
    ax.grid(ls=":")
    ax.set_ylabel("magnification")
    ax1.plot(t, jnp.abs(magnifications - magnifications2)/magnifications2)
    ax1.grid(ls=":")
    ax1.set_ylabel("relative diff")
    ax1.set_yscale("log")
    #ax1.set_ylim(1e-4, 1)
    plt.show()