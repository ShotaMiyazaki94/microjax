import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
from microjax.inverse_ray.merge_area import calc_source_limb, calculate_overlap_and_range, _compute_in_mask, merge_final
import jax.numpy as jnp
from jax import jit, vmap

import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnames=("u1"))
def Is_limb_1st(d, u1=0.0):
    """
    Calculate the normalized limb-darkened intensity using a linear limb-darkening law.

    Parameters
    ----------
    r : array-like or float
        Radial distance from the center of the star's disk, 
        normalized such that r = 1 corresponds to the edge of the stellar disk.
        Should be in the range [0, 1].
    u1 : float, optional
        Linear limb-darkening coefficient. Defaults to 0.0, which corresponds 
        to a uniform disk with no limb darkening.

    Returns
    -------
    I : array-like or float
        Normalized intensity at the given radial distance(s), calculated as:
        I(r) = (3 / (π * (3 - u1))) * (1 - u1 * (1 - sqrt(1 - r))).

    Notes
    -----
    - The returned intensity is normalized such that the integral over the stellar disk is 1.
    - The equation implements the linear limb-darkening law:
      I(r) = I0 * (1 - u1 * (1 - sqrt(1 - r))),
      where I0 is a normalization constant ensuring that the total flux is conserved.
    - For physically meaningful results, `u1` should be in the range [0, 1], though 
      values outside this range can be used for testing or hypothetical scenarios.
    """
    mu = jnp.sqrt(1.0 - d**2)
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    I  = I0 * (1.0 - u1 * (1.0 - mu))
    return jnp.where(d < 1.0, I, 0.0) 

@partial(jit, static_argnames=("r_resolution", "th_resolution", "Nlimb", "u1",
                               "offset_r", "offset_th", "delta_c"))
def mag_binary(w_center, rho, r_resolution=4000, th_resolution=4000, Nlimb=200, u1=0.0, 
                offset_r = 1.0, offset_th = 10.0, delta_c=0.05, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    # one-dimensional overlap search and merging.
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    # if merging is correct, 5 may be emperically sufficient for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:,1])][-10:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-10:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    # select matched regions including image limbs. binary-lens microlensing should have less than 5 images.
    # note: theta boundary is 0 and 2pi so that images containing the boundary are divided into two.
    # The reason why the 6 is chosen is that never all five images align on the binary axis, though three may be.
    in_mask = _compute_in_mask(r_limb.ravel()*mask_limb.ravel(), th_limb.ravel()*mask_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    r_vmap_excess   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:10]
    th_vmap_excess  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:10]
    r_vmap, th_vmap = merge_final(r_vmap_excess, th_vmap_excess)
    r_vmap          = r_vmap[:5]
    th_vmap         = th_vmap[:5]

    r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

    def compute_for_range(r_range, th_range):
        dr = (r_range[1] - r_range[0]) / r_resolution
        dth = (th_range[1] - th_range[0]) / th_resolution
        r_values  = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
        th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]
        def process_r(r0):
            z_th = r0 * (jnp.cos(th_values) + 1j * jnp.sin(th_values))
            image_mesh = lens_eq(z_th - shifted, **_params)
            distances = jnp.abs(image_mesh - w_center_shifted)
            Is        = Is_limb_1st(distances / rho, u1=u1)
            in_source = Is > 0.0
            in0, in1, in2  = in_source[:-2], in_source[1:-1], in_source[2:]
            d0, d1, d2     = distances[:-2], distances[1:-1], distances[2:]
            in_segment = in0 & in1 & in2
            B1_segment = (~in0) & in1 & in2
            B2_segment = in0 & in1 & (~in2)
            zero_term  = 1e-12 
            delta_B1   = jnp.clip((rho - d0) / (d1 - d0 + zero_term), 0.0, 1.0) 
            delta_B2   = jnp.clip((d2 - rho) / (d2 - d1 + zero_term), 0.0, 1.0)
            fac_B1 = jnp.where(delta_B1 > delta_c, 
                               (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B1) * (0.5 + delta_B1), 
                               (2.0 / 3.0) * delta_B1 + 0.5)
            fac_B2 = jnp.where(delta_B2 > delta_c, 
                               (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B2) * (0.5 + delta_B2), 
                               (2.0 / 3.0) * delta_B2 + 0.5)
            area_inside = r0 * dth * Is[1:-1] * in_segment
            area_B1     = r0 * dth * Is[1:-1] * fac_B1 * B1_segment
            area_B2     = r0 * dth * Is[1:-1] * fac_B2 * B2_segment
            return jnp.sum(area_inside + area_B1 + area_B2)
        area_r = vmap(process_r)(r_values) # (Nr) array
        total_area = dr * jnp.sum(area_r)
        return total_area
    compute_vmap = vmap(compute_for_range, in_axes=(0, 0))
    image_areas = compute_vmap(r_vmap, th_vmap)
    magnification = jnp.sum(image_areas) / rho**2 #/ jnp.pi 
    return magnification 

@partial(jit, static_argnames=("r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th", "cubic"))
def mag_uniform(w_center, rho, r_resolution=4000, th_resolution=4000, Nlimb=500, 
                offset_r=0.5, offset_th=5.0, cubic=True, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    
    # one-dimensional overlap search and merging.
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    
    # if merging is correct, 5 may be emperically sufficient for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:,1])][-10:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-10:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    
    # select matched regions including image limbs. binary-lens microlensing should have less than 5 images.
    # note: theta boundary is 0 and 2pi so that images containing the boundary are divided into two.
    # The reason why the 6 is chosen is that never all five images align on the binary axis, though three may be.
    in_mask = _compute_in_mask(r_limb.ravel()*mask_limb.ravel(), th_limb.ravel()*mask_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    # select the first 5 regions for the integration.
    r_vmap_excess   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:10]
    th_vmap_excess  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:10]
    r_vmap, th_vmap = merge_final(r_vmap_excess, th_vmap_excess)
    r_vmap          = r_vmap[:5]
    th_vmap         = th_vmap[:5]

    r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

    def compute_for_range(r_range, th_range):
        dr = (r_range[1] - r_range[0]) / r_resolution
        dth = (th_range[1] - th_range[0]) / th_resolution
        r_values  = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
        th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]
        def process_r(r0):
            x_th = r0 * jnp.cos(th_values)
            y_th = r0 * jnp.sin(th_values)
            z_th = x_th + 1j * y_th 
            image_mesh = lens_eq(z_th - shifted, **_params)
            distances = jnp.abs(image_mesh - w_center_shifted)
            in_source = distances - rho < 0.0
            zero_term = 1e-10
            
            if cubic:
                # cubic interpolation. The boundaries locate  at between in1 and in2.
                def cubic_interp(x, xs, ys, eps=1e-8):
                    x_min = jnp.min(xs, axis=1, keepdims=True)  # shape=(N, 1)
                    x_max = jnp.max(xs, axis=1, keepdims=True)  # shape=(N, 1)
                    scale = jnp.maximum(x_max - x_min, eps)     # shape=(N, 1)
                    xs_hat = (xs - x_min) / scale               # shape=(N, 4) 
                    x_hat  = (x - x_min) / scale                 # shape=(N, 1)
                    diffs_x = x_hat - xs_hat  # shape=(N, 4)
                    diag_mask = jnp.eye(4, dtype=bool) # shape=(4, 4)
                    diffs_x_mat = jnp.where(diag_mask, 1.0, diffs_x[:, None, :]) # shape=(N, 4, 4)
                    numer = jnp.prod(diffs_x_mat, axis=2)  # shape=(N, 4)
                    diffs_xs = xs_hat[:, :, None] - xs_hat[:, None, :]  # shape=(N, 4, 4)
                    diffs_xs_mat = jnp.where(diag_mask, 1.0, diffs_xs)  # shape=(N, 4, 4)
                    denom = jnp.prod(diffs_xs_mat, axis=2)  # shape=(N, 4)
                    basis = numer / (denom + eps)  # shape=(N, 4)
                    return jnp.sum(basis * ys, axis=1)  # shape=(N,)

                def _cubic_interp(x, x0, x1, x2, x3, y0, y1, y2, y3):
                    # In this case, x is distance, y is coordinate.
                    epsilon = zero_term
                    x_min = jnp.min(jnp.array([x0, x1, x2, x3]))
                    x_max = jnp.max(jnp.array([x0, x1, x2, x3]))
                    scale = jnp.maximum(x_max - x_min, zero_term)
                    x_hat = (x - x_min) / scale
                    x0_hat, x1_hat, x2_hat, x3_hat = (x0 - x_min) / scale, (x1 - x_min) / scale, (x2 - x_min) / scale, (x3 - x_min) / scale
                    L0 = ((x_hat - x1_hat) * (x_hat - x2_hat) * (x_hat - x3_hat)) / \
                        ((x0_hat - x1_hat + epsilon) * (x0_hat - x2_hat + epsilon) * (x0_hat - x3_hat + epsilon))
                    L1 = ((x_hat - x0_hat) * (x_hat - x2_hat) * (x_hat - x3_hat)) / \
                        ((x1_hat - x0_hat + epsilon) * (x1_hat - x2_hat + epsilon) * (x1_hat - x3_hat + epsilon))
                    L2 = ((x_hat - x0_hat) * (x_hat - x1_hat) * (x_hat - x3_hat)) / \
                        ((x2_hat - x0_hat + epsilon) * (x2_hat - x1_hat + epsilon) * (x2_hat - x3_hat + epsilon))
                    L3 = ((x_hat - x0_hat) * (x_hat - x1_hat) * (x_hat - x2_hat)) / \
                        ((x3_hat - x0_hat + epsilon) * (x3_hat - x1_hat + epsilon) * (x3_hat - x2_hat + epsilon))
                    return y0 * L0 + y1 * L1 + y2 * L2 + y3 * L3
                in0, in1, in2, in3 = in_source[:-3], in_source[1:-2], in_source[2:-1], in_source[3:]
                d0, d1, d2, d3 = distances[:-3], distances[1:-2], distances[2:-1], distances[3:]
                th0, th1, th2, th3 = -1.5, -0.5, 0.5, 1.5
                segment_inside = in1 * in2
                segment_in2out = in1 * (~in2)
                segment_out2in = (~in1) * in2
                #th_est = cubic_interp(rho, jnp.array([d0, d1, d2, d3]), jnp.array([th0, th1, th2, th3]))
                th_est = _cubic_interp(rho, d0, d1, d2, d3, th0, th1, th2, th3)
                frac_in2out = jnp.clip((th_est - th1), 0.0, 1.0)
                frac_out2in = jnp.clip((th2 - th_est), 0.0, 1.0)
                area_inside = r0 * dth * segment_inside
                area_crossing = r0 * dth * (segment_in2out * frac_in2out + segment_out2in * frac_out2in)
            else:
                # linear interpolation. The boundaries locate at between in0 and in1.
                in0, in1 = in_source[:-1], in_source[1:]
                d0, d1   = distances[:-1], distances[1:]
                segment_inside = in0 * in1
                segment_in2out = in0 * (~in1)
                segment_out2in = (~in0) * in1
                frac     = jnp.clip((rho - d0) / (d1 - d0 + zero_term), 0.0, 1.0)
                area_inside    = r0 * dth * segment_inside
                area_crossing  = r0 * dth * (segment_in2out * frac + segment_out2in * (1.0 - frac))
            return jnp.sum(area_inside + area_crossing)
        area_r = vmap(process_r)(r_values) # (Nr, Ntheta -1) array
        #area_r = jnp.sum(area_r, axis=1)
        #total_area = 0.5 * dr * jnp.sum(area_r[:-1] + area_r[1:])
        #total_area = dr * (0.5 * area_r[0] + jnp.sum(area_r[1:-1]) + 0.5 * area_r[-1])
        total_area = dr * jnp.sum(area_r)
        #total_area = (dr / 3.0) * (area_r[0] + area_r[-1] 
        #                  + 4 * jnp.sum(area_r[1:-1:2])
        #                  + 2 * jnp.sum(area_r[2:-2:2]))
        #total_area = (3 * dr / 8.0) * (area_r[0] + area_r[-1] 
        #                               + 3 * jnp.sum(area_r[1:-1:3] + area_r[2:-1:3]) 
        #                               + 2 * jnp.sum(area_r[3:-3:3]))
        return total_area
    compute_vmap = vmap(compute_for_range, in_axes=(0, 0))
    image_areas = compute_vmap(r_vmap, th_vmap)
    magnification = jnp.sum(image_areas) / rho**2 / jnp.pi 
    return magnification 

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    import jax
    jax.config.update("jax_debug_nans", True)
    q = 0.1
    s = 1.0
    alpha = jnp.deg2rad(30) # angle between lens axis and source trajectory
    tE = 20 # einstein radius crossing time
    t0 = 0.0 # time of peak magnification
    u0 = 0.1 # impact parameter
    rho = 2e-2

    num_points = 500
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
    #magn  = lambda w: mag_uniform(w, rho, r_resolution=1000, th_resolution=500, **test_params)
    magn  = lambda w: mag_uniform(w, rho, r_resolution=500, th_resolution=500, **test_params, cubic=True)
    #magn  = lambda w: mag_binary(w, rho, r_resolution=200, th_resolution=200, u1=0.0, **test_params)
    magn2  = lambda w0: jnp.array([mag_vbbl(w, rho) for w in w0])
    #magn2 =  jit(vmap(magn2, in_axes=(0,)))
    magn =  vmap(magn, in_axes=(0,))

    _ = magn(w_points).block_until_ready()

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
    ax1.plot(t, jnp.abs(magnifications - magnifications2)/magnifications2, "-", ms=1)
    ax1.grid(ls=":")
    ax1.set_ylabel("relative diff")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 1e-2)
    plt.savefig("lc.pdf", bbox_inches="tight")
    plt.show()