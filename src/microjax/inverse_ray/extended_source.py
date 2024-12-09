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

def limb_1st_norm(r, u1=0.0):
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

    Examples
    --------
    import jax.numpy as jnp
    limb_1st_norm(0.5, u1=0.3)
    DeviceArray(0.565955, dtype=float32)
    
    r = jnp.linspace(0, 1, 5)
    limb_1st_norm(r, u1=0.3)
    DeviceArray([1.042716 , 0.8890367, 0.7364623, 0.5852573, 0.        ], dtype=float32)
    """
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    I  = I0 * (1.0 - u1 * (1.0 - jnp.sqrt(1.0 - r)))
    return jnp.where(r <= 1.0, I, 0.0) 

def mag_binary(w_center, rho, r_resolution=250, th_resolution=4000, Nlimb=200, 
                offset_r = 1.0, offset_th = 10.0, u1=0.0, **_params):
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
    # if merging is correct, 6 is emperically sufficient for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:,1])][-6:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-6:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    in_mask = _compute_in_mask(r_limb.ravel()*mask_limb.ravel(), th_limb.ravel()*mask_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    # binary-lens should have less than 5 images.
    r_vmap   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:6]
    th_vmap  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:6]

    r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

    def compute_for_range(r_range, th_range):
        r_in  = (r_limb > r_range[0]) & (r_limb < r_range[1])
        th_in = (th_limb > th_range[0]) & (th_limb < th_range[1]) 
        in_mask = jnp.any(r_in & th_in)
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
                #th0, th1 = th_values[:-1], th_values[1:]
                d0, d1   = distances[:-1], distances[1:]
                segment_inside = (in0 == 1) & (in1 == 1)
                segment_in2out = (in0 == 1) & (in1 == 0)
                segment_out2in = (in0 == 0) & (in1 == 1)
                frac = jnp.clip((rho - d0) / (d1 - d0), 0.0, 1.0)
                area_inside    = r0 * dth * segment_inside
                area_crossing  = r0 * dth * (segment_in2out * frac + segment_out2in * (1.0 - frac))
                return area_inside + area_crossing
                #return jnp.sum(area_inside + area_crossing)
            area_r = vmap(process_r)(r_values) # (Nr, Ntheta -1) array
            trapezoid = area_r[:-1] + area_r[1:] 
            total_area = 0.5 * dr * jnp.sum(trapezoid)
            #jax.debug.print("{}",area_each_r.shape)
            #total_area = dr * jnp.sum(area_r)
            return total_area
        return jnp.where(in_mask, compute_if_in(), 0.0)
    compute_vmap = vmap(compute_for_range, in_axes=(0, 0))
    image_areas = compute_vmap(r_vmap, th_vmap)
    magnification = jnp.sum(image_areas) / rho**2 / jnp.pi 
    return magnification 

def mag_uniform_bisection(w_center, rho, r_resolution=250, th_resolution=4000, 
                          Nlimb=200, offset_r = 1.0, offset_th = 10.0, **_params):
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
    # if merging is correct, 6 is emperically sufficient for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    in_mask = _compute_in_mask(r_limb.ravel()*mask_limb.ravel(), th_limb.ravel()*mask_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    # binary-lens should have less than 5 images.
    r_vmap   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:6]
    th_vmap  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:6]

    r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
    th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

    def d_residual(r0, theta):
        x_mid = r0 * jnp.cos(theta)
        y_mid = r0 * jnp.sin(theta)
        z_mid = x_mid + 1j * y_mid
        dist_mid = jnp.abs(lens_eq(z_mid - shifted, **_params) - w_center_shifted)
        return dist_mid - rho
    
    def bisection(r0, th_low, th_high, max_iter=3):
        def body_fn(carry, _):
            th_l, th_h = carry
            th_mid = 0.5 * (th_l + th_h)
            d_res_mid = d_residual(r0, th_mid) # plus: out, minus: in
            mid_is_in = d_res_mid < 0.0
            th_l_new = jnp.where(mid_is_in, th_l, th_mid) # if in, lower one is NOT replaced.
            th_h_new = jnp.where(mid_is_in, th_mid, th_h) # if in, higher one is replaced.
            return (th_l_new, th_h_new), None
        
        (th_l_final, th_h_final), _ = lax.scan(body_fn, (th_low, th_high), None, length=max_iter)
        return 0.5 * (th_l_final + th_h_final)




@partial(jit, static_argnames=("r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th"))
def mag_uniform(w_center, rho, r_resolution=250, th_resolution=4000, Nlimb=200, 
                offset_r = 1.0, offset_th = 10.0, **_params):
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
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    # select matched regions including image limbs. binary-lens microlensing should have less than 5 images.
    # note: theta boundary is 0 and 2pi so that images containing the boundary are divided into two.
    # The reason why the 6 is chosen is that never all five images align on the binary axis, though three may be.
    in_mask = _compute_in_mask(r_limb.ravel()*mask_limb.ravel(), th_limb.ravel()*mask_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    r_vmap   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:6]
    th_vmap  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:6]

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
            in_source = (distances - rho < 0.0).astype(float)
            in0, in1 = in_source[:-1], in_source[1:]
            #th0, th1 = th_values[:-1], th_values[1:]
            d0, d1   = distances[:-1], distances[1:]
            segment_inside = (in0 == 1) & (in1 == 1)
            segment_in2out = (in0 == 1) & (in1 == 0)
            segment_out2in = (in0 == 0) & (in1 == 1)
            frac = jnp.clip((rho - d0) / (d1 - d0), 0.0, 1.0)
            area_inside    = r0 * dth * segment_inside
            area_crossing  = r0 * dth * (segment_in2out * frac + segment_out2in * (1.0 - frac))
            return area_inside + area_crossing
        area_r = vmap(process_r)(r_values) # (Nr, Ntheta -1) array
        #trapezoid = area_r[:-1] + area_r[1:] 
        #total_area = 0.5 * dr * jnp.sum(trapezoid)
        total_area = dr * jnp.sum(area_r)
        return total_area
    compute_vmap = vmap(compute_for_range, in_axes=(0, 0))
    image_areas = compute_vmap(r_vmap, th_vmap)
    magnification = jnp.sum(image_areas) / rho**2 / jnp.pi 
    return magnification 

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    q = 0.1
    s = 1.0
    alpha = jnp.deg2rad(30) # angle between lens axis and source trajectory
    tE = 20 # einstein radius crossing time
    t0 = 0.0 # time of peak magnification
    u0 = 0.1 # impact parameter
    rho = 5e-2

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
    magn  = lambda w: mag_uniform(w, rho, r_resolution=500, th_resolution=500, **test_params, Nlimb=100)
    #magn  = lambda w: mag_simple(w, rho, r_resolution=100, th_resolution=100, **test_params, Nlimb=100)
    magn2  = lambda w0: jnp.array([mag_vbbl(w, rho) for w in w0])
    #magn2 =  jit(vmap(magn2, in_axes=(0,)))
    magn =  jit(vmap(magn, in_axes=(0,)))

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
    ax1.plot(t, jnp.abs(magnifications - magnifications2)/magnifications2)
    ax1.grid(ls=":")
    ax1.set_ylabel("relative diff")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 1e-2)
    plt.show()