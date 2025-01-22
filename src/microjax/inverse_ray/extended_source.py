import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap, custom_jvp
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, determine_grid_regions
from microjax.inverse_ray.limb_darkening import Is_limb_1st

@partial(jit, static_argnames=("nlenses", "cubic", "r_resolution", "th_resolution", "Nlimb", "u1",
                               "offset_r", "offset_th", "delta_c"))
def mag_binary(w_center, rho, nlenses=2, cubic=True, u1=0.0, r_resolution=1000, th_resolution=4000, 
               Nlimb=1000, offset_r = 0.5, offset_th = 10.0, delta_c=0.05, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r, offset_th, nlenses=nlenses)

    @partial(jit, static_argnames=("cubic")) 
    def _process_r(r0, th_values, cubic=True):
        dth = (th_values[1] - th_values[0])
        distances = distance_from_source(r0, th_values, w_center_shifted, shifted, nlenses=nlenses, **_params)
        in_num = in_source(distances, rho)
        Is     = Is_limb_1st(distances / rho, u1=u1)
        zero_term = 1e-10
        if cubic:
            in0_num, in1_num, in2_num, in3_num, in4_num = in_num[:-4], in_num[1:-3], in_num[2:-2], in_num[3:-1], in_num[4:]
            d0, d1, d2, d3, d4 = distances[:-4], distances[1:-3], distances[2:-2], distances[3:-1], distances[4:]
            th0, th1, th2, th3 = jnp.arange(4)
            num_inside  = in1_num * in2_num * in3_num
            num_B1      = (1.0 - in1_num) * in2_num * in3_num
            num_B2      = in1_num * in2_num * (1.0 - in3_num)
            th_est_B1   = cubic_interp(rho, d0, d1, d2, d3, th0, th1, th2, th3, epsilon=zero_term)
            th_est_B2   = cubic_interp(rho, d1, d2, d3, d4, th0, th1, th2, th3, epsilon=zero_term)
            delta_B1    = jnp.clip(th2 - th_est_B1, 0.0, 1.0)
            delta_B2    = jnp.clip(th_est_B2 - th1, 0.0, 1.0)
            fac_B1      = jnp.where(delta_B1 > delta_c,
                                    (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B1) * (0.5 + delta_B1),
                                    (2.0 / 3.0) * delta_B1 + 0.5)
            fac_B2      = jnp.where(delta_B2 > delta_c,
                                    (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B2) * (0.5 + delta_B2),
                                    (2.0 / 3.0) * delta_B2 + 0.5)
            area_inside = r0 * dth * Is[2:-2] * num_inside
            area_B1     = r0 * dth * Is[2:-2] * fac_B1 * num_B1
            area_B2     = r0 * dth * Is[2:-2] * fac_B2 * num_B2
        else:
            in0_num, in1_num, in2_num = in_num[:-2], in_num[1:-1], in_num[2:]
            d0, d1, d2 = distances[:-2], distances[1:-1], distances[2:]
            num_inside     = in0_num * in1_num * in2_num
            num_B1         = (1.0 - in0_num) * in1_num * in2_num 
            num_B2         = in0_num * in1_num * (1.0 - in2_num)
            delta_B1   = jnp.clip((rho - d0) / (d1 - d0 + zero_term), 0.0, 1.0) 
            delta_B2   = jnp.clip((d2 - rho) / (d2 - d1 + zero_term), 0.0, 1.0)
            fac_B1 = jnp.where(delta_B1 > delta_c, 
                               (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B1) * (0.5 + delta_B1), 
                               (2.0 / 3.0) * delta_B1 + 0.5)
            fac_B2 = jnp.where(delta_B2 > delta_c, 
                               (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B2) * (0.5 + delta_B2), 
                               (2.0 / 3.0) * delta_B2 + 0.5)
            area_inside = r0 * dth * Is[1:-1] * num_inside
            area_B1     = r0 * dth * Is[1:-1] * fac_B1 * num_B1
            area_B2     = r0 * dth * Is[1:-1] * fac_B2 * num_B2
        return jnp.sum(area_inside + area_B1 + area_B2)
    
    @partial(jit, static_argnames=("cubic"))  
    def _compute_for_range(r_range, th_range, cubic=True):
        r_values = jnp.linspace(r_range[0], r_range[1], r_resolution, endpoint=False)
        th_values = jnp.linspace(th_range[0], th_range[1], th_resolution, endpoint=False)
        area_r = vmap(lambda r: _process_r(r, th_values, cubic=cubic))(r_values)
        dr = r_values[1] - r_values[0]
        total_area = dr * jnp.sum(area_r) # trapezoidal integration
        return total_area
    
    def scan_images(carry, inputs):
        r_range, th_range = inputs
        total_area = _compute_for_range(r_range, th_range, cubic=cubic)
        return carry + total_area, None

    inputs = (r_scan, th_scan)
    magnification_unnorm, _ = lax.scan(scan_images, 0.0, inputs, unroll=1)
    magnification = magnification_unnorm / rho**2 
    return magnification 



    """
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
    """

@partial(jit, static_argnames=("nlenses", "r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th", "cubic",))
def mag_uniform(w_center, rho, nlenses=2, r_resolution=1000, th_resolution=4000, 
                Nlimb=2000, offset_r=0.5, offset_th=10.0, cubic=True, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r, offset_th, nlenses=nlenses)
    
    @partial(jit, static_argnames=("cubic")) 
    def _process_r(r0, th_values, cubic=True):
        dth = (th_values[1] - th_values[0])
        distances = distance_from_source(r0, th_values, w_center_shifted, shifted, nlenses=nlenses, **_params)
        in_num = in_source(distances, rho)
        zero_term = 1e-10
        if cubic:
            in0_num, in1_num, in2_num, in3_num = in_num[:-3], in_num[1:-2], in_num[2:-1], in_num[3:]
            d0, d1, d2, d3 = distances[:-3], distances[1:-2], distances[2:-1], distances[3:]
            th0, th1, th2, th3 = jnp.arange(4)
            num_inside  = in1_num * in2_num
            num_in2out  = in1_num * (1.0 - in2_num)
            num_out2in  = (1.0 - in1_num) * in2_num
            th_est      = cubic_interp(rho, d0, d1, d2, d3, th0, th1, th2, th3, epsilon=zero_term)
            frac_in2out = jnp.clip((th_est - th1), 0.0, 1.0)
            frac_out2in = jnp.clip((th2 - th_est), 0.0, 1.0)
            area_inside = r0 * dth * num_inside
            area_crossing = r0 * dth * (num_in2out * frac_in2out + num_out2in * frac_out2in)
        else:
            in0_num, in1_num = in_num[:-1], in_num[1:]
            d0, d1   = distances[:-1], distances[1:]
            num_inside     = in0_num * in1_num
            area_inside    = r0 * dth * num_inside
            num_in2out = in0_num * (1.0 - in1_num)
            num_out2in = (1.0 - in0_num) * in1_num
            frac = jnp.clip((rho - d0) / (d1 - d0 + zero_term), 0.0, 1.0)
            area_crossing  = r0 * dth * (num_in2out * frac + num_out2in * (1.0 - frac))
        return jnp.sum(area_inside + area_crossing)  
    
    @partial(jit, static_argnames=("cubic"))  
    def _compute_for_range(r_range, th_range, cubic=True):
        r_values = jnp.linspace(r_range[0], r_range[1], r_resolution, endpoint=False)
        th_values = jnp.linspace(th_range[0], th_range[1], th_resolution, endpoint=False)
        area_r = vmap(lambda r: _process_r(r, th_values, cubic=cubic))(r_values)
        dr = r_values[1] - r_values[0]
        total_area = dr * jnp.sum(area_r) # trapezoidal integration
        return total_area
    
    def scan_images(carry, inputs):
        r_range, th_range = inputs
        total_area = _compute_for_range(r_range, th_range, cubic=cubic)
        return carry + total_area, None

    inputs = (r_scan, th_scan)
    magnification_unnorm, _ = lax.scan(scan_images, 0.0, inputs, unroll=1)
    magnification = magnification_unnorm / rho**2 / jnp.pi
    return magnification 

def cubic_interp(x, x0, x1, x2, x3, y0, y1, y2, y3, epsilon=1e-12):
    # Implemented algebraically, much faster and memory efficient than polyfit that uses matrix manipulation.
    # In this case, x is distance, y is coordinate.
    x_min = jnp.min(jnp.array([x0, x1, x2, x3]))
    x_max = jnp.max(jnp.array([x0, x1, x2, x3]))
    scale = jnp.maximum(x_max - x_min, epsilon)
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

@custom_jvp 
def in_source(distances, rho):
    return jnp.where(rho - distances < 0.0, 0.0, 1.0)

@in_source.defjvp
def in_source_jvp(primal, tangent):
    distances, rho = primal
    distances_dot, rho_dot = tangent
    primal_out = in_source(distances, rho)

    z = (rho - distances) / rho 
    factor = 100.0 
    sigmoid_input = factor * z
    sigmoid = jax.nn.sigmoid(sigmoid_input)
    sigmoid_derivative = sigmoid * (1.0 - sigmoid) * factor
    dz_distances = -1.0 / rho
    dz_rho = distances / rho**2
    tangent_out = sigmoid_derivative * (dz_distances * distances_dot + dz_rho * rho_dot)
    primal_out = sigmoid
    return primal_out, tangent_out

def distance_from_source(r0, th_values, w_center_shifted, shifted, nlenses=2, **_params):
    x_th = r0 * jnp.cos(th_values)
    y_th = r0 * jnp.sin(th_values)
    z_th = x_th + 1j * y_th
    image_mesh = lens_eq(z_th - shifted, nlenses=nlenses, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    return distances

if __name__ == "__main__":
    import time
    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_debug_nans", True)
    q = 0.5
    s = 0.9
    alpha = jnp.deg2rad(30) # angle between lens axis and source trajectory
    tE = 10 # einstein radius crossing time
    t0 = 0.0 # time of peak magnification
    u0 = 0.1 # impact parameter
    rho = 2e-2

    num_points = 1000
    t  =  jnp.linspace(-5.0, 5.0, num_points)
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
    #magn  = lambda w: mag_uniform(w, rho, r_resolution=2000, th_resolution=1000, **test_params, cubic=True)
    #magn  = lambda w: mag_uniform(w, rho, r_resolution=1000, th_resolution=1000, **test_params, cubic=True)
    magn  = lambda w: mag_binary(w, rho, r_resolution=1000, th_resolution=1000, u1=0.0, **test_params, cubic=True)
    magn2  = lambda w0: jnp.array([mag_vbbl(w, rho) for w in w0])
    #magn2 =  jit(vmap(magn2, in_axes=(0,)))
    magn =  jit(vmap(magn, in_axes=(0,)))

    _ = magn(w_points).block_until_ready()

    print("start computation")
    start = time.time()
    magnifications = magn(w_points).block_until_ready() 
    end = time.time()
    print("computation time: %.6f sec per points"%((end - start)/num_points))
    start = time.time()
    magnifications2 = magn2(w_points) 
    end = time.time()
    print("computation time: %.6f sec per points"%((end - start)/num_points))
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

    ax.plot(t, magnifications, ".", label="microjax")
    ax.plot(t, magnifications2, "-", label="VBBinaryLensing")
    ax.set_title("extended source evaluation")
    #ax.plot(t, mags_poi, ls="--")
    ax.grid(ls=":")
    ax.set_ylabel("magnification")
    ax1.plot(t, jnp.abs(magnifications - magnifications2)/magnifications2, "-", ms=1)
    ax1.grid(ls=":")
    ax1.set_ylabel("relative diff")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 1e-2)
    ax.legend(loc="upper left")
    plt.savefig("lc.pdf", bbox_inches="tight")
    plt.close()