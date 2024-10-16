import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
from microjax.inverse_ray.merge_area import calc_source_limb, calculate_overlap_and_range 

@partial(jit, static_argnums=(2, 3, 4, 5, 6, ))
def mag_simple(w_center, rho, resolution=200, Nlimb=1000, offset_r = 1.0, offset_th = 5.0, GRID_RATIO=1, **_params):
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
    # 10 is maximum number of images for triple-lens 
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)

    th_resolution = resolution * GRID_RATIO
    r_grid_normalized = jnp.linspace(0, 1, resolution, endpoint=False)
    th_grid_normalized = jnp.linspace(0, 1, th_resolution, endpoint=False)
    r_mesh_norm, th_mesh_norm = jnp.meshgrid(r_grid_normalized, th_grid_normalized, indexing='ij') 
    
    def compute_for_range(r_range, th_range):
        in_mask = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) &
                          (th_limb > th_range[0]) & (th_limb < th_range[1]))
        def compute_if_in():
            dr = (r_range[1] - r_range[0]) / resolution
            dth = (th_range[1] - th_range[0]) / (resolution * GRID_RATIO)
            r_mesh = r_mesh_norm * (r_range[1] - r_range[0]) + r_range[0]
            th_mesh = th_mesh_norm * (th_range[1] - th_range[0]) + th_range[0]
            z_mesh = jnp.ravel(r_mesh * (jnp.cos(th_mesh) + 1j * jnp.sin(th_mesh)))
            image_mesh = lens_eq(z_mesh - shifted, **_params)
            distances  = jnp.abs(image_mesh - w_center_shifted) 
            image_mask = distances < rho
            area = dr * dth * jnp.sum(r_mesh.ravel() * image_mask.astype(float))
            return area

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
    rho = 1e-4

    t  =  jnp.linspace(-10, 12.5, 1000)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    test_params = {"q": q, "s": s}  # Lens parameters

    from microjax.caustics.extended_source import mag_extended_source
    #magnification  = lambda w: mag_extended_source(w, rho, **test_params)
    magnification  = lambda w: mag_simple(w, rho, resolution=400, **test_params)
    magnifications = vmap(magnification, in_axes=(0, ))(w_points) 

    # Print out the result
    import seaborn as sns
    sns.set_theme(font="Arial", style="ticks")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from microjax.point_source import critical_and_caustic_curves, mag_point_source
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    mags_poi = mag_point_source(w_points, s=s, q=q)
    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

    fig, ax = plt.subplots(figsize=(8,8))
    ax_in = inset_axes(ax,
        width="60%", height="60%", 
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(-0.1, 0.3, 0.6, 0.6)
    )
    ax_in.set_aspect(1)
    ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")
    for cc in caustic_curves:
        ax_in.plot(cc.real, cc.imag, color='black', lw=0.7)
    circles = [
        plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, zorder=-1) for xi, yi in zip(w_points.real, w_points.imag)
    ]
    c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.8)
    ax_in.add_collection(c)
    ax_in.set_aspect(1)
    ax_in.set(xlim=(-1., 1.2), ylim=(-0.8, 1.))

    ax.plot(t, magnifications)
    ax.plot(t, mags_poi, ls="--")
    ax.grid(ls=":")
    #ax.set_yscale("log")
    #fig.savefig("")
    plt.show()