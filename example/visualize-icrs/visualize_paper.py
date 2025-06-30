import jax
from jax import lax, vmap, jit
import numpy as np
import jax.numpy as jnp
from microjax.point_source import _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, define_regions
from microjax.point_source import critical_and_caustic_curves
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [6,6.0]
#plt.rcParams['text.usetex'] = True
import seaborn as sns
sns.set_theme(font="serif", font_scale=1.2,style="ticks")
jax.config.update("jax_enable_x64", True)
from functools import partial

w_center = jnp.complex128(-0.0510682 +0.1j)
#w_center = jnp.complex128(0.20732831+0.23517111j)
s = 1.1
q = 0.1
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1, "q": q, "s": s}
rho = 0.05

nlenses = 2
Nlimb = 200
margin_r  = 0.5
margin_th = 1.0
bins_r = 50
bins_th= 120

critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
image_limb = image_limb.ravel()
mask_limb = mask_limb.ravel()
r_scan, th_scan = define_regions(image_limb, mask_limb, rho, bins_r=bins_r, bins_th=bins_th, 
                                 margin_r=margin_r, margin_th=margin_th, nlenses=nlenses)
for i, (r, th) in enumerate(zip(r_scan, th_scan)):
    print(i, r, th)

from microjax.point_source import lens_eq
shifted = 0.5 * s * (1 - q) / (1 + q)
w_center_shifted = w_center - shifted
r_resolution  = 50
th_resolution = 50
r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=True)
th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=True)
@jit
def plot(r_range, th_range):
    r_values = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
    th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]
    r_mesh, th_mesh = jnp.meshgrid(r_values, th_values, indexing='ij')
    z_grid = r_mesh * (jnp.cos(th_mesh) + 1j * jnp.sin(th_mesh))
    image_mesh = lens_eq(z_grid - shifted, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    in_source = (distances - rho < 0.0)
    return z_grid.real, z_grid.imag, in_source
vmap_plot = vmap(plot, in_axes=(0, 0))
x_grids, y_grids, in_sources = vmap_plot(r_scan, th_scan)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
for x_grid, y_grid, in_source in zip(x_grids, y_grids, in_sources):
    ax.scatter(x_grid.ravel(), y_grid.ravel(), c='lightgray', s=0.5, zorder=0)
    ax.scatter(x_grid[in_source].ravel(), y_grid[in_source].ravel(), c='orange', s=0.5, zorder=1)

w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
ax.scatter(w_limb.real, w_limb.imag, color="blue", s=1, label="source limb")
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
ax.scatter(critical_curves.ravel().real, critical_curves.ravel().imag, 
            marker=".", color="green", s=3, label="critical curve")
ax.scatter(caustic_curves.ravel().real, caustic_curves.ravel().imag, 
            marker=".", color="crimson", s=3, label="caustic")
ax.plot(-q/(1+q) * s, 0 , "o",c="k")
ax.plot((1.0)/(1+q) * s, 0 ,"o",c="k")
ax.scatter(image_limb[mask_limb].real, image_limb[mask_limb].imag, s=1, color="purple", label="image limb")
ax.scatter(0, 0, marker="x", color="k")
ax.grid(ls="--")
#ax.scatter(image_limb[~mask_limb].real, image_limb[~mask_limb].imag, s=1, color="lightblue")
ax.axis("equal")
ax.legend(loc="lower right", fontsize=9)

axins = inset_axes(ax, width="25%", height="25%", loc="upper right")
for x_grid, y_grid, in_source in zip(x_grids, y_grids, in_sources):
    axins.scatter(x_grid.ravel(), y_grid.ravel(), c='lightgray', s=0.5, zorder=0)
    axins.scatter(x_grid[in_source].ravel(), y_grid[in_source].ravel(), c='orange', s=0.5, zorder=1)
axins.scatter(image_limb[mask_limb].real, image_limb[mask_limb].imag, s=1, color="purple")
axins.scatter(w_limb.real, w_limb.imag, color="blue", s=1)
x0, y0 = 1.17, -0.04
dx = 0.025
x1, x2 = x0-dx, x0+dx  
y1, y2 = y0-dx, y0+dx  
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
ax.set_title(rf"$N_{{\rm limb}} = {Nlimb:d}$, $N_r={r_resolution:d}$, $N_\theta={th_resolution:d}$")
ax.set_xlabel("$x$ [$R_{\\rm E}$]")
ax.set_ylabel("$y$ [$R_{\\rm E}$]")
plt.savefig("example/visualize-icrs/visualize_example.png", bbox_inches="tight", dpi=300)
plt.show()