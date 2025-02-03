import jax 
import jax.numpy as jnp
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, vmap, lax
from microjax.inverse_ray.merge_area import calc_source_limb, determine_grid_regions
jax.config.update("jax_enable_x64", True)

#rho = 0.1
#w_center = jnp.complex128(-0.21044241+0.12124181j)
rho = 1e-4
w_center = jnp.complex128(0.23053418+8.75822851e-02j)

q = 1.0
s = 1.0
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"q": q, "s": s, "a": a, "e1": e1}

r_resolution  = 100
th_resolution = 400
Nlimb = 1000
offset_r = 0.5
offset_th  = 1.0

shifted = 0.5 * s * (1 - q) / (1 + q)
w_center_shifted = w_center - shifted
image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r, offset_th, nlenses=2)

r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

print("-vmaps---")
for r ,th in zip(r_scan, th_scan):
    print(r, th)
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
fig = plt.figure(figsize=(6,6))
ax = plt.axes()
for x_grid, y_grid, in_source in zip(x_grids, y_grids, in_sources):
    ax.scatter(x_grid.ravel(), y_grid.ravel(), c='lightgray', s=1)
    ax.scatter(x_grid[in_source].ravel(), y_grid[in_source].ravel(), c='orange', s=1, zorder=2)
from microjax.point_source import critical_and_caustic_curves
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
plt.scatter(critical_curves.ravel().real, 
            critical_curves.ravel().imag, marker=".", color="black", s=3, label="critical curve")
plt.scatter(caustic_curves.ravel().real, 
            caustic_curves.ravel().imag, marker=".", color="crimson", s=3, label="caustic")
plt.scatter(image_limb[mask_limb].ravel().real, 
            image_limb[mask_limb].ravel().imag, 
            s=1,color="purple", zorder=2, label="true image limb")
plt.scatter((image_limb[~mask_limb].real).ravel(), 
            (image_limb[~mask_limb].imag).ravel(), 
            s=1,color="green", zorder=1, label="false image limb")
#plt.scatter((image_limb.real*mask_limb).ravel(), 
#            (image_limb.imag*mask_limb).ravel(), 
#            s=1,color="blue", zorder=2)
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
ax.scatter(w_limb.real, w_limb.imag, color="blue", s=1, label="source limb")
plt.plot(w_center.real, w_center.imag, "*", color="k")
plt.plot(-q/(1+q) * s, 0 , "o",c="k")
plt.plot((1.0)/(1+q) * s, 0 ,"o",c="k")
ax.set_aspect('equal')
plt.legend(fontsize=8)
plt.show()
#import mpld3
#html_string = mpld3.fig_to_html(fig)
#with open("/Users/shotamiyazaki/Desktop/figure.html", "w") as f:
#    f.write(html_string)