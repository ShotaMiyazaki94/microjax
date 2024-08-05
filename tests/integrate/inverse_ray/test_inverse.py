import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source
from microjax.inverse_ray import image_area0, image_area_all, CarryData

NBIN=20
nlenses=2

w_center = jnp.complex128(-0.05 - 0.1j)
q  = 0.5
s  = 1.0
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}
rho = 0.1

incr  = jnp.abs(rho / NBIN)
incr2 = incr * 0.5

w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
z_inits_mid, z_mask = _images_point_source(w_center_mid, nlenses=nlenses, **_params)
z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)


yi         = 0
area_all   = 0.0
area_image = jnp.zeros(10)
max_iter   = int(1e+6)
indx       = jnp.zeros((max_iter * 2, 5), dtype=int) # index for checking the overlaps
Nindx      = jnp.zeros((max_iter * 2), dtype=int)     # Number of images at y_index
xmin       = jnp.zeros((max_iter * 2))
xmax       = jnp.zeros((max_iter * 2)) 
area_x     = jnp.zeros((max_iter * 2)) 
y          = jnp.zeros((max_iter * 2)) 
dys        = jnp.zeros((max_iter * 2))
CM2MD = -0.5 * s * (1 - q)/(1 + q) 
dz2 = jnp.inf
dx = incr 
count_x = 0.0
count_all = 0.0
rho2 = rho * rho
finish = jnp.bool_(False)

z_init = z_inits[z_mask][0]
dy     = incr
carry  = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)