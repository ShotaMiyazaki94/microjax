import jax.numpy as jnp
import jax 
import sys
from dataclasses import dataclass
sys.path.append('/Users/shotamiyazaki/Analysis/ulens/microjax/src/microjax')
import numpy as np
import jax.numpy as jnp
from jax import jit
from microjax.inverse_ray_old import image_area0
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import _images_point_source
from microjax.inverse_ray_old import image_area0, CarryData

q  = 0.1
s  = 1.0
rho = 0.1
NBIN = 20
incr  = jnp.abs(rho/NBIN)

a  = 0.5 * s
e1 = q / (1.0 + q) 
w_center = jnp.complex128(0.0 + 0.0j)  # center of mass
w_center -= 0.5*s*(1 - q)/(1 + q)      # mid-point
z_inits, z_mask = _images_point_source(w_center, nlenses=2, a=a, e1=e1) 
w_center += 0.5*s*(1 - q)/(1 + q)      # center of mass
z_inits  += 0.5*s*(1 - q)/(1 + q)      # center of mass

max_iter = int(1e+6)
indx  = jnp.zeros((max_iter * 2, 5), dtype=int) # index for checking the overlaps
Nindx = jnp.zeros((max_iter * 2), dtype=int)     # Number of images at y_index
xmin  = jnp.zeros((max_iter * 2))
xmax  = jnp.zeros((max_iter * 2)) 
area_x= jnp.zeros((max_iter * 2)) 
y     = jnp.zeros((max_iter * 2)) 
dys   = jnp.zeros((max_iter * 2))

yi    = 0
area_all  = 0.0
area_image = jnp.zeros(6)

carry_init = CarryData(yi=yi, indx=indx, Nindx=Nindx, xmin=xmin, xmax=xmax, area_x=area_x, y=y, dys=dys)

for i in range(len(z_inits[z_mask])):
    z_init = z_inits[z_mask][i] 
    xmin   = xmin.at[yi].set(z_init.real) 
    xmax   = xmax.at[yi].set(z_init.real)
    dy     = incr
    carry  = carry_init 
    area_i, carry_all = image_area0(w_center, rho, z_inits[z_mask][i], dy, carry, nlenses=2, s=s, q=q)
