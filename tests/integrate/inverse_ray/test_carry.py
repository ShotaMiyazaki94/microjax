import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.image_area_all import image_area0, image_area_all, CarryData


w_center = jnp.array([-0.05 - 0.1j])
q  = 0.5
s  = 1.0
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}
rho = 0.1

yi         = 0
area_all   = 0.0
area_image = jnp.zeros(10)
max_iter   = int(1e+6)
indx       = jnp.zeros((max_iter * 2, 6), dtype=int) # index for checking the overlaps
Nindx      = jnp.zeros((max_iter * 2), dtype=int)     # Number of images at y_index
xmin       = jnp.zeros((max_iter * 2))
xmax       = jnp.zeros((max_iter * 2)) 
area_x     = jnp.zeros((max_iter * 2)) 
y          = jnp.zeros((max_iter * 2)) 
dys        = jnp.zeros((max_iter * 2))
CM2MD = -0.5 * s * (1 - q)/(1 + q) 
dy = 0.01
z_init = jnp.complex128(0.0)
z_current = z_init
x0 = z_init.real
dz2 = jnp.inf
incr = jnp.abs(dy)
incr_inv = 1.0 / incr
dx = incr 
count_x = 0.0
count_all = 0.0
rho2 = rho * rho
finish = jnp.bool_(False)
nlenses=2

carry = CarryData(yi=yi, indx=indx, Nindx=Nindx, xmin=xmin, 
                            xmax=xmax, area_x=area_x, y=y, dys=dys,
                            z_current=z_current, x0=x0, count_x=count_x, 
                            count_all=count_all, dz2=dz2, dz2_last=dz2, dx=dx, finish=finish,
                            w_center=w_center, rho2=rho2, a=a, e1=e1, CM2MD=CM2MD, 
                            incr=incr, incr_inv=incr_inv, max_iter=max_iter, nlenses=nlenses) 

cond = carry.rho2<=carry.dz2 
print(cond.shape, cond)

#area_all, mag, carry = image_area_all(w_center, rho, NBIN=10, nlenses=2, **_params)