import unittest
import numpy as np
import jax.numpy as jnp
from jax import jit
from microjax.extend_source import image_area0_binary
from microjax.point_source import _images_point_source_binary, critical_and_caustic_curves_binary
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

def test():
    w_center = jnp.array([0.0 + 0.0j])
    q = 0.3
    s = 1.0
    rho = 0.1
    NBIN = 10
    incr = jnp.abs(rho/NBIN)
    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    w_center -= 0.5*s*(1 - q)/(1 + q) # mid-point
    z_inits, z_mask = _images_point_source_binary(w_center, a, e1) 
    w_center += 0.5*s*(1 - q)/(1 + q) # center of mass
    z_inits  += 0.5*s*(1 - q)/(1 + q) # center of mass
    max_iter = 10000000
    indx  = jnp.zeros((max_iter * 2, 10), dtype=int)
    Nindx = jnp.zeros((max_iter * 2), dtype=int)
    xmin  = jnp.zeros((max_iter * 2))
    xmax  = jnp.zeros((max_iter * 2)) 
    area_x= jnp.zeros((max_iter * 2)) 
    y     = jnp.zeros((max_iter * 2)) 
    dys   = jnp.zeros((max_iter * 2)) 
    count = 0.0
    yi    = 0
    for i in range(len(z_inits[z_mask])):
        print("%d image positive"%(i))
        xmin = xmin.at[yi].set(z_inits[z_mask][i].real) 
        xmax = xmax.at[yi].set(z_inits[z_mask][i].real)
        dy     = incr
        z_init = z_inits[z_mask][i] 
        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
        area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry)
        (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
        count += area
        print("%d image negative"%(i))
        dy     = -incr
        z_init = z_init + 1j * dy
        #z_init = jnp.complex128(xmax[0] + 1j * z_inits[z_mask][i].imag + 1j * dy)
        xmin = xmin.at[yi].set(0) 
        xmax = xmax.at[yi].set(0)
        #xmin = xmin.at[yi].set(xmin[yi - 1]) 
        #xmax = xmax.at[yi].set(xmax[yi - 1]) 
        #y    = y.at[yi].set(y[yi - 1]) 
        #dys  = dys.at[yi].set(dy)
        yi  += 1 
        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
        area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry) 
        (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
        count += area 
        xmin = xmin.at[yi].set(0) 
        xmax = xmax.at[yi].set(0)
        yi  += 1 
    print("count_all:", count)
    print("dy       :", dy)
    print("area*dy  :", count * dy)
    (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry

    N_limb = 5000
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
    w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
    image, mask = _images_point_source_binary(w_limb_shift, a=a, e1=e1) # half-axis coordinate
    image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
    
    fig = plt.figure()
    ax = plt.axes()
    mask_x = jnp.bool_(xmin!=0)
    for i in range(len(xmin[mask_x])):
        plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i])
    for i in range(len(z_inits[z_mask])):
        plt.scatter(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, marker="*", zorder=2)
        plt.text(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, s="%d"%(i), zorder=2)
    source = plt.Circle((w_center.real, w_center.imag), rho, color='b', fill=False)
    ax.add_patch(source)
    plt.plot(-q*s, 0 ,".",c="k")
    plt.plot((1.0-q)*s, 0 ,".",c="k")
    crit_tri, cau_tri = critical_and_caustic_curves_binary(npts=1000, q=q, s=s)
    plt.scatter(image_limb[mask].ravel().real, image_limb[mask].ravel().imag, s=1,color="purple")
    plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag,   marker=".", color="red", s=1)
    plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1) 
    plt.axis("equal")
    plt.show()    

if __name__ == "__main__":
    test()