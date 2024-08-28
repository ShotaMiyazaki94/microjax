import unittest
import numpy as np
import jax.numpy as jnp
from jax import jit
from microjax.inverse_ray.extend_source_python import image_area0_binary
from microjax.point_source import _images_point_source_binary, critical_and_caustic_curves_binary
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

def test_image4():
    w_center = jnp.array([-0.0 + 0.0j])
    q = 0.5
    s = 1.0
    rho = 0.2
    NBIN = 5
    incr  = jnp.abs(rho/NBIN)
    incr2 = incr*0.5
    incr2margin = incr2*1.01

    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    w_center -= 0.5*s*(1 - q)/(1 + q) # mid-point
    z_inits, z_mask = _images_point_source_binary(w_center, a, e1) 
    w_center += 0.5*s*(1 - q)/(1 + q) # center of mass
    z_inits  += 0.5*s*(1 - q)/(1 + q) # center of mass

    max_iter = 10000000
    indx  = jnp.zeros((max_iter * 2, 10), dtype=int) # index for checking the overlaps
    Nindx = jnp.zeros((max_iter * 2), dtype=int)     # Number of images at y_index
    xmin  = jnp.zeros((max_iter * 2))
    xmax  = jnp.zeros((max_iter * 2)) 
    area_x= jnp.zeros((max_iter * 2)) 
    y     = jnp.zeros((max_iter * 2)) 
    dys   = jnp.zeros((max_iter * 2))

    yi    = 0
    area_all  = 0.0
    area_i    = 0.0
    area_image = jnp.zeros(6)
    overlap    = jnp.zeros(6)
    for i in range(len(z_inits[z_mask])):
        if overlap[i] == 1:
            continue
        print("%d image positive"%(i))
        z_init = z_inits[z_mask][i] 
        xmin = xmin.at[yi].set(z_init.real) 
        xmax = xmax.at[yi].set(z_init.real)
        dy     = incr
        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
        area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry)
        (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
        area_i += area
        print("%d image negative"%(i))
        dy     = -incr
        z_init = z_init + 1j * dy
        yi  += 1 
        carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
        area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry) 
        (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
        area_i += area 
        yi  += 1
        print("%d image extra-image expolaration"%(i))
        nyi = yi
        area_bound = 0.0;
        for j in range(yi):
            dxmax = xmax[j + 1] - xmax[j]
            dxmin = xmin[j + 1] - xmin[j]
            if area_x[j + 1] > 0.0:
                if dxmax > 1.1 * incr:
                    z_init  = jnp.complex128(xmax[j + 1] + 1j * y[j])
                    xmin = xmin.at[j].set(xmax[j])
                    xmax = xmax.at[j].set(xmax[j + 1])
                    dy   = -dys[j]
                    dys  = dys.at[j].set(dy)
                    yi  += 1
                    carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
                    area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry) 
                    (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
                    area_i     += area
                    area_bound += area
                    if area<=0.0:
                        yi -= 1
                if dxmin > 1.1 * incr:
                    z_init = jnp.complex128(xmax[j + 1] - incr + 1j *  y[j + 1])
                    xmin   = xmin.at[yi].set(xmin[j])
                    xmax   = xmax.at[yi].set(xmin[j + 1])
                    dy     = dys[j]
                    dys    = dys.at[yi].set(dy)
                    yi    += 1
                    carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
                    area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry) 
                    (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
                    area_i     += area
                    area_bound += area
                    if area<=0.0:
                        yi -= 1
                if dxmin < -1.1 * incr:
                    z_init = jnp.complex128(xmin[j] - incr + 1j *  y[j])
                    xmin   = xmin.at[yi].set(xmin[j + 1])
                    xmax   = xmax.at[yi].set(xmin[j])
                    dy     = -dys[j]
                    dys    = dys.at[yi].set(dy)
                    yi    += 1
                    carry = (yi, indx, Nindx, xmax, xmin, area_x, y, dys) 
                    area, carry = image_area0_binary(w_center, z_init, q, s, rho, dy, carry) 
                    (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
                    area_i     += area
                    area_bound += area
                    if area<=0.0:
                        yi -= 1
            if (j == nyi - 1) and (area_bound > 0.0) and (yi > nyi):
                nyi = yi
        
        area_all += area_i
        area_image = area_image.at[i].set(area_i)
        nimage    = len(z_inits[z_mask]) 
        for ii in range(nimage):
            if ii == i:
                continue
            for j in range(nyi):
                if z_inits[z_mask][ii].imag >= y[j] - incr2margin and z_inits[z_mask][ii].imag <= y[j] + incr2margin:
                    if z_inits[z_mask][ii].real >= xmin[j] - incr2margin and z_inits[z_mask][ii].real <= xmax[j] + incr2margin:
                        if ii < i:
                            area_all -= area_image[ii]
                        else:
                            overlap = overlap.at[ii].set(1)
                    break
        
        for j in range(nyi):
            index = int(y[j] / incr + max_iter)
            Nindx = Nindx.at[index].set(0)
                             
    print("count_all:", area_all)
    (yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry

    N_limb = 5000
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
    w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
    image, mask = _images_point_source_binary(w_limb_shift, a=a, e1=e1) # half-axis coordinate
    image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
    
    fig = plt.figure()
    ax = plt.axes()
    mask_x = jnp.bool_(area_x!=0)
    cmap = plt.get_cmap("tab10")
    pos_neg = jnp.where(dys[mask_x] > 0, 1.0, 0.0)
    for i in range(len(xmin[mask_x])):
        plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i], color=cmap(pos_neg[i]))
        plt.plot(xmin[mask_x][i], y[mask_x][i], ".", color="k")
        plt.plot(xmax[mask_x][i], y[mask_x][i], ".", color="k")
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
    test_image4()