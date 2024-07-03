from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.poly_solver import poly_roots_EA_multi as poly_roots
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple 
from microjax.coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary
from microjax.point_source import _lens_eq_binary, _lens_eq_triple
from microjax.point_source import mag_point_source_binary, mag_point_source_triple

def source_profile_limb1(dz, u1=0.0):
    mu = jnp.sqrt(1.0 - dz*dz)
    return 1 - u1 * (1.0 - mu)

def image_area_binary(w_center, z_inits, q, s, rho, NBIN=20, max_iter=100000):

    # used arrays
    indx   = jnp.zeros((max_iter * 2, 4), dtype=int)
    Nindx  = jnp.zeros((max_iter * 2,),   dtype=int)
    xmax   = jnp.zeros((max_iter * 4,))
    xmin   = jnp.zeros((max_iter * 4,))
    area_x = jnp.zeros((max_iter * 4,))
    y      = jnp.zeros((max_iter * 4,))
    dys    = jnp.zeros((max_iter * 4,))

    nimage  = len(z_inits)
    overlap = jnp.zeros(6) # binary-lens
    incr    = rho / NBIN  
    incr2   = 0.5 * incr 
    incr2margin = incr2 * 1.01  
    area_i  = jnp.zeros(6)
    
    if rho <= 0:
        return 0.0
    
    for i in range(1, nimage+1):
        if overlap[i] == 1:
            continue
        # search image toward +y
        yi = 0
        dy = incr
        z_init = z_inits[i]
        xmin = xmin.at[0].set(z_init[i].real)
        xmax = xmax.at[0].set(z_init[i].real)
        area_i, yi = image_area0_binary(w_center, z_init, q, s, dy, indx, Nindx, 
                                        xmax, xmin, area_x, y, dys, max_iter)
        # search image toward -y
        dy       = -incr
        z_init   = jnp.complex128(xmax[0], z_inits[i].imag + dy)
        xmin = xmin.at[yi].set(xmin[0])
        xmax = xmax.at[yi].set(xmax[0])
        y    = y.at[yi].set(y[0])




def image_area0_binary(w_center, z_init, q, s, rho, dy, indx, Nindx, xmax, xmin, area_x, y, dys, max_iter):
    """
    calculate an image area with binary-lens by inverse-ray shooting.

    Args:
        w_center (complex) : The complex position of the source.
        z_init (complex)   : Initial position of the image point.
        q (float)          : Mass ratio of the binary lens.
        s (float)          : Separation between the two lenses.
        rho (float)        : Radius of the source.
        dy (float,opt)     : Step size in the y-direction, default is 1e-4.
        max_iter (int, opt): Maximum number of iterations, default is 10000.

    Returns:
        float: Total brightness of the image area, 
                proportional to its physical area and adjusted for limb darkening.
    """

    z_current = z_init
    x0   = z_init.real
    a    = 0.5 * s
    e1   = q / (1.0 + q) 
    dz2  = jnp.inf
    incr      = jnp.abs(dy)
    incr_inv  = 1.0 / incr
    yi        = 0
    dx        = incr 
    count_x   = 0.0
    count_all = 0.0
    rho2      = rho * rho
    
    yi = 0
    while True:
        zis = _lens_eq_binary(z_current, a=a, e1=e1) # inversed point from image into source
        dz2_last = dz2
        dz  = jnp.abs(w_center - zis)
        dz2 = dz**2
        if dz2 <= rho2: # inside of the source
            if dx == -incr and count_x == 0.0: # update xmax value if negative run
                xmax = xmax.at[yi].set(z_current.real - dx)
            Ar = source_profile_limb1(dz) # brightness with limb-darkening
            count_x   += Ar
        else: # outside of the source
            if dx == incr: # if dx is positive
                if dz2_last <= rho2: # if previous ray is inside
                    xmax = xmax.at[yi].set(z_current.real) # store the previous ray as xmax
                # parepare negative run
                dx = -incr 
                z_current = jnp.complex128(x0 + z_current.imag)
                xmin = xmin.at[yi].set(z_current.real + dx) # set xmin in positive run
            else: # negative run with outside of the source 
                if dz2_last <= rho2: # if previous ray is inside
                    xmin = xmin.at[yi].set(z_current.real) # set xmin in negative run 
                if z_current.real >= xmin[yi-1] - dx and yi!=0 and count_x==0: # nothing in negative run
                    z_current.real += dx
                    continue
                # collect numbers in the current y coordinate
                count_all += count_x
                area_x = area_x.at[yi].set(count_x)
                y      = y.at[yi].set(z_current.imag)
                dys    = dys.at[yi].set(dy)
                if count_x == 0.0: # This means top in y
                    dys = dys.at[yi].set(-dy)
                    break
                # check if this y is already counted
                y_index = int(z_current.imag * incr_inv + max_iter) #the index based on the current y coordinate (+offset)
                for j in range(Nindx[y_index]):
                    ind = indx[y_index][j]
                    #ind = indx[y_index][j+1]
                    if xmin[yi] + incr < xmax[ind] and xmax[yi] - incr > xmin[ind]: # already counted.
                        return count_all - count_x 
                # save index yi if counted
                indx = indx.at[[y_index][Nindx[y_index]]].set(yi)
                Nindx = Nindx.at[y_index].add(1)
                # move next y-row 
                yi += 1
                dx        = incr               # switch to positive run
                x0        = xmax[yi-1]         # starting x in next negative run.  
                z_current = x0 - dx + 1j * dy  # starting point in next positive run.
                count_x = 0.0
        # update the z value 
        z_current = z_current + dx
    
    return count_all, yi

