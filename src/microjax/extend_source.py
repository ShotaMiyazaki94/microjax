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

def imagearea_binary(w_center, z_inits, q, s, rho, NBIN=20):
    
    nimage  = len(z_inits)
    overlap = jnp.zeros(6) # binary-lens
    incr    = rho / NBIN  
    incr2   = 0.5 * incr   


def imagearea0_binary(w_center, z_init, q, s, rho, dy=1e-4, max_iter=10000):
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
    # used arrays
    indx   = jnp.zeros((max_iter * 2, 4), dtype=int)
    Nindx  = jnp.zeros((max_iter * 2,),   dtype=int)
    xmax   = jnp.zeros((max_iter * 4,))
    xmin   = jnp.zeros((max_iter * 4,))
    area_x = jnp.zeros((max_iter * 4,))
    y      = jnp.zeros((max_iter * 4,))
    dys    = jnp.zeros((max_iter * 4,))
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
            else: # negative run
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
                for j in range(Nindx[y_index]-1):
                    ind = indx[y_index][j+1]
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
    return count_all





def body_fun(carry):
    (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
     yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry

    zis = _lens_eq_binary(z, a=a, e1=e1)
    dz2_last = dz2
    dz = jnp.abs(w - zis)
    dz2 = dz**2

    def in_radius(carry):
        (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
         yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry

        xmax = lax.cond(
            (dx == -incr) & (count_x == 0.0),
            lambda xmax: xmax.at[yi].set(z.real - dx),
            lambda xmax: xmax,
            xmax
        )
        Ar = source_profile_limb1(dz)
        count_x += Ar
        count_all += Ar
        return (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
                yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0)
    
    def out_radius(carry):
        (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
         yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry
        
        def reverse_direction(carry):
            (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
             yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry
            # update the xmax value 
            xmax = lax.cond(
                dz2_last <= rr,
                lambda xmax: xmax.at[yi].set(z.real),
                lambda xmax: xmax,
                xmax
            )
            dx = -incr
            z = jnp.complex128(x0 + 1j * z.imag)
            xmin = xmin.at[yi].set(z.real + dx)
            return (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
                    yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0)
        def settle_row(carry):
            (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, 
             yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry

            xmin = lax.cond(
                dz2_last <= rr,
                lambda xmin: xmin.at[yi].set(z.real),
                lambda xmin: xmin,
                xmin
            )
            skip = (z.real >= xmin[yi-1] - dx) & (yi != 0) & (count_x == 0.0)

            Ax = Ax.at[yi].set(count_x)
            y = y.at[yi].set(z.imag)
            dys = dys.at[yi].set(dy)
            dys = lax.cond(
                count_x == 0.0,
                lambda dys: dys.at[yi].set(-dy),
                lambda dys: dys,
                dys
            )
            yii = jnp.array(z.imag * incr_inv + INDX0)
            is_counted = lax.fori_loop(
                0,
                Nindx[yii],
                lambda j, is_counted: is_counted | ((xmin[yi] + incr < xmax[indx[yii][j]]) & (xmax[yi] - incr > xmin[indx[yii][j]])),
                False
            )

            def return_count(carry):
                (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry
                return (z, dz2, dx, 0.0, count_all - count_x, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0), False

            def save_index(carry):
                (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry

                indx = indx.at[yii, Nindx[yii]].set(yi)
                Nindx = Nindx.at[yii].add(1.0)
                yi += 1
                dx = incr
                x0 = xmax[yi-1]
                z = z - dx + 1j * dy
                return (z, dz2, dx, 0.0, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0), True

            return lax.cond(
                is_counted,
                return_count,
                save_index,
                (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0)
            )

        return lax.cond(
            dx == incr,
            reverse_direction,
            settle_row,
            (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0)
        )

    return lax.cond(
        dz2 <= rr,
        in_radius,
        out_radius,
        (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0)
    ), True

def cond_fun(carry):
    (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0) = carry
    return count_x != 0.0

def image_area0_binary(w, z_init, q, s, rho, dy=1e-4, INDX0=2000000):
    indx  = jnp.zeros((INDX0 * 2, 4), dtype=int)
    Nindx = jnp.zeros((INDX0 * 2,), dtype=int)
    xmax  = jnp.zeros((INDX0 * 4,))
    xmin  = jnp.zeros((INDX0 * 4,))
    Ax    = jnp.zeros((INDX0 * 4,))
    y     = jnp.zeros((INDX0 * 4,))
    dys   = jnp.zeros((INDX0 * 4,))

    z    = z_init
    x0   = z.real
    a    = 0.5 * s
    e1   = q / (1.0 + q) 
    dz2  = 99999999.9
    incr      = jnp.abs(dy)
    incr_inv  = 1.0 / incr
    yi        = 0
    dx        = incr 
    count_x   = 0.0
    count_all = 0.0
    rr        = rho * rho

    carry = (z, dz2, dx, count_x, count_all, xmax, xmin, Ax, y, dys, yi, indx, Nindx, rr, incr, incr_inv, w, rho, a, e1, x0, INDX0)
    carry, _ = lax.while_loop(cond_fun, body_fun, carry)
    _, _, _, _, count_all, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = carry

    return count_all
