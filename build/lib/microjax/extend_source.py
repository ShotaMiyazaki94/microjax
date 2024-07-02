from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.poly_solver import poly_roots_EA_multi as poly_roots
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple 
from microjax.coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary
from microjax.point_source import _lens_eq_binary, _lens_eq_triple
from microjax.point_source import mag_point_source_binary, mag_point_source_triple

INDX0 = 2000000
indx = jnp.zeros((INDX0 * 2, 4), dtype=int)
# indx: A two-dimensional array storing x indices for each y index.
# It can hold up to 4 x indices for each y value, for up to INDX0*2 y indices.
Nindx = jnp.zeros((INDX0 * 2,), dtype=int)
# Nindx: An array that holds the count of active x indices used in the indx array for each y index.
xmax = jnp.zeros((INDX0 * 4,))
# xmax: An array storing the maximum x value for each y index.
xmin = jnp.zeros((INDX0 * 4,))
# xmin: An array storing the minimum x value for each y index.
Ax = jnp.zeros((INDX0 * 4,))
# Ax: An array storing the total area calculated for each y index.
y = jnp.zeros((INDX0 * 4,))
# y: An array storing the actual y values corresponding to each index.
dys = jnp.zeros((INDX0 * 4,))
# dys: An array storing the dy values (increment in the y-direction) for each y index.


def limb_lin(dz, u1=0.0):
    mu = jnp.sqrt(1.0 - dz*dz)
    return 1 - u1 * (1.0 - mu)

def compute_intersection(yi, yii, xmin, xmax, incr, indx, Nindx):
    """
    Check if the current range intersects with any previously computed ranges.
    
    Args:
    yi (int): Current y index being processed.
    yii (int): Computed index based on the current z.imag value.
    xmin (jax.numpy.ndarray): Array of minimum x values for each y index.
    xmax (jax.numpy.ndarray): Array of maximum x values for each y index.
    incr (float): Increment value used for adjustments in x direction.
    indx (jax.numpy.ndarray): Array storing indices of y coordinates.
    Nindx (jax.numpy.ndarray): Array storing the number of indices stored for each y index.
    
    Returns:
    tuple: (bool, int) where bool indicates if there is an intersection and int is the intersecting index.
    """
    for j in range(Nindx[yii]):
        ind = indx[yii][j]
        if xmin[yi] + incr < xmax[ind] and xmax[yi] - incr > xmin[ind]:
            return True, ind
    return False, None


def image_area0_binary(w, z_init, q, s, rho, dy=1e-4):
    #incr=rho/NBIN
    z = z_init
    x0 = z.real # starting point in x
    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    dz2 = 99999999.9
    incr  = jnp.abs(dy) # positive
    Oincr = 1.0 / incr
    yi    = 0
    dx    = incr 
    count_x = 0.0
    count_all = 0.0
    rr = rho * rho
    Orr   = 1.0 / rr
    while True:
        zis = _lens_eq_binary(z, a=a, e1=e1) # inversed point from image into source
        dz2_last = dz2
        dz  = jnp.abs(w - zis)
        dz2 = dz**2
        # point is within radius
        if dz2 <= rho:
            if dx == -incr and count_x == 0.0:
                xmax = xmax.at[yi].set(z.real - dx)
            Ar = limb_lin(dz)
            count_x += Ar 
        # outside of radius
        else:
            # reverse x direction if x-direction is positive
            if dx == incr:
                if dz2_last <= rr:
                    xmax = xmax.at[yi].set(z.real)
                dx = -incr
                z = jnp.complex128(x0 + 1j * z.imag)
                xmin = xmin.at[yi].set(z.real + dx)
            # settle the x-row if x-direction is negative
            else:
                if dz2_last <= rr:
                    xmin = xmin.at[yi].set(z.real)
                if z.real >= xmin[yi-1] - dx and yi != 0 and count_x == 0.0:
                    z = jnp.complex128(z.real + dx + 1j * z.imag)
                    continue
                count_all += count_x
                Ax  = Ax.at[yi].set(count_x)
                y   = y.at[yi].set(z.imag)
                dys = dys.at[yi].set(dy)
                if count_x == 0.0:
                    dys = dys.at[yi].set(-dy)
                    break
                # Additional index checking and setting logic needed here
                yi += 1
                dx = incr
                x0 = xmax[yi-1]
                z = jnp.complex128(x0 - dx + 1j * (z.imag + dy))
                count_x = 0.0
        z = jnp.complex128(z.real + dx + 1j * z.imag)
    return count_all


def compute_intersection(yi, yii, xmin, xmax, incr, indx, Nindx):
    """
    Check if the current range intersects with any previously computed ranges.
    
    Args:
        yi (int): Current y index being processed.
        yii (int): Computed index based on the current z.imag value.
        xmin (jax.numpy.ndarray): Array of minimum x values for each y index.
        xmax (jax.numpy.ndarray): Array of maximum x values for each y index.
        incr (float): Increment value used for adjustments in x direction.
        indx (jax.numpy.ndarray): Array storing indices of y coordinates.
        Nindx (jax.numpy.ndarray): Array storing the number of indices stored for each y index.
    
    Returns:
    tuple: (bool, int) where bool indicates if there is an intersection and int is the intersecting index.
    """
    for j in range(Nindx[yii]):
        ind = indx[yii][j]
        if xmin[yi] + incr < xmax[ind] and xmax[yi] - incr > xmin[ind]:
            return True, ind
    return False, None 

def update_indices(yi, yii, indx, Nindx):
    """
    Update indices after confirming no intersection.
    
    Args:
    yi (int): Current y index being processed.
    yii (int): Computed index based on the current z.imag value.
    indx (jax.numpy.ndarray): Array storing indices of y coordinates.
    Nindx (jax.numpy.ndarray): Array storing the number of indices stored for each y index.
    
    Returns:
    tuple: Updated (indx, Nindx) arrays.
    """
    indx = indx.at[yii, Nindx[yii]].set(yi)
    Nindx = Nindx.at[yii].set(Nindx[yii] + 1)
    return indx, Nindx