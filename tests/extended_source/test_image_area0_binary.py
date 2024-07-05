import unittest
import numpy as np
import jax.numpy as jnp
from jax import jit
from microjax.extend_source import image_area0_binary
from microjax.point_source import _images_point_source_binary, mag_point_source_binary
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

def test():
    w_center = jnp.array([0.0 + 0.0j])
    q = 1.0
    s = 1.0
    rho = 5.2e-2
    NBIN = 10
    dy = jnp.abs(rho/NBIN)
    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    w_center -= 0.5*s*(1 - q)/(1 + q) # mid-point
    z_inits, z_mask = _images_point_source_binary(w_center, a, e1) 
    w_center += 0.5*s*(1 - q)/(1 + q) # center of mass
    z_inits  += 0.5*s*(1 - q)/(1 + q) # center of mass
    max_iter = 10000000
    carry =(
        0, 
        jnp.zeros((max_iter * 2, 10), dtype=int),  # indx
        jnp.zeros((max_iter * 2), dtype=int),    # Nindx
        jnp.zeros((max_iter * 2)),               # xmax
        jnp.zeros((max_iter * 2)),               # xmin
        jnp.zeros((max_iter * 2)),               # area_x
        jnp.zeros((max_iter * 2)),               # y
        jnp.zeros((max_iter * 2))                # dys
    )
    count = 0.0
    for i in range(len(z_inits[z_mask])):
    #for i in range(1):
        area, carry = image_area0_binary(w_center, z_inits[z_mask][i], q, s, rho, dy, carry)
        count += area
    print("count_all:", count)
    print("dy       :", dy)
    print("area*dy  :", count * dy)

if __name__ == "__main__":
    test()