from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.poly_solver import poly_roots_EA_multi as poly_roots
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple, 
from microjax.coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary

