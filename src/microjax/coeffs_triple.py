import jax.numpy as jnp
from jax import jit

def _poly_coeffs_triple(w, a, r3, e1, e2):
    p_0 = _poly_coeffs_triple_p0(w, a, r3, e1, e2)
    p_1 = _poly_coeffs_triple_p1(w, a, r3, e1, e2)
    p_2 = _poly_coeffs_triple_p2(w, a, r3, e1, e2)
    p_3 = _poly_coeffs_triple_p3(w, a, r3, e1, e2)
    p_4 = _poly_coeffs_triple_p4(w, a, r3, e1, e2)
    p_5 = _poly_coeffs_triple_p5(w, a, r3, e1, e2)
    p_6 = _poly_coeffs_triple_p6(w, a, r3, e1, e2)
    p_7 = _poly_coeffs_triple_p7(w, a, r3, e1, e2)
    p_8 = _poly_coeffs_triple_p8(w, a, r3, e1, e2)
    p_9 = _poly_coeffs_triple_p9(w, a, r3, e1, e2)
    p_10 = _poly_coeffs_triple_p10(w, a, r3, e1, e2)

    p = jnp.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10])
    return jnp.moveaxis(p, 0, -1)

def _poly_coeffs_critical_triple(phi, a, r3, e1, e2):
    x = jnp.exp(-1j * phi)

    p_0 = x
    p_1 = -2 * x * r3
    p_2 = -2 * a**2 * x - 1 + x * r3**2
    p_3 = 4 * a**2 * x * r3 - 2 * a * e1 + 2 * a * e2 + 2 * e1 * r3 + 2 * e2 * r3
    p_4 = (
        a**4 * x
        - 3 * a**2 * e1
        - 3 * a**2 * e2
        + 2 * a**2
        - 2 * a**2 * x * r3**2
        + 4 * a * e1 * r3
        - 4 * a * e2 * r3
        - e1 * r3**2
        - e2 * r3**2
    )
    p_5 = (
        -2 * a**4 * x * r3
        + 2 * a**2 * e1 * r3
        + 2 * a**2 * e2 * r3
        - 2 * a * e1 * r3**2
        + 2 * a * e2 * r3**2
    )
    p_6 = (
        a**4 * e1
        + a**4 * e2
        - a**4
        + a**4 * x * r3**2
        - a**2 * e1 * r3**2
        - a**2 * e2 * r3**2
    )

    p = jnp.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6])

    return p


def _poly_coeffs_triple_p0(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_0 = -(a**2) * wbar + a**2 * r3bar + wbar**3 - wbar**2 * r3bar
    return p_0

def _poly_coeffs_triple_p1(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_1 = (
        a**2 * w * wbar
        - a**2 * w * r3bar
        + 3 * a**2 * wbar * r3
        - 3 * a**2 * r3bar * r3
        - a**2 * e1
        - a**2 * e2
        - a * wbar * e1
        + a * wbar * e2
        + a * r3bar * e1
        - a * r3bar * e2
        - w * wbar**3
        + w * wbar**2 * r3bar
        - 3 * wbar**3 * r3
        + 3 * wbar**2 * r3bar * r3
        + 2 * wbar**2
        + wbar * r3bar * e1
        + wbar * r3bar * e2
        - 2 * wbar * r3bar
    )
    return p_1

def _poly_coeffs_triple_p2(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_2 = (
        3 * a**4 * wbar
        - 3 * a**4 * r3bar
        - a**3 * e1
        + a**3 * e2
        - 3 * a**2 * w * wbar * r3
        + 3 * a**2 * w * r3bar * r3
        + a**2 * w
        - 3 * a**2 * wbar**3
        + 3 * a**2 * wbar**2 * r3bar
        - 3 * a**2 * wbar * r3**2
        + 3 * a**2 * r3bar * r3**2
        + 4 * a**2 * e1 * r3
        + 4 * a**2 * e2 * r3
        - a**2 * r3
        + 3 * a * wbar**2 * e1
        - 3 * a * wbar**2 * e2
        - 2 * a * wbar * r3bar * e1
        + 2 * a * wbar * r3bar * e2
        + 3 * a * wbar * e1 * r3
        - 3 * a * wbar * e2 * r3
        - 3 * a * r3bar * e1 * r3
        + 3 * a * r3bar * e2 * r3
        - a * e1
        + a * e2
        + 3 * w * wbar**3 * r3
        - 3 * w * wbar**2 * r3bar * r3
        - 3 * w * wbar**2
        + 2 * w * wbar * r3bar
        + 3 * wbar**3 * r3**2
        - 3 * wbar**2 * r3bar * r3**2
        - 3 * wbar**2 * e1 * r3
        - 3 * wbar**2 * e2 * r3
        - 3 * wbar**2 * r3
        - wbar * r3bar * e1 * r3
        - wbar * r3bar * e2 * r3
        + 4 * wbar * r3bar * r3
        + wbar
        + r3bar * e1
        + r3bar * e2
        - r3bar
    )
    return p_2

def _poly_coeffs_triple_p3(w, a, r3, e1, e2):
   wbar = jnp.conjugate(w)
   r3bar = jnp.conjugate(r3) 
   p_3 = (
        -3 * a**4 * w * wbar
        + 3 * a**4 * w * r3bar
        - 9 * a**4 * wbar * r3
        + 9 * a**4 * r3bar * r3
        + 2 * a**4 * e1
        + 2 * a**4 * e2
        + a**3 * w * e1
        - a**3 * w * e2
        + 3 * a**3 * wbar * e1
        - 3 * a**3 * wbar * e2
        - 3 * a**3 * r3bar * e1
        + 3 * a**3 * r3bar * e2
        + 3 * a**3 * e1 * r3
        - 3 * a**3 * e2 * r3
        + 3 * a**2 * w * wbar**3
        - 3 * a**2 * w * wbar**2 * r3bar
        + 3 * a**2 * w * wbar * r3**2
        - 3 * a**2 * w * r3bar * r3**2
        - a**2 * w * e1 * r3
        - a**2 * w * e2 * r3
        - 2 * a**2 * w * r3
        + 9 * a**2 * wbar**3 * r3
        - 9 * a**2 * wbar**2 * r3bar * r3
        + 3 * a**2 * wbar**2 * e1
        + 3 * a**2 * wbar**2 * e2
        - 6 * a**2 * wbar**2
        - 5 * a**2 * wbar * r3bar * e1
        - 5 * a**2 * wbar * r3bar * e2
        + 6 * a**2 * wbar * r3bar
        + a**2 * wbar * r3**3
        - a**2 * r3bar * r3**3
        - a**2 * e1**2
        + 2 * a**2 * e1 * e2
        - 5 * a**2 * e1 * r3**2
        - a**2 * e2**2
        - 5 * a**2 * e2 * r3**2
        + 2 * a**2 * r3**2
        - 3 * a * w * wbar**2 * e1
        + 3 * a * w * wbar**2 * e2
        + 2 * a * w * wbar * r3bar * e1
        - 2 * a * w * wbar * r3bar * e2
        - 9 * a * wbar**2 * e1 * r3
        + 9 * a * wbar**2 * e2 * r3
        + 6 * a * wbar * r3bar * e1 * r3
        - 6 * a * wbar * r3bar * e2 * r3
        - 3 * a * wbar * e1 * r3**2
        + 4 * a * wbar * e1
        + 3 * a * wbar * e2 * r3**2
        - 4 * a * wbar * e2
        + a * r3bar * e1**2
        + 3 * a * r3bar * e1 * r3**2
        - 2 * a * r3bar * e1
        - a * r3bar * e2**2
        - 3 * a * r3bar * e2 * r3**2
        + 2 * a * r3bar * e2
        + a * e1**2 * r3
        + 2 * a * e1 * r3
        - a * e2**2 * r3
        - 2 * a * e2 * r3
        - 3 * w * wbar**3 * r3**2
        + 3 * w * wbar**2 * r3bar * r3**2
        + 3 * w * wbar**2 * e1 * r3
        + 3 * w * wbar**2 * e2 * r3
        + 6 * w * wbar**2 * r3
        - 2 * w * wbar * r3bar * e1 * r3
        - 2 * w * wbar * r3bar * e2 * r3
        - 4 * w * wbar * r3bar * r3
        - 3 * w * wbar
        + w * r3bar
        - wbar**3 * r3**3
        + wbar**2 * r3bar * r3**3
        + 6 * wbar**2 * e1 * r3**2
        + 6 * wbar**2 * e2 * r3**2
        - wbar * r3bar * e1 * r3**2
        - wbar * r3bar * e2 * r3**2
        - 2 * wbar * r3bar * r3**2
        - 4 * wbar * e1 * r3
        - 4 * wbar * e2 * r3
        + wbar * r3
        - r3bar * e1**2 * r3
        - 2 * r3bar * e1 * e2 * r3
        - r3bar * e2**2 * r3
        + r3bar * r3
    )
   return p_3

def _poly_coeffs_triple_p4(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3) 
    p_4 = (
        -3 * a**6 * wbar
        + 3 * a**6 * r3bar
        + 2 * a**5 * e1
        - 2 * a**5 * e2
        + 9 * a**4 * w * wbar * r3
        - 9 * a**4 * w * r3bar * r3
        + a**4 * w * e1
        + a**4 * w * e2
        - 3 * a**4 * w
        + 3 * a**4 * wbar**3
        - 3 * a**4 * wbar**2 * r3bar
        + 9 * a**4 * wbar * r3**2
        - 9 * a**4 * r3bar * r3**2
        - 9 * a**4 * e1 * r3
        - 9 * a**4 * e2 * r3
        + 3 * a**4 * r3
        - 3 * a**3 * w * e1 * r3
        + 3 * a**3 * w * e2 * r3
        - 6 * a**3 * wbar**2 * e1
        + 6 * a**3 * wbar**2 * e2
        + 4 * a**3 * wbar * r3bar * e1
        - 4 * a**3 * wbar * r3bar * e2
        - 9 * a**3 * wbar * e1 * r3
        + 9 * a**3 * wbar * e2 * r3
        + 9 * a**3 * r3bar * e1 * r3
        - 9 * a**3 * r3bar * e2 * r3
        - a**3 * e1**2
        - 3 * a**3 * e1 * r3**2
        + 3 * a**3 * e1
        + a**3 * e2**2
        + 3 * a**3 * e2 * r3**2
        - 3 * a**3 * e2
        - 9 * a**2 * w * wbar**3 * r3
        + 9 * a**2 * w * wbar**2 * r3bar * r3
        - 3 * a**2 * w * wbar**2 * e1
        - 3 * a**2 * w * wbar**2 * e2
        + 9 * a**2 * w * wbar**2
        + 2 * a**2 * w * wbar * r3bar * e1
        + 2 * a**2 * w * wbar * r3bar * e2
        - 6 * a**2 * w * wbar * r3bar
        - a**2 * w * wbar * r3**3
        + a**2 * w * r3bar * r3**3
        + 2 * a**2 * w * e1 * r3**2
        + 2 * a**2 * w * e2 * r3**2
        + a**2 * w * r3**2
        - 9 * a**2 * wbar**3 * r3**2
        + 9 * a**2 * wbar**2 * r3bar * r3**2
        + 9 * a**2 * wbar**2 * r3
        + 9 * a**2 * wbar * r3bar * e1 * r3
        + 9 * a**2 * wbar * r3bar * e2 * r3
        - 12 * a**2 * wbar * r3bar * r3
        + 3 * a**2 * wbar * e1**2
        - 6 * a**2 * wbar * e1 * e2
        + 4 * a**2 * wbar * e1
        + 3 * a**2 * wbar * e2**2
        + 4 * a**2 * wbar * e2
        - 3 * a**2 * wbar
        + 4 * a**2 * r3bar * e1 * e2
        - 5 * a**2 * r3bar * e1
        - 5 * a**2 * r3bar * e2
        + 3 * a**2 * r3bar
        + 3 * a**2 * e1**2 * r3
        - 6 * a**2 * e1 * e2 * r3
        + 2 * a**2 * e1 * r3**3
        + 3 * a**2 * e2**2 * r3
        + 2 * a**2 * e2 * r3**3
        - a**2 * r3**3
        + 9 * a * w * wbar**2 * e1 * r3
        - 9 * a * w * wbar**2 * e2 * r3
        - 6 * a * w * wbar * r3bar * e1 * r3
        + 6 * a * w * wbar * r3bar * e2 * r3
        - 6 * a * w * wbar * e1
        + 6 * a * w * wbar * e2
        + 2 * a * w * r3bar * e1
        - 2 * a * w * r3bar * e2
        + 9 * a * wbar**2 * e1 * r3**2
        - 9 * a * wbar**2 * e2 * r3**2
        - 6 * a * wbar * r3bar * e1 * r3**2
        + 6 * a * wbar * r3bar * e2 * r3**2
        - 6 * a * wbar * e1**2 * r3
        + a * wbar * e1 * r3**3
        - 6 * a * wbar * e1 * r3
        + 6 * a * wbar * e2**2 * r3
        - a * wbar * e2 * r3**3
        + 6 * a * wbar * e2 * r3
        - a * r3bar * e1**2 * r3
        - a * r3bar * e1 * r3**3
        + 4 * a * r3bar * e1 * r3
        + a * r3bar * e2**2 * r3
        + a * r3bar * e2 * r3**3
        - 4 * a * r3bar * e2 * r3
        - 2 * a * e1**2 * r3**2
        - a * e1 * r3**2
        + a * e1
        + 2 * a * e2**2 * r3**2
        + a * e2 * r3**2
        - a * e2
        + w * wbar**3 * r3**3
        - w * wbar**2 * r3bar * r3**3
        - 6 * w * wbar**2 * e1 * r3**2
        - 6 * w * wbar**2 * e2 * r3**2
        - 3 * w * wbar**2 * r3**2
        + 4 * w * wbar * r3bar * e1 * r3**2
        + 4 * w * wbar * r3bar * e2 * r3**2
        + 2 * w * wbar * r3bar * r3**2
        + 6 * w * wbar * e1 * r3
        + 6 * w * wbar * e2 * r3
        + 3 * w * wbar * r3
        - 2 * w * r3bar * e1 * r3
        - 2 * w * r3bar * e2 * r3
        - w * r3bar * r3
        - w
        - 3 * wbar**2 * e1 * r3**3
        - 3 * wbar**2 * e2 * r3**3
        + wbar**2 * r3**3
        + wbar * r3bar * e1 * r3**3
        + wbar * r3bar * e2 * r3**3
        + 3 * wbar * e1**2 * r3**2
        + 6 * wbar * e1 * e2 * r3**2
        + 2 * wbar * e1 * r3**2
        + 3 * wbar * e2**2 * r3**2
        + 2 * wbar * e2 * r3**2
        - 2 * wbar * r3**2
        + r3bar * e1**2 * r3**2
        + 2 * r3bar * e1 * e2 * r3**2
        - r3bar * e1 * r3**2
        + r3bar * e2**2 * r3**2
        - r3bar * e2 * r3**2
        - e1 * r3
        - e2 * r3
        + r3
    )
    return p_4

def _poly_coeffs_triple_p5(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_5 = (
        3 * a**6 * w * wbar
        - 3 * a**6 * w * r3bar
        + 9 * a**6 * wbar * r3
        - 9 * a**6 * r3bar * r3
        - a**6 * e1
        - a**6 * e2
        - 2 * a**5 * w * e1
        + 2 * a**5 * w * e2
        - 3 * a**5 * wbar * e1
        + 3 * a**5 * wbar * e2
        + 3 * a**5 * r3bar * e1
        - 3 * a**5 * r3bar * e2
        - 6 * a**5 * e1 * r3
        + 6 * a**5 * e2 * r3
        - 3 * a**4 * w * wbar**3
        + 3 * a**4 * w * wbar**2 * r3bar
        - 9 * a**4 * w * wbar * r3**2
        + 9 * a**4 * w * r3bar * r3**2
        + 6 * a**4 * w * r3
        - 9 * a**4 * wbar**3 * r3
        + 9 * a**4 * wbar**2 * r3bar * r3
        - 6 * a**4 * wbar**2 * e1
        - 6 * a**4 * wbar**2 * e2
        + 6 * a**4 * wbar**2
        + 7 * a**4 * wbar * r3bar * e1
        + 7 * a**4 * wbar * r3bar * e2
        - 6 * a**4 * wbar * r3bar
        - 3 * a**4 * wbar * r3**3
        + 3 * a**4 * r3bar * r3**3
        + 2 * a**4 * e1**2
        - 4 * a**4 * e1 * e2
        + 12 * a**4 * e1 * r3**2
        + 2 * a**4 * e2**2
        + 12 * a**4 * e2 * r3**2
        - 6 * a**4 * r3**2
        + 6 * a**3 * w * wbar**2 * e1
        - 6 * a**3 * w * wbar**2 * e2
        - 4 * a**3 * w * wbar * r3bar * e1
        + 4 * a**3 * w * wbar * r3bar * e2
        + 3 * a**3 * w * e1 * r3**2
        - 3 * a**3 * w * e2 * r3**2
        + 18 * a**3 * wbar**2 * e1 * r3
        - 18 * a**3 * wbar**2 * e2 * r3
        - 12 * a**3 * wbar * r3bar * e1 * r3
        + 12 * a**3 * wbar * r3bar * e2 * r3
        + 6 * a**3 * wbar * e1**2
        + 9 * a**3 * wbar * e1 * r3**2
        - 8 * a**3 * wbar * e1
        - 6 * a**3 * wbar * e2**2
        - 9 * a**3 * wbar * e2 * r3**2
        + 8 * a**3 * wbar * e2
        - 4 * a**3 * r3bar * e1**2
        - 9 * a**3 * r3bar * e1 * r3**2
        + 4 * a**3 * r3bar * e1
        + 4 * a**3 * r3bar * e2**2
        + 9 * a**3 * r3bar * e2 * r3**2
        - 4 * a**3 * r3bar * e2
        + a**3 * e1 * r3**3
        - 6 * a**3 * e1 * r3
        - a**3 * e2 * r3**3
        + 6 * a**3 * e2 * r3
        + 9 * a**2 * w * wbar**3 * r3**2
        - 9 * a**2 * w * wbar**2 * r3bar * r3**2
        - 18 * a**2 * w * wbar**2 * r3
        + 12 * a**2 * w * wbar * r3bar * r3
        - 3 * a**2 * w * wbar * e1**2
        + 6 * a**2 * w * wbar * e1 * e2
        - 6 * a**2 * w * wbar * e1
        - 3 * a**2 * w * wbar * e2**2
        - 6 * a**2 * w * wbar * e2
        + 9 * a**2 * w * wbar
        + a**2 * w * r3bar * e1**2
        - 2 * a**2 * w * r3bar * e1 * e2
        + 2 * a**2 * w * r3bar * e1
        + a**2 * w * r3bar * e2**2
        + 2 * a**2 * w * r3bar * e2
        - 3 * a**2 * w * r3bar
        - a**2 * w * e1 * r3**3
        - a**2 * w * e2 * r3**3
        + 3 * a**2 * wbar**3 * r3**3
        - 3 * a**2 * wbar**2 * r3bar * r3**3
        - 9 * a**2 * wbar**2 * e1 * r3**2
        - 9 * a**2 * wbar**2 * e2 * r3**2
        - 3 * a**2 * wbar * r3bar * e1 * r3**2
        - 3 * a**2 * wbar * r3bar * e2 * r3**2
        + 6 * a**2 * wbar * r3bar * r3**2
        - 15 * a**2 * wbar * e1**2 * r3
        + 6 * a**2 * wbar * e1 * e2 * r3
        + 6 * a**2 * wbar * e1 * r3
        - 15 * a**2 * wbar * e2**2 * r3
        + 6 * a**2 * wbar * e2 * r3
        - 3 * a**2 * wbar * r3
        + 5 * a**2 * r3bar * e1**2 * r3
        - 2 * a**2 * r3bar * e1 * e2 * r3
        + 4 * a**2 * r3bar * e1 * r3
        + 5 * a**2 * r3bar * e2**2 * r3
        + 4 * a**2 * r3bar * e2 * r3
        - 3 * a**2 * r3bar * r3
        - 3 * a**2 * e1**2 * r3**2
        + 2 * a**2 * e1**2
        + 6 * a**2 * e1 * e2 * r3**2
        - 4 * a**2 * e1 * e2
        + a**2 * e1
        - 3 * a**2 * e2**2 * r3**2
        + 2 * a**2 * e2**2
        + a**2 * e2
        - 9 * a * w * wbar**2 * e1 * r3**2
        + 9 * a * w * wbar**2 * e2 * r3**2
        + 6 * a * w * wbar * r3bar * e1 * r3**2
        - 6 * a * w * wbar * r3bar * e2 * r3**2
        + 6 * a * w * wbar * e1**2 * r3
        + 12 * a * w * wbar * e1 * r3
        - 6 * a * w * wbar * e2**2 * r3
        - 12 * a * w * wbar * e2 * r3
        - 2 * a * w * r3bar * e1**2 * r3
        - 4 * a * w * r3bar * e1 * r3
        + 2 * a * w * r3bar * e2**2 * r3
        + 4 * a * w * r3bar * e2 * r3
        - 3 * a * w * e1
        + 3 * a * w * e2
        - 3 * a * wbar**2 * e1 * r3**3
        + 3 * a * wbar**2 * e2 * r3**3
        + 2 * a * wbar * r3bar * e1 * r3**3
        - 2 * a * wbar * r3bar * e2 * r3**3
        + 12 * a * wbar * e1**2 * r3**2
        - 12 * a * wbar * e2**2 * r3**2
        - a * r3bar * e1**2 * r3**2
        - 2 * a * r3bar * e1 * r3**2
        + a * r3bar * e2**2 * r3**2
        + 2 * a * r3bar * e2 * r3**2
        + a * e1**2 * r3**3
        - 4 * a * e1**2 * r3
        + a * e1 * r3
        - a * e2**2 * r3**3
        + 4 * a * e2**2 * r3
        - a * e2 * r3
        + 3 * w * wbar**2 * e1 * r3**3
        + 3 * w * wbar**2 * e2 * r3**3
        - 2 * w * wbar * r3bar * e1 * r3**3
        - 2 * w * wbar * r3bar * e2 * r3**3
        - 3 * w * wbar * e1**2 * r3**2
        - 6 * w * wbar * e1 * e2 * r3**2
        - 6 * w * wbar * e1 * r3**2
        - 3 * w * wbar * e2**2 * r3**2
        - 6 * w * wbar * e2 * r3**2
        + w * r3bar * e1**2 * r3**2
        + 2 * w * r3bar * e1 * e2 * r3**2
        + 2 * w * r3bar * e1 * r3**2
        + w * r3bar * e2**2 * r3**2
        + 2 * w * r3bar * e2 * r3**2
        + 3 * w * e1 * r3
        + 3 * w * e2 * r3
        - 3 * wbar * e1**2 * r3**3
        - 6 * wbar * e1 * e2 * r3**3
        + 2 * wbar * e1 * r3**3
        - 3 * wbar * e2**2 * r3**3
        + 2 * wbar * e2 * r3**3
        + 2 * e1**2 * r3**2
        + 4 * e1 * e2 * r3**2
        - 2 * e1 * r3**2
        + 2 * e2**2 * r3**2
        - 2 * e2 * r3**2
    )
    return p_5 
    

def _poly_coeffs_triple_p6(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_6 = (
        a**8 * wbar
        - a**8 * r3bar
        - a**7 * e1
        + a**7 * e2
        - 9 * a**6 * w * wbar * r3
        + 9 * a**6 * w * r3bar * r3
        - 2 * a**6 * w * e1
        - 2 * a**6 * w * e2
        + 3 * a**6 * w
        - a**6 * wbar**3
        + a**6 * wbar**2 * r3bar
        - 9 * a**6 * wbar * r3**2
        + 9 * a**6 * r3bar * r3**2
        + 6 * a**6 * e1 * r3
        + 6 * a**6 * e2 * r3
        - 3 * a**6 * r3
        + 6 * a**5 * w * e1 * r3
        - 6 * a**5 * w * e2 * r3
        + 3 * a**5 * wbar**2 * e1
        - 3 * a**5 * wbar**2 * e2
        - 2 * a**5 * wbar * r3bar * e1
        + 2 * a**5 * wbar * r3bar * e2
        + 9 * a**5 * wbar * e1 * r3
        - 9 * a**5 * wbar * e2 * r3
        - 9 * a**5 * r3bar * e1 * r3
        + 9 * a**5 * r3bar * e2 * r3
        + 2 * a**5 * e1**2
        + 6 * a**5 * e1 * r3**2
        - 3 * a**5 * e1
        - 2 * a**5 * e2**2
        - 6 * a**5 * e2 * r3**2
        + 3 * a**5 * e2
        + 9 * a**4 * w * wbar**3 * r3
        - 9 * a**4 * w * wbar**2 * r3bar * r3
        + 6 * a**4 * w * wbar**2 * e1
        + 6 * a**4 * w * wbar**2 * e2
        - 9 * a**4 * w * wbar**2
        - 4 * a**4 * w * wbar * r3bar * e1
        - 4 * a**4 * w * wbar * r3bar * e2
        + 6 * a**4 * w * wbar * r3bar
        + 3 * a**4 * w * wbar * r3**3
        - 3 * a**4 * w * r3bar * r3**3
        - 3 * a**4 * w * e1 * r3**2
        - 3 * a**4 * w * e2 * r3**2
        - 3 * a**4 * w * r3**2
        + 9 * a**4 * wbar**3 * r3**2
        - 9 * a**4 * wbar**2 * r3bar * r3**2
        + 9 * a**4 * wbar**2 * e1 * r3
        + 9 * a**4 * wbar**2 * e2 * r3
        - 9 * a**4 * wbar**2 * r3
        - 15 * a**4 * wbar * r3bar * e1 * r3
        - 15 * a**4 * wbar * r3bar * e2 * r3
        + 12 * a**4 * wbar * r3bar * r3
        + 12 * a**4 * wbar * e1 * e2
        - 8 * a**4 * wbar * e1
        - 8 * a**4 * wbar * e2
        + 3 * a**4 * wbar
        - 2 * a**4 * r3bar * e1**2
        - 8 * a**4 * r3bar * e1 * e2
        + 7 * a**4 * r3bar * e1
        - 2 * a**4 * r3bar * e2**2
        + 7 * a**4 * r3bar * e2
        - 3 * a**4 * r3bar
        - 6 * a**4 * e1**2 * r3
        + 12 * a**4 * e1 * e2 * r3
        - 5 * a**4 * e1 * r3**3
        - 6 * a**4 * e2**2 * r3
        - 5 * a**4 * e2 * r3**3
        + 3 * a**4 * r3**3
        - 18 * a**3 * w * wbar**2 * e1 * r3
        + 18 * a**3 * w * wbar**2 * e2 * r3
        + 12 * a**3 * w * wbar * r3bar * e1 * r3
        - 12 * a**3 * w * wbar * r3bar * e2 * r3
        - 6 * a**3 * w * wbar * e1**2
        + 12 * a**3 * w * wbar * e1
        + 6 * a**3 * w * wbar * e2**2
        - 12 * a**3 * w * wbar * e2
        + 2 * a**3 * w * r3bar * e1**2
        - 4 * a**3 * w * r3bar * e1
        - 2 * a**3 * w * r3bar * e2**2
        + 4 * a**3 * w * r3bar * e2
        - a**3 * w * e1 * r3**3
        + a**3 * w * e2 * r3**3
        - 18 * a**3 * wbar**2 * e1 * r3**2
        + 18 * a**3 * wbar**2 * e2 * r3**2
        + 12 * a**3 * wbar * r3bar * e1 * r3**2
        - 12 * a**3 * wbar * r3bar * e2 * r3**2
        - 6 * a**3 * wbar * e1**2 * r3
        - 3 * a**3 * wbar * e1 * r3**3
        + 12 * a**3 * wbar * e1 * r3
        + 6 * a**3 * wbar * e2**2 * r3
        + 3 * a**3 * wbar * e2 * r3**3
        - 12 * a**3 * wbar * e2 * r3
        + 8 * a**3 * r3bar * e1**2 * r3
        + 3 * a**3 * r3bar * e1 * r3**3
        - 8 * a**3 * r3bar * e1 * r3
        - 8 * a**3 * r3bar * e2**2 * r3
        - 3 * a**3 * r3bar * e2 * r3**3
        + 8 * a**3 * r3bar * e2 * r3
        + a**3 * e1**3
        - 3 * a**3 * e1**2 * e2
        + 3 * a**3 * e1**2 * r3**2
        + 4 * a**3 * e1**2
        + 3 * a**3 * e1 * e2**2
        + 3 * a**3 * e1 * r3**2
        - 2 * a**3 * e1
        - a**3 * e2**3
        - 3 * a**3 * e2**2 * r3**2
        - 4 * a**3 * e2**2
        - 3 * a**3 * e2 * r3**2
        + 2 * a**3 * e2
        - 3 * a**2 * w * wbar**3 * r3**3
        + 3 * a**2 * w * wbar**2 * r3bar * r3**3
        + 9 * a**2 * w * wbar**2 * e1 * r3**2
        + 9 * a**2 * w * wbar**2 * e2 * r3**2
        + 9 * a**2 * w * wbar**2 * r3**2
        - 6 * a**2 * w * wbar * r3bar * e1 * r3**2
        - 6 * a**2 * w * wbar * r3bar * e2 * r3**2
        - 6 * a**2 * w * wbar * r3bar * r3**2
        + 15 * a**2 * w * wbar * e1**2 * r3
        - 6 * a**2 * w * wbar * e1 * e2 * r3
        - 6 * a**2 * w * wbar * e1 * r3
        + 15 * a**2 * w * wbar * e2**2 * r3
        - 6 * a**2 * w * wbar * e2 * r3
        - 9 * a**2 * w * wbar * r3
        - 5 * a**2 * w * r3bar * e1**2 * r3
        + 2 * a**2 * w * r3bar * e1 * e2 * r3
        + 2 * a**2 * w * r3bar * e1 * r3
        - 5 * a**2 * w * r3bar * e2**2 * r3
        + 2 * a**2 * w * r3bar * e2 * r3
        + 3 * a**2 * w * r3bar * r3
        - 3 * a**2 * w * e1**2
        + 6 * a**2 * w * e1 * e2
        - 3 * a**2 * w * e1
        - 3 * a**2 * w * e2**2
        - 3 * a**2 * w * e2
        + 3 * a**2 * w
        + 6 * a**2 * wbar**2 * e1 * r3**3
        + 6 * a**2 * wbar**2 * e2 * r3**3
        - 3 * a**2 * wbar**2 * r3**3
        - a**2 * wbar * r3bar * e1 * r3**3
        - a**2 * wbar * r3bar * e2 * r3**3
        + 12 * a**2 * wbar * e1**2 * r3**2
        - 12 * a**2 * wbar * e1 * e2 * r3**2
        - 6 * a**2 * wbar * e1 * r3**2
        + 12 * a**2 * wbar * e2**2 * r3**2
        - 6 * a**2 * wbar * e2 * r3**2
        + 6 * a**2 * wbar * r3**2
        - 7 * a**2 * r3bar * e1**2 * r3**2
        - 2 * a**2 * r3bar * e1 * e2 * r3**2
        + a**2 * r3bar * e1 * r3**2
        - 7 * a**2 * r3bar * e2**2 * r3**2
        + a**2 * r3bar * e2 * r3**2
        - 3 * a**2 * e1**3 * r3
        + 3 * a**2 * e1**2 * e2 * r3
        + a**2 * e1**2 * r3**3
        - 7 * a**2 * e1**2 * r3
        + 3 * a**2 * e1 * e2**2 * r3
        - 2 * a**2 * e1 * e2 * r3**3
        - 2 * a**2 * e1 * e2 * r3
        + 4 * a**2 * e1 * r3
        - 3 * a**2 * e2**3 * r3
        + a**2 * e2**2 * r3**3
        - 7 * a**2 * e2**2 * r3
        + 4 * a**2 * e2 * r3
        - 3 * a**2 * r3
        + 3 * a * w * wbar**2 * e1 * r3**3
        - 3 * a * w * wbar**2 * e2 * r3**3
        - 2 * a * w * wbar * r3bar * e1 * r3**3
        + 2 * a * w * wbar * r3bar * e2 * r3**3
        - 12 * a * w * wbar * e1**2 * r3**2
        - 6 * a * w * wbar * e1 * r3**2
        + 12 * a * w * wbar * e2**2 * r3**2
        + 6 * a * w * wbar * e2 * r3**2
        + 4 * a * w * r3bar * e1**2 * r3**2
        + 2 * a * w * r3bar * e1 * r3**2
        - 4 * a * w * r3bar * e2**2 * r3**2
        - 2 * a * w * r3bar * e2 * r3**2
        + 6 * a * w * e1**2 * r3
        + 3 * a * w * e1 * r3
        - 6 * a * w * e2**2 * r3
        - 3 * a * w * e2 * r3
        - 6 * a * wbar * e1**2 * r3**3
        + 2 * a * wbar * e1 * r3**3
        + 6 * a * wbar * e2**2 * r3**3
        - 2 * a * wbar * e2 * r3**3
        + a * r3bar * e1**2 * r3**3
        - a * r3bar * e2**2 * r3**3
        + 3 * a * e1**3 * r3**2
        + 3 * a * e1**2 * e2 * r3**2
        + 2 * a * e1**2 * r3**2
        - 3 * a * e1 * e2**2 * r3**2
        - 2 * a * e1 * r3**2
        - 3 * a * e2**3 * r3**2
        - 2 * a * e2**2 * r3**2
        + 2 * a * e2 * r3**2
        + 3 * w * wbar * e1**2 * r3**3
        + 6 * w * wbar * e1 * e2 * r3**3
        + 3 * w * wbar * e2**2 * r3**3
        - w * r3bar * e1**2 * r3**3
        - 2 * w * r3bar * e1 * e2 * r3**3
        - w * r3bar * e2**2 * r3**3
        - 3 * w * e1**2 * r3**2
        - 6 * w * e1 * e2 * r3**2
        - 3 * w * e2**2 * r3**2
        - e1**3 * r3**3
        - 3 * e1**2 * e2 * r3**3
        + e1**2 * r3**3
        - 3 * e1 * e2**2 * r3**3
        + 2 * e1 * e2 * r3**3
        - e2**3 * r3**3
        + e2**2 * r3**3
    )
    return p_6

def _poly_coeffs_triple_p7(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_7 = (
        -(a**8) * w * wbar
        + a**8 * w * r3bar
        - 3 * a**8 * wbar * r3
        + 3 * a**8 * r3bar * r3
        + a**7 * w * e1
        - a**7 * w * e2
        + a**7 * wbar * e1
        - a**7 * wbar * e2
        - a**7 * r3bar * e1
        + a**7 * r3bar * e2
        + 3 * a**7 * e1 * r3
        - 3 * a**7 * e2 * r3
        + a**6 * w * wbar**3
        - a**6 * w * wbar**2 * r3bar
        + 9 * a**6 * w * wbar * r3**2
        - 9 * a**6 * w * r3bar * r3**2
        + 3 * a**6 * w * e1 * r3
        + 3 * a**6 * w * e2 * r3
        - 6 * a**6 * w * r3
        + 3 * a**6 * wbar**3 * r3
        - 3 * a**6 * wbar**2 * r3bar * r3
        + 3 * a**6 * wbar**2 * e1
        + 3 * a**6 * wbar**2 * e2
        - 2 * a**6 * wbar**2
        - 3 * a**6 * wbar * r3bar * e1
        - 3 * a**6 * wbar * r3bar * e2
        + 2 * a**6 * wbar * r3bar
        + 3 * a**6 * wbar * r3**3
        - 3 * a**6 * r3bar * r3**3
        - a**6 * e1**2
        + 2 * a**6 * e1 * e2
        - 9 * a**6 * e1 * r3**2
        - a**6 * e2**2
        - 9 * a**6 * e2 * r3**2
        + 6 * a**6 * r3**2
        - 3 * a**5 * w * wbar**2 * e1
        + 3 * a**5 * w * wbar**2 * e2
        + 2 * a**5 * w * wbar * r3bar * e1
        - 2 * a**5 * w * wbar * r3bar * e2
        - 6 * a**5 * w * e1 * r3**2
        + 6 * a**5 * w * e2 * r3**2
        - 9 * a**5 * wbar**2 * e1 * r3
        + 9 * a**5 * wbar**2 * e2 * r3
        + 6 * a**5 * wbar * r3bar * e1 * r3
        - 6 * a**5 * wbar * r3bar * e2 * r3
        - 6 * a**5 * wbar * e1**2
        - 9 * a**5 * wbar * e1 * r3**2
        + 4 * a**5 * wbar * e1
        + 6 * a**5 * wbar * e2**2
        + 9 * a**5 * wbar * e2 * r3**2
        - 4 * a**5 * wbar * e2
        + 3 * a**5 * r3bar * e1**2
        + 9 * a**5 * r3bar * e1 * r3**2
        - 2 * a**5 * r3bar * e1
        - 3 * a**5 * r3bar * e2**2
        - 9 * a**5 * r3bar * e2 * r3**2
        + 2 * a**5 * r3bar * e2
        - 3 * a**5 * e1**2 * r3
        - 2 * a**5 * e1 * r3**3
        + 6 * a**5 * e1 * r3
        + 3 * a**5 * e2**2 * r3
        + 2 * a**5 * e2 * r3**3
        - 6 * a**5 * e2 * r3
        - 9 * a**4 * w * wbar**3 * r3**2
        + 9 * a**4 * w * wbar**2 * r3bar * r3**2
        - 9 * a**4 * w * wbar**2 * e1 * r3
        - 9 * a**4 * w * wbar**2 * e2 * r3
        + 18 * a**4 * w * wbar**2 * r3
        + 6 * a**4 * w * wbar * r3bar * e1 * r3
        + 6 * a**4 * w * wbar * r3bar * e2 * r3
        - 12 * a**4 * w * wbar * r3bar * r3
        - 12 * a**4 * w * wbar * e1 * e2
        + 12 * a**4 * w * wbar * e1
        + 12 * a**4 * w * wbar * e2
        - 9 * a**4 * w * wbar
        + 4 * a**4 * w * r3bar * e1 * e2
        - 4 * a**4 * w * r3bar * e1
        - 4 * a**4 * w * r3bar * e2
        + 3 * a**4 * w * r3bar
        + 2 * a**4 * w * e1 * r3**3
        + 2 * a**4 * w * e2 * r3**3
        - 3 * a**4 * wbar**3 * r3**3
        + 3 * a**4 * wbar**2 * r3bar * r3**3
        + 9 * a**4 * wbar * r3bar * e1 * r3**2
        + 9 * a**4 * wbar * r3bar * e2 * r3**2
        - 6 * a**4 * wbar * r3bar * r3**2
        + 12 * a**4 * wbar * e1**2 * r3
        - 12 * a**4 * wbar * e1 * e2 * r3
        + 12 * a**4 * wbar * e2**2 * r3
        + 3 * a**4 * wbar * r3
        - a**4 * r3bar * e1**2 * r3
        + 10 * a**4 * r3bar * e1 * e2 * r3
        - 8 * a**4 * r3bar * e1 * r3
        - a**4 * r3bar * e2**2 * r3
        - 8 * a**4 * r3bar * e2 * r3
        + 3 * a**4 * r3bar * r3
        + 3 * a**4 * e1**3
        - 3 * a**4 * e1**2 * e2
        + 6 * a**4 * e1**2 * r3**2
        - 3 * a**4 * e1 * e2**2
        - 12 * a**4 * e1 * e2 * r3**2
        + 8 * a**4 * e1 * e2
        - 2 * a**4 * e1
        + 3 * a**4 * e2**3
        + 6 * a**4 * e2**2 * r3**2
        - 2 * a**4 * e2
        + 18 * a**3 * w * wbar**2 * e1 * r3**2
        - 18 * a**3 * w * wbar**2 * e2 * r3**2
        - 12 * a**3 * w * wbar * r3bar * e1 * r3**2
        + 12 * a**3 * w * wbar * r3bar * e2 * r3**2
        + 6 * a**3 * w * wbar * e1**2 * r3
        - 24 * a**3 * w * wbar * e1 * r3
        - 6 * a**3 * w * wbar * e2**2 * r3
        + 24 * a**3 * w * wbar * e2 * r3
        - 2 * a**3 * w * r3bar * e1**2 * r3
        + 8 * a**3 * w * r3bar * e1 * r3
        + 2 * a**3 * w * r3bar * e2**2 * r3
        - 8 * a**3 * w * r3bar * e2 * r3
        - a**3 * w * e1**3
        + 3 * a**3 * w * e1**2 * e2
        - 6 * a**3 * w * e1**2
        - 3 * a**3 * w * e1 * e2**2
        + 6 * a**3 * w * e1
        + a**3 * w * e2**3
        + 6 * a**3 * w * e2**2
        - 6 * a**3 * w * e2
        + 6 * a**3 * wbar**2 * e1 * r3**3
        - 6 * a**3 * wbar**2 * e2 * r3**3
        - 4 * a**3 * wbar * r3bar * e1 * r3**3
        + 4 * a**3 * wbar * r3bar * e2 * r3**3
        - 6 * a**3 * wbar * e1**2 * r3**2
        + 6 * a**3 * wbar * e2**2 * r3**2
        - 4 * a**3 * r3bar * e1**2 * r3**2
        + 4 * a**3 * r3bar * e1 * r3**2
        + 4 * a**3 * r3bar * e2**2 * r3**2
        - 4 * a**3 * r3bar * e2 * r3**2
        - 9 * a**3 * e1**3 * r3
        + 3 * a**3 * e1**2 * e2 * r3
        - 2 * a**3 * e1**2 * r3**3
        + 2 * a**3 * e1**2 * r3
        - 3 * a**3 * e1 * e2**2 * r3
        - 2 * a**3 * e1 * r3
        + 9 * a**3 * e2**3 * r3
        + 2 * a**3 * e2**2 * r3**3
        - 2 * a**3 * e2**2 * r3
        + 2 * a**3 * e2 * r3
        - 6 * a**2 * w * wbar**2 * e1 * r3**3
        - 6 * a**2 * w * wbar**2 * e2 * r3**3
        + 4 * a**2 * w * wbar * r3bar * e1 * r3**3
        + 4 * a**2 * w * wbar * r3bar * e2 * r3**3
        - 12 * a**2 * w * wbar * e1**2 * r3**2
        + 12 * a**2 * w * wbar * e1 * e2 * r3**2
        + 12 * a**2 * w * wbar * e1 * r3**2
        - 12 * a**2 * w * wbar * e2**2 * r3**2
        + 12 * a**2 * w * wbar * e2 * r3**2
        + 4 * a**2 * w * r3bar * e1**2 * r3**2
        - 4 * a**2 * w * r3bar * e1 * e2 * r3**2
        - 4 * a**2 * w * r3bar * e1 * r3**2
        + 4 * a**2 * w * r3bar * e2**2 * r3**2
        - 4 * a**2 * w * r3bar * e2 * r3**2
        + 3 * a**2 * w * e1**3 * r3
        - 3 * a**2 * w * e1**2 * e2 * r3
        + 12 * a**2 * w * e1**2 * r3
        - 3 * a**2 * w * e1 * e2**2 * r3
        - 6 * a**2 * w * e1 * r3
        + 3 * a**2 * w * e2**3 * r3
        + 12 * a**2 * w * e2**2 * r3
        - 6 * a**2 * w * e2 * r3
        + 12 * a**2 * wbar * e1 * e2 * r3**3
        - 4 * a**2 * wbar * e1 * r3**3
        - 4 * a**2 * wbar * e2 * r3**3
        + 2 * a**2 * r3bar * e1**2 * r3**3
        + 2 * a**2 * r3bar * e2**2 * r3**3
        + 9 * a**2 * e1**3 * r3**2
        + 3 * a**2 * e1**2 * e2 * r3**2
        - 4 * a**2 * e1**2 * r3**2
        + 3 * a**2 * e1 * e2**2 * r3**2
        - 8 * a**2 * e1 * e2 * r3**2
        + 4 * a**2 * e1 * r3**2
        + 9 * a**2 * e2**3 * r3**2
        - 4 * a**2 * e2**2 * r3**2
        + 4 * a**2 * e2 * r3**2
        + 6 * a * w * wbar * e1**2 * r3**3
        - 6 * a * w * wbar * e2**2 * r3**3
        - 2 * a * w * r3bar * e1**2 * r3**3
        + 2 * a * w * r3bar * e2**2 * r3**3
        - 3 * a * w * e1**3 * r3**2
        - 3 * a * w * e1**2 * e2 * r3**2
        - 6 * a * w * e1**2 * r3**2
        + 3 * a * w * e1 * e2**2 * r3**2
        + 3 * a * w * e2**3 * r3**2
        + 6 * a * w * e2**2 * r3**2
        - 3 * a * e1**3 * r3**3
        - 3 * a * e1**2 * e2 * r3**3
        + 2 * a * e1**2 * r3**3
        + 3 * a * e1 * e2**2 * r3**3
        + 3 * a * e2**3 * r3**3
        - 2 * a * e2**2 * r3**3
        + w * e1**3 * r3**3
        + 3 * w * e1**2 * e2 * r3**3
        + 3 * w * e1 * e2**2 * r3**3
        + w * e2**3 * r3**3
    )
    return p_7

def _poly_coeffs_triple_p8(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_8 = (
        3 * a**8 * w * wbar * r3
        - 3 * a**8 * w * r3bar * r3
        + a**8 * w * e1
        + a**8 * w * e2
        - a**8 * w
        + 3 * a**8 * wbar * r3**2
        - 3 * a**8 * r3bar * r3**2
        - a**8 * e1 * r3
        - a**8 * e2 * r3
        + a**8 * r3
        - 3 * a**7 * w * e1 * r3
        + 3 * a**7 * w * e2 * r3
        - 3 * a**7 * wbar * e1 * r3
        + 3 * a**7 * wbar * e2 * r3
        + 3 * a**7 * r3bar * e1 * r3
        - 3 * a**7 * r3bar * e2 * r3
        - a**7 * e1**2
        - 3 * a**7 * e1 * r3**2
        + a**7 * e1
        + a**7 * e2**2
        + 3 * a**7 * e2 * r3**2
        - a**7 * e2
        - 3 * a**6 * w * wbar**3 * r3
        + 3 * a**6 * w * wbar**2 * r3bar * r3
        - 3 * a**6 * w * wbar**2 * e1
        - 3 * a**6 * w * wbar**2 * e2
        + 3 * a**6 * w * wbar**2
        + 2 * a**6 * w * wbar * r3bar * e1
        + 2 * a**6 * w * wbar * r3bar * e2
        - 2 * a**6 * w * wbar * r3bar
        - 3 * a**6 * w * wbar * r3**3
        + 3 * a**6 * w * r3bar * r3**3
        + 3 * a**6 * w * r3**2
        - 3 * a**6 * wbar**3 * r3**2
        + 3 * a**6 * wbar**2 * r3bar * r3**2
        - 6 * a**6 * wbar**2 * e1 * r3
        - 6 * a**6 * wbar**2 * e2 * r3
        + 3 * a**6 * wbar**2 * r3
        + 7 * a**6 * wbar * r3bar * e1 * r3
        + 7 * a**6 * wbar * r3bar * e2 * r3
        - 4 * a**6 * wbar * r3bar * r3
        - 3 * a**6 * wbar * e1**2
        - 6 * a**6 * wbar * e1 * e2
        + 4 * a**6 * wbar * e1
        - 3 * a**6 * wbar * e2**2
        + 4 * a**6 * wbar * e2
        - a**6 * wbar
        + 2 * a**6 * r3bar * e1**2
        + 4 * a**6 * r3bar * e1 * e2
        - 3 * a**6 * r3bar * e1
        + 2 * a**6 * r3bar * e2**2
        - 3 * a**6 * r3bar * e2
        + a**6 * r3bar
        + 3 * a**6 * e1**2 * r3
        - 6 * a**6 * e1 * e2 * r3
        + 4 * a**6 * e1 * r3**3
        + 3 * a**6 * e2**2 * r3
        + 4 * a**6 * e2 * r3**3
        - 3 * a**6 * r3**3
        + 9 * a**5 * w * wbar**2 * e1 * r3
        - 9 * a**5 * w * wbar**2 * e2 * r3
        - 6 * a**5 * w * wbar * r3bar * e1 * r3
        + 6 * a**5 * w * wbar * r3bar * e2 * r3
        + 6 * a**5 * w * wbar * e1**2
        - 6 * a**5 * w * wbar * e1
        - 6 * a**5 * w * wbar * e2**2
        + 6 * a**5 * w * wbar * e2
        - 2 * a**5 * w * r3bar * e1**2
        + 2 * a**5 * w * r3bar * e1
        + 2 * a**5 * w * r3bar * e2**2
        - 2 * a**5 * w * r3bar * e2
        + 2 * a**5 * w * e1 * r3**3
        - 2 * a**5 * w * e2 * r3**3
        + 9 * a**5 * wbar**2 * e1 * r3**2
        - 9 * a**5 * wbar**2 * e2 * r3**2
        - 6 * a**5 * wbar * r3bar * e1 * r3**2
        + 6 * a**5 * wbar * r3bar * e2 * r3**2
        + 12 * a**5 * wbar * e1**2 * r3
        + 3 * a**5 * wbar * e1 * r3**3
        - 6 * a**5 * wbar * e1 * r3
        - 12 * a**5 * wbar * e2**2 * r3
        - 3 * a**5 * wbar * e2 * r3**3
        + 6 * a**5 * wbar * e2 * r3
        - 7 * a**5 * r3bar * e1**2 * r3
        - 3 * a**5 * r3bar * e1 * r3**3
        + 4 * a**5 * r3bar * e1 * r3
        + 7 * a**5 * r3bar * e2**2 * r3
        + 3 * a**5 * r3bar * e2 * r3**3
        - 4 * a**5 * r3bar * e2 * r3
        + 3 * a**5 * e1**3
        + 3 * a**5 * e1**2 * e2
        - 4 * a**5 * e1**2
        - 3 * a**5 * e1 * e2**2
        - 3 * a**5 * e1 * r3**2
        + a**5 * e1
        - 3 * a**5 * e2**3
        + 4 * a**5 * e2**2
        + 3 * a**5 * e2 * r3**2
        - a**5 * e2
        + 3 * a**4 * w * wbar**3 * r3**3
        - 3 * a**4 * w * wbar**2 * r3bar * r3**3
        - 9 * a**4 * w * wbar**2 * r3**2
        + 6 * a**4 * w * wbar * r3bar * r3**2
        - 12 * a**4 * w * wbar * e1**2 * r3
        + 12 * a**4 * w * wbar * e1 * e2 * r3
        - 6 * a**4 * w * wbar * e1 * r3
        - 12 * a**4 * w * wbar * e2**2 * r3
        - 6 * a**4 * w * wbar * e2 * r3
        + 9 * a**4 * w * wbar * r3
        + 4 * a**4 * w * r3bar * e1**2 * r3
        - 4 * a**4 * w * r3bar * e1 * e2 * r3
        + 2 * a**4 * w * r3bar * e1 * r3
        + 4 * a**4 * w * r3bar * e2**2 * r3
        + 2 * a**4 * w * r3bar * e2 * r3
        - 3 * a**4 * w * r3bar * r3
        - 3 * a**4 * w * e1**3
        + 3 * a**4 * w * e1**2 * e2
        + 3 * a**4 * w * e1 * e2**2
        - 12 * a**4 * w * e1 * e2
        + 6 * a**4 * w * e1
        - 3 * a**4 * w * e2**3
        + 6 * a**4 * w * e2
        - 3 * a**4 * w
        - 3 * a**4 * wbar**2 * e1 * r3**3
        - 3 * a**4 * wbar**2 * e2 * r3**3
        + 3 * a**4 * wbar**2 * r3**3
        - a**4 * wbar * r3bar * e1 * r3**3
        - a**4 * wbar * r3bar * e2 * r3**3
        - 15 * a**4 * wbar * e1**2 * r3**2
        + 6 * a**4 * wbar * e1 * e2 * r3**2
        + 6 * a**4 * wbar * e1 * r3**2
        - 15 * a**4 * wbar * e2**2 * r3**2
        + 6 * a**4 * wbar * e2 * r3**2
        - 6 * a**4 * wbar * r3**2
        + 5 * a**4 * r3bar * e1**2 * r3**2
        - 2 * a**4 * r3bar * e1 * e2 * r3**2
        + a**4 * r3bar * e1 * r3**2
        + 5 * a**4 * r3bar * e2**2 * r3**2
        + a**4 * r3bar * e2 * r3**2
        - 9 * a**4 * e1**3 * r3
        - 3 * a**4 * e1**2 * e2 * r3
        - 2 * a**4 * e1**2 * r3**3
        + 8 * a**4 * e1**2 * r3
        - 3 * a**4 * e1 * e2**2 * r3
        + 4 * a**4 * e1 * e2 * r3**3
        + 4 * a**4 * e1 * e2 * r3
        - 5 * a**4 * e1 * r3
        - 9 * a**4 * e2**3 * r3
        - 2 * a**4 * e2**2 * r3**3
        + 8 * a**4 * e2**2 * r3
        - 5 * a**4 * e2 * r3
        + 3 * a**4 * r3
        - 6 * a**3 * w * wbar**2 * e1 * r3**3
        + 6 * a**3 * w * wbar**2 * e2 * r3**3
        + 4 * a**3 * w * wbar * r3bar * e1 * r3**3
        - 4 * a**3 * w * wbar * r3bar * e2 * r3**3
        + 6 * a**3 * w * wbar * e1**2 * r3**2
        + 12 * a**3 * w * wbar * e1 * r3**2
        - 6 * a**3 * w * wbar * e2**2 * r3**2
        - 12 * a**3 * w * wbar * e2 * r3**2
        - 2 * a**3 * w * r3bar * e1**2 * r3**2
        - 4 * a**3 * w * r3bar * e1 * r3**2
        + 2 * a**3 * w * r3bar * e2**2 * r3**2
        + 4 * a**3 * w * r3bar * e2 * r3**2
        + 9 * a**3 * w * e1**3 * r3
        - 3 * a**3 * w * e1**2 * e2 * r3
        + 3 * a**3 * w * e1 * e2**2 * r3
        - 6 * a**3 * w * e1 * r3
        - 9 * a**3 * w * e2**3 * r3
        + 6 * a**3 * w * e2 * r3
        + 6 * a**3 * wbar * e1**2 * r3**3
        - 4 * a**3 * wbar * e1 * r3**3
        - 6 * a**3 * wbar * e2**2 * r3**3
        + 4 * a**3 * wbar * e2 * r3**3
        + 9 * a**3 * e1**3 * r3**2
        - 3 * a**3 * e1**2 * e2 * r3**2
        - 4 * a**3 * e1**2 * r3**2
        + 3 * a**3 * e1 * e2**2 * r3**2
        + 4 * a**3 * e1 * r3**2
        - 9 * a**3 * e2**3 * r3**2
        + 4 * a**3 * e2**2 * r3**2
        - 4 * a**3 * e2 * r3**2
        - 12 * a**2 * w * wbar * e1 * e2 * r3**3
        + 4 * a**2 * w * r3bar * e1 * e2 * r3**3
        - 9 * a**2 * w * e1**3 * r3**2
        - 3 * a**2 * w * e1**2 * e2 * r3**2
        - 3 * a**2 * w * e1 * e2**2 * r3**2
        + 12 * a**2 * w * e1 * e2 * r3**2
        - 9 * a**2 * w * e2**3 * r3**2
        - 3 * a**2 * e1**3 * r3**3
        + 3 * a**2 * e1**2 * e2 * r3**3
        + 3 * a**2 * e1 * e2**2 * r3**3
        - 4 * a**2 * e1 * e2 * r3**3
        - 3 * a**2 * e2**3 * r3**3
        + 3 * a * w * e1**3 * r3**3
        + 3 * a * w * e1**2 * e2 * r3**3
        - 3 * a * w * e1 * e2**2 * r3**3
        - 3 * a * w * e2**3 * r3**3
    )
    return p_8 

def _poly_coeffs_triple_p9(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_9 = (
        -3 * a**8 * w * wbar * r3**2
        + 3 * a**8 * w * r3bar * r3**2
        - 2 * a**8 * w * e1 * r3
        - 2 * a**8 * w * e2 * r3
        + 2 * a**8 * w * r3
        - a**8 * wbar * r3**3
        + a**8 * r3bar * r3**3
        + 2 * a**8 * e1 * r3**2
        + 2 * a**8 * e2 * r3**2
        - 2 * a**8 * r3**2
        + 3 * a**7 * w * e1 * r3**2
        - 3 * a**7 * w * e2 * r3**2
        + 3 * a**7 * wbar * e1 * r3**2
        - 3 * a**7 * wbar * e2 * r3**2
        - 3 * a**7 * r3bar * e1 * r3**2
        + 3 * a**7 * r3bar * e2 * r3**2
        + 2 * a**7 * e1**2 * r3
        + a**7 * e1 * r3**3
        - 2 * a**7 * e1 * r3
        - 2 * a**7 * e2**2 * r3
        - a**7 * e2 * r3**3
        + 2 * a**7 * e2 * r3
        + 3 * a**6 * w * wbar**3 * r3**2
        - 3 * a**6 * w * wbar**2 * r3bar * r3**2
        + 6 * a**6 * w * wbar**2 * e1 * r3
        + 6 * a**6 * w * wbar**2 * e2 * r3
        - 6 * a**6 * w * wbar**2 * r3
        - 4 * a**6 * w * wbar * r3bar * e1 * r3
        - 4 * a**6 * w * wbar * r3bar * e2 * r3
        + 4 * a**6 * w * wbar * r3bar * r3
        + 3 * a**6 * w * wbar * e1**2
        + 6 * a**6 * w * wbar * e1 * e2
        - 6 * a**6 * w * wbar * e1
        + 3 * a**6 * w * wbar * e2**2
        - 6 * a**6 * w * wbar * e2
        + 3 * a**6 * w * wbar
        - a**6 * w * r3bar * e1**2
        - 2 * a**6 * w * r3bar * e1 * e2
        + 2 * a**6 * w * r3bar * e1
        - a**6 * w * r3bar * e2**2
        + 2 * a**6 * w * r3bar * e2
        - a**6 * w * r3bar
        - a**6 * w * e1 * r3**3
        - a**6 * w * e2 * r3**3
        + a**6 * wbar**3 * r3**3
        - a**6 * wbar**2 * r3bar * r3**3
        + 3 * a**6 * wbar**2 * e1 * r3**2
        + 3 * a**6 * wbar**2 * e2 * r3**2
        - 5 * a**6 * wbar * r3bar * e1 * r3**2
        - 5 * a**6 * wbar * r3bar * e2 * r3**2
        + 2 * a**6 * wbar * r3bar * r3**2
        + 3 * a**6 * wbar * e1**2 * r3
        + 6 * a**6 * wbar * e1 * e2 * r3
        - 2 * a**6 * wbar * e1 * r3
        + 3 * a**6 * wbar * e2**2 * r3
        - 2 * a**6 * wbar * e2 * r3
        - a**6 * wbar * r3
        - 3 * a**6 * r3bar * e1**2 * r3
        - 6 * a**6 * r3bar * e1 * e2 * r3
        + 4 * a**6 * r3bar * e1 * r3
        - 3 * a**6 * r3bar * e2**2 * r3
        + 4 * a**6 * r3bar * e2 * r3
        - a**6 * r3bar * r3
        + a**6 * e1**3
        + 3 * a**6 * e1**2 * e2
        - 3 * a**6 * e1**2 * r3**2
        - 2 * a**6 * e1**2
        + 3 * a**6 * e1 * e2**2
        + 6 * a**6 * e1 * e2 * r3**2
        - 4 * a**6 * e1 * e2
        + a**6 * e1
        + a**6 * e2**3
        - 3 * a**6 * e2**2 * r3**2
        - 2 * a**6 * e2**2
        + a**6 * e2
        - 9 * a**5 * w * wbar**2 * e1 * r3**2
        + 9 * a**5 * w * wbar**2 * e2 * r3**2
        + 6 * a**5 * w * wbar * r3bar * e1 * r3**2
        - 6 * a**5 * w * wbar * r3bar * e2 * r3**2
        - 12 * a**5 * w * wbar * e1**2 * r3
        + 12 * a**5 * w * wbar * e1 * r3
        + 12 * a**5 * w * wbar * e2**2 * r3
        - 12 * a**5 * w * wbar * e2 * r3
        + 4 * a**5 * w * r3bar * e1**2 * r3
        - 4 * a**5 * w * r3bar * e1 * r3
        - 4 * a**5 * w * r3bar * e2**2 * r3
        + 4 * a**5 * w * r3bar * e2 * r3
        - 3 * a**5 * w * e1**3
        - 3 * a**5 * w * e1**2 * e2
        + 6 * a**5 * w * e1**2
        + 3 * a**5 * w * e1 * e2**2
        - 3 * a**5 * w * e1
        + 3 * a**5 * w * e2**3
        - 6 * a**5 * w * e2**2
        + 3 * a**5 * w * e2
        - 3 * a**5 * wbar**2 * e1 * r3**3
        + 3 * a**5 * wbar**2 * e2 * r3**3
        + 2 * a**5 * wbar * r3bar * e1 * r3**3
        - 2 * a**5 * wbar * r3bar * e2 * r3**3
        - 6 * a**5 * wbar * e1**2 * r3**2
        + 6 * a**5 * wbar * e2**2 * r3**2
        + 5 * a**5 * r3bar * e1**2 * r3**2
        - 2 * a**5 * r3bar * e1 * r3**2
        - 5 * a**5 * r3bar * e2**2 * r3**2
        + 2 * a**5 * r3bar * e2 * r3**2
        - 3 * a**5 * e1**3 * r3
        - 3 * a**5 * e1**2 * e2 * r3
        + a**5 * e1**2 * r3**3
        + 2 * a**5 * e1**2 * r3
        + 3 * a**5 * e1 * e2**2 * r3
        + a**5 * e1 * r3
        + 3 * a**5 * e2**3 * r3
        - a**5 * e2**2 * r3**3
        - 2 * a**5 * e2**2 * r3
        - a**5 * e2 * r3
        + 3 * a**4 * w * wbar**2 * e1 * r3**3
        + 3 * a**4 * w * wbar**2 * e2 * r3**3
        - 2 * a**4 * w * wbar * r3bar * e1 * r3**3
        - 2 * a**4 * w * wbar * r3bar * e2 * r3**3
        + 15 * a**4 * w * wbar * e1**2 * r3**2
        - 6 * a**4 * w * wbar * e1 * e2 * r3**2
        - 6 * a**4 * w * wbar * e1 * r3**2
        + 15 * a**4 * w * wbar * e2**2 * r3**2
        - 6 * a**4 * w * wbar * e2 * r3**2
        - 5 * a**4 * w * r3bar * e1**2 * r3**2
        + 2 * a**4 * w * r3bar * e1 * e2 * r3**2
        + 2 * a**4 * w * r3bar * e1 * r3**2
        - 5 * a**4 * w * r3bar * e2**2 * r3**2
        + 2 * a**4 * w * r3bar * e2 * r3**2
        + 9 * a**4 * w * e1**3 * r3
        + 3 * a**4 * w * e1**2 * e2 * r3
        - 12 * a**4 * w * e1**2 * r3
        + 3 * a**4 * w * e1 * e2**2 * r3
        + 3 * a**4 * w * e1 * r3
        + 9 * a**4 * w * e2**3 * r3
        - 12 * a**4 * w * e2**2 * r3
        + 3 * a**4 * w * e2 * r3
        + 3 * a**4 * wbar * e1**2 * r3**3
        - 6 * a**4 * wbar * e1 * e2 * r3**3
        + 2 * a**4 * wbar * e1 * r3**3
        + 3 * a**4 * wbar * e2**2 * r3**3
        + 2 * a**4 * wbar * e2 * r3**3
        - 2 * a**4 * r3bar * e1**2 * r3**3
        - 2 * a**4 * r3bar * e2**2 * r3**3
        + 3 * a**4 * e1**3 * r3**2
        - 3 * a**4 * e1**2 * e2 * r3**2
        + 2 * a**4 * e1**2 * r3**2
        - 3 * a**4 * e1 * e2**2 * r3**2
        + 4 * a**4 * e1 * e2 * r3**2
        - 2 * a**4 * e1 * r3**2
        + 3 * a**4 * e2**3 * r3**2
        + 2 * a**4 * e2**2 * r3**2
        - 2 * a**4 * e2 * r3**2
        - 6 * a**3 * w * wbar * e1**2 * r3**3
        + 6 * a**3 * w * wbar * e2**2 * r3**3
        + 2 * a**3 * w * r3bar * e1**2 * r3**3
        - 2 * a**3 * w * r3bar * e2**2 * r3**3
        - 9 * a**3 * w * e1**3 * r3**2
        + 3 * a**3 * w * e1**2 * e2 * r3**2
        + 6 * a**3 * w * e1**2 * r3**2
        - 3 * a**3 * w * e1 * e2**2 * r3**2
        + 9 * a**3 * w * e2**3 * r3**2
        - 6 * a**3 * w * e2**2 * r3**2
        - a**3 * e1**3 * r3**3
        + 3 * a**3 * e1**2 * e2 * r3**3
        - 2 * a**3 * e1**2 * r3**3
        - 3 * a**3 * e1 * e2**2 * r3**3
        + a**3 * e2**3 * r3**3
        + 2 * a**3 * e2**2 * r3**3
        + 3 * a**2 * w * e1**3 * r3**3
        - 3 * a**2 * w * e1**2 * e2 * r3**3
        - 3 * a**2 * w * e1 * e2**2 * r3**3
        + 3 * a**2 * w * e2**3 * r3**3
    )
    return p_9

def _poly_coeffs_triple_p10(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    p_10 = (
        a**8 * w * wbar * r3**3
        - a**8 * w * r3bar * r3**3
        + a**8 * w * e1 * r3**2
        + a**8 * w * e2 * r3**2
        - a**8 * w * r3**2
        - a**8 * e1 * r3**3
        - a**8 * e2 * r3**3
        + a**8 * r3**3
        - a**7 * w * e1 * r3**3
        + a**7 * w * e2 * r3**3
        - a**7 * wbar * e1 * r3**3
        + a**7 * wbar * e2 * r3**3
        + a**7 * r3bar * e1 * r3**3
        - a**7 * r3bar * e2 * r3**3
        - a**7 * e1**2 * r3**2
        + a**7 * e1 * r3**2
        + a**7 * e2**2 * r3**2
        - a**7 * e2 * r3**2
        - a**6 * w * wbar**3 * r3**3
        + a**6 * w * wbar**2 * r3bar * r3**3
        - 3 * a**6 * w * wbar**2 * e1 * r3**2
        - 3 * a**6 * w * wbar**2 * e2 * r3**2
        + 3 * a**6 * w * wbar**2 * r3**2
        + 2 * a**6 * w * wbar * r3bar * e1 * r3**2
        + 2 * a**6 * w * wbar * r3bar * e2 * r3**2
        - 2 * a**6 * w * wbar * r3bar * r3**2
        - 3 * a**6 * w * wbar * e1**2 * r3
        - 6 * a**6 * w * wbar * e1 * e2 * r3
        + 6 * a**6 * w * wbar * e1 * r3
        - 3 * a**6 * w * wbar * e2**2 * r3
        + 6 * a**6 * w * wbar * e2 * r3
        - 3 * a**6 * w * wbar * r3
        + a**6 * w * r3bar * e1**2 * r3
        + 2 * a**6 * w * r3bar * e1 * e2 * r3
        - 2 * a**6 * w * r3bar * e1 * r3
        + a**6 * w * r3bar * e2**2 * r3
        - 2 * a**6 * w * r3bar * e2 * r3
        + a**6 * w * r3bar * r3
        - a**6 * w * e1**3
        - 3 * a**6 * w * e1**2 * e2
        + 3 * a**6 * w * e1**2
        - 3 * a**6 * w * e1 * e2**2
        + 6 * a**6 * w * e1 * e2
        - 3 * a**6 * w * e1
        - a**6 * w * e2**3
        + 3 * a**6 * w * e2**2
        - 3 * a**6 * w * e2
        + a**6 * w
        - a**6 * wbar**2 * r3**3
        + a**6 * wbar * r3bar * e1 * r3**3
        + a**6 * wbar * r3bar * e2 * r3**3
        - 2 * a**6 * wbar * e1 * r3**2
        - 2 * a**6 * wbar * e2 * r3**2
        + 2 * a**6 * wbar * r3**2
        + a**6 * r3bar * e1**2 * r3**2
        + 2 * a**6 * r3bar * e1 * e2 * r3**2
        - a**6 * r3bar * e1 * r3**2
        + a**6 * r3bar * e2**2 * r3**2
        - a**6 * r3bar * e2 * r3**2
        + a**6 * e1**2 * r3**3
        - a**6 * e1**2 * r3
        - 2 * a**6 * e1 * e2 * r3**3
        - 2 * a**6 * e1 * e2 * r3
        + 2 * a**6 * e1 * r3
        + a**6 * e2**2 * r3**3
        - a**6 * e2**2 * r3
        + 2 * a**6 * e2 * r3
        - a**6 * r3
        + 3 * a**5 * w * wbar**2 * e1 * r3**3
        - 3 * a**5 * w * wbar**2 * e2 * r3**3
        - 2 * a**5 * w * wbar * r3bar * e1 * r3**3
        + 2 * a**5 * w * wbar * r3bar * e2 * r3**3
        + 6 * a**5 * w * wbar * e1**2 * r3**2
        - 6 * a**5 * w * wbar * e1 * r3**2
        - 6 * a**5 * w * wbar * e2**2 * r3**2
        + 6 * a**5 * w * wbar * e2 * r3**2
        - 2 * a**5 * w * r3bar * e1**2 * r3**2
        + 2 * a**5 * w * r3bar * e1 * r3**2
        + 2 * a**5 * w * r3bar * e2**2 * r3**2
        - 2 * a**5 * w * r3bar * e2 * r3**2
        + 3 * a**5 * w * e1**3 * r3
        + 3 * a**5 * w * e1**2 * e2 * r3
        - 6 * a**5 * w * e1**2 * r3
        - 3 * a**5 * w * e1 * e2**2 * r3
        + 3 * a**5 * w * e1 * r3
        - 3 * a**5 * w * e2**3 * r3
        + 6 * a**5 * w * e2**2 * r3
        - 3 * a**5 * w * e2 * r3
        + 2 * a**5 * wbar * e1 * r3**3
        - 2 * a**5 * wbar * e2 * r3**3
        - a**5 * r3bar * e1**2 * r3**3
        + a**5 * r3bar * e2**2 * r3**3
        + 2 * a**5 * e1**2 * r3**2
        - 2 * a**5 * e1 * r3**2
        - 2 * a**5 * e2**2 * r3**2
        + 2 * a**5 * e2 * r3**2
        - 3 * a**4 * w * wbar * e1**2 * r3**3
        + 6 * a**4 * w * wbar * e1 * e2 * r3**3
        - 3 * a**4 * w * wbar * e2**2 * r3**3
        + a**4 * w * r3bar * e1**2 * r3**3
        - 2 * a**4 * w * r3bar * e1 * e2 * r3**3
        + a**4 * w * r3bar * e2**2 * r3**3
        - 3 * a**4 * w * e1**3 * r3**2
        + 3 * a**4 * w * e1**2 * e2 * r3**2
        + 3 * a**4 * w * e1**2 * r3**2
        + 3 * a**4 * w * e1 * e2**2 * r3**2
        - 6 * a**4 * w * e1 * e2 * r3**2
        - 3 * a**4 * w * e2**3 * r3**2
        + 3 * a**4 * w * e2**2 * r3**2
        - a**4 * e1**2 * r3**3
        + 2 * a**4 * e1 * e2 * r3**3
        - a**4 * e2**2 * r3**3
        + a**3 * w * e1**3 * r3**3
        - 3 * a**3 * w * e1**2 * e2 * r3**3
        + 3 * a**3 * w * e1 * e2**2 * r3**3
        - a**3 * w * e2**3 * r3**3
    )
    return p_10