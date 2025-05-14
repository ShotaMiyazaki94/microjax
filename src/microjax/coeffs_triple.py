import jax.numpy as jnp
from jax import jit

def _poly_coeffs_triple_CM(w, a, r3, e1, e2):
    eps1 = e2 # primary lens
    eps2 = e1 # secondary lens
    eps3 = 1.0 - e1 - e2 # third lens
    shift_cm = (eps1 * (-a) + eps2 * a + eps3 * r3.real) \
    + 1j*(r3.imag * eps3) # mid-point to center of mass
    w_cm = w - shift_cm 
    r1 = -a - shift_cm
    r2 = +a - shift_cm
    r3 = r3 - shift_cm
    #bool_cm = jnp.abs(eps1*r1+eps2*r2+eps3*r3) < 1e-12
    #print("CM:", bool_cm)

    cc1 = r1
    cc2 = r2
    cc3 = r3
    aa = -(cc1+cc2+cc3)
    bb = cc1*cc2 + cc1*cc3 + cc2*cc3
    cc = -cc1*cc2*cc3
    dd = eps1*cc2*cc3 + eps2*cc1*cc3 + eps3*cc1*cc2

    hh39 = 1.0
    hh38 = 3.0*aa
    hh37 = 3.0*bb + 3.0*aa*aa
    hh36 = 3.0*cc + 6.0*aa*bb + aa*aa*aa
    hh35 = 6.0*aa*cc + 3.0*bb*bb + 3.0*aa*aa*bb
    hh34 = 6.0*bb*cc + 3.0*aa*aa*cc + 3.0*aa*bb*bb
    hh33 = 3.0*cc*cc + 6.0*aa*bb*cc + bb*bb*bb
    hh32 = 3.0*aa*cc*cc + 3.0*bb*bb*cc
    hh31 = 3.0*bb*cc*cc
    hh30 = cc*cc*cc

    hh28 = 1.0
    hh27 = 3.0*aa
    hh26 = dd + 2.0*bb + 3.0*aa*aa
    hh25 = 2.0*aa*dd + 4.0*aa*bb + aa*aa*aa + 2.0*cc
    hh24 = 2.0*dd*bb + dd*aa*aa + 4.0*aa*cc +2.0*aa*aa*bb + bb*bb
    hh23 = 2.0*dd*cc + 2.0*dd*aa*bb + 2.0*aa*aa*cc +aa*bb*bb + 2.0*bb*cc
    hh22 = 2.0*cc*aa*dd + dd*bb*bb + 2.0*aa*bb*cc + cc*cc
    hh21 = 2.0*bb*cc*dd + aa*cc*cc
    hh20 = cc*cc*dd

    hh17 = 1.0
    hh16 = 3.0*aa
    hh15 = 2.0*dd + 3.0*aa*aa + bb
    hh14 = 4.0*aa*dd + aa*aa*aa + 2.0*aa*bb + cc
    hh13 = dd*dd + 2.0*aa*aa*dd + 2.0*bb*dd + bb*aa*aa + 2.0*aa*cc
    hh12 = aa*dd*dd + 2.0*aa*bb*dd + 2.0*cc*dd + cc*aa*aa
    hh11 = bb*dd*dd + 2.0*aa*cc*dd
    hh10 = cc*dd*dd

    hh06 = 1.0
    hh05 = 3.0*aa
    hh04 = 3.0*dd + 3.0*aa*aa
    hh03 = 6.0*aa*dd + aa*aa*aa
    hh02 = 3.0*dd*dd + 3.0*aa*aa*dd
    hh01 = 3.0*aa*dd*dd
    hh00 = dd*dd*dd

    ww = w_cm
    ww1 = ww - cc1
    ww2 = ww - cc2
    ww3 = ww - cc3
    
    wwbar  = jnp.conjugate(ww)
    ww1bar = jnp.conjugate(ww1)
    ww2bar = jnp.conjugate(ww2)
    ww3bar = jnp.conjugate(ww3)

    wwaa = ww1bar+ww2bar+ww3bar
    wwbb = ww1bar*ww2bar + ww2bar*ww3bar + ww1bar*ww3bar
    wwcc = ww1bar*ww2bar*ww3bar
    wwdd = eps1*ww2bar*ww3bar + eps2*ww1bar*ww3bar + eps3*ww1bar*ww2bar

    p_10 = hh39*wwcc 
    p_9  = hh38*wwcc + hh28*wwbb - (ww*wwcc+wwdd)*hh39
    p_8  = hh37*wwcc + hh27*wwbb + hh17*wwaa - (ww*wwcc + wwdd)*hh38 \
    - (ww*wwbb + wwaa - wwbar)*hh28
    
    p_7  = hh36*wwcc + hh26*wwbb + hh16*wwaa + hh06 - (ww*wwcc + wwdd)*hh37 \
    - (ww*wwbb + wwaa-wwbar)*hh27 - (ww*wwaa + 1.0)*hh17
    p_6  = hh35*wwcc + hh25*wwbb + hh15*wwaa + hh05 - (ww*wwcc + wwdd)*hh36 \
    - (ww*wwbb + wwaa-wwbar)*hh26 - (ww*wwaa + 1.0)*hh16  - ww*hh06
    p_5  = hh34*wwcc + hh24*wwbb + hh14*wwaa + hh04 - (ww*wwcc + wwdd)*hh35 \
    - (ww*wwbb + wwaa-wwbar)*hh25 - (ww*wwaa + 1.0)*hh15  - ww*hh05
    p_4  = hh33*wwcc + hh23*wwbb + hh13*wwaa + hh03 - (ww*wwcc + wwdd)*hh34 \
    - (ww*wwbb + wwaa-wwbar)*hh24 - (ww*wwaa + 1.0)*hh14  - ww*hh04
    p_3  = hh32*wwcc + hh22*wwbb + hh12*wwaa + hh02 - (ww*wwcc + wwdd)*hh33 \
    - (ww*wwbb + wwaa-wwbar)*hh23 - (ww*wwaa + 1.0)*hh13  - ww*hh03
    p_2  = hh31*wwcc + hh21*wwbb + hh11*wwaa + hh01 - (ww*wwcc + wwdd)*hh32 \
    - (ww*wwbb + wwaa-wwbar)*hh22 - (ww*wwaa + 1.0)*hh12  - ww*hh02
    p_1  = hh30*wwcc + hh20*wwbb + hh10*wwaa + hh00 - (ww*wwcc + wwdd)*hh31 \
    - (ww*wwbb + wwaa-wwbar)*hh21 - (ww*wwaa + 1.0)*hh11  - ww*hh01
    p_0 = - (ww*wwcc + wwdd)*hh30 - (ww*wwbb + wwaa-wwbar)*hh20 - (ww*wwaa + 1)*hh10  - ww*hh00;

    p = jnp.stack([p_10, p_9, p_8, p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0])
    
    return jnp.moveaxis(p, 0, -1), shift_cm



def pre_poly_coeffs_triple_CM(w, a, r3, e1, e2):
    """
    Args:
        w : complex position of the source at the mid-point coordinate
        a : distance from the primary and secondary lens to the mid-point
        r3 : complex position of the third lens w.r.t. the mid-point 
        e1 : mass fraction of the secondary lens, (a, 0) in the mid-point coordinate
        e2 : mass fraction of the primary lens, (-a, 0) in the mid-point coordinate
    Returns:
        p : coefficients of the polynomial 
        shift : complex shift from the mid-point to the center of mass
    """
    eps1 = e2 # primary lens
    eps2 = e1 # secondary lens
    eps3 = 1.0 - e1 - e2 # third lens
    shift_cm = (eps1 * (-a) + eps2 * a + eps3 * r3.real) + 1j*(r3.imag * eps3) # mid-point to center of mass
    w_cm = w - shift_cm 
    r1 = -a - shift_cm
    r2 = +a - shift_cm
    r3 = r3 - shift_cm

    aa = -(r1 + r2 + r3)
    bb = r1 * r2 + r1 * r3 + r2 * r3
    cc = -r1 * r2 * r3
    dd = eps1 * r2 * r3 + eps2 * r1 * r3 + eps3 * r1 * r2

    hh39 = 1.0
    hh38 = 3.0 * aa
    hh37 = 3.0 * bb + 3.0 * aa * aa
    hh36 = 3.0 * cc + 6.0 * aa * bb + aa * aa * aa
    hh35 = 6.0 * aa * cc + 3.0 * bb * bb + 3.0 * aa * aa * bb
    hh34 = 6.0 * bb * cc + 3.0 * aa * aa * cc + 3.0 * aa * bb * bb
    hh33 = 3.0 * cc * cc + 6.0 * aa * bb * cc + bb * bb * bb
    hh32 = 3.0 * aa * cc * cc + 3.0 * bb * bb * cc
    hh31 = 3.0 * bb * cc * cc
    hh30 = cc * cc * cc

    hh28 = 1.0
    hh27 = 3.0 * aa
    hh26 = dd + 2.0 * bb + 3.0 * aa * aa
    hh25 = 2.0 * aa * dd + 4.0 * aa * bb + aa * aa * aa + 2.0 * cc
    hh24 = 2.0 * dd * bb + dd * aa * aa + 4.0 * aa * cc + 2.0 * aa * aa * bb + bb * bb
    hh23 = 2.0 * dd * cc + 2.0 * dd * aa * bb + 2.0 * aa * aa * cc + aa * bb * bb + 2.0 * bb * cc
    hh22 = 2.0 * cc * aa * dd + dd * bb * bb + 2.0 * aa * bb * cc + cc * cc
    hh21 = 2.0 * bb * cc * dd + aa * cc * cc
    hh20 = cc * cc * dd

    hh17 = 1.0
    hh16 = 3.0 * aa
    hh15 = 2.0 * dd + 3.0 * aa * aa + bb
    hh14 = 4.0 * aa * dd + aa * aa * aa + 2.0 * aa * bb + cc
    hh13 = dd * dd + 2.0 * aa * aa * dd + 2.0 * bb * dd + bb * aa * aa + 2.0 * aa * cc
    hh12 = aa * dd * dd + 2.0 * aa * bb * dd + 2.0 * cc * dd + cc * aa * aa
    hh11 = bb * dd * dd + 2.0 * aa * cc * dd
    hh10 = cc * dd * dd

    hh06 = 1.0
    hh05 = 3.0 * aa
    hh04 = 3.0 * dd + 3.0 * aa * aa
    hh03 = 6.0 * aa * dd + aa * aa * aa
    hh02 = 3.0 * dd * dd + 3.0 * aa * aa * dd * dd
    hh01 = 3.0 * aa * dd * dd
    hh00 = dd * dd * dd

    wbar = jnp.conjugate(w_cm)
    wr1bar = wbar - jnp.conjugate(r1)
    wr2bar = wbar - jnp.conjugate(r2)
    wr3bar = wbar - jnp.conjugate(r3)
    aw = wr1bar + wr2bar + wr3bar
    bw = wr1bar * wr2bar + wr1bar * wr3bar + wr2bar * wr3bar
    cw = wr1bar * wr2bar * wr3bar
    dw = eps1 * wr2bar * wr3bar + eps2 * wr1bar * wr3bar + eps3 * wr1bar * wr2bar

    waw_1 = w_cm * aw - 1.0
    wbw_aw_wbar = w_cm * bw + aw - wbar
    wcw_bw = w_cm * cw + dw #+ dw

    p_0  = jnp.ones_like(w_cm) * 1e-6
    p_1 = (hh00 + hh10 * aw + hh20 * bw + hh30 * cw) \
        - (hh01 * w_cm + hh11 * waw_1 + hh21 * wbw_aw_wbar + hh31 * wcw_bw)
    p_2 = (hh01 + hh11 * aw + hh21 * bw + hh31 * cw) \
        - (hh02 * w_cm + hh12 * waw_1 + hh22 * wbw_aw_wbar + hh32 * wcw_bw)
    p_3 = (hh02 + hh12 * aw + hh22 * bw + hh32 * cw) \
        - (hh03 * w_cm + hh13 * waw_1 + hh23 * wbw_aw_wbar + hh33 * wcw_bw)
    p_4 = (hh03 + hh13 * aw + hh23 * bw + hh33 * cw) \
        - (hh04 * w_cm + hh14 * waw_1 + hh24 * wbw_aw_wbar + hh34 * wcw_bw)
    p_5 = (hh04 + hh14 * aw + hh24 * bw + hh34 * cw) \
        - (hh05 * w_cm + hh15 * waw_1 + hh25 * wbw_aw_wbar + hh35 * wcw_bw) 
    p_6 = (hh05 + hh15 * aw + hh25 * bw + hh35 * cw) \
        - (hh06 * w_cm + hh16 * waw_1 + hh26 * wbw_aw_wbar + hh36 * wcw_bw)
    p_7 = (hh06 + hh16 * aw + hh26 * bw + hh36 * cw) \
        - (hh17 * waw_1 + hh27 * wbw_aw_wbar + hh37 * wcw_bw)
    p_8 = (hh17 * aw + hh27 * bw + hh37 * cw) \
        - (hh28 * wbw_aw_wbar + hh38 * wcw_bw)
    p_9 = (hh28 * bw + hh38 * cw) - (hh39 * wcw_bw)
    p_10 = hh39 * cw

    p = jnp.stack([p_10, p_9, p_8, p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0])
    #p = jnp.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10])
    
    return jnp.moveaxis(p, 0, -1), shift_cm


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


def _poly_coeffs_triple_p0(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    x0 = jnp.conjugate(w)
    x1 = jnp.conjugate(r3)
    x2 = a**2
    p_0 = x0**3 - x0**2*x1 - x0*x2 + x1*x2
    #p_0 = -(a**2) * wbar + a**2 * r3bar + wbar**3 - wbar**2 * r3bar
    return p_0

def _poly_coeffs_triple_p1(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    x0 = a**2
    x1 = jnp.conjugate(w)
    x2 = jnp.conjugate(r3)
    x3 = x1**3
    x4 = x1**2
    x5 = 3*r3
    p_1 = (-a*e1*x1 + a*e1*x2 + a*e2*x1 - a*e2*x2 - e1*x0 + e1*x1*x2 
           - e2*x0 + e2*x1*x2 + 3*r3*x0*x1 + 3*r3*x2*x4 + w*x0*x1 
           - w*x0*x2 + w*x2*x4 - w*x3 - x0*x2*x5 - 2*x1*x2 - x3*x5 + 2*x4)
    return p_1

def _poly_coeffs_triple_p2(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    x0 = jnp.conjugate(r3)
    x1 = jnp.conjugate(w)
    x2 = a**2
    x3 = a**3
    x4 = a**4
    x5 = x1**2
    x6 = 3*x5
    x7 = r3*x6
    x8 = e1*x0
    x9 = a*x8
    x10 = a*e2
    x11 = 3*x1
    x12 = r3*x11
    x13 = r3*x1
    x14 = x1**3
    x15 = r3**2
    p_2 = (3*a*e1*r3*x1 + 3*a*e1*x5 - a*e1 + 3*a*e2*r3*x0 + 2*a*e2*x0*x1 
           + a*e2 + 4*e1*r3*x2 + e1*x0 - e1*x3 - e1*x7 + 4*e2*r3*x2 
           - e2*x0*x13 + e2*x0 + e2*x3 - e2*x7 + 3*r3*w*x0*x2 
           + 3*r3*w*x14 + 4*r3*x0*x1 - r3*x2 - 3*r3*x9 + 2*w*x0*x1 
           - w*x0*x7 - w*x12*x2 + w*x2 - w*x6 + 3*x0*x15*x2 - x0*x15*x6 
           + 3*x0*x2*x5 - 3*x0*x4 - x0 + 3*x1*x4 - 2*x1*x9 + x1 - x10*x12 
           - x10*x6 - x11*x15*x2 - x13*x8 + 3*x14*x15 - 3*x14*x2 - x7)
    return p_2

def _poly_coeffs_triple_p3(w, a, r3, e1, e2):
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)
    x0 = jnp.conjugate(r3)
    x1 = jnp.conjugate(w)
    x2 = 3*w
    x3 = x1*x2
    x4 = e2*r3
    x5 = 2*a
    x6 = a**4
    x7 = e1*x0
    x8 = e2*x1
    x9 = e2**2
    x10 = a*x9
    x11 = a**3
    x12 = r3*x1
    x13 = 4*x12
    x14 = a**2
    x15 = e1**2
    x16 = w*x14
    x17 = 3*x11
    x18 = r3*x0
    x19 = 2*e1
    x20 = r3**2
    x21 = r3**3
    x22 = x1**3
    x23 = x1**2
    x24 = w*x18
    x25 = x1*x24
    x26 = x14*x23
    x27 = x14*x20
    x28 = 5*x27
    x29 = a*e1
    x30 = x23*x29
    x31 = 3*x20
    x32 = x1*x7
    x33 = x0*x8
    x34 = w*x0
    x35 = 5*x14
    x36 = 3*x34
    p_3 = (6*a*e1*r3*x0*x1 + 2*a*e1*r3 + 2*a*e1*w*x0*x1 + 3*a*e1*x0*x20 
           + 4*a*e1*x1 + 9*a*e2*r3*x23 + 3*a*e2*w*x23 - a*e2*x0*x31 
           + 2*a*e2*x0 + 3*a*e2*x1*x20 + a*r3*x15 + a*x0*x15 - 6*a*x18*x8 
           - 4*a*x8 + 2*e1*e2*x14 + 3*e1*r3*w*x23 + 3*e1*r3*x11 - e1*r3*x16 
           + e1*w*x11 + 3*e1*x1*x11 - e1*x13 + 3*e1*x14*x23 + 6*e1*x20*x23 
           - e1*x28 + 2*e1*x6 + 3*e2*r3*w*x23 - e2*w*x11 + 3*e2*x0*x11 - e2*x13 
           + 3*e2*x14*x23 - e2*x18*x19 + 6*e2*x20*x23 - e2*x28 + 2*e2*x6 
           + 6*r3*w*x23 + 9*r3*x0*x6 + r3*x0 + r3*x1 - r3*x10 + 9*r3*x14*x22 
           - 2*r3*x16 - 9*r3*x30 + 3*w*x0*x20*x23 + 3*w*x0*x6 + w*x0 
           + 3*w*x1*x14*x20 + 3*w*x14*x22 + 6*x0*x1*x14 - 2*x0*x1*x20 
           - x0*x10 - x0*x14*x21 + x0*x21*x23 + x1*x14*x21 - x1*x29*x31 
           - 9*x12*x6 - x14*x15 + 2*x14*x20 - x14*x9 - x15*x18 - x16*x4 
           - x17*x4 - x17*x7 - x17*x8 - 9*x18*x26 - x18*x9 - x19*x25 - x2*x20*x22 
           - x2*x30 - x20*x32 - x20*x33 - x21*x22 - 2*x24*x8 - 4*x25 - x26*x36 
           - 6*x26 - x27*x36 - x3*x6 - x3 - x32*x35 - x33*x35 - x34*x5*x8 - x4*x5 - x5*x7)
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