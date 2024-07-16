import jax.numpy as jnp
from jax import jit, jacfwd 
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

from microjax.caustics.lightcurve import magnifications
from microjax.caustics.extended_source import mag_extended_source
from microjax.point_source import critical_and_caustic_curves, _images_point_source, mag_point_source
from microjax.multipole import _mag_hexadecapole

def _planetary_caustic_test(w, rho, c_p=2., **params):
    e1, a = params["e1"], params["a"]
    s = 2 * a
    q = e1 / (1.0 - e1)
    x_cm = (2.0 * e1 - 1.0) * a
    w_pc = -1.0 / s 
    delta_pc = 3 * jnp.sqrt(q) / s
    return (w_pc - w).real**2 + (w_pc - w).imag**2 > c_p * (rho**2 + delta_pc**2)

@partial(jit, static_argnames=("nlenses"))
def _caustics_proximity_test(
    w, z, z_mask, rho, delta_mu_multi, nlenses=2, c_m=1e-02, gamma=0.02, c_f=4., rho_min=1e-03, **params
):
    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        # Derivatives
        f = lambda z: -e1 / (z - a) - (1 - e1) / (z + a)
        f_p = lambda z: e1 / (z - a) ** 2 + (1 - e1) / (z + a) ** 2
        f_pp = lambda z: 2 * (e1 / (a - z) ** 3 - (1 - e1) / (a + z) ** 3)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        # Derivatives
        f = lambda z: -e1 / (z - a) - e2 / (z + a) - (1 - e1 - e2) / (z + r3)
        f_p = (
            lambda z: e1 / (z - a) ** 2
            + e2 / (z + a) ** 2
            + (1 - e1 - e2) / (z + r3) ** 2
        )
        f_pp = (
            lambda z: 2 * (e1 / (a - z) ** 3 - e2 / (a + z) ** 3)
            + (1 - e1 - e2) / (z + r3) ** 3
        )
    zbar = jnp.conjugate(z)
    zhat = jnp.conjugate(w) - f(z)

    # Derivatives
    fp_z = f_p(z)
    fpp_z = f_pp(z)
    fp_zbar = f_p(zbar)
    fp_zhat = f_p(zhat)
    fpp_zbar = f_pp(zbar)
    J = 1.0 - jnp.abs(fp_z * fp_zbar)

    # Multipole test and cusp test
    mu_cusp = 6 * jnp.imag(3 * fp_zbar**3.0 * fpp_z**2.0) / J**5 * (rho + rho_min)**2
    mu_cusp = jnp.sum(jnp.abs(mu_cusp) * z_mask, axis=0)
    test_multipole_and_cusp = gamma*mu_cusp + delta_mu_multi < c_m

    # False images test
    Jhat = 1 - jnp.abs(fp_z*fp_zhat)
    factor = jnp.abs(
        J*Jhat**2/(Jhat*fpp_zbar*fp_z - jnp.conjugate(Jhat)*fpp_z*fp_zbar*fp_zhat)
    )
    test_false_images = 0.5*(~z_mask*factor).sum(axis=0) > c_f*(rho + rho_min)
    test_false_images = jnp.where(
        (~z_mask).sum(axis=0)==0, 
        jnp.ones_like(test_false_images, dtype=jnp.bool_), 
        test_false_images
    )

    return test_false_images & test_multipole_and_cusp


s  = 1.1  
q  = 0.1  
a = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"a": a, "e1": e1, "s": s, "q": q}

critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)
alpha = jnp.deg2rad(65) 
tE = 20 
t0 = 0.0 
u0 = 0.1 
rho = 1e-3

t  =  jnp.linspace(-15, 12.5, 1000)
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 

w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
x_cm = 0.5 * s * (1.0 - q) / (1.0 + q) # mid-point -> center-of-mass

z, z_mask = _images_point_source(w_points - x_cm, **_params)
mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, **_params)

test1 = _caustics_proximity_test(w_points - x_cm, 
                                 z, 
                                 z_mask, 
                                 rho, 
                                 delta_mu_multi, 
                                 **_params)
test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

mag_points = mag_point_source(w_points, nlenses=2, **_params) 
mag_full = lambda w: mag_extended_source(w, rho, nlenses=2, **_params)
mag_ext  = jax.jit(jax.vmap(mag_full))(w_points)

fig, ax = plt.subplots(
    2, 1,
    figsize=(6, 6),
    gridspec_kw={'height_ratios': [4, 1], 'wspace':0.3},
    sharex=True,
)

ax[0].plot(t, mag_ext, ".")
ax[0].plot(t[test1], mag_ext[test1], ".")
ax[0].plot(t, mu_multi, c="red", alpha=0.5)
ax[0].plot(t, mag_points, "--", c="k", alpha=1.0)
ax[0].legend(["extended-source (needed)", "extended-source", "hexadecapole", "point-source"])
ax[0].set_yscale("log")
ax[1].plot(t, jnp.abs(mu_multi - mag_ext) / mag_ext, c="red", alpha=0.5)
ax[1].plot(t, jnp.abs(mag_points - mag_ext) / mag_ext, "--", c="k", alpha=1.0)
ax[1].set_yscale("log")
ax[1].set(ylim=(1e-4,1e-2))
ax[1].grid(ls="--")
fig.savefig("tests/caustics/lightcurve_cond.pdf", bbox_inches="tight")
plt.show()