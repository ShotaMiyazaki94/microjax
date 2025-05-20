import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from microjax.point_source import _images_point_source, critical_and_caustic_curves

N_limb = 5000

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.5  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
q3 = 0.1
r3 = 0.2+1.0j
psi = jnp.arctan2(r3.imag, r3.real)

alpha = np.deg2rad(65) # angle between lens axis and source trajectory
tE = 10 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.0 # impact parameter
rho = 0.05

a  = 0.5 * s
e1 = q/(1 + q + q3)
e2 = 1/(1 + q + q3)

# Position of the center of the source with respect to the center of mass.
t  =  np.linspace(-12, 12, 100)
tau = (t - t0)/tE
y1 = -u0*np.sin(alpha) + tau*np.cos(alpha)
y2 = u0*np.cos(alpha) + tau*np.sin(alpha)

crit, cau = critical_and_caustic_curves(npts=1000, q=q, s=s, q3=q3, r3=jnp.abs(r3), psi=psi, nlenses=3)
w = jnp.array(y1[:,None] + 1.0j * y2[:,None] 
              + rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)

import seaborn as sns
sns.set_theme(font_scale=1.2, style="ticks", font="serif")
def plot(frame):
    plt.cla()
    plt.scatter(crit.ravel().real, crit.ravel().imag, s=1, color="green",label="critical curve")
    plt.scatter(cau.ravel().real, cau.ravel().imag, s=1, color="red", label="caustic")
    w_limb       = w[frame,:]                         # center-of-mass coordinate
    w_limb_shift = w[frame,:] - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1, e2=e2,
                                       r3=jnp.abs(r3), psi=psi, nlenses=3) # half-axis coordinate
    image_shift = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
    plt.scatter(w_limb.real, w_limb.imag, s=1, color="blue",label="source limb")
    plt.scatter(image_shift[mask].ravel().real, image_shift[mask].ravel().imag, s=1,color="purple",label="image limb")
    plt.legend(loc="upper left")
    plt.plot(-q*s, 0 ,".",c="k")
    plt.plot((1.0-q)*s, 0 ,".",c="k")
    plt.plot(r3.real - (0.5*s - s/(1 + q)), r3.imag ,".",c="k")
    plt.axis("equal")
    plt.xlim(-1.8,1.8)
    plt.ylim(-1.8,1.8)
    plt.xlabel(r"x $(R_{\rm E})$")
    plt.ylabel(r"y $(R_{\rm E})$")
    plt.tight_layout()

from matplotlib import animation
fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, plot, interval=1.0, frames=100, blit=False)
ani.save('tests/integrate/point_source/animation_triple.mp4', writer='ffmpeg', fps=10)
plt.show()