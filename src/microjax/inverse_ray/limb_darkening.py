import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax import custom_jvp

#@partial(jit, static_argnames=("u1"))
@custom_jvp
def Is_limb_1st(d, u1=0.0):
    """
    Calculate the normalized limb-darkened intensity using a linear limb-darkening law.
    - The equation implements the linear limb-darkening law:
      I(r) = I0 * (1 - u1 * (1 - sqrt(1 - r))),
      where I0 is a normalization constant ensuring that the total flux is conserved.
    - `d` is normalized by rho.
    - `u1` should be in the range [0, 1].
    """
    mu = jnp.sqrt(1.0 - d**2)
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    I  = I0 * (1.0 - u1 * (1.0 - mu))
    return jnp.where(d < 1.0, I, 0.0)

@Is_limb_1st.defjvp
def Is_limb_1st_jvp(primals, tangents):
    d, u1 = primals
    d_dot, u1_dot = tangents
    #primal_out = Is_limb_1st(d, u1)

    # I = I0(u1) * profile(mu(d), u1) * sigmoid(z(d))
    fac = 100.0
    mu = jnp.sqrt(jnp.maximum(1e-6, 1.0 - d**2))
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    dI0_du1 = -3.0 / jnp.pi / (3.0 - u1)**2
    prof = 1.0 - u1 * (1.0 - mu)
    dprof_du1 = - (1.0 - mu)
    dprof_dmu = u1
    dmu_dd = -d / mu
    z = 1.0 - d
    sigmoid = jax.nn.sigmoid(fac * z)
    dsig_dz = fac * sigmoid * (1.0 - sigmoid)
    dz_dd = -1.0

    primal_out = I0 * prof * sigmoid
    tangent_out = u1_dot * (dI0_du1 * prof * sigmoid + I0 * dprof_du1 * sigmoid) \
      + d_dot * (I0 * dprof_dmu * dmu_dd * sigmoid + I0 * prof * dsig_dz * dz_dd)
    return primal_out, tangent_out
    
    """
    mask = (d < 1.0)
    mu      = jnp.where(mask, jnp.sqrt(1.0 - d**2), 1.0)        # ダミー値でいい
    dmu_dd  = jnp.where(mask, -d / jnp.maximum(1e-15, mu), 0.0) # d>=1 なら0
    profile = jnp.where(mask, 1.0 - u1*(1.0 - mu), 0.0)

    I0         = 3.0 / (jnp.pi * (3.0-u1))
    dI0_du1    = -3.0/ (jnp.pi * (3.0-u1)**2)
    dprof_dd  = u1 * dmu_dd
    dprof_du1 = - (1.0 - mu)
    
    primal_out = jnp.where(mask, I0*profile, 0.0)
    tangent_out = (u1_dot * (dI0_du1*profile + I0*dprof_du1) + d_dot * (I0*dprof_dd))

    # For eval., I = I0(u1) * profile(d, u1) 
    # For AD, I = I0(u1) * profile(mu(d), u1) * sigmoid(z(d))
    # 安全な範囲制限
    mask = d < 1.0
    mu = jnp.where(mask, jnp.sqrt(jnp.maximum(0.0, 1.0 - d**2)), 0.0)  # 安全な sqrt
    I0 = 3.0 / jnp.pi / (3.0 - u1)

    # 微分計算
    dI0_du1 = -3.0 / jnp.pi / (3.0 - u1)**2
    profile = jnp.where(mask, 1.0 - u1 * (1.0 - mu), 0.0)
    dmu_dd = jnp.where(mask, -d / jnp.maximum(mu, 1e-15), 0.0)
    dprof_dd = jnp.where(mask, u1 * dmu_dd, 0.0)
    dprof_du1 = jnp.where(mask, -(1.0 - mu), 0.0)

    # 合成微分
    tangent_out = (
        u1_dot * (dI0_du1 * profile + I0 * dprof_du1)
        + d_dot * (I0 * dprof_dd)
    )
    fac = 100.0
    
    mask = d < 1.0
    mu = jnp.where(mask, jnp.sqrt(jnp.maximum(0.0, 1.0 - d**2)), 0.0)
    dmu_dd = jnp.where(mask, -d / jnp.maximum(mu, 1e-15), 0.0)

    #mu = jnp.sqrt(jnp.maximum(1e-6, 1.0 - d**2))
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    z = 1.0 - d
    profile = 1.0 - u1 * (1.0 - mu)
    sigmoid = jax.nn.sigmoid(fac * z)
    dp_dmu = u1
    #dmu_dd = -d / mu * sigmoid #jnp.where(d < 1.0, -d / mu, 0.0) 
    dp_du1 = mu - 1.0
    dI0_du1 = -3.0 / jnp.pi / (3.0 - u1)**2    
    sigmoid_dot = fac * sigmoid * (1.0 - sigmoid)
    dz_dd = -1.0
    tangent_out = u1_dot * (dI0_du1 * profile * sigmoid + I0 * dp_du1 * sigmoid) \
                  + d_dot * (I0 * dp_dmu * dmu_dd * sigmoid + I0 * profile * sigmoid_dot * dz_dd) 
    return primal_out, tangent_out 
    """