import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from jax import custom_jvp

# from jax.scipy.special import digamma
# from jax.lax import digamma
# from microjax.fastlens.gamma_jax import gamma_jax as gamma

__all__ = ["gamma", "ellipk", "ellipe", "j0", "j1", "j2", "j1p5"]


@custom_jvp
def gamma_(z):
    """
    Custom gamma function with JAX similar to scipy.special.gamma.
    I see that the accuracy compared with scipy.special.gamma is met at 1e-10 within -100<Re(z)<100 and -100<Im(z)<100.
    I adopted the algorothm from https://ui.adsabs.harvard.edu/abs/2021arXiv210400697C/abstract.

    Arg   : z (complex)
    return: Gamma(z) (complex)
    """
    K = 5
    N = 16
    c = jnp.log(2 * jnp.pi) / 2
    a = jnp.array([1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188])

    z = jnp.atleast_1d(z)
    original_shape = z.shape
    z = z.ravel()

    negative_real_mask = jnp.real(z) <= 0
    not_special_case = ~negative_real_mask

    def compute_case(z_val):
        t = z_val + N
        gam = a[0] / t
        t_squared = t**2
        for k in range(1, K):
            t *= t_squared
            gam += a[k] / t
        u = jnp.prod(z_val[:, None] + jnp.arange(N), axis=1)
        lg = c - (z_val + N) + (z_val + N - 0.5) * jnp.log(z_val + N) - jnp.log(u) + gam
        return jnp.exp(lg)

    def compute_reflection(z_val):
        reflected_z = 1 - z_val
        sin_term = jnp.sin(jnp.pi * z_val)
        gamma_reflected = compute_case(reflected_z)
        return jnp.pi / (sin_term * gamma_reflected)

    # Compute results for all elements
    positive_results = compute_case(z)
    negative_results = compute_reflection(z)

    # Select the appropriate results based on the masks
    g = jnp.where(not_special_case, positive_results, negative_results)

    non_positive_integer_mask = negative_real_mask & (
        jnp.real(z) == jnp.floor(jnp.real(z))
    )
    g = jnp.where(non_positive_integer_mask, jnp.nan, g)

    return g.reshape(original_shape)


def digamma_(z):
    """
    Custom digamma function with JAX, implemented from the Julia algorithm
    Args:
        z (complex): Input value(s).
    Returns:
        complex: Digamma of z.
    """
    # constants
    coeffs = jnp.array(
        [
            0.08333333333333333,
            -0.008333333333333333,
            0.003968253968253968,
            -0.004166666666666667,
            0.007575757575757576,
            -0.021092796092796094,
            0.08333333333333333,
            -0.4432598039215686,
        ],
        dtype=jnp.float64,
    )
    z = jnp.atleast_1d(z)
    original_shape = z.shape
    z = z.ravel()

    # Reflection for negative real part
    negative_real_mask = jnp.real(z) < 0
    reflection = jnp.pi / jnp.tan(jnp.pi * z)
    z = jnp.where(negative_real_mask, 1.0 - z, z)
    psi = jnp.where(negative_real_mask, -reflection, 0.0 + 0.0j)

    # Adjustment for x < X
    X = 8.0
    x = jnp.real(z)
    small_mask = x < X
    n_add = jnp.sum(1.0 / (z[:, None] + jnp.arange(X)), axis=-1)
    z = jnp.where(small_mask, z + X, z)
    psi = jnp.where(small_mask, psi - n_add, psi)

    # Asymptotic expansion
    t = 1 / z
    psi += jnp.log(z) - 0.5 * t
    t *= t  # 1/z^2
    psi -= t * jnp.polyval(coeffs[::-1], t)

    return psi.reshape(original_shape)


@gamma_.defjvp
def gamma_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = gamma_(x)
    tangent_out = primal_out * digamma_(x) * x_dot
    return primal_out, tangent_out


gamma = jit(gamma_)


def agm(a, b, num_iter=5):
    def agm_step(carry, _):
        a, b = carry
        return ((a + b) / 2, jnp.sqrt(a * b)), None

    (a, b), _ = lax.scan(agm_step, (a, b), None, length=num_iter)
    return a


@custom_jvp
def ellipk_(m):
    num_iter = 5
    a, b = 1.0, jnp.sqrt(1 - m)
    c = agm(a, b, num_iter=num_iter)
    return jnp.pi / (2 * c)


@ellipk_.defjvp
def ellipk_jvp(primals, tangents):
    (m,) = primals
    (m_dot,) = tangents
    k_val = ellipk_(
        m,
    )
    # Derivative of the complete elliptic integral of the first kind with respect to m
    dk_dm = (ellipe_(m) - (1 - m) * k_val) / (2 * m * (1 - m))
    return k_val, m_dot * dk_dm


def agm2(a0, b0, s_sum0, num_iter=5):
    def agm_step2(carry, _):
        a, b, s_sum, n = carry
        a_next, b_next = (a + b) / 2, jnp.sqrt(a * b)
        c_next = 0.5 * (a - b)
        s_sum_next = s_sum + 2 ** (n - 1) * c_next**2
        n += 1
        return (a_next, b_next, s_sum_next, n), None

    n = 1
    (a, b, s_sum, n), _ = lax.scan(
        agm_step2, (a0, b0, s_sum0, n), None, length=num_iter
    )
    return a, s_sum


@custom_jvp
def ellipe_(m):
    num_iter = 5
    a0, b0 = 1.0, jnp.sqrt(1 - m)
    c0 = jnp.sqrt(jnp.abs(a0**2 - b0**2))
    # c0     = jnp.sqrt(a0**2 - b0**2)
    s_sum0 = 0.5 * c0**2
    a, s_sum = agm2(a0, b0, s_sum0, num_iter=num_iter)
    return jnp.pi / (2 * a) * (1 - s_sum)


@ellipe_.defjvp
def ellipe_jvp(primals, tangents):
    (m,) = primals
    (m_dot,) = tangents
    e_val = ellipe_(m)
    k_val = ellipk_(m)
    # Derivative of the complete elliptic integral of the second kind with respect to m
    de_dm = (e_val - k_val) / (2 * m)
    return e_val, m_dot * de_dm


# @jit
"""
def ellipk(m):
    return lax.cond(m > 0, 
                    lambda _: ellipk_(m), 
                    lambda _: 1.0 / jnp.sqrt(jnp.abs(m) + 1.0) * ellipk_(jnp.abs(m) / (jnp.abs(m) + 1.0)), 
                    None)
#@jit
def ellipe(m):
    return lax.cond(m > 0, 
                    lambda _: ellipe_(m), 
                    lambda _: jnp.sqrt(jnp.abs(m) + 1.0) * ellipe_(jnp.abs(m) / (jnp.abs(m) + 1.0)), 
                    None)
"""


def ellipk(m):
    result = jnp.where(
        m > 0,
        ellipk_(m),
        1.0 / jnp.sqrt(jnp.abs(m) + 1.0) * ellipk_(jnp.abs(m) / (jnp.abs(m) + 1.0)),
    )
    return result


def ellipe(m):
    result = jnp.where(
        m > 0,
        ellipe_(m),
        jnp.sqrt(jnp.abs(m) + 1.0) * ellipe_(jnp.abs(m) / (jnp.abs(m) + 1.0)),
    )
    return result


# ellipk = jit(vmap(ellipk, in_axes=(0,)))
# ellipe = jit(vmap(ellipe, in_axes=(0,)))


# I imported the code from https://github.com/google/jax/pull/17038/files
# polynomial coefficients for J1

RP1 = [
    -8.99971225705559398224e8,
    4.52228297998194034323e11,
    -7.27494245221818276015e13,
    3.68295732863852883286e15,
]
RQ1 = [
    1.0,
    6.20836478118054335476e2,
    2.56987256757748830383e5,
    8.35146791431949253037e7,
    2.21511595479792499675e10,
    4.74914122079991414898e12,
    7.84369607876235854894e14,
    8.95222336184627338078e16,
    5.32278620332680085395e18,
]

PP1 = [
    7.62125616208173112003e-4,
    7.31397056940917570436e-2,
    1.12719608129684925192e0,
    5.11207951146807644818e0,
    8.42404590141772420927e0,
    5.21451598682361504063e0,
    1.00000000000000000254e0,
]
PQ1 = [
    5.71323128072548699714e-4,
    6.88455908754495404082e-2,
    1.10514232634061696926e0,
    5.07386386128601488557e0,
    8.39985554327604159757e0,
    5.20982848682361821619e0,
    9.99999999999999997461e-1,
]

QP1 = [
    5.10862594750176621635e-2,
    4.98213872951233449420e0,
    7.58238284132545283818e1,
    3.66779609360150777800e2,
    7.10856304998926107277e2,
    5.97489612400613639965e2,
    2.11688757100572135698e2,
    2.52070205858023719784e1,
]
QQ1 = [
    1.0,
    7.42373277035675149943e1,
    1.05644886038262816351e3,
    4.98641058337653607651e3,
    9.56231892404756170795e3,
    7.99704160447350683650e3,
    2.82619278517639096600e3,
    3.36093607810698293419e2,
]

YP1 = [
    1.26320474790178026440e9,
    -6.47355876379160291031e11,
    1.14509511541823727583e14,
    -8.12770255501325109621e15,
    2.02439475713594898196e17,
    -7.78877196265950026825e17,
]
YQ1 = [
    5.94301592346128195359e2,
    2.35564092943068577943e5,
    7.34811944459721705660e7,
    1.87601316108706159478e10,
    3.88231277496238566008e12,
    6.20557727146953693363e14,
    6.87141087355300489866e16,
    3.97270608116560655612e18,
]

Z1 = 1.46819706421238932572e1
Z2 = 4.92184563216946036703e1
PIO4 = 0.78539816339744830962  # pi/4
THPIO4 = 2.35619449019234492885  # 3*pi/4
SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)

# polynomial coefficients for J0

PP0 = [
    7.96936729297347051624e-4,
    8.28352392107440799803e-2,
    1.23953371646414299388e0,
    5.44725003058768775090e0,
    8.74716500199817011941e0,
    5.30324038235394892183e0,
    9.99999999999999997821e-1,
]
PQ0 = [
    9.24408810558863637013e-4,
    8.56288474354474431428e-2,
    1.25352743901058953537e0,
    5.47097740330417105182e0,
    8.76190883237069594232e0,
    5.30605288235394617618e0,
    1.00000000000000000218e0,
]

QP0 = [
    -1.13663838898469149931e-2,
    -1.28252718670509318512e0,
    -1.95539544257735972385e1,
    -9.32060152123768231369e1,
    -1.77681167980488050595e2,
    -1.47077505154951170175e2,
    -5.14105326766599330220e1,
    -6.05014350600728481186e0,
]
QQ0 = [
    1.0,
    6.43178256118178023184e1,
    8.56430025976980587198e2,
    3.88240183605401609683e3,
    7.24046774195652478189e3,
    5.93072701187316984827e3,
    2.06209331660327847417e3,
    2.42005740240291393179e2,
]

YP0 = [
    1.55924367855235737965e4,
    -1.46639295903971606143e7,
    5.43526477051876500413e9,
    -9.82136065717911466409e11,
    8.75906394395366999549e13,
    -3.46628303384729719441e15,
    4.42733268572569800351e16,
    -1.84950800436986690637e16,
]
YQ0 = [
    1.04128353664259848412e3,
    6.26107330137134956842e5,
    2.68919633393814121987e8,
    8.64002487103935000337e10,
    2.02979612750105546709e13,
    3.17157752842975028269e15,
    2.50596256172653059228e17,
]

DR10 = 5.78318596294678452118e0
DR20 = 3.04712623436620863991e1

RP0 = [
    -4.79443220978201773821e9,
    1.95617491946556577543e12,
    -2.49248344360967716204e14,
    9.70862251047306323952e15,
]
RQ0 = [
    1.0,
    4.99563147152651017219e2,
    1.73785401676374683123e5,
    4.84409658339962045305e7,
    1.11855537045356834862e10,
    2.11277520115489217587e12,
    3.10518229857422583814e14,
    3.18121955943204943306e16,
    1.71086294081043136091e18,
]


def j0_small(x):
    """
    Implementation of J0 for x < 5
    """
    z = x * x
    # if x < 1.0e-5:
    #     return 1.0 - z/4.0
    p = (z - jnp.array(DR10)) * (z - jnp.array(DR20))
    p = p * jnp.polyval(jnp.array(RP0), z) / jnp.polyval(jnp.array(RQ0), z)
    return jnp.where(x < 1e-5, 1 - z / 4.0, p)


def j0_large(x):
    """
    Implementation of J0 for x >= 5
    """

    w = 5.0 / x
    q = 25.0 / (x * x)
    p = jnp.polyval(jnp.array(PP0), q) / jnp.polyval(jnp.array(PQ0), q)
    q = jnp.polyval(jnp.array(QP0), q) / jnp.polyval(jnp.array(QQ0), q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@custom_jvp
def j0_(z):
    """
    Bessel function of the first kind of order zero and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.
    Reference:
    Cephes Mathematical Library.
    Args:
        z: The sampling point(s) at which the Bessel function of the first kind are
        computed.
    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.array(z)
    return jnp.where(jnp.abs(z) < 5.0, j0_small(jnp.abs(z)), j0_large(jnp.abs(z)))


@j0_.defjvp
def j0_jvp(primals, tangents):
    (m,) = primals
    (m_dot,) = tangents
    # Derivative of the complete elliptic integral of the first kind with respect to m
    return j0_(m), -1.0 * m_dot * j1_(m)


def j1_small(x):
    """
    Implementation of J1 for x < 5
    """
    z = x * x
    w = jnp.polyval(jnp.array(RP1), z) / jnp.polyval(jnp.array(RQ1), z)
    w = w * x * (z - jnp.array(Z1)) * (z - jnp.array(Z2))
    return w


def j1_large(x):
    """
    Implementation of J1 for x > 5
    """
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(jnp.array(PP1), z) / jnp.polyval(jnp.array(PQ1), z)
    q = jnp.polyval(jnp.array(QP1), z) / jnp.polyval(jnp.array(QQ1), z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@custom_jvp
def j1_(z):
    """
    Bessel function of the first kind of order one and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.
    Reference:
    Cephes mathematical library.
    Args:
        x: The sampling point(s) at which the Bessel function of the first kind are
        computed.
    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.array(z)

    return jnp.sign(z) * jnp.where(
        jnp.abs(z) < 5.0, j1_small(jnp.abs(z)), j1_large(jnp.abs(z))
    )


@j1_.defjvp
def j1_jvp(primals, tangents):
    (m,) = primals
    (m_dot,) = tangents
    primal_out = j1_(m)
    tangent_out = j0_(m) * m_dot - j1_(m) * m_dot / m
    # tangent_out = lax.cond(m == 0, lambda _: 0.0, lambda _: tangent_out - j1(m) * m_dot / m, operand=None)
    return primal_out, tangent_out


j0 = jit(vmap(j0_, in_axes=(0,)))
j1 = jit(vmap(j1_, in_axes=(0,)))


def j2_(x):
    return (2 / x) * j1_(x) - j0_(x)


j2 = jit(vmap(j2_, in_axes=(0,)))


def j1p5_(x):
    return jnp.sqrt(2 / (jnp.pi * x)) * (jnp.sin(x) / x - jnp.cos(x))


j1p5 = jit(vmap(j1p5_, in_axes=(0,)))
