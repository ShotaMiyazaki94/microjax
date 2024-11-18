import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, lax

from microjax.inverse_ray.extended_source import mag_simple2
from microjax.point_source import critical_and_caustic_curves
import MulensModel as mm

def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    e1 = 1 / (1 + q)
    e2 = 1 - e1
    bl = mm.BinaryLens(e1, e2, s)
    return bl.vbbl_magnification(
        w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1
    )

def mag_binary(w_points, rho, s, q, r_resolution=200, th_resolution=200):
    def body_fn(_, w):
        mag = mag_simple2(w, rho, s=s, q=q, 
                          r_resolution=r_resolution, 
                          th_resolution=th_resolution,
                          Nlimb=500,)
        return 0, mag
    _, mags = lax.scan(body_fn, 0, w_points)
    return mags

# 設定
s, q = 1.0, 0.1
npts = 100
critical_curves, caustic_curves = critical_and_caustic_curves(
    npts=npts, nlenses=2, s=s, q=q
)
caustic_curves = caustic_curves.reshape(-1)

acc_vbb = 1e-05
r_resolution = 500
th_resolution = 500

rho_list = [1e-01, 1e-02, 1e-03, 1e-04]
results = []  # 結果を保存するリスト

for rho in rho_list:
    print(f"Processing rho = {rho}")

    # ランダムなテストポイントを生成
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-jnp.pi, maxval=jnp.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=rho)
    w_test = caustic_curves + r * jnp.exp(1j * phi)

    # Magnifications を計算
    mags_vbb = jnp.array([mag_vbb_binary(complex(w), rho, s, q, u1=0.0, accuracy=acc_vbb)
                          for w in w_test])
    mags = mag_binary(w_test, rho, s, q, r_resolution=r_resolution, th_resolution=th_resolution)

    # relative_error を計算
    relative_error = jnp.abs((mags - mags_vbb) / mags_vbb)

    # relative_error > 0.5 を満たすデータを抽出
    indices = jnp.where(relative_error > 3e-3)[0]
    if len(indices) > 0:
        results.append({
            "rho": rho,
            "w_test": w_test[indices],
            "relative_error": relative_error[indices]
        })

# 結果を出力
for result in results:
    print(f"rho: {result['rho']}")
    print(f"w_test with relative error > 3e-3: {result['w_test']}")
    print(f"Relative error values: {result['relative_error']}")
    print("-" * 40)
