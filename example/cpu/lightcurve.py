"""Plot a limb-darkened binary-lens light curve and its source-radius derivative."""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microjax.inverse_ray import mag_binary


def main():
    with jax.default_device(jax.devices("cpu")[0]):
        tau = jnp.linspace(-0.7, 0.7, 501)  # (t - t0) / tE
        sources = tau + 0.1j
        rho, s, q, u1 = 0.02, 1.0, 0.3, 0.5

        def light_curve(radius):
            return mag_binary(sources, radius, s=s, q=q, u1=u1, backend="cpu")

        # One forward-mode pass returns both A and dA/drho; the first call compiles.
        @jax.jit
        def evaluate(radius):
            return jax.jvp(light_curve, (radius,), (jnp.ones_like(radius),))

        values, derivative = evaluate(jnp.asarray(rho))
        tau, values, derivative = map(np.asarray, (tau, values, derivative))

    if not (np.isfinite(values).all() and np.isfinite(derivative).all()):
        raise RuntimeError("The example produced non-finite values or derivatives.")

    output = Path(__file__).resolve().parents[2] / "results" / "cpu" / "example"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output / "lightcurve.csv",
        np.column_stack((tau, values, derivative)),
        delimiter=",",
        header="tau,magnification,dA_drho",
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, layout="constrained")
    axes[0].plot(tau, values, color="#2166ac")
    axes[0].set_ylabel("Magnification A")
    axes[0].set_title(f"CPU binary lens · s={s:g}, q={q:g}, ρ={rho:g}, u₁={u1:g}")
    axes[1].plot(tau, derivative, color="#b35806")
    axes[1].axhline(0, color="0.6", linewidth=0.7)
    axes[1].set_ylabel(r"Forward derivative $\partial A/\partial\rho$")
    axes[1].set_xlabel(r"$(t-t_0)/t_E$   (impact parameter $u_0=0.1$, trajectory angle $0$)")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "lightcurve.png", dpi=180)
    plt.close(fig)
    print(f"Saved lightcurve.png and lightcurve.csv to {output}")


if __name__ == "__main__":
    main()
