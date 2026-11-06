"""CPU counterpart of the GPU limb-darkened seven-parameter Jacobian example."""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from microjax.inverse_ray import mag_binary
from microjax.point_source import critical_and_caustic_curves

PARAMETER_NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho")
PARAMETER_LABELS = (r"t_0", r"t_E", r"u_0", "q", "s", r"\alpha", r"\rho")


def trajectory(params, times):
    t0, t_e, u0, _, _, alpha, _ = params
    tau = (times - t0) / t_e
    return (-u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)) + 1j * (u0 * jnp.cos(alpha) + tau * jnp.sin(alpha))


def save_plot(output, times, values, jacobian, params):
    """Match the GPU example's eight panels and lens/trajectory inset."""
    _, _, _, q, s, _, rho = params
    positions = np.asarray(trajectory(params, times))
    critical, caustic = critical_and_caustic_curves(npts=500, nlenses=2, q=q, s=s)
    fig, axes = plt.subplots(
        8,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [7] + [1] * 7},
    )
    axes[0].plot(times, values, color="black")
    axes[0].set_ylabel("Magnification")
    inset = inset_axes(
        axes[0],
        width="68%",
        height="68%",
        bbox_transform=axes[0].transAxes,
        bbox_to_anchor=(0.06, 0.05, 0.9, 0.9),
    )
    for curve in caustic:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="red", lw=0.7)
    for curve in critical:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="green", lw=0.7)
    circles = [plt.Circle((w.real, w.imag), radius=rho, fill=False) for w in positions]
    inset.add_collection(PatchCollection(circles, match_original=True, alpha=0.08, edgecolor="blue", linewidth=0.5))
    inset.plot(-q / (1 + q) * s, 0.0, "x", color="black", ms=3)
    inset.plot(1 / (1 + q) * s, 0.0, "x", color="black", ms=3)
    inset.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    inset.set_aspect("equal")
    for index, label in enumerate(PARAMETER_LABELS):
        axes[index + 1].plot(times, jacobian[:, index])
        axes[index + 1].set_ylabel(r"$\partial A / \partial " + label + "$")
    axes[-1].set_xlabel("Time (day)")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    with jax.default_device(jax.devices("cpu")[0]):
        # Match the GPU limb-darkened example's event and 500 observation times.
        params = jnp.asarray([0.0, 10.0, 0.1, 0.001, 1.0, jnp.deg2rad(50.0), 0.005])
        times = jnp.linspace(-5.0, 10.0, 500)
        u1 = 0.5  # Fixed, not an eighth differentiated parameter.

        def light_curve(p):
            values = mag_binary(trajectory(p, times), p[6], s=p[4], q=p[3], u1=u1, backend="cpu")
            # Keep the value as auxiliary data to obtain A and its Jacobian together.
            return values, values

        jacobian, values = jax.jit(jax.jacfwd(light_curve, has_aux=True))(params)
        times, values, jacobian, params = map(np.asarray, (times, values, jacobian, params))

    if not (np.isfinite(values).all() and np.isfinite(jacobian).all()):
        raise RuntimeError("The example produced non-finite values or derivatives.")

    output = Path(__file__).resolve().parents[2] / "results" / "cpu" / "example"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output / "lightcurve.csv",
        np.column_stack((times, values)),
        delimiter=",",
        header="time,magnification",
        comments="",
    )

    np.save(output / "jacobian_forward.npy", jacobian)
    save_plot(output / "lightcurve.png", times, values, jacobian, params)
    print(f"Saved lightcurve.png, lightcurve.csv and jacobian_forward.npy to {output}")


if __name__ == "__main__":
    main()
