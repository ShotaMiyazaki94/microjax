"""Binary/triple CPU counterparts of the GPU limb-darkened Jacobian examples."""

import argparse
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

from microjax import make_binary_lightcurve, make_triple_lightcurve
from microjax.point_source import critical_and_caustic_curves

PARAMETER_NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho", "q3", "r3", "psi")
PARAMETER_LABELS = (r"t_0", r"t_E", r"u_0", "q", "s", r"\alpha", r"\rho", r"q_3", r"r_3", r"\psi")


def trajectory(params, times):
    t0, t_e, u0, _, _, alpha, _ = params[:7]
    tau = (times - t0) / t_e
    return (-u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)) + 1j * (u0 * jnp.cos(alpha) + tau * jnp.sin(alpha))


def save_plot(output, times, values, jacobian, params, *, u1=0.5):
    """Match the corresponding GPU example's panels and lens/trajectory inset."""
    _, _, _, q, s, _, rho = params[:7]
    triple = params.size == 10
    lens_parameters = dict(q=q, s=s)
    if triple:
        lens_parameters.update(q3=params[7], r3=params[8], psi=params[9])
    positions = np.asarray(trajectory(params, times))
    critical, caustic = critical_and_caustic_curves(
        npts=1000 if triple else 500, nlenses=3 if triple else 2, **lens_parameters
    )
    fig, axes = plt.subplots(
        params.size + 1,
        1,
        figsize=(12, 13) if triple else (12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [8] + [1.2] * 10 if triple else [7] + [1] * 7},
    )
    axes[0].plot(times, values, color="black")
    axes[0].set_ylabel("Magnification")
    if triple:
        axes[0].set_title(rf"Linear limb darkening: $u_1={u1:g}$ (held fixed)")
    inset = inset_axes(
        axes[0],
        width="70%" if triple else "68%",
        height="70%" if triple else "68%",
        bbox_transform=axes[0].transAxes,
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.9) if triple else (0.06, 0.05, 0.9, 0.9),
    )
    for curve in caustic:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="red", lw=0.7)
    for curve in critical:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="green", lw=0.7)
    circles = [plt.Circle((w.real, w.imag), radius=rho, fill=False) for w in positions]
    inset.add_collection(
        PatchCollection(circles, match_original=True, alpha=0.05 if triple else 0.08, edgecolor="blue", linewidth=0.5)
    )
    inset.plot(-q / (1 + q) * s, 0.0, "x", color="black", ms=3)
    inset.plot(1 / (1 + q) * s, 0.0, "x", color="black", ms=3)
    if triple:
        third = params[8] * np.exp(1j * params[9]) + 0.5 * s * (1 - q) / (1 + q)
        inset.plot(third.real, third.imag, "x", color="black", ms=3)
    inset.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    inset.set_aspect("equal")
    for index, label in enumerate(PARAMETER_LABELS[: params.size]):
        axes[index + 1].plot(times, jacobian[:, index])
        alignment = dict(rotation=0, ha="right", va="center") if triple else {}
        axes[index + 1].set_ylabel(r"$\partial A / \partial " + label + "$", **alignment)
    axes[-1].set_xlabel("Time (day)")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", choices=("binary", "triple"), default="binary")
    triple = parser.parse_args().lens == "triple"
    with jax.default_device(jax.devices("cpu")[0]):
        # Match each GPU limb-darkened example's event and observation times.
        params = (
            [0.0, 10.0, 0.1, 0.1, 1.1, np.deg2rad(50.0), 0.01, 0.01, np.abs(0.3 + 1.2j), np.angle(0.3 + 1.2j)]
            if triple
            else [0.0, 10.0, 0.1, 0.001, 1.0, np.deg2rad(50.0), 0.005]
        )
        params = jnp.asarray(params)
        times = jnp.linspace(-5.0, 10.0, 1000 if triple else 500)
        u1 = 0.5  # Fixed; the example differentiates physical and trajectory parameters.

        make_lightcurve = make_triple_lightcurve if triple else make_binary_lightcurve
        magnification = make_lightcurve(times, backend="cpu")
        names = PARAMETER_NAMES[:params.size]

        def light_curve(p):
            values = magnification({**dict(zip(names, p)), "u1": u1})
            # Keep the value as auxiliary data to obtain A and its Jacobian together.
            return values, values

        jacobian, values = jax.jit(jax.jacfwd(light_curve, has_aux=True))(params)
        times, values, jacobian, params = map(np.asarray, (times, values, jacobian, params))

    if not (np.isfinite(values).all() and np.isfinite(jacobian).all()):
        raise RuntimeError("The example produced non-finite values or derivatives.")

    output = Path(__file__).resolve().parents[2] / "results" / "cpu" / "example"
    if triple:
        output /= "triple"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output / "lightcurve.csv",
        np.column_stack((times, values)),
        delimiter=",",
        header="time,magnification",
        comments="",
    )

    np.save(output / "jacobian_forward.npy", jacobian)
    save_plot(output / "lightcurve.png", times, values, jacobian, params, u1=u1)
    print(f"Saved lightcurve.png, lightcurve.csv and jacobian_forward.npy to {output}")


if __name__ == "__main__":
    main()
