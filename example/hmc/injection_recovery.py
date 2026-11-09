"""Generate a binary-lens event and recover its parameters with NumPyro NUTS."""

import argparse
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_uniform

from microjax import make_binary_lightcurve


NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho", "fs", "fb")
LOG_INDICES = np.array([1, 3, 4, 6, 7])

# Times are in days, alpha in radians. The last two entries are fluxes.
TRUTH = np.array([0.0, 10.0, 0.1, 0.001, 1.0, np.deg2rad(50.0), 0.005, 1.0, 0.2])
PRIOR_CENTER = np.array([0.1, 10.5, 0.11, 0.0012, 1.02, np.deg2rad(49.0), 0.006, 1.05, 0.15])
# Standard deviations in physical space, or natural-log space at LOG_INDICES.
PRIOR_SCALE = np.array([1.0, 0.3, 0.05, 0.8, 0.15, 0.3, 0.7, 0.3, 0.5])
PRIOR_LOCATION = PRIOR_CENTER.copy()
PRIOR_LOCATION[LOG_INDICES] = np.log(PRIOR_LOCATION[LOG_INDICES])
U1 = 0.5
DATA_SEED = 20261109
RELATIVE_NOISE = 0.01
MAX_TREE_DEPTH = 8
POSTERIOR_CURVES = 64


def physical_parameters(z):
    """Independent standard normals give normal and log-normal physical priors."""
    theta = jnp.asarray(PRIOR_LOCATION) + jnp.asarray(PRIOR_SCALE) * z
    return theta.at[..., LOG_INDICES].set(jnp.exp(theta[..., LOG_INDICES]))


def flux_model(parameters, times, *, lightcurve):
    magnification = lightcurve(parameters[:7], times, u1=U1)
    return parameters[7] * magnification + parameters[8]


def model(times, flux, error, *, forward_model):
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand((len(NAMES),)).to_event(1))
    parameters = physical_parameters(z)
    mean = forward_model(parameters, times)
    numpyro.sample("flux", dist.Normal(mean, error).to_event(1), obs=flux)


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500, help="Retained draws per chain")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed; the mock-data seed stays fixed")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hmc/binary"))
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 4 or args.chains < 1:
        parser.error("Use at least one warm-up step, four retained draws, and one chain.")

    device = jax.devices(args.backend)[0]
    output = args.output_dir
    # Keep separate runs separate, including an interrupted run's saved data.
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "backend": args.backend,
        "device": str(device),
        "float64": True,
        "warmup": args.warmup,
        "samples_per_chain": args.samples,
        "chains": args.chains,
        "sampler_seed": args.seed,
        "data_seed": DATA_SEED,
        "u1": U1,
        "relative_noise": RELATIVE_NOISE,
        "max_tree_depth": MAX_TREE_DEPTH,
        "target_accept_prob": 0.9,
        "forward_mode_differentiation": True,
        "parameter_names": NAMES,
        "truth": TRUTH.tolist(),
        "prior_center": PRIOR_CENTER.tolist(),
        "prior_scale": PRIOR_SCALE.tolist(),
        "log_prior_parameters": [NAMES[i] for i in LOG_INDICES],
        "versions": {name: version(name) for name in ("microjaxx", "jax", "jaxlib", "numpyro", "numpy")},
    }
    (output / "run.json").write_text(json.dumps(config, indent=2) + "\n")

    with jax.default_device(device):
        lightcurve = make_binary_lightcurve(backend=args.backend)
        forward_model = jax.jit(partial(flux_model, lightcurve=lightcurve))
        # Sample the baseline, broad peak, and short feature near day -0.65.
        times = jnp.asarray(
            np.unique(np.r_[np.linspace(-30, 30, 80), np.linspace(-5, 5, 40), np.linspace(-0.9, -0.4, 60)])
        )
        print("Generating the mock light curve (the first call compiles the model).", flush=True)
        injected_flux = np.asarray(forward_model(jnp.asarray(TRUTH), times))
        if not np.all(np.isfinite(injected_flux) & (injected_flux > 0)):
            raise RuntimeError("The injected light curve contains invalid fluxes.")
        error = RELATIVE_NOISE * injected_flux
        flux = injected_flux + np.random.default_rng(DATA_SEED).normal(size=error.size) * error
        np.savez_compressed(
            output / "data.npz",
            times=np.asarray(times),
            flux=flux,
            error=error,
            injected_flux=injected_flux,
            truth=TRUTH,
        )

        kernel = NUTS(
            partial(model, forward_model=forward_model),
            forward_mode_differentiation=True,
            dense_mass=True,
            target_accept_prob=0.9,
            max_tree_depth=MAX_TREE_DEPTH,
            init_strategy=init_to_uniform(radius=0.1),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
            chain_method="sequential",
            progress_bar=True,
        )
        print("Starting NUTS; chains run sequentially and compilation precedes the first progress update.", flush=True)
        start = time.perf_counter()
        mcmc.run(
            jax.random.PRNGKey(args.seed),
            times,
            jnp.asarray(flux),
            jnp.asarray(error),
            extra_fields=("diverging", "num_steps", "accept_prob"),
        )
        z = np.asarray(mcmc.get_samples(group_by_chain=True)["z"])
        elapsed = time.perf_counter() - start
        extra = {name: np.asarray(value) for name, value in mcmc.get_extra_fields(group_by_chain=True).items()}
        physical = np.asarray(physical_parameters(jnp.asarray(z)))
        samples = {name: physical[..., i] for i, name in enumerate(NAMES)}
        # Save the expensive part before computing curves or making figures.
        np.savez_compressed(output / "samples.npz", **samples)
        np.savez_compressed(output / "sampler.npz", z=z, **extra)

        diagnostics = summary(samples, group_by_chain=True)
        parameters = {}
        for i, name in enumerate(NAMES):
            lo95, lo68, median, hi68, hi95 = np.quantile(samples[name], [0.025, 0.16, 0.5, 0.84, 0.975])
            parameters[name] = {
                "injected": float(TRUTH[i]),
                "median": float(median),
                "ci68": [float(lo68), float(hi68)],
                "ci95": [float(lo95), float(hi95)],
                "effective_samples": finite_float(diagnostics[name]["n_eff"]),
                "split_r_hat": finite_float(diagnostics[name]["r_hat"]),
            }
        report = {
            "sampling_seconds_including_compilation": elapsed,
            "parameters": parameters,
            "divergences_per_chain": extra["diverging"].sum(axis=1).tolist(),
            "step_limit_hits_per_chain": (extra["num_steps"] >= 2**MAX_TREE_DEPTH - 1).sum(axis=1).tolist(),
            "mean_acceptance_per_chain": extra["accept_prob"].mean(axis=1).tolist(),
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Posterior samples and summary saved to {output}", flush=True)

        # Draw a small, reproducible subset for the uncertainty in the model curve.
        flattened = physical.reshape(-1, len(NAMES))
        indices = np.random.default_rng(args.seed + 1).choice(
            len(flattened), size=min(POSTERIOR_CURVES, len(flattened)), replace=False
        )
        curves = np.stack([np.asarray(forward_model(jnp.asarray(flattened[i]), times)) for i in indices])
        np.savez_compressed(output / "curves.npz", model_flux=curves)

    from plot import plot_results

    plot_results(output)
    print(f"Done. Share {output / 'summary.json'} and the PNG figures together with run.json.")


if __name__ == "__main__":
    main()
