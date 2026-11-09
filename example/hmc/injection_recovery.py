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
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import minimize
from scipy.special import expit, logit

from microjax import make_binary_lightcurve


NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho", "fs", "fb")
LOG_INDICES = np.array([1, 3, 4, 6])

# Times are in days, alpha in radians. The last two entries are fluxes.
TRUTH = np.array([0.0, 10.0, 0.1, 0.001, 1.0, np.deg2rad(50.0), 0.005, 1.0, 0.2])
# A rough event solution starts the local fit; it does not define the prior.
INITIAL = np.array([0.5, 12.0, 0.12, 0.002, 1.05, np.deg2rad(45.0), 0.01, 1.0, 0.0])
# Uniform in physical coordinates, or log-uniform at LOG_INDICES.
PRIOR_BOUNDS = np.array(
    [
        [-30.0, 30.0],  # t0 [day]
        [0.1, 300.0],  # tE [day]
        [-1.0, 1.0],  # u0
        [1e-6, 1.0],  # q
        [0.2, 5.0],  # s
        [-np.pi, np.pi],  # alpha [radian]
        [1e-5, 0.1],  # rho
        [0.0, 10.0],  # source flux
        [-5.0, 5.0],  # blend flux
    ]
)
COORDINATE_BOUNDS = PRIOR_BOUNDS.copy()
COORDINATE_BOUNDS[LOG_INDICES] = np.log(COORDINATE_BOUNDS[LOG_INDICES])
U1 = 0.5
DATA_SEED = 20261109
RELATIVE_NOISE = 0.01
MAX_TREE_DEPTH = 8
POSTERIOR_CURVES = 64


def physical_parameters(unit):
    """Map independent Uniform(0, 1) variables to the specified physical priors."""
    lower, upper = jnp.asarray(COORDINATE_BOUNDS).T
    theta = lower + (upper - lower) * unit
    return theta.at[..., LOG_INDICES].set(jnp.exp(theta[..., LOG_INDICES]))


def flux_model(parameters, times, *, lightcurve):
    magnification = lightcurve(parameters[:7], times, u1=U1)
    return parameters[7] * magnification + parameters[8]


def model(times, flux, error, *, forward_model):
    unit = numpyro.sample("unit", dist.Uniform(0.0, 1.0).expand((len(NAMES),)).to_event(1))
    parameters = physical_parameters(unit)
    mean = forward_model(parameters, times)
    numpyro.sample("flux", dist.Normal(mean, error).to_event(1), obs=flux)


def initialize_hmc(initial, times, flux, error, *, forward_model, maxiter):
    """Fit locally, then estimate covariance in NumPyro's unconstrained coordinates."""
    coordinates = np.array(initial, dtype=float)
    coordinates[LOG_INDICES] = np.log(coordinates[LOG_INDICES])
    lower, upper = COORDINATE_BOUNDS.T
    unit = (coordinates - lower) / (upper - lower)
    if not np.all((unit > 0) & (unit < 1)):
        raise ValueError("INITIAL must lie strictly inside PRIOR_BOUNDS.")

    def residuals(z):
        parameters = physical_parameters(jax.nn.sigmoid(z))
        residual = (forward_model(parameters, times) - flux) / error
        return residual, residual

    # has_aux shares the forward evaluation between residuals and their Jacobian.
    residual_jacobian = jax.jit(jax.jacfwd(residuals, has_aux=True))

    def objective(z):
        jacobian, residual = map(np.asarray, residual_jacobian(jnp.asarray(z)))
        # NumPyro samples z = logit(unit). Include the transform's Jacobian:
        # -log p(z) = softplus(z) + softplus(-z), for a uniform physical/log prior.
        value = 0.5 * (residual @ residual) + np.sum(np.logaddexp(0, z) + np.logaddexp(0, -z))
        gradient = jacobian.T @ residual + 2 * expit(z) - 1
        return float(value), gradient

    fit = minimize(objective, logit(unit), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    jacobian, residual = map(np.asarray, residual_jacobian(jnp.asarray(fit.x)))
    if not np.isfinite(fit.fun) or not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(residual)):
        raise RuntimeError("The local fit returned non-finite values; check INITIAL and the data.")

    unit_map = expit(fit.x)
    # Gaussian-likelihood Fisher information plus the transformed-prior curvature.
    precision = jacobian.T @ jacobian + np.diag(2 * unit_map * (1 - unit_map))
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    floor = max(1e-8, 1e-10 * eigenvalues[-1])
    covariance = (eigenvectors / np.maximum(eigenvalues, floor)) @ eigenvectors.T
    info = {
        "converged": bool(fit.success),
        "message": str(fit.message),
        "iterations": int(fit.nit),
        "negative_log_posterior": float(fit.fun),
        "chi_squared": float(residual @ residual),
        "parameters": dict(zip(NAMES, np.asarray(physical_parameters(jnp.asarray(unit_map))).tolist())),
        "precision_eigenvalue_floor": float(floor),
    }
    return fit.x, covariance, info


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500, help="Retained draws per chain")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--map-maxiter", type=int, default=300, help="Maximum local optimization iterations")
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed; the mock-data seed stays fixed")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hmc/binary"))
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 4 or args.chains < 1 or args.map_maxiter < 1:
        parser.error("Use positive warm-up, chain and optimization counts, and at least four retained draws.")

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
        "initial_parameters": INITIAL.tolist(),
        "priors": {
            name: {"distribution": "log_uniform" if i in LOG_INDICES else "uniform", "bounds": bounds.tolist()}
            for i, (name, bounds) in enumerate(zip(NAMES, PRIOR_BOUNDS))
        },
        "map_maxiter": args.map_maxiter,
        "versions": {name: version(name) for name in ("microjaxx", "jax", "jaxlib", "numpyro", "numpy", "scipy")},
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

        print("Fitting from INITIAL and estimating the local covariance (forward-mode compilation first).", flush=True)
        start = time.perf_counter()
        map_z, covariance, initialization = initialize_hmc(
            INITIAL, times, jnp.asarray(flux), jnp.asarray(error), forward_model=forward_model, maxiter=args.map_maxiter
        )
        initialization["seconds_including_compilation"] = time.perf_counter() - start
        config["initialization"] = initialization
        (output / "run.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        # Small independent displacements in the fitted covariance, not a narrower prior.
        chain_z = (
            map_z
            + 0.1
            * np.random.default_rng(args.seed).normal(size=(args.chains, len(NAMES)))
            @ np.linalg.cholesky(covariance).T
        )
        np.savez_compressed(output / "initialization.npz", map_z=map_z, inverse_mass_matrix=covariance, chain_z=chain_z)
        print(f"Local fit: {initialization['message']} (chi-squared = {initialization['chi_squared']:.1f})", flush=True)

        kernel = NUTS(
            partial(model, forward_model=forward_model),
            forward_mode_differentiation=True,
            dense_mass=True,
            # NumPyro's inverse_mass_matrix takes a covariance, not a precision.
            inverse_mass_matrix=jnp.asarray(covariance),
            target_accept_prob=0.9,
            max_tree_depth=MAX_TREE_DEPTH,
            init_strategy=init_to_value(values={"unit": jax.nn.sigmoid(jnp.asarray(map_z))}),
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
            init_params={"unit": jnp.asarray(chain_z[0] if args.chains == 1 else chain_z)},
            extra_fields=("diverging", "num_steps", "accept_prob"),
        )
        unit = np.asarray(mcmc.get_samples(group_by_chain=True)["unit"])
        elapsed = time.perf_counter() - start
        extra = {name: np.asarray(value) for name, value in mcmc.get_extra_fields(group_by_chain=True).items()}
        physical = np.asarray(physical_parameters(jnp.asarray(unit)))
        samples = {name: physical[..., i] for i, name in enumerate(NAMES)}
        # Save the expensive part before computing curves or making figures.
        np.savez_compressed(output / "samples.npz", **samples)
        np.savez_compressed(output / "sampler.npz", unit=unit, **extra)

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
