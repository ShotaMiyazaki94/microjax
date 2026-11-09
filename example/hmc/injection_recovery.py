"""Generate a binary-lens event and recover its parameters with NumPyro NUTS."""

from functools import partial
from importlib.metadata import version
import json
import shutil
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size
from numpyro.distributions.transforms import LowerCholeskyAffine
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import minimize
from scipy.special import expit, logit

NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho")
LOG_INDICES = np.array([1, 3, 4, 6])
CPU_WORKERS = 6
NUM_WARMUP = 500
# Total retained draws: one chain, with time-parallel model evaluations.
NUM_SAMPLES = 1000
SAMPLER_SEED = 42
MAP_MAXITER = 300

# Times are in days, alpha in radians.
TRUTH = np.array([0.0, 10.0, 0.1, 0.001, 1.0, np.deg2rad(50.0), 0.005])
FLUX_TRUTH = np.array([1.0, 0.2])
# A rough event solution starts the local fit; it does not define the prior.
INITIAL = np.array([0.5, 12.0, 0.12, 0.002, 1.05, np.deg2rad(45.0), 0.01])
# Uniform in physical coordinates, or log-uniform at LOG_INDICES.
# Coarse bounds for this weeks-long event, independent of TRUTH and INITIAL.
PRIOR_BOUNDS = np.array(
    [
        [-1.0, 1.0],  # t0 [day]
        [1.0, 100.0],  # tE [day]
        [0.0, 1.0],  # u0
        [1e-4, 1.0],  # q
        [0.5, 2.0],  # s
        [-np.pi, np.pi],  # alpha [radian]
        [1e-4, 0.05],  # rho
    ]
)
# Independent zero-mean Gaussian flux priors, in the chosen flux units.
FLUX_PRIOR_SCALE = 10.0
COORDINATE_BOUNDS = PRIOR_BOUNDS.copy()
COORDINATE_BOUNDS[LOG_INDICES] = np.log(COORDINATE_BOUNDS[LOG_INDICES])
U1 = 0.5
DATA_SEED = 20261109
RELATIVE_NOISE = 0.03
MAX_TREE_DEPTH = 10
POSTERIOR_CURVES = 64


def physical_parameters(unit):
    """Map independent Uniform(0, 1) variables to the specified physical priors."""
    lower, upper = jnp.asarray(COORDINATE_BOUNDS).T
    theta = lower + (upper - lower) * unit
    return theta.at[..., LOG_INDICES].set(jnp.exp(theta[..., LOG_INDICES]))


def make_parallel_magnification(lightcurve):
    """Distribute observation times across CPUs while preserving their order."""
    parallel = jax.pmap(partial(lightcurve, u1=U1), in_axes=(None, 0))

    @jax.jit
    def evaluate(parameters, times):
        padded = jnp.pad(times, (0, (-times.size) % CPU_WORKERS), mode="edge")
        # Interleave times so the short, expensive feature is shared by all CPUs.
        blocks = padded.reshape(-1, CPU_WORKERS).T
        return parallel(parameters, blocks).T.reshape(-1)[: times.size]

    return evaluate


def flux_terms(magnification, flux, error):
    """Analytic Gaussian marginal likelihood and conditional flux posterior."""
    # Import after main configures the CPU devices.
    from microjax.likelihood import nll_ulens

    design = jnp.column_stack((magnification, jnp.ones_like(magnification)))
    variance = FLUX_PRIOR_SCALE**2
    precision = design.T @ (design / error[:, None] ** 2) + jnp.eye(2) / variance
    covariance = jnp.linalg.inv(precision)
    mean = covariance @ (design.T @ (flux / error**2))
    residual = (design @ mean - flux) / error
    # nll_ulens omits only the data-independent normalizing constant.
    nll = nll_ulens(flux, design, error**2, variance, variance) + 0.5 * flux.size * jnp.log(2 * jnp.pi)
    return residual, nll, mean, covariance


def model(times, flux, error, *, forward_model, to_logit):
    # logit(Uniform(0, 1)) is Logistic(0, 1). Transform this same prior into
    # whitened coordinates, including the affine Jacobian automatically.
    prior = dist.TransformedDistribution(dist.Logistic(0.0, 1.0).expand((len(NAMES),)).to_event(1), to_logit.inv)
    whitened = numpyro.sample("whitened", prior)
    parameters = physical_parameters(jax.nn.sigmoid(to_logit(whitened)))
    magnification = forward_model(parameters, times)
    _, nll, _, _ = flux_terms(magnification, flux, error)
    numpyro.factor("flux", -nll)


def initialize_hmc(initial, times, flux, error, *, forward_model, maxiter):
    """Fit locally, then estimate covariance in unbounded logit coordinates."""
    coordinates = np.array(initial, dtype=float)
    coordinates[LOG_INDICES] = np.log(coordinates[LOG_INDICES])
    lower, upper = COORDINATE_BOUNDS.T
    unit = (coordinates - lower) / (upper - lower)
    if not np.all((unit > 0) & (unit < 1)):
        raise ValueError("INITIAL must lie strictly inside PRIOR_BOUNDS.")

    def fit_terms(z):
        parameters = physical_parameters(jax.nn.sigmoid(z))
        magnification = forward_model(parameters, times)
        residual, nll, _, _ = flux_terms(magnification, flux, error)
        terms = jnp.concatenate((residual, jnp.atleast_1d(nll)))
        return terms, terms

    # has_aux shares the forward evaluation between residuals and their Jacobian.
    terms_jacobian = jax.jit(jax.jacfwd(fit_terms, has_aux=True))

    def objective(z):
        jacobian, terms = map(np.asarray, terms_jacobian(jnp.asarray(z)))
        # Fit in z = logit(unit). Include the transform's Jacobian:
        # -log p(z) = softplus(z) + softplus(-z), for a uniform physical/log prior.
        value = terms[-1] + np.sum(np.logaddexp(0, z) + np.logaddexp(0, -z))
        gradient = jacobian[-1] + 2 * expit(z) - 1
        return float(value), gradient

    fit = minimize(objective, logit(unit), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    jacobian, terms = map(np.asarray, terms_jacobian(jnp.asarray(fit.x)))
    residual, jacobian = terms[:-1], jacobian[:-1]
    if not np.isfinite(fit.fun) or not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(residual)):
        raise RuntimeError("The local fit returned non-finite values; check INITIAL and the data.")

    unit_map = expit(fit.x)
    # Profile-residual Gauss–Newton estimate plus transformed-prior curvature.
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


def save_model_curves(output, forward_model):
    """Evaluate saved posterior draws on a dense plotting grid and at the data."""
    config = json.loads((output / "run.json").read_text())
    with np.load(output / "data.npz") as saved:
        times, flux, error = (saved[key] for key in ("times", "flux", "error"))
    with np.load(output / "samples.npz") as saved:
        parameters = np.column_stack([saved[name].ravel() for name in config["parameter_names"]])
    # Fine spacing around the peak; keep the observation times to condition fluxes.
    grid = np.unique(np.r_[times, np.linspace(times[0], times[-1], 1201), np.linspace(-5.0, 5.0, 4001)])
    observed = np.searchsorted(grid, times)
    grid = jnp.asarray(grid)
    rng = np.random.default_rng(config["sampler_seed"] + 1)
    indices = rng.choice(len(parameters), size=min(POSTERIOR_CURVES, len(parameters)), replace=False)
    curves, flux_draws = [], []
    injected = config["flux_truth"][0] * forward_model(jnp.asarray(config["truth"]), grid) + config["flux_truth"][1]
    print(f"Evaluating {len(indices)} posterior curves at {grid.size} plotting times.", flush=True)
    for i in indices:
        magnification = forward_model(jnp.asarray(parameters[i]), grid)
        # Condition the fluxes on the observations, not the extra plotting times.
        _, _, mean, covariance = flux_terms(magnification[observed], jnp.asarray(flux), jnp.asarray(error))
        draw = rng.multivariate_normal(np.asarray(mean), np.asarray(covariance))
        curves.append(np.asarray(magnification) * draw[0] + draw[1])
        flux_draws.append(draw)
    curves = np.stack(curves)
    np.savez_compressed(
        output / "curves.npz",
        times=np.asarray(grid),
        model_flux=curves,
        injected_flux=np.asarray(injected),
        flux_draws=np.stack(flux_draws),
        sample_indices=indices,
    )


def main():
    # Configure CPU devices before importing microJAX, which creates JAX arrays.
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(CPU_WORKERS)

    from microjax import make_binary_lightcurve
    from plot import OUTPUT, plot_results

    devices = jax.local_devices(backend="cpu")
    device = devices[0]
    output = OUTPUT
    # A rerun replaces this example's outputs, including any incomplete run.
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    config = {
        "backend": "cpu",
        "devices": [str(d) for d in devices],
        "parallelism": "observation_times",
        "cpu_workers": CPU_WORKERS,
        "float64": True,
        "warmup": NUM_WARMUP,
        "samples_per_chain": NUM_SAMPLES,
        "samples_total": NUM_SAMPLES,
        "chains": 1,
        "sampler_seed": SAMPLER_SEED,
        "data_seed": DATA_SEED,
        "u1": U1,
        "relative_noise": RELATIVE_NOISE,
        "max_tree_depth": MAX_TREE_DEPTH,
        "target_accept_prob": 0.9,
        "sampling_coordinates": "map_whitened",
        "adapt_mass_matrix": True,
        "forward_mode_differentiation": True,
        "parameter_names": NAMES,
        "truth": TRUTH.tolist(),
        "flux_truth": FLUX_TRUTH.tolist(),
        "flux_treatment": "marginalized_gaussian",
        "flux_priors": {
            name: {"distribution": "normal", "mean": 0.0, "standard_deviation": FLUX_PRIOR_SCALE}
            for name in ("fs", "fb")
        },
        "observations": {"count": 180, "start_day": -30.0, "end_day": 30.0, "spacing": "uniform"},
        "initial_parameters": INITIAL.tolist(),
        "priors": {
            name: {"distribution": "log_uniform" if i in LOG_INDICES else "uniform", "bounds": bounds.tolist()}
            for i, (name, bounds) in enumerate(zip(NAMES, PRIOR_BOUNDS))
        },
        "map_maxiter": MAP_MAXITER,
        "versions": {name: version(name) for name in ("microjaxx", "jax", "jaxlib", "numpyro", "numpy", "scipy")},
    }
    (output / "run.json").write_text(json.dumps(config, indent=2) + "\n")

    with jax.default_device(device):
        lightcurve = make_binary_lightcurve(backend="cpu")
        forward_model = make_parallel_magnification(lightcurve)
        times = jnp.linspace(-30.0, 30.0, 180)
        print("Generating the mock light curve (the first call compiles the model).", flush=True)
        injected_flux = FLUX_TRUTH[0] * np.asarray(forward_model(jnp.asarray(TRUTH), times)) + FLUX_TRUTH[1]
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
            INITIAL, times, jnp.asarray(flux), jnp.asarray(error), forward_model=forward_model, maxiter=MAP_MAXITER
        )
        initialization["seconds_including_compilation"] = time.perf_counter() - start
        config["initialization"] = initialization
        (output / "run.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        cholesky = np.linalg.cholesky(covariance)
        # z = map_z + L @ whitened makes the local covariance approximately I.
        to_logit = LowerCholeskyAffine(jnp.asarray(map_z), jnp.asarray(cholesky))
        initial_whitened = jnp.asarray(0.1 * np.random.default_rng(SAMPLER_SEED).normal(size=len(NAMES)))
        np.savez_compressed(
            output / "initialization.npz", map_z=map_z, cholesky=cholesky, initial_whitened=np.asarray(initial_whitened)
        )
        print(f"Local fit: {initialization['message']} (chi-squared = {initialization['chi_squared']:.1f})", flush=True)

        kernel = NUTS(
            partial(model, forward_model=forward_model, to_logit=to_logit),
            forward_mode_differentiation=True,
            dense_mass=True,
            inverse_mass_matrix=jnp.eye(len(NAMES)),
            adapt_mass_matrix=True,
            target_accept_prob=0.9,
            max_tree_depth=MAX_TREE_DEPTH,
            init_strategy=init_to_value(values={"whitened": initial_whitened}),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            num_chains=1,
            progress_bar=True,
        )
        print(
            f"Starting NUTS: {NUM_WARMUP} warm-up + {NUM_SAMPLES} retained draws, one chain, {CPU_WORKERS} CPU workers.",
            flush=True,
        )
        start = time.perf_counter()
        mcmc.run(
            jax.random.PRNGKey(SAMPLER_SEED),
            times,
            jnp.asarray(flux),
            jnp.asarray(error),
            init_params={"whitened": initial_whitened},
            extra_fields=("diverging", "num_steps", "accept_prob"),
        )
        whitened = mcmc.get_samples(group_by_chain=True)["whitened"]
        physical = np.asarray(physical_parameters(jax.nn.sigmoid(to_logit(whitened))))
        elapsed = time.perf_counter() - start
        extra = {name: np.asarray(value) for name, value in mcmc.get_extra_fields(group_by_chain=True).items()}
        samples = {name: physical[..., i] for i, name in enumerate(NAMES)}
        # Save the expensive part before computing curves or making figures.
        np.savez_compressed(output / "samples.npz", **samples)
        np.savez_compressed(
            output / "sampler.npz",
            whitened=np.asarray(whitened),
            inverse_mass_matrix=np.asarray(mcmc.last_state.adapt_state.inverse_mass_matrix[("whitened",)]),
            step_size=np.asarray(mcmc.last_state.adapt_state.step_size),
            **extra,
        )

        parameters = {}
        for i, name in enumerate(NAMES):
            lo95, lo68, median, hi68, hi95 = np.quantile(samples[name], [0.025, 0.16, 0.5, 0.84, 0.975])
            parameters[name] = {
                "injected": float(TRUTH[i]),
                "median": float(median),
                "ci68": [float(lo68), float(hi68)],
                "ci95": [float(lo95), float(hi95)],
                "effective_samples": finite_float(effective_sample_size(samples[name])),
            }
        report = {
            "sampling_seconds_including_compilation": elapsed,
            "parameters": parameters,
            "divergences": int(extra["diverging"].sum()),
            "step_limit_hits": int((extra["num_steps"] >= 2**MAX_TREE_DEPTH - 1).sum()),
            "mean_acceptance": float(extra["accept_prob"].mean()),
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Posterior samples and summary saved to {output}", flush=True)

        save_model_curves(output, forward_model)

    plot_results()
    print(f"Done. Share {output / 'summary.json'} and the PNG figures together with run.json.")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python example/hmc/injection_recovery.py")
    main()
