"""Compare JAX FSPL magnification against VBBinaryLensing and plot time-domain residuals.

Outputs fspl_vs_vbbl.png showing, for a Paczynski trajectory u(t):
 - Top row: magnification curves A(t) (per rho, and per LD coefficient)
 - Bottom row: relative residuals |A_FSPL - A_VBBL|/A_VBBL (log scale)
 for uniform disk (left) and linear limb-darkening (right).
"""

import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

# Allow running from any working directory by adding repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force JAX to stay on CPU for consistent benchmarking
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    import VBBinaryLensing
except ImportError:
    print("VBBinaryLensing not installed; run `pip install VBBinaryLensing`.")
    sys.exit(0)

from microjax.fastlens import fspl_disk, fspl_ld1


def paczynski_u(t: np.ndarray, u0: float = 0.01, tE: float = 1.0) -> np.ndarray:
    """Compute dimensionless impact parameter u(t) for a single-lens trajectory."""
    tau = (t) / tE
    return np.sqrt(u0 * u0 + tau * tau)


def eval_curve(mag_fn, vbbl_fn, rhos: Sequence[float], t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return A_fspl, relerr over t for each rho; also wall time in seconds (excluding JIT compile)."""
    A = np.zeros((len(rhos), len(t)))
    R = np.zeros_like(A)
    u_t = paczynski_u(t)
    u_t_jax = jnp.array(u_t)
    # jit compile FSPL once for this mag_fn; warm-up to exclude compile time
    fspl_fn = jax.jit(lambda u, rho: mag_fn.A(u, rho))
    _ = fspl_fn(u_t_jax, rhos[0])  # warm-up
    _ = jax.device_get(_)  # ensure execution
    t0 = time.perf_counter()
    per_rho_times = []
    for i, rho in enumerate(rhos):
        vb_fn = lambda u: vbbl_fn(float(u), float(rho))
        t_rho = time.perf_counter()
        ref = np.array([vb_fn(u) for u in u_t])
        ref_time = time.perf_counter() - t_rho

        t_rho = time.perf_counter()
        got = np.array(jax.device_get(fspl_fn(u_t_jax, rho)))
        fspl_time = time.perf_counter() - t_rho

        A[i] = got
        R[i] = np.abs(got - ref) / ref
        per_rho_times.append((rho, ref_time, fspl_time))
    total = time.perf_counter() - t0
    return A, R, total, per_rho_times


def main():
    vb = VBBinaryLensing.VBBinaryLensing()
    table_path = VBBinaryLensing.__file__.replace("__init__.py", "data/ESPL.tbl")
    vb.LoadESPLTable(table_path)

    rhos = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    t = np.linspace(-2.0, 2.0, 401)  # in units of tE

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={"height_ratios": [2.0, 1.0]}, sharex="col")

    # Uniform disk
    mag_disk = fspl_disk()
    A_d, R_d, t_disk, per_rho_disk = eval_curve(mag_disk, vb.ESPLMag, rhos, t)
    ax_mag_d, ax_res_d = axes[0, 0], axes[1, 0]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, rho in enumerate(rhos):
        ls = linestyles[i % len(linestyles)]
        ax_mag_d.plot(t, A_d[i], label=f"rho={rho}", ls=ls)
        ax_res_d.semilogy(t, R_d[i], label=f"rho={rho}", ls=ls)
    ax_mag_d.set_title("Uniform disk")
    ax_mag_d.set_ylabel("A(t)")
    ax_res_d.set_ylabel("rel. error")

    # Linear limb darkening (multiple a1)
    a1_list = [0.2, 0.5, 0.8]
    colors = ["C0", "C1", "C2"]
    ax_mag_l, ax_res_l = axes[0, 1], axes[1, 1]
    max_t_ld = 0.0
    per_rho_ld_all = []
    for a1, c in zip(a1_list, colors):
        vb.a1 = a1
        mag_ld1 = fspl_ld1(a1=a1)
        A_l, R_l, t_ld, per_rho_ld = eval_curve(mag_ld1, vb.ESPLMagDark, rhos, t)
        max_t_ld = max(max_t_ld, t_ld)
        for i, rho in enumerate(rhos):
            ls = linestyles[i % len(linestyles)]
            ax_mag_l.plot(t, A_l[i], color=c, alpha=0.85, ls=ls, label=f"a1={a1}, rho={rho}")
            ax_res_l.semilogy(t, R_l[i], color=c, alpha=0.85, ls=ls)
        # store times tagged by a1
        per_rho_ld_all.extend([(a1, rho, tref, tfspl) for (rho, tref, tfspl) in per_rho_ld])
    ax_mag_l.set_title("Linear LD")

    # Residual panels formatting
    for ax in (ax_res_d, ax_res_l):
        ax.axhline(0.01, color="k", ls="--", lw=0.8, label="1%")
        ax.set_xlabel("t / tE")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=7)

    for ax in (ax_mag_d, ax_mag_l):
        ax.set_ylim(bottom=0.9)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=7)

    # Reduce overlap between legends and curves
    fig.tight_layout()
    out = Path(__file__).parent / "fspl_vs_vbbl.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    print(f"CPU timings (per rho, ms):")
    for rho, tref, tfspl in per_rho_disk:
        slowdown = (tfspl / max(tref, 1e-9))
        print(f"  disk rho={rho:.3g}: VBBL {tref*1e3:.2f}ms, FSPL {tfspl*1e3:.2f}ms, x{slowdown:.1f}")
    for a1, rho, tref, tfspl in per_rho_ld_all:
        slowdown = (tfspl / max(tref, 1e-9))
        print(f"  LD1 a1={a1:.1f} rho={rho:.3g}: VBBL {tref*1e3:.2f}ms, FSPL {tfspl*1e3:.2f}ms, x{slowdown:.1f}")
    print(f"Total: disk {t_disk*1e3:.2f}ms, LD1 (max over a1) {max_t_ld*1e3:.2f}ms")


if __name__ == "__main__":
    main()
