# GPU full-solve benchmarks

This directory is the accelerator counterpart of `benchmark_cpu/`. It reuses
the CPU benchmark's parameter-grid, VBM reference, reporting, and plotting
helpers so both backends see the same sampled configurations. The numerical
evaluation is independent: `code/xdense_gpu.py` calls the accelerator boundary
integrators directly and never calls a solver from `microjax.inverse_ray.cpu`.

The default `quick` run uses the same 25 x 13 x 9 configuration grid and six
points per configuration as the CPU benchmark:

```bash
python benchmark_gpu/code/xdense_gpu.py --preset quick
```

Useful development runs are:

```bash
# Kernel smoke test without a live VBM reference.
python benchmark_gpu/code/xdense_gpu.py --preset smoke --reference none --no-plot

# Default fast A100 route.
python benchmark_gpu/code/xdense_gpu.py --preset quick --n-limb 64 \
  --configuration-batch-size 128

# Higher-accuracy dense run: 4608 ordinary points or 576 cascade points/launch.
python benchmark_gpu/code/xdense_gpu.py --input reference.csv \
  --reference-column reference_magnification --n-limb 64 --deep-topology \
  --chart optimized --radial-subdivisions 2 \
  --configuration-batch-size 128 --cascade-configuration-batch-size 16

# Replay a CPU/GPU miss CSV with an independent reference column.
python benchmark_gpu/code/xdense_gpu.py --input reference.csv \
  --reference-column reference_magnification --n-limb 64
```

`n_limb` is constrained to 64 or more. JIT compilation is warmed up and
excluded from reported timings. The runner requires a CUDA GPU unless
`--allow-non-gpu` is passed explicitly for a smoke check.

The CLI defaults are the audited public binary fast settings: 64 limbs, a
40-cell radial chunk, `fast` chart, and 128 configurations per launch. The
explicit options in the command above document those values rather than alter
them.

`fast` is the default chart. It uses a shallow 64-limb polar trace and one
measured 40-interval local buffer across the full mass-ratio range. Every
radial cell uses one fixed 19-point Gauss rule. It does not compute a second
rule as a non-guaranteeing comparison and it does not retry. `optimized`
retains the slower low-q polar/Cartesian cascade when the tighter tail is more
important than throughput.

The quick-grid optimization measurements on an NVIDIA A100-PCIE-40GB were:

| one-pass configuration | concurrent points | median ms/point | p95 relative error | error > 1e-3 |
| --- | ---: | ---: | ---: | ---: |
| 64 limbs, shallow topology | 768 | 0.224 | 1.45e-4 | 360 |
| 64 limbs, radial x2 | 768 | 0.429 | 8.22e-5 | 348 |
| 64 limbs, robust roots + deep topology | 768 | 0.273 | 9.59e-5 | 278 |
| 64 limbs, fast roots + deep topology | 768 | 0.243 | 9.58e-5 | 281 |
| 64 limbs, fast roots + deep topology | 3072 | 0.233 | 9.58e-5 | 281 |

The ordinary route keeps the fast angular root solve and spends the small
extra budget on denser source-limb topology discovery. Uniform radial
subdivision and the 47-point radial rule remain rejected for ordinary points:
both roughly doubled time while barely changing the difficult-tail count.

The first full 443,700-point dense run measured 0.230 ms/point versus 0.920
ms/point for the MacBook CPU benchmark. Analysis of its low-q tail exposed a
structural error in the planetary local chart: point-source polynomial root
slots were treated as independent image-area owners. Near a fold, several
slots bound one connected planetary image group, so using one polar origin per
slot could integrate the same area repeatedly without setting a status bit.

The corrected polar path assigns the complete planetary image group to one
local chart. Its origin may lie outside the image: radial topology represents
that component as an annulus. Removing the old, unnecessary “origin inside”
gate reduced structural failures in the `q <= 1e-3`, `rho <= 1e-3` audit from
12,681 to 58 without changing its approximately 0.23 ms/point cost.

The residual low-q problem is not solved safely by applying one Cartesian
projection everywhere: it is fast (0.178 ms/point for 576 concurrent points)
but some caustic topologies are incomplete in that projection. The optimized
route therefore evaluates polar first and compacts only its nonzero radial
diagnostics into one GPU batch. Those points use a source-radial Cartesian
strip chart; invalid or >0.5% cross-chart disagreements use the already batched
two-cell local-polar rule. This is a bounded fail-closed cascade, not an
unbounded accuracy retry ladder. On 81,000 low-q/low-rho points it left eight
nonzero statuses. After refreshing 216 stale VBM values with the current VBM at
`Tol=RelTol=1e-5`, 99% of the 78,678 trustworthy references were below
8.69e-5 relative error and only 39 exceeded 1e-3.

The higher-accuracy dense result is in
`outputs/dense_n64_optimized_cb128/`. Its per-batch median was 0.234 ms/point;
the padding-aware aggregate was 0.286 ms/point (3,501 points/s and 3.22x the
MacBook CPU throughput). It returned 62 nonzero
structural statuses. Among 439,185 references whose strict and timing VBM
solutions agree to 1e-3, the relative-error median, p95, and p99 were
2.49e-5, 5.84e-5, and 1.51e-4. The remaining 761 values above 1e-3 are mostly
outside the optimized low-q/low-rho route; targeted tests showed that doubling
the limb count, profile subdivisions, or radial quadrature order does not
remove that tail. It is therefore tracked separately as a limb-topology
detection problem rather than hidden by globally increasing fixed work.

The default fast dense result is in `outputs/dense_n64_fast_cb128/`. Auditing
all 443,700 traces found a mean of 17.2 and a maximum of 37 active radial
intervals, so its local kernel uses a status-checked capacity of 40 instead of
evaluating all 64 general-purpose slots. A full-grid audit showed that the
local chart produces the same magnifications and statuses above ``q=1e-2`` as
the former global chart, so fast no longer branches on mass ratio. Together
with the externally audited 19-node rule, this raises the final aggregate from
5,799 to 10,583 points/s (0.0945 ms/point), 1.82x faster. For 439,186
trustworthy references, p50, p95, and p99 relative errors are 2.60e-5, 7.39e-5,
and 5.12e-4; 2,441 points exceed 1e-3 and 67 return a nonzero structural status.
In the low-q/low-rho subset, p99 is 5.15e-4 and 369 of 78,679 references exceed
1e-3.

The public binary accelerator API uses the same fast policy by default:
`Nlimb=64`, one fixed 19-point radial rule, 512 source points per outer tile, a
40-lane local radial kernel across all mass ratios, and no mass-ratio chart
branch. Explicit `BinaryMagConfig` values remain static JAX compilation
parameters.

These numbers are binary-lens settings, not universal accelerator constants.
The public triple-lens path uses its separately audited
`TripleMagConfig(n_limb=128, source_tile_size=100, radial_chunk_size=8)`;
widening it to the binary 512/40 scheduler reduced A100 forward-Jacobian
throughput because triple-lens tangent intermediates are substantially larger.

The Cartesian implementation lives in the accelerator integrator layer and
does not import `microjax.inverse_ray.cpu`. It reuses the existing source-limb
trace, constructs q-aware factored line sextics, and solves all strip roots as
one regular GPU batch. The report keeps `diagnostic_valid_relative_error`
separate so detected structural failures are not mixed with accepted accuracy.

Outputs are written below `benchmark_gpu/outputs/<preset>/`. CSV and JSON data
are stored in `data/`; figures are stored in `figures/`.
