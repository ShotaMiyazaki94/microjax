# FP32 and GPU feasibility audit (2026-07-27)

## Decision

The final acceptance criterion in this audit is

```text
abs(A_microjax - A_VB) / abs(A_VB) <= 1e-3
```

where `A_VB` is VBBinaryLensing for binary lenses and VBMicrolensing for
triple lenses. A new non-finite value is a failure. FP64 microJAX is retained
only as a regression reference.

Whole-program FP32 does not meet this criterion. The recommended near-term
route is to keep the numerical solver in FP64 and remove the serial root
continuation bottleneck. This already gives substantial speedups without
sacrificing the external-reference accuracy.

## Environment and workloads

- NVIDIA A100-PCIE-40GB, compute capability 8.0
- JAX and jaxlib 0.10.2, CUDA backend
- Python 3.13.12
- `n_points=1000`, `n_boundary=100`, `n_limb=500`
- uniform sources
- VBBinaryLensing and VBMicrolensing with `Tol=RelTol=1e-6`

The 500-limb value above records this historical precision experiment; it is
not the current public accelerator default. Current defaults are 64 limbs for
binary lenses and 128 limbs for triple lenses.

Representative cases:

| Case | Parameters |
|---|---|
| binary | `s=0.85`, `q=0.03`, `rho=5e-3` |
| binary low-q | `s=1.2`, `q=1e-6`, `rho=1e-5` |
| triple | `s=1.1`, `q=0.1`, `q3=0.01`, `rho=1e-2` |

## FP64 baseline and bottleneck

Steady-state A100 timings before parallel continuation:

| Stage | Binary | Triple |
|---|---:|---:|
| point roots, 1000 points | 2.01 ms | 5.90 ms |
| prefilter, 1000 points | 2.12 ms | 6.01 ms |
| source limb, 100 boundary points | 9.57 ms | 48.83 ms |
| full boundary, 100 points | 53.00 ms | 149.37 ms |
| end-to-end, 1000 points | 224.41 ms | 1345.37 ms |
| end-to-end JVP | 252.21 ms | 1487.99 ms |

The binary prefilter rejected 363/1000 points and the triple prefilter rejected
844/1000, so boundary work determines the end-to-end time.

Nsight Systems and XLA metadata identify the main binary boundary costs:

| Operation | Approximate time per 100-point call |
|---|---:|
| source-limb polynomial roots | 5.2 ms |
| sequential 5! root continuation | 32.8 ms |
| fixed 32-step angular roots | 11.5 ms |

The largest cost is a serial `lax.scan` over limb samples, not FP64 arithmetic.

## Why whole-program FP32 fails

Two accidental promotions were found and removed for a fair experiment:

- the EA initial circle used an untyped `2j`, creating complex128 roots;
- multipole factorials used `gammaln(int64)`, lifting the prefilter to FP64.

After those corrections, whole-program FP32 still fails for structural
reasons:

1. The general EA solver uses a fixed `tol=1e-12`. This is unreachable in
   float32, so it often runs to the 100-iteration cap.
2. Global complex64 lens polynomials lose the small root separations described
   in the local-chart analysis. This is particularly severe at low mass ratio
   and for triple-lens near-multiple roots.
3. Physical-root masks and the multipole acceptance decision therefore change,
   so the FP32 end-to-end workload is not merely a cheaper version of the FP64
   workload.

Direct VBBL/VBML results:

| Case | FP32 time | Non-finite | Finite values over relative 1e-3 |
|---|---:|---:|---:|
| binary | 112 ms | 102/1000 | 428/898 |
| binary low-q | 74 ms | 0/1000 | 131/1000 |
| triple | 496 ms | 942/1000 | 38/58 |

For the controlled 100-point boundary batches, 26 binary, 100 low-q binary,
and 46 triple results exceed `1e-3`. The low-q boundary failures are especially
decisive: all 100 differ from VBBL by approximately 100 percent.

Thus the attractive raw end-to-end speed of FP32 is partly caused by incorrect
prefilter acceptance and cannot be treated as a valid speedup.

## Mixed-precision experiment

An isolated binary experiment kept source-limb roots and topology in FP64,
then cast the completed support to FP32 for radial/angular integration.

- FP64 boundary: 53.1 ms
- mixed boundary: 59.4 ms in the ordinary benchmark (42.7 ms in the profiled
  run)
- 82/100 controlled values met relative `1e-3`
- every mixed result carried an angular-root or tolerance status

The FP32 angular solver follows different polish/classification paths, and
fallback of the remaining 18 percent removes the possible speed benefit.
This simple precision boundary is therefore a no-go.

## Precision-preserving structural result

For binary roots, each adjacent source-limb pair can compute its optimal 5!
assignment independently. The transition permutations are then composed by
`lax.associative_scan`. This is mathematically equivalent to the sequential
continuation except for exact assignment ties.

Measured FP64 results:

| Workload | Sequential | Parallel continuation | Speedup |
|---|---:|---:|---:|
| binary boundary, 100 | 53.09 ms | 23.77 ms | 2.23x |
| binary end-to-end, 1000 | 224.50 ms | 94.25 ms | 2.38x |
| binary JVP, 1000 | 252.40 ms | 114.66 ms | 2.20x |
| low-q boundary, 100 | 70.84 ms | 41.55 ms | 1.70x |
| low-q end-to-end, 1000 | 77.97 ms | 46.05 ms | 1.69x |
| triple boundary, 100 | 150.73 ms | 125.18 ms | 1.20x |
| triple end-to-end, 1000 | 1346.99 ms | 1129.39 ms | 1.19x |
| triple JVP, 1000 | 1495.64 ms | 1263.43 ms | 1.18x |

Binary outputs and JVPs were identical in the measured trajectories. Triple
originally used an input-slot-order-dependent greedy ten-root assignment. The
parallel path now canonicalizes both root sets by physical complex coordinate
and mask before applying a mask-first greedy assignment. It is therefore
equivariant to independent polynomial-root permutations at every limb sample.
Randomly permuting all ten slots independently at every sample left the
complete tracked branches identical. In the 1000-point audit, its maximum
magnification difference from sequential FP64 was `6.42e-9` relative and its
maximum JVP difference was `3.58e-6` relative.

A fixed-shape ten-root Hungarian prototype was also implemented and checked
against exact assignment. It made sequential triple end-to-end tracking
prohibitively slow (`17.65 s`) and the parallel result `1.246 s`; the latter
was only about `1.08x` faster than the original sequential workload. The
canonical mask-first matcher retains the required permutation equivariance at
`1.129 s`, so SciPy Hungarian assignment is used only as a unit-test oracle
and the rejected device solver is not retained in the hot module.

Against the external solvers, the parallel FP64 end-to-end maximum relative
errors over the 999 finite trajectory points were:

| Case | Maximum relative error versus VBBL/VBML |
|---|---:|
| binary | 5.16e-7 |
| binary low-q | 7.47e-9 |
| triple | 6.60e-6 |

All are well below `1e-3`.

The same parallel topology construction was propagated through the generic
radial-profile and linear limb-darkening paths. For `u1=0.5`, 1000-point FP64
light curves gave:

| Workload | Sequential | Parallel continuation | Speedup | Parallel numerical change |
|---|---:|---:|---:|---:|
| binary | 237.4 ms | 107.5 ms | 2.21x | exactly zero on finite points |
| binary low-q | 85.9 ms | 49.1 ms | 1.75x | exactly zero on finite points |
| triple | 1391.9 ms | 1168.9 ms | 1.19x | max `5.17e-7` relative |

The parallel triple limb-darkened result was checked directly against
VBMicrolensing 5.5 `MultiMagDark`: all 999 common finite points were below
`1e-3`, with maximum relative error `1.57e-5`. One microJAX value was the same
pre-existing status-rejected NaN discussed below.

An offset caustic-crossing trajectory supplied a second direct external check
with no non-finite microJAX values. Against VBMicrolensing at `1e-6` accuracy,
the 500-point maximum relative errors were `4.75e-4` for a uniform source and
`4.42e-4` for `u1=0.5`; both remain inside the requested `1e-3` bound.

For binary linear limb darkening, VBBinaryLensing `BinaryMagDark` is not used
as the sole oracle because its process-global state and one-annulus early stop
can itself exceed `1e-3` near caustics. The repository's state-isolated VBBL
uniform-contour layer-cake certificate has 42/42 cases below `1e-3`, with
maximum relative error `1.11e-5`. Parallel binary continuation is
mathematically equivalent and was bit-identical to the sequential result in
both 1000-point trajectories, so that external certificate transfers without
an additional numerical approximation.

The safety-first binary retry scheduler was also replayed on the saved
768-point VBBL stress matrix:

- 768/768 finite;
- 768/768 inside the stricter `1e-5 + 1e-4 abs(A)` contract;
- zero false accepts;
- maximum relative error `4.24e-5`;
- sum of the 24 per-case steady times: 5.31 s to 3.99 s (1.33x);
- median per-case steady time: 0.126 s to 0.082 s (1.54x).

## One-pass NaN distinction

The retry-free public one-pass trajectory returned one NaN at the source-plane
origin in each representative case. VBBL/VBML returned finite values there.
The internal best-effort values were within `4.7e-7` relative, but status bit 4
(`ANGULAR_ROOT_FAILURE`) intentionally rejected them. The parallel algorithm
did not create these failures.

For a strict “every final magnification is valid” requirement, the validated
`inverse_ray_retry.mag_binary_safe` scheduler, or an equivalent fixed-shape
compact retry, is required. Globally ignoring status bit 4 is not justified by
the three symmetric examples.

## Feasibility assessment

| Proposal | Accuracy feasibility | Expected value on A100 | Decision |
|---|---|---|---|
| disable x64 / whole FP32 | fails badly | invalid 2-3x apparent speed | no-go |
| FP32 global roots, FP64 accumulation | global polynomial already damaged | low | no-go |
| FP64 topology, FP32 radial/angular | 82% pass in binary sample | no reliable gain | no-go as tested |
| FP64 plus parallel continuation | preserves VBBL/VBML accuracy | 1.18-2.4x | go |
| compact scalar-batch retry | already validated for binary safe path | workload-dependent | go |
| chart-native FP32 roots `(center, scale, u)` | plausible, especially on Ada | high R&D cost | research |

Chart-native FP32 remains technically possible, but it is a new numerical
representation rather than a dtype switch. Coefficients must be built directly
in local coordinates, roots must remain represented as `(center, scale, u)`,
and residual/Jacobian/area calculations must avoid rematerialising an
ill-resolved global complex64 value. Central/resonant cancellation needs
mass-graded coefficient formulas. A fixed-size compact FP64 retry should be
driven by backward error, physical lens residual, topology consistency, and
the final magnification error estimate.

On A100, where ordinary FP32 throughput is only about twice FP64, the measured
structural optimization should be completed before this larger redesign. On
consumer Ada GPUs with a much larger FP32:FP64 ratio, the chart-native research
path may have a stronger payoff, but it must be re-profiled on that hardware.

## Recommended sequence

1. Keep parallel binary continuation enabled by default and retain the safe
   retry path for strict accuracy.
2. Keep the existing VBBL stress, caustic-map, limb-darkening, and gradient
   suites as release regressions.
3. Keep the triple canonical-assignment permutation tests and direct
   VBMicrolensing uniform/limb-darkening comparisons as release regressions.
4. Profile the remaining triple radial/angular loop and reduce redundant
   fixed-shape work without changing precision.
5. Treat chart-native FP32 as a separate algorithm branch with VBBL/VBML as the
   sole final magnification authority.
