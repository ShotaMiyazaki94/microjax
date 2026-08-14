# A100 inverse-ray scheduler sweep — 2026-07-23

This is the historical scheduler measurement record used to select the
triple scheduler and the original binary scheduler. A later dense binary audit
selected the current 512/40 fast route documented in `benchmark_gpu/README.md`.

## Definitions

- **Trajectory size**: number of source positions passed to `mag_binary` or
  `mag_triple`.
- **Full-solve count**: number of positions rejected by the multipole
  prefilter and sent to image-boundary integration.
- **Source tile**: `source_tile_size`, the fixed outer `vmap` width. A
  partially occupied final tile is evaluated in full.
- **Radial chunk**: `radial_chunk_size`, the fixed inner radial-region width.
  The region capacity is 64.
- A scheduler is written as `source tile / radial chunk`.

“Sparse” and “dense” are intentionally not used in this record because they
do not define a full-solve-count threshold.

## Environment and protocol

| Item | Value |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB |
| GPU memory | 40,960 MiB |
| Driver | 570.211.01 |
| JAX | 0.10.2 |
| Precision | `jax_enable_x64=True` |
| Source-boundary samples | `n_limb=500` (historical sweep workload) |
| Timing | `block_until_ready`, compilation excluded |
| Repeats | 7 for the binary-uniform 25-point grid; 5 for the other 100-position grids; 3–5 for 1000-position and JVP checks |
| Source tiles tested | 8, 16, 32, 64, 100 |
| Radial chunks tested | 4, 8, 16, 32, 64 in the initial grid; 8, 32, 64 in verification grids |

The 100-position controlled workload used one repeated source position that
always entered the full calculation and one far-field position that always
passed the prefilter. Varying their counts isolated scheduler behavior from
changes in lens geometry.

All reported configurations returned finite values. On the two repository
example trajectories, radial chunks 8 and 64 produced bitwise-identical
magnification arrays (`max_abs=0`, `max_rel=0`, identical `NaN` masks).

## Controlled 100-position sweep

The table below averages execution time over binary/triple and
uniform/linear-limb-darkened workloads. It reports the fastest common setting
among radial chunks 8, 32, and 64.

| Full-solve count | Fastest setting | Mean time | Mean time at former `100 / 8` default | Speedup |
|---:|---:|---:|---:|---:|
| 1 | `8 / 64` | 30.47 ms | 96.17 ms | 3.16x |
| 8 | `8 / 64` | 30.45 ms | 96.18 ms | 3.16x |
| 32 | `32 / 64` | 50.90 ms | 96.09 ms | 1.89x |
| 64 | `64 / 64` | 69.96 ms | 96.11 ms | 1.37x |
| 100 | `100 / 64` | 91.52 ms | 94.82 ms | 1.04x |

Lens-specific qualifications at 100 full solves:

- Binary uniform: `100 / 64` = 54.34 ms; `100 / 8` = 60.80 ms.
- Binary limb darkening: `100 / 64` = 57.09 ms; `100 / 8` = 64.05 ms.
- Triple uniform: `100 / 64` = 125.11 ms; `100 / 8` = 125.17 ms.
- Triple limb darkening: `100 / 8` = 129.28 ms; `100 / 64` = 129.54 ms.

The binary result gives a radial-64 speed benefit of 10.6% for uniform and
10.9% for limb darkening. At triple source tile 100, the sum of the uniform
and limb-darkened medians is 254.44 ms at radial 8 and 254.65 ms at radial 64,
so radial 8 is lower by 0.08%.

## Controlled 1000-position sweep

Radial chunk was fixed at 64 to isolate the source tile.

### Binary uniform

| Full-solve count | Fastest source tile | Median |
|---:|---:|---:|
| 1 | 8 | 26.30 ms |
| 8 | 8 | 26.29 ms |
| 32 | 32 | 40.08 ms |
| 64 | 64 | 46.13 ms |
| 100 | 100 | 55.23 ms |
| 256 | 100 | 163.10 ms |
| 1000 | 100 | 540.52 ms |

### Triple uniform

| Full-solve count | Fastest source tile | Median |
|---:|---:|---:|
| 1 | 8 | 42.24 ms |
| 8 | 8 | 42.28 ms |
| 32 | 32 | 62.30 ms |
| 64 | 64 | 93.38 ms |
| 100 | 100 | 128.65 ms |
| 256 | 64 | 358.99 ms |
| 1000 | 100 | 1245.15 ms |

The 256-position result shows why “full-solve count greater than 100” is not a
sufficient recommendation rule. For triple lenses, four 64-position tiles
were faster than three 100-position tiles with 44 padded evaluations. For
binary lenses at the same count, tile 100 was faster.

## Default-selection benchmark trajectories

### Binary VBML-comparison benchmark trajectory

- Trajectory size: 1000.
- Lens/source parameters: `s=0.85`, `q=0.03`, `rho=5e-3`, `u1=0`.
- Full-solve count: 544.
- Compared source tiles: 8, 16, 32, 64, 100.
- Compared radial chunks: 8, 64.

| Calculation | `100 / 8` | `100 / 64` | Relative improvement from 64 |
|---|---:|---:|---:|
| Primal | 386.43 ms | 345.17 ms | 10.7% |
| JVP with respect to `q` | 410.34 ms | 388.37 ms | 5.4% |

`100 / 64` was the fastest measured binary setting and became the binary
default.

### Triple Jacobian trajectory

- Trajectory size: 1000.
- Lens/source parameters: `s=1.1`, `q=0.1`, `q3=0.01`, `rho=0.01`,
  `u1=0`, with the third lens at `0.3+1.2j`.
- Full-solve count: 888.
- Compared source tiles: 8, 16, 32, 64, 100.
- Compared radial chunks: 8, 64.

| Calculation | `100 / 8` | `100 / 64` | Relative change from 64 |
|---|---:|---:|---:|
| Primal | 1348.57 ms | 1355.67 ms | 0.5% slower |
| JVP with respect to `q3` | 1488.87 ms | 1504.29 ms | 1.0% slower |

`100 / 8` was the fastest measured triple setting and remains the triple
default.

## Peak device memory

Memory comparisons used independent processes with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. They are primal peaks for trajectory
size 100, full-solve count 100, and source tile 100.

| Lens | Radial 8 | Radial 64 | Ratio |
|---|---:|---:|---:|
| Binary | 160 MiB | 256 MiB | 1.60x |
| Triple | 144 MiB | 256 MiB | 1.78x |

These peaks occupy less than 1% of the measured 40 GiB device. The current
default decision therefore prioritizes measured execution time rather than
the radial-8 memory reduction.

## Current default decision

| Config | `n_limb` | `source_tile_size` | `radial_chunk_size` | Measurement basis |
|---|---:|---:|---:|---|
| `BinaryMagConfig` | 64 | 512 | 40 | Later 443,700-point dense fast-route audit |
| `TripleMagConfig` | 128 | 100 | 8 | Scheduler sweep here plus the 1000-point VBML accuracy/Jacobian audit |

The triple benchmark trajectory sends 888/1000 uniform and 885/1000
limb-darkened points to full ICRS. At 128 limbs its maximum relative errors
against VBMicrolensing were `9.903e-4` and `5.919e-4`, respectively. On the
same A100, the 100/8 forward Jacobian took about 2.21 s; widening the source
tile to 512 took 3.71 s and widening the radial chunk to 16 took 3.02 s.

For a known full-solve count at or below 100, use the smallest candidate tile
in `8, 16, 32, 64, 100` that is not smaller than the count, then benchmark the
adjacent tile. For counts above 100, benchmark source tiles 64 and 100 because
the winner depends on lens type, number of executed tiles, and final-tile
padding.
