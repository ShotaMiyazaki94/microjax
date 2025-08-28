#!/usr/bin/env bash
set -euo pipefail

export MICROJAX_GPU_TESTS=${MICROJAX_GPU_TESTS:-1}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}

python - <<'PY'
import sys
try:
    import jax
except Exception as e:
    print("[ERROR] JAX not importable:", e)
    sys.exit(1)

cuda = jax.devices("cuda")
if not cuda:
    print("[ERROR] No CUDA devices detected by JAX.")
    sys.exit(2)

print("Detected CUDA devices:")
for d in cuda:
    print(" -", d)

if not any("A100" in d.device_kind for d in cuda):
    print("[ERROR] No NVIDIA A100 detected (device_kind mismatch).")
    sys.exit(3)

print("A100 detected. Proceeding to run gpu-marked tests.\n")
PY

pytest -m gpu -q

