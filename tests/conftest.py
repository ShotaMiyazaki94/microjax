import os, sys

def _gpu_marker_selected() -> bool:
    # Detect if pytest was invoked with a GPU marker selection (e.g., `-m gpu`)
    for i, arg in enumerate(sys.argv):
        if arg == "-m" and i + 1 < len(sys.argv):
            if "gpu" in sys.argv[i + 1]:
                return True
        if arg.startswith("-m="):
            if "gpu" in arg.split("=", 1)[1]:
                return True
    return False

# Default to CPU unless GPU tests are explicitly selected via marker.
if not _gpu_marker_selected():
    # Avoid initializing GPU plugins in default runs/CI
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    from jax import config
    config.update("jax_enable_x64", True)
except Exception:
    pass
