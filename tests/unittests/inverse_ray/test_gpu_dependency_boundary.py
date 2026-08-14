"""GPU numerical layers stay independent from the CPU implementation."""

from __future__ import annotations

import ast
from pathlib import Path

from microjax.inverse_ray.integrators.profile import _binary_profile_limb_count


def test_profile_topology_density_is_independent_of_root_budget():
    assert _binary_profile_limb_count(64, False) == 127
    assert _binary_profile_limb_count(64, True) == 253


def test_accelerator_numerical_layers_do_not_import_cpu_modules():
    package = Path(__file__).parents[3] / "src" / "microjax" / "inverse_ray"
    layers = ("geometry", "integrators", "quadrature", "roots")
    offenders = []
    for layer in layers:
        for path in (package / layer).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "cpu" in node.module.split("."):
                        offenders.append(f"{path.name}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "cpu" in alias.name.split("."):
                            offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
