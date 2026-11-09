"""CPU and GPU implementations share utilities without importing each other."""

from __future__ import annotations

import ast
from pathlib import Path

from microjax._magnification.gpu.integrators.profile import _binary_profile_limb_count


def test_profile_topology_density_is_independent_of_root_budget():
    assert _binary_profile_limb_count(64, False) == 127
    assert _binary_profile_limb_count(64, True) == 253


def test_backend_dependencies_stay_separate():
    package = Path(__file__).parents[3] / "src" / "microjax" / "_magnification"
    groups = (
        (list((package / "cpu").rglob("*.py")), {"gpu"}),
        (list((package / "gpu").rglob("*.py")), {"cpu"}),
        ([p for p in package.glob("*.py") if p.name not in {"api.py", "__init__.py"}], {"cpu", "gpu"}),
    )
    for paths, forbidden in groups:
        assert paths
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                assert not any(forbidden.intersection(module.split(".")) for module in modules), path
