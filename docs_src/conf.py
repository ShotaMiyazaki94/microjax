"""Sphinx configuration for the microJAX user guide."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microjax import __version__

project = "microJAX"
author = "Shota Miyazaki"
copyright = "2025–2026, Shota Miyazaki"
version = release = __version__
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_typehints = "none"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "microJAX"
html_logo = "../logo/microjax.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/ShotaMiyazaki94/microjax",
    "repository_branch": "develop",
    "path_to_docs": "docs_src",
    "use_repository_button": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
}
