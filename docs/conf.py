"""Sphinx configuration for the AudioSig API overview."""

from __future__ import annotations

project = "AudioSig"
copyright = "2026, AudioSig contributors"
author = "AudioSig contributors"
version = "0.1"
release = "0.1.0"

extensions = ["myst_parser"]
templates_path: list[str] = []
exclude_patterns: list[str] = []
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

# MyST parser configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
