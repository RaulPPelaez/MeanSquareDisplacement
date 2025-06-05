import os
import subprocess

# Project info
project = "mean_square_displacement"
author = "Raul P. Pelaez"
# Get the latest git tag as release version
try:
    release = (
        subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], cwd=os.path.abspath("..")
        )
        .decode("utf-8")
        .strip()
    )
except Exception:
    release = "unknown"


# General config
extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

# HTML options, rtd
html_theme = "sphinx_rtd_theme"

# Breathe config
breathe_projects = {"msd": "../doxygen/xml"}
breathe_default_project = "msd"
breathe_show_include = True
# Autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "show-inheritance": False,
}
