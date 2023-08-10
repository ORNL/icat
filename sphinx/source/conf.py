# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ICAT"
copyright = "2023, UT Battelle, LLC"
author = "Nathan Martindale, Scott L. Stewart"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

autosummary_generate = True
autosummary_imported_members = False
# add_module_names = False  # NOTE: this doesn't apply to toc stuff


napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"
autodoc_default_options = {"inherited-members": False, "undoc-members": True}

templates_path = ["_templates"]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_nav_level": 3,
    "navigation_depth": 6,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ORNL/icat",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/icat-iml/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "external_links": [
        {"name": "Changelog", "url": "https://github.com/ORNL/icat/releases"},
    ],
}

html_context = {"default_mode": "dark"}
