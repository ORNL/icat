<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="sphinx/source/_static/icat_large_full_dark.svg" />
        <source media="(prefers-color-scheme: light)" srcset="sphinx/source/_static/icat_large_full_light.svg" />
        <img alt='ICAT logo' src="https://raw.githubusercontent.com/ORNL/icat/main/sphinx/source/_static/icat_large_full_light.svg" />
    </picture>
</p>

# Interactive Corpus Analysis Tool

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/icat-iml.svg)](https://badge.fury.io/py/icat-iml)
[![tests](https://github.com/ORNL/icat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ORNL/icat/actions/workflows/tests.yml)
[![License](https://img.shields.io/pypi/l/curifactory)](https://github.com/ORNL/curifactory/blob/main/LICENSE)
[![status](https://joss.theoj.org/papers/0528d60ff4f251069d15456fdb83bd0f/status.svg)](https://joss.theoj.org/papers/0528d60ff4f251069d15456fdb83bd0f)



The Interactive Corpus Analysis Tool (ICAT) is an interactive machine learning (IML) dashboard for unlabeled text datasets that allows a user to iteratively and visually define features, explore and label instances of their dataset, and train a logistic regression model on the fly as they do so to assist in filtering, searching, and labeling tasks.

![ICAT Screenshot](https://raw.githubusercontent.com/ORNL/icat/main/sphinx/source/_static/screenshot1.png)

ICAT is implemented using holoviz's [panel](https://panel.holoviz.org/) library, so it can either directly be rendered like a widget in a jupyter lab/notebook instance, or incorporated as part of a standalone panel website.

## Installation

ICAT can be installed via `pip` with:

```
pip install icat-iml
```

<!-- usage/examples -->

## Documentation

The user guide and API documentation can be found at [https://ornl.github.io/icat](https://ornl.github.io/icat).

## Visualization

The primary ring visualization is called AnchorViz, a technique from IML literature. (See Chen, Nan-Chen, et al. "[AnchorViz: Facilitating classifier error discovery through interactive semantic data exploration](https://dl.acm.org/doi/abs/10.1145/3172944.3172950)")

We implemented an ipywidget version of AnchorViz and use it in this project, it can be found separately at [https://github.com/ORNL/ipyanchorviz](https://github.com/ORNL/ipyanchorviz)

<!-- documentation section -->

## Citation

To cite usage of ICAT, please use the following bibtex:

```bibtex
@misc{doecode_105653,
    title = {Interactive Corpus Analysis Tool},
    author = {Martindale, Nathan and Stewart, Scott},
    abstractNote = {The Interactive Corpus Analysis Tool (ICAT) is an interactive machine learning dashboard for unlabeled text/natural language processing datasets that allows a user to iteratively and visually define features, explore and label instances of their dataset, and simultaneously train a logistic regression model. ICAT was created to allow subject matter experts in a specific domain to directly train their own models for unlabeled datasets visually, without needing to be a machine learning expert or needing to know how to code the models themselves. This approach allows users to directly leverage the power of machine learning, but critically, also involves the user in the development of the machine learning model.},
    year = {2023},
    month = {apr}
}
```
