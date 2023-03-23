Endstate correction from MM to QML potential
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/wiederm/endstate_correction/workflows/CI/badge.svg)](https://github.com/wiederm/endstate_correction/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/wiederm/endstate_correction/branch/main/graph/badge.svg)](https://codecov.io/gh/wiederm/endstate_correction/branch/main)
[![Github release](https://badgen.net/github/release/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/)
[![GitHub license](https://img.shields.io/github/license/wiederm/endstate_correction?color=green)](https://github.com/wiederm/endstate_correction/blob/main/LICENSE)
[![GH Pages](https://github.com/wiederm/endstate_correction/actions/workflows/build_page.yaml/badge.svg)](https://github.com/wiederm/endstate_correction/actions/workflows/build_page.yaml)
[![CodeQL](https://github.com/wiederm/endstate_correction/actions/workflows/codeql.yml/badge.svg)](https://github.com/wiederm/endstate_correction/actions/workflows/codeql.yml)
[![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://wiederm.github.io/endstate_correction/)
[![GitHub forks](https://img.shields.io/github/forks/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/network)
[![Github tag](https://badgen.net/github/tag/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/tags/)
[![GitHub issues](https://img.shields.io/github/issues/wiederm/endstate_correction?style=flat)](https://github.com/wiederm/endstate_correction/issues)
[![GitHub stars](https://img.shields.io/github/stars/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/stargazers)



Following an arbritary thermodynamic cycle to calculate a free energy for a given molecular system at a certain level of theory, we can perform endstate corrections at the nodes of the thermodynamic cycle to a desired target level of theory.
In this work we have performed the endstate corrections using equilibrium free energy calculations, non-equilibrium (NEQ) work protocols and free energy perturbation (FEP).

![TD_cycle](https://user-images.githubusercontent.com/64199149/183875405-be049fa2-7ba7-40ba-838f-e2d43c4801f4.PNG)


# Installation

We recommend setting up a new python environment with `python=3.9` and installing the packages defined here using `mamba`: https://github.com/wiederm/endstate_correction/blob/main/devtools/conda-envs/test_env.yaml.
This package can be installed using:
`pip install git+https://github.com/wiederm/endstate_correction.git`.

### Copyright

Copyright (c) 2022, Sara Tkaczyk, Johannes Karwounopoulos & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
