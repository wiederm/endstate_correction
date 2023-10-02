Endstate correction from MM to NNP
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/wiederm/endstate_correction/workflows/CI/badge.svg)](https://github.com/wiederm/endstate_correction/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/wiederm/endstate_correction/branch/main/graph/badge.svg)](https://codecov.io/gh/wiederm/endstate_correction/branch/main)
[![Github release](https://badgen.net/github/release/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/)
[![GitHub license](https://img.shields.io/github/license/wiederm/endstate_correction?color=green)](https://github.com/wiederm/endstate_correction/blob/main/LICENSE)
[![GH Pages](https://github.com/wiederm/endstate_correction/actions/workflows/build_page.yaml/badge.svg)](https://github.com/wiederm/endstate_correction/actions/workflows/build_page.yaml)
[![CodeQL](https://github.com/wiederm/endstate_correction/actions/workflows/codeql.yml/badge.svg)](https://github.com/wiederm/endstate_correction/actions/workflows/codeql.yml)
[![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://wiederm.github.io/endstate_correction/)
[![GitHub issues](https://img.shields.io/github/issues/wiederm/endstate_correction?style=flat)](https://github.com/wiederm/endstate_correction/issues)
[![GitHub stars](https://img.shields.io/github/stars/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/stargazers)



Following an arbritary thermodynamic cycle to calculate a free energy for a given molecular system at a certain level of theory, we can perform endstate corrections at the nodes of the thermodynamic cycle to a desired target level of theory.
In this work we have performed the endstate corrections using equilibrium free energy calculations, non-equilibrium (NEQ) work protocols and free energy perturbation (FEP).

![TD_cycle](https://user-images.githubusercontent.com/64199149/183875405-be049fa2-7ba7-40ba-838f-e2d43c4801f4.PNG)


# Installation

We recommend setting up a new python environment with `python=3.9` and installing the packages defined here using `mamba`: https://github.com/wiederm/endstate_correction/blob/main/devtools/conda-envs/test_env.yaml.
This package can be installed using:
`pip install git+https://github.com/wiederm/endstate_correction.git`.

### Contributors

<!-- readme: collaborators,contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/xiki-tempula">
            <img src="https://avatars.githubusercontent.com/u/6242032?v=4" width="100;" alt="xiki-tempula"/>
            <br />
            <sub><b>Zhiyi Wu</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/wiederm">
            <img src="https://avatars.githubusercontent.com/u/31651017?v=4" width="100;" alt="wiederm"/>
            <br />
            <sub><b>Marcus Wieder</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/msuruzhon">
            <img src="https://avatars.githubusercontent.com/u/36005076?v=4" width="100;" alt="msuruzhon"/>
            <br />
            <sub><b>Miroslav Suruzhon</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/saratk1">
            <img src="https://avatars.githubusercontent.com/u/64199149?v=4" width="100;" alt="saratk1"/>
            <br />
            <sub><b>Sara Tkaczyk</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/JohannesKarwou">
            <img src="https://avatars.githubusercontent.com/u/72743318?v=4" width="100;" alt="JohannesKarwou"/>
            <br />
            <sub><b>Johannes Karwounopoulos</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: collaborators,contributors -end -->


### Copyright

Copyright (c) 2022, Sara Tkaczyk, Johannes Karwounopoulos & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
