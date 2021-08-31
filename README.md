# Main code for my master thesis


# Installation
- If you use pipenv and pyenv:
```
pipenv install -e git+https://github.com/bernharl/ealstm_regional_modeling_camels_gb.git#egg=camelsml --python 3.8
```
- If not using pipenv, this repository should be installable using pip as well.

## Content of the repository
This repo is structured like a Python package. All relevant code is found within the `camelsml` directory.


## Citation

As you can see on the Github page, this repository is a fork of [this repository](https://github.com/kratzert/ealstm_regional_modeling).
Therefore, if you use this code, make sure to cite:

```
@article{kratzert2019universal,
author = {Kratzert, F. and Klotz, D. and Shalev, G. and Klambauer, G. and Hochreiter, S. and Nearing, G.},
title = {Towards learning universal, regional, and local hydrological behaviors via machine learning 
applied to large-sample datasets},
journal = {Hydrology and Earth System Sciences},
volume = {23},
year = {2019},
number = {12},
pages = {5089--5110},
url = {https://www.hydrol-earth-syst-sci.net/23/5089/2019/},
doi = {10.5194/hess-23-5089-2019}
}
```
, as well as the thesis connected to this code.

## License
[Apache License 2.0](https://github.com/kratzert/ealstm_regional_modeling/blob/master/LICENSE)
