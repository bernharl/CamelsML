# Main code for my master thesis


# Installation
- If you use pipenv and pyenv:
```
pipenv install -e git+https://github.com/bernharl/ealstm_regional_modeling_camels_gb.git#egg=camelsml --python 3.8
```
- If only using pipenv, make sure to manually fix the Python version. Pytorch doesn't support Python >= 3.9 as of December 4th 2020.
- If not using pipenv, this repository should be installable using pip as well.

## Content of the repository
- will add later


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

## License of our code
[Apache License 2.0](https://github.com/kratzert/ealstm_regional_modeling/blob/master/LICENSE)

## License of the CAMELS GB dataset and pre-trained models.
