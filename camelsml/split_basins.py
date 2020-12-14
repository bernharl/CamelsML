from typing import Union, Dict, List
from pathlib import Path
from datetime import date

import numpy as np
from tqdm import tqdm
import pandas as pd

from .utils import get_basin_list
from .datautils import load_forcing, load_discharge


def split_basins(
    camels_root: Union[str, Path],
    basin_list: Union[str, Path],
    split: List[float],
    store_folder: Union[str, Path],
    seed: int,
    normalize: bool = True,
):
    if isinstance(basin_list, str):
        basin_list = Path(basin_list)
    elif not isinstance(basin_list, Path):
        raise TypeError(f"basin_list must be Path or str, not {type(basin_list)}")
    if isinstance(store_folder, str):
        store_folder = Path(store_folder)
    elif not isinstance(store_folder, Path):
        raise TypeError(f"basin_list must be Path or str, not {type(basin_list)}")
    if sum(split) > 1:
        raise ValueError(f"sum of splits must be 1, not {sum(split)}")
    if len(split) not in (2, 3):
        raise ValueError(f"length of split must be 2 or 3, not {len(split)}")
    np.random.seed(seed)
    store_folder = store_folder / f"split_seed_{seed}"
    store_folder.mkdir(parents=True, exist_ok=True)
    basins = np.loadtxt(basin_list, dtype="str")
    np.random.shuffle(basins)
    basins_train = basins[: int(len(basins) * split[0])]
    if len(split) == 2:
        basins_test = basins[int(len(basins) * split[0]) :]
    else:
        basins_validation = basins[
            int(len(basins) * split[0]) : int(len(basins) * split[0])
            + int(len(basins) * split[1])
        ]
        basins_test = basins[
            int(len(basins) * split[0]) + int(len(basins) * split[1]) :
        ]
    np.savetxt(store_folder / "basins_test.txt", basins_test, fmt="%s")
    np.savetxt(store_folder / "basins_train.txt", basins_train, fmt="%s")
    if len(split) == 3:
        np.savetxt(store_folder / "basins_validation.txt", basins_validation, fmt="%s")
    if normalize:
        create_normalization_file(camels_root, store_folder / "basins_train.txt")


def create_normalization_file(camels_root: Union[str, Path], train_basin_list: Path):
    basin_list = get_basin_list(train_basin_list)
    ignore_columns = np.array(["Year", "Mnth", "Day", "Hr"])
    mean = np.array([0, 0, 0, 0, 0, 0]).reshape(1, -1)
    mean_squared = np.zeros_like(mean)
    length = 0
    for i, basin in enumerate(tqdm(basin_list)):
        forcing, _ = load_forcing(camels_root, basin)
        forcing = forcing.drop(ignore_columns, axis=1)
        discharge = load_discharge(camels_root, basin, _)
        if i == 0:
            mean = pd.DataFrame(mean, columns=forcing.columns)
            mean["discharge"] = np.array([0])
            mean_squared = pd.DataFrame(mean_squared, columns=forcing.columns)
            mean_squared["discharge"] = np.array([0])
        tmp_mean = forcing.sum(axis=0)
        tmp_mean_squared = (forcing ** 2).sum(axis=0)
        tmp_mean["discharge"] = discharge.sum()
        tmp_mean_squared["discharge"] = (discharge ** 2).sum()
        mean += tmp_mean
        mean_squared += tmp_mean_squared
        length += len(forcing)
    mean = mean / length
    mean_squared = mean_squared / length
    std = np.sqrt(mean_squared - mean ** 2)
    mean.to_csv(train_basin_list.parent / "means_train.csv")
    std.to_csv(train_basin_list.parent / "stds_train.csv")


if __name__ == "__main__":
    split_basins("data/basin_list.txt", [0.65, 0.1, 0.25], "data/split", 1010)
