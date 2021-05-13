from typing import Union, Dict, List, Optional
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
    timeseries: List[str],
    dataset: List[str],
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
    if len(split) == 2:
        basins_test = basins[: int(len(basins) * split[1])]
        basins_train = basins[int(len(basins) * split[1]) :]
    else:
        basins_test = basins[: int(len(basins) * split[2])]
        basins_validation = basins[
            int(len(basins) * split[2])
            : int(len(basins) * split[1]) + int(len(basins) * split[2])
        ]
        basins_train = basins[
            int(len(basins) * split[1]) + int(len(basins) * split[2]) :
        ]
    np.savetxt(store_folder / "basins_test.txt", basins_test, fmt="%s")
    np.savetxt(store_folder / "basins_train.txt", basins_train, fmt="%s")
    if len(split) == 3:
        np.savetxt(store_folder / "basins_validation.txt", basins_validation, fmt="%s")
    if normalize:
        create_normalization_file(
            camels_root,
            store_folder / "basins_train.txt",
            dataset=dataset,
            timeseries=timeseries,
        )


def cross_validation_split(
    camels_root: Union[str, Path],
    basin_list: Union[str, Path],
    k: int,
    test_split: float,
    store_folder: Union[str, Path],
    seed: int,
    dataset: List[str],
    timeseries: List[str],
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
    store_folder = store_folder / f"cross_validation_seed_{seed}"
    store_folder.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    basins = np.loadtxt(basin_list, dtype="str")
    np.random.shuffle(basins)
    basins_test = basins[: int(len(basins) * test_split)]
    basins = basins[int(len(basins) * test_split) :]
    basins_split = np.array_split(basins, k)
    np.savetxt(store_folder / "basins_test.txt", basins_test, fmt="%s")
    for i, basins_val in enumerate(basins_split):
        split_folder = store_folder / f"{i}"
        split_folder.mkdir(parents=True, exist_ok=True)
        basins_train = np.delete(basins_split, i)
        basins_train_list = []
        for sub_split in basins_train:
            basins_train_list.extend(list(sub_split))
        basins_train = np.array(basins_train_list, dtype=object)
        del basins_train_list
        np.savetxt(split_folder / "basins_val.txt", basins_val, fmt="%s")
        np.savetxt(split_folder / "basins_train.txt", basins_train, fmt="%s")
        create_normalization_file(
            camels_root,
            split_folder / "basins_train.txt",
            dataset=dataset,
            timeseries=timeseries,
        )


def combine_cv_datasets(
    cv_folder_1: Path,
    cv_folder_2: Path,
    store_folder: Path,
    seed: int,
    k: int = 5,
    normalize: bool = True,
    timeseries: Optional[List[str]] = None,
    dataset: Optional[List[str]] = None,
    camels_root: Optional[Union[Path, str, List[Union[Path, str]]]] = None,
):
    store_folder = store_folder / f"cross_validation_seed_{seed}"
    store_folder.mkdir(exist_ok=True, parents=True)
    cv1 = cv_folder_1 / f"cross_validation_seed_{seed}"
    cv2 = cv_folder_2 / f"cross_validation_seed_{seed}"

    test1 = np.loadtxt(cv1 / "basins_test.txt", dtype="str")
    test2 = np.loadtxt(cv2 / "basins_test.txt", dtype="str")
    test = np.append(test1, test2)
    np.savetxt(store_folder / "basins_test.txt", test, fmt="%s")

    for i in range(k):
        train1, train2 = (
            np.loadtxt(cv1 / f"{i}" / "basins_train.txt", dtype="str"),
            np.loadtxt(cv2 / f"{i}" / "basins_train.txt", dtype="str"),
        )
        train = np.append(train1, train2)
        cv_folder = store_folder / f"{i}"
        cv_folder.mkdir(exist_ok=True)
        np.savetxt(cv_folder / "basins_train.txt", train, fmt="%s")
        if normalize:
            create_normalization_file(
                camels_root=camels_root,
                dataset=dataset,
                train_basin_list=cv_folder / "basins_train.txt",
                timeseries=timeseries,
            )
        val1, val2 = (
            np.loadtxt(cv1 / f"{i}" / "basins_val.txt", dtype="str"),
            np.loadtxt(cv2 / f"{i}" / "basins_val.txt", dtype="str"),
        )
        val = np.append(val1, val2)
        np.savetxt(cv_folder / "basins_val.txt", val, fmt="%s")


def create_normalization_file(
    camels_root: Union[str, Path],
    train_basin_list: Path,
    dataset: List[str],
    timeseries: List[str],
):
    basin_list = get_basin_list(train_basin_list)
    mean = np.zeros(len(timeseries)).reshape(1, -1)
    mean_squared = np.zeros_like(mean)
    length = 0
    for i, basin in enumerate(tqdm(basin_list)):
        forcing, _ = load_forcing(camels_root, basin, dataset=dataset)
        forcing = forcing[timeseries]
        discharge = load_discharge(camels_root, basin, _, dataset=dataset)
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
