from typing import Union, Dict
from pathlib import Path
from datetime import date

import numpy as np


def split_basin_list_simple(
    basin_list: Union[str, Path], store_folder: Union[str, Path], cfg: Dict
):
    if isinstance(basin_list, str):
        basin_list = Path(basin_list)
    elif not isinstance(basin_list, Path):
        raise TypeError(f"basin_list must be Path or str, not {type(basin_list)}")
    if isinstance(store_folder, str):
        store_folder = Path(store_folder)
    elif not isinstance(store_folder, Path):
        raise TypeError(f"basin_list must be Path or str, not {type(basin_list)}")
    current_date = date.today().strftime("%d%m")
    store_folder = store_folder / f"single_split_{current_date}_{cfg['seed']}"
    store_folder.mkdir(parents=True, exist_ok=True)
    basins = np.loadtxt(basin_list, dtype="str")
    np.random.shuffle(basins)
    basins_train = basins[: 2 * len(basins) // 3]
    basins_test = basins[2 * len(basins) // 3 :]
    np.savetxt(store_folder / "basins_test.txt", basins_test, fmt="%s")
    np.savetxt(store_folder / "basins_train.txt", basins_train, fmt="%s")


if __name__ == "__main__":
    split_basin_list_simple("data/basin_list.txt", "data/split", {"seed": 1010})
