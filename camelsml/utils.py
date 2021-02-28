"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
"""
In compliance with the Apache-2.0 license I must inform that this file has been modified
by Bernhard Nornes Lotsberg. The original code by Kratzert et. al can be found at 
https://github.com/kratzert/ealstm_regional_modeling
"""

import sys
from pathlib import Path, PosixPath
from typing import List, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .datasets import CamelsTXT


def create_h5_files(
    camels_root: PosixPath,
    out_file: PosixPath,
    basins: List,
    dates: List,
    scaler_dir: Path,
    timeseries: List[str],
    dataset_name: str,
    with_basin_str: bool = True,
    seq_length: int = 270,
):
    """[summary]

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    out_file : PosixPath
        Path of the location, where the hdf5 file should be stored
    basins : List
        List containing the 8-digit USGS gauge id
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    with_basin_str : bool, optional
        If True, stores for each sample the corresponding USGS gauged id, by default True
    seq_length : int, optional
        Length of the requested input sequences., by default 270

    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")
    dataset = CamelsTXT(
        camels_root=camels_root,
        basin=basins[0],
        is_train=True,
        seq_length=seq_length,
        dates=dates,
        scaler_dir=scaler_dir,
        dataset=dataset_name,
        timeseries=timeseries,
    )
    num_timeseries = dataset.x.shape[-1]
    with h5py.File(out_file, "w") as out_f:
        input_data = out_f.create_dataset(
            "input_data",
            shape=(0, seq_length, num_timeseries),
            maxshape=(None, seq_length, num_timeseries),
            chunks=True,
            dtype=np.float32,
            compression="gzip",
        )
        target_data = out_f.create_dataset(
            "target_data",
            shape=(0, 1),
            maxshape=(None, 1),
            chunks=True,
            dtype=np.float32,
            compression="gzip",
        )

        q_stds = out_f.create_dataset(
            "q_stds",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )

        if with_basin_str:
            sample_2_basin = out_f.create_dataset(
                "sample_2_basin",
                shape=(0,),
                maxshape=(None,),
                dtype="S11",
                compression="gzip",
                chunks=True,
            )
        for basin in tqdm(basins, file=sys.stdout):

            dataset = CamelsTXT(
                camels_root=camels_root,
                basin=basin,
                is_train=True,
                seq_length=seq_length,
                dates=dates,
                scaler_dir=scaler_dir,
                dataset=dataset_name,
                timeseries=timeseries,
            )

            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples
            # store input and output samples
            input_data.resize((total_samples, seq_length, num_timeseries))
            target_data.resize((total_samples, 1))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :] = dataset.y

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array(
                [dataset.q_std] * num_samples, dtype=np.float32
            ).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            if with_basin_str:
                sample_2_basin.resize((total_samples,))
                str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
                sample_2_basin[-num_samples:] = str_arr

            out_f.flush()


def get_basin_list(split: Union[str, Path, None] = None) -> List:
    """Read list of basins from text file.

    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    if split is not None:
        folder = split
    else:
        folder = "data/basin_list.txt"
    basins = np.genfromtxt(folder, dtype=str)
    # These two basins only have data until 1996, skipping them.
    remove = np.array(["18017", "18018"])
    for basin in remove:
        basins = np.delete(basins, np.argwhere(basins == basin))
    return basins
