"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    "gauge_name",
    "area_geospa_fabric",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "dom_land_cover_frac",
    "dom_land_cover",
    "high_prec_timing",
    "low_prec_timing",
    "huc",
    "q_mean",
    "runoff_ratio",
    "stream_elas",
    "slope_fdc",
    "baseflow_index",
    "hfd_mean",
    "q5",
    "q95",
    "high_q_freq",
    "high_q_dur",
    "low_q_freq",
    "low_q_dur",
    "zero_q_freq",
    "geol_porostiy",
    "root_depth_50",
    "root_depth_99",
    "organic_frac",
    "water_frac",
    "other_frac",
    # Added for GB
    "flow_period_start",
    "flow_period_end",
    "quncert_meta",
    # Parameters describing missing values should be ignored?
    "sand_perc_missing",
    "silt_perc_missing",
    "clay_perc_missing",
    "organic_perc_missing",
    "bulkdens_missing",
    "tawc_missing",
    "porosity_cosby_missing",
    "porosity_hypres_missing",
    "conductivity_cosby_missing",
    "conductivity_hypres_missing",
    "root_depth_missing",
    "soil_depth_pelletier_missing",
]

# Maurer mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {}


def correct_scaler_inputs(df: pd.DataFrame):
    means = df.drop(["Year", "Mnth", "Day", "Hr"], axis=1).mean(axis=0).to_numpy()
    SCALER["input_means"] = means
    stds = df.drop(["Year", "Mnth", "Day", "Hr"], axis=1).std(axis=0).to_numpy()


def correct_scaler_output(df: pd.DataFrame):
    mean = df.mean()
    SCALER["output_mean"] = mean
    std = df.std()
    SCALER["output_std"] = std


def add_camels_us_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the
        `data` directory in the main folder of this repository., by default None

    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
    attributes_path = Path(camels_root) / "camels_attributes_v2.0"

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob("camels_*.txt")

    # Read-in attributes into one big dataframe
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=";", header=0, dtype={"gauge_id": str})
        df_temp = df_temp.set_index("gauge_id")

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    # convert huc column to double digit strings
    df["huc"] = df["huc_02"].apply(lambda x: str(x).zfill(2))
    df = df.drop("huc_02", axis=1)
    # tmp to check
    return df
    if db_path is None:
        db_path = str(
            Path(__file__).absolute().parent.parent / "data" / "attributes.db"
        )

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql("basin_attributes", conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the
        `data` directory in the main folder of this repository., by default None

    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
    attributes_path = (
        Path(camels_root)
        / "catalogue.ceh.ac.uk/datastore/eidchub/8344e4f3-d2ea-44f5-8afa-86d2987543a9/"
    )

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob("CAMELS_GB_*.csv")

    # Read-in attributes into one big dataframe
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f)
        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)
        # df_temp = pd.read_csv(f, sep=";", header=0, dtype={"gauge_id": str})
        # df_temp = df_temp.set_index("gauge_id")

        # if df is None:
        #    df = df_temp.copy()
        # else:
        #    df = pd.concat([df, df_temp], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.select_dtypes(exclude=["object"])
    # np.savetxt("data/basin_list.txt", df["gauge_id"].values, fmt="%s")
    df.set_index("gauge_id", inplace=True)
    df.index = df.index.astype("str")
    # tmp to check
    df = df.dropna(axis=1)
    # convert huc column to double digit strings
    if db_path is None:
        db_path = str(
            Path(__file__).absolute().parent.parent / "data" / "attributes.db"
        )

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql("basin_attributes", conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(
    db_path: str,
    basins: List[str],
    drop_lat_lon: bool = True,
    keep_features: List = None,
) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col="gauge_id")
    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if str(b) not in basins]
    # return drop_basins, df
    df = df.drop(drop_basins, axis=0)
    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(["gauge_lat", "gauge_lon"], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)
    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    # Temp before actually fixing scaling.
    return feature
    if variable == "inputs":
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == "output":
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    # Temp as well
    return feature
    if variable == "inputs":
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == "output":
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(
    x: np.ndarray, y: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))
    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i : i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(
    camels_root: Path, basin: str, remove_nan: bool = True
) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.

    :return: pd.DataFrame containing the meteorological forcing data and the
        area of the basin as integer.
    """
    if isinstance(camels_root, str):
        camels_root = Path(camels_root)
    elif not isinstance(camels_root, Path):
        raise ValueError(f"camels_root must be Path or str, not {type(camels_root)}")
    path = (
        camels_root
        / "catalogue.ceh.ac.uk"
        / "datastore"
        / "eidchub"
        / "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
        / "timeseries"
        / f"CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv"
    )
    exclude = ["pet", "discharge_vol", "discharge_spec", "peti"]
    df = pd.read_csv(path)
    # print(df[df.isna().any(axis=1)])
    # tqdm.write(f"Basin {basin} before dropna: {len(df)}")
    if remove_nan:
        df = df.dropna()
    # tqdm.write(f"Basin {basin} after dropna: {len(df)}")
    columns = df.columns.values
    df = df.drop(exclude, axis=1)
    dates = pd.to_datetime(df["date"])
    year = []
    day = []
    month = []
    hour = np.ones(len(dates)) * 12
    for date in df["date"]:
        date_split = date.split("-")
        year.append(int(date_split[0]))
        month.append(int(date_split[1]))
        day.append(int(date_split[2]))
    df["Year"] = np.array(year)
    df["Mnth"] = np.array(month)
    df["Day"] = np.array(day)
    df["Hr"] = hour
    df["Date"] = dates
    df.drop("date", axis=1, inplace=True)
    df.set_index("Date", inplace=True)
    correct_scaler_inputs(df)
    return df, 1


def load_discharge(camels_root: Path, basin: str, area: int) -> pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters

    :return: A pd.Series containng the catchment normalized discharge.
    """
    if isinstance(camels_root, str):
        camels_root = Path(camels_root)
    elif not isinstance(camels_root, Path):
        raise ValueError(f"camels_root must be Path or str, not {type(camels_root)}")
    discharge_path = (
        camels_root
        / "catalogue.ceh.ac.uk"
        / "datastore"
        / "eidchub"
        / "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
        / "timeseries"
        / f"CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv"
    )
    df = pd.read_csv(discharge_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df["discharge_spec"]
    df.fillna(0, inplace=True)
    df = pd.to_numeric(df)

    correct_scaler_output(df)
    return df


if __name__ == "__main__":
    # add_camels_attributes(camels_root="/home/bernhard/git/datasets_masters/camels_us/basin_dataset_public_v1p2", db_path="camels_us")
    # add_camels_attributes(
    #     camels_root="/home/bernhard/git/datasets_masters/camels_gb",
    #     db_path="converted_gb/attributes.db",
    # )
    forcing, area = load_forcing(
        camels_root="/home/bernhard/git/datasets_masters/camels_gb", basin=1001
    )
    load_discharge(
        camels_root="/home/bernhard/git/datasets_masters/camels_gb", area=1, basin=1001
    )
    print(SCALER)
    # print(load_attributes(db_path="camels_us", basins=["01013500"]))