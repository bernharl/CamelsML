"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import argparse
import json
import pickle
import random
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import CamelsH5, CamelsTXT
from .datautils import (
    add_camels_attributes,
    load_attributes,
    rescale_features,
    load_forcing,
)
from .ealstm import EALSTM
from .lstm import LSTM
from .metrics import calc_nse
from .nseloss import NSELoss
from .utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
"""GLOBAL_SETTINGS = {
    "batch_size": 1536,
    "clip_norm": True,
    "clip_value": 1,
    "dropout": 0.4,
    "epochs": 30,
    "hidden_size": 256,
    "initial_forget_gate_bias": 5,
    "log_interval": 50,
    "learning_rate": 1e-3,
    "seq_length": 270,
    # "train_start": pd.to_datetime("01101971", format="%d%m%Y"),
    # When to start?
    "train_start": pd.to_datetime("01101988", format="%d%m%Y"),
    "train_end": pd.to_datetime("30092015", format="%d%m%Y"),
    "val_start": pd.to_datetime("01101971", format="%d%m%Y"),
    # "val_start": pd.to_datetime("01101979", format="%d%m%Y"),
    # "val_end": pd.to_datetime("30091988", format="%d%m%Y"),
    "val_end": pd.to_datetime("30092015", format="%d%m%Y"),
}"""

# check if GPU is available
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###############
# Prepare run #
###############


def load_config(cfg_file: Union[Path, str], device="cuda:0", num_workers=1) -> Dict:
    if not isinstance(cfg_file, Path):
        try:
            cfg_file = Path(cfg_file)
        except TypeError:
            raise TypeError(
                f"cfg file must be convertible to Path, not {type(cfg_file)}"
            )
    cfg = {}
    date_type = lambda date: pd.to_datetime(date, format="%d%m%Y")

    def bool_type(var):
        if var in ("True", "False"):
            return var == "True"
        elif var in (0, 1):
            return bool(var)
        else:
            raise TypeError("A variable could not be converted to bool, check config")

    types = {
        "epochs": int,
        "camels_root": Path,
        "run_dir": Path,
        "train_start": date_type,
        "train_end": date_type,
        "val_start": date_type,
        "val_end": date_type,
        "test_start": date_type,
        "test_end": date_type,
        "device": str,
        "learning_rate": float,
        "seq_length": int,
        "batch_size": int,
        "clip_norm": bool_type,
        "clip_value": int,
        "dropout": float,
        "seed": int,
        "cache_data": bool_type,
        "num_workers": int,
        "no_static": bool_type,
        "concat_static": bool_type,
        "eval_epoch": int,
        "hidden_size": int,
        "log_interval": int,
        "initial_forget_gate_bias": float,
        "invalid_attr_file": Path,
        "train_basin_file": Path,
        "val_basin_file": Path,
        "test_basin_file": Path,
        "evaluate_on_epoch": bool_type,
    }
    cfg["num_workers"] = num_workers
    cfg["device"] = device
    with open(cfg_file, "r") as infile:
        for line in infile:
            if line[0] == "#":
                continue
            for i, sign in enumerate(line):
                if sign == "#":
                    line = line[:i]
            line = line.split(": ")
            key = line[0]
            value = line[1][:-1]
            try:
                cfg[key] = types[key](value)
            except KeyError:
                raise NotImplementedError(
                    f"No functionality for setting {key} implemented"
                )
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    # cfg["run_dir"] = Path(__file__).absolute().parent / "runs" / run_name
    cfg["run_dir"] = cfg["run_dir"] / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / "data" / "train"
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / "data" / "val"
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / "cfg.json").open("w") as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / "train_data.h5"
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=cfg["train_file"],
        basins=basins,
        dates=[cfg["train_start"], cfg["train_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"],
        scaler_dir=cfg["train_basin_file"].parent,
    )

    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        initial_forget_bias: int = 5,
        dropout: float = 0.0,
        concat_static: bool = False,
        no_static: bool = False,
    ):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(
                input_size=input_size_dyn,
                hidden_size=hidden_size,
                initial_forget_bias=initial_forget_bias,
            )
        else:
            self.lstm = EALSTM(
                input_size_dyn=input_size_dyn,
                input_size_stat=input_size_stat,
                hidden_size=hidden_size,
                initial_forget_bias=initial_forget_bias,
            )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self, x_d: torch.Tensor, x_s: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    try:
        basins = get_basin_list(cfg["train_basin_file"])
    except KeyError:
        raise KeyError(f"train_basin_file not found in config file")

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)
    # prepare PyTorch DataLoader
    ds = CamelsH5(
        h5_file=cfg["train_file"],
        basins=basins,
        db_path=cfg["db_path"],
        concat_static=cfg["concat_static"],
        cache=cfg["cache_data"],
        no_static=cfg["no_static"],
    )
    input_size_dyn = ds[0][0].size()[1]
    if cfg["no_static"] or cfg["concat_static"]:
        input_size_stat = 0
    else:
        input_size_stat = ds[0][1].size()[1]
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )

    # create model and optimizer
    model = Model(
        input_size_dyn=input_size_dyn,
        input_size_stat=input_size_stat,
        hidden_size=cfg["hidden_size"],
        initial_forget_bias=cfg["initial_forget_gate_bias"],
        dropout=cfg["dropout"],
        concat_static=cfg["concat_static"],
        no_static=cfg["no_static"],
    ).to(cfg["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Temp fix
    cfg["use_mse"] = False
    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))
        if cfg["evaluate_on_epoch"]:
            model = model.to("cpu")
            tqdm.write(f"Validating epoch {epoch}")
            evaluate(user_cfg=cfg, split="val", epoch=epoch)
            model = model.to(cfg["device"])


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: nn.Module,
    loader: DataLoader,
    cfg: Dict,
    epoch: int,
    use_mse: bool,
):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f"# Epoch {epoch}")

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = (
                x.to(cfg["device"]),
                y.to(cfg["device"]),
                q_stds.to(cfg["device"]),
            )
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = (
                x_d.to(cfg["device"]),
                x_s.to(cfg["device"]),
                y.to(cfg["device"]),
            )
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(cfg["device"])
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict, split: str = "test", epoch: Optional[int] = None):
    """

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    split: str, ["train", "validation", "test"]
        What part of the dataset to evaluate on.

    """
    if not split in ["train", "val", "test"]:
        raise NotImplementedError(
            f"split must be either train, val or test, not {split}"
        )
    with open(user_cfg["run_dir"] / "cfg.json", "r") as fp:
        run_cfg = json.load(fp)
    try:
        basins = get_basin_list(user_cfg[f"{split}_basin_file"])
    except KeyError:
        raise KeyError(
            f"split it set to {split}, but that is not defined in your config."
        )
    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()
    attrs_count = len(attributes.columns)
    timeseries_count = 6
    # create model
    input_size_stat = timeseries_count if run_cfg["no_static"] else attrs_count
    input_size_dyn = (
        timeseries_count
        if (run_cfg["no_static"] or not run_cfg["concat_static"])
        else timeseries_count + attrs_count
    )
    model = Model(
        input_size_dyn=input_size_dyn,
        input_size_stat=input_size_stat,
        hidden_size=run_cfg["hidden_size"],
        dropout=run_cfg["dropout"],
        concat_static=run_cfg["concat_static"],
        no_static=run_cfg["no_static"],
    ).to(user_cfg["device"])

    # load trained model
    if epoch is None:
        epoch = cfg["epochs"]
    weight_file = user_cfg["run_dir"] / f"model_epoch{epoch}.pt"
    model.load_state_dict(torch.load(weight_file, map_location=user_cfg["device"]))

    results = {}
    for basin in tqdm(basins):
        try:
            ds_test = CamelsTXT(
                camels_root=user_cfg["camels_root"],
                basin=basin,
                dates=[user_cfg[f"{split}_start"], user_cfg[f"{split}_end"]],
                is_train=False,
                seq_length=run_cfg["seq_length"],
                with_attributes=True,
                attribute_means=means,
                attribute_stds=stds,
                concat_static=run_cfg["concat_static"],
                db_path=db_path,
                scaler_dir=user_cfg["train_basin_file"].parent,
            )
        except ValueError as e:
            # raise e
            tqdm.write(f"Skipped {basin} because CamelsTXT crashed")
            continue
        except IndexError as e:
            # raise e
            tqdm.write(f"Skipped {basin} because 0 length")
            continue
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
        preds, obs = evaluate_basin(model, loader, user_cfg)
        try:
            df = pd.DataFrame(
                data={"qobs": obs.flatten(), "qsim": preds.flatten()},
                index=ds_test.dates_index[run_cfg["seq_length"] - 1 :],
            )
        except ValueError as e:
            tqdm.write(f"Skipped {basin} because of missing data")
            continue
        results[basin] = df
    _store_results(user_cfg, run_cfg, results, epoch)


def evaluate_basin(
    model: nn.Module, loader: DataLoader, user_cfg: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None
    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(user_cfg["device"]), y.to(user_cfg["device"])
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = (
                    x_d.to(user_cfg["device"]),
                    x_s.to(user_cfg["device"]),
                    y.to(user_cfg["device"]),
                )
                p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(
            preds.numpy(),
            variable="output",
            scaler_dir=user_cfg["train_basin_file"].parent,
        )
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs


def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame, epoch: int):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if run_cfg["no_static"]:
        file_name = (
            user_cfg["run_dir"]
            / f"lstm_no_static_seed{run_cfg['seed']}_epoch_{epoch}.p"
        )
    else:
        if run_cfg["concat_static"]:
            file_name = (
                user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}_epoch_{epoch}.p"
            )
        else:
            file_name = (
                user_cfg["run_dir"] / f"ealstm_seed{run_cfg['seed']}_epoch_{epoch}.p"
            )

    with (file_name).open("wb") as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
