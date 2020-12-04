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
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (
    add_camels_attributes,
    load_attributes,
    rescale_features,
    load_forcing,
)
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
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
    #"train_start": pd.to_datetime("01101971", format="%d%m%Y"),
    # When to start?
    "train_start": pd.to_datetime("01101988", format="%d%m%Y"),
    "train_end": pd.to_datetime("30092015", format="%d%m%Y"),
    "val_start": pd.to_datetime("01101971", format="%d%m%Y"),
    #"val_start": pd.to_datetime("01101979", format="%d%m%Y"),
    # "val_end": pd.to_datetime("30091988", format="%d%m%Y"),
    "val_end": pd.to_datetime("30092015", format="%d%m%Y"),
}

# check if GPU is available
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###############
# Prepare run #
###############


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "eval_robustness"])
    parser.add_argument(
        "--camels_root", type=str, help="Root directory of CAMELS data set"
    )
    parser.add_argument("--seed", type=int, required=False, help="Random seed")
    parser.add_argument(
        "--run_dir", type=str, help="For evaluation mode. Path to run directory."
    )
    parser.add_argument(
        "--cache_data",
        type=bool,
        default=False,
        help="If True, loads all data into memory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of parallel threads for data loading",
    )
    parser.add_argument(
        "--no_static",
        type=bool,
        default=False,
        help="If True, trains LSTM without static features",
    )
    parser.add_argument(
        "--concat_static",
        type=bool,
        default=False,
        help="If True, train LSTM with static feats concatenated at each time step",
    )
    parser.add_argument(
        "--use_mse",
        type=bool,
        default=False,
        help="If True, uses mean squared error as loss function.",
    )
    parser.add_argument(
        "--split_train_test_folder",
        type=str,
        default=None,
        help="If defined, the training will only train on the train split in the folder, while evaluation will only happen on the test state",
    )
    parser.add_argument(
        "--cross_validation_run",
        type=bool,
        default=False,
        help="NOT IMPLEMENTED. Whether to use cross validation.",
    )
    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=GLOBAL_SETTINGS["epochs"],
        help="What epoch to evaluate",
    )
    cfg = vars(parser.parse_args())
    # print(cfg["no_static"])
    # exit()
    if cfg["cross_validation_run"]:
        raise NotImplementedError("Cross validation is not yet implemented.")
    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError(
            "In evaluation mode a run directory (--run_dir) has to be specified"
        )

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
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
    cfg["run_dir"] = Path(__file__).absolute().parent / "runs" / run_name
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

    if cfg["split_train_test_folder"] is not None:
        basins = get_basin_list(f"{cfg['split_train_test_folder']}/basins_train.txt")
    else:
        basins = get_basin_list()

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
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

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
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config

    """
    with open(user_cfg["run_dir"] / "cfg.json", "r") as fp:
        run_cfg = json.load(fp)

    if user_cfg["split_train_test_folder"] is not None:
        basins = get_basin_list(
            f"{user_cfg['split_train_test_folder']}/basins_test.txt"
        )
    else:
        basins = get_basin_list()

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
    ).to(DEVICE)

    # load trained model
    weight_file = user_cfg["run_dir"] / f"model_epoch{user_cfg['eval_epoch']}.pt"
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    results = {}
    for basin in tqdm(basins):
        try:
            ds_test = CamelsTXT(
                camels_root=user_cfg["camels_root"],
                basin=basin,
                dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                is_train=False,
                seq_length=run_cfg["seq_length"],
                with_attributes=True,
                attribute_means=means,
                attribute_stds=stds,
                concat_static=run_cfg["concat_static"],
                db_path=db_path,
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
        preds, obs = evaluate_basin(model, loader)
        try:
            df = pd.DataFrame(
                data={"qobs": obs.flatten(), "qsim": preds.flatten()}, index=ds_test.dates_index[run_cfg["seq_length"]-1:]
            )
        except ValueError as e:
            tqdm.write(f"Skipped {basin} because of missing data")
            continue
        results[basin] = df
    print(f"Saved {len(results)} basins")
    _store_results(user_cfg, run_cfg, results)


def evaluate_basin(
    model: nn.Module, loader: DataLoader
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
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(preds.numpy(), variable="output")
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs




def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
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
        file_name = user_cfg["run_dir"] / f"lstm_no_static_seed{run_cfg['seed']}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}.p"
        else:
            file_name = user_cfg["run_dir"] / f"ealstm_seed{run_cfg['seed']}.p"

    with (file_name).open("wb") as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)