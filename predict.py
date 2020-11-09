from typing import Union
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from main import evaluate_basin, GLOBAL_SETTINGS, Model, DEVICE
from papercode.datautils import load_attributes
import papercode.lstm
import papercode.ealstm
from papercode.datasets import CamelsTXT
from papercode.utils import get_basin_list


def predict_basin(
    basin: str,
    run_dir: Union[str, Path],
    camels_dir: Union[str, Path],
    period: str = "train",
    epoch: int = 30,
):
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    elif not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be str or Path, not {type(run_dir)}")
    if isinstance(camels_dir, str):
        camels_dir = Path(camels_dir)
    elif not isinstance(camels_dir, Path):
        raise TypeError(f"run_dir must be str or Path, not {type(camels_dir)}")

    with open(run_dir / "cfg.json", "r") as fp:
        run_cfg = json.load(fp)

    if not period in ["train", "val"]:
        raise ValueError("period must be either train or val")
    basins = get_basin_list()
    db_path = str(run_dir / "attributes.db")
    attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()
    attrs_count = len(attributes.columns)
    timeseries_count = 6
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
    weight_file = run_dir / f"model_epoch{epoch}.pt"
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(
        start=GLOBAL_SETTINGS[f"{period}_start"], end=GLOBAL_SETTINGS[f"{period}_end"]
    )
    ds_test = CamelsTXT(
        camels_root=camels_dir,
        basin=basin,
        dates=[GLOBAL_SETTINGS[f"{period}_start"], GLOBAL_SETTINGS[f"{period}_end"]],
        is_train=False,
        seq_length=run_cfg["seq_length"],
        with_attributes=True,
        attribute_means=means,
        attribute_stds=stds,
        concat_static=run_cfg["concat_static"],
        db_path=db_path,
    )
    loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
    preds, obs = evaluate_basin(model, loader)
    df = pd.DataFrame(
        data={"qobs": obs.flatten(), "qsim": preds.flatten()}, index=date_range
    )

    results = df
    # plt.plot(date_range, results["qobs"], label="Obs")
    # plt.plot(date_range, results["qsim"], label="Preds")
    # plt.legend()
    # plt.savefig(f"{run_dir}/pred_basin_{basin}.pdf")
    # plt.close()
    return results, date_range


if __name__ == "__main__":
    ealstm, date_range = predict_basin(
        "96004",
        "runs/run_2310_1224_seed473836",
        "/home/bernhard/git/datasets_masters/camels_gb",
        "val",
    )
    concat = predict_basin(
        "96004",
        "runs/run_2910_1513_seed298997",
        "/home/bernhard/git/datasets_masters/camels_gb",
        "val",
        20,
    )[0]
    lstm = predict_basin(
        "96004",
        "runs/run_3010_1252_seed465791",
        "/home/bernhard/git/datasets_masters/camels_gb",
        "val",
    )[0]
    plt.plot(date_range, ealstm["qsim"], "--", label=f"EALSTM, nse={0.58}")
    plt.plot(date_range, concat["qsim"], "--", label=f"Concat lstm, nse={0.84}")
    plt.plot(date_range, lstm["qsim"], "--", label=f"Lstm no static, nse={0.27}")
    plt.plot(date_range, concat["qobs"], ":", label="Observations")
    plt.title("Basin 96004")
    plt.legend()
    plt.savefig("figures/comparison.pdf")
