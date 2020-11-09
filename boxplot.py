from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from main import Model
from papercode.evalutils import eval_lstm_models
from papercode.metrics import calc_nse
from predict import predict_basin


def boxplot(run_dir: Union[str, Path]) -> Tuple[Tuple[str, float], Tuple[str, float], List]:
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    elif not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be str or Path, not {type(run_dir)}")
    seed = str(run_dir).split("_")[-1]

    results = eval_lstm_models([run_dir], calc_nse)[seed]
    #plt.boxplot(results.values())
    #plt.title(f"Boxplot showing validation metrics of {len(results)} basins")
    #plt.savefig(f"{run_dir}/boxplot.pdf")
    #plt.clf()
    results_list = list(results.values())
    keys_list = list(results.keys())
    return (keys_list[np.argmax(results_list)], np.max(results_list)), (
        keys_list[np.argmin(results_list)],
        np.min(results_list),
    ), results_list


if __name__ == "__main__":
    camels_dir = "/home/bernhard/git/datasets_masters/camels_gb"
    run_dir_ealstm = "runs/run_2310_1224_seed473836/"
    run_dir_concat = "runs/run_2910_1513_seed298997/"
    run_dir_no_static = "runs/run_3010_1252_seed465791"
    (best_ealstm, best_nse_ealstm), (worst_ealstm, worst_nse_ealstm), results_ealstm = boxplot(run_dir_ealstm)
    pred_best_ealstm, date_range = predict_basin(best_ealstm, run_dir_ealstm, camels_dir, period="val")
    pred_worst_ealstm = predict_basin(worst_ealstm, run_dir_ealstm, camels_dir, "val")[0]
    fig, ax = plt.subplots(3, 2, figsize=[10, 10])
    ax = ax.flatten()
    ax[0].plot(date_range, pred_best_ealstm["qsim"], label="pred")
    ax[0].plot(date_range, pred_best_ealstm["qobs"], label="obs")
    ax[0].set_title(f"EA-LSTM. Best basin [{best_ealstm}]. NSE: {best_nse_ealstm: .2f}")
    ax[0].legend()
    ax[1].plot(date_range, pred_worst_ealstm["qsim"], label="pred")
    ax[1].plot(date_range, pred_worst_ealstm["qobs"], label="obs")
    ax[1].set_title(f"EA-LSTM. Worst basin [{worst_ealstm}]. NSE: {worst_nse_ealstm: .2f}")
    ax[1].legend()

    (best_concat, best_nse_concat), (worst_concat, worst_nse_concat), results_concat = boxplot(run_dir_concat)
    pred_best_concat = predict_basin(best_concat, run_dir_concat, camels_dir, period="val", epoch=20)[0]
    pred_worst_concat = predict_basin(worst_concat, run_dir_concat, camels_dir, "val", epoch=20)[0]
    ax[2].plot(date_range, pred_best_concat["qsim"], label="pred")
    ax[2].plot(date_range, pred_best_concat["qobs"], label="obs")
    ax[2].set_title(f"LSTM concat. Best basin [{best_concat}]. NSE: {best_nse_concat: .2f}")
    ax[2].legend()
    ax[3].plot(date_range, pred_worst_concat["qsim"], label="pred")
    ax[3].plot(date_range, pred_worst_concat["qobs"], label="obs")
    ax[3].set_title(f"LSTM concat. Worst basin [{worst_concat}]. NSE: {worst_nse_concat: .2f}")
    ax[3].legend()


    (best_no_static, best_nse_no_static), (worst_no_static, worst_nse_no_static), results_no_static = boxplot(run_dir_no_static)
    pred_best_no_static = predict_basin(best_no_static, run_dir_no_static, camels_dir, period="val")[0]
    pred_worst_no_static = predict_basin(worst_no_static, run_dir_no_static, camels_dir, "val")[0]
    ax[4].plot(date_range, pred_best_no_static["qsim"], label="pred")
    ax[4].plot(date_range, pred_best_no_static["qobs"], label="obs")
    ax[4].set_title(f"LSTM no static. Best basin [{best_no_static}]. NSE: {best_nse_no_static: .2f}")
    ax[4].legend()
    ax[5].plot(date_range, pred_worst_no_static["qsim"], label="pred")
    ax[5].plot(date_range, pred_worst_no_static["qobs"], label="obs")
    ax[5].set_title(f"LSTM no static. Worst basin [{worst_no_static}]. NSE: {worst_nse_no_static: .2f}")
    ax[5].legend()

    fig.tight_layout()
    #fig.savefig(f"{run_dir_ealstm}/best_and_worst.pdf")
    fig.savefig("figures/best_and_worst.pdf")
    
    fig, ax = plt.subplots(1, 3)
    ax = ax.flatten()
    ax[0].boxplot(results_ealstm)
    ax[0].set_title(f"EA-LSTM, avg = {np.mean(results_ealstm): .2f}")
    ax[1].boxplot(results_concat)
    ax[1].set_title(f"Concat, avg = {np.mean(results_concat): .2f}")
    ax[2].boxplot(results_no_static)
    ax[2].set_title(f"No static, avg = {np.mean(results_no_static): .2f}")
    fig.tight_layout()
    fig.savefig("figures/boxplot.pdf")
