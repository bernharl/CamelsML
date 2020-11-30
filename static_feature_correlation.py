from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

from papercode.evalutils import eval_lstm_models
from papercode.metrics import calc_nse
from papercode.datautils import load_attributes


def static_feature_relation(
    run_dir: Union[str, Path], camels_dir: Union[str, Path]
) -> pd.DataFrame:
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    elif not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be str or Path, not {type(run_dir)}")
    if isinstance(camels_dir, str):
        camels_dir = Path(camels_dir)
    elif not isinstance(camels_dir, Path):
        raise TypeError(f"camels_dir must be str or Path, not {type(camels_dir)}")
    seed = str(run_dir).split("_")[-1]
    results = eval_lstm_models([run_dir], calc_nse)[seed]
    # for basin in results.keys():
    # print(f"Basin {basin}: {results[basin]}")
    features = load_attributes(
        run_dir / "attributes.db", basins=list(results.keys())
    ).sort_index()
    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["NSE"])
    df_results.index.name = "gauge_id"
    df_results = df_results.sort_index()
    df = pd.concat([features, df_results], axis=1)
    return df


def least_squares_relation(
    df: pd.DataFrame,
) -> Tuple[sm.OLS, sm.regression.linear_model.RegressionResultsWrapper]:
    # clf = LinearRegression()
    scaler = StandardScaler()
    x = df.drop("NSE", axis=1)  # .to_numpy()
    x = scaler.fit_transform(x)
    y = df["NSE"]  # .to_numpy()
    # clf.fit(x, y)
    # print(clf.score(x, y))
    statsmod = sm.OLS(y, sm.add_constant(x))
    result = statsmod.fit()
    # print(np.argmax(result.pvalues))
    # print(result.summary())
    # print(result.aic)
    # print(type(result))
    return statsmod, result


# return clf


def backwards_selection(df: pd.DataFrame) -> pd.DataFrame:
    drop_list = []
    features = df.columns[:-1]
    print(features)
    for i in range(len(features)):
        print(f"Using {len(features) - 1} features.")
        model, result = least_squares_relation(df.drop(drop_list, axis=1))
        # print(result.aic)
        pvalues = result.pvalues
        new_index = ["intercept"] + list(features)
        pvalues.index = new_index
        print(f"Current AIC: {result.aic}")
        drop = pvalues.idxmax()
        drop_value = pvalues.max()
        if drop_value < 0.05:
            print(f"Stopped reducing features as the highest p-value is {drop_value}")
            print(f"This best model has an AIC: {result.aic}, R^2: {result.rsquared}")
            break
        print(f"Dropping {drop}, p-value: {drop_value}")
        features = features.drop(drop)
        drop_list.append(drop)
    return pd.DataFrame(result.params.to_numpy(), columns=["Coeff"], index=new_index)


def create_tex_table(df: pd.DataFrame, tex_path: Union[Path, str], filename: str):
    if isinstance(tex_path, str):
        tex_path = Path(tex_path)
    elif not isinstance(tex_path, Path):
        raise TypeError(f"tex_path must be str or Path, not {type(tex_path)}")
    tex_path.mkdir(exist_ok=True)
    with open(tex_path / filename, "w") as outfile:
        print(df.sort_values(r"Coeff", ascending=False).to_latex(), file=outfile)


def correlation(
    df: pd.DataFrame, title: str, fig_path: Union[str, Path], fig_name: str
):
    if isinstance(fig_path, str):
        fig_path = Path(fig_path)
    elif not isinstance(tex_path, Path):
        raise TypeError(f"fig_path must be str or Path, not {type(fig_path)}")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(4.7747, 4.7747))
    #ax.tick_params(axis="both", which="major", labelsize=1)
    ax.tick_params(axis="both", which="both", labelsize=5)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    ax.xaxis.label.set_size(5)
    ax.yaxis.label.set_size(5)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_path / fig_name)
    plt.close()


if __name__ == "__main__":
    run_dir = "/home/bernhard/git/ealstm_regional_modeling_camels_gb/runs/run_no_static_split_basin_seed571030"
    camels_dir = "/home/bernhard/git/datasets_masters/camels_gb"
    tex_path = "/home/bernhard/git/Master-Thesis/doc/thesis/tables"
    tex_file = "linreg_no_static.tex"
    fig_path = "/home/bernhard/git/Master-Thesis/doc/thesis/figures"
    df = static_feature_relation(run_dir, camels_dir)
    correlation(df, "Full correlation matrix", fig_path, "full_matrix.pdf")
    best_coefs = backwards_selection(df)
    correlation(
        df[np.append("NSE", best_coefs.index.drop("intercept").to_numpy())],
        "Correlation matrix after backwards elimination",
        fig_path,
        "reduced_matrix.pdf",
    )
    create_tex_table(best_coefs, tex_path, tex_file)
    plt.plot(best_coefs)
    # plt.show()
