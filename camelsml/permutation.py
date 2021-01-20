from typing import Optional, Dict
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from camelsml.datautils import load_attributes
from camelsml.datasets import CamelsTXT
from camelsml import evaluate, get_basin_list, Model


def permutation_test(
    cfg, k: int = 2, epoch: Optional[int] = 30, split: str = "val"
) -> Dict:
    db_path = str(cfg["run_dir"] / "attributes.db")
    if "attribute_selection_file" in cfg.keys():
        attribute_selection = np.genfromtxt(
            cfg["attribute_selection_file"], dtype="str"
        )
        tqdm.write(f"Using {len(attribute_selection)} static features")
    else:
        attribute_selection = None
    i_dict = defaultdict(dict)
    basins = get_basin_list(cfg[f"{split}_basin_file"])
    features = load_attributes(
        db_path=db_path, basins=basins, keep_features=attribute_selection
    ).columns
    for i, feature in enumerate(tqdm(features)):
        for k_ in range(k):
            nse_values = evaluate(
                user_cfg=cfg,
                split=split,
                epoch=epoch,
                store=False,
                permutate_feature=feature,
            )
            i_dict[feature][k_] = nse_values
    return i_dict
