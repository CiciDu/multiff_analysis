import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import hashlib
import json
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

from itertools import product
from data_wrangling import general_utils


def add_interaction_terms_and_features(concat_behav_trials):
    cols_added = []
    for var_a, var_b in product(
            ['cur_vis', 'nxt_vis', 'nxt_in_memory'],
            ['cur_ff_distance', 'nxt_ff_distance']):
        col_name = f'{var_a}*{var_b}'
        concat_behav_trials[col_name] = (
            concat_behav_trials[var_a] * concat_behav_trials[var_b]
        )
        cols_added.append(col_name)

    return concat_behav_trials, cols_added


def prep_behav(df,
               cont_cols=('cur_ff_distance', 'nxt_ff_distance',
                          'time_since_last_capture', 'speed', 'accel'),
               cat_vars=('cur_vis', 'nxt_vis', 'nxt_in_memory', 'log1p_num_ff_visible')):
    # keep only requested features (that exist), copy to avoid side effects
    out = df.copy()
    added_cols = []

    # add log1p features (clip negatives to 0 to keep log1p valid)
    for c in cont_cols:
        if c in out.columns:
            out[f'log1p_{c}'] = np.log1p(pd.to_numeric(
                out[c], errors='coerce').clip(lower=0))
            added_cols.append(f'log1p_{c}')

    # binarize categorical indicators ( > 0 → 1; else 0 )
    for v in cat_vars:
        if v in out.columns:
            x = pd.to_numeric(out[v], errors='coerce')
            out[v] = (x.fillna(0) > 0).astype('int8')
            added_cols.append(v)
    return out, added_cols


def get_band_conditioned_save_path(pn, reg_or_clf):
    neural_data_tag = make_raw_neural_data_processing_tag(pn)
    bin_width_str = f"{pn.bin_width:.4f}".rstrip(
        '0').rstrip('.').replace('.', 'p')
    seg_str = f'bin{bin_width_str}_{pn.cur_or_nxt}_{pn.first_or_last}_st{general_utils.clean_float(pn.start_t_rel_event)}_et{general_utils.clean_float(pn.end_t_rel_event)}'
    if reg_or_clf == 'reg':
        save_path = os.path.join(pn.planning_and_neural_folder_path,
                                 'pn_decoding', 'band_conditioned_reg', neural_data_tag, seg_str)
    elif reg_or_clf == 'clf':
        save_path = os.path.join(pn.planning_and_neural_folder_path,
                                 'pn_decoding', 'band_conditioned_clf', neural_data_tag, seg_str)
    else:
        raise ValueError(f'Invalid reg_or_clf: {reg_or_clf}')
    return save_path















