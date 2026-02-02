from typing import Optional, Union

import numpy as np
import pandas as pd
from neural_data_analysis.design_kits.design_by_segment import (
    create_pn_design_df,
    temporal_feats
)

# your modules

ArrayLike = Union[np.ndarray, pd.Series, list]


def get_initial_full_session_design_df(
    data: pd.DataFrame,
    dt: float,
    trial_ids: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict, dict]:

    # work on a copy
    data = data.copy()
    data = temporal_feats.add_stop_and_capture_columns(data, trial_ids)

    specs, meta0 = temporal_feats.init_predictor_specs(
        data,
        dt,
        trial_ids,
    )

    specs, meta0 = temporal_feats.add_event_predictors(
        specs,
        meta0,
        data,
        events_to_include=['stop', 'capture_ff'],
        basis_family_event='rc',
        n_basis_event=6,
    )

    # ---------- assemble temporal design ----------
    design_df, meta = temporal_feats.specs_to_design_df(
        specs,
        meta0['trial_ids'],
        edge='zero',
        add_intercept=True,
    )
    rows_mask = meta.get('valid_rows_mask')  # None for edge='zero'

    # ---------- refactored block ----------
    design_df, meta = create_pn_design_df.add_state_and_spatial_features(
        design_df=design_df,
        data=data,
        meta=meta,
    )

    # NOTE: no finalize/normalize shims needed; grouping now matches the checker logic
    return design_df, meta0, meta


def merge_design_blocks(fs_df, best_arc_df, pn_df, stop_df):
    fs_cols = set(fs_df.columns) - {'bin'}
    best_arc_cols = set(best_arc_df.columns) - {'bin'}
    pn_cols = set(pn_df.columns) - {'bin'}
    stop_cols = set(stop_df.columns) - {'bin'}

    print(f'Duplicated FS–Best Arc columns ({len(fs_cols & best_arc_cols)}):')
    print(sorted(fs_cols & best_arc_cols))

    print(f'Duplicated FS–PN columns ({len(fs_cols & pn_cols)}):')
    print(sorted(fs_cols & pn_cols))

    print(f'Duplicated FS–STOP columns ({len(fs_cols & stop_cols)}):')
    print(sorted(fs_cols & stop_cols))

    return (
        fs_df
        .merge(best_arc_df, on='bin', how='left', suffixes=('', ''))
        .merge(pn_df, on='bin', how='left', suffixes=('', '_pn'))
        .merge(stop_df, on='bin', how='left', suffixes=('', '_stop'))
        .fillna(0.0)
        .sort_values('bin')
        .reset_index(drop=True)
    )
