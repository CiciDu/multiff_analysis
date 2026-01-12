from neural_data_analysis.design_kits.design_by_segment import create_design_df


import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import BSpline

# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.design_kits.design_by_segment import temporal_feats, spatial_feats, predictor_utils, other_feats


import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, pd.Series, list]



def get_initial_full_session_design_df(
    data: pd.DataFrame,
    dt: float,
    trial_ids: np.ndarray | None = None,
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
    design_df, meta = create_design_df.add_state_and_spatial_features(
        design_df=design_df,
        data=data,
        meta=meta,
    )

    # NOTE: no finalize/normalize shims needed; grouping now matches the checker logic
    return design_df, meta0, meta

