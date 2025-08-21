from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_bases import raised_cosine_basis, spline_basis, onset_from_mask_trials, angle_sin_cos, wrap_angle, _unique_trials, safe_poisson_lambda
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_design import build_glm_design_with_trials, lagged_design_from_signal_trials
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg.glm_fit import fit_poisson_glm_trials, predict_mu, poisson_deviance, pseudo_R2, per_trial_deviance
from neural_data_analysis.neural_analysis_tools.model_neural_data import drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.topic_based_neural_analysis.target_decoder import prep_target_decoder

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

def design_bases_for_behav_data(
    *,
    dt: float,
    trial_ids: np.ndarray,
    cur_vis: np.ndarray,
    nxt_vis: np.ndarray,
    cur_dist: np.ndarray,
    nxt_dist: np.ndarray,
    cur_angle: np.ndarray,
    nxt_angle: np.ndarray,
    speed: Optional[np.ndarray] = None,
    angular_speed: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    stop: Optional[np.ndarray] = None,
    capture: Optional[np.ndarray] = None,
    use_trial_FE: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Construct the MultiFF GLM glm_design and return (design_df, y, meta).

    Now supports multiple bases per feature by setting stimulus_basis_dict[name]
    to a list/tuple of basis matrices. build_glm_design_with_trials handles it.
    """
    T = len(trial_ids)

    # --- glm_bases ---
    _, B_event = raised_cosine_basis(n_basis=6, t_max=0.60, dt=dt, t_min=0.0, log_spaced=True)
    _, B_short = raised_cosine_basis(n_basis=5, t_max=0.30, dt=dt, t_min=0.0, log_spaced=True)
    _, B_hist  = raised_cosine_basis(n_basis=5, t_max=0.20, dt=dt, t_min=dt,  log_spaced=True)

    _, B_spline_event = spline_basis(n_basis=6, t_max=0.60, dt=dt, t_min=0.0, degree=3, log_spaced=True)
    _, B_spline_short = spline_basis(n_basis=5, t_max=0.30, dt=dt, t_min=0.0, degree=3, log_spaced=False)
    _, B_spline_hist  = spline_basis(n_basis=5, t_max=0.20, dt=dt, t_min=dt,  degree=3, log_spaced=True)


    # --- Onsets and gated features ---
    cur_on = onset_from_mask_trials(cur_vis, trial_ids)
    nxt_on = onset_from_mask_trials(nxt_vis, trial_ids)

    cur_dist_g = cur_dist * (cur_vis > 0)
    nxt_dist_g = nxt_dist * (nxt_vis > 0)

    cur_sin, cur_cos = angle_sin_cos(cur_angle)
    nxt_sin, nxt_cos = angle_sin_cos(nxt_angle)
    cur_sin *= (cur_vis > 0); cur_cos *= (cur_vis > 0)
    nxt_sin *= (nxt_vis > 0); nxt_cos *= (nxt_vis > 0)

    stimulus_dict: Dict[str, np.ndarray] = {
        "cur_on": cur_on,
        "nxt_on": nxt_on,
        "cur_dist": cur_dist_g,
        "nxt_dist": nxt_dist_g,
        "cur_angle_sin": cur_sin,
        "cur_angle_cos": cur_cos,
        "nxt_angle_sin": nxt_sin,
        "nxt_angle_cos": nxt_cos,
        "speed": speed,
        "angular_speed": angular_speed,
        "curvature": curvature,
    }

    extra_covariates: Dict[str, np.ndarray] = {}
    if speed is not None:     extra_covariates["speed"] = speed
    if curvature is not None: extra_covariates["curvature"] = curvature

    # ✅ Minimal change: allow any value to be either a single basis or a list of bases.
    # Example below: give cur_on both B_event and B_short; others keep a single basis.
    stimulus_basis_dict: Dict[str, object] = {
        "cur_on": [B_event, B_spline_event],        # multiple bases (example)
        "nxt_on": [B_event, B_spline_event],
        "stop": [B_event, B_spline_event],
        "capture": [B_event, B_spline_event],
        "cur_dist": [B_short, B_spline_short],
        "nxt_dist": [B_short, B_spline_short],
        "cur_angle_sin": [B_short, B_spline_short],
        "cur_angle_cos": [B_short, B_spline_short],
        "nxt_angle_sin": [B_short, B_spline_short],
        "nxt_angle_cos": [B_short, B_spline_short],
        "speed": [B_short, B_spline_short],
        "angular_speed": [B_short, B_spline_short],
        "curvature": [B_short, B_spline_short],
    }

    design_df, y = build_glm_design_with_trials(
        dt=dt,
        trial_ids=trial_ids,
        stimulus_dict=stimulus_dict,
        stimulus_basis_dict=stimulus_basis_dict,  # <— now can contain lists
        history_basis=B_hist,
        extra_covariates=extra_covariates,
        spike_counts=None,  # only transform behavioral data here
        use_trial_FE=use_trial_FE,
    )

    meta = {"B_event": B_event, "B_short": B_short, "B_hist": B_hist}
    return design_df, y, meta



def reduce_df(df, corr_threshold_for_lags_of_a_feature=0.9,
                        vif_threshold_for_initial_subset=5, vif_threshold=5, verbose=True,
                        filter_corr_by_all_columns=True,
                        filter_vif_by_all_columns=True,
                        ):
    # adapted from _reduce_y_var_base

    df_reduced = prep_target_decoder.remove_zero_var_cols(
        df)

    # Call the function to iteratively drop lags with high correlation for each feature
    df_reduced_corr = drop_high_corr_vars.drop_columns_with_high_corr(df_reduced,
                                                                        corr_threshold_for_lags=corr_threshold_for_lags_of_a_feature,
                                                                        verbose=verbose,
                                                                        filter_by_feature=False,
                                                                        filter_by_subsets=False,
                                                                        filter_by_all_columns=filter_corr_by_all_columns)

    df_reduced = drop_high_vif_vars.drop_columns_with_high_vif(df_reduced_corr,
                                                                        vif_threshold_for_initial_subset=vif_threshold_for_initial_subset,
                                                                        vif_threshold=vif_threshold,
                                                                        verbose=verbose,
                                                                        filter_by_feature=False,
                                                                        filter_by_subsets=False,
                                                                        filter_by_all_columns=filter_vif_by_all_columns)
    return df_reduced