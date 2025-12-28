import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_fit import stop_glm_fit, cv_stop_glm
from neural_data_analysis.design_kits.design_by_segment import create_design_df
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from data_wrangling import general_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import cvn_from_ref_class


def init_decoding_data(raw_data_folder_path):
    planning_data_by_point_exists_ok = True

    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path)

    pn.prep_data_to_analyze_planning(
        planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)

    pn.rebin_data_in_new_segments(
        cur_or_nxt='cur',
        first_or_last='first',
        time_limit_to_count_sighting=2,
        pre_event_window=0,
        post_event_window=1.5,
        rebinned_max_x_lag_number=2
    )

    for col in ['cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory']:
        pn.rebinned_y_var[col] = (pn.rebinned_y_var[col] > 0).astype(int)

    return pn

def get_data_for_decoding_vis(rebinned_x_var, rebinned_y_var, dt):

    data = rebinned_y_var.copy()
    trial_ids = data['new_segment']

    design_df, meta0, meta = create_design_df.get_initial_design_df(
        data, dt, trial_ids)

    df_X = design_df[
        [
            'speed_z',
            'time_since_last_capture',
            'ang_accel_mag_spline:s0',
            'ang_accel_mag_spline:s1',
            'ang_accel_mag_spline:s2',
            'ang_accel_mag_spline:s3',
            'cur_vis',
            'nxt_vis',
        ]
    ].copy()

    # neural matrix
    cluster_cols = [c for c in rebinned_x_var.columns if c.startswith('cluster_')]
    df_Y = rebinned_x_var[cluster_cols]
    df_Y.columns = df_Y.columns.str.replace('cluster_', '').astype(int)

    return df_X, df_Y


