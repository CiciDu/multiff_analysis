import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import general_glm_fit, cv_stop_glm
from neural_data_analysis.design_kits.design_by_segment import create_pn_design_df
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from data_wrangling import general_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import cvn_from_ref_class
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import ff_vis_epochs
import neural_data_analysis.design_kits.design_around_event.event_binning as event_binning

def init_decoding_data(raw_data_folder_path,
                       cur_or_nxt='cur',
                       first_or_last='first',
                       time_limit_to_count_sighting=2,
                       start_t_rel_event=0,
                       end_t_rel_event=1.5,
                       end_at_stop_time=False):

    planning_data_by_point_exists_ok = True

    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path)

    pn.prep_data_to_analyze_planning(
        planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)

    pn.rebin_data_in_new_segments(
        cur_or_nxt=cur_or_nxt,
        first_or_last=first_or_last,
        time_limit_to_count_sighting=time_limit_to_count_sighting,
        start_t_rel_event=start_t_rel_event,
        end_t_rel_event=end_t_rel_event,
        end_at_stop_time=end_at_stop_time,
    )

    for col in ['cur_vis', 'nxt_vis', 'cur_in_memory', 'nxt_in_memory']:
        pn.rebinned_y_var[col] = (pn.rebinned_y_var[col] > 0).astype(int)

    return pn


def get_data_for_decoding_vis(rebinned_x_var, rebinned_y_var, dt):

    data = rebinned_y_var.copy()
    trial_ids = data['new_segment']

    design_df, meta0, meta = create_pn_design_df.get_initial_design_df(
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
    cluster_cols = [
        c for c in rebinned_x_var.columns if c.startswith('cluster_')]
    df_Y = rebinned_x_var[cluster_cols]
    df_Y.columns = df_Y.columns.str.replace('cluster_', '').astype(int)

    return df_X, df_Y



def prepare_new_seg_info(ff_dataframe, bin_width):

    # minimal: detect runs, no merging (each run is its own cluster)
    df2 = ff_vis_epochs.compute_visibility_runs_and_clusters(
        ff_dataframe.copy(), ff_col='ff_index', t_col='point_index', time_col='time', vis_col='visible',
        chunk_merge_gap=0.05,    # seconds: merge *raw* runs into chunks if gap <= this
        cluster_merge_gap=1
    )

    df2 = ff_vis_epochs.add_global_visibility_bursts(df2, global_merge_gap=0.25)
    #df2 = ff_vis_epochs.add_global_vis_cluster_id(df2, group_cols=None, nullable_int=True)
    df2 = ff_vis_epochs.add_global_vis_chunk_id(df2, group_cols=None, nullable_int=True)
    #df2 = ff_vis_epochs.add_global_vis_cluster_id(df2, group_cols=None, nullable_int=True)

    vis_df = df2.loc[df2['visible'] == 1].copy()

    # based on any ff visible
    sequential_vis_df = vis_df[['ff_index', 'ff_vis_start_time', 'ff_vis_end_time',
                            'global_vis_chunk_id', 'global_burst_id', 'global_burst_start_time','global_burst_end_time',
                            'global_burst_duration','global_burst_size',
                            #'global_burst_prev_start_time','global_burst_prev_end_time'
                            ]].drop_duplicates().reset_index(drop=True)
    sequential_vis_df = sequential_vis_df.sort_values('ff_vis_start_time').reset_index(drop=True)
    sequential_vis_df['prev_time'] = sequential_vis_df['ff_vis_start_time'].shift(1)
    sequential_vis_df['next_time'] = sequential_vis_df['ff_vis_start_time'].shift(-1)


    new_seg_info = event_binning.pick_event_window(sequential_vis_df,
                                                    event_time_col='ff_vis_start_time',
                                                    prev_event_col='prev_time',
                                                    next_event_col='next_time',
                                                    pre_s=0.1, post_s=0.5, min_pre_bins=2, min_post_bins=3, bin_dt=bin_width)
    new_seg_info['event_id'] = new_seg_info['global_vis_chunk_id']
    new_seg_info['event_time'] = new_seg_info['ff_vis_start_time']


    events_with_stats = sequential_vis_df[['global_vis_chunk_id', 'global_burst_id', 'ff_vis_start_time', 'ff_vis_end_time']].copy()
    events_with_stats = sequential_vis_df.rename(columns={'global_vis_chunk_id': 'event_id', 
                                            'global_burst_id': 'event_cluster_id', 
                                            'ff_vis_start_time': 'event_id_start_time', 
                                            'ff_vis_end_time': 'event_id_end_time'})

    return new_seg_info, events_with_stats