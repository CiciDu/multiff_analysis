import gc
from pathlib import Path
import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import hashlib

for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break


from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import assemble_stop_design
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import pn_decoding_model_specs
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.design_kits.design_by_segment import spike_history, temporal_feats, create_pn_design_df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_folder_path',
        default=None,
        help='If provided, run GLM only on this raw data folder'
    )
    
    bin_width = 0.04
    t_max = 0.20
    
    args = parser.parse_args()
    raw_data_folder_path = args.raw_data_folder_path


    planning_data_by_point_exists_ok = True

    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(raw_data_folder_path=raw_data_folder_path)
    pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)
    #pn.get_x_and_y_data_for_modeling(exists_ok=y_data_exists_ok, reduce_y_var_lags=reduce_y_var_lags)

    pn.rebin_data_in_new_segments(cur_or_nxt='cur', first_or_last='first', time_limit_to_count_sighting=2,
                                    start_t_rel_event=0, end_t_rel_event=1.5, rebinned_max_x_lag_number=2)


    data = pn.rebinned_y_var.copy()
    trial_ids = data['new_segment']
    dt = pn.bin_width


    data = temporal_feats.add_stop_and_capture_columns(data, trial_ids, pn.ff_caught_T_new)
    design_df, meta0, meta = create_pn_design_df.get_initial_design_df(data, dt, trial_ids)

    cluster_cols = [col for col in pn.rebinned_x_var.columns if col.startswith('cluster_')]
    df_Y = pn.rebinned_x_var[cluster_cols]
    df_Y.columns = df_Y.columns.str.replace('cluster_', '').astype(int)

    
    bin_df = create_pn_design_df.make_bin_df_for_pn(pn.rebinned_x_var, pn.bin_edges)


    spike_data_w_history, basis, colnames, meta_groups = (
        spike_history.build_design_with_spike_history_from_bins(
            spikes_df=pn.spikes_df,
            bin_df=bin_df,
            X_pruned=df_Y,
            meta_groups={},
            dt=bin_width,
            t_max=t_max,
        )
    )
    
    save_dir = os.path.join(
        pn.planning_and_neural_folder_path,
        'stop_decoding',
    )
    
    all_results = []

    for model_name, spec in pn_decoding_model_specs.MODEL_SPECS.items():
            
        config = cv_decoding.DecodingRunConfig(
            # regression
            regression_model_class=spec.get('regression_model_class', None),
            regression_model_kwargs=spec.get('regression_model_kwargs', {}),

            # classification
            classification_model_class=spec.get('classification_model_class', None),
            classification_model_kwargs=spec.get('classification_model_kwargs', {}),

            # shared
            use_early_stopping=False,
        )
        
        print(f'model_name: ', model_name)
        print(f'config: ', config)
        
    
        results_df = cv_decoding.run_cv_decoding(
            X=spike_data_w_history,
            y_df=design_df,
            behav_features=None,
            groups=trial_ids,
            n_splits=5,
            config=config,
            context_label='pooled',
            save_dir=save_dir,
        )

        results_df['model_name'] = model_name
        all_results.append(results_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df


if __name__ == '__main__':
    main()
