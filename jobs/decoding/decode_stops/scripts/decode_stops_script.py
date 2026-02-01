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


    pn, stop_binned_spikes, stop_binned_feats, offset_log, stop_meta_used, stop_meta_groups = assemble_stop_design.assemble_stop_design_func(
        raw_data_folder_path,
        bin_width,
    )
    
    bin_df = spike_history.make_bin_df_from_stop_meta(stop_meta_used)


    spike_data_w_history, basis, colnames, meta_groups = (
        spike_history.build_design_with_spike_history_from_bins(
            spikes_df=pn.spikes_df,
            bin_df=bin_df,
            X_pruned=stop_binned_spikes,
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
            y_df=stop_binned_feats,
            behav_features=None,
            groups=stop_meta_used['event_id'].values,
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
