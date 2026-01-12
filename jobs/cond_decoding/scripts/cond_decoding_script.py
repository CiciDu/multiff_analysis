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
# isort: off
# fmt: off
from data_wrangling import specific_utils, process_monkey_information, general_utils, combine_info_utils
from pattern_discovery import pattern_by_trials, pattern_by_trials, cluster_analysis, organize_patterns_and_features
from visualization.matplotlib_tools import plot_behaviors_utils
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data, plot_modeling_result
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import planning_and_neural_class, pn_utils, pn_helper_class, pn_aligned_by_seg, pn_aligned_by_event
from neural_data_analysis.neural_analysis_tools.cca_methods import cca_class
from neural_data_analysis.neural_analysis_tools.cca_methods import cca_class, cca_utils, cca_cv_utils
from neural_data_analysis.neural_analysis_tools.cca_methods.cca_plotting import cca_plotting, cca_plot_lag_vs_no_lag, cca_plot_cv
from machine_learning.ml_methods import regression_utils, regz_regression_utils, ml_methods_class, classification_utils, ml_plotting_utils, ml_methods_utils
from planning_analysis.show_planning import nxt_ff_utils, show_planning_utils
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, plot_gpfa_utils, gpfa_helper_class
from neural_data_analysis.neural_analysis_tools.align_trials import time_resolved_regression, time_resolved_gpfa_regression,plot_time_resolved_regression
from neural_data_analysis.neural_analysis_tools.align_trials import align_trial_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import pn_decoding_utils, plot_pn_decoding, pn_decoding_model_specs

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions import add_interactions, discrete_decoders
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions.band_conditioned import conditional_decoding_clf, conditional_decoding_reg, cond_decoding_plots, agg_cond_decoding
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding.interactions.band_conditioned.specified_pairs import CONTINUOUS_INTERACTIONS, DISCRETE_INTERACTIONS


import sys
import math
import gc
import subprocess
from pathlib import Path
from importlib import reload
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from scipy import linalg, interpolate
from scipy.signal import fftconvolve
from scipy.io import loadmat
from scipy import sparse
import torch
from numpy import pi

# Machine Learning imports
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.multivariate.cancorr import CanCorr

# Neuroscience specific imports
import neo
import rcca
import quantities as pq

# fmt: on
# isort: on

# ----------------------------------------------------------------------
# Project path setup
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    reduce_y_var_lags = False
    planning_data_by_point_exists_ok = True
    y_data_exists_ok = True
    bin_width = 0.1

    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name, 'monkey_Bruno')

    for index, row in sessions_df_for_one_monkey.iterrows():
        print('='*100)
        print('='*100)
        print(row['data_name'])
        raw_data_folder_path = os.path.join(
            raw_data_dir_name, row['monkey_name'], row['data_name'])
        
        try:
            pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(raw_data_folder_path=raw_data_folder_path, bin_width=bin_width)
            pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)
            pn.planning_data_by_point, cols_to_drop = general_utils.drop_columns_with_many_nans(
                pn.planning_data_by_point)
            pn.get_x_and_y_data_for_modeling(exists_ok=y_data_exists_ok, reduce_y_var_lags=reduce_y_var_lags)
            pn.prepare_seg_aligned_data(start_t_rel_event=-0.25, end_t_rel_event=1.25, end_at_stop_time=False)

            flags = {
                'use_raw_spike_data_instead': True,
                'apply_pca_on_raw_spike_data': False,
                'use_lagged_raw_spike_data': False,
            }
            pn.get_concat_data_for_regression(**flags) 
            df = pn.concat_behav_trials.copy()
            df, added_cols = pn_decoding_utils.prep_behav(df)
            df = add_interactions.add_behavior_bands(df)

            key_features2 = (['cur_ff_distance', 'log1p_cur_ff_distance', 'speed', 'accel', 'time_since_last_capture'] + added_cols)

            save_path_reg = pn_decoding_utils.get_band_conditioned_save_path(pn, 'reg')

            outs_reg = conditional_decoding_reg.run_band_conditioned_reg_decoding(df,
                pn.concat_neural_trials,
                CONTINUOUS_INTERACTIONS,
                max_pairs=100,
                save_path=save_path_reg,
                make_plots=False,
                )

            save_path_clf = pn_decoding_utils.get_band_conditioned_save_path(pn, 'clf')
            outs_clf = conditional_decoding_clf.run_band_conditioned_clf_decoding(df,
                pn.concat_neural_trials,
                DISCRETE_INTERACTIONS,
                max_pairs=100,
                save_path=save_path_clf,
                make_plots=False,
                )
            
        except Exception as e:
            print(f"Error processing {row['data_name']}: {e}")
            continue

if __name__ == '__main__':
    main()
