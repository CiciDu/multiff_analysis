%load_ext autoreload
%autoreload 2

import os, sys, sys
from pathlib import Path
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
    
from data_wrangling import specific_utils, process_monkey_information, general_utils
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
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import core_stops_psth, psth_postprocessing, psth_stats, compare_events, dpca_utils, prep_stop_psth_data
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import get_stops_utils

import sys
import math
import gc
import subprocess
from pathlib import Path

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
import cProfile
import pstats

# Machine Learning imports
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.multivariate.cancorr import CanCorr

# Neuroscience specific imports
import neo
import rcca

# To fit gpfa
import numpy as np
from importlib import reload
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from elephant.gpfa import gpfa_core, gpfa_util

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

print("done")


parser.add_argument("--raw_data_folder_path", type=float, default=1e-3,
                    help="Raw data folder path")

reduce_y_var_lags = False
planning_data_by_point_exists_ok = True
y_data_exists_ok = True

pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(raw_data_folder_path=raw_data_folder_path)
pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=planning_data_by_point_exists_ok)
# pn.planning_data_by_point, cols_to_drop = general_utils.drop_columns_with_many_nans(
#     pn.planning_data_by_point)
#pn.get_x_and_y_data_for_modeling(exists_ok=y_data_exists_ok, reduce_y_var_lags=reduce_y_var_lags)

if not hasattr(pn, 'spikes_df'):
    pn.retrieve_or_make_monkey_data()
    pn.spikes_df = neural_data_processing.make_spikes_df(pn.raw_data_folder_path, pn.ff_caught_T_sorted,
                                                            sampling_rate=pn.sampling_rate)
    
pn.make_or_retrieve_stop_category_df()

# Example wiring (mirrors your original usage)
captures_df, valid_captures_df, filtered_no_capture_stops_df, stops_with_stats = get_stops_utils.prepare_no_capture_and_captures(
    monkey_information=pn.monkey_information,
    closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
    ff_caught_T_new=pn.ff_caught_T_new,
    distance_col="distance_from_ff_to_stop",
)   

columns_to_add = ["stop_id", "stop_id_duration", "stop_id_start_time", "stop_id_end_time"]

pn.make_one_stop_w_ff_df()
one_stop_miss_df = pn.one_stop_w_ff_df[['first_stop_point_index', 'first_stop_time', 'latest_visible_ff']].copy()
one_stop_miss_df.rename(columns={'first_stop_point_index': 'stop_point_index', 'first_stop_time': 'stop_time'}, inplace=True)
one_stop_miss_df[columns_to_add] = pn.monkey_information.loc[one_stop_miss_df['stop_point_index'], columns_to_add].values

pn.make_or_retrieve_stop_category_df()
pn.get_try_a_few_times_info()
pn.get_give_up_after_trying_info()


columns_to_add = ["stop_id", "stop_id_duration", "stop_id_start_time", "stop_id_end_time"]
shared_columns = ["stop_point_index", "stop_time"] + columns_to_add

# --- Build expanded + ordered tables for GUAT / TAFT ---
GUAT_expanded = get_stops_utils._expand_trials(pn.GUAT_trials_df, pn.monkey_information)
TAFT_expanded = get_stops_utils._expand_trials(pn.TAFT_trials_df, pn.monkey_information)

# add stop_id to GUAT_trials_df and TAFT_trials_df
GUAT_expanded[columns_to_add] = pn.monkey_information.loc[GUAT_expanded['stop_point_index'], columns_to_add].values
TAFT_expanded[columns_to_add] = pn.monkey_information.loc[TAFT_expanded['stop_point_index'], columns_to_add].values


GUAT = get_stops_utils._add_cluster_ordering(GUAT_expanded)
TAFT = get_stops_utils._add_cluster_ordering(TAFT_expanded)

# --- Per-cluster slices (consistent, vectorized) ---
# First stop in each cluster
GUAT_first = GUAT[GUAT["is_first"]].reset_index(drop=True)
TAFT_first = TAFT[TAFT["is_first"]].reset_index(drop=True)

# Last stop in each cluster
GUAT_last = GUAT[GUAT["is_last"]].reset_index(drop=True)
capture_TAFT_last = TAFT[TAFT["is_last"]].reset_index(drop=True)

# Middle stops (exclude first and last)
GUAT_middle = GUAT[GUAT["is_middle"]].reset_index(drop=True)
TAFT_middle = TAFT[TAFT["is_middle"]].reset_index(drop=True)

# “First several” = all but the last stop in each cluster
GUAT_nonfinal = GUAT[GUAT["order_in_cluster"] < GUAT["cluster_size"] - 1].reset_index(drop=True)
TAFT_nonfinal = TAFT[TAFT["order_in_cluster"] < TAFT["cluster_size"] - 1].reset_index(drop=True)

# Combine the “first several” from both, keep only columns you care about, then sort by index
both_nonfinal = (
    pd.concat(
        [
            GUAT_nonfinal[shared_columns],
            TAFT_nonfinal[shared_columns],
        ],
        ignore_index=True
    )
    .sort_values("stop_point_index")
    .reset_index(drop=True)
)

persist_both_first = pd.concat([GUAT_first[shared_columns], 
                         TAFT_first[shared_columns]])

both_middle = pd.concat([GUAT_middle[shared_columns], 
                         TAFT_middle[shared_columns]])

# Optional: if you also want “last several” (all but the first), it’s symmetrical:
# GUAT_last_several = GUAT[GUAT["order_in_cluster"] > 0].reset_index(drop=True)
# capture_TAFT_last_several = TAFT[TAFT["order_in_cluster"] > 0].reset_index(drop=True)

GUAT_last_plus_single_miss = pd.concat([GUAT_last[shared_columns], 
                                         one_stop_miss_df[shared_columns]])

all_misses = pd.concat([one_stop_miss_df[shared_columns], 
                                         GUAT_expanded[shared_columns],
                                         TAFT_nonfinal[shared_columns]
                                         ])

all_first_misses = pd.concat(
    [one_stop_miss_df[shared_columns], GUAT_first[shared_columns], TAFT_first[shared_columns]],
    ignore_index=True
)

all_last_misses = pd.concat(
    [one_stop_miss_df[shared_columns], GUAT_last[shared_columns]],
    ignore_index=True
)

# captures not in TAFT last (assuming capture_TAFT_last is a subset of captures)
captures_minus_TAFT_last = compare_events.diff_by(valid_captures_df, capture_TAFT_last, key='stop_id')

# non-captures excluding those flagged as 'all_misses'
non_captures_minus_all_misses = compare_events.diff_by(filtered_no_capture_stops_df, all_misses, key='stop_id')
