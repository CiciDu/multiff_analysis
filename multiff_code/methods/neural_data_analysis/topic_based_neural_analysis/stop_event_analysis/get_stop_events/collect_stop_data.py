from neural_data_analysis.neural_analysis_tools.align_trials import time_resolved_regression, time_resolved_gpfa_regression, plot_time_resolved_regression
from elephant.gpfa import gpfa_core, gpfa_util
from mpl_toolkits.mplot3d import Axes3D
from elephant.gpfa import GPFA
from elephant.spike_train_generation import inhomogeneous_poisson_process
import quantities as pq
from scipy.integrate import odeint
from importlib import reload
import neo
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pstats
import cProfile
from numpy import pi
import torch
from scipy import sparse
from scipy.io import loadmat
from scipy.signal import fftconvolve
from scipy import linalg, interpolate
from matplotlib import rc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import subprocess
import gc
import math
import sys
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import get_stops_utils, collect_stop_data
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import core_stops_psth, psth_postprocessing, psth_stats, compare_events, dpca_utils, prep_stop_psth_data
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from neural_data_analysis.neural_analysis_tools.align_trials import align_trial_utils
from neural_data_analysis.neural_analysis_tools.gpfa_methods import elephant_utils, fit_gpfa_utils, plot_gpfa_utils, gpfa_helper_class
from planning_analysis.show_planning import nxt_ff_utils, show_planning_utils
from machine_learning.ml_methods import regression_utils, regz_regression_utils, ml_methods_class, classification_utils, ml_plotting_utils, ml_methods_utils
from neural_data_analysis.neural_analysis_tools.cca_methods.cca_plotting import cca_plotting, cca_plot_lag_vs_no_lag, cca_plot_cv
from neural_data_analysis.neural_analysis_tools.cca_methods import cca_class, cca_utils, cca_cv_utils
from neural_data_analysis.neural_analysis_tools.cca_methods import cca_class
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import planning_and_neural_class, pn_utils, pn_helper_class, pn_aligned_by_seg, pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_neural_data, plot_modeling_result
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from visualization.matplotlib_tools import plot_behaviors_utils
from pattern_discovery import pattern_by_trials, pattern_by_trials, cluster_analysis, organize_patterns_and_features
from data_wrangling import specific_utils, process_monkey_information, general_utils
import os
import sys
import sys
from pathlib import Path
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break


# Third-party imports

# Machine Learning imports

# Neuroscience specific imports

# To fit gpfa

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)


def collect_stop_data_func(raw_data_folder_path):
    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path)
    pn.retrieve_neural_data()

    if not hasattr(pn, 'spikes_df'):
        pn.retrieve_or_make_monkey_data()
        pn.spikes_df = neural_data_processing.make_spikes_df(pn.raw_data_folder_path, pn.ff_caught_T_sorted,
                                                             sampling_rate=pn.sampling_rate)

    pn.make_or_retrieve_stop_category_df()

    captures_df, valid_captures_df, filtered_no_capture_stops_df, stops_with_stats = get_stops_utils.prepare_no_capture_and_captures(
        monkey_information=pn.monkey_information,
        closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
        ff_caught_T_new=pn.ff_caught_T_new,
        distance_col="distance_from_ff_to_stop",
    )

    columns_to_add = ["stop_id", "stop_id_duration",
                      "stop_id_start_time", "stop_id_end_time"]

    pn.make_one_stop_w_ff_df()
    one_stop_miss = pn.one_stop_w_ff_df[[
        'first_stop_point_index', 'first_stop_time', 'latest_visible_ff']].copy()
    one_stop_miss.rename(columns={
        'first_stop_point_index': 'stop_point_index', 'first_stop_time': 'stop_time'}, inplace=True)
    one_stop_miss[columns_to_add] = pn.monkey_information.loc[one_stop_miss['stop_point_index'],
                                                              columns_to_add].values

    pn.make_or_retrieve_stop_category_df()
    pn.get_try_a_few_times_info()
    pn.get_give_up_after_trying_info()

    columns_to_add = ["stop_id", "stop_id_duration",
                      "stop_id_start_time", "stop_id_end_time"]
    shared_columns = ["stop_point_index", "stop_time"] + columns_to_add

    # --- Build expanded + ordered tables for GUAT / TAFT ---
    GUAT_expanded = get_stops_utils._expand_trials(
        pn.GUAT_trials_df, pn.monkey_information)
    TAFT_expanded = get_stops_utils._expand_trials(
        pn.TAFT_trials_df, pn.monkey_information)

    # add stop_id to GUAT_trials_df and TAFT_trials_df
    GUAT_expanded[columns_to_add] = pn.monkey_information.loc[GUAT_expanded['stop_point_index'],
                                                              columns_to_add].values
    TAFT_expanded[columns_to_add] = pn.monkey_information.loc[TAFT_expanded['stop_point_index'],
                                                              columns_to_add].values

    GUAT = get_stops_utils._add_cluster_ordering(GUAT_expanded)
    TAFT = get_stops_utils._add_cluster_ordering(TAFT_expanded)

    # --- Per-cluster slices (consistent, vectorized) ---
    # First stop in each cluster
    GUAT_first = GUAT[GUAT["is_first"]].reset_index(drop=True)
    TAFT_first = TAFT[TAFT["is_first"]].reset_index(drop=True)

    # Last stop in each cluster
    GUAT_last = GUAT[GUAT["is_last"]].reset_index(drop=True)
    TAFT_last = TAFT[TAFT["is_last"]].reset_index(drop=True)

    # Middle stops (exclude first and last)
    GUAT_middle = GUAT[GUAT["is_middle"]].reset_index(drop=True)
    TAFT_middle = TAFT[TAFT["is_middle"]].reset_index(drop=True)

    # “First several” = all but the last stop in each cluster
    GUAT_nonfinal = GUAT[GUAT["order_in_cluster"] <
                         GUAT["cluster_size"] - 1].reset_index(drop=True)
    TAFT_nonfinal = TAFT[TAFT["order_in_cluster"] <
                         TAFT["cluster_size"] - 1].reset_index(drop=True)

    # Combine the “first several” from both, keep only columns you care about, then sort by index
    retry_after_miss = (
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

    both_first_miss = pd.concat([GUAT_first[shared_columns],
                                 TAFT_first[shared_columns]])

    both_middle = pd.concat([GUAT_middle[shared_columns],
                            TAFT_middle[shared_columns]])

    # Optional: if you also want “last several” (all but the first), it’s symmetrical:
    # GUAT_last_several = GUAT[GUAT["order_in_cluster"] > 0].reset_index(drop=True)
    # TAFT_last_several = TAFT[TAFT["order_in_cluster"] > 0].reset_index(drop=True)

    switch_after_miss = pd.concat([GUAT_last[shared_columns],
                                   one_stop_miss[shared_columns]])

    all_misses = pd.concat([one_stop_miss[shared_columns],
                            GUAT_expanded[shared_columns],
                            TAFT_nonfinal[shared_columns]
                            ])

    all_first_misses = pd.concat(
        [one_stop_miss[shared_columns],
            GUAT_first[shared_columns], TAFT_first[shared_columns]],
        ignore_index=True
    )

    all_last_misses = pd.concat(
        [one_stop_miss[shared_columns], GUAT_last[shared_columns]],
        ignore_index=True
    )

    # captures not in TAFT last (assuming TAFT_last is a subset of captures)
    captures_minus_TAFT_last = compare_events.diff_by(
        valid_captures_df, TAFT_last, key='stop_id')

    # non-captures excluding those flagged as 'all_misses'
    non_captures_minus_all_misses = compare_events.diff_by(
        filtered_no_capture_stops_df, all_misses, key='stop_id')

    datasets_raw = {
        'captures': valid_captures_df.copy(),
        'no_capture': filtered_no_capture_stops_df.copy(),
        'retry_after_miss': retry_after_miss.copy(),
        'both_middle': both_middle.copy(),
        'TAFT_first': TAFT_first.copy(),
        'GUAT_first': GUAT_first.copy(),
        'GUAT_middle': GUAT_middle.copy(),
        'TAFT_middle': TAFT_middle.copy(),
        'GUAT_last': GUAT_last.copy(),
        'TAFT_last': TAFT_last.copy(),
        'GUAT_nonfinal': GUAT_nonfinal.copy(),
        'TAFT_nonfinal': TAFT_nonfinal.copy(),
        'one_stop_miss': one_stop_miss.copy(),
        'both_first_miss': both_first_miss.copy(),
        'switch_after_miss': switch_after_miss.copy(),
        'captures_minus_TAFT_last': captures_minus_TAFT_last.copy(),
        'all_misses': all_misses.copy(),
        'non_captures_minus_all_misses': non_captures_minus_all_misses.copy(),
        'all_first_misses': all_first_misses.copy(),
        'all_last_misses': all_last_misses.copy(),
    }

    comparisons = compare_events.build_comparisons([

        {'a': 'GUAT_first', 'b': 'TAFT_first',
         'key': 'guat_first_vs_taft_first',
         'title': 'GUAT First vs TAFT First'},

        {'a': 'GUAT_middle', 'b': 'TAFT_middle',
         'key': 'guat_middle_vs_taft_middle',
         'title': 'GUAT Middle vs TAFT Middle'},

        {'a': 'GUAT_last', 'b': 'TAFT_last',
         'key': 'guat_last_vs_taft_last',
         'title': 'GUAT Last vs TAFT Last'},

        {'a': 'GUAT_middle', 'b': 'GUAT_last',
         'key': 'guat_middle_vs_guat_last',
         'title': 'GUAT Middle vs GUAT Last'},

        # ========= Maybe less interpretable because of the confounds in stops & captures surrounding the current stop =========

        {'a': 'switch_after_miss', 'b': 'retry_after_miss',
         'key': 'switch_vs_retry_after_miss',
         'title': 'Switch vs Retry After Miss'},

        {'a': 'GUAT_last', 'b': 'both_middle',
         'key': 'switch_vs_retry_after_retry',
         'title': 'Switch vs Retry After Retry'},

         # ==================


        {'a': 'one_stop_miss', 'b': 'both_first_miss',
         'key': 'first_giveup_vs_first_persist',
         'title': 'First-Attempt Give-up vs First-Attempt Persist'},



        {'a': 'captures', 'b': 'no_capture', 'key': 'captures_vs_no_capture',
            'title': 'Capture vs No Capture'},

            
        {'a': 'captures', 'b': 'all_misses',
            'key': 'captures_vs_all_misses', 'title': 'Capture vs Miss'},

        {'a': 'captures_minus_TAFT_last', 'b': 'TAFT_last',
         'key': 'first_vs_eventual_capture',
         'title': 'First-Shot Capture vs Eventual Capture'},

        {'a': 'one_stop_miss', 'b': 'GUAT_last',
         'key': 'first_giveup_vs_retry_then_giveup',
         'title': 'First-Attempt Give-up vs Retry then Give-up'},


        {'a': 'one_stop_miss', 'b': 'GUAT_first',
         'key': 'first_giveup_vs_first_GUAT',
         'title': 'First-Attempt Give-up vs First-Attempt GUAT'},


        {'a': 'GUAT_nonfinal', 'b': 'TAFT_nonfinal',
         'key': 'guat_persist_vs_taft_persist',
         'title': 'GUAT Early Persists vs TAFT Early Persists'},

        {'a': 'all_misses', 'b': 'non_captures_minus_all_misses',
         'key': 'non_attempts_vs_all_misses',
         'title': 'Miss vs Non-Attempt'},

        {'a': 'non_captures_minus_all_misses', 'b': 'one_stop_miss',
         'key': 'non_attempts_vs_single_miss',
         'title': 'Non-Attempt vs Single-Attempt Miss'},

        {'a': 'non_captures_minus_all_misses', 'b': 'all_first_misses',
         'key': 'non_attempts_vs_first_miss',
         'title': 'Non-Attempt vs First-Attempt Miss'},

        {'a': 'non_captures_minus_all_misses', 'b': 'all_last_misses',
         'key': 'non_attempts_vs_last_misses',
         'title': 'Non-Attempt vs Last-Attempt Miss'},
    ])


    datasets = {k: compare_events.dedupe_within(compare_events.ensure_event_schema(v)) for k, v in datasets_raw.items()}

    return pn, datasets, comparisons
