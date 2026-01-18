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
from decision_making_analysis.event_detection import detect_rsw_and_rcap
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


def collect_stop_data_func(raw_data_folder_path, bin_width=0.04):
    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path, bin_width=bin_width)
    pn.retrieve_neural_data()

    if not hasattr(pn, 'spikes_df'):
        pn.retrieve_or_make_monkey_data()
        pn.spikes_df = neural_data_processing.make_spikes_df(pn.raw_data_folder_path, pn.ff_caught_T_sorted,
                                                             pn.monkey_information, sampling_rate=pn.sampling_rate)

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
        'stop_1_point_index', 'stop_1_time', 'candidate_target']].copy()
    one_stop_miss.rename(columns={
        'stop_1_point_index': 'stop_point_index', 'stop_1_time': 'stop_time'}, inplace=True)
    one_stop_miss[columns_to_add] = pn.monkey_information.loc[one_stop_miss['stop_point_index'],
                                                              columns_to_add].values

    pn.make_or_retrieve_stop_category_df()
    pn.get_retry_capture_info()
    pn.get_retry_switch_info()

    columns_to_add = ["stop_id", "stop_id_duration",
                      "stop_id_start_time", "stop_id_end_time"]
    shared_columns = ["stop_point_index", "stop_time"] + columns_to_add

    # --- Build expanded + ordered tables for rsw / rcap ---
    rsw_expanded = get_stops_utils._expand_trials(
        pn.rsw_events_df, pn.monkey_information)
    rcap_expanded = get_stops_utils._expand_trials(
        pn.rcap_events_df, pn.monkey_information)

    # add stop_id to rsw_events_df and rcap_events_df
    rsw_expanded[columns_to_add] = pn.monkey_information.loc[rsw_expanded['stop_point_index'],
                                                             columns_to_add].values
    rcap_expanded[columns_to_add] = pn.monkey_information.loc[rcap_expanded['stop_point_index'],
                                                              columns_to_add].values

    rsw = get_stops_utils._add_cluster_ordering(rsw_expanded)
    rcap = get_stops_utils._add_cluster_ordering(rcap_expanded)

    # --- Per-cluster slices (consistent, vectorized) ---
    # First stop in each cluster
    rsw_first = rsw[rsw["is_first"]].reset_index(drop=True)
    rcap_first = rcap[rcap["is_first"]].reset_index(drop=True)

    # Last stop in each cluster
    rsw_last = rsw[rsw["is_last"]].reset_index(drop=True)
    rcap_last = rcap[rcap["is_last"]].reset_index(drop=True)

    # Middle stops (exclude first and last)
    rsw_middle = rsw[rsw["is_middle"]].reset_index(drop=True)
    rcap_middle = rcap[rcap["is_middle"]].reset_index(drop=True)

    # “First several” = all but the last stop in each cluster
    rsw_nonfinal = rsw[rsw["order_in_cluster"] <
                       rsw["cluster_size"] - 1].reset_index(drop=True)
    rcap_nonfinal = rcap[rcap["order_in_cluster"] <
                         rcap["cluster_size"] - 1].reset_index(drop=True)

    # Combine the “first several” from both, keep only columns you care about, then sort by index
    will_retry_after_miss = (
        pd.concat(
            [rsw_nonfinal[shared_columns],
             rcap_nonfinal[shared_columns],
             ],
            ignore_index=True
        )
        .sort_values("stop_point_index")
        .reset_index(drop=True)
    )

    both_first_miss = pd.concat([rsw_first[shared_columns],
                                 rcap_first[shared_columns]])

    both_middle = pd.concat([rsw_middle[shared_columns],
                            rcap_middle[shared_columns]])

    # Optional: if you also want “last several” (all but the first), it’s symmetrical:
    # rsw_last_several = rsw[rsw["order_in_cluster"] > 0].reset_index(drop=True)
    # rcap_last_several =  rcap[ rcap["order_in_cluster"] > 0].reset_index(drop=True)

    will_switch_after_miss = pd.concat([rsw_last[shared_columns],
                                        one_stop_miss[shared_columns]])

    all_misses = pd.concat([one_stop_miss[shared_columns],
                            rsw_expanded[shared_columns],
                            rcap_nonfinal[shared_columns]
                            ])

    all_first_misses = pd.concat(
        [one_stop_miss[shared_columns],
            rsw_first[shared_columns], rcap_first[shared_columns]],
        ignore_index=True
    )

    all_last_misses = pd.concat(
        [one_stop_miss[shared_columns], rsw_last[shared_columns]],
        ignore_index=True
    )

    # captures not in rcap last (assuming rcap_last is a subset of captures)
    captures_minus_rcap_last = compare_events.diff_by(
        valid_captures_df, rcap_last, key='stop_id')

    # non-captures excluding those flagged as 'all_misses'
    non_captures_minus_all_misses = compare_events.diff_by(
        filtered_no_capture_stops_df, all_misses, key='stop_id')

    datasets_raw = {
        'captures': valid_captures_df.copy(),
        'no_capture': filtered_no_capture_stops_df.copy(),
        # note: will_retry_after_miss excludes rcap_last
        'will_retry_after_miss': will_retry_after_miss.copy(),
        'both_middle': both_middle.copy(),
        'rcap_first': rcap_first.copy(),
        'rsw_first': rsw_first.copy(),
        'rsw_middle': rsw_middle.copy(),
        'rcap_middle': rcap_middle.copy(),
        'rsw_last': rsw_last.copy(),
        'rcap_last': rcap_last.copy(),
        'rsw_nonfinal': rsw_nonfinal.copy(),
        'rcap_nonfinal': rcap_nonfinal.copy(),
        'one_stop_miss': one_stop_miss.copy(),
        'both_first_miss': both_first_miss.copy(),
        'will_switch_after_miss': will_switch_after_miss.copy(),
        'captures_minus_rcap_last': captures_minus_rcap_last.copy(),
        'all_misses': all_misses.copy(),
        'non_captures_minus_all_misses': non_captures_minus_all_misses.copy(),
        'all_first_misses': all_first_misses.copy(),
        'all_last_misses': all_last_misses.copy(),
        'rsw': rsw.copy(),
        'rcap': rcap.copy(),
    }

    comparisons = compare_events.build_comparisons([

        {'a': 'rsw_first', 'b': 'rcap_first',
         'key': 'rsw_first_vs_rcap_first',
         'title': 'rsw First vs rcap First'},

        {'a': 'rsw_middle', 'b': 'rcap_middle',
         'key': 'rsw_middle_vs_rcap_middle',
         'title': 'rsw Middle vs rcap Middle'},

        {'a': 'rsw_last', 'b': 'rcap_last',
         'key': 'rsw_last_vs_rcap_last',
         'title': 'rsw Last vs rcap Last'},

        {'a': 'rsw_middle', 'b': 'rsw_last',
         'key': 'rsw_middle_vs_rsw_last',
         'title': 'rsw Middle vs rsw Last'},

        # ========= Maybe less interpretable because of the confounds in stops & captures surrounding the current stop =========

        {'a': 'will_switch_after_miss', 'b': 'will_retry_after_miss',  # will_retry_after_miss excludes rcap_last
         'key': 'will_switch_vs_retry_after_miss',
         'title': 'Switch vs Retry After Miss'},

        {'a': 'rsw_last', 'b': 'both_middle',
         'key': 'will_switch_vs_retry_after_retry_miss',
         'title': 'Switch vs Retry After Retry'},

        # ==================


        {'a': 'one_stop_miss', 'b': 'both_first_miss',
         'key': 'one_stop_vs_both_first_miss',
         'title': 'First-Attempt Give-up vs First-Attempt Persist'},

        {'a': 'one_stop_miss', 'b': 'rsw_last',
         'key': 'one_stop_vs_rsw_last',
         'title': 'First-Attempt Give-up vs Retry then Give-up'},

        {'a': 'one_stop_miss', 'b': 'rsw_first',
         'key': 'one_stop_vs_rsw_first',
         'title': 'First-Attempt Give-up vs First-Attempt rsw'},

        {'a': 'rsw_nonfinal', 'b': 'rcap_nonfinal',
         'key': 'rsw_nonfinal_vs_rcap_nonfinal',
         'title': 'rsw Early Persists vs rcap Early Persists'},

        {'a': 'captures', 'b': 'no_capture',
         'key': 'captures_vs_no_capture',
         'title': 'Capture vs No Capture'},

        {'a': 'captures', 'b': 'all_misses',
         'key': 'captures_vs_all_misses',
         'title': 'Capture vs Miss'},

        {'a': 'captures_minus_rcap_last', 'b': 'rcap_last',
         'key': 'captures_minus_rcap_last_vs_rcap_last',
         'title': 'First-Shot Capture vs Eventual Capture'},

        {'a': 'all_misses', 'b': 'non_captures_minus_all_misses',
         'key': 'miss_vs_non_attemtp',
         'title': 'Miss vs Non-Attempt'},

        {'a': 'non_captures_minus_all_misses', 'b': 'one_stop_miss',
         'key': 'non_attempt_vs_one_stop_miss',
         'title': 'Non-Attempt vs Single-Attempt Miss'},

        {'a': 'non_captures_minus_all_misses', 'b': 'all_first_misses',
         'key': 'non_attempts_vs_first_miss',
         'title': 'Non-Attempt vs First-Attempt Miss'},

        {'a': 'non_captures_minus_all_misses', 'b': 'all_last_misses',
         'key': 'non_attempt_vs_last_miss',
         'title': 'Non-Attempt vs Last-Attempt Miss'},
    ])

    datasets = {k: compare_events.dedupe_within(
        compare_events.ensure_event_schema(v)) for k, v in datasets_raw.items()}

    return pn, datasets, comparisons
