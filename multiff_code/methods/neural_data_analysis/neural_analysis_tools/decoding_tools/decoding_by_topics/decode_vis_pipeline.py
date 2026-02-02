import os
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
import json

# Machine Learning imports
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.multivariate.cancorr import CanCorr
import statsmodels.api as sm

# Neuroscience specific imports
import neo
import rcca
import quantities as pq
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA, gpfa_core, gpfa_util
from mpl_toolkits.mplot3d import Axes3D

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
from decision_making_analysis.event_detection import detect_rsw_and_rcap

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import core_stops_psth, psth_postprocessing, psth_stats, compare_events, dpca_utils
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import get_stops_utils, collect_stop_data

from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import general_glm_fit, cv_stop_glm, glm_fit_utils, variance_explained, glm_runner
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_plotting import plot_spikes, plot_glm_fit, plot_tuning_func, compare_glm_fit
from neural_data_analysis.design_kits.design_around_event import event_binning, stop_design, cluster_design, design_checks
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_hyperparams import compare_glm_configs, glm_hyperparams_class
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import ff_vis_epochs, vis_design

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import prepare_stop_design

# import decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.event_decoding import decoding_utils, decoding_analysis, plot_decoding, cmp_decode, load_results
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import pn_decoding_model_specs
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import assemble_stop_design
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis

# Additional imports
from importlib import reload
from scipy.integrate import odeint


class FFVisDecodingRunner:
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        t_max=0.20,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.t_max = t_max

        # will be filled during setup
        self.pn = None
        self.datasets = None
        self.comparisons = None

    def _collect_data(self):
        self.pn, self.datasets, self.comparisons = (
            collect_stop_data.collect_stop_data_func(
                self.raw_data_folder_path
            )
        )

        self.pn.make_or_retrieve_ff_dataframe()

    def _prepare_design_matrices(self):
        new_seg_info, events_with_stats = decode_vis.prepare_new_seg_info(
            self.pn.ff_dataframe,
            self.pn.bin_width,
        )

        (
            binned_spikes,
            binned_feats,
            offset_log,
            meta_used,
            meta_groups,
        ) = prepare_stop_design.build_stop_design(
            new_seg_info,
            events_with_stats,
            self.pn.monkey_information,
            self.pn.spikes_df,
            self.pn.ff_dataframe,
            datasets=self.datasets,
            bin_dt=self.pn.bin_width,
            add_ff_visible_info=True,
            add_retries_info=False,
        )

        if 'global_burst_id' not in meta_used.columns:
            meta_used = meta_used.merge(
                new_seg_info[['event_id', 'global_burst_id']],
                on='event_id',
                how='left',
            )

        binned_feats_sc, scaled_cols = event_binning.selective_zscore(
            binned_feats
        )
        binned_feats_sc = sm.add_constant(
            binned_feats_sc,
            has_constant='add',
        )

        bin_df = spike_history.make_bin_df_from_stop_meta(meta_used)

        (
            spike_data_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=binned_spikes,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        return spike_data_w_history, binned_feats_sc, meta_used

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'ff_vis_decoding',
        )

    def run(self, n_splits=5):
        self._collect_data()

        (
            spike_data_w_history,
            binned_feats_sc,
            meta_used,
        ) = self._prepare_design_matrices()

        save_dir = self._get_save_dir()

        all_results = []

        for model_name, spec in pn_decoding_model_specs.MODEL_SPECS.items():
            config = cv_decoding.DecodingRunConfig(
                # regression
                regression_model_class=spec.get(
                    'regression_model_class', None
                ),
                regression_model_kwargs=spec.get(
                    'regression_model_kwargs', {}
                ),
                # classification
                classification_model_class=spec.get(
                    'classification_model_class', None
                ),
                classification_model_kwargs=spec.get(
                    'classification_model_kwargs', {}
                ),
                # shared
                use_early_stopping=False,
            )

            print('model_name:', model_name)
            print('config:', config)

            results_df = cv_decoding.run_cv_decoding(
                X=spike_data_w_history,
                y_df=binned_feats_sc,
                behav_features=None,
                groups=meta_used['event_id'].values,
                n_splits=n_splits,
                config=config,
                context_label='pooled',
                save_dir=save_dir,
            )

            results_df['model_name'] = model_name
            all_results.append(results_df)

        all_results_df = pd.concat(all_results, ignore_index=True)
        return all_results_df
