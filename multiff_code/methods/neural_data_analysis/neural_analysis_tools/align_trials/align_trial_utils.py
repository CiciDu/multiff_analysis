import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.topic_based_neural_analysis.target_decoder import prep_target_decoder, behav_features_to_keep
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from os.path import exists
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_rebinned_var_lags(rebinned_var, trial_vector, rebinned_max_lag_number=2):
    lag_numbers = np.arange(-rebinned_max_lag_number,
                            rebinned_max_lag_number+1)
    rebinned_var_lags = neural_data_processing.add_lags_to_each_feature(
        rebinned_var, lag_numbers, trial_vector=trial_vector)

    if 'new_bin_0' in rebinned_var_lags.columns:
        rebinned_var_lags['new_bin'] = rebinned_var_lags['new_bin_0'].astype(
            int)
        rebinned_var_lags = rebinned_var_lags.drop(
            columns=[col for col in rebinned_var_lags.columns if 'new_bin_' in col])
    if 'new_segment_0' in rebinned_var_lags.columns:
        rebinned_var_lags['new_segment'] = rebinned_var_lags['new_segment_0'].astype(
            int)
        rebinned_var_lags = rebinned_var_lags.drop(
            columns=[col for col in rebinned_var_lags.columns if 'new_segment_' in col])

    assert rebinned_var_lags['new_bin'].equals(
        rebinned_var['new_bin'])

    return rebinned_var_lags
