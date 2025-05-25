import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class, specific_utils, general_utils
from non_behavioral_analysis.neural_data_analysis.model_neural_data import neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from non_behavioral_analysis.neural_data_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data, neural_vs_behavioral_class
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
from null_behaviors import curvature_utils, curv_of_traj_utils
import warnings
import os
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


def find_single_vis_target_df(target_clust_last_vis_df, monkey_information, ff_caught_T_new):
    # check if target_clust_last_vis_df['nearby_vis_ff_indices'] is a string
    if isinstance(target_clust_last_vis_df['nearby_vis_ff_indices'].iloc[0], str):
        target_clust_last_vis_df['nearby_vis_ff_indices'] = target_clust_last_vis_df['nearby_vis_ff_indices'].apply(
            lambda x: [int(i) for i in x.strip('[]').split(',') if i.strip().isdigit()])

    target_clust_last_vis_df['num_nearby_vis_ff'] = target_clust_last_vis_df['nearby_vis_ff_indices'].apply(
        lambda x: len(x))

    # add ff_caught_time and ff_caught_point_index
    target_clust_last_vis_df['ff_caught_time'] = ff_caught_T_new[target_clust_last_vis_df['target_index'].values]
    target_clust_last_vis_df['ff_caught_point_index'] = np.searchsorted(
        monkey_information['time'], target_clust_last_vis_df['ff_caught_time'].values)

    single_vis_target_df = target_clust_last_vis_df[
        target_clust_last_vis_df['num_nearby_vis_ff'] == 1].copy()

    single_vis_target_df['last_vis_time'] = monkey_information.loc[
        single_vis_target_df['last_vis_point_index'].values, 'time'].values

    # print percentage of single_vis_target_df
    print("Percentage of targets not in a visible cluster out of all targets", len(
        single_vis_target_df) / len(target_clust_last_vis_df) * 100)
    return single_vis_target_df
