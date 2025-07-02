import sys
from data_wrangling import process_monkey_information, specific_utils, further_processing_class
from neural_data_analysis.neural_analysis_tools.model_neural_data import transform_vars, neural_data_modeling, drop_high_corr_vars, drop_high_vif_vars
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
from neural_data_analysis.neural_analysis_by_topic.neural_vs_behavioral import prep_monkey_data, prep_monkey_data, prep_monkey_data, prep_target_data
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.cca_methods import cca_utils, cca_cv_utils
from neural_data_analysis.neural_analysis_tools.cca_methods.cca_plotting import cca_plotting, cca_plot_lag_vs_no_lag, cca_plot_cv

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
from sklearn.cross_decomposition import CCA
import rcca
from sklearn.preprocessing import StandardScaler


class CCAclass():

    def __init__(self, X1, X2, lagging_included=False):
        self.X1 = X1
        self.X2 = X2
        self.lagging_included = lagging_included

        # Scale data
        self.scaler = StandardScaler()
        self.X1_sc, self.X2_sc = self.scaler.fit_transform(
            self.X1), self.scaler.fit_transform(self.X2)

        self.X1_sc_df = pd.DataFrame(self.X1_sc, columns=self.X1.columns)
        self.X2_sc_df = pd.DataFrame(self.X2_sc, columns=self.X2.columns)

    def conduct_cca(self, n_components=10, plot_correlations=True):
        self.n_components = n_components
        self.cca, self.X1_c, self.X2_c, self.canon_corr = neural_data_modeling.conduct_cca(
            self.X1_sc, self.X2_sc, n_components=n_components, plot_correlations=plot_correlations)
        self.X1_weights = self.cca.ws[0]
        self.X2_weights = self.cca.ws[1]
        self.X1_loading = neural_data_modeling.calculate_loadings(
            self.X1_sc, self.X1_c)
        self.X2_loading = neural_data_modeling.calculate_loadings(
            self.X2_sc, self.X2_c)
        self.get_weight_df()
        self.get_loading_df()

        # also put the results into one dict
        self.results = {
            'X1_canon_vars': self.X1_c,
            'X2_canon_vars': self.X2_c,
            'canon_corr': self.canon_corr,
            'X1_weights': self.X1_weights,
            'X2_weights': self.X2_weights,
            'X1_loading': self.X1_loading,
            'X2_loading': self.X2_loading,
        }

    def get_loading_df(self):
        self.X1_loading_df = neural_data_modeling.make_loading_or_weight_df(
            self.X1_loading, self.X1.columns, lagging_included=False)
        self.X2_loading_df = neural_data_modeling.make_loading_or_weight_df(
            self.X2_loading, self.X2.columns, lagging_included=self.lagging_included)

    def get_weight_df(self):
        self.X1_weight_df = neural_data_modeling.make_loading_or_weight_df(
            self.X1_weights, self.X1.columns, lagging_included=False)
        self.X2_weight_df = neural_data_modeling.make_loading_or_weight_df(
            self.X2_weights, self.X2.columns, lagging_included=self.lagging_included)

    def get_squared_loading_df(self):
        self.X1_squared_loading_df = neural_data_modeling.make_loading_or_weight_df(
            self.X1_loading**2, self.X1.columns, lagging_included=False)
        self.X2_squared_loading_df = neural_data_modeling.make_loading_or_weight_df(
            self.X2_loading**2, self.X2.columns, lagging_included=self.lagging_included)


    def plot_X1_loadings(self, max_components=20, features_per_fig=30):
        cca_plotting.plot_loading_heatmap(
            loadings=self.X1_loading,
            feature_names=self.X1_loading_df.feature.values,
            matrix_label='X1',
            max_components=max_components,
            features_per_fig=features_per_fig,
            canonical_corrs = self.canon_corr
        )
        

    def plot_X2_loadings(self, max_components=20, features_per_fig=30):
        cca_plotting.plot_loading_heatmap(
            loadings=self.X2_loading,
            feature_names=self.X2_loading_df.feature.values,
            matrix_label='X2',
            max_components=max_components,
            features_per_fig=features_per_fig,
            canonical_corrs = self.canon_corr
        )
        

    def plot_ranked_loadings(self, keep_one_value_for_each_feature=False, max_plots_to_show=5, max_features_to_show_per_plot=10,
                             X1_or_X2='X1', squared=True):
        if squared:
            if not hasattr(self, 'X1_squared_loading_df'):
                self.get_squared_loading_df()
            loading_df = self.X1_squared_loading_df if X1_or_X2 == 'X1' else self.X2_squared_loading_df
            squared = True
        else:
            if not hasattr(self, 'X1_loading_df'):
                self.get_loading_df()
            loading_df = self.X1_loading_df if X1_or_X2 == 'X1' else self.X2_loading_df
            squared = False
        num_variates = self.X1_loading.shape[1] if X1_or_X2 == 'X1' else self.X2_loading.shape[1]

        cca_plotting.make_a_series_of_barplots_of_ranked_loadings_or_weights(loading_df, self.canon_corr, num_variates,
                                                                             max_plots_to_show=max_plots_to_show,
                                                                             keep_one_value_for_each_feature=keep_one_value_for_each_feature,
                                                                             max_features_to_show_per_plot=max_features_to_show_per_plot,
                                                                             squared=squared
                                                                             )

    def plot_ranked_weights(self, keep_one_value_for_each_feature=False, max_plots_to_show=5, max_features_to_show_per_plot=10,
                            X1_or_X2='X1', abs_value=True):
        if not hasattr(self, 'X1_weight_df'):
            self.get_weight_df()
        weight_df = self.X1_weight_df.copy() if X1_or_X2 == 'X1' else self.X2_weight_df.copy()
        num_variates = self.X1_weights.shape[1] if X1_or_X2 == 'X1' else self.X2_weights.shape[1]

        if abs_value:
            # Apply abs() only to columns that are not of type str
            numeric_columns = weight_df.select_dtypes(
                include=[np.number]).columns
            weight_df[numeric_columns] = weight_df[numeric_columns].abs()

        cca_plotting.make_a_series_of_barplots_of_ranked_loadings_or_weights(weight_df, self.canon_corr, num_variates,
                                                                             max_plots_to_show=max_plots_to_show,
                                                                             keep_one_value_for_each_feature=keep_one_value_for_each_feature,
                                                                             max_features_to_show_per_plot=max_features_to_show_per_plot
                                                                             )
