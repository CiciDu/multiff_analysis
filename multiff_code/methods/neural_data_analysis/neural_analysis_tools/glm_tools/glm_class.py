
import sys
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_feature_selection

import os
import sys
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler



class GLMclass():

    # for

    # for the original neural vs behavior analysis
    # temporal_vars = ['catching_ff', 'any_ff_visible', 'monkey_speeddummy',
    #                  'min_target_has_disappeared_for_last_time_dummy',
    #                  'min_target_cluster_has_disappeared_for_last_time_dummy',
    #                  'max_target_visible_dummy', 'max_target_cluster_visible_dummy',
    #                  'try_a_few_times_indice_dummy', 'give_up_after_trying_indice_dummy',
    #                  'ignore_sudden_flash_indice_dummy', 'two_in_a_row', 'waste_cluster_around_target',
    #                  'visible_before_last_one', 'disappear_latest', 'ignore_sudden_flash',
    #                  'try_a_few_times', 'give_up_after_trying', 'cluster_around_target',
    #                  ]

    def __init__(self, x_var, y_var, bin_width, processed_neural_data_folder_path):
        self.x_var = x_var
        self.y_var = y_var
        self.bin_width = bin_width
        self.processed_neural_data_folder_path = processed_neural_data_folder_path

    def streamline_pgam(self, temporal_vars=None, neural_cluster_number=10, num_total_trials=10):
        self.prepare_for_pgam(temporal_vars, num_total_trials)
        self._add_temporal_features_to_model(plot_each_feature=False)
        self._add_spatial_features_to_model(plot_each_feature=False)
        self.run_pgam(neural_cluster_number=neural_cluster_number)
        self.post_processing()
        self.save_results()

    def prepare_for_pgam(self, temporal_vars=None, num_total_trials=10):
        if temporal_vars is None:
            temporal_vars = self.temporal_vars
        self._categorize_features(temporal_vars)
        self._scale_features()
        self._get_mock_trials_df(num_total_trials)


    def _categorize_features(self, temporal_vars):
        # Keep only valid temporal variables that exist in y_var
        self.temporal_vars = [x for x in temporal_vars if x in self.y_var.columns]

        # Convert to set for subtraction
        rest_of_vars = set(pn_feature_selection.all_features) - set(self.temporal_vars)

        # Spatial vars are those in y_var that are in the "rest"
        self.spatial_vars = [x for x in self.y_var.columns if x in rest_of_vars]

        print("Spatial variables:", np.array(self.spatial_vars))

        # Sub-dataframes
        self.temporal_sub = self.y_var.loc[:, self.temporal_vars]
        self.spatial_sub_unscaled = self.y_var.loc[:, self.spatial_vars]


    def _scale_features(self):
        # since temporal variables are all dummy variables, we only need to scale the spatial variables
        scaler = StandardScaler()
        spatial_sub = scaler.fit_transform(self.spatial_sub_unscaled)
        self.spatial_sub = pd.DataFrame(
            spatial_sub, columns=self.spatial_sub_unscaled.columns)

    def _get_mock_trials_df(self, num_total_trials=10):
        self.num_total_trials = num_total_trials
        num_data_points = self.y_var.shape[0]
        num_repeats = math.ceil(num_data_points/num_total_trials)
        trial_ids = np.repeat(np.arange(num_total_trials), num_repeats)
        self.trial_ids = trial_ids[:num_data_points]
        # take out 2/3 of the trials for training
        self.train_trials = self.trial_ids % 3 != 1

    def _add_temporal_features_to_model(self,
                                        # Duration of the kernel h(t) in seconds
                                        kernel_time_window=10,
                                        num_internal_knots=8,  # Number of internal knots used to represent h
                                        # the order of the base spline, the number of coefficient in the polinomial (ord =4 is cubic spline)
                                        order=4,
                                        plot_each_feature=True,
                                        ):
        # Define the B-spline parameters

        # length in time points of the kernel
        self.kernel_h_length = int(kernel_time_window / self.bin_width)
        if self.kernel_h_length % 2 == 0:
            self.kernel_h_length += 1

        # Iterate over columns in the temporal subset
        for column in self.temporal_sub.columns:
            # Add the covariate & evaluate the convolution
            self.sm_handler.add_smooth(
                column,
                [self.temporal_sub[column].values],
                is_temporal_kernel=True,
                ord=order,
                knots_num=num_internal_knots,
                trial_idx=self.trial_ids,
                kernel_length=self.kernel_h_length,
                kernel_direction=0,
                time_bin=self.bin_width
            )

            if plot_each_feature:
                plot_modeling_result.plot_smoothed_temporal_feature(
                    self.temporal_sub, column, self.sm_handler, self.kernel_h_length)

    def _add_spatial_features_to_model(self,
                                       # the order of the base spline, the number of coefficient in the polinomial (ord =4 is cubic spline)
                                       order=4,
                                       plot_each_feature=True,
                                       ):

        # Add the 1D spatial variable
        for column in self.spatial_sub.columns:
            column_values = self.spatial_sub[column].values

            # Remove the variable from smooths_var and smooths_dict if it exists
            if column in self.sm_handler.smooths_var:
                self.sm_handler.smooths_var.remove(column)
                self.sm_handler.smooths_dict.pop(column)

            # Define internal knots and extend them for boundary conditions
            int_knots = np.linspace(min(column_values), max(column_values), 6)
            knots = np.hstack(([int_knots[0]] * (order - 1),
                              int_knots, [int_knots[-1]] * (order - 1)))

            # Add the smooth variable
            self.sm_handler.add_smooth(
                column, [column_values],
                knots=[knots],
                ord=order,
                is_temporal_kernel=False,
                trial_idx=self.trial_ids,
                is_cyclic=[False]
            )

            if plot_each_feature:
                plot_modeling_result.plot_smoothed_spatial_feature(
                    self.spatial_sub, column, self.sm_handler)

    def _rename_variables_in_results(self):
        variable = self.res['variable']
        # rename each variable to the corresponding label
        variable[variable ==
                 'catching_ff'] = 'whether catching ff at current time bin'
        variable[variable == 'min_target_cluster_has_disappeared_for_last_time_dummy'] = 'whether target cluster has disappeared for last time'
        variable[variable ==
                 'max_target_cluster_visible_dummy'] = 'whether target cluster is visible'
        variable[variable == 'gaze_world_y'] = 'gaze y-coordinate'
        variable[variable == 'monkey_speed'] = 'monkey linear speed'
        variable[variable == 'monkey_dw'] = 'monkey linear acceleration'
        variable[variable == 'monkey_ddw'] = 'change in monkey linear acceleration'
        variable[variable == 'monkey_ddv'] = 'change in monkey angular acceleration'
        variable[variable == 'avg_target_cluster_last_seen_distance'] = 'distance of target cluster last seen'
        variable[variable ==
                 'avg_target_cluster_last_seen_angle'] = 'angle of target cluster last seen'
        self.res['variable'] = variable
