# edited from cluster
from pathlib import Path
from neural_data_analysis.neural_analysis_tools.pgam_tools import pgam_utils
import sys
import os

def find_project_root(marker="multiff_analysis"):
    """Search upward until we find a folder containing `marker`."""
    cur = Path(os.getcwd()).resolve()   # use CWD instead of __file__
    for parent in [cur] + list(cur.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'")

project_root = find_project_root()

# Build the paths relative to project root
pgam_src = project_root / "multiff_analysis" / "external" / "pgam" / "src"
pgam_src_pg = pgam_src / "PGAM"

for path in [pgam_src, pgam_src_pg]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

    
from sklearn.preprocessing import StandardScaler
import math
import os
import numpy as np
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import plot_modeling_result
from scipy.io import savemat
from post_processing import postprocess_results
import pandas as pd
import matplotlib.pylab as plt
import PGAM.gam_data_handlers as gdh
from PGAM.GAM_library import *
import sys


class PGAMclass():

    # temporal_vars = ['catching_ff', 'any_ff_visible', 'monkey_speeddummy',
    #                  'min_target_has_disappeared_for_last_time_dummy',
    #                  'min_target_cluster_has_disappeared_for_last_time_dummy',
    #                  'max_target_visible_dummy', 'max_target_cluster_visible_dummy',
    #                  'try_a_few_times_indice_dummy', 'give_up_after_trying_indice_dummy',
    #                  'ignore_sudden_flash_indice_dummy', 'two_in_a_row', 'waste_cluster_around_target',
    #                  'visible_before_last_one', 'disappear_latest', 'ignore_sudden_flash',
    #                  'try_a_few_times', 'give_up_after_trying', 'cluster_around_target',
    #                  ]

    temporal_vars = ['capture_ff', 'any_ff_visible', 'any_ff_in_memory', 'turning_right', 'stop', 'whether_test',
                    'cur_in_memory', 'nxt_in_memory', 'cur_vis', 'nxt_vis', 'target_cluster_has_disappeared_for_last_time_dummy']

    def __init__(self, x_var, y_var, bin_width, processed_neural_data_folder_path):
        self.x_var = x_var
        self.y_var = y_var
        self.bin_width = bin_width
        self.processed_neural_data_folder_path = processed_neural_data_folder_path

    def _categorize_features(self, temporal_vars):
        self.temporal_vars = [
            x for x in temporal_vars if x in self.y_var.columns]
        self.spatial_vars = [
            x for x in self.y_var.columns if x not in temporal_vars]
        print('Spatial variables:', np.array(self.spatial_vars))

        self.temporal_sub = self.y_var[self.temporal_vars]
        self.spatial_sub_unscaled = self.y_var[self.spatial_vars]

    def streamline_pgam(self, temporal_vars=None, neural_cluster_number=10, num_total_trials=10):
        self.prepare_for_pgam(temporal_vars, num_total_trials)
        self._add_temporal_features_to_model(plot_each_feature=False)
        self._add_spatial_features_to_model(plot_each_feature=False)
        self.run_pgam(neural_cluster_number=neural_cluster_number)
        self.post_processing_results()
        self.save_results()

    def prepare_for_pgam(self, temporal_vars=None, num_total_trials=10):
        if temporal_vars is None:
            temporal_vars = self.temporal_vars
        self._categorize_features(temporal_vars)
        self._scale_features()
        self._get_mock_trials_df(num_total_trials)
        self.sm_handler = gdh.smooths_handler()

    def run_pgam(self, neural_cluster_number=5):
        self.neural_cluster_number = neural_cluster_number
        link = sm.genmod.families.links.log()
        self.poissFam = sm.genmod.families.family.Poisson(link=link)
        self.spk_counts = self.x_var.iloc[:, neural_cluster_number].values
        self.cluster_name = self.x_var.columns[neural_cluster_number]

        # create the pgam model
        self.pgam = general_additive_model(self.sm_handler,
                                           self.sm_handler.smooths_var,  # list of covariate we want to include in the model
                                           self.spk_counts,  # vector of spike counts
                                           self.poissFam  # poisson family with exponential link from statsmodels.api
                                           )

        # with all covariate, remove according to stat testing, and then refit
        self.full, self.reduced = self.pgam.fit_full_and_reduced(self.sm_handler.smooths_var,
                                                                 th_pval=0.001,  # pval for significance of covariate icluseioon
                                                                 max_iter=10 ** 2,  # max number of iteration
                                                                 use_dgcv=True,  # learn the smoothing penalties by dgcv
                                                                 trial_num_vec=self.trial_ids,
                                                                 filter_trials=self.train_trials,
                                                                 )
        try:
            print('Minimal subset of variables driving the activity:')
            print(self.reduced.var_list)
        except Exception as e:
            print(f"Error occurred while printing reduced variable list: {e}")

    def post_processing_results(self):
        # string with the neuron identifier
        neuron_id = 'neuron_000_session_1_monkey_001'
        # dictionary containing some information about the neuron, keys must be strings and values can be anything since are stored with type object.
        info_save = {'x': 100,
                     'y': 801.2,
                     'brain_region': 'V1',
                     'subject': 'monkey_001'
                     }

        # assume that we used 90% of the trials for training, 10% for evaluation
        self.res = postprocess_results(neuron_id, self.spk_counts, self.full, self.reduced, self.train_trials, self.sm_handler, self.poissFam, self.trial_ids,
                                       var_zscore_par=None, info_save=info_save, bins=self.kernel_h_length)


    def plot_results(self, plot_vars_in_reduced_list_only=True):
        # find which variables in res['variable'] are in reduced.var_list
        # and then plot the corresponding x_rate_Hz
        if plot_vars_in_reduced_list_only:
            try:
                indices_of_vars_to_plot = np.where(
                    np.isin(self.res['variable'], self.reduced.var_list))[0]
            except Exception as e:
                print(f"Error occurred while plotting results: {e}. Skipping...")
                return
        else:
            indices_of_vars_to_plot = np.arange(self.res.shape[0])

        self._rename_variables_in_results()
        plot_modeling_result.plot_pgam_tuning_curvetions(
            self.res, indices_of_vars_to_plot=indices_of_vars_to_plot)



    def load_pgam_pgam_results(self, neural_cluster_number):
        self.cluster_name = self.x_var.columns[neural_cluster_number]
        
        self.res, self.reduced_vars, self.meta = pgam_utils.load_full_results_npz(self.processed_neural_data_folder_path,
                                                            self.cluster_name)


    def save_results(self):
                # after you compute self.res = postprocess_results(...):
        extra_meta = {
            "bin_width": float(self.bin_width),
            "neuron_index": int(self.neural_cluster_number),
            "trial_count": int(len(np.unique(self.trial_ids))),
            "reduced_AIC": float(getattr(self.reduced, "AIC", np.nan)) if hasattr(self.reduced, "AIC") else np.nan,
            "full_AIC": float(getattr(self.full, "AIC", np.nan)) if hasattr(self.full, "AIC") else np.nan,
        }
        pgam_utils.save_full_results_npz(self.processed_neural_data_folder_path,
                            self.cluster_name,
                            self.res,                       # the structured array
                            getattr(self.reduced, "var_list", []),
                            extra_meta)




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
        variable[variable ==
                 'target_cluster_has_disappeared_for_last_time_dummy'] = 'whether target cluster has disappeared for last time'
        self.res['variable'] = variable

    def _rename_variables_in_results(self):
        variable = self.res['variable']
        # rename each variable to the corresponding label
        variable[variable == 'catching_ff'] = 'caught firefly (this bin)'
        variable[variable == 'min_target_cluster_has_disappeared_for_last_time_dummy'] = 'target cluster disappeared (first time)'
        variable[variable == 'max_target_cluster_visible_dummy'] = 'target cluster visible'
        variable[variable == 'gaze_world_y'] = 'gaze position (y)'
        variable[variable == 'monkey_speed'] = 'monkey speed'
        variable[variable == 'monkey_dw'] = 'monkey acceleration'
        variable[variable == 'monkey_ddw'] = 'change in acceleration (jerk)'
        variable[variable == 'monkey_ddv'] = 'change in angular accel'
        variable[variable == 'avg_target_cluster_last_seen_distance'] = 'distance to last-seen target cluster'
        variable[variable == 'avg_target_cluster_last_seen_angle'] = 'angle to last-seen target cluster'
        variable[variable == 'target_cluster_has_disappeared_for_last_time_dummy'] = 'target cluster disappeared (final time)'
        self.res['variable'] = variable
