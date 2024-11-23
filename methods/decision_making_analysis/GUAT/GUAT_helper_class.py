import sys
from decision_making_analysis.cluster_replacement import cluster_replacement_class, cluster_replacement_utils
from decision_making_analysis.decision_making import decision_making_utils
from decision_making_analysis.GUAT import GUAT_and_TAFT
from decision_making_analysis import free_selection, trajectory_info
from null_behaviors import curvature_utils, curv_of_traj_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)




class GUATHelperClass(cluster_replacement_class.ClusterReplacement):


    def process_current_and_alternative_ff_info(self, 
                                                num_old_ff_per_row=2, 
                                                num_new_ff_per_row=2, 
                                                selection_criterion_if_too_many_ff='time_since_last_visible'):

        # The following columns are added so the function further_process_df_related_to_cluster_replacement can be called without errors.
        # But the values in these columns will be updated later
        self.GUAT_current_ff_info[['whether_changed', 'whether_intended_target']] = False
        self.GUAT_alt_ff_info[['whether_changed', 'whether_intended_target']] = False    

        self.GUAT_current_ff_info, self.GUAT_alt_ff_info = cluster_replacement_utils.further_process_df_related_to_cluster_replacement(self.GUAT_current_ff_info, self.GUAT_alt_ff_info, \
            num_old_ff_per_row=num_old_ff_per_row, num_new_ff_per_row=num_new_ff_per_row, selection_criterion_if_too_many_ff=selection_criterion_if_too_many_ff)
        self.GUAT_alt_ff_info['order'] = self.GUAT_alt_ff_info['order'] + num_old_ff_per_row

        self.num_old_ff_per_row = num_old_ff_per_row
        self.num_new_ff_per_row = num_new_ff_per_row
        self.num_ff_per_row = num_old_ff_per_row + num_new_ff_per_row


    def find_input_and_output(self, 
                              add_arc_info=False, 
                              add_current_curvature_of_traj=False, 
                              curvature_df=None, 
                              curv_of_traj_df=None,
                              curv_of_traj_mode='time',
                              window_for_curv_of_traj=[-1, 1],
                              truncate_curv_of_traj_by_time_of_capture=False,
                              ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_visible', 'time_till_next_visible'],
                              arc_info_to_add=['optimal_curvature', 'curv_diff']):
        if curv_of_traj_df is None:
            self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_sorted, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            self.curv_of_traj_df = curv_of_traj_df

        attributes_for_plotting = ['ff_distance', 'ff_angle', 'time_since_last_visible']
        if 'time_till_next_visible' in ff_attributes:
            #self.add_time_till_next_visible()
            attributes_for_plotting.append('time_till_next_visible')

        if curvature_df is not None:
            if self.additional_curvature_df is not None:
                curvature_df = pd.concat([curvature_df, self.additional_curvature_df], axis=0)
                curvature_df_sub = curvature_df[['point_index', 'ff_index']].copy()
                curvature_df = curvature_df[~curvature_df_sub.duplicated()]

        self.GUAT_joined_ff_info = pd.concat([self.GUAT_current_ff_info, self.GUAT_alt_ff_info], axis=0)
        if add_arc_info:
            curvature_utils.add_arc_info_to_df(self.GUAT_joined_ff_info, curvature_df, arc_info_to_add=arc_info_to_add)
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))               
        self.free_selection_inputs_df, self.free_selection_inputs_df_for_plotting, self.sequence_of_obs_ff_indices, self.point_index_array, self.pred_var = free_selection.find_free_selection_inputs_from_info_of_n_ff_per_point(self.GUAT_joined_ff_info, self.monkey_information, ff_attributes=ff_attributes, attributes_for_plotting=attributes_for_plotting,
                                                                                    num_ff_per_row=self.num_old_ff_per_row + self.num_new_ff_per_row, add_current_curvature_of_traj=add_current_curvature_of_traj, ff_caught_T_sorted=self.ff_caught_T_sorted, curv_of_traj_df=self.curv_of_traj_df)        
        self.free_selection_time = self.monkey_information.loc[self.point_index_array, 'monkey_t'].values     
        self.num_stops = self.num_stops_df.set_index('point_index').loc[self.point_index_array, 'num_stops'].values
        self.output = (self.num_stops > 2).astype(int)       


    def add_additional_info_to_plot(self, time_range_of_trajectory, 
                                    num_time_points_for_trajectory, 
                                    ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_visible', 'time_till_next_visible']):
        
        if (self.all_available_ff_in_near_future is not None) & ('time_till_next_visible' in ff_attributes):
            self.more_ff_df, self.more_ff_inputs_df_for_plotting = cluster_replacement_utils.find_more_ff_inputs_for_plotting(self.point_index_all, self.sequence_of_obs_ff_indices, self.ff_dataframe, self.ff_real_position_sorted, 
                                                                                                                              self.monkey_information, ff_attributes=ff_attributes, all_available_ff_in_near_future=self.all_available_ff_in_near_future)
            _, self.more_traj_points, self.more_traj_stops, _ = trajectory_info.furnish_machine_learning_data_with_trajectory_data_func(self.X_all, self.time_all, self.monkey_information,
                                                                                                    trajectory_data_kind=self.trajectory_data_kind, time_range_of_trajectory=time_range_of_trajectory, 
                                                                                                    num_time_points_for_trajectory=num_time_points_for_trajectory, add_traj_stops=self.add_traj_stops)
            self.more_ff_inputs = self.more_ff_inputs_df_for_plotting.values
            self.more_ff_inputs_to_plot = self.more_ff_inputs[self.indices_test]
            self.more_traj_points_to_plot = self.more_traj_points[self.indices_test]
            self.more_traj_stops_to_plot = self.more_traj_stops[self.indices_test]
        else:
            super().add_additional_info_to_plot(time_range_of_trajectory, num_time_points_for_trajectory, ff_attributes=ff_attributes)





