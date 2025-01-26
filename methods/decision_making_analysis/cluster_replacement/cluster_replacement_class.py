import sys
from decision_making_analysis.cluster_replacement import cluster_replacement_utils, plot_cluster_replacement
from decision_making_analysis.decision_making import decision_making_class, plot_decision_making
from decision_making_analysis import free_selection
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



class ClusterReplacement(decision_making_class.DecisionMaking):

    def __init__(self, ff_dataframe, ff_caught_T_new, ff_real_position_sorted, monkey_information, ff_life_sorted,
                 time_range_of_trajectory=[-1,1], num_time_points_for_trajectory=10):
        super().__init__(ff_dataframe, ff_caught_T_new, ff_real_position_sorted, monkey_information, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)
        self.ff_life_sorted = ff_life_sorted
        self.converting_multi_class_for_free_selection = False
        self.data_kind = 'free_selection'
        

    def find_input_and_output_for_cluster_replacement(self, num_old_ff_per_row=2, num_new_ff_per_row=2, selection_criterion_if_too_many_ff='time_since_last_vis', sorting_criterion=None,\
                                                      add_arc_info=False, add_current_curv_of_traj=False, curvature_df=None, curv_of_traj_df=None, ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis'],
                                                      window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False,
                                                      arc_info_to_add=['optimal_curvature', 'curv_diff']):
        if curv_of_traj_df is None:
            self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            self.curv_of_traj_df = curv_of_traj_df
        
        
        self.num_old_ff_per_row = num_old_ff_per_row
        self.num_new_ff_per_row = num_new_ff_per_row
        self.old_ff_cluster_df, self.new_ff_cluster_df, self.parallel_old_ff_cluster_df, self.non_chosen_ff_cluster_df = \
            cluster_replacement_utils.find_df_related_to_cluster_replacement(self.replacement_df, self.prior_to_replacement_df, self.non_chosen_df, self.manual_anno, \
                                                                                self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted, self.ff_life_sorted, sample_size=None, equal_sample_from_two_cases=True)
        
        self.joined_old_ff_cluster_df = pd.concat([self.old_ff_cluster_df, self.parallel_old_ff_cluster_df], axis=0)
        self.joined_new_ff_cluster_df = pd.concat([self.new_ff_cluster_df, self.non_chosen_ff_cluster_df], axis=0)

        self.joined_old_ff_cluster_df, self.joined_new_ff_cluster_df = cluster_replacement_utils.further_process_df_related_to_cluster_replacement(self.joined_old_ff_cluster_df, self.joined_new_ff_cluster_df, num_old_ff_per_row=num_old_ff_per_row, num_new_ff_per_row=num_new_ff_per_row,\
                selection_criterion_if_too_many_ff=selection_criterion_if_too_many_ff, sorting_criterion=sorting_criterion)
        self.joined_new_ff_cluster_df['order'] = self.joined_new_ff_cluster_df['order'] + num_old_ff_per_row
        self.joined_cluster_df = pd.concat([self.joined_old_ff_cluster_df, self.joined_new_ff_cluster_df], axis=0)
        if add_arc_info:
            curvature_utils.add_arc_info_to_df(self.joined_cluster_df, curvature_df, arc_info_to_add=arc_info_to_add)
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))      
        self.free_selection_inputs_df, self.free_selection_inputs_df_for_plotting, self.sequence_of_obs_ff_indices, self.point_index_array, self.pred_var = free_selection.find_free_selection_inputs_from_info_of_n_ff_per_point(self.joined_cluster_df, self.monkey_information, ff_attributes=ff_attributes, 
                                                                                    num_ff_per_row=num_old_ff_per_row + num_new_ff_per_row, add_current_curv_of_traj=add_current_curv_of_traj, ff_caught_T_new=self.ff_caught_T_new, curv_of_traj_df=self.curv_of_traj_df)        
        self.free_selection_time = self.monkey_information.loc[self.point_index_array, 'time'].values
        # incorporate whether_changed
        self.whether_changed = self.joined_cluster_df[['point_index', 'whether_changed']].drop_duplicates()        
        self.output = self.whether_changed['whether_changed'].values.astype(int).copy()        
        
        

    def prepare_data_for_machine_learning(self, furnish_with_trajectory_data=True, trajectory_data_kind="position", add_traj_stops=True): 
        # trajectory_data_kind can also be "velocity"
        self.free_selection_inputs = self.free_selection_inputs_df.values
        self.free_selection_point_index = self.point_index_array
        self.free_selection_labels = self.output.copy()
        super().prepare_data_for_machine_learning(kind="free selection", furnish_with_trajectory_data=furnish_with_trajectory_data, trajectory_data_kind=trajectory_data_kind, add_traj_stops=add_traj_stops)


    def plot_prediction_results(self, selected_cases=None, max_plot_to_make=40, show_direction_of_monkey_on_trajectory=False, show_reward_boundary=False, 
                                use_more_ff_inputs=False, use_more_traj_points=False, max_time_since_last_vis=2.5, predict_num_stops=False, additional_plotting_kwargs={}):

        #if self.polar_plots_kwargs is None:
        self.prepare_to_plot_prediction_results(use_more_ff_inputs=use_more_ff_inputs, use_more_traj_points=use_more_traj_points, show_direction_of_monkey_on_trajectory=show_direction_of_monkey_on_trajectory)
        for key, value in additional_plotting_kwargs.items():
            self.polar_plots_kwargs[key] = value

        self.polar_plots_kwargs['y_prob'] = None
        
        self.make_polar_plots_for_cluster_replacement(selected_cases=selected_cases,
                                                        max_plot_to_make=max_plot_to_make,
                                                        show_reward_boundary=show_reward_boundary,
                                                        max_time_since_last_vis=max_time_since_last_vis,
                                                        predict_num_stops=predict_num_stops,
                                                        )




    def make_polar_plots_for_cluster_replacement(self, selected_cases=None, max_plot_to_make=5, show_reward_boundary=False,
                                                max_time_since_last_vis=2.5, ff_colormap = 'Greens', 
                                                predict_num_stops=False, ):
        
        if selected_cases is not None:
            instance_to_plot = selected_cases[:max_plot_to_make]
        else:
            instance_to_plot = np.arange(self.polar_plots_kwargs['ff_inputs'].shape[0])[:max_plot_to_make]


        for i in instance_to_plot:
            self.current_polar_plot_kargs = plot_decision_making.get_current_polar_plot_kargs(i, max_time_since_last_vis=max_time_since_last_vis, 
                                                                                              show_reward_boundary=show_reward_boundary, ff_colormap=ff_colormap, **self.polar_plots_kwargs)

            ax = plot_cluster_replacement.make_one_polar_plot_for_cluster_replacement(self.num_old_ff_per_row, predict_num_stops=predict_num_stops, **self.current_polar_plot_kargs)

            plt.show()