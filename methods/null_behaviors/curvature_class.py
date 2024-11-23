
import sys
from decision_making_analysis.decision_making import intended_targets_classes
from null_behaviors import show_null_trajectory, find_best_arc, curvature_utils, curv_of_traj_utils


import math
import numpy as np



class CurvatureOfPath():
    def __init__(self, ff_dataframe, monkey_information, ff_caught_T_sorted, ff_real_position_sorted, seed=None):

        if seed is not None:
            np.random.seed(seed)
        monkey_information['point_index'] = monkey_information.index

        self.ff_dataframe = ff_dataframe[ff_dataframe.ff_angle_boundary.between(-45*math.pi/180, 45*math.pi/180)]
        self.monkey_information = monkey_information
        self.ff_caught_T_sorted = ff_caught_T_sorted
        self.ff_real_position_sorted = ff_real_position_sorted
        self.seed = seed

    def make_curvature_df(self, window_for_curv_of_traj, curv_of_traj_mode='time', truncate_curv_of_traj_by_time_of_capture=False, ff_radius_for_optimal_arc=15, clean=True,
                          include_curv_to_ff_center=True, include_optimal_curvature=True,
                          optimal_arc_stop_at_visible_boundary=False,
                          ignore_error=False):
        ff_dataframe = self.ff_dataframe.copy()
        self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_sorted, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        self.curvature_df = curvature_utils.make_curvature_df(ff_dataframe, self.curv_of_traj_df, ff_radius_for_optimal_arc=ff_radius_for_optimal_arc, clean=clean,
                                                              include_curv_to_ff_center=include_curv_to_ff_center, include_optimal_curvature=include_optimal_curvature,
                                                              optimal_arc_stop_at_visible_boundary=optimal_arc_stop_at_visible_boundary, ignore_error=ignore_error)
        self.curvature_point_index = self.curvature_df.point_index.values


    def find_optimal_arc_info_from_curvature_df(self):
        self.all_point_index = self.curvature_df.point_index.values
        self.all_ff_indices = self.curvature_df.ff_index.values

        self.all_arc_lengths = self.curvature_df.optimal_arc_length.values
        self.all_arc_radius = self.curvature_df.optimal_arc_radius.values
        self.arc_end_direction = np.sign(self.curvature_df.optimal_curvature.values)

        # find the angle from the monkey to the end point of the arc
        self.arc_end_angles = self.curvature_df.optimal_arc_measure.values/2 #if remembered correctly, this is based on a formula
        # for ff to the right
        self.arc_end_angles[self.arc_end_direction < 0] = - self.arc_end_angles[self.arc_end_direction < 0]


    def make_best_arc_df(self):
        self.best_arc_df, self.best_arc_original_columns = find_best_arc.make_best_arc_df(self.curvature_df, self.monkey_information, self.ff_real_position_sorted)


    def add_column_monkey_passed_by_to_best_arc_df(self):
        find_best_arc.add_column_monkey_passed_by_to_best_arc_df(self.best_arc_df, self.ff_dataframe)   


    def get_elements_for_plotting(self, optimal_arc_stop_at_visible_boundary=False, ignore_error=False):
        arc_ff_xy = self.ff_real_position_sorted[self.all_ff_indices]
        monkey_xy = self.monkey_information.loc[self.curvature_df.point_index.values, ['monkey_x', 'monkey_y']].values
        monkey_angles = self.monkey_information.loc[self.curvature_df.point_index.values, 'monkey_angles'].values
        ff_distance = self.curvature_df.ff_distance.values
        ff_angle = self.curvature_df.ff_angle.values
        whether_ff_behind = (np.abs(ff_angle) > math.pi/2)
        self.center_x, self.center_y, self.arc_starting_angle, self.arc_ending_angle = optimal_arc_utils.find_cartesian_arc_center_and_angle_for_optimal_arc(arc_ff_xy, monkey_xy, monkey_angles, ff_distance, ff_angle, self.all_arc_radius, np.sign(self.arc_end_angles), 
                                                                                                                                                           whether_ff_behind=whether_ff_behind, optimal_arc_stop_at_visible_boundary=optimal_arc_stop_at_visible_boundary,
                                                                                                                                                           ignore_error=ignore_error)





