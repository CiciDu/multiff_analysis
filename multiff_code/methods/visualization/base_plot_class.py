from null_behaviors import show_null_trajectory
from planning_analysis.show_planning.cur_vs_nxt_ff import plot_monkey_heading_helper_class
from data_wrangling import further_processing_class


class BasePlotter():

    null_arcs_plotting_kwargs = {'player': 'monkey',
                                 'show_stops': True,
                                 'show_believed_target_positions': True,
                                 'show_reward_boundary': True,
                                 'show_scale_bar': True,
                                 'hitting_arena_edge_ok': True,
                                 'trial_too_short_ok': True,
                                 'show_connect_path_ff': False,
                                 'vary_color_for_connecting_path_ff': True,
                                 'show_points_when_ff_stop_being_visible': False,
                                 'show_alive_fireflies': False,
                                 'show_visible_fireflies': True,
                                 'show_in_memory_fireflies': True,
                                 'connect_path_ff_max_distance': 500,
                                 'adjust_xy_limits': True,
                                 'show_null_agent_trajectory': False,
                                 'show_only_ff_that_monkey_has_passed_by_closely': False,
                                 'show_null_trajectory_reaching_boundary_ok': False,
                                 'zoom_in': False,
                                 'truncate_part_before_crossing_arena_edge': True}


    def add_additional_info_for_plotting(self, **kwargs):
        # put each element of kargs into self
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_null_arc_info_for_counted_points(self, use_fixed_arc_length=False, fixed_arc_length=None, use_curv_to_ff_center=False):

        if use_curv_to_ff_center:
            self.cur_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(
                self.ref_point_index_counted, self.cur_ff_counted_df.ff_index.values, self.monkey_information, self.ff_real_position_sorted)
            self.nxt_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(
                self.ref_point_index_counted, self.nxt_ff_counted_df.ff_index.values, self.monkey_information, self.ff_real_position_sorted)
        else:
            self.cur_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
                self.cur_ff_counted_df, self.monkey_information)
            self.nxt_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_opt_arc_info_for_plotting(
                self.nxt_ff_counted_df, self.monkey_information)

        if use_fixed_arc_length:
            self._update_null_arc_info_using_fixed_arc_length(fixed_arc_length)

    def _update_null_arc_info_using_fixed_arc_length(self, fixed_arc_length):
        if fixed_arc_length is not None:
            self.cur_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(
                fixed_arc_length, self.cur_null_arc_info_for_counted_points)
            self.nxt_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(
                fixed_arc_length, self.nxt_null_arc_info_for_counted_points)
        else:
            temp_curv_of_traj_df = self.curv_of_traj_df.set_index(
                ['point_index']).loc[self.cur_null_arc_info_for_counted_points['arc_point_index'].values]
            fixed_arc_length = temp_curv_of_traj_df['delta_distance'].values
            self.cur_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(
                fixed_arc_length, self.cur_null_arc_info_for_counted_points)
            self.nxt_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(
                fixed_arc_length, self.nxt_null_arc_info_for_counted_points)

    def _find_null_arcs_for_cur_and_nxt_ff_for_the_point_from_info_for_counted_points(self, i):
        self.cur_null_arc_info_for_the_point = self.cur_null_arc_info_for_counted_points.iloc[[
            i]].copy()
        self.nxt_null_arc_info_for_the_point = self.nxt_null_arc_info_for_counted_points.iloc[[
            i]].copy()
        if len(self.cur_null_arc_info_for_the_point) > 1:
            raise ValueError('More than one cur null arc found for the point')
        if len(self.nxt_null_arc_info_for_the_point) > 1:
            raise ValueError('More than one nxt null arc found for the point')
