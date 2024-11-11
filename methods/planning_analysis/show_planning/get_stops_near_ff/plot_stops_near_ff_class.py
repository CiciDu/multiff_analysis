import sysfrom null_behaviors import show_null_trajectory
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_class, find_stops_near_ff_utils, plot_stops_near_ff_utils, plot_monkey_heading_helper_class
from planning_analysis.plan_factors import plan_factors_utils
from visualization.plotly_tools import plotly_for_monkey, plotly_preparation, plotly_for_null_arcs
from visualization import plot_behaviors_utils
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class _PlotStopsNearFF(find_stops_near_ff_class._FindStopsNearFF):

    traj_curv_descr = 'Traj Curv: From Current Point to Right Before Stop'
    
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

    default_overall_params = {'heading_instead_of_curv': True}

    default_monkey_plot_params = {'eliminate_irrelevant_points_beyond_boundaries': True,
                                    'show_reward_boundary': True,
                                    'rotation_matrix': None,
                                    'hoverdata_multi_columns': ['rel_time'],
                                    'show_null_arcs_to_ff': True,
                                    }

    def __init__(self, 
                 optimal_arc_type='norm_opt_arc', # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
                 ):
        super().__init__()
        self._update_optimal_arc_type_and_related_paths(optimal_arc_type=optimal_arc_type)

    def prepare_to_make_plots(self, use_fixed_arc_length=False, fixed_arc_length=None):
        #self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(self.traj_curv_counted, self.alt_curv_counted)
        self.get_null_arc_info_for_counted_points(fixed_arc_length=fixed_arc_length, use_fixed_arc_length=use_fixed_arc_length)
        plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.find_all_mheading_for_counted_points(self)
        if self.ff_dataframe is None: 
            self.get_more_monkey_data()
        self.stop_point_index_counted = self.stops_near_ff_df_counted['stop_point_index'].values
        self.heading_info_df_counted = self.heading_info_df.set_index('stop_point_index').loc[self.stop_point_index_counted].reset_index()
        self.heading_info_df_counted = plan_factors_utils.process_heading_info_df(self.heading_info_df_counted)
        self.make_PlotTrials_args()


    def make_individual_plots_in_matplotlib(self, current_i, max_num_plot_to_make = 5, additional_plotting_kwargs={'show_connect_path_ff_specific_indices': None, 'show_ff_indices': True},
                              show_position_in_scatter_plot=True, show_monkey_heading=True, show_null_arcs=True):
        
        for i in range(len(self.stops_near_ff_df_counted))[current_i:current_i+max_num_plot_to_make]:
            stops_near_ff_row = self.stops_near_ff_df_counted.iloc[i]
            heading_row = self.heading_info_df_counted.iloc[i]
            diff_in_abs = heading_row['diff_in_abs']
            print(f'diff_in_abs: {diff_in_abs}')

            print('alt_ff_index:', stops_near_ff_row.alt_ff_index)
            print('stop_ff_index:', stops_near_ff_row.stop_ff_index)
            
            current_i = i+1

            fig, R, x0, y0 = plot_stops_near_ff_utils.plot_stops_near_ff_func(stops_near_ff_row, self.monkey_information, self.ff_real_position_sorted, self.ff_dataframe, self.null_arcs_plotting_kwargs, self.PlotTrials_args, 
                                                                ff_max_distance_to_path_to_show_visible_segments=None,
                                                                additional_plotting_kwargs=additional_plotting_kwargs
            )

            axes = fig.axes[0]
            if show_monkey_heading:
                plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.show_all_monkey_heading_in_matplotlib(self, axes, i, R, x0, y0)

            if show_null_arcs:
                current_arc_point_index = self.stop_null_arc_info_for_counted_points.arc_point_index.iloc[i]
                self._find_null_arcs_for_stop_and_alt_ff_for_the_point_from_info_for_counted_points(i=i)
                axes = plot_stops_near_ff_utils.show_null_arcs_func(axes, current_arc_point_index, self.monkey_information, R, x0=x0, y0=y0,
                        stop_null_arc_info_for_the_point=self.stop_null_arc_info_for_the_point, 
                        alt_null_arc_info_for_the_point=self.alt_null_arc_info_for_the_point,
                        )

            if show_position_in_scatter_plot:
                axes = find_stops_near_ff_utils.plot_relationship(self.alt_curv_counted, self.traj_curv_counted, show_plot=False, change_units_to_degrees_per_m=self.overall_params['change_units_to_degrees_per_m'])
                axes.scatter(self.traj_curv_counted[i], self.alt_curv_counted[i], color='red')
            plt.show()
            
        return current_i



    def make_individual_plotly_plots(self, current_i, max_num_plot_to_make=5, 
                                     **additional_plotting_kwargs):


        self.monkey_plot_params.update(additional_plotting_kwargs)
        
        if self.PlotTrials_args is None:
            self.PlotTrials_args = self.make_PlotTrials_args()

        for i in range(len(self.stops_near_ff_df_counted))[current_i: current_i+max_num_plot_to_make]:
            self.stops_near_ff_row = self.stops_near_ff_df_counted.iloc[i]
            heading_row = self.heading_info_df_counted.iloc[i]
            diff_in_abs = heading_row['diff_in_abs']
            print(f'diff_in_abs: {diff_in_abs}')

            if self.monkey_plot_params['show_null_arcs_to_ff']:
                self._find_null_arcs_for_stop_and_alt_ff_for_the_point_from_info_for_counted_points(i=i)

            current_i = i+1
            self.current_plotly_plot_key_comp, self.fig = self.plot_stops_near_ff_in_plotly_func(
                                                                self.monkey_plot_params,
                                                                plot_counter_i=i)
            
            self.fig.show()
            
        return current_i
    



    def plot_stops_near_ff_in_plotly_func(self,
                                        monkey_plot_params={},
                                        plot_counter_i=None,
                                        ):

        default_params = {
            "show_reward_boundary": True,
            "show_alive_fireflies": False,
            "show_visible_fireflies": False,  # only meaningful when show_alive_fireflies is False
            "show_in_memory_fireflies": False, # only meaningful when show_alive_fireflies is False
            "show_visible_segments": True,
            "show_stops": True,
            "show_all_eye_positions": False,
            "show_current_eye_positions": True,
            "show_eye_positions_for_both_eyes": False,
            "show_connect_path_ff": False,
            "show_points_on_trajectory": True,
            "connect_path_ff_max_distance": 500,
            "eliminate_irrelevant_points_beyond_boundaries": True,
            # ahh the below is not used yet
            "show_monkey_heading": False,
            "show_traj_portion": False,
            "show_null_arcs_to_ff": False,
        }

        default_params.update(self.monkey_plot_params)
        default_params.update(monkey_plot_params)
        self.monkey_plot_params = default_params
        
        self.current_plotly_plot_key_comp = plotly_preparation.prepare_to_plot_a_planning_instance_in_plotly(self.stops_near_ff_row, self.PlotTrials_args, 
                                                                                                            self.monkey_plot_params)
        self.point_index_to_show_traj_curv = self.stops_near_ff_row.stop_point_index

        if self.monkey_plot_params['show_stops']:
            #show_stop_point_indices = [int(row.stop_point_index), int(row.next_stop_point_index)]
            trajectory_df = self.current_plotly_plot_key_comp['trajectory_df']
            show_stop_point_indices = trajectory_df[trajectory_df['monkey_speeddummy']==0]['point_index'].values
        else:
            show_stop_point_indices = None
        self.monkey_plot_params['show_stop_point_indices'] = show_stop_point_indices


        self.fig = plotly_for_monkey.make_one_monkey_plotly_plot(self.current_plotly_plot_key_comp, 
                                                            monkey_plot_params=self.monkey_plot_params,                                                            
                                                            )

        if self.monkey_plot_params['show_monkey_heading']:
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.get_all_triangle_df_for_the_point_from_triangle_df_for_all_counted_points(self, plot_counter_i, self.current_plotly_plot_key_comp['R'])
            self.fig = plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.show_all_monkey_heading_in_plotly(self)

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.fig = self._show_null_arcs_for_stop_and_alt_ff_in_plotly()

        self.fig.update_layout(
            autosize=False,
            width=900,  # Set the desired width
            height=700,  # Set the desired height
            margin={'l': 10, 'b': 0, 't': 20, 'r': 10},
        )

        return self.current_plotly_plot_key_comp, self.fig
    

    def _produce_fig_for_dash(self, mark_reference_point=True):

        self._prepare_to_plot_eye_positions_for_dash()

        if self.monkey_plot_params['show_stops']:
            self.show_stop_point_indices = [int(self.stops_near_ff_row.stop_point_index), int(self.stops_near_ff_row.next_stop_point_index)]
        else:
            self.show_stop_point_indices = None

        if self.monkey_plot_params['show_traj_portion']:
            traj_portion = self.traj_portion
        else:
            traj_portion = None

        self.monkey_plot_params=self.monkey_plot_params.update({'show_reward_boundary': True, 
                                                           'show_points_on_trajectory': True,
                                                           'eye_positions_trace_name': 'all_eye_positions',
                                                           'show_stop_point_indices': self.show_stop_point_indices,
                                                           'hoverdata_multi_columns': self.hoverdata_multi_columns,
                                                           'traj_portion': traj_portion,
                                                           'monkey_plot_params': self.monkey_plot_params,
                                                           })

        self.fig = plotly_for_monkey.make_one_monkey_plotly_plot(self.current_plotly_plot_key_comp, 
                                                                monkey_plot_params=self.monkey_plot_params) 
        
        if self.monkey_plot_params['show_monkey_heading']:
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper._get_all_triangle_df_for_the_point_from_triangle_df_in_duration(self)
            self.fig = plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.show_all_monkey_heading_in_plotly(self)

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self._find_null_arcs_for_stop_and_alt_ff_for_the_point_from_info_for_duration()
            self.fig = self._show_null_arcs_for_stop_and_alt_ff_in_plotly()

        if mark_reference_point:
            self.fig = plotly_for_monkey.mark_reference_point_in_monkey_plot(self.fig, self.trajectory_ref_row)
        

        for i, trace in enumerate(self.fig.data):
            if trace.name == 'trajectory_data':
                self.trajectory_data_trace_index = i
                break

        self.traj_portion_trace_index = -1
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'to_show_the_scope_for_curv':
                self.traj_portion_trace_index = i
                break

        return self.fig


    def compare_test_and_control_in_polar_plots(self, max_instances_each=10, test_color='green', ctrl_color='purple',
                                                start='stop_point_index', end='next_stop_point_index', rmax=400):
        
        self._prepare_data_to_compare_test_and_control()

        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        axes = plot_behaviors_utils.set_polar_background_for_plotting(axes, rmax, color_visible_area_in_background=False)
        axes = find_stops_near_ff_utils.add_instances_to_polar_plot(axes, self.stops_near_ff_df_ctrl, self.alt_ff_df2_ctrl, self.monkey_information, 
                                                               max_instances_each, color=ctrl_color, start=start, end=end)
        axes = find_stops_near_ff_utils.add_instances_to_polar_plot(axes, self.stops_near_ff_df_test, self.alt_ff_df2_test, self.monkey_information, 
                                                               max_instances_each, color=test_color, start=start, end=end)
        # make a legend

        colors = [test_color, ctrl_color]
        labels = ['test', 'control']

        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='solid') for c in colors]
        axes.legend(lines, labels, loc='lower right')

        plt.show()


    def compare_test_and_control_in_plotly_polar_plots(self, max_instances_each=10, test_color='green', ctrl_color='purple',
                                                    start='stop_point_index', end='next_stop_point_index'):
        
        if (start=='ref_point_index') & (end=='next_stop_point_index'):
            rmax = 600
        else:
            rmax = 350

        self._prepare_data_to_compare_test_and_control()

        fig = go.Figure()

        # Add control group instances
        fig = find_stops_near_ff_utils.add_instances_to_plotly_polar_plot(fig, self.stops_near_ff_df_ctrl, self.alt_ff_df2_ctrl, self.monkey_information,
                                                                    max_instances_each, color=ctrl_color, point_color='red', start=start, end=end, legendgroup='Control data')

        # Add test group instances
        fig = find_stops_near_ff_utils.add_instances_to_plotly_polar_plot(fig, self.stops_near_ff_df_test, self.alt_ff_df2_test, self.monkey_information,
                                                                    max_instances_each, color=test_color, point_color='blue', start=start, end=end, legendgroup='Test data')

        # Set up radial ticks based on rmax
        radial_ticks = list(range(25, rmax + 1, 25)) if rmax < 150 else []
        # Define custom angular tick labels
        angular_tickvals = np.linspace(0, 360, 8, endpoint=False)  # 12 angular positions
        # Adjusting labels as per the original logic, assuming it's meant to manipulate angular labels
        adjusted_angular_tickvals = np.copy(angular_tickvals)
        adjusted_angular_tickvals[4:8] = -adjusted_angular_tickvals[1:5][::-1]
        adjusted_angular_tickvals = -adjusted_angular_tickvals
        labels_in_degrees = [f"{int(val)}°" for val in adjusted_angular_tickvals]

        fig.update_layout(
            width=800,
            height=800,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, rmax],
                    tickvals=radial_ticks
                ),
                angularaxis=dict(
                    direction="clockwise",
                    tickmode='array',
                    tickvals=angular_tickvals,
                    ticktext=labels_in_degrees
                ),
            )
        )
        fig.show()


    def _prepare_to_make_plotly_fig_for_dash_given_stop_point_index(self, stop_point_index):
        if self.ff_dataframe is None:
            self.get_more_monkey_data()        
        
        self.stop_point_index = stop_point_index
        self.stops_near_ff_row = self.stops_near_ff_df_counted[self.stops_near_ff_df_counted['stop_point_index']==self.stop_point_index].copy()
        self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_sorted)
        
        self.current_plotly_plot_key_comp = plotly_preparation.prepare_to_plot_a_planning_instance_in_plotly(self.stops_near_ff_row, self.PlotTrials_args, self.monkey_plot_params)
        # self.trajectory_ref_row = plotly_for_monkey.find_trajectory_ref_row(self.current_plotly_plot_key_comp['trajectory_df'], self.ref_point_params['ref_point_mode'], self.ref_point_params['ref_point_value'])
        
        self.trajectory_ref_row = self._find_trajectory_ref_row()
        self.trajectory_next_stop_row = self._find_trajectory_next_stop_row()
        self.monkey_hoverdata_value = self.trajectory_ref_row[self.hoverdata_column]
        self.point_index_to_show_traj_curv = self.trajectory_ref_row['point_index'].astype(int)
              
        self._further_prepare_plotting_info_for_the_duration()

    

    def _further_prepare_plotting_info_for_the_duration(self):
        self.curv_of_traj_df_in_duration = self._get_curv_of_traj_df_in_duration()
        try:
            self.curv_of_traj_current_row = self.curv_of_traj_df_in_duration[self.curv_of_traj_df_in_duration['point_index']==self.point_index_to_show_traj_curv].copy()
            self.traj_portion, self.traj_length = plotly_preparation.find_traj_portion_for_traj_curv(self.current_plotly_plot_key_comp['trajectory_df'], self.curv_of_traj_current_row)
        except: 
            self.traj_portion = None
            self.traj_length = 0

        if self.monkey_plot_params['show_null_arcs_to_ff']:
            self.find_null_arcs_info_for_plotting_for_the_duration() 

        if self.monkey_plot_params['show_monkey_heading']:
            plot_monkey_heading_helper_class.PlotMonkeyHeadingHelper.find_all_mheading_and_triangle_df_for_the_duration(self)


    def find_null_arcs_info_for_plotting_for_the_duration(self):

        duration = self.current_plotly_plot_key_comp['duration_to_plot']
        self.stop_ff_index = self.stops_near_ff_row.stop_ff_index
        self.alt_ff_index = self.stops_near_ff_row.alt_ff_index 

        # for the stop ff, we eliminate the point index after the capture

        if self.overall_params['use_curvature_to_ff_center']:
            all_point_index = self.curv_of_traj_df_in_duration['point_index'].values
            self.stop_null_arc_info_for_duration = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(all_point_index, np.repeat(np.array([self.stop_ff_index]), len(all_point_index)), self.monkey_information, self.ff_real_position_sorted, verbose=False)
            self.alt_null_arc_info_for_duration = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(all_point_index, np.repeat(np.array([self.alt_ff_index]), len(all_point_index)), self.monkey_information, self.ff_real_position_sorted, verbose=False)
        else:
            optimal_arc_stop_at_visible_boundary = True if (self.optimal_arc_type == 'opt_arc_stop_first_vis_bdry') else False
            self.stop_ff_best_arc_df = plotly_for_null_arcs.find_best_arc_df_for_ff_in_duration([self.stop_ff_index], duration, self.curv_of_traj_df, self.monkey_information, self.ff_real_position_sorted,
                                                                                                optimal_arc_stop_at_visible_boundary=optimal_arc_stop_at_visible_boundary)
            self.stop_null_arc_info_for_duration = show_null_trajectory.find_and_package_optimal_arc_info_for_plotting(self.stop_ff_best_arc_df, self.monkey_information)
            self.alt_ff_best_arc_df = plotly_for_null_arcs.find_best_arc_df_for_ff_in_duration([self.alt_ff_index], duration, self.curv_of_traj_df, self.monkey_information, self.ff_real_position_sorted,
                                                                                               optimal_arc_stop_at_visible_boundary=optimal_arc_stop_at_visible_boundary)
            self.alt_null_arc_info_for_duration = show_null_trajectory.find_and_package_optimal_arc_info_for_plotting(self.alt_ff_best_arc_df, self.monkey_information)

    
    def get_null_arc_info_for_counted_points(self, use_fixed_arc_length=False, fixed_arc_length=None):

        #with basic_func.HiddenPrints():
        if self.overall_params['use_curvature_to_ff_center']:
            self.stop_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(self.ref_point_index_counted, self.stop_ff_counted_df.ff_index.values, self.monkey_information, self.ff_real_position_sorted)
            self.alt_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(self.ref_point_index_counted, self.alt_ff_counted_df.ff_index.values, self.monkey_information, self.ff_real_position_sorted)
        else:
            self.stop_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_optimal_arc_info_for_plotting(self.stop_ff_counted_df, self.monkey_information)
            self.alt_null_arc_info_for_counted_points = show_null_trajectory.find_and_package_optimal_arc_info_for_plotting(self.alt_ff_counted_df, self.monkey_information)
    
        if use_fixed_arc_length:
            if fixed_arc_length is not None:
                self.stop_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(fixed_arc_length, self.stop_null_arc_info_for_counted_points)
                self.alt_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(fixed_arc_length, self.alt_null_arc_info_for_counted_points)
            else:
                temp_curv_of_traj_df = self.curv_of_traj_df.set_index(['point_index']).loc[self.stop_null_arc_info_for_counted_points['arc_point_index'].values]
                fixed_arc_length = temp_curv_of_traj_df['delta_distance'].values
                self.stop_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(fixed_arc_length, self.stop_null_arc_info_for_counted_points)
                self.alt_null_arc_info_for_counted_points = show_null_trajectory.update_null_arc_info_based_on_fixed_arc_length(fixed_arc_length, self.alt_null_arc_info_for_counted_points)



    def _find_null_arcs_for_stop_and_alt_ff_for_the_point_from_info_for_duration(self):
        self.stop_null_arc_info_for_the_point = self.stop_null_arc_info_for_duration[self.stop_null_arc_info_for_duration['arc_point_index']==self.point_index_to_show_traj_curv]
        self.alt_null_arc_info_for_the_point = self.alt_null_arc_info_for_duration[self.alt_null_arc_info_for_duration['arc_point_index']==self.point_index_to_show_traj_curv]


    def _find_null_arcs_for_stop_and_alt_ff_for_the_point_from_info_for_counted_points(self, i):
        self.stop_null_arc_info_for_the_point = self.stop_null_arc_info_for_counted_points.iloc[[i]].copy()
        self.alt_null_arc_info_for_the_point = self.alt_null_arc_info_for_counted_points.iloc[[i]].copy()
        if len(self.stop_null_arc_info_for_the_point) > 1:
            raise ValueError('More than one stop null arc found for the point')
        if len(self.alt_null_arc_info_for_the_point) > 1:
            raise ValueError('More than one alt null arc found for the point')



    def _show_null_arcs_for_stop_and_alt_ff_in_plotly(self):
        rotation_matrix = self.current_plotly_plot_key_comp['R']
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.alt_null_arc_info_for_the_point, rotation_matrix=rotation_matrix,
                                                                color=self.alt_ff_color, trace_name='alt null arc', linewidth=4)
        self.fig = plotly_for_null_arcs.plot_null_arcs_in_plotly(self.fig, self.stop_null_arc_info_for_the_point, rotation_matrix=rotation_matrix, 
                                                                color=self.stop_ff_color, trace_name='stop null arc', linewidth=3)
        return self.fig

