from visualization.dash_tools.dash_main_class_methods import dash_main_class
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_class
from planning_analysis.show_planning import show_planning_utils
from planning_analysis.show_planning import nxt_ff_utils, show_planning_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import copy

# Import neural data visualization tools

# Import shared configuration
from visualization.dash_tools.dash_config import configure_plotting_environment
configure_plotting_environment()

# Configuration - moved to a shared config module or base class
plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

# https://dash.plotly.com/interactive-graphing


'''
# Things to keep in mind:
probably will need to make heading_info_df and stuff afresh...while either not saving them or only in GUAT_vs_TAFT
Rather than test or control, we'll want GUAT or TAFT

'''

class DashForGUATandTAFT(GUAT_vs_TAFT_class.GUATvsTAFTclass, dash_main_class.DashMainPlots):

    def __init__(self, raw_data_folder_path=None, 
                 GUAT_or_TAFT='GUAT',
                 stop_period_duration=2,
                 opt_arc_type='opt_arc_stop_closest',
                 ):
        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         ref_point_mode=None,
                         ref_point_value=None,
                         stop_period_duration=stop_period_duration,
                         )
        
        self.GUAT_or_TAFT = GUAT_or_TAFT
        self.test_or_ctrl = 'test' # this is more like a placeholder
        self.opt_arc_type = opt_arc_type



    def prepare_to_make_dash_for_main_plots(self,
                                            ref_point_params={},
                                            curv_of_traj_params={},
                                            overall_params={},
                                            monkey_plot_params={},
                                            time_series_plot_params={},
                                            stops_near_ff_df_exists_ok=True,
                                            heading_info_df_exists_ok=True,
                                            test_or_control='test',
                                            stop_point_index=None):

        self.ref_point_params = ref_point_params
        self.curv_of_traj_params = curv_of_traj_params
        self.overall_params = {
            **copy.deepcopy(self.default_overall_params),
            **overall_params
        }
        self.monkey_plot_params = {
            **copy.deepcopy(self.default_monkey_plot_params),
            **monkey_plot_params
        }
        self.time_series_plot_params = time_series_plot_params

        self.ref_point_mode = self.ref_point_params['ref_point_mode']
        self.ref_point_value = self.ref_point_params['ref_point_value']


        self.snf_streamline_organizing_info_kwargs = find_cvn_utils.organize_snf_streamline_organizing_info_kwargs(
            ref_point_params, curv_of_traj_params, overall_params)
        self.streamline_organizing_info_2(**self.snf_streamline_organizing_info_kwargs, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                          heading_info_df_exists_ok=heading_info_df_exists_ok, test_or_control=test_or_control)

        
        self._get_stops_near_ff_row(stop_point_index)

        self._prepare_static_main_plots()
        
      
    def streamline_organizing_info_2(self, **kwargs):
        
        self.ref_point_mode = kwargs['ref_point_mode']
        self.ref_point_value = kwargs['ref_point_value']
        self.remove_i_o_modify_rows_with_big_ff_angles = kwargs['remove_i_o_modify_rows_with_big_ff_angles']
        self.curv_traj_window_before_stop = kwargs.get('curv_traj_window_before_stop', [-25, 0])

        self.get_relevant_monkey_data()
        self.get_GUAT_or_TAFT_df()
        self._get_stops_near_ff_df(already_made_ok=True)
        
        self.stops_near_ff_df, self.nxt_ff_df, self.cur_ff_df = nxt_ff_utils.get_nxt_ff_df_and_cur_ff_df(
            self.stops_near_ff_df)
        self.find_nxt_ff_df_and_cur_ff_df_from_ref(self.ref_point_value, self.ref_point_mode)
        self.add_info_to_nxt_ff_and_cur_ff_df(remove_i_o_modify_rows_with_big_ff_angles=self.remove_i_o_modify_rows_with_big_ff_angles)
        
        self.cur_and_nxt_ff_from_ref_df = show_planning_utils.make_cur_and_nxt_ff_from_ref_df(
           self.nxt_ff_df_final, self.cur_ff_df_final)
        
        self.make_heading_info_df_2()
        self._take_out_info_counted()


    def make_heading_info_df_2(self, merge_diff_in_curv_df_to_heading_info=True):
        self.heading_info_df = show_planning_utils.make_heading_info_df(
            self.cur_and_nxt_ff_from_ref_df, self.stops_near_ff_df_modified, self.monkey_information, self.ff_real_position_sorted)

        if 'nxt_ff_angle_at_ref' not in self.heading_info_df.columns:
            self.add_both_ff_at_ref_to_heading_info_df()

        self.diff_in_curv_df = self.make_diff_in_curv_df(
                    curv_traj_window_before_stop=self.curv_traj_window_before_stop)
        
        if merge_diff_in_curv_df_to_heading_info:
            if hasattr(self, 'heading_info_df'):
                columns_to_add = [
                    col for col in self.diff_in_curv_df.columns if col not in self.heading_info_df.columns]
                self.heading_info_df = self.heading_info_df.merge(
                    self.diff_in_curv_df[['ref_point_index'] + columns_to_add], on='ref_point_index', how='left')

