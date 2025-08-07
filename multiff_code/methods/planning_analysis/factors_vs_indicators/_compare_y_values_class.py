
from planning_analysis.plan_factors import test_vs_control_utils, test_vs_control_utils
from planning_analysis.factors_vs_indicators import make_variations_utils, process_variations_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp
from machine_learning.ml_methods import ml_methods_class, prep_ml_data_utils, ml_methods_class
import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class _CompareYValues:

    def __init__(self):
        pass

    def make_or_retrieve_overall_median_info(self,
                                             exists_ok=True,
                                             all_median_info_exists_ok=True,
                                             ref_point_params_based_on_mode=None,
                                             list_of_curv_traj_window_before_stop=[
                                                 [-50, 0]],
                                             save_data=True,
                                             combd_heading_df_x_sessions_exists_ok=True,
                                             stops_near_ff_df_exists_ok=True,
                                             heading_info_df_exists_ok=True,
                                             process_info_for_plotting=True):

        if exists_ok & exists(self.overall_median_info_path):
            self.overall_median_info = pd.read_csv(
                self.overall_median_info_path).drop(columns=['Unnamed: 0'])
            print('Successfully retrieved overall_median_info from ',
                  self.overall_median_info_path)
        else:
            if ref_point_params_based_on_mode is None:
                ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode

            self.overall_median_info = pd.DataFrame([])
            for curv_traj_window_before_stop in list_of_curv_traj_window_before_stop:
                temp_overall_median_info = make_variations_utils.make_variations_df_across_ref_point_values(self.make_all_median_info,
                                                                                                            ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                                                            monkey_name=self.monkey_name,
                                                                                                            variation_func_kwargs={'all_median_info_exists_ok': all_median_info_exists_ok,
                                                                                                                                   'curv_traj_window_before_stop': curv_traj_window_before_stop,
                                                                                                                                   'save_data': save_data,
                                                                                                                                   'combd_heading_df_x_sessions_exists_ok': combd_heading_df_x_sessions_exists_ok,
                                                                                                                                   'stops_near_ff_df_exists_ok': stops_near_ff_df_exists_ok,
                                                                                                                                   'heading_info_df_exists_ok': heading_info_df_exists_ok,
                                                                                                                                   },
                                                                                                            path_to_save=None,
                                                                                                            )
                temp_overall_median_info['curv_traj_window_before_stop'] = str(
                    curv_traj_window_before_stop)
                self.overall_median_info = pd.concat(
                    [self.overall_median_info, temp_overall_median_info], axis=0)

        self.overall_median_info.reset_index(drop=True, inplace=True)
        self.overall_median_info['monkey_name'] = self.monkey_name
        self.overall_median_info['opt_arc_type'] = self.opt_arc_type
        self.overall_median_info.to_csv(self.overall_median_info_path)
        print(
            f'Saved overall_median_info_path to {self.overall_median_info_path}')
        if process_info_for_plotting:
            self.process_overall_median_info_to_plot_heading_and_curv()

        return self.overall_median_info

    def make_or_retrieve_all_perc_info(self, exists_ok=True, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                       ref_point_mode='distance', ref_point_value=-50, verbose=False, save_data=True, process_info_for_plotting=True):
        # These two parameters (ref_point_mode, ref_point_value) are actually not important here as long as the corresponding data can be successfully retrieved,
        # since the results are the same regardless

        if exists_ok & exists(self.all_perc_info_path):
            self.all_perc_info = pd.read_csv(self.all_perc_info_path).drop(
                ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                   heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok)
            self.all_perc_info = make_variations_utils.make_all_perc_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
                                                                                                             self.ctrl_heading_info_df, verbose=verbose)

            if save_data:
                self.all_perc_info.to_csv(self.all_perc_info_path)
            print('Stored new all_perc_info in ', self.all_perc_info_path)

        self.all_perc_info['monkey_name'] = self.monkey_name
        self.all_perc_info['opt_arc_type'] = self.opt_arc_type

        if process_info_for_plotting:
            self.process_all_perc_info_to_plot_direction()

        return self.all_perc_info

    def make_all_median_info(self, ref_point_mode='time after cur ff visible',
                             ref_point_value=0.1,
                             curv_traj_window_before_stop=[-50, 0],
                             all_median_info_exists_ok=True,
                             combd_heading_df_x_sessions_exists_ok=True,
                             stops_near_ff_df_exists_ok=True,
                             heading_info_df_exists_ok=True,
                             verbose=False, save_data=True):

        df_name = find_cvn_utils.find_diff_in_curv_df_name(
            ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        df_path = os.path.join(self.all_median_info_folder_path, df_name)
        if all_median_info_exists_ok & exists(df_path):
            self.all_median_info = pd.read_csv(df_path).drop(
                ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
            print('Successfully retrieved all_median_info from ', df_path)
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                   curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                   heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                   stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data,
                                                                   combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok)
            self.all_median_info = make_variations_utils.make_all_median_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
                                                                                                                 self.ctrl_heading_info_df, verbose=verbose)
            self.all_median_info['ref_point_mode'] = ref_point_mode
            self.all_median_info['ref_point_value'] = ref_point_value
            time_calibration = {'ref_point_mode': ref_point_mode,
                                'ref_point_value': ref_point_value, 'monkey_name': self.monkey_name}
            self.all_median_info.attrs.update(time_calibration)
            os.makedirs(self.all_median_info_folder_path, exist_ok=True)
            self.all_median_info.to_csv(df_path)
            print('Stored new all_median_info in ',
                  self.all_median_info_folder_path)
        return self.all_median_info

    def process_overall_median_info_to_plot_heading_and_curv(self):
        self.all_median_info_heading = process_variations_utils.make_new_df_for_plotly_comparison(
            self.overall_median_info)
        self.all_median_info_curv = self.all_median_info_heading.copy()
        self.all_median_info_curv['sample_size'] = self.all_median_info_curv['sample_size_for_curv']

    def process_all_perc_info_to_plot_direction(self):
        self.all_perc_info_new = process_variations_utils.make_new_df_for_plotly_comparison(self.all_perc_info,
                                                                                            match_rows_based_on_ref_columns_only=False)
