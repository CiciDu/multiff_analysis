from planning_analysis.variations_of_factors_vs_results import make_variations_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.show_planning.get_cur_vs_nxt_ff_data import find_cvn_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.variations_of_factors_vs_results import _variations_base_class
from data_wrangling import specific_utils, combine_info_utils, base_processing_class
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import gc
import warnings

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class PlanAcrossSessions(_variations_base_class._VariationsBase):

    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    default_ref_point_params_based_on_mode = {'time after cur ff visible': [-0.2, -0.1, 0, 0.1, 0.2],
                                              'distance': [-100, -150]}

    # default_ref_point_params_based_on_mode = {'time after cur ff visible': [-0.2]}

    def __init__(self,
                 monkey_name='monkey_Bruno',
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 opt_arc_type='opt_arc_stop_closest',
                 ):

        super().__init__(opt_arc_type=opt_arc_type)
        self.monkey_name = monkey_name
        self.sessions_df = None
        self.sessions_df_for_one_monkey = None
        self.combd_planning_info_folder_path = make_variations_utils.make_combd_planning_info_folder_path(
            self.monkey_name)
        self.combd_cur_and_nxt_folder_path = make_variations_utils.make_combd_cur_and_nxt_folder_path(
            self.monkey_name)
        self.make_key_paths()

    def retrieve_all_plan_data_for_one_session(self, raw_data_folder_path, ref_point_mode='distance', ref_point_value=-150,
                                               curv_traj_window_before_stop=[-50, 0]):
        self.pf = plan_factors_class.PlanFactors()
        self.pf.monkey_name = self.monkey_name
        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(
            self.pf, raw_data_folder_path)
        self.pf.retrieve_all_plan_data_for_one_session(
            ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        self.pf.get_plan_x_and_y_tc()
        print('Successfully retrieved plan_x and plan_y data for session: ',
              raw_data_folder_path)

    def get_plan_x_and_plan_y_across_sessions(self,
                                              exists_ok=True,
                                              plan_x_exists_ok=True,
                                              plan_y_exists_ok=True,
                                              heading_info_df_exists_ok=False,
                                              stops_near_ff_df_exists_ok=True,
                                              curv_of_traj_mode='distance',
                                              window_for_curv_of_traj=[-25, 25],
                                              use_curv_to_ff_center=False,
                                              ref_point_mode='distance',
                                              ref_point_value=-150,
                                              curv_traj_window_before_stop=[
                                                  -50, 0],
                                              save_data=True,
                                              **plan_xy_tc_kwargs
                                              ):

        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_of_traj_mode = curv_of_traj_mode
        self.window_for_curv_of_traj = window_for_curv_of_traj
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        self.use_curv_to_ff_center = use_curv_to_ff_center

        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)
        df_name = find_cvn_utils.find_diff_in_curv_df_name(ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                           curv_traj_window_before_stop=self.curv_traj_window_before_stop)
        # df_name = find_cvn_utils.find_diff_in_curv_df_name(ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        combd_plan_x_tc_path = os.path.join(
            self.combd_plan_x_tc_folder_path, df_name)
        combd_plan_y_tc_path = os.path.join(
            self.combd_plan_y_tc_folder_path, df_name)

        if exists_ok:
            if exists(combd_plan_x_tc_path) & exists(combd_plan_y_tc_path):
                self.combd_plan_x_tc = pd.read_csv(combd_plan_x_tc_path)
                self.combd_plan_y_tc = pd.read_csv(combd_plan_y_tc_path)
                return
            else:
                print(
                    'Retrieving combd_plan_y_tc and combd_plan_x_tc failed. Will recreate them.')

        self.make_combd_plan_xy_tc(plan_x_exists_ok=plan_x_exists_ok,
                                   plan_y_exists_ok=plan_y_exists_ok,
                                   heading_info_df_exists_ok=heading_info_df_exists_ok,
                                   stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                   **plan_xy_tc_kwargs)

        if save_data:
            os.makedirs(self.combd_plan_x_tc_folder_path, exist_ok=True)
            os.makedirs(self.combd_plan_y_tc_folder_path, exist_ok=True)
            self.combd_plan_x_tc.to_csv(combd_plan_x_tc_path, index=False)
            self.combd_plan_y_tc.to_csv(combd_plan_y_tc_path, index=False)
            print('Saved combd_plan_x_tc to: ', combd_plan_x_tc_path)
            print('Saved combd_plan_y_tc to: ', combd_plan_y_tc_path)

        return

    def make_combd_plan_xy_tc(self,
                              plan_x_exists_ok=True,
                              plan_y_exists_ok=True,
                              heading_info_df_exists_ok=True,
                              stops_near_ff_df_exists_ok=True,
                              ):

        self.combd_plan_y_tc = pd.DataFrame()
        self.combd_plan_x_tc = pd.DataFrame()

        self.initialize_monkey_sessions_df_for_one_monkey()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, row in self.sessions_df_for_one_monkey.iterrows():
                if row['finished'] is True:
                    continue
                raw_data_folder_path = os.path.join(
                    self.raw_data_dir_name, row['monkey_name'], row['data_name'])
                print(raw_data_folder_path)
                # first just try retrieving the data directly
                try:
                    if plan_x_exists_ok & plan_y_exists_ok:
                        self.retrieve_all_plan_data_for_one_session(raw_data_folder_path=raw_data_folder_path, ref_point_mode=self.ref_point_mode,
                                                                    ref_point_value=self.ref_point_value, curv_traj_window_before_stop=self.curv_traj_window_before_stop)
                    else:
                        raise Exception(
                            'plan_x_exists_ok is False or plan_y_exists_ok is False')
                except Exception as e:
                    print(e)
                    print('Will recreate the plan_x and plan_y data for this session')
                    self.pf = plan_factors_class.PlanFactors(raw_data_folder_path=raw_data_folder_path,
                                                             opt_arc_type=self.opt_arc_type,
                                                             curv_of_traj_mode=self.curv_of_traj_mode, window_for_curv_of_traj=self.window_for_curv_of_traj)
                    gc.collect()
                    self.pf.make_plan_x_and_y_for_both_test_and_ctrl(plan_x_exists_ok=plan_x_exists_ok, plan_y_exists_ok=plan_y_exists_ok,
                                                                     ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                                     curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                                                     use_curv_to_ff_center=self.use_curv_to_ff_center,
                                                                     heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                     stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok)

                self._add_new_plan_data_to_combd_data(row['data_name'])

        self.combd_plan_y_tc.reset_index(drop=True, inplace=True)
        self.combd_plan_x_tc.reset_index(drop=True, inplace=True)

    def initialize_monkey_sessions_df(self):
        self.sessions_df = specific_utils.initialize_monkey_sessions_df(
            raw_data_dir_name=self.raw_data_dir_name)
        return self.sessions_df

    def initialize_monkey_sessions_df_for_one_monkey(self):
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
            self.raw_data_dir_name, self.monkey_name)

    def get_combd_heading_df_x_sessions_across_sessions(self,
                                                        ref_point_mode='distance', ref_point_value=-150,
                                                        curv_traj_window_before_stop=[
                                                            -50, 0],
                                                        heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                        use_curv_to_ff_center=False,
                                                        exists_ok=True, save_data=True):

        self.sp = show_planning_class.ShowPlanning(monkey_name=self.monkey_name,
                                                   opt_arc_type=self.opt_arc_type,
                                                   )
        self.combd_heading_df_x_sessions_test, self.combd_heading_df_x_sessions_ctrl = self.sp.make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(ref_point_mode, ref_point_value,
                                                                                                                                                                       curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                                                                                       combd_heading_df_x_sessions_exists_ok=exists_ok, use_curv_to_ff_center=use_curv_to_ff_center,
                                                                                                                                                                       show_printed_output=True, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                                                                                                       stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)

    def _add_new_plan_data_to_combd_data(self, data_name):
        self.pf.plan_y_tc['data_name'] = data_name
        self.pf.plan_x_tc['data_name'] = data_name
        self.combd_plan_y_tc = pd.concat(
            [self.combd_plan_y_tc, self.pf.plan_y_tc], axis=0)
        self.combd_plan_x_tc = pd.concat(
            [self.combd_plan_x_tc, self.pf.plan_x_tc], axis=0)
        self.sessions_df_for_one_monkey.loc[self.sessions_df_for_one_monkey['data_name']
                                            == data_name, 'finished'] = True

    def combine_overall_median_info_across_monkeys_and_opt_arc_types(self):
        self.overall_median_info = make_variations_utils.combine_overall_median_info_across_monkeys_and_opt_arc_types()
        self.process_overall_median_info_to_plot_heading_and_curv()
        return self.overall_median_info

    def combine_all_perc_info_across_monkeys(self):
        self.all_perc_info = make_variations_utils.combine_all_perc_info_across_monkeys()
        self.process_all_perc_info_to_plot_direction()
        return self.all_perc_info
