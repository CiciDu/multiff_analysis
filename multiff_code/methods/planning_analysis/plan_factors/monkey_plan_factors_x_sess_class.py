from planning_analysis.factors_vs_indicators import make_variations_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.factors_vs_indicators import _variations_base_class
from data_wrangling import specific_utils, combine_info_utils, base_processing_class
from planning_analysis.factors_vs_indicators import process_variations_utils

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

    default_ref_point_params_based_on_mode = {
        'distance': np.arange(-150, -30, 10)}

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
                                               curv_traj_window_before_stop=[-25, 0]):
        self.pf = plan_factors_class.PlanFactors()
        self.pf.monkey_name = self.monkey_name
        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(
            self.pf, raw_data_folder_path)
        self.pf.retrieve_all_plan_data_for_one_session(
            ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        self.pf.get_plan_features_tc()
        print('Successfully retrieved plan_features data for session: ',
              raw_data_folder_path)

    def get_plan_features_df_across_sessions(self,
                                             exists_ok=True,
                                             plan_features_exists_ok=True,
                                             heading_info_df_exists_ok=False,
                                             stops_near_ff_df_exists_ok=True,
                                             curv_of_traj_mode='distance',
                                             window_for_curv_of_traj=[-25, 0],
                                             use_curv_to_ff_center=False,
                                             ref_point_mode='distance',
                                             ref_point_value=-150,
                                             curv_traj_window_before_stop=[
                                                 -25, 0],
                                             save_data=True,
                                             **plan_features_tc_kwargs
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
        combd_plan_features_tc_path = os.path.join(
            self.combd_plan_features_tc_folder_path, df_name)

        if exists_ok:
            if exists(combd_plan_features_tc_path):
                self.combd_plan_features_tc = pd.read_csv(combd_plan_features_tc_path).drop(
                    ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
                return
            else:
                print(
                    'Retrieving combd_plan_features_tc failed. Will recreate them.')

        self.make_combd_plan_features_tc(plan_features_exists_ok=plan_features_exists_ok,
                                         heading_info_df_exists_ok=heading_info_df_exists_ok,
                                         stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                         **plan_features_tc_kwargs)

        if save_data:
            os.makedirs(self.combd_plan_features_tc_folder_path, exist_ok=True)
            self.combd_plan_features_tc.to_csv(
                combd_plan_features_tc_path, index=False)
            print('Saved combd_plan_features_tc to: ',
                  combd_plan_features_tc_path)

        return

    def make_combd_plan_features_tc(self,
                                    plan_features_exists_ok=True,
                                    heading_info_df_exists_ok=True,
                                    stops_near_ff_df_exists_ok=True,
                                    ):

        self.combd_plan_features_tc = pd.DataFrame()

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
                    if plan_features_exists_ok:
                        self.retrieve_all_plan_data_for_one_session(raw_data_folder_path=raw_data_folder_path, ref_point_mode=self.ref_point_mode,
                                                                    ref_point_value=self.ref_point_value, curv_traj_window_before_stop=self.curv_traj_window_before_stop)
                    else:
                        raise Exception(
                            'plan_features_exists_ok is False')
                except Exception as e:
                    print(e)
                    print('Will recreate the plan_features data for this session')
                    self.pf = plan_factors_class.PlanFactors(raw_data_folder_path=raw_data_folder_path,
                                                             opt_arc_type=self.opt_arc_type,
                                                             curv_of_traj_mode=self.curv_of_traj_mode, window_for_curv_of_traj=self.window_for_curv_of_traj)
                    gc.collect()
                    self.pf.make_plan_features_df_both_test_and_ctrl(plan_features_exists_ok=plan_features_exists_ok,                                                                      ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value,
                                                                     curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                                                     use_curv_to_ff_center=self.use_curv_to_ff_center,
                                                                     heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                     stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok)

                self._add_new_plan_data_to_combd_data(row['data_name'])

        self.combd_plan_features_tc.reset_index(drop=True, inplace=True)

    def initialize_monkey_sessions_df(self):
        self.sessions_df = specific_utils.initialize_monkey_sessions_df(
            raw_data_dir_name=self.raw_data_dir_name)
        return self.sessions_df

    def initialize_monkey_sessions_df_for_one_monkey(self):
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
            self.raw_data_dir_name, self.monkey_name)

    def _add_new_plan_data_to_combd_data(self, data_name):
        self.pf.plan_features_tc['data_name'] = data_name
        self.combd_plan_features_tc = pd.concat(
            [self.combd_plan_features_tc, self.pf.plan_features_tc], axis=0)
        self.sessions_df_for_one_monkey.loc[self.sessions_df_for_one_monkey['data_name']
                                            == data_name, 'finished'] = True

    def combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types(self):
        self.all_ref_pooled_median_info = make_variations_utils.combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types()
        self.process_all_ref_pooled_median_info_to_plot_heading_and_curv()
        return self.all_ref_pooled_median_info

    def combine_all_ref_per_sess_median_info_across_monkeys_and_opt_arc_types(self):
        self.all_ref_per_sess_median_info = make_variations_utils.combine_all_ref_per_sess_median_info_across_monkeys_and_opt_arc_types()
        self.process_all_ref_per_sess_median_info_to_plot_heading_and_curv()
        return self.all_ref_per_sess_median_info

    def combine_pooled_perc_info_across_monkeys(self):
        self.pooled_perc_info = make_variations_utils.combine_pooled_perc_info_across_monkeys()
        return self.pooled_perc_info

    def combine_per_sess_perc_info_across_monkeys(self):
        self.per_sess_perc_info = make_variations_utils.combine_per_sess_perc_info_across_monkeys()
        return self.per_sess_perc_info