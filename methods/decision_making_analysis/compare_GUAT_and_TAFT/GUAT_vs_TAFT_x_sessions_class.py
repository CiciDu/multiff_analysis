from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils, GUAT_vs_TAFT_class
from planning_analysis.plan_factors import plan_factors_class
from planning_analysis.only_cur_ff import only_cur_ff_utils, features_to_keep_utils
from planning_analysis.show_planning import nxt_ff_utils, show_planning_utils
from planning_analysis.show_planning.get_cur_vs_nxt_ff_data import find_cvn_utils, cur_vs_nxt_ff_from_ref_class
from planning_analysis.plan_factors import plan_factors_utils, build_factor_comp
from null_behaviors import curv_of_traj_utils
from data_wrangling import combine_info_utils, specific_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class GUATandTAFTacrossSessionsClass():

    def __init__(self):
        self.raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    def streamline_getting_combd_GUAT_or_TAFT_x_df(self,
                                                   individual_df_exists_ok=True,
                                                   monkey_name='monkey_Bruno',
                                                   ref_point_params_based_on_mode={'time': [-1.5, -1],
                                                                                   'distance': [-150, -100]}
                                                   ):

        variations_list = specific_utils.init_variations_list_func(
            ref_point_params_based_on_mode)

        for index, row in variations_list.iterrows():
            ref_point_mode = row['ref_point_mode']
            ref_point_value = row['ref_point_value']
            self.combd_TAFT_x_df = pd.DataFrame()
            self.combd_GUAT_x_df = pd.DataFrame()
            sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
                self.raw_data_dir_name, monkey_name)

            for index, row in sessions_df_for_one_monkey.iterrows():
                print(row['data_name'])
                raw_data_folder_path = os.path.join(
                    self.raw_data_dir_name, row['monkey_name'], row['data_name'])
                cgt = GUAT_vs_TAFT_class.GUATvsTAFTclass(
                    ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, raw_data_folder_path=raw_data_folder_path)
                cgt.streamline_getting_GUAT_or_TAFT_x_df(
                    GUAT_or_TAFT='TAFT', exists_ok=individual_df_exists_ok)
                cgt.streamline_getting_GUAT_or_TAFT_x_df(
                    GUAT_or_TAFT='GUAT', exists_ok=individual_df_exists_ok)
                self.combd_TAFT_x_df = pd.concat(
                    [self.combd_TAFT_x_df, cgt.TAFT_x_df], axis=0)
                self.combd_GUAT_x_df = pd.concat(
                    [self.combd_GUAT_x_df, cgt.GUAT_x_df], axis=0)

        self.combd_GUAT_x_df.reset_index(drop=True, inplace=True)
        self.combd_TAFT_x_df.reset_index(drop=True, inplace=True)

        return self.combd_GUAT_x_df, self.combd_TAFT_x_df
