from decision_making_analysis.GUAT import GUAT_collect_info_class
from decision_making_analysis.compare_GUAT_and_TAFT import GUAT_vs_TAFT_utils, helper_GUAT_vs_TAFT_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from data_wrangling import base_processing_class

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


class CompareGUATandTAFTclass(helper_GUAT_vs_TAFT_class.HelperGUATandTAFTclass):
    
    def __init__(self, 
                 raw_data_folder_path='all_monkey_data/raw_monkey_data/monkey_Bruno/data_0330',
                 ref_point_mode='distance', 
                 ref_point_value=-150,
                 stop_period_duration=2):
        
        base_processing_class.BaseProcessing.get_related_folder_names_from_raw_data_folder_path(self, raw_data_folder_path)
        self.curv_of_traj_df = None
        self.curv_of_traj_df_w_one_sided_window = None

        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.stop_period_duration = stop_period_duration

    def streamline_getting_GUAT_or_TAFT_x_df(self, GUAT_or_TAFT='TAFT',
                                    save_data=True,
                                    exists_ok=True,
                                    ):
        self.GUAT_or_TAFT = GUAT_or_TAFT
        # delete self.stops_near_ff_df
        if hasattr(self, 'stops_near_ff_df'):
            del self.stops_near_ff_df
        
        self.GUAT_vs_TAFT_folder_path = os.path.join(self.decision_making_folder_path, 'GUAT_vs_TAFT')
        os.makedirs(self.GUAT_vs_TAFT_folder_path, exist_ok=True)

        if exists_ok:
            try:
                self.try_retrieving_GUAT_or_TAFT_x_df()
                return
            except FileNotFoundError:
                pass

        self.get_relevant_monkey_data(raw_data_folder_path=self.raw_data_folder_path)
        self.get_GUAT_or_TAFT_df()
        self.get_GUAT_or_TAFT_x_df(save_data=save_data)


    def try_retrieving_GUAT_or_TAFT_x_df(self):
        if self.GUAT_or_TAFT == 'TAFT':
            if (os.path.exists(os.path.join(self.GUAT_vs_TAFT_folder_path, 'TAFT_x_df.csv'))):
                self.TAFT_x_df = pd.read_csv(os.path.join(self.GUAT_vs_TAFT_folder_path, 'TAFT_x_df.csv'))
                return
            else:
                raise FileNotFoundError('TAFT_x_df.csv does not exist')
        elif self.GUAT_or_TAFT == 'GUAT':
            if (os.path.exists(os.path.join(self.GUAT_vs_TAFT_folder_path, 'GUAT_x_df.csv'))):
                self.GUAT_x_df = pd.read_csv(os.path.join(self.GUAT_vs_TAFT_folder_path, 'GUAT_x_df.csv'))
                return
            else:
                raise FileNotFoundError('GUAT_x_df.csv does not exist')
        else:
            raise ValueError('GUAT_or_TAFT must be either TAFT or GUAT')
            

    def get_relevant_monkey_data(self,
                                 already_retrieved_ok=True,
                                raw_data_folder_path='all_monkey_data/raw_monkey_data/monkey_Bruno/data_0330'
                                ):
        self.monkey_name = os.path.basename(raw_data_folder_path)
        if not hasattr(self, 'gcc'):
            self.gcc = GUAT_collect_info_class.GUATCollectInfoForSession(raw_data_folder_path=raw_data_folder_path, 
                                                                         gc_kwargs=helper_GUAT_vs_TAFT_class.gc_kwargs, new_point_index_start=0)

        include_TAFT_data = True if self.GUAT_or_TAFT == 'TAFT' else False
        include_GUAT_data = True if self.GUAT_or_TAFT == 'GUAT' else False
        self.gcc.get_monkey_data(already_retrieved_ok=already_retrieved_ok, include_GUAT_data=include_GUAT_data, include_TAFT_data=include_TAFT_data)
        
        for df in ['monkey_information', 'ff_dataframe', 'ff_flash_sorted', 'ff_real_position_sorted', 'ff_life_sorted',
                   'ff_believed_position_sorted', 'ff_caught_T_new', 'closest_stop_to_capture_df']:
            setattr(self, df, getattr(self.gcc.data_item, df).copy())
        if self.GUAT_or_TAFT == 'TAFT':
            self.TAFT_trials_df = self.gcc.data_item.TAFT_trials_df.copy()
            self.TAFT_trials_df['first_stop_time'] = self.monkey_information.loc[self.TAFT_trials_df['first_stop_point_index'], 'time'].values
            self.TAFT_trials_df['ff_index'] = self.TAFT_trials_df['trial']
            # because we need to have alt_ff, we will limit the max number of ff_index to len(self.ff_caught_T_new - 2)
            self.TAFT_trials_df = self.TAFT_trials_df[self.TAFT_trials_df['ff_index'] < len(self.ff_caught_T_new) - 2]
            GUAT_vs_TAFT_utils.add_stop_point_index(self.TAFT_trials_df, self.monkey_information, self.ff_real_position_sorted)
        elif self.GUAT_or_TAFT == 'GUAT':
            self.GUAT_w_ff_df = self.gcc.data_item.GUAT_w_ff_df.copy()
        self.ff_dataframe_visible = self.ff_dataframe[self.ff_dataframe['visible'] == 1]


    def get_GUAT_or_TAFT_df(self):
        if self.GUAT_or_TAFT == 'TAFT':
            self._get_TAFT_df()
            self._get_TAFT_df2_based_on_ref_point()
        elif self.GUAT_or_TAFT == 'GUAT':
            self._get_GUAT_df()
            self._get_GUAT_df2_based_on_ref_point()
        else:
            raise ValueError('GUAT_or_TAFT must be either TAFT or GUAT')


    def get_GUAT_or_TAFT_x_df(self, save_data=True):
        self._get_stops_near_ff_df(already_made_ok=True)
        self._make_plan_y_df()
        self._make_plan_x_df(list_of_stop_ff_cluster_radius=[100],
                            list_of_alt_ff_cluster_radius=[200])
        
        self._get_x_features_df(list_of_stop_ff_cluster_radius=[],
                                list_of_stop_ff_ang_cluster_radius=[20],
                                list_of_start_dist_cluster_radius=[100],
                                list_of_start_ang_cluster_radius=[20],
                                list_of_flash_cluster_period=[[1.0, 1.5], [1.5, 2.0]])
        self._make_only_stop_ff_df()
        self._get_GUAT_or_TAFT_x_df(save_data=save_data)


    def get_x_and_y_var_df(self):
        self.x_var_df = pd.concat([self.TAFT_x_df, self.GUAT_x_df], axis=0).reset_index(drop=True).drop(columns=['stop_point_index'])
        self.x_var_df['dir_from_stop_ff_same_side'] = self.x_var_df['dir_from_stop_ff_same_side'].astype(int)
        self.y_var_df = pd.DataFrame(np.array([1]*len(self.TAFT_x_df) + [0]*len(self.GUAT_x_df)).reshape(-1, 1), columns=['y_var'])

        columns_with_na = self.x_var_df.columns[self.x_var_df.isna().any()].tolist()
        print(f'There are {len(columns_with_na)} columns with NA that are dropped. {self.x_var_df.shape[1]} columns are left. The dropped columns with number of NA are:')
        print(self.x_var_df[columns_with_na].isna().sum())
        self.x_var_df = self.x_var_df.drop(columns=columns_with_na)


    def take_out_subsets_to_plot(self, list_of_stop_point_index, GUAT_or_TAFT):
        if GUAT_or_TAFT == 'GUAT':
            sub = self.GUAT_w_ff_df[self.GUAT_w_ff_df['stop_point_index'].isin(list_of_stop_point_index)].copy()
        else:
            sub = self.TAFT_trials_df[self.TAFT_trials_df['stop_point_index'].isin(list_of_stop_point_index)].copy()

        GUAT_or_TAFT_x_df = getattr(self, f"{GUAT_or_TAFT}_x_df")
        sub = sub.merge(GUAT_or_TAFT_x_df[['stop_point_index', 'stop_ff_distance_at_ref']], on='stop_point_index', how='left')

        sub2 = find_stops_near_ff_utils.find_ff_info_based_on_ref_point(sub, self.monkey_information, self.ff_real_position_sorted,
                                                                        ref_point_mode=self.ref_point_mode, ref_point_value=self.ref_point_value)
        
        sub2.rename(columns={'point_index': 'ref_point_index'}, inplace=True)
        sub = sub.merge(sub2[['stop_point_index', 'ref_point_index']], on='stop_point_index', how='left')

        # also prepare some plotting args and kwargs
        self.plotting_kwargs = {'show_stops': True,
                        'show_believed_target_positions': True,
                        'show_reward_boundary': True,
                        'show_scale_bar': True,
                        'truncate_part_before_crossing_arena_edge': True,
                        'trial_too_short_ok': True,
                        'show_connect_path_ff': False,
                        'show_visible_fireflies': True}

        self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, None, self.ff_caught_T_new)
        self.max_distance_to_stop_for_GUAT_target = 50
        
        return sub
