from decision_making_analysis.data_compilation import miss_events_class
from decision_making_analysis.data_enrichment import trajectory_utils
from decision_making_analysis.ff_data_acquisition import cluster_replacement_utils, free_selection, get_missed_ff_data
from null_behaviors import curvature_utils
from data_wrangling import combine_info_utils, specific_utils
from decision_making_analysis.modeling import ml_for_decision_making_class
from decision_making_analysis.data_enrichment import miss_events_enricher, rsw_vs_rcap_utils


import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import copy


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class MissEventsAcrossSessions(ml_for_decision_making_class.MLForDecisionMakingClass):

    df_names = ['miss_events_df', 'miss_event_alt_ff', 'miss_event_cur_ff', 'traj_data_df']
    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    def __init__(self, gc_kwargs=None, monkey_name='monkey_Bruno'):
        self.gc_kwargs = copy.deepcopy(miss_events_enricher.gc_kwargs)
        if gc_kwargs is not None:
            self.gc_kwargs.update(gc_kwargs)
        self.monkey_information = None
        self.polar_plots_kwargs = {}
        self.monkey_name = monkey_name
        self.combd_miss_events_info_folder_path = f"all_monkey_data/decision_making/{self.monkey_name}/combined_data/GUAT_info"

    def streamline_getting_combd_rsw_or_rcap_x_df(self,
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
            self.combd_rcap_x_df = pd.DataFrame()
            self.combd_rsw_x_df = pd.DataFrame()
            sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
                self.raw_data_dir_name, monkey_name)

            for index, row in sessions_df_for_one_monkey.iterrows():
                print(row['data_name'])
                raw_data_folder_path = os.path.join(
                    self.raw_data_dir_name, row['monkey_name'], row['data_name'])
                self.cgt = miss_events_class.MissEventsClass(
                    ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, raw_data_folder_path=raw_data_folder_path)
                self.cgt.streamline_getting_rsw_or_rcap_x_df(
                    rsw_or_rcap='rcap', exists_ok=individual_df_exists_ok)
                self.cgt.streamline_getting_rsw_or_rcap_x_df(
                    rsw_or_rcap='rsw', exists_ok=individual_df_exists_ok)
                self.combd_rcap_x_df = pd.concat(
                    [self.combd_rcap_x_df, self.cgt.rcap_x_df], axis=0)
                self.combd_rsw_x_df = pd.concat(
                    [self.combd_rsw_x_df, self.cgt.rsw_x_df], axis=0)

        self.combd_rsw_x_df.reset_index(drop=True, inplace=True)
        self.combd_rcap_x_df.reset_index(drop=True, inplace=True)

        return self.combd_rsw_x_df, self.combd_rcap_x_df

    def streamline_getting_combd_decision_making_basic_ff_info(self,
                                                               exists_ok=True,
                                                               monkey_name='monkey_Bruno',
                                                               ):

        self.monkey_name = monkey_name
        df_path = f'all_monkey_data/decision_making/{monkey_name}/combd_decision_making_basic_ff_info.csv'

        if exists_ok and os.path.exists(df_path):
            self.combd_decision_making_basic_ff_info = pd.read_csv(df_path)
        else:
            self.combd_decision_making_basic_ff_info = pd.DataFrame()
            sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
                self.raw_data_dir_name, monkey_name)

            for index, row in sessions_df_for_one_monkey.iterrows():
                print(row['data_name'])
                raw_data_folder_path = os.path.join(
                    self.raw_data_dir_name, row['monkey_name'], row['data_name'])
                self.cgt = miss_events_class.MissEventsClass(
                    raw_data_folder_path=raw_data_folder_path)
                self.cgt.make_decision_making_basic_ff_info()
                self.cgt.decision_making_basic_ff_info['data_name'] = row['data_name']

                self.combd_decision_making_basic_ff_info = pd.concat(
                    [self.combd_decision_making_basic_ff_info, self.cgt.decision_making_basic_ff_info], axis=0)

            self.combd_decision_making_basic_ff_info.reset_index(
                drop=True, inplace=True)
            self.combd_decision_making_basic_ff_info.to_csv(
                df_path, index=False)

    def retrieve_or_make_combined_info(
        self,
        combined_info_exists_ok=True,
        curv_of_traj_data_df_exists_ok=True,
    ):
        # Try to load existing combined info, or mark for regeneration
        if combined_info_exists_ok:
            self.combined_info, self.collect_info_flag = combine_info_utils.try_to_retrieve_combined_info(
                self.combd_miss_events_info_folder_path, df_names=self.df_names
            )
        else:
            self.collect_info_flag = True

        self.trajectory_features = self.gc_kwargs['trajectory_features']

        if self.collect_info_flag:
            self.combined_info, self.all_traj_feature_names = self.collect_combined_info_for_miss_events(
                self.gc_kwargs,
                curv_of_traj_data_df_exists_ok=curv_of_traj_data_df_exists_ok
            )
        else:
            self.all_traj_feature_names = self.find_all_traj_feature_names(
                self.gc_kwargs, traj_point_features=self.gc_kwargs['trajectory_features']
            )

    # ===============================================================
    # Collect rsw info across sessions
    # ===============================================================
    def collect_combined_info_for_miss_events(
        self,
        gc_kwargs,
        curv_of_traj_data_df_exists_ok=True,
        traj_data_df_exists_ok=True
    ):

        # Get all sessions and check which DFs exist
        sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
            self.raw_data_dir_name, self.monkey_name
        )
        sessions_df_for_one_monkey = combine_info_utils.check_which_df_exists_for_each_session(
            sessions_df_for_one_monkey, df_names=self.df_names
        )

        # Regenerate absent miss events info if needed
        print(f'Making absent miss events info for monkey {self.monkey_name}')
        sessions_df_for_one_monkey = self.make_absent_df(
            sessions_df_for_one_monkey,
            gc_kwargs,
            curv_of_traj_data_df_exists_ok=curv_of_traj_data_df_exists_ok,
            traj_df_exist_ok=traj_data_df_exists_ok
        )

        # Collect info across sessions
        print(
            f'Collecting important info from all sessions of monkey {self.monkey_name}')
        self.all_important_info, self.all_point_index_to_new_number = combine_info_utils.collect_info_from_all_sessions(
            sessions_df_for_one_monkey, df_names=self.df_names
        )
        self.all_traj_feature_names = self.find_all_traj_feature_names(
            gc_kwargs, traj_point_features=gc_kwargs['trajectory_features']
        )

        # Combine into a single dataset
        self.combined_info = combine_info_utils.turn_all_important_info_into_combined_info(
            self.all_important_info,
            self.combd_miss_events_info_folder_path,
            save_each_df_as_csv=True
        )

        return self.combined_info, self.all_traj_feature_names

    # ===============================================================
    # Helper: Generate all trajectory feature names
    # ===============================================================
    def find_all_traj_feature_names(self, gc_kwargs, traj_point_features=['monkey_distance', 'monkey_angle']):
        return trajectory_utils.make_all_traj_feature_names(
            time_range_of_trajectory=gc_kwargs['time_range_of_trajectory'],
            num_time_points_for_trajectory=gc_kwargs['num_time_points_for_trajectory'],
            time_range_of_trajectory_to_plot=gc_kwargs['time_range_of_trajectory_to_plot'],
            num_time_points_for_trajectory_to_plot=gc_kwargs['num_time_points_for_trajectory_to_plot'],
            traj_point_features=traj_point_features
        )

    # ===============================================================
    # Helper: Create missing miss events info for incomplete sessions
    # ===============================================================
    def make_absent_df(
        self,
        sessions_df_for_one_monkey,
        gc_kwargs,
        curv_of_traj_data_df_exists_ok=True,
        traj_df_exist_ok=True
    ):
        sessions_df_for_one_monkey['finished'] = sessions_df_for_one_monkey[self.df_names].all(
            axis=1)
        sessions_df_for_one_monkey['remade'] = False

        for index, row in sessions_df_for_one_monkey.iterrows():
            if row['finished']:
                continue

            raw_data_folder_path = os.path.join(
                self.raw_data_dir_name, row['monkey_name'], row['data_name'])
            print('raw_data_folder_path:', raw_data_folder_path)

            gcc = miss_events_class.MissEventsClass(
                raw_data_folder_path, gc_kwargs)
            important_info = gcc.streamline_process_to_collect_info_from_one_session(
                curv_of_traj_data_df_exists_ok=curv_of_traj_data_df_exists_ok,
                update_point_index=False
            )

            for df in self.df_names:
                important_info[df].to_csv(os.path.join(
                    gcc.miss_events_folder_path, df + '.csv'))

            sessions_df_for_one_monkey.loc[index, [
                'finished', 'remade']] = True, True

        # Remake trajectory DF if needed
        if not traj_df_exist_ok:
            for index, row in sessions_df_for_one_monkey.iterrows():
                if row['remade']:
                    continue

                raw_data_folder_path = os.path.join(
                    self.raw_data_dir_name, row['monkey_name'], row['data_name'])
                gcc = miss_events_class.MissEventsClass(
                    raw_data_folder_path, gc_kwargs)
                print('Remaking traj_data_df...')

                miss_event_alt_ff = pd.read_csv(os.path.join(
                    gcc.miss_events_folder_path, 'miss_event_alt_ff.csv'))
                all_point_index = np.unique(
                    miss_event_alt_ff['point_index'].values)

                gcc.streamline_process_to_collect_traj_data_only(
                    all_point_index, curv_of_traj_data_df_exists_ok=curv_of_traj_data_df_exists_ok
                )
                gcc.traj_data_df.to_csv(os.path.join(
                    gcc.miss_events_folder_path, 'traj_data_df.csv'))

        return sessions_df_for_one_monkey

    # ===============================================================
    # Reload and unpack combined info
    # ===============================================================
    def unpack_combined_info(self, combined_info=None, all_traj_feature_names=None, new_point_index_col='new_point_index'):
        if combined_info is not None:
            self.combined_info = copy.deepcopy(combined_info)
            
        # for all df in self.combined_info, change the point_index to the new_point_index
        for df_name in self.combined_info.keys():
            self.combined_info[df_name]['point_index'] = self.combined_info[df_name][new_point_index_col]
        print('Note: point_index column has been updated to new_point_index column in all dfs in combined_info.')
            
        if all_traj_feature_names is not None:
            self.all_traj_feature_names = copy.deepcopy(all_traj_feature_names)

        self.miss_event_alt_ff_reloaded = self.combined_info['miss_event_alt_ff']
        self.miss_event_cur_ff_reloaded = self.combined_info['miss_event_cur_ff']
        self.traj_data_df = self.combined_info['traj_data_df']

        # Extract trajectory components
        self.traj_points_df = self.traj_data_df[self.all_traj_feature_names['traj_points']]
        self.traj_stops_df = self.traj_data_df[self.all_traj_feature_names['traj_stops']]
        self.relevant_curv_of_traj_df = self.traj_data_df[self.all_traj_feature_names['relevant_curv_of_traj']].copy(
        )
        self.relevant_curv_of_traj_df['point_index'] = self.traj_data_df['point_index']

        self.num_stops_df = self.miss_event_cur_ff_reloaded[[
            'point_index', 'num_stops']].drop_duplicates()


    def process_current_and_alternative_ff_info(self):
        self.miss_event_cur_ff = self.miss_event_cur_ff_reloaded.copy()
        self.miss_event_alt_ff = self.miss_event_alt_ff_reloaded.copy()
        super().process_current_and_alternative_ff_info()