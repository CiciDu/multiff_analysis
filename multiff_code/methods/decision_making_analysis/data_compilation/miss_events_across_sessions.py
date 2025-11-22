from decision_making_analysis.data_compilation import miss_events_class
from decision_making_analysis.data_enrichment import trajectory_utils
from decision_making_analysis.ff_data_acquisition import cluster_replacement_utils, free_selection, get_missed_ff_data
from null_behaviors import curvature_utils
from data_wrangling import combine_info_utils, specific_utils
from decision_making_analysis.modeling import ml_for_decision_making_class
from decision_making_analysis.data_enrichment import miss_events_enricher
from decision_making_analysis.data_enrichment import rsw_vs_rcap_utils

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

    # ===============================================================
    # Process and align current vs alternative firefly info
    # ===============================================================
    def process_current_and_alternative_ff_info(self):
        self.miss_event_cur_ff = self.miss_event_cur_ff_reloaded.copy()
        self.miss_event_alt_ff = self.miss_event_alt_ff_reloaded.copy()

        for df in [self.miss_event_cur_ff, self.miss_event_alt_ff]:
            df[['whether_changed', 'whether_intended_target']] = False

        self.num_old_ff_per_row = self.gc_kwargs['num_old_ff_per_row']
        self.num_new_ff_per_row = self.gc_kwargs['num_new_ff_per_row']

        self.miss_event_cur_ff, self.miss_event_alt_ff = (
            cluster_replacement_utils.further_process_df_related_to_cluster_replacement(
                self.miss_event_cur_ff,
                self.miss_event_alt_ff,
                num_old_ff_per_row=self.num_old_ff_per_row,
                num_new_ff_per_row=self.num_new_ff_per_row,
                selection_criterion_if_too_many_ff=self.gc_kwargs['selection_criterion_if_too_many_ff']
            )
        )

        self.miss_event_alt_ff['order'] += self.num_old_ff_per_row

        # Fill curvature placeholders
        self.miss_event_cur_ff = curvature_utils.fill_up_NAs_for_placeholders_in_columns_related_to_curvature(
            self.miss_event_cur_ff, self.relevant_curv_of_traj_df, curv_of_traj_df=self.relevant_curv_of_traj_df
        )
        self.miss_event_alt_ff = curvature_utils.fill_up_NAs_for_placeholders_in_columns_related_to_curvature(
            self.miss_event_alt_ff, self.relevant_curv_of_traj_df, curv_of_traj_df=self.relevant_curv_of_traj_df
        )

        # also fill in eventual_outcome and event_type
        self.miss_event_cur_ff = get_missed_ff_data.fill_in_eventual_outcome_and_event_type(
            self.miss_event_cur_ff)
        self.miss_event_alt_ff = get_missed_ff_data.fill_in_eventual_outcome_and_event_type(
            self.miss_event_alt_ff)

    # ===============================================================
    # Construct ML-ready input/output
    # ===============================================================

    def prepare_data_to_predict_num_stops(
        self,
        add_arc_info=False,
        add_current_curv_of_traj=False,
        ff_attributes=['ff_distance', 'ff_angle',
                       'time_since_last_vis'],
        add_num_ff_in_cluster=False,
        arc_info_to_add=['curv_diff', 'abs_curv_diff']
    ):

        # Merge current & next firefly info
        self.miss_event_cur_ff['group'] = 'Original'
        self.miss_event_alt_ff['group'] = 'Alternative'
        self.miss_to_switch_joined_ff_info = pd.concat(
            [self.miss_event_cur_ff[self.miss_event_cur_ff['eventual_outcome'] == 'switch'],
                self.miss_event_alt_ff[self.miss_event_alt_ff['eventual_outcome'] == 'switch']], axis=0
        )
        
        self._prepare_data_using_current_and_alternative_ff_info(
            add_arc_info=add_arc_info,
            add_current_curv_of_traj=add_current_curv_of_traj,
            ff_attributes=ff_attributes,
            add_num_ff_in_cluster=add_num_ff_in_cluster,
            arc_info_to_add=arc_info_to_add
        )

        self.num_stops = self.num_stops_df.set_index('point_index').loc[
            self.point_index_array, 'num_stops'
        ].values
        self.y_value = self.num_stops

        self.y_var_df = pd.DataFrame(self.num_stops.astype(int), columns=['num_stops'])
        self.y_var_df.loc[self.y_var_df['num_stops'] > 2, 'num_stops'] = 2
        self.y_var_df['num_stops'] = self.y_var_df['num_stops'] - 1  # to avoid problem during classification
        
        
    def prepare_data_to_predict_rsw_vs_rcap(self,
                                            add_arc_info=False,
                                            add_current_curv_of_traj=False,
                                            ff_attributes=['ff_distance', 'ff_angle',
                                                        'time_since_last_vis'],
                                            add_num_ff_in_cluster=False,
                                            arc_info_to_add=['curv_diff', 'abs_curv_diff']):
        # Merge current & next firefly info
        self.miss_event_cur_ff['group'] = 'Original'
        self.miss_event_alt_ff['group'] = 'Alternative'
        self.miss_to_switch_joined_ff_info = pd.concat(
            [self.miss_event_cur_ff[self.miss_event_cur_ff['event_type'].isin(['rcap', 'rsw'])],
                self.miss_event_alt_ff[self.miss_event_alt_ff['event_type'].isin(['rcap', 'rsw'])]], axis=0
        )
        
        self._prepare_data_using_current_and_alternative_ff_info(
            add_arc_info=add_arc_info,
            add_current_curv_of_traj=add_current_curv_of_traj,
            ff_attributes=ff_attributes,
            add_num_ff_in_cluster=add_num_ff_in_cluster,
            arc_info_to_add=arc_info_to_add
        )

        self.y_var_df = rsw_vs_rcap_utils.extract_binary_event_label(
                                    df=self.miss_event_cur_ff_reloaded,
                                    point_index_array=self.point_index_array
                                )
        self.y_value = self.y_var_df['whether_rcap'].values


    def _prepare_data_using_current_and_alternative_ff_info(self,
                                                            add_arc_info=False,
                                                            add_current_curv_of_traj=False,
                                                            ff_attributes=['ff_distance', 'ff_angle',
                                                                        'time_since_last_vis'],
                                                            add_num_ff_in_cluster=False,
                                                            arc_info_to_add=['curv_diff', 'abs_curv_diff']):

        if add_current_curv_of_traj:
            existing_curv_feats = [
                n for n in self.all_traj_feature_names['traj_points'] if 'curv_of_traj' in n]
            if existing_curv_feats:
                add_current_curv_of_traj = False
                print(
                    "Warning: 'curv_of_traj' already present in trajectory features. Ignoring add_current_curv_of_traj.")


        self.ff_attributes = ff_attributes.copy()
        self.attributes_for_plotting = [a for a in [
            'ff_distance', 'ff_angle', 'time_since_last_vis'] if a in self.ff_attributes]

        if add_arc_info:
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))

        # Build ML design matrix
        (self.free_selection_x_df, self.free_selection_x_df_for_plotting,
         self.sequence_of_obs_ff_indices, self.point_index_array, self.pred_var) = free_selection.find_free_selection_x_from_info_of_n_ff_per_point(
            self.miss_to_switch_joined_ff_info,
            ff_attributes=ff_attributes,
            attributes_for_plotting=self.attributes_for_plotting,
            num_ff_per_row=self.num_old_ff_per_row + self.num_new_ff_per_row
        )

        if add_current_curv_of_traj:
            curv_of_traj = self.relevant_curv_of_traj_df.set_index('point_index').loc[
                self.point_index_array, 'curv_of_traj'
            ].values
            self.free_selection_x_df['curv_of_traj'] = curv_of_traj

        if add_num_ff_in_cluster:
            self.free_selection_x_df['num_current_ff_in_cluster'] = (
                self.miss_event_cur_ff.groupby('point_index').first()[
                    'num_ff_in_cluster'].values
            )
            self.free_selection_x_df['num_nxt_ff_in_cluster'] = (
                self.miss_event_alt_ff.groupby('point_index').first()[
                    'num_ff_in_cluster'].values
            )

        # Extract labels and times
        self.free_selection_time = self.miss_to_switch_joined_ff_info.set_index('point_index').loc[
            self.point_index_array, 'time'
        ].values

    # ===============================================================
    # Prepare for ML pipeline
    # ===============================================================
    def prepare_data_for_machine_learning(self, furnish_with_trajectory_data=False):
        """
        Prepares all rsw-derived inputs and labels for ML models.
        Optionally augments with trajectory points and stops.
        """
        self.free_selection_point_index = self.point_index_array
        self.free_selection_labels = self.y_value.copy()

        self.time_range_of_trajectory = self.gc_kwargs['time_range_of_trajectory']
        self.num_time_points_for_trajectory = self.gc_kwargs['num_time_points_for_trajectory']

        super().prepare_data_for_machine_learning(
            kind='free selection',
            furnish_with_trajectory_data=furnish_with_trajectory_data,
            trajectory_data_kind='position',
            add_traj_stops=True
        )

        if furnish_with_trajectory_data:
            self.furnish_with_trajectory_data = True
            self.X_all = np.concatenate(
                [self.X_all, self.traj_points_df.values, self.traj_stops_df.values], axis=1)
            self.input_features = np.concatenate(
                [self.input_features, self.traj_points_df.columns,
                    self.traj_stops_df.columns], axis=0
            )
            self.traj_points = self.traj_points_df.values
            self.traj_stops = self.traj_stops_df.values
        else:
            self.furnish_with_trajectory_data = False
