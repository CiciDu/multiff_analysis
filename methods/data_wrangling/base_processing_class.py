from data_wrangling import specific_utils, process_monkey_information, retrieve_raw_data, time_calib_utils
from pattern_discovery import make_ff_dataframe
from non_behavioral_analysis import eye_positions
from null_behaviors import curv_of_traj_utils
from planning_analysis.test_params_for_planning import params_utils
from planning_analysis.show_planning import nxt_ff_utils
from pattern_discovery import cluster_analysis


import os
import os.path
import pandas as pd
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


class BaseProcessing:

    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    def __init__(self):
        self.monkey_information = None
        self.ff_dataframe = None
        self.curv_of_traj_params = {}

    def load_raw_data(self, raw_data_folder_path, monkey_data_exists_ok=True, window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False):
        self.extract_info_from_raw_data_folder_path(raw_data_folder_path)
        self.retrieve_or_make_monkey_data(exists_ok=monkey_data_exists_ok)
        if curv_of_traj_mode is not None:
            self.curv_of_traj_df = self.get_curv_of_traj_df(window_for_curv_of_traj=window_for_curv_of_traj, curv_of_traj_mode=curv_of_traj_mode,
                                                            truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)

    def extract_info_from_raw_data_folder_path(self, raw_data_folder_path):
        self.get_related_folder_names_from_raw_data_folder_path(
            raw_data_folder_path)
        self.monkey_name = raw_data_folder_path.split('/')[2]
        self.data_name = raw_data_folder_path.split('/')[3]
        self.player = 'monkey'

        if 'monkey_' not in self.monkey_name:
            raise Exception("The monkey name should start with 'monkey_")
        if 'data_' not in self.data_name:
            raise Exception("The data name should start with 'data_'")

    def get_related_folder_names_from_raw_data_folder_path(self, raw_data_folder_path):
        # replace raw_monkey_data with other folder names
        self.raw_data_folder_path = raw_data_folder_path
        self.processed_data_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'processed_data')
        self.planning_data_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'planning')
        self.patterns_and_features_data_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'patterns_and_features')
        self.decision_making_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'decision_making')
        self.neural_data_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'neural_data')
        self.processed_neural_data_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'processed_neural_data')
        self.time_calibration_folder_path = raw_data_folder_path.replace(
            'raw_monkey_data', 'time_calibration')

        # make sure all the folders above exist
        os.makedirs(self.processed_data_folder_path, exist_ok=True)
        os.makedirs(self.planning_data_folder_path, exist_ok=True)
        os.makedirs(self.patterns_and_features_data_folder_path, exist_ok=True)
        os.makedirs(self.decision_making_folder_path, exist_ok=True)
        os.makedirs(self.neural_data_folder_path, exist_ok=True)
        os.makedirs(self.processed_neural_data_folder_path, exist_ok=True)
        os.makedirs(self.time_calibration_folder_path, exist_ok=True)

    def try_retrieving_df(self, df_name, exists_ok=True, data_folder_name_for_retrieval=None):
        if data_folder_name_for_retrieval is None:
            raise Exception("data_folder_name_for_retrieval is None")
        csv_name = df_name + '.csv'
        filepath = os.path.join(data_folder_name_for_retrieval, csv_name)
        if exists(filepath) & exists_ok:
            df_of_interest = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)
            print("Retrieved", df_name)
        else:
            df_of_interest = None
        return df_of_interest

    def make_or_retrieve_ff_dataframe(self, exists_ok=True, already_made_ok=False, save_into_h5=True, to_furnish_ff_dataframe=True, **kwargs):

        if already_made_ok & (getattr(self, 'ff_dataframe', None) is not None):
            return

        h5_file_pathway = os.path.join(os.path.join(
            self.processed_data_folder_path, 'ff_dataframe.h5'))
        try:
            if not exists_ok:
                raise Exception(
                    'ff_dataframe exists_ok is False. Will make new ff_dataframe')

            self.ff_dataframe = pd.read_hdf(h5_file_pathway, 'ff_dataframe')
            print("Retrieved ff_dataframe from", h5_file_pathway)
        # otherwise, recreate the dataframe
        except Exception as e:
            print(
                "Failed to retrieve ff_dataframe. Will make new ff_dataframe. Error: ", e)
            ff_dataframe_args = (self.monkey_information, self.ff_caught_T_new,
                                 self.ff_flash_sorted,  self.ff_real_position_sorted, self.ff_life_sorted)
            ff_dataframe_kargs = {"max_distance": 500,
                                  "to_add_essential_columns": False,
                                  "to_furnish_ff_dataframe": False}

            for kwarg, value in kwargs.items():
                ff_dataframe_kargs[kwarg] = value

            if self.player != 'monkey':
                ff_dataframe_kargs['obs_ff_indices_in_ff_dataframe'] = self.obs_ff_indices_in_ff_dataframe
                ff_dataframe_kargs['ff_in_obs_df'] = self.ff_in_obs_df

            self.ff_dataframe = make_ff_dataframe.make_ff_dataframe_func(
                *ff_dataframe_args, **ff_dataframe_kargs, player=self.player)
            print("made ff_dataframe")

            if save_into_h5:
                with pd.HDFStore(h5_file_pathway) as h5_store:
                    h5_store['ff_dataframe'] = self.ff_dataframe

        self.ff_dataframe = self.ff_dataframe.drop_duplicates()
        # do some final processing
        self.ff_dataframe['memory'] = self.ff_dataframe['memory'].astype('int')
        if len(self.ff_dataframe) > 0:
            self.min_point_index, self.max_point_index = np.min(np.array(
                self.ff_dataframe['point_index'])), np.max(np.array(self.ff_dataframe['point_index']))
        else:
            self.min_point_index, self.max_point_index = 0, 0
        self.caught_ff_num = len(self.ff_caught_T_new)

        make_ff_dataframe.add_essential_columns_to_ff_dataframe(
            self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted, 10, 25)
        if to_furnish_ff_dataframe:
            self.ff_dataframe = make_ff_dataframe.furnish_ff_dataframe(self.ff_dataframe, self.ff_real_position_sorted,
                                                                       self.ff_caught_T_new, self.ff_life_sorted)

    def make_or_retrieve_closest_stop_to_capture_df(self, exists_ok=True):
        path = os.path.join(self.processed_data_folder_path,
                            'closest_stop_to_capture_df.csv')
        if exists_ok & exists(path):
            self.closest_stop_to_capture_df = pd.read_csv(
                path).drop(["Unnamed: 0"], axis=1)
        else:
            self.closest_stop_to_capture_df = nxt_ff_utils.get_closest_stop_time_to_all_capture_time(self.ff_caught_T_sorted, self.monkey_information, self.ff_real_position_sorted,
                                                                                                     cur_ff_index_array=np.arange(len(self.ff_caught_T_sorted)))
            self.closest_stop_to_capture_df.to_csv(path)
        return

    def make_ff_caught_T_new(self, max_time_apart=0.3):
        self.ff_closest_stop_time_sorted = self.closest_stop_to_capture_df['time'].values
        self.ff_caught_T_new = self.ff_closest_stop_time_sorted.copy()

        # if the time difference between the closest stop time and the capture time is too large, then we should use the original capture time
        time_too_far_apart_points = np.where(np.abs(
            self.ff_closest_stop_time_sorted - self.ff_caught_T_sorted) > max_time_apart)[0]
        if len(time_too_far_apart_points) > 0:
            print(f"Warning: ff_closest_stop_time_sorted has {len(time_too_far_apart_points)} points out of {len(self.ff_caught_T_sorted)} points that are significantly larger than ff_caught_T_sorted, "
                  f"which is {len(time_too_far_apart_points)/len(self.ff_caught_T_sorted)*100:.2f}% of the points. "
                  f"Max value of closest_time - capture time is {abs(self.ff_closest_stop_time_sorted - self.ff_caught_T_sorted).max()}. "
                  f"They are replaced with the original ff_caught_T in ff_caught_T_new.")
            # replace the ff_caught_T_new with the original ff_caught_T_sorted, or previous element in ff_caught_T_new, whichever is bigger. In this way, we can make sure that the ff_caught_T_new is monotonically increasing
            prev_ff_caught_T = np.insert(self.ff_caught_T_new[:-1], 0, 0)
            self.ff_caught_T_new[time_too_far_apart_points] = np.maximum(
                self.ff_caught_T_sorted[time_too_far_apart_points], prev_ff_caught_T[time_too_far_apart_points])

        # also, if the new stop position is outside of the reward boundary, then we should use the original capture time
        outside_boundary_points = np.where(
            self.closest_stop_to_capture_df['whether_stop_inside_boundary'].values == 0)[0]
        if len(outside_boundary_points) > 0:
            print(f"Warning: ff_closest_stop_time_sorted has {len(outside_boundary_points)} points out of {len(self.ff_caught_T_sorted)} points that are outside of the reward boundary, "
                  f"which is {len(outside_boundary_points)/len(self.ff_caught_T_sorted)*100:.2f}% of the points. "
                  f"They are replaced with the original ff_caught_T in ff_caught_T_new.")
            prev_ff_caught_T = np.insert(self.ff_caught_T_new[:-1], 0, 0)
            self.ff_caught_T_new[outside_boundary_points] = np.maximum(
                self.ff_caught_T_sorted[outside_boundary_points], prev_ff_caught_T[outside_boundary_points])

        # now, check if self.ff_caught_T_new is monotonically increasing
        if not np.all(np.diff(self.ff_caught_T_new) >= 0):
            print("Warning: ff_caught_T_new is not monotonically increasing. Will make it monotonically increasing.")
            self.ff_caught_T_new = np.maximum.accumulate(self.ff_caught_T_new)

        self.closest_stop_to_capture_df['ff_caught_T_new'] = self.ff_caught_T_new
        self.closest_stop_to_capture_df['ff_caught_T_new_point_index'] = np.searchsorted(
            self.monkey_information['time'].values, self.ff_caught_T_new)

        print('Note: ff_caught_T_sorted is replaced with ff_caught_T_new')

        self.monkey_information['trial'] = np.searchsorted(
            self.ff_caught_T_new, self.monkey_information['time'])

        assert len(self.ff_caught_T_new) == len(self.ff_caught_T_sorted)

    def make_or_retrieve_target_clust_last_vis_df(self, exists_ok=True, max_distance_to_target_in_cluster=50):
        path = os.path.join(self.processed_data_folder_path,
                            'target_clust_last_vis_df.csv')
        if exists_ok & exists(path):
            self.target_clust_last_vis_df = pd.read_csv(
                path).drop(["Unnamed: 0"], axis=1)
            print("Retrieved target_clust_last_vis_df")
        else:
            self.target_clust_last_vis_df = cluster_analysis.get_target_clust_last_vis_df(self.ff_dataframe, self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted,
                                                                                          self.ff_life_sorted, max_distance_to_target_in_cluster=max_distance_to_target_in_cluster, keep_all_rows=True)
            self.target_clust_last_vis_df.to_csv(path)
            print("Made target_clust_last_vis_df and saved it at ", path)
        return

    def make_or_retrieve_target_last_vis_df(self, exists_ok=True):
        path = os.path.join(self.processed_data_folder_path,
                            'target_last_vis_df.csv')
        if exists_ok & exists(path):
            self.target_last_vis_df = pd.read_csv(
                path).drop(["Unnamed: 0"], axis=1)
            print("Retrieved target_last_vis_df")
        else:
            self.target_last_vis_df = cluster_analysis.get_target_last_vis_df(self.ff_dataframe, self.monkey_information,
                                                                              self.ff_caught_T_new, self.ff_real_position_sorted, duration_of_evaluation=3)
            self.target_last_vis_df.to_csv(path)
            print("Made target_last_vis_df and saved it at ", path)
        return

    def retrieve_or_make_monkey_data(self, exists_ok=True, already_made_ok=False, save_data=True, speed_threshold_for_distinct_stop=1, min_distance_to_calculate_angle=5):
        if already_made_ok & (getattr(self, 'monkey_information', None) is not None):
            return

        self.ff_caught_T_sorted, self.ff_index_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.ff_life_sorted, \
            self.ff_flash_end_sorted = retrieve_raw_data.make_or_retrieve_ff_info_from_txt_data(
                self.raw_data_folder_path, exists_ok=exists_ok, save_data=save_data)

        self.smr_markers_start_time, self.smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
            self.raw_data_folder_path)

        self.ff_flash_sorted = retrieve_raw_data.make_or_retrieve_ff_flash_sorted_from_txt_data(
            self.raw_data_folder_path, exists_ok=exists_ok, save_data=save_data)

        self.make_or_retrieve_monkey_information(
            exists_ok=exists_ok, save_data=save_data, min_distance_to_calculate_angle=min_distance_to_calculate_angle, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)
        self.make_or_retrieve_closest_stop_to_capture_df()
        self.make_ff_caught_T_new()

    def make_or_retrieve_monkey_information(self, exists_ok=True, save_data=True, min_distance_to_calculate_angle=5, speed_threshold_for_distinct_stop=1):
        self.interocular_dist = 4 if self.monkey_name == 'monkey_Bruno' else 3
        self.monkey_information = process_monkey_information.make_or_retrieve_monkey_information(self.raw_data_folder_path, self.interocular_dist, min_distance_to_calculate_angle=min_distance_to_calculate_angle,
                                                                                                 exists_ok=exists_ok, save_data=save_data, speed_threshold_for_distinct_stop=speed_threshold_for_distinct_stop)
        return

    def get_more_monkey_data(self, exists_ok=True):
        self.make_or_retrieve_ff_dataframe(
            exists_ok=exists_ok, to_furnish_ff_dataframe=False)
        self.crudely_furnish_ff_dataframe()
        # self.find_patterns()
        self.cluster_around_target_indices = None
        self.make_PlotTrials_args()

    def crudely_furnish_ff_dataframe(self):
        # instead of furnishing ff_dataframe in the line above, we just add a few columns, so as not to make ff_dataframe_too_big
        self.ff_dataframe[['monkey_angle', 'monkey_angle', 'monkey_dw', 'dt', 'cum_distance']] = self.monkey_information.loc[self.ff_dataframe['point_index'].values, [
            'monkey_angle', 'monkey_angle', 'monkey_dw', 'dt', 'cum_distance']].values
        self.ff_dataframe = self.ff_dataframe.drop(
            columns=['left_right', 'abs_delta_ff_angle', 'abs_delta_ff_angle_boundary'], errors='ignore')

    def get_curv_of_traj_df(self, window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False):
        self.curv_of_traj_params = {}
        self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
        self.curv_of_traj_params['window_for_curv_of_traj'] = window_for_curv_of_traj
        self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = truncate_curv_of_traj_by_time_of_capture
        self.curv_of_traj_df, self.traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new,
                                                                                                                        curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        self.curv_of_traj_trace_name = curv_of_traj_utils.get_curv_of_traj_trace_name(
            curv_of_traj_mode, window_for_curv_of_traj)

        return self.curv_of_traj_df

    def init_variations_list(self, folder_path=None):
        self.get_ref_point_params_based_on_mode()
        self.init_variations_list_func(folder_path)

    def get_ref_point_params_based_on_mode(self):
        self.ref_point_info = params_utils.add_values_and_marks_to_ref_point_info(
            self.ref_point_info)
        self.ref_point_params_based_on_mode = dict(
            [(k, v['values']) for k, v in self.ref_point_info.items()])

    def init_variations_list_func(self, folder_path=None, ref_point_params_based_on_mode=None):
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.ref_point_params_based_on_mode
        self.variations_list = specific_utils.init_variations_list_func(ref_point_params_based_on_mode, folder_path=folder_path,
                                                                        monkey_name=self.monkey_name)

    def make_PlotTrials_args(self):
        try:
            self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted,
                                    self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_new)
        except AttributeError:
            self.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted,
                                    self.ff_real_position_sorted, self.ff_believed_position_sorted, None, self.ff_caught_T_new)

    def _update_optimal_arc_type_and_related_paths(self, optimal_arc_type='norm_opt_arc'):
        # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
        self.optimal_arc_type = optimal_arc_type
        self.heading_info_partial_path = f'heading_info_df/{optimal_arc_type}'
        self.diff_in_curv_partial_path = f'diff_in_curv_df/{optimal_arc_type}'
        self.plan_x_partial_path = f'plan_x_df/{optimal_arc_type}'
        self.plan_y_partial_path = f'plan_y_df/{optimal_arc_type}'
