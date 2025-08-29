from data_wrangling import base_processing_class, general_utils
from pattern_discovery import pattern_by_trials, organize_patterns_and_features, monkey_landing_in_ff
from visualization.matplotlib_tools import plot_behaviors_utils
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from decision_making_analysis.GUAT import GUAT_utils
from pattern_discovery import pattern_by_points
from null_behaviors import find_best_arc, curvature_utils, curv_of_traj_utils, opt_arc_utils
from decision_making_analysis.decision_making import decision_making_utils
from neural_data_analysis.topic_based_neural_analysis.neural_vs_behavioral import prep_monkey_data, prep_target_data

import math
import numpy as np


import os
import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists


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


class FurtherProcessing(base_processing_class.BaseProcessing):
    def __init__(self, raw_data_folder_path=None):
        super().__init__()

        if raw_data_folder_path is not None:
            if not exists(raw_data_folder_path):
                raise ValueError(
                    f"raw_data_folder_path {raw_data_folder_path} does not exist.")
            self.extract_info_from_raw_data_folder_path(raw_data_folder_path)
            self.cluster_around_target_indices = None
        else:
            print("Warning: raw_data_folder_path is None")

        self.max_visibility_window = 10

    def make_df_related_to_patterns_and_features(self, exists_ok=True):

        self.make_or_retrieve_all_trial_patterns(exists_ok=exists_ok)

        self.make_or_retrieve_pattern_frequencies(exists_ok=exists_ok)

        self.make_or_retrieve_all_trial_features(exists_ok=exists_ok)

        self.make_or_retrieve_feature_statistics(exists_ok=exists_ok)

        self.make_or_retrieve_scatter_around_target_df(exists_ok=exists_ok)

    def _prepare_to_find_patterns_and_features(self, find_patterns=True):
        self.retrieve_or_make_monkey_data(already_made_ok=True)
        self.make_or_retrieve_ff_dataframe(
            already_made_ok=True, exists_ok=True)
        if find_patterns:
            self.find_patterns()

    def find_patterns(self):
        self.get_n_ff_in_a_row_info()
        self.on_before_last_one_trials = pattern_by_trials.on_before_last_one_func(
            self.ff_flash_end_sorted, self.ff_caught_T_new, self.caught_ff_num)
        self.on_before_last_one_simul, self.on_before_last_one_non_simul = pattern_by_trials.whether_current_and_last_targets_are_captured_simultaneously(
            self.on_before_last_one_trials, self.ff_caught_T_new)
        self.get_visible_before_last_one_trials_info()
        self.used_cluster = np.intersect1d(
            self.two_in_a_row_non_simul, self.visible_before_last_one_trials)
        self.disappear_latest_trials = pattern_by_trials.disappear_latest_func(
            self.ff_dataframe)
        self.cluster_around_target_trials, self.used_cluster_trials, self.cluster_around_target_indices, self.cluster_around_target_positions = pattern_by_trials.cluster_around_target_func(
            self.ff_dataframe, self.caught_ff_num, self.ff_caught_T_new, self.ff_real_position_sorted)
        # take out trials in cluster_around_target_trials but not in used_cluster_trials
        self.waste_cluster_around_target_trials = np.setdiff1d(
            self.cluster_around_target_trials, self.used_cluster_trials)
        self.ignore_sudden_flash_trials, self.sudden_flash_trials, self.ignore_sudden_flash_indices, self.ignore_sudden_flash_indices_for_anim, self.ignored_ff_target_pairs = pattern_by_trials.ignore_sudden_flash_func(
            self.ff_dataframe, self.max_point_index, max_ff_distance_from_monkey=50)
        self.ignore_sudden_flash_time = self.monkey_information[
            'time'][self.ignore_sudden_flash_indices]
        self.get_give_up_after_trying_info()
        self.get_try_a_few_times_info()

        self.all_categories = {'visible_before_last_one': self.visible_before_last_one_trials,
                               'disappear_latest': self.disappear_latest_trials,
                               'two_in_a_row': self.two_in_a_row,
                               'waste_cluster_around_target': self.waste_cluster_around_target_trials,
                               'try_a_few_times': self.try_a_few_times_trials,
                               'give_up_after_trying': self.give_up_after_trying_trials,
                               'ignore_sudden_flash': self.ignore_sudden_flash_trials,

                               # additional categories:
                               'two_in_a_row_simul': self.two_in_a_row_simul,
                               'two_in_a_row_non_simul': self.two_in_a_row_non_simul,
                               'used_cluster': self.used_cluster, }

    def get_n_ff_in_a_row_info(self):
        self.n_ff_in_a_row = pattern_by_trials.n_ff_in_a_row_func(
            self.ff_believed_position_sorted, distance_between_ff=50)
        self.two_in_a_row = np.where(self.n_ff_in_a_row == 2)[0]
        self.two_in_a_row_simul, self.two_in_a_row_non_simul = pattern_by_trials.whether_current_and_last_targets_are_captured_simultaneously(
            self.two_in_a_row, self.ff_caught_T_new)
        self.three_in_a_row = np.where(self.n_ff_in_a_row == 3)[0]
        self.four_in_a_row = np.where(self.n_ff_in_a_row == 4)[0]

    def get_visible_before_last_one_trials_info(self):
        self.make_or_retrieve_target_clust_df_short()
        self.visible_before_last_one_trials, self.vblo_target_cluster_df, self.selected_base_trials = pattern_by_trials.visible_before_last_one_func(
            self.target_clust_df_short, self.ff_caught_T_new)

    def get_try_a_few_times_info(self):

        self.try_a_few_times_trials, self.TAFT_indices_df, self.TAFT_trials_df, self.try_a_few_times_indices_for_anim = find_GUAT_or_TAFT_trials.try_a_few_times_func(
            self.monkey_information, self.ff_caught_T_new,  self.ff_real_position_sorted, max_point_index=self.max_point_index)

    def get_give_up_after_trying_info(self):

        self.give_up_after_trying_trials, self.GUAT_indices_df, self.GUAT_trials_df, self.GUAT_point_indices_for_anim, self.GUAT_w_ff_df = find_GUAT_or_TAFT_trials.give_up_after_trying_func(
            self.ff_dataframe, self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, max_point_index=self.max_point_index)
        self.give_up_after_trying_indices = self.GUAT_indices_df['point_index'].values
        self.give_up_after_trying_info_bundle = (
            self.give_up_after_trying_trials, self.GUAT_point_indices_for_anim, self.GUAT_indices_df, self.GUAT_trials_df)

    def make_or_retrieve_all_trial_patterns(self, exists_ok=True):
        self.all_trial_patterns = self.try_retrieving_df(
            df_name='all_trial_patterns', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.all_trial_patterns is None:
            if getattr(self, 'n_ff_in_a_row', None) is None:
                self._prepare_to_find_patterns_and_features()
            self.all_trial_patterns = self.make_all_trial_patterns()
            print("made all_trial_patterns")

    def make_or_retrieve_pattern_frequencies(self, exists_ok=True):
        self.pattern_frequencies = self.try_retrieving_df(
            df_name='pattern_frequencies', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)
        self.make_one_stop_w_ff_df()
        # Count one-stop misses (stops near but not at fireflies) for pattern frequency analysis
        self.one_stop_w_ff_frequency = len(self.one_stop_w_ff_df)
        if getattr(self, 'GUAT_w_ff_df', None) is None:
            self.get_give_up_after_trying_info()
        # Count give-up-after-trying events (multiple stops with firefly proximity) for pattern frequency analysis
        self.GUAT_w_ff_frequency = len(self.GUAT_w_ff_df)

        if self.pattern_frequencies is None:
            if getattr(self, 'monkey_information', None) is None:
                self.retrieve_or_make_monkey_data(already_made_ok=True)
            self.pattern_frequencies = organize_patterns_and_features.make_pattern_frequencies(self.all_trial_patterns, self.ff_caught_T_new, self.monkey_information,
                                                                                               self.GUAT_w_ff_frequency, self.one_stop_w_ff_frequency,
                                                                                               data_folder_name=self.patterns_and_features_data_folder_path)
            print("made pattern_frequencies")

    def make_or_retrieve_all_trial_features(self, exists_ok=True):
        self.all_trial_features = self.try_retrieving_df(
            df_name='all_trial_features', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.all_trial_features is None:
            if getattr(self, 'cluster_around_target_indices', None) is None:
                self._prepare_to_find_patterns_and_features()
            self.all_trial_features = organize_patterns_and_features.make_all_trial_features(self.ff_dataframe, self.monkey_information, self.ff_caught_T_new, self.cluster_around_target_indices,
                                                                                             self.ff_real_position_sorted, self.ff_believed_position_sorted, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made all_trial_features")

    def make_or_retrieve_feature_statistics(self, exists_ok=True):
        self.feature_statistics = self.try_retrieving_df(
            df_name='feature_statistics', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.feature_statistics is None:
            self.feature_statistics = organize_patterns_and_features.make_feature_statistics(
                self.all_trial_features, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made feature_statistics")

    def make_or_retrieve_scatter_around_target_df(self, exists_ok=True):
        self.scatter_around_target_df = self.try_retrieving_df(
            df_name='scatter_around_target_df', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.scatter_around_target_df is None:
            self.scatter_around_target_df = monkey_landing_in_ff.make_scatter_around_target_df(self.monkey_information,
                                                                                               self.closest_stop_to_capture_df,
                                                                                               self.ff_real_position_sorted,
                                                                                               data_folder_name=self.patterns_and_features_data_folder_path)
            print("made scatter_around_target_df")

    def plot_scatter_around_target_df(self):
        monkey_landing_in_ff.plot_scatter_around_target_df(
            self.closest_stop_to_capture_df, self.monkey_information, self.ff_real_position_sorted)

    def make_info_of_monkey(self):
        self.info_of_monkey = {"monkey_information": self.monkey_information,
                               "ff_dataframe": self.ff_dataframe,
                               "ff_caught_T_new": self.ff_caught_T_new,
                               "ff_real_position_sorted": self.ff_real_position_sorted,
                               "ff_believed_position_sorted": self.ff_believed_position_sorted,
                               "ff_life_sorted": self.ff_life_sorted,
                               "ff_flash_sorted": self.ff_flash_sorted,
                               "ff_flash_end_sorted": self.ff_flash_end_sorted,
                               "cluster_around_target_indices": self.cluster_around_target_indices}

    def make_all_trial_patterns(self):
        zero_array = np.zeros(self.caught_ff_num + 1, dtype=int)

        multiple_in_a_row = np.where(self.n_ff_in_a_row >= 2)[0]
        # multiple_in_a_row_all means it also includes the first ff that's caught in a cluster
        multiple_in_a_row_all = np.union1d(
            multiple_in_a_row, multiple_in_a_row - 1)
        multiple_in_a_row2 = zero_array.copy()
        multiple_in_a_row_all2 = zero_array.copy()
        multiple_in_a_row2[multiple_in_a_row] = 1
        multiple_in_a_row_all2[multiple_in_a_row_all] = 1

        two_in_a_row = np.where(self.n_ff_in_a_row == 2)[0]
        three_in_a_row = np.where(self.n_ff_in_a_row == 3)[0]
        four_in_a_row = np.where(self.n_ff_in_a_row == 4)[0]

        two_in_a_row2 = zero_array.copy()
        if len(two_in_a_row) > 0:
            two_in_a_row2[two_in_a_row] = 1

        three_in_a_row2 = zero_array.copy()
        if len(three_in_a_row) > 0:
            three_in_a_row2[three_in_a_row] = 1

        four_in_a_row2 = zero_array.copy()
        if len(four_in_a_row) > 0:
            four_in_a_row2[four_in_a_row] = 1

        one_in_a_row = np.where(self.n_ff_in_a_row < 2)[0]
        one_in_a_row2 = zero_array.copy()
        if len(one_in_a_row) > 0:
            one_in_a_row2[one_in_a_row] = 1

        visible_before_last_one2 = zero_array.copy()
        if len(self.visible_before_last_one_trials) > 0:
            visible_before_last_one2[self.visible_before_last_one_trials] = 1

        disappear_latest2 = zero_array.copy()
        if len(self.disappear_latest_trials) > 0:
            disappear_latest2[self.disappear_latest_trials] = 1

        sudden_flash_trials2 = zero_array.copy()
        if len(self.sudden_flash_trials) > 0:
            sudden_flash_trials2[self.sudden_flash_trials] = 1

        ignore_sudden_flash2 = zero_array.copy()
        if len(self.ignore_sudden_flash_trials) > 0:
            ignore_sudden_flash2[self.ignore_sudden_flash_trials] = 1

        try_a_few_times2 = zero_array.copy()
        if len(self.try_a_few_times_trials) > 0:
            try_a_few_times2[self.try_a_few_times_trials] = 1

        give_up_after_trying2 = zero_array.copy()
        if len(self.give_up_after_trying_trials) > 0:
            give_up_after_trying2[self.give_up_after_trying_trials] = 1

        cluster_around_target2 = zero_array.copy()
        if len(self.cluster_around_target_trials) > 0:
            cluster_around_target2[self.cluster_around_target_trials] = 1

        use_cluster2 = zero_array.copy()
        if len(self.used_cluster_trials) > 0:
            use_cluster2[self.used_cluster_trials] = 1

        waste_cluster_around_target2 = zero_array.copy()
        if len(self.waste_cluster_around_target_trials) > 0:
            waste_cluster_around_target2[self.waste_cluster_around_target_trials] = 1

        all_trial_patterns_dict = {
            # bool
            'two_in_a_row': two_in_a_row2,
            'three_in_a_row': three_in_a_row2,
            'four_in_a_row': four_in_a_row2,
            'one_in_a_row': one_in_a_row2,
            'multiple_in_a_row': multiple_in_a_row2,
            'multiple_in_a_row_all': multiple_in_a_row_all2,
            'visible_before_last_one': visible_before_last_one2,
            'disappear_latest': disappear_latest2,
            'sudden_flash': sudden_flash_trials2,
            'ignore_sudden_flash': ignore_sudden_flash2,
            'try_a_few_times': try_a_few_times2,
            'give_up_after_trying': give_up_after_trying2,
            'cluster_around_target': cluster_around_target2,
            'use_cluster': use_cluster2,
            'waste_cluster_around_target': waste_cluster_around_target2
        }

        for key, value in all_trial_patterns_dict.items():
            all_trial_patterns_dict[key] = value[:-1]

        self.all_trial_patterns = pd.DataFrame(all_trial_patterns_dict)

        if self.patterns_and_features_data_folder_path:
            general_utils.save_df_to_csv(
                self.all_trial_patterns, 'all_trial_patterns', self.patterns_and_features_data_folder_path)

        return self.all_trial_patterns

    def make_one_stop_w_ff_df(self):
        self._prepare_to_find_patterns_and_features(find_patterns=False)

        # one_stop_df: Long format dataframe with one row per (stop × nearby firefly)
        # Contains stops that are 25-50cm from fireflies, representing potential decision points
        # where the monkey stopped near but not at a firefly (one-stop misses)
        self.one_stop_df = GUAT_utils.streamline_getting_one_stop_df(
            self.monkey_information, self.ff_dataframe, self.ff_caught_T_new)
        
        # one_stop_w_ff_df: Wide format dataframe with one row per stop
        # Aggregates one_stop_df by grouping stops and selecting the most recently visible firefly
        # Contains nearby_alive_ff_indices (list of all nearby fireflies) and latest_visible_ff (primary firefly)
        self.one_stop_w_ff_df = GUAT_utils.make_one_stop_w_ff_df(
            self.one_stop_df)

    def make_distance_and_num_stops_df(self):
        self.distance_df = organize_patterns_and_features.make_distance_df(
            self.ff_caught_T_new, self.monkey_information, self.ff_believed_position_sorted)
        self.num_stops_df = organize_patterns_and_features.make_num_stops_df(
            self.distance_df, self.closest_stop_to_capture_df, self.ff_caught_T_new, self.monkey_information)

    def make_PlotTrials_kargs(self, classic_plot_kwargs=None, combined_plot_kwargs=None, animation_plot_kwargs=None, all_category_kwargs=None):

        if classic_plot_kwargs is not None:
            self.classic_plot_kwargs = classic_plot_kwargs

        if combined_plot_kwargs is not None:
            self.combined_plot_kwargs = combined_plot_kwargs

        if animation_plot_kwargs is not None:
            self.animation_plot_kwargs = animation_plot_kwargs
            self.all_category_animation_kwargs = plot_behaviors_utils.customize_kwargs_by_category(
                self.animation_plot_kwargs, images_dir=None)

        if all_category_kwargs is not None:
            self.all_category_kwargs = all_category_kwargs
        elif classic_plot_kwargs is not None:
            self.all_category_kwargs = plot_behaviors_utils.customize_kwargs_by_category(
                classic_plot_kwargs, images_dir=None)

    def plot_trials_from_a_category(self, category_name, max_trial_to_plot, trials=None, additional_kwargs=None, images_dir=None, using_subplots=False, figsize=(10, 10)):
        category = self.all_categories[category_name]
        plot_behaviors_utils.plot_trials_from_a_category(category, category_name, max_trial_to_plot, self.PlotTrials_args, self.all_category_kwargs,
                                                         self.ff_caught_T_new, trials=trials, additional_kwargs=additional_kwargs, images_dir=images_dir, using_subplots=using_subplots, figsize=figsize)

    # these may need to be run again if they're to be used
    def make_or_retrieve_target_closest(self, exists_ok=False):
        filepath = os.path.join(
            self.patterns_and_features_data_folder_path, 'target_closest.csv')
        if exists(filepath) & exists_ok:
            self.target_closest = np.genfromtxt(
                filepath, delimiter=',').astype('int')
            print("Retrieved target_closest")
        else:
            self.target_closest = pattern_by_points.make_target_closest(
                self.ff_dataframe, self.max_point_index, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made target_closest")

    def make_or_retrieve_target_angle_smallest(self, exists_ok=False):
        filepath = self.patterns_and_features_data_folder_path + '/target_angle_smallest.csv'
        if exists(filepath) & exists_ok:
            self.target_angle_smallest = np.genfromtxt(
                filepath, delimiter=',').astype('int')
            print("Retrieved target_angle_smallest")
        else:
            # make target_angle_smallest:
            # 2 means target is has the smallest absolute angle at that point (visible or in memory)
            # 1 means the target does not have the smallest absolute angle. In the subset of 1:
            # 1 means both the target and a non-target are visible or in memory (which we call present)
            # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
            # -1 means both the target and other ff are neither visible or in memory
            self.target_angle_smallest = pattern_by_points.make_target_angle_smallest(
                self.ff_dataframe, self.max_point_index, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made target_angle_smallest")

    def make_curvature_df(self, window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False, ff_radius_for_opt_arc=10, clean=True,
                          include_cntr_arc_curv=True, include_opt_arc_curv=True,
                          opt_arc_stop_first_vis_bdry=False,  # whether optimal arc stop at visible boundary
                          ignore_error=False):
        ff_dataframe = self.ff_dataframe.copy()
        self.window_for_curv_of_traj = window_for_curv_of_traj
        self.curv_of_traj_mode = curv_of_traj_mode
        self.truncate_curv_of_traj_by_time_of_capture = truncate_curv_of_traj_by_time_of_capture
        self.ff_radius_for_opt_arc = ff_radius_for_opt_arc

        self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
            window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        self.curvature_df = curvature_utils.make_curvature_df(ff_dataframe, self.curv_of_traj_df, ff_radius_for_opt_arc=ff_radius_for_opt_arc, clean=clean,
                                                              include_cntr_arc_curv=include_cntr_arc_curv, include_opt_arc_curv=include_opt_arc_curv,
                                                              opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry, ignore_error=ignore_error)
        self.curvature_point_index = self.curvature_df.point_index.values

    def find_opt_arc_info_from_curvature_df(self):
        self.all_point_index = self.curvature_df.point_index.values
        self.all_ff_indices = self.curvature_df.ff_index.values

        self.all_arc_lengths = self.curvature_df.opt_arc_length.values
        self.all_arc_radius = self.curvature_df.opt_arc_radius.values
        self.arc_end_direction = np.sign(
            self.curvature_df.opt_arc_curv.values)

        # find the angle from the monkey to the end point of the arc
        self.arc_end_angles = self.curvature_df.opt_arc_measure.values / \
            2  # if remembered correctly, this is based on a formula
        # for ff to the right
        self.arc_end_angles[self.arc_end_direction < 0] = - \
            self.arc_end_angles[self.arc_end_direction < 0]

    def make_or_retrieve_best_arc_df(self, exists_ok=True):
        self.best_arc_df = self.try_retrieving_df(
            df_name='best_arc_df', exists_ok=exists_ok, data_folder_name_for_retrieval=self.processed_data_folder_path)
        if self.best_arc_df is None:
            self.make_best_arc_df()
            print("made best_arc_df")
            self._save_best_arc_df()

    def make_best_arc_df(self):
        self.best_arc_df, self.best_arc_original_columns = find_best_arc.make_best_arc_df(
            self.curvature_df, self.monkey_information, self.ff_real_position_sorted)

    def _save_best_arc_df(self):
        general_utils.save_df_to_csv(
            self.best_arc_df, 'best_arc_df', self.processed_data_folder_path)

    def add_column_monkey_passed_by_to_best_arc_df(self):
        if 'monkey_passed_by' not in self.best_arc_df.columns:
            self.best_arc_df = find_best_arc.add_column_monkey_passed_by_to_best_arc_df(
                self.best_arc_df, self.ff_dataframe)

    def get_elements_for_plotting(self, opt_arc_stop_first_vis_bdry=False, ignore_error=False):
        arc_ff_xy = self.ff_real_position_sorted[self.all_ff_indices]
        monkey_xy = self.monkey_information.loc[self.curvature_df.point_index.values, [
            'monkey_x', 'monkey_y']].values
        monkey_angles = self.monkey_information.loc[self.curvature_df.point_index.values,
                                                    'monkey_angle'].values
        ff_distance = self.curvature_df.ff_distance.values
        ff_angle = self.curvature_df.ff_angle.values
        arc_point_index = self.curvature_df.point_index.values
        whether_ff_behind = (np.abs(ff_angle) > math.pi/2)
        self.center_x, self.center_y, self.arc_starting_angle, self.arc_ending_angle = opt_arc_utils.find_cartesian_arc_center_and_angle_for_opt_arc(arc_ff_xy, arc_point_index, monkey_xy, monkey_angles, ff_distance, ff_angle, self.all_arc_radius, np.sign(self.arc_end_angles),
                                                                                                                                                     whether_ff_behind=whether_ff_behind, opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry,
                                                                                                                                                     ignore_error=ignore_error)

    def make_auto_annot(self):
        if getattr(self, 'best_arc_df', None) is None:
            self.make_best_arc_df()
        self.auto_annot, self.auto_annot_long = decision_making_utils.make_auto_annot(
            self.best_arc_df, self.monkey_information, self.ff_caught_T_new)

    def make_or_retrieve_target_cluster_df(self, exists_ok=True, fill_na=False):
        target_cluster_df_filepath = os.path.join(
            self.patterns_and_features_data_folder_path, 'target_cluster_df.csv')
        if exists(target_cluster_df_filepath) & exists_ok:
            self.target_cluster_df = pd.read_csv(target_cluster_df_filepath)
            print("Retrieved target_cluster_df")
        else:
            self.target_cluster_df = prep_target_data.make_target_cluster_df(
                self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, self.ff_dataframe,
                self.ff_life_sorted, max_visibility_window=self.max_visibility_window)
            self.target_cluster_df.to_csv(
                target_cluster_df_filepath, index=False)
            print("Made new target_cluster_df")

        if fill_na:
            self.target_df = prep_target_data.fill_na_in_target_df(
                self.target_df)

    def make_or_retrieve_target_clust_df_short(self, exists_ok=True, fill_na=False):
        target_clust_df_short_filepath = os.path.join(
            self.patterns_and_features_data_folder_path, 'target_clust_df_short.csv')
        if exists(target_clust_df_short_filepath) & exists_ok:
            self.target_clust_df_short = pd.read_csv(
                target_clust_df_short_filepath)
            print("Retrieved target_clust_df_short")
        else:
            self.target_clust_df_short = prep_target_data.make_target_clust_df_short(
                self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, self.ff_dataframe, self.ff_life_sorted, max_visibility_window=self.max_visibility_window)
            self.target_clust_df_short.to_csv(
                target_clust_df_short_filepath, index=False)
            print("Made new target_clust_df_short and saved to ",
                  target_clust_df_short_filepath)

        if fill_na:
            self.target_clust_df_short = prep_target_data.fill_na_in_target_df(
                self.target_clust_df_short)
