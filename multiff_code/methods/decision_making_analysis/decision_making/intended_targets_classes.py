
from decision_making_analysis.decision_making import decision_making_utils, decision_making_class, plot_decision_making

import math
import numpy as np


class ModelOfIntendedTargets(decision_making_class.DecisionMaking):
    def __init__(self, raw_data_folder_path=None, time_range_of_trajectory=[-1, 1], num_time_points_for_trajectory=10):
        super().__init__(raw_data_folder_path=raw_data_folder_path, time_range_of_trajectory=time_range_of_trajectory,
                         num_time_points_for_trajectory=num_time_points_for_trajectory)

    def retrieve_manual_anno(self):
        super().retrieve_manual_anno()

    def get_and_process_manual_anno_long(self, n_seconds_before_crossing_boundary=None, n_seconds_after_crossing_boundary=None):
        self.get_manual_anno_long()
        self.eliminate_crossing_boundary_cases(n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary,
                                               n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary)
        self.invalidate_ff_already_caught_as_intended_target()
        self.update_invalid_ff_indices_in_manual_anno_long()

    def get_manual_anno_long(self):
        super().retrieve_manual_anno()
        self.manual_anno_trimmed = self.manual_anno[self.manual_anno.target_index !=
                                                    self.manual_anno.target_index.min()]
        self.manual_anno_trimmed = self.manual_anno_trimmed[self.manual_anno_trimmed.target_index !=
                                                            self.manual_anno_trimmed.target_index.min()]
        # furnish manual_anno so that every point index has a row
        min_point = self.manual_anno_trimmed.starting_point_index.min()
        max_point = self.manual_anno_trimmed.starting_point_index.max()
        self.manual_anno_long = self.manual_anno_trimmed.copy()
        self.manual_anno_long['original_starting_point_index'] = self.manual_anno['starting_point_index']
        self.manual_anno_long = self.manual_anno_long.set_index(
            'starting_point_index')
        self.manual_anno_long = self.manual_anno_long.reindex(
            range(min_point, max_point+1))
        self.manual_anno_long = self.manual_anno_long.fillna(method='ffill')
        self.manual_anno_long = self.manual_anno_long.reset_index()

        # recalculate time and target_index based on starting_point_index
        self.manual_anno_long['time'] = self.monkey_information['time'].loc[self.manual_anno_long.starting_point_index.values].values
        self.manual_anno_long['target_index'] = np.searchsorted(
            self.ff_caught_T_new, self.manual_anno_long['time'].values)

        # change data type
        self.manual_anno_long['original_starting_point_index'] = self.manual_anno_long['original_starting_point_index'].astype(
            'int')
        self.manual_anno_long['starting_point_index'] = self.manual_anno_long['starting_point_index'].astype(
            'int')
        self.manual_anno_long['target_index'] = self.manual_anno_long['target_index'].astype(
            'int')
        self.manual_anno_long['ff_index'] = self.manual_anno_long['ff_index'].astype(
            'int')

    def eliminate_crossing_boundary_cases(self, n_seconds_before_crossing_boundary=None, n_seconds_after_crossing_boundary=None):
        n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary = self.determine_n_seconds_before_or_after_crossing_boundary(
            n_seconds_before_crossing_boundary, n_seconds_after_crossing_boundary
        )

        crossing_boundary_time = self.monkey_information.loc[
            self.monkey_information['crossing_boundary'] == 1, 'time'].values

        input_time = self.manual_anno_long.time.values
        original_length = len(input_time)
        CB_indices, non_CB_indices, left_input_time = decision_making_utils.find_time_points_that_are_within_n_seconds_after_crossing_boundary(input_time, crossing_boundary_time, n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary,
                                                                                                                                               n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary)
        self.manual_anno_long = self.manual_anno_long.iloc[non_CB_indices]
        # print("self.manual_anno_long:", self.manual_anno_long.shape[0], "out of", original_length, "rows remains")

    def invalidate_ff_already_caught_as_intended_target(self):
        # take out all the points where the ff_index is a ff that has just been captured, and change the ff_index = -9

        self.manual_anno_long['ff_capture_time'] = 99999
        valid_caught_ff_indices = self.manual_anno_long[(self.manual_anno_long['ff_index'] >= 0) & (
            self.manual_anno_long['ff_index'] < len(self.ff_caught_T_new))].index.values
        self.manual_anno_long.loc[valid_caught_ff_indices, 'ff_capture_time'] = self.ff_caught_T_new[
            self.manual_anno_long.loc[valid_caught_ff_indices, 'ff_index'].values.astype(int)]
        self.manual_anno_long.loc[self.manual_anno_long['time'] >
                                  self.manual_anno_long['ff_capture_time'], 'ff_index'] = -9

    def update_invalid_ff_indices_in_manual_anno_long(self):
        manual_anno_long = self.manual_anno_long.copy()
        # get sub_ff_index which is a candidate for substituting ff_index
        manual_anno_long['sub_ff_index'] = manual_anno_long['ff_index']
        # make sub_ff_index below 0 to be NA
        manual_anno_long.loc[manual_anno_long['sub_ff_index']
                             < 0, 'sub_ff_index'] = np.nan
        # fill NA with backward fill
        manual_anno_long['sub_ff_index'] = manual_anno_long['sub_ff_index'].fillna(
            method='bfill')
        # And then, in case there are leftover NAs near the end of the df, we fill it with -9
        manual_anno_long['sub_ff_index'] = manual_anno_long['sub_ff_index'].fillna(
            -9)
        manual_anno_long['sub_ff_index'] = manual_anno_long['sub_ff_index'].astype(
            int)

        # Now, we'll test whether the substitute (if it's a valid ff_index) is actually usable
        valid_sub_ff_index = manual_anno_long[manual_anno_long['sub_ff_index'] >= 0]['sub_ff_index'].unique(
        )
        # for each valid sub_ff_index
        for ff_index in valid_sub_ff_index:
            # get an array of visible time from ff_dataframe
            ff_time = self.ff_dataframe[self.ff_dataframe['ff_index']
                                        == ff_index]['time']
            # find correponding part in manual_anno_long where the new sub_ff_index cannot be used (because it's not available at that time point)
            manual_anno_long_invalid_indices = manual_anno_long[(manual_anno_long['sub_ff_index'] == ff_index) & (
                ~manual_anno_long['time'].isin(ff_time))].index.values
            if len(manual_anno_long_invalid_indices) > 0:
                # for those rows, discard the substitution (a.k.a. replace the sub_ff_index with ff_index)
                manual_anno_long.loc[manual_anno_long_invalid_indices,
                                     'sub_ff_index'] = manual_anno_long.loc[manual_anno_long_invalid_indices, 'ff_index']

        # Also change the invalid ff_index back to the original numbers
        manual_anno_long.loc[manual_anno_long['sub_ff_index'] < 0,
                             'sub_ff_index'] = manual_anno_long.loc[manual_anno_long['sub_ff_index'] < 0, 'ff_index']

        # print what percentage of manual_anno_long has invalid ff_index, and out of those, what percentage has been updated, and what percentage has not been updated
        print('Percentage of manual_anno_long has invalid ff_index: ', round(len(
            manual_anno_long[manual_anno_long['ff_index'] < 0])/len(manual_anno_long)*100, 1), '%')
        print('Out of the above:')
        print('Percentage of invalid ff_index that has been updated: ', round(len(manual_anno_long[(manual_anno_long['ff_index'] < 0) & (
            manual_anno_long['sub_ff_index'] >= 0)])/len(manual_anno_long[manual_anno_long['ff_index'] < 0])*100, 1), '%')
        print('percentage of invalid ff_index that has not been updated: ', round(len(manual_anno_long[(manual_anno_long['ff_index'] < 0) & (
            manual_anno_long['sub_ff_index'] < 0)])/len(manual_anno_long[manual_anno_long['ff_index'] < 0])*100, 1), '%')

        self.manual_anno_long['ff_index'] = manual_anno_long['sub_ff_index']

    def get_input_data(self, num_ff_per_row=5, select_every_nth_row=1, add_arc_info=False, arc_info_to_add=['opt_arc_curv', 'curv_diff'], curvature_df=None, curv_of_traj_df=None, **kwargs):
        self.free_selection_df = self.manual_anno_long
        super().get_free_selection_x(num_ff_per_row=num_ff_per_row, select_every_nth_row=select_every_nth_row,
                                     add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, curvature_df=curvature_df, curv_of_traj_df=curv_of_traj_df, **kwargs)

    def prepare_data_for_machine_learning(self, furnish_with_trajectory_data=True, trajectory_data_kind="position", add_traj_stops=True):
        # kind can also be "replacement"
        # trajectory_data_kind can also be "velocity"
        super().prepare_data_for_machine_learning(kind="free selection", furnish_with_trajectory_data=furnish_with_trajectory_data,
                                                  trajectory_data_kind=trajectory_data_kind, add_traj_stops=add_traj_stops)


# ==========================================================================


class ModelOfMultipleIntendedTargets(ModelOfIntendedTargets):
    def __init__(self, ff_dataframe, ff_caught_T_new, ff_real_position_sorted, monkey_information, ff_flash_sorted, ff_life_sorted,
                 time_range_of_trajectory=[-1, 1], num_time_points_for_trajectory=10):
        super().__init__(ff_dataframe, ff_caught_T_new, ff_real_position_sorted, monkey_information, ff_flash_sorted, ff_life_sorted,
                         time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)

    def prepare_data_for_machine_learning(self, furnish_with_trajectory_data=True, trajectory_data_kind="position", allow_multi_label=True):
        super().prepare_data_for_machine_learning(
            furnish_with_trajectory_data=furnish_with_trajectory_data, trajectory_data_kind=trajectory_data_kind)
        super().turn_y_label_into_multi_class(allow_multi_label=True)

    def prepare_to_plot_prediction_results(self, **kwargs):
        super(). prepare_to_plot_prediction_results(**kwargs)
        self.polar_plots_kwargs['labels'] = (self.y_test_to_plot == 1)
        self.polar_plots_kwargs['y_pred'] = (self.y_pred == 1)

    def plot_prediction_results(self, selected_cases=None, max_plot_to_make=30, **kwargs):
        if selected_cases is None:
            selected_cases = np.arange(len(self.X_test_to_plot))
        # Turn the formats into T/F for plotting

        if self.polar_plots_kwargs is None:
            self.prepare_to_plot_prediction_results()

        plot_decision_making.make_polar_plots_for_decision_making(**self.polar_plots_kwargs,
                                                                  selected_cases=selected_cases,
                                                                  max_plot_to_make=max_plot_to_make,
                                                                  data_kind=self.data_kind,
                                                                  **kwargs)


# ==========================================================================
# ==========================================================================
# ==========================================================================


# this needs to be updated since ModelOfIntendedTargets was updated
def test_moit_hyperparameters(ff_dataframe, ff_caught_T_new, ff_real_position_sorted, monkey_information, ff_flash_sorted, ff_life_sorted, auto_annot, auto_annot_long,
                              num_ff_per_row=5, select_every_nth_row=10, add_arc_info=True, arc_info_to_add=['opt_arc_curv', 'curv_diff'],
                              add_current_curv_of_traj=True, furnish_with_trajectory_data=True, keep_whole_chunks=False,
                              ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis'], trajectory_data_kind=['position'], curvature_df=None, curv_of_traj_df=None,
                              time_range_of_trajectory=[-0.8, 0], n_seconds_before_crossing_boundary=0, n_seconds_after_crossing_boundary=0.8,):

    ff_dataframe_temp = ff_dataframe.copy()
    ff_dataframe_temp = ff_dataframe_temp[abs(
        ff_dataframe_temp['ff_angle']) <= math.pi/4]
    ff_dataframe_truncated = ff_dataframe_temp[ff_dataframe_temp['time_since_last_vis'] <= 2.5]

    moit = ModelOfIntendedTargets(ff_dataframe_truncated, ff_caught_T_new, ff_real_position_sorted,
                                  monkey_information, ff_flash_sorted, ff_life_sorted, time_range_of_trajectory=time_range_of_trajectory)
    moit.manual_anno = auto_annot
    moit.manual_anno_long = auto_annot_long
    moit.eliminate_crossing_boundary_cases(n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary,
                                           n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary)
    moit.get_input_data(num_ff_per_row=num_ff_per_row, select_every_nth_row=select_every_nth_row,
                        add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, add_current_curv_of_traj=add_current_curv_of_traj,
                        curvature_df=curvature_df, curv_of_traj_df=curv_of_traj_df, ff_attributes=ff_attributes)

    moit.prepare_data_for_machine_learning(
        furnish_with_trajectory_data=furnish_with_trajectory_data, trajectory_data_kind=trajectory_data_kind)
    moit.split_data_to_train_and_test(
        scaling_data=True, keep_whole_chunks=keep_whole_chunks)
    #moit.use_machine_learning_model_for_classification(model=None)

    return moit.model_comparison_df
