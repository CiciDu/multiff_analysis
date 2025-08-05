import sys
from null_behaviors import curvature_utils, curv_of_traj_utils, show_null_trajectory, opt_arc_utils
from planning_analysis.show_planning import nxt_ff_utils, show_planning_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils, plot_cvn_class, cvn_helper_class
from planning_analysis.plan_factors import plan_factors_utils
from visualization.matplotlib_tools import monkey_heading_utils
import pandas as pd
import os
import sys
import copy


class CurVsNxtFfFromRefClass(cvn_helper_class._FindCurVsNxtFF, plot_cvn_class._PlotCurVsNxtFF):

    def __init__(self,
                 raw_data_folder_path=None,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
                 opt_arc_type='opt_arc_stop_closest',
                 ):
        super().__init__()

        self._init_empty_vars()
        self.update_opt_arc_type(opt_arc_type=opt_arc_type)

        self.overall_params = {
            **copy.deepcopy(self.default_overall_params),
            **self.overall_params
        }

        self.monkey_plot_params = {
            **copy.deepcopy(self.default_monkey_plot_params),
            **self.monkey_plot_params
        }

        if raw_data_folder_path is not None:
            self.load_raw_data(
                raw_data_folder_path, monkey_data_exists_ok=True, curv_of_traj_mode=None)
        else:
            self.monkey_information = None

    def update_opt_arc_type(self, opt_arc_type='opt_arc_stop_closest'):
        # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest
        super()._update_opt_arc_type_and_related_paths(opt_arc_type)

    def streamline_organizing_info(self,
                                   # ref_point_mode can be 'time', 'distance', or 'time after cur ff visible'
                                   ref_point_mode='distance', ref_point_value=-150,
                                   curv_traj_window_before_stop=[-50, 0],
                                   curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25], truncate_curv_of_traj_by_time_of_capture=False,
                                   eliminate_outliers=False, use_curv_to_ff_center=False, deal_with_rows_with_big_ff_angles=True,
                                   remove_i_o_modify_rows_with_big_ff_angles=False,
                                   stops_near_ff_df_exists_ok=True,
                                   heading_info_df_exists_ok=True, test_or_control='test',
                                   ):

        self.make_stops_near_ff_and_ff_comparison_dfs(
            test_or_control=test_or_control, exists_ok=stops_near_ff_df_exists_ok, save_data=True)
        # self._make_info_based_on_monkey_angle()
        # curv_of_traj_mode can be 'time', 'distance', or 'now to stop'
        self.curv_of_traj_params['curv_of_traj_mode'] = curv_of_traj_mode
        self.curv_of_traj_params['window_for_curv_of_traj'] = window_for_curv_of_traj
        self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = truncate_curv_of_traj_by_time_of_capture
        self.curv_of_traj_lower_end = window_for_curv_of_traj[0]
        self.curv_of_traj_upper_end = window_for_curv_of_traj[1]

        self.ref_point_params['ref_point_mode'] = ref_point_mode
        # ref_point_mode can be 'time', 'distance', or 'specific index', or 'time after cur ff visible'
        self.ref_point_params['ref_point_value'] = ref_point_value
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop

        self.overall_params['remove_i_o_modify_rows_with_big_ff_angles'] = remove_i_o_modify_rows_with_big_ff_angles
        self.overall_params['use_curv_to_ff_center'] = use_curv_to_ff_center

        self._get_nxt_ff_and_cur_ff_info_based_on_ref_point(ref_point_mode, ref_point_value,
                                                            deal_with_rows_with_big_ff_angles=deal_with_rows_with_big_ff_angles,
                                                            remove_i_o_modify_rows_with_big_ff_angles=remove_i_o_modify_rows_with_big_ff_angles)

        self._take_out_info_counted()
        self._find_curv_of_traj_counted()
        self.find_relative_curvature()
        if eliminate_outliers:
            self._eliminate_outliers_in_cur_ff_curv()
        # self._find_relative_heading_info()
        # below is more for plotting
        self._find_mheading_before_stop()

        # make some other useful df
        self.cur_and_nxt_ff_from_ref_df = self._make_cur_and_nxt_ff_from_ref_df()
        self.heading_info_df, self.diff_in_curv_df = self._make_heading_info_df(
            test_or_control, heading_info_df_exists_ok)

        self.kwargs_for_heading_plot = self._make_kwargs_for_heading_plot()
        if 'rank_by_angle_to_nxt_ff' not in self.stops_near_ff_df.columns:
            self.stops_near_ff_df = self.stops_near_ff_df.merge(self.heading_info_df[[
                                                                'stop_point_index', 'rank_by_angle_to_nxt_ff']], on='stop_point_index', how='left')

    def make_heading_info_df_without_long_process(self, test_or_control='test', ref_point_mode='time after cur ff visible', ref_point_value=0.0,
                                                  curv_traj_window_before_stop=[
                                                      -50, 0],
                                                  use_curv_to_ff_center=False, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                                  save_data=True, merge_diff_in_curv_df_to_heading_info=True,
                                                  deal_with_rows_with_big_ff_angles=True,
                                                  remove_i_o_modify_rows_with_big_ff_angles=False):

        self.ref_point_params = {
            'ref_point_mode': ref_point_mode, 'ref_point_value': ref_point_value}
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop
        self.overall_params['use_curv_to_ff_center'] = use_curv_to_ff_center

        self.make_stops_near_ff_and_ff_comparison_dfs(
            test_or_control=test_or_control, exists_ok=stops_near_ff_df_exists_ok, save_data=True)
        self._get_nxt_ff_and_cur_ff_info_based_on_ref_point(ref_point_mode, ref_point_value,
                                                            deal_with_rows_with_big_ff_angles=deal_with_rows_with_big_ff_angles,
                                                            remove_i_o_modify_rows_with_big_ff_angles=remove_i_o_modify_rows_with_big_ff_angles)
        self.cur_and_nxt_ff_from_ref_df = self._make_cur_and_nxt_ff_from_ref_df()
        self.heading_info_df, self.diff_in_curv_df = self._make_heading_info_df(test_or_control, heading_info_df_exists_ok=heading_info_df_exists_ok, save_data=save_data,
                                                                                merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)

    def _get_nxt_ff_and_cur_ff_info_based_on_ref_point(self, ref_point_mode, ref_point_value,
                                                       deal_with_rows_with_big_ff_angles=True,
                                                       remove_i_o_modify_rows_with_big_ff_angles=False):
        self.nxt_ff_df_from_ref, self.cur_ff_df_from_ref = self.find_nxt_ff_df_and_cur_ff_df_from_ref(
            ref_point_value, ref_point_mode)
        if deal_with_rows_with_big_ff_angles:
            self._deal_with_rows_with_big_ff_angles(
                remove_i_o_modify_rows_with_big_ff_angles=remove_i_o_modify_rows_with_big_ff_angles)
        else:
            self.nxt_ff_df_modified = self.nxt_ff_df_from_ref.copy()
            self.cur_ff_df_modified = self.cur_ff_df_from_ref.copy()
            self.stop_point_index_modified = self.nxt_ff_df_modified.stop_point_index.values.copy()
            self.stops_near_ff_df_modified = self.stops_near_ff_df.copy()
        self._add_curvature_info()
        self._add_d_heading_info()

    def make_or_retrieve_diff_in_curv_df(self, ref_point_mode, ref_point_value, test_or_control, curv_traj_window_before_stop=[-50, 0], exists_ok=True, save_data=True,
                                         merge_diff_in_curv_df_to_heading_info=True,
                                         only_try_retrieving=False):
        folder_path = os.path.join(
            self.planning_data_folder_path, self.diff_in_curv_partial_path, test_or_control)
        os.makedirs(folder_path, exist_ok=True)
        df_name = find_cvn_utils.find_diff_in_curv_df_name(
            ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        df_path = os.path.join(folder_path, df_name)
        if exists_ok & os.path.exists(df_path):
            self.diff_in_curv_df = pd.read_csv(df_path)
            print(f'Retrieving {df_name} from {df_path} succeeded')
        else:
            if not only_try_retrieving:
                self.make_diff_in_curv_df(
                    curv_traj_window_before_stop=curv_traj_window_before_stop)
                if save_data:
                    self.diff_in_curv_df.to_csv(df_path)
                    print(f'Stored {df_name} in {df_path}')
            else:
                raise FileNotFoundError(
                    f'{df_name} is not in the folder: {folder_path}')

        if merge_diff_in_curv_df_to_heading_info:
            if hasattr(self, 'heading_info_df'):
                self.heading_info_df = self.heading_info_df.merge(
                    self.diff_in_curv_df, on='stop_point_index', how='left')

        return self.diff_in_curv_df

    def make_diff_in_curv_df(self, curv_traj_window_before_stop=[-50, 0]):
        self.nxt_ff_info_for_null_arc = show_planning_utils.make_nxt_ff_info_for_null_arc(
            self.nxt_ff_df_modified, self.cur_ff_df_final, self.heading_info_df)
        self.nxt_ff_info_for_monkey = show_planning_utils.make_nxt_ff_info_for_monkey(self.nxt_ff_df_modified, self.heading_info_df, self.monkey_information,
                                                                                      self.ff_real_position_sorted, self.ff_caught_T_new,
                                                                                      curv_traj_window_before_stop=curv_traj_window_before_stop)
        self.diff_in_curv_df = show_planning_utils.make_diff_in_curv_df(
            self.nxt_ff_info_for_monkey, self.nxt_ff_info_for_null_arc)
        self.diff_in_curv_df = show_planning_utils.furnish_diff_in_curv_df(
            self.diff_in_curv_df)
        return self.diff_in_curv_df

    def _init_empty_vars(self):
        self.slope = None
        self.ff_dataframe = None
        self.nxt_ff_df_from_ref_test = None
        self.nxt_ff_df_from_ref_ctrl = None
        self.curv_of_traj_df = None
        self.shared_stops_near_ff_df = None
        self.curv_of_traj_params = {}
        self.ref_point_params = {}
        self.overall_params = {}
        self.monkey_plot_params = {}

    def find_nxt_ff_df_and_cur_ff_df_from_ref(self, ref_point_value, ref_point_mode
                                              # Note: ref_point_mode can be 'time', 'distance', ‘time after both ff visible’, or ‘time after cur ff visible’, etc
                                              ):

        # first get the description labels
        self.get_ref_point_descr_and_column(ref_point_mode, ref_point_value)

        # then get the actual nxt_ff_df_from_ref and cur_ff_df_from_ref
        self.nxt_ff_df_from_ref = find_cvn_utils.find_ff_info_based_on_ref_point(self.nxt_ff_df, self.monkey_information, self.ff_real_position_sorted,
                                                                                 ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                                 point_index_cur_ff_first_seen=self.cur_ff_df['point_index_ff_first_seen'].values)
        self.cur_ff_df_from_ref = find_cvn_utils.find_ff_info_based_on_ref_point(self.cur_ff_df, self.monkey_information, self.ff_real_position_sorted,
                                                                                 ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)

        return self.nxt_ff_df_from_ref, self.cur_ff_df_from_ref

    def get_ref_point_descr_and_column(self, ref_point_mode, ref_point_value):
        if ref_point_mode == 'time':
            if ref_point_value >= 0:
                raise ValueError(
                    'ref_point_value must be negative for ref_point_mode = "time"')
            self.ref_point_descr = 'based on %d s into past' % ref_point_value
            self.ref_point_column = 'rel_time'
            self.used_points_n_seconds_or_cm_ago = True
        elif ref_point_mode == 'distance':
            if ref_point_value >= 0:
                raise ValueError(
                    'ref_point_value must be negative for ref_point_mode = "distance"')
            self.ref_point_descr = 'based on %d cm into past' % ref_point_value
            self.ref_point_column = 'rel_distance'
            self.used_points_n_seconds_or_cm_ago = True
        elif ref_point_mode == 'time after cur ff visible':
            self.ref_point_descr = 'based on %d s ' % ref_point_value + \
                ref_point_mode[5:]
            self.ref_point_column = 'rel_time'
            self.used_points_n_seconds_or_cm_ago = True
        else:
            raise ValueError(
                'ref_point_mode must be either "time" or "distance" or "time after cur ff visible"')
        return

    def find_relative_curvature(self):

        # if 'heading_instead_of_curv' in self.overall_params:
        #     if self.overall_params['heading_instead_of_curv']:
        #         return
        # else:
        #     return

        if self.overall_params['use_curv_to_ff_center']:
            self.curv_var = 'cntr_arc_curv'
        else:
            self.curv_var = 'opt_arc_curv'

        self.traj_curv_counted, self.nxt_curv_counted = find_cvn_utils.find_relative_curvature(
            self.nxt_ff_counted_df, self.cur_ff_counted_df, self.curv_of_traj_counted, self.overall_params['use_curv_to_ff_center'])

        self.curv_for_correlation_df = pd.DataFrame({'traj_curv_counted': self.traj_curv_counted,
                                                     'nxt_curv_counted': self.nxt_curv_counted,
                                                     'stop_point_index': self.nxt_ff_counted_df['stop_point_index'].values,
                                                     'point_index': self.nxt_ff_counted_df['point_index'].values,
                                                     })

        self.curv_for_correlation_df['rank_by_traj_curv'] = self.curv_for_correlation_df['traj_curv_counted'].rank(
            method='first')
        self.curv_for_correlation_df['rank_by_traj_curv'] = self.curv_for_correlation_df['rank_by_traj_curv'].astype(
            'int')
        # add the column rank_by_angle_to_nxt_ff to stops_near_ff_df
        if 'rank_by_traj_curv' in self.stops_near_ff_df.columns:
            self.stops_near_ff_df.drop(
                columns=['rank_by_traj_curv'], inplace=True)
        self.stops_near_ff_df = self.stops_near_ff_df.merge(self.curv_for_correlation_df[[
                                                            'stop_point_index', 'rank_by_traj_curv']], on='stop_point_index', how='left')

    def find_relationships_from_info(self, normalize=False,
                                     change_units_to_degrees_per_m=True,
                                     show_plot=True):

        # try and see if traj_curv_counted_cleaned and nxt_curv_counted_cleaned are already made
        if 'traj_curv_counted' not in self.__dict__:
            self.find_relative_curvature()

        self.overall_params['change_units_to_degrees_per_m'] = change_units_to_degrees_per_m
        if normalize:
            traj_curv_counted_cleaned = (
                self.traj_curv_counted - self.traj_curv_counted.mean()) / self.traj_curv_counted.std()
            nxt_curv_counted_cleaned = (
                self.nxt_curv_counted - self.nxt_curv_counted.mean()) / self.nxt_curv_counted.std()
        else:
            traj_curv_counted_cleaned = self.traj_curv_counted.copy()
            nxt_curv_counted_cleaned = self.nxt_curv_counted.copy()

        ax_for_corr = find_cvn_utils.plot_relationship(
            nxt_curv_counted_cleaned, traj_curv_counted_cleaned, show_plot=show_plot, change_units_to_degrees_per_m=change_units_to_degrees_per_m)
        return ax_for_corr

    def _find_relative_heading_info(self):
        nxt_ff_df = self.nxt_ff_df_final.copy()

        self.d_heading_of_traj = nxt_ff_df['d_heading_of_traj'].values
        self.d_heading_cur = self.cur_ff_df_final['opt_arc_d_heading'].values
        self.d_heading_nxt = nxt_ff_df['opt_arc_d_heading'].values

        self.d_heading_of_traj = find_cvn_utils.confine_angle_to_within_one_pie(
            self.d_heading_of_traj)
        self.d_heading_cur = find_cvn_utils.confine_angle_to_within_one_pie(
            self.d_heading_cur)
        self.d_heading_nxt = find_cvn_utils.confine_angle_to_within_one_pie(
            self.d_heading_nxt)

        self.rel_heading_traj = self.d_heading_of_traj - self.d_heading_cur
        self.rel_heading_alt = self.d_heading_nxt - self.d_heading_cur

        self.rel_heading_traj = find_cvn_utils.confine_angle_to_within_one_pie(
            self.rel_heading_traj)
        self.rel_heading_alt = find_cvn_utils.confine_angle_to_within_one_pie(
            self.rel_heading_alt)

        self.rel_heading_df = pd.DataFrame({'rel_heading_traj': self.rel_heading_traj,
                                            'rel_heading_alt': self.rel_heading_alt,
                                            'stop_point_index': nxt_ff_df['stop_point_index'].values,
                                            'point_index': nxt_ff_df['point_index'].values,
                                            })

        # self.kwargs_for_heading_plot = self._make_kwargs_for_heading_plot()

    def _find_mheading_before_stop(self):
        # this is more for plotting
        traj_point_index_2d = self.stops_near_ff_df.loc[:, [
            'point_index_before_stop']].values
        self.mheading_before_stop_dict = monkey_heading_utils.find_mheading_in_xy(
            traj_point_index_2d, self.monkey_information)

        # transfer each value in the dict above using reshape(-1)
        for key in self.mheading_before_stop_dict.keys():
            self.mheading_before_stop_dict[key] = self.mheading_before_stop_dict[key].reshape(
                -1)
        self.mheading_before_stop = pd.DataFrame(
            self.mheading_before_stop_dict)
        self.mheading_before_stop[['stop_point_index', 'point_index_before_stop']] = self.stops_near_ff_df[[
            'stop_point_index', 'point_index_before_stop']].values

    def _find_nxt_ff_df_2_and_cur_ff_df_2_based_on_specific_point_index(self, all_point_index=None):
        print('nxt_ff_df_from_ref and cur_ff_df_from_ref are based on specific point_index')
        self.used_points_n_seconds_or_cm_ago = False
        if all_point_index is None:
            # all_point_index = self.stops_near_ff_df['earlest_point_index_when_nxt_ff_and_cur_ff_have_both_been_seen_bbas'].values
            all_point_index = self.cur_ff_df['point_index_ff_first_seen'].values
        self.nxt_ff_df_from_ref = find_cvn_utils.find_ff_info(
            self.nxt_ff_df.ff_index.values, all_point_index, self.monkey_information, self.ff_real_position_sorted)
        self.cur_ff_df_from_ref = find_cvn_utils.find_ff_info(
            self.cur_ff_df.ff_index.values, all_point_index, self.monkey_information, self.ff_real_position_sorted)

    def _make_cur_and_nxt_ff_from_ref_df(self):
        self.cur_and_nxt_ff_from_ref_df = show_planning_utils.make_cur_and_nxt_ff_from_ref_df(
            self.nxt_ff_df_final, self.cur_ff_df_final)
        return self.cur_and_nxt_ff_from_ref_df

    def _retrieve_heading_info_df(self, ref_point_mode, ref_point_value, test_or_control,
                                  curv_traj_window_before_stop=[-50, 0],
                                  merge_diff_in_curv_df_to_heading_info=True):

        self.diff_in_curv_df = self.make_or_retrieve_diff_in_curv_df(ref_point_mode, ref_point_value, test_or_control,
                                                                     curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                     exists_ok=True, merge_diff_in_curv_df_to_heading_info=False,
                                                                     only_try_retrieving=True)

        self.heading_info_df = show_planning_utils.retrieve_df_based_on_ref_point(
            ref_point_mode, ref_point_value, test_or_control, self.planning_data_folder_path, self.heading_info_partial_path, self.monkey_name)

        if merge_diff_in_curv_df_to_heading_info:
            self.heading_info_df = self.heading_info_df.merge(
                self.diff_in_curv_df, on='stop_point_index', how='left')
        return self.heading_info_df, self.diff_in_curv_df

    def _make_heading_info_df(self, test_or_control='test', heading_info_df_exists_ok=True, diff_in_curv_df_exists_ok=True, save_data=True,
                              merge_diff_in_curv_df_to_heading_info=True):
        self.heading_info_path = os.path.join(
            self.planning_data_folder_path, self.heading_info_partial_path, test_or_control)
        os.makedirs(self.heading_info_path, exist_ok=True)
        try:
            if heading_info_df_exists_ok is False:
                print(
                    'Will make new heading_info_df because heading_info_df_exists_ok is False')
                raise Exception
            self.heading_info_df, self.diff_in_curv_df = self._retrieve_heading_info_df(self.ref_point_params['ref_point_mode'], self.ref_point_params['ref_point_value'], test_or_control,
                                                                                        curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                                                                        merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)
        except Exception as e:
            if heading_info_df_exists_ok:
                print(
                    f'Failed to retrieve heading_info_df because of {e}; will make new heading_info_df')
            self.heading_info_df = show_planning_utils.make_heading_info_df(
                self.cur_and_nxt_ff_from_ref_df, self.stops_near_ff_df_modified, self.monkey_information, self.ff_real_position_sorted)
            self.heading_info_df['rank_by_angle_to_nxt_ff'] = self.heading_info_df['angle_from_m_before_stop_to_nxt_ff'].rank(
                method='first')
            self.heading_info_df['rank_by_angle_to_nxt_ff'] = self.heading_info_df['rank_by_angle_to_nxt_ff'].astype(
                'int')
            # add the column rank_by_angle_to_nxt_ff to stops_near_ff_df
            if 'rank_by_angle_to_nxt_ff' not in self.stops_near_ff_df.columns:
                self.stops_near_ff_df = self.stops_near_ff_df.merge(self.heading_info_df[[
                                                                    'stop_point_index', 'rank_by_angle_to_nxt_ff']], on='stop_point_index', how='left')

            if 'nxt_ff_angle_at_ref' not in self.heading_info_df.columns:
                self.add_both_ff_at_ref_to_heading_info_df()

            df_name = find_cvn_utils.get_df_name_by_ref(
                self.monkey_name, self.ref_point_params['ref_point_mode'], self.ref_point_params['ref_point_value'])
            if save_data:
                self.heading_info_df.to_csv(
                    os.path.join(self.heading_info_path, df_name))
                print(
                    f'Stored new heading_info_df ({df_name}) ({len(self.heading_info_df)} rows) in {os.path.join(self.heading_info_path, df_name)}')

            self.diff_in_curv_df = self.make_or_retrieve_diff_in_curv_df(self.ref_point_params['ref_point_mode'], self.ref_point_params['ref_point_value'], test_or_control,
                                                                         curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                                                         exists_ok=diff_in_curv_df_exists_ok, save_data=save_data, merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)

        return self.heading_info_df, self.diff_in_curv_df

    # choose one function from the above
    # ===================================================================================================

    def add_both_ff_at_ref_to_heading_info_df(self):
        self.both_ff_at_ref_df = self.get_both_ff_at_ref_df()
        self.both_ff_at_ref_df['stop_point_index'] = self.nxt_ff_df_from_ref['stop_point_index'].values
        columns_to_add = [
            col for col in self.both_ff_at_ref_df.columns if col not in self.heading_info_df.columns]
        self.heading_info_df = self.heading_info_df.merge(
            self.both_ff_at_ref_df[columns_to_add + ['stop_point_index']], on='stop_point_index', how='left')

    def _deal_with_rows_with_big_ff_angles(self, remove_i_o_modify_rows_with_big_ff_angles=False, verbose=True, delete_the_same_rows=True):

        if 'heading_instead_of_curv' in self.overall_params:
            if not self.overall_params['heading_instead_of_curv']:
                # if we want to focus on curv rather than heading, then delete the same rows
                delete_the_same_rows = True

        # prepare each df for curvature calculation, since optimal curvature cannot be calculated by the algorithm
        # when the absolute angle is greater than 45 degrees
        # Note, even when using remove_i_o_modify_rows_with_big_ff_angles, some rows are deleted if the ff is behind the monkey
        self.nxt_ff_df_modified, indices_of_kept_rows = find_cvn_utils.modify_position_of_ff_with_big_angle_for_finding_null_arc(
            self.nxt_ff_df_from_ref, remove_i_o_modify_rows_with_big_ff_angles=remove_i_o_modify_rows_with_big_ff_angles, verbose=verbose)
        self.cur_ff_df_modified = self.cur_ff_df_from_ref.copy()
        if delete_the_same_rows:
            self.cur_ff_df_modified = self.cur_ff_df_modified.iloc[indices_of_kept_rows].copy(
            )
        self.cur_ff_df_modified, indices_of_kept_rows = find_cvn_utils.modify_position_of_ff_with_big_angle_for_finding_null_arc(
            self.cur_ff_df_modified, remove_i_o_modify_rows_with_big_ff_angles=remove_i_o_modify_rows_with_big_ff_angles, verbose=verbose)
        self.cur_ff_df_modified = self.cur_ff_df_modified.reset_index(
            drop=True)
        if delete_the_same_rows:
            self.nxt_ff_df_modified = self.nxt_ff_df_modified.iloc[indices_of_kept_rows].reset_index(
                drop=True)
        self.stop_point_index_modified = self.nxt_ff_df_modified.stop_point_index.values.copy()
        self.stops_near_ff_df_modified = self.stops_near_ff_df.set_index(
            'stop_point_index').loc[self.stop_point_index_modified].reset_index()

    def _make_curv_of_traj_df_if_not_already_made(self, window_for_curv_of_traj=[-25, 25], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False):
        if getattr(self, 'curv_of_traj_df', None) is None:
            self.curv_of_traj_df, _ = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new,
                                                                                                         curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)

    def _add_curvature_info(self):

        self._make_curv_of_traj_df_if_not_already_made(
            **self.curv_of_traj_params)

        opt_arc_stop_first_vis_bdry = True if (
            self.opt_arc_type == 'opt_arc_stop_first_vis_bdry') else False

        self.nxt_curv_df = curvature_utils.make_curvature_df(self.nxt_ff_df_modified, self.curv_of_traj_df, clean=False, monkey_information=self.monkey_information,
                                                             ff_caught_T_new=self.ff_caught_T_new, remove_invalid_rows=False,
                                                             invalid_curvature_ok=True, ignore_error=True, opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry)
        self.cur_curv_df = curvature_utils.make_curvature_df(self.cur_ff_df_modified, self.curv_of_traj_df, clean=False, monkey_information=self.monkey_information,
                                                             ff_caught_T_new=self.ff_caught_T_new, remove_invalid_rows=False,
                                                             invalid_curvature_ok=False, ignore_error=True, opt_arc_stop_first_vis_bdry=opt_arc_stop_first_vis_bdry)

        if self.opt_arc_type == 'opt_arc_stop_closest':
            stop_and_ref_point_info = self.cur_ff_df_modified[[
                'stop_point_index', 'point_index', 'ff_index', 'ff_x', 'ff_y', 'monkey_x', 'monkey_y']].copy()
            stop_and_ref_point_info = stop_and_ref_point_info.merge(self.stops_near_ff_df_modified[[
                                                                    'stop_point_index', 'stop_x', 'stop_y']], on='stop_point_index', how='left')
            # if stop_x and stop_y don't exist
            # stop_and_ref_point_info['stop_x'], stop_and_ref_point_info['stop_y'] = self.monkey_information.loc[stop_and_ref_point_info['stop_point_index'], [
            #     'monkey_x', 'monkey_y']].values.T

            self.cur_curv_df = opt_arc_utils.update_curvature_df_to_let_opt_arc_stop_at_closest_point_to_monkey_stop(self.cur_curv_df, stop_and_ref_point_info,
                                                                                                                     self.ff_real_position_sorted, self.monkey_information)

        # use merge to add curvature_info
        shared_columns = ['ff_index', 'point_index', 'opt_arc_curv', 'opt_arc_measure', 'opt_arc_radius', 'opt_arc_end_direction', 'curv_of_traj', 'cntr_arc_curv',
                          'cntr_arc_radius', 'cntr_arc_d_heading', 'opt_arc_d_heading', 'opt_arc_end_x', 'opt_arc_end_y', 'cntr_arc_end_x', 'cntr_arc_end_y']
        self.nxt_ff_df_final = self.nxt_ff_df_modified.merge(
            self.nxt_curv_df[shared_columns], on=['ff_index', 'point_index'], how='left')
        self.cur_ff_df_final = self.cur_ff_df_modified.merge(
            self.cur_curv_df[shared_columns], on=['ff_index', 'point_index'], how='left')

    def _add_d_heading_info(self):

        self.nxt_ff_df_final = self.add_d_heading_of_traj_to_df(
            self.nxt_ff_df_final)
        self.cur_ff_df_final = self.add_d_heading_of_traj_to_df(
            self.cur_ff_df_final)

    def add_d_heading_of_traj_to_df(self, df):
        df = df.merge(self.stops_near_ff_df[[
                      'stop_point_index', 'monkey_angle_before_stop']], on='stop_point_index', how='left')
        df = plan_factors_utils.add_d_heading_of_traj_to_df(df)
        return df

    def _take_out_info_counted(self):
        # before eliminating outliers, the counted rows are just the same as the original ones
        self.cur_ff_counted_df = self.cur_ff_df_final.copy().reset_index(drop=True)
        self.nxt_ff_counted_df = self.nxt_ff_df_final.copy().reset_index(drop=True)

        self.ref_point_index_counted = self.nxt_ff_df_final.point_index.values.copy()
        self.stop_point_index_counted = self.nxt_ff_df_final.stop_point_index.values.copy()
        self.stops_near_ff_df_counted = self.stops_near_ff_df.set_index(
            'stop_point_index').loc[self.stop_point_index_counted].reset_index()

    def _find_curv_of_traj_counted(self):
        self.curv_of_traj_counted = self.cur_ff_counted_df['curv_of_traj'].values

    def _eliminate_outliers_in_cur_ff_curv(self):

        if 'heading_instead_of_curv' in self.overall_params:
            if self.overall_params['heading_instead_of_curv']:
                return

        self.outlier_positions, self.non_outlier_positions = find_cvn_utils.find_outliers_in_a_column(
            self.cur_ff_counted_df, self.curv_var)
        self.traj_curv_counted = self.traj_curv_counted[self.non_outlier_positions].copy(
        )
        self.nxt_curv_counted = self.nxt_curv_counted[self.non_outlier_positions]
        self.curv_for_correlation_df = self.curv_for_correlation_df.iloc[self.non_outlier_positions].reset_index(
            drop=True)
        self.ref_point_index_counted = self.ref_point_index_counted[self.non_outlier_positions]
        self.stop_point_index_counted = self.stop_point_index_counted[self.non_outlier_positions]
        self.curv_of_traj_counted = self.curv_of_traj_counted[self.non_outlier_positions].copy(
        )
        self.nxt_ff_counted_df = self.nxt_ff_df_final.iloc[self.non_outlier_positions].copy(
        ).reset_index(drop=True)
        self.cur_ff_counted_df = self.cur_ff_df_final.iloc[self.non_outlier_positions].copy(
        ).reset_index(drop=True)
        self.stops_near_ff_df_counted = self.stops_near_ff_df.set_index(
            'stop_point_index').loc[self.stop_point_index_counted].reset_index()

    def _prepare_data_to_compare_test_and_control(self):
        if self.nxt_ff_df_from_ref_test is None:
            self.prepare_stop_near_ff_and_ff_comparison_dfs_test(
                exists_ok=True)
            self._find_nxt_ff_df_2_and_cur_ff_df_2_based_on_specific_point_index(
                all_point_index=self.cur_ff_df['point_index_ff_first_seen'].values)
            self.nxt_ff_df_from_ref_test = self.nxt_ff_df_from_ref.copy()
        if self.nxt_ff_df_from_ref_ctrl is None:
            self.prepare_stop_near_ff_and_ff_comparison_dfs_ctrl(
                exists_ok=True)
            self._find_nxt_ff_df_2_and_cur_ff_df_2_based_on_specific_point_index(
                all_point_index=self.cur_ff_df['point_index_ff_first_seen'].values)
            self.nxt_ff_df_from_ref_ctrl = self.nxt_ff_df_from_ref.copy()

    def select_control_data_based_on_whether_nxt_ff_cluster_visible_pre_stop(self, max_distance_between_ffs_in_cluster=50):
        self.stops_near_ff_df_ctrl = nxt_ff_utils.find_if_nxt_ff_cluster_visible_pre_stop(self.stops_near_ff_df_ctrl, self.ff_dataframe,
                                                                                          self.ff_real_position_sorted, max_distance_between_ffs_in_cluster=max_distance_between_ffs_in_cluster)
        # print the number of rows out of total rows such that if_nxt_ff_cluster_visible_pre_stop is True
        print('Number of rows out of total rows such that if_nxt_ff_cluster_visible_pre_stop is True:',
              self.stops_near_ff_df_ctrl['if_nxt_ff_cluster_visible_pre_stop'].sum(), 'out of', len(self.stops_near_ff_df_ctrl))
        self.stops_near_ff_df_ctrl = self.stops_near_ff_df_ctrl[self.stops_near_ff_df_ctrl[
            'if_nxt_ff_cluster_visible_pre_stop'] == False].reset_index(drop=True).copy()

    def get_both_ff_at_ref_df(self):
        self.nxt_ff_df_from_ref, self.cur_ff_df_from_ref = self.find_nxt_ff_df_and_cur_ff_df_from_ref(
            self.ref_point_value, self.ref_point_mode)
        self.both_ff_at_ref_df = self.nxt_ff_df_from_ref[[
            'ff_distance', 'ff_angle']].copy()
        self.both_ff_at_ref_df.rename(columns={'ff_distance': 'nxt_ff_distance_at_ref',
                                               'ff_angle': 'nxt_ff_angle_at_ref'}, inplace=True)
        self.both_ff_at_ref_df = pd.concat([self.both_ff_at_ref_df.reset_index(drop=True), self.cur_ff_df_from_ref[['ff_distance', 'ff_angle',
                                                                                                                    'ff_angle_boundary']].reset_index(drop=True)], axis=1)
        self.both_ff_at_ref_df.rename(columns={'ff_distance': 'cur_ff_distance_at_ref',
                                               'ff_angle': 'cur_ff_angle_at_ref',
                                               'ff_angle_boundary': 'cur_ff_angle_boundary_at_ref'}, inplace=True)

        # self.both_ff_at_ref_df = self.heading_info_df[['nxt_ff_distance_at_ref', 'nxt_ff_angle_at_ref',
        #                                             'cur_ff_distance_at_ref', 'cur_ff_angle_at_ref',
        #                                             'cur_ff_angle_boundary_at_ref']].copy()
        return self.both_ff_at_ref_df

    def _make_info_based_on_monkey_angle(self):
        self.info_based_on_monkey_angle_before_stop = find_cvn_utils.calculate_info_based_on_monkey_angles(
            self.stops_near_ff_df, self.stops_near_ff_df.monkey_angle_before_stop.values)

    def _make_kwargs_for_correlation_plot(self):

        self.kwargs_for_correlation_plot = {
            'curv_for_correlation_df': self.curv_for_correlation_df.copy(),
            'change_units_to_degrees_per_m': self.overall_params['change_units_to_degrees_per_m'],
            'ref_point_descr': self.ref_point_descr,
            'traj_curv_descr': self.traj_curv_descr}
        return self.kwargs_for_correlation_plot

    def _make_kwargs_for_heading_plot(self):
        self.kwargs_for_heading_plot = {'heading_info_df': self.heading_info_df,
                                        # 'rel_heading_df': self.rel_heading_df,
                                        'change_units_to_degrees': True,
                                        'ref_point_descr': self.ref_point_descr,
                                        'traj_curv_descr': self.traj_curv_descr, }

        return self.kwargs_for_heading_plot
