
# where's arc type determined

oh, in combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types


def combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types(self):
    self.all_ref_pooled_median_info = make_variations_utils.combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types()
    self.process_all_ref_pooled_median_info_to_plot_heading_and_curv()
    return self.all_ref_pooled_median_info


def combine_all_ref_pooled_median_info_across_monkeys_and_opt_arc_types(all_ref_pooled_median_info_exists_ok=True,
                                                                        pooled_median_info_exists_ok=True):
    all_ref_pooled_median_info = pd.DataFrame([])
    for monkey_name in ['monkey_Schro', 'monkey_Bruno']:
        for opt_arc_type in ['norm_opt_arc', 'opt_arc_stop_closest', 'opt_arc_stop_first_vis_bdry']:
            # suppress printed output
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                ps = monkey_plan_factors_x_sess_class.PlanAcrossSessions(monkey_name=monkey_name,
                                                                         opt_arc_type=opt_arc_type)
                ref_pooled_median_info = ps.make_or_retrieve_all_ref_pooled_median_info(exists_ok=all_ref_pooled_median_info_exists_ok,
                                                                                        pooled_median_info_exists_ok=pooled_median_info_exists_ok,
                                                                                        process_info_for_plotting=False
                                                                                        )
                all_ref_pooled_median_info = pd.concat(
                    [all_ref_pooled_median_info, ref_pooled_median_info], axis=0)
    all_ref_pooled_median_info.reset_index(drop=True, inplace=True)
    return all_ref_pooled_median_info


AND


def make_pooled_median_info(self, ref_point_mode='time after cur ff visible',
                             ref_point_value=0.1,
                             curv_traj_window_before_stop=[-25, 0],
                             pooled_median_info_exists_ok=True,
                             combd_heading_df_x_sessions_exists_ok=True,
                             stops_near_ff_df_exists_ok=True,
                             heading_info_df_exists_ok=True,
                             verbose=False, save_data=True):

     df_name = find_cvn_utils.find_diff_in_curv_df_name(
          ref_point_mode, ref_point_value, curv_traj_window_before_stop)
      df_path = os.path.join(self.pooled_median_info_folder_path, df_name)
       if pooled_median_info_exists_ok & exists(df_path):
            self.pooled_median_info =

            pd.read_csv(df_path).drop(
                ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
            print('Successfully retrieved pooled_median_info from ', df_path)
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                   curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                   heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                   stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data,
                                                                   combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok)
            self.pooled_median_info = make_variations_utils.make_pooled_median_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
                                                                                                                       self.ctrl_heading_info_df, verbose=verbose)
            self.pooled_median_info['ref_point_mode'] = ref_point_mode
            self.pooled_median_info['ref_point_value'] = ref_point_value
            time_calibration = {'ref_point_mode': ref_point_mode,
                                'ref_point_value': ref_point_value, 'monkey_name': self.monkey_name}
            self.pooled_median_info.attrs.update(time_calibration)
            os.makedirs(self.pooled_median_info_folder_path, exist_ok=True)
            self.pooled_median_info.to_csv(df_path)
            print('Stored new pooled_median_info in ',
                  self.pooled_median_info_folder_path)
        return self.pooled_median_info


def make_temp_median_info_func(test_heading_info_df, ctrl_heading_info_df):
    row_from_test, row_from_ctrl = get_rows_from_test_and_ctrl(
        test_heading_info_df, ctrl_heading_info_df)
    row_from_test, row_from_ctrl = add_boostrap_median_std_to_df(test_heading_info_df, ctrl_heading_info_df,
                                                                 row_from_test, row_from_ctrl,
                                                                 columns=['diff_in_abs_angle_to_nxt_ff', 'diff_in_abs_d_curv'])

    def get_test_and_ctrl_heading_info_df_across_sessions(self, ref_point_mode='distance', ref_point_value=-150,
                                                          curv_traj_window_before_stop=[
                                                              -25, 0],
                                                          heading_info_df_exists_ok=True, combd_heading_df_x_sessions_exists_ok=True, stops_near_ff_df_exists_ok=True, save_data=True):
        self.sp = show_planning_class.ShowPlanning(monkey_name=self.monkey_name,
                                                   opt_arc_type=self.opt_arc_type)
        self.test_heading_info_df, self.ctrl_heading_info_df = self.sp.make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(ref_point_mode, ref_point_value,
                                                                                                                                               curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                                                               combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
                                                                                                                                               show_printed_output=True, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                                                                               stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)

    def make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(self, ref_point_mode='distance', ref_point_value=-100,
                                                                                curv_traj_window_before_stop=[
                                                                                    -25, 0],
                                                                                combd_heading_df_x_sessions_exists_ok=True,
                                                                                heading_info_df_exists_ok=True, stops_near_ff_df_exists_ok=True,
                                                                                show_printed_output=False, use_curv_to_ff_center=False, save_data=True):
        for test_or_control in ['control', 'test']:
            stops_near_ff_df_exists_ok = stops_near_ff_df_exists_ok if test_or_control == 'test' else stops_near_ff_df_exists_ok
            self.handle_heading_info_df(ref_point_mode, ref_point_value, combd_heading_df_x_sessions_exists_ok, heading_info_df_exists_ok, stops_near_ff_df_exists_ok,
                                        show_printed_output, test_or_control, curv_traj_window_before_stop=curv_traj_window_before_stop, use_curv_to_ff_center=use_curv_to_ff_center, save_data=save_data)
        return self.test_heading_info_df, self.ctrl_heading_info_df

    def _make_combd_heading_df_x_sessions(self, test_or_control='test',
                                          ref_point_mode='distance', ref_point_value=-100,
                                          curv_traj_window_before_stop=[
                                              -25, 0],
                                          stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                          sessions_df_for_one_monkey=None,
                                          use_curv_to_ff_center=False):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.curv_traj_window_before_stop = curv_traj_window_before_stop

        if sessions_df_for_one_monkey is not None:
            self.sessions_df_for_one_monkey = sessions_df_for_one_monkey
        else:
            self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
                self.raw_data_dir_name, self.monkey_name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, row in self.sessions_df_for_one_monkey.iterrows():
                if row['finished'] is True:
                    continue
                print(
                    f'Making heading_info_df for: {row["monkey_name"]} {row["data_name"]}')
                self.heading_info_df = self._make_heading_info_df_for_a_data_session(row['monkey_name'], row['data_name'], ref_point_mode=ref_point_mode,
                                                                                     ref_point_value=ref_point_value, test_or_control=test_or_control,
                                                                                     curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                     stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                     use_curv_to_ff_center=use_curv_to_ff_center,
                                                                                     merge_diff_in_curv_df_to_heading_info=False,
                                                                                     )
                self.heading_info_df['data_name'] = row['data_name']
                self.combd_heading_df_x_sessions = pd.concat(
                    [self.combd_heading_df_x_sessions, self.heading_info_df], axis=0)
                self.combd_diff_in_curv_df = pd.concat(
                    [self.combd_diff_in_curv_df, self.snf.diff_in_curv_df], axis=0)
                self.sessions_df_for_one_monkey.loc[self.sessions_df_for_one_monkey['data_name']
                                                    == row['data_name'], 'finished'] = True
        return self.combd_heading_df_x_sessions
