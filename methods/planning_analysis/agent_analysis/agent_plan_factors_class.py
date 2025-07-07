from planning_analysis.plan_factors import plan_factors_class
from machine_learning.RL.SB3 import sb3_for_multiff_class, rl_for_multiff_class


class PlanFactorsOfAgent():

    def __init__(self,
                 # overall_folder_name='RL_models/SB3_stored_models/all_agents/env1_relu',
                 model_folder_name='RL_models/SB3_stored_models/all_agents/env1_relu/ff3/dv10_dw10_w10_mem3',
                 data_name='data_0',
                 use_curvature_to_ff_center=False,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 optimal_arc_type='opt_arc_stop_closest',
                 ):

        self.monkey_name = None
        self.model_folder_name = model_folder_name
        self.data_name = data_name
        self.use_curvature_to_ff_center = use_curvature_to_ff_center
        self.optimal_arc_type = optimal_arc_type
        rl_for_multiff_class._RLforMultifirefly.get_related_folder_names_from_model_folder_name(
            self, self.model_folder_name, data_name=data_name)

    def get_agent_data(self, n_steps=8000, exists_ok=False, save_data=False, **env_kwargs):
        episode_len = int(n_steps * 1.2)
        env_kwargs['episode_len'] = episode_len

        self.rl_ff = sb3_for_multiff_class.SB3forMultifirefly(model_folder_name=self.model_folder_name,
                                                              data_name=self.data_name,
                                                              overall_folder='',
                                                              **env_kwargs)
        self.rl_ff.streamline_getting_data_from_agent(
            n_steps=n_steps, exists_ok=exists_ok, save_data=save_data)

    def make_animation(self, currentTrial=None, num_trials=None, duration=[10, 20], video_dir=None):
        self.rl_ff.set_animation_parameters(
            currentTrial=currentTrial, num_trials=num_trials, k=1, duration=duration)
        self.rl_ff.call_animation_function(video_dir=video_dir)

    def _copy_df_from_pn_to_self(self):

        try:
            self.plan_y_tc = self.pf.plan_y_tc
            self.plan_x_tc = self.pf.plan_x_tc
            self.plan_x_test = self.pf.plan_x_test
            self.plan_y_test = self.pf.plan_y_test
            self.plan_x_ctrl = self.pf.plan_x_ctrl
            self.plan_y_ctrl = self.pf.plan_y_ctrl
        except AttributeError:
            pass
        try:
            self.test_heading_info_df = self.pf.test_heading_info_df
            self.ctrl_heading_info_df = self.pf.ctrl_heading_info_df
        except AttributeError:
            pass

    def _load_agent_data_onto_pf(self):
        for attr in ['ff_dataframe',
                     'monkey_information',
                     'ff_caught_T_new',
                     'ff_real_position_sorted',
                     'ff_believed_position_sorted',
                     'ff_life_sorted',
                     'ff_flash_sorted',
                     'closest_stop_to_capture_df'
                     ]:
            setattr(self.pf, attr, getattr(self.rl_ff, attr))

    def _initialize_pf(self, **kwargs):

        self.pf = plan_factors_class.PlanFactors(raw_data_folder_path=None,
                                                 optimal_arc_type=self.optimal_arc_type,
                                                 **kwargs)

        self.pf.processed_data_folder_path = self.processed_data_folder_path
        self.pf.planning_data_folder_path = self.planning_data_folder_path
        self.pf.patterns_and_features_data_folder_path = self.patterns_and_features_data_folder_path
        self.pf.decision_making_folder_path = self.decision_making_folder_path
        self.pf.monkey_name = 'monkey_agent'

    def get_plan_x_and_plan_y_for_one_session(self, ref_point_mode='distance', ref_point_value=-150,
                                              curv_traj_window_before_stop=[
                                                  -50, 0],
                                              monkey_data_exists_ok=True,
                                              plan_x_exists_ok=False,
                                              plan_y_exists_ok=False,
                                              heading_info_df_exists_ok=False,
                                              stops_near_ff_df_exists_ok=False,
                                              use_curvature_to_ff_center=False,
                                              save_data=True,
                                              curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25],
                                              n_steps=8000,
                                              **env_kwargs):

        self._initialize_pf(curv_of_traj_mode=curv_of_traj_mode,
                            window_for_curv_of_traj=window_for_curv_of_traj)

        kwargs = dict(plan_x_exists_ok=plan_x_exists_ok,
                      plan_y_exists_ok=plan_y_exists_ok,
                      heading_info_df_exists_ok=heading_info_df_exists_ok,
                      stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                      ref_point_mode=ref_point_mode,
                      ref_point_value=ref_point_value,
                      curv_traj_window_before_stop=curv_traj_window_before_stop,
                      use_curvature_to_ff_center=use_curvature_to_ff_center,
                      use_eye_data=False,
                      already_made_ok=False,
                      save_data=save_data)

        try:  # needs agent data
            self.pf.make_plan_x_and_y_for_both_test_and_ctrl(**kwargs)
        except AttributeError as e:
            print('Data missing. Will get agent data first. Error message: ', e)
            self.get_agent_data(
                n_steps=n_steps, exists_ok=monkey_data_exists_ok, save_data=save_data, **env_kwargs)
            self._load_agent_data_onto_pf()
            self.pf.make_plan_x_and_y_for_both_test_and_ctrl(**kwargs)
        self._copy_df_from_pn_to_self()

    def get_test_and_ctrl_heading_info_df_for_one_session(self, ref_point_mode='distance', ref_point_value=-150,
                                                          curv_traj_window_before_stop=[
                                                              -50, 0],
                                                          monkey_data_exists_ok=True,
                                                          heading_info_df_exists_ok=False,
                                                          stops_near_ff_df_exists_ok=False,
                                                          use_curvature_to_ff_center=False,
                                                          save_data=True,
                                                          curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 25],
                                                          n_steps=8000,
                                                          merge_diff_in_curv_df_to_heading_info=True,
                                                          **env_kwargs):

        self._initialize_pf(curv_of_traj_mode=curv_of_traj_mode,
                            window_for_curv_of_traj=window_for_curv_of_traj)

        try:  # needs agent data
            for test_or_control in ['test', 'control']:
                heading_info_df, diff_in_curv_df = self.pf._retrieve_heading_info_df(ref_point_mode, ref_point_value, test_or_control,
                                                                                     curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                     merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)
                test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'
                setattr(self, f'{test_or_ctrl}_heading_info_df',
                        heading_info_df)
        except Exception as e:
            print('Data missing. Will get agent data first. Error message: ', e)
            self.get_agent_data(
                n_steps=n_steps, exists_ok=monkey_data_exists_ok, save_data=save_data, **env_kwargs)
            self._load_agent_data_onto_pf()
            for test_or_control in ['test', 'control']:
                self.pf.make_heading_info_df_without_long_process(test_or_control=test_or_control, ref_point_mode=ref_point_mode,
                                                                  curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                  ref_point_value=ref_point_value, use_curvature_to_ff_center=use_curvature_to_ff_center,
                                                                  heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                                  save_data=save_data,
                                                                  merge_diff_in_curv_df_to_heading_info=merge_diff_in_curv_df_to_heading_info)
                heading_info_df = self.pf.heading_info_df.copy()
                diff_in_curv_df = self.pf.diff_in_curv_df.copy()
                test_or_ctrl = 'test' if test_or_control == 'test' else 'ctrl'
                setattr(self, f'{test_or_ctrl}_heading_info_df',
                        heading_info_df)
                setattr(self, f'{test_or_ctrl}_diff_in_curv_df',
                        diff_in_curv_df)
