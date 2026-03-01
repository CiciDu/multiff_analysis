
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class
from planning_analysis.agent_analysis import agent_plan_factors_class
from planning_analysis.factors_vs_indicators import variations_base_class
from planning_analysis.factors_vs_indicators import make_variations_utils
from reinforcement_learning.base_classes import rl_base_class
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils
from reinforcement_learning.base_classes import rl_base_utils

import pandas as pd
import os
import warnings
from os.path import exists


class PlanFactorsAcrossAgentSessions(variations_base_class._VariationsBase):

    def __init__(self,
                 model_folder_name,
                 # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 opt_arc_type='opt_arc_stop_closest',
                 # note, currently we use 900s / dt = 9000 steps (15 mins)
                 backend='matplotlib',
                 ):

        super().__init__(opt_arc_type=opt_arc_type, backend=backend)
        self.model_folder_name = model_folder_name
        self.opt_arc_type = opt_arc_type
        rl_base_class._RLforMultifirefly.get_related_folder_names_from_model_folder_name(
            self, self.model_folder_name)
        self.monkey_name = None

        self.combd_planning_info_folder_path = os.path.join(model_folder_name.replace(
            'all_agents', 'all_collected_data/planning'), 'combined_data')
        self.combd_cur_and_nxt_folder_path = os.path.join(
            self.combd_planning_info_folder_path, 'cur_and_nxt')
        self.make_key_paths()
        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)
        self.default_ref_point_params_based_on_mode = monkey_plan_factors_x_sess_class.PlanAcrossSessions.default_ref_point_params_based_on_mode

        self.num_obs_ff, self.max_in_memory_time = compare_monkey_and_agent_utils.extract_ff_num(
            model_folder_name)
        if self.num_obs_ff == 1:
            self.test_or_control_filter = 'control'
        else:
            self.test_or_control_filter = None

    def streamline_getting_y_values(self,
                                    num_datasets_to_collect=2,
                                    num_steps_per_dataset=9000,
                                    ref_point_mode='distance',
                                    ref_point_value=-100,
                                    save_data=True,
                                    final_products_exist_ok=True,
                                    intermediate_products_exist_ok=True,
                                    agent_data_exists_ok=True,
                                    model_folder_name=None,
                                    ref_point_params_based_on_mode=None,
                                    use_stored_data_only=False,
                                    **env_kwargs
                                    ):

        save_data = False if use_stored_data_only else save_data
        self.num_steps_per_dataset = num_steps_per_dataset

        if use_stored_data_only:
            # first check if any data exists at all
            planning_data_folder_path = rl_base_utils.build_path(model_folder_name, 'planning')
            if not os.path.exists(planning_data_folder_path) or not os.listdir(planning_data_folder_path):
                msg = (f'Planning data folder does not exist or is empty for model_folder_name. '
                        f'Since use_stored_data_only is True, skipping data collection.')
                raise Exception(msg)

        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode

        if model_folder_name is None:
            model_folder_name = self.model_folder_name
        if not use_stored_data_only:
            # make sure there's enough data from agent
            for i in range(num_datasets_to_collect):
                data_name = f'data_{i}'
                print(' ')
                print('model_folder_name:', model_folder_name)
                print('data_name:', data_name)
                # make a new instance of PlanFactorsOfAgent for each dataset for convenience
                self.pfa = agent_plan_factors_class.PlanFactorsOfAgent(model_folder_name=model_folder_name,
                                                                       data_name=data_name,
                                                                       opt_arc_type=self.opt_arc_type,
                                                                       )

                # check to see if data exists:
                if agent_data_exists_ok & exists(os.path.join(self.pfa.processed_data_folder_path, 'monkey_information.csv')):
                    print('Data exists for this agent.')
                    continue
                print('Getting agent data ......')
                env_kwargs['print_ff_capture_incidents'] = False

                self.pfa.get_agent_data(**env_kwargs, exists_ok=agent_data_exists_ok,
                                        save_data=save_data, n_steps=self.num_steps_per_dataset)

        try:
            print(' ')
            print('Making overall all median info ......')

            self.make_or_retrieve_all_ref_pooled_median_info(ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                             list_of_curv_traj_window_before_stop=[
                                                                 [-25, 0]],
                                                             save_data=save_data,
                                                             exists_ok=final_products_exist_ok,
                                                             pooled_median_info_exists_ok=intermediate_products_exist_ok,
                                                             combd_heading_df_x_sessions_exists_ok=intermediate_products_exist_ok,
                                                             stops_near_ff_df_exists_ok=intermediate_products_exist_ok,
                                                             heading_info_df_exists_ok=intermediate_products_exist_ok,
                                                             use_stored_data_only=use_stored_data_only,
                                                             num_datasets_to_collect=num_datasets_to_collect)

            self.agent_all_ref_pooled_median_info = self.all_ref_pooled_median_info.copy()

            print(' ')
            print('Making all perc info ......')
            self.make_or_retrieve_pooled_perc_info(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                   verbose=True,
                                                   exists_ok=final_products_exist_ok,
                                                   stops_near_ff_df_exists_ok=intermediate_products_exist_ok,
                                                   heading_info_df_exists_ok=intermediate_products_exist_ok,
                                                   save_data=save_data,
                                                   use_stored_data_only=use_stored_data_only,
                                                   num_datasets_to_collect=num_datasets_to_collect)
            self.agent_all_perc_df = self.pooled_perc_info.copy()
        except Exception as e:
            print(f'Error making overall all median info: {e}')
            raise Exception(
                f'Error making either overall all median info or perc info: {e}')

    def get_plan_features_df_across_sessions(self,
                                             num_datasets_to_collect=1,
                                             ref_point_mode='distance',
                                             ref_point_value=-150,
                                             curv_traj_window_before_stop=[
                                                 -25, 0],
                                             exists_ok=True,
                                             plan_features_exists_ok=True,

                                             heading_info_df_exists_ok=True,
                                             stops_near_ff_df_exists_ok=True,
                                             curv_of_traj_mode='distance',
                                             window_for_curv_of_traj=[-25, 0],
                                             use_curv_to_ff_center=False,
                                             save_data=True,
                                             **env_kwargs
                                             ):
        plan_features_tc_kwargs = dict(num_datasets_to_collect=num_datasets_to_collect,
                                       save_data=save_data,
                                       **env_kwargs)

        monkey_plan_factors_x_sess_class.PlanAcrossSessions.get_plan_features_df_across_sessions(self,
                                                                                                 ref_point_mode=ref_point_mode,
                                                                                                 ref_point_value=ref_point_value,
                                                                                                 curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                 exists_ok=exists_ok,
                                                                                                 plan_features_exists_ok=plan_features_exists_ok,
                                                                                                 heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                                                 stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                                                                 curv_of_traj_mode=curv_of_traj_mode,
                                                                                                 window_for_curv_of_traj=window_for_curv_of_traj,
                                                                                                 use_curv_to_ff_center=use_curv_to_ff_center,
                                                                                                 **plan_features_tc_kwargs)

    def make_combd_plan_features_tc(self,
                                    plan_features_exists_ok=True,
                                    heading_info_df_exists_ok=True,
                                    stops_near_ff_df_exists_ok=True,
                                    num_datasets_to_collect=1,
                                    save_data=True,
                                    **env_kwargs
                                    ):

        self.combd_plan_features_tc = pd.DataFrame()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(num_datasets_to_collect):
                data_name = f'data_{i}'
                print(' ')
                print('model_folder_name:', self.model_folder_name)
                print('data_name:', data_name)
                self.pfa = agent_plan_factors_class.PlanFactorsOfAgent(model_folder_name=self.model_folder_name,
                                                                       data_name=data_name,
                                                                       opt_arc_type=self.opt_arc_type,
                                                                       )
                print(' ')
                print('Getting plan x and plan y data ......')
                self.pfa.get_plan_features_df_for_one_session(ref_point_mode=self.ref_point_mode,
                                                              ref_point_value=self.ref_point_value,
                                                              curv_traj_window_before_stop=self.curv_traj_window_before_stop,
                                                              plan_features_exists_ok=plan_features_exists_ok,
                                                              heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                              stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                              curv_of_traj_mode=self.curv_of_traj_mode,
                                                              window_for_curv_of_traj=self.window_for_curv_of_traj,
                                                              use_curv_to_ff_center=self.use_curv_to_ff_center,
                                                              save_data=save_data,
                                                              n_steps=self.num_steps_per_dataset,
                                                              **env_kwargs)

                self._add_plan_features_to_combd_plan_features(data_name)

    def retrieve_combd_heading_df_x_sessions(self, ref_point_mode='distance', ref_point_value=-150,
                                             curv_traj_window_before_stop=[
                                                 -25, 0],
                                             test_or_control_filter=None):
        df_name_dict = {'control': 'ctrl_heading_info_df',
                        'test': 'test_heading_info_df'}

        # Determine which types to retrieve
        types_to_retrieve = ['control', 'test'] if test_or_control_filter is None else [
            test_or_control_filter]

        # Initialize both dataframes as empty
        self.ctrl_heading_info_df = pd.DataFrame()
        self.test_heading_info_df = pd.DataFrame()

        try:
            for test_or_control in types_to_retrieve:
                try:
                    combd_heading_df_x_sessions = show_planning_class.ShowPlanning.retrieve_combd_heading_df_x_sessions(self, ref_point_mode=ref_point_mode,
                                                                                                                        ref_point_value=ref_point_value,
                                                                                                                        curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                                                                        test_or_control=test_or_control)
                    setattr(self, df_name_dict[test_or_control],
                            combd_heading_df_x_sessions)
                except Exception as e:
                    print(f'Could not retrieve {test_or_control} data: {e}')
                    setattr(
                        self, df_name_dict[test_or_control], pd.DataFrame())
        finally:
            # Always ensure both dataframes have proper column structure, even if retrieval failed
            # Ensure both dataframes have the same columns even if one is empty
            if len(self.test_heading_info_df) > 0 and len(self.ctrl_heading_info_df) == 0:
                self.ctrl_heading_info_df = pd.DataFrame(
                    columns=self.test_heading_info_df.columns)
            elif len(self.ctrl_heading_info_df) > 0 and len(self.test_heading_info_df) == 0:
                self.test_heading_info_df = pd.DataFrame(
                    columns=self.ctrl_heading_info_df.columns)

            # however, if both are empty, then delete the vars to prevent downstream problems
            if len(self.test_heading_info_df) == 0 and len(self.ctrl_heading_info_df) == 0:
                print('No heading_info_df retrieved for either test or control.')

    def make_combd_heading_df_x_sessions(self, num_datasets_to_collect=1,
                                         ref_point_mode='distance', ref_point_value=-150,
                                         curv_traj_window_before_stop=[-25, 0],
                                         heading_info_df_exists_ok=True,
                                         stops_near_ff_df_exists_ok=True,
                                         curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 0],
                                         use_curv_to_ff_center=False,
                                         save_data=True,
                                         use_stored_data_only=False,
                                         **env_kwargs
                                         ):
        self.test_heading_info_df = pd.DataFrame()
        self.ctrl_heading_info_df = pd.DataFrame()

        save_data = False if use_stored_data_only else save_data

        for i in range(num_datasets_to_collect):
            data_name = f'data_{i}'
            print(' ')
            print('model_folder_name:', self.model_folder_name)
            print('data_name:', data_name)
            self.pfa = agent_plan_factors_class.PlanFactorsOfAgent(model_folder_name=self.model_folder_name,
                                                                   data_name=data_name,
                                                                   opt_arc_type=self.opt_arc_type,
                                                                   )
            print(' ')

            # Get heading info for the requested data type(s)
            filter_msg = f'{self.test_or_control_filter} heading info only' if self.test_or_control_filter else 'test and control heading info'
            print(f'Getting {filter_msg} ......')

            try:
                # Exclude test_or_control_filter from env_kwargs - it is planning-specific and
                # causes KeyError if passed to agent/env code
                if self.test_or_control_filter in env_kwargs.keys():
                    env_kwargs.pop(self.test_or_control_filter)
                    print('test_or_control_filter removed from env_kwargs')
                self.pfa.get_test_and_ctrl_heading_info_df_for_one_session(
                    ref_point_mode=ref_point_mode,
                    ref_point_value=ref_point_value,
                    curv_traj_window_before_stop=curv_traj_window_before_stop,
                    heading_info_df_exists_ok=heading_info_df_exists_ok,
                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                    curv_of_traj_mode=curv_of_traj_mode,
                    window_for_curv_of_traj=window_for_curv_of_traj,
                    use_curv_to_ff_center=use_curv_to_ff_center,
                    save_data=save_data,
                    n_steps=self.num_steps_per_dataset,
                    use_stored_data_only=use_stored_data_only,
                    test_or_control_filter=self.test_or_control_filter,
                    **env_kwargs)
                self._add_heading_info_to_combd_heading_info(
                    data_name)
            except Exception as e:
                print(f'Warning: Could not get heading info for data_{i}: {e}')
                if self.test_or_control_filter:
                    print(
                        f'{self.test_or_control_filter.capitalize()} data might not exist for this session. Skipping.')
                else:
                    print('Data might not exist for this session. Skipping.')

        # Ensure both dataframes have the same columns even if one is empty
        # This prevents KeyErrors in downstream processing
        if len(self.test_heading_info_df) > 0 and len(self.ctrl_heading_info_df) == 0:
            self.ctrl_heading_info_df = pd.DataFrame(
                columns=self.test_heading_info_df.columns)
        elif len(self.ctrl_heading_info_df) > 0 and len(self.test_heading_info_df) == 0:
            self.test_heading_info_df = pd.DataFrame(
                columns=self.ctrl_heading_info_df.columns)

        self.test_heading_info_df.reset_index(drop=True, inplace=True)
        self.ctrl_heading_info_df.reset_index(drop=True, inplace=True)

        if save_data:
            # Determine which types to save
            types_to_save = ['test', 'control'] if self.test_or_control_filter is None else [
                self.test_or_control_filter]

            for test_or_control in types_to_save:
                df_to_save = self.test_heading_info_df if test_or_control == 'test' else self.ctrl_heading_info_df

                # Only save if the dataframe is not empty
                if len(df_to_save) > 0:
                    path = self.dict_of_combd_heading_info_folder_path[test_or_control]
                    df_name = find_cvn_utils.get_df_name_by_ref(
                        'monkey_agent', ref_point_mode, ref_point_value)
                    df_path = os.path.join(path, df_name)
                    os.makedirs(path, exist_ok=True)
                    df_to_save.to_csv(df_path)
                    print(
                        f'Stored new combd_heading_df_x_sessions for {test_or_control} data in {df_path}')
                else:
                    print(
                        f'Skipping save for {test_or_control} data - dataframe is empty')

    def get_test_and_ctrl_heading_info_df_across_sessions(self,
                                                          num_datasets_to_collect=1,
                                                          num_steps_per_dataset=9000,
                                                          ref_point_mode='distance', ref_point_value=-150,
                                                          curv_traj_window_before_stop=[
                                                              -25, 0],
                                                          heading_info_df_exists_ok=True,
                                                          combd_heading_df_x_sessions_exists_ok=True,
                                                          stops_near_ff_df_exists_ok=True,
                                                          curv_of_traj_mode='distance', window_for_curv_of_traj=[-25, 0],
                                                          use_curv_to_ff_center=False,
                                                          save_data=True,
                                                          use_stored_data_only=False,
                                                          **env_kwargs
                                                          ):

        try:
            if combd_heading_df_x_sessions_exists_ok:
                print(
                    f'Attempting to retrieve existing combd_heading_df_x_sessions (filter: {self.test_or_control_filter})...')
                self.retrieve_combd_heading_df_x_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                          curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                          test_or_control_filter=self.test_or_control_filter)

                # Only check for empty dataframes if not filtering, or check only the filtered type
                if self.test_or_control_filter is None:
                    if (len(self.ctrl_heading_info_df) == 0) or (len(self.test_heading_info_df) == 0):
                        raise Exception('Empty combd_heading_df_x_sessions.')
                elif self.test_or_control_filter == 'control':
                    if len(self.ctrl_heading_info_df) == 0:
                        raise Exception(
                            'Empty combd_heading_df_x_sessions for control data.')
                elif self.test_or_control_filter == 'test':
                    if len(self.test_heading_info_df) == 0:
                        raise Exception(
                            'Empty combd_heading_df_x_sessions for test data.')
                print(
                    f'Retrieved: test={len(self.test_heading_info_df)} rows, ctrl={len(self.ctrl_heading_info_df)} rows')

            else:
                raise Exception(
                    'combd_heading_df_x_sessions_exists_ok is False.')

        except Exception as e:
            self.num_steps_per_dataset = num_steps_per_dataset
            print(
                f'Will make new combd_heading_df_x_sessions for the agent because: {e}')
            print(
                f'Creating data from scratch with filter: {self.test_or_control_filter}')

            self.make_combd_heading_df_x_sessions(num_steps_per_dataset=self.num_steps_per_dataset, num_datasets_to_collect=num_datasets_to_collect,
                                                  ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                  curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                  heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                  stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                  curv_of_traj_mode=curv_of_traj_mode, window_for_curv_of_traj=window_for_curv_of_traj,
                                                  use_curv_to_ff_center=use_curv_to_ff_center,
                                                  save_data=save_data,
                                                  use_stored_data_only=use_stored_data_only,
                                                  **env_kwargs)

            print(
                f'After creation: test={len(self.test_heading_info_df)} rows, ctrl={len(self.ctrl_heading_info_df)} rows')

    def _add_plan_features_to_combd_plan_features(self, data_name):
        plan_features_tc = self.pfa.plan_features_tc.copy()
        plan_features_tc['data_name'] = data_name
        self.combd_plan_features_tc = pd.concat(
            [self.combd_plan_features_tc, plan_features_tc], axis=0)

    def _add_heading_info_to_combd_heading_info(self, data_name):
        # Get dataframes from pfa, handling cases where they might not exist
        test_df = self.pfa.test_heading_info_df.copy() if hasattr(
            self.pfa, 'test_heading_info_df') else pd.DataFrame()
        ctrl_df = self.pfa.ctrl_heading_info_df.copy() if hasattr(
            self.pfa, 'ctrl_heading_info_df') else pd.DataFrame()

        print(f'_add_heading_info_to_combd_heading_info for {data_name}:')
        print(
            f'  From pfa: test={len(test_df)} rows, ctrl={len(ctrl_df)} rows')
        print(f'  Filter: {self.test_or_control_filter}')

        # Only process the requested type if filter is specified
        if self.test_or_control_filter == 'test':
            if len(test_df) > 0:
                test_df['data_name'] = data_name
                test_df['whether_test'] = 1
                self.test_heading_info_df = pd.concat(
                    [self.test_heading_info_df, test_df], axis=0)
                print(f'  Added {len(test_df)} test rows')
            else:
                print('  No test data to add')
        elif self.test_or_control_filter == 'control':
            if len(ctrl_df) > 0:
                ctrl_df['data_name'] = data_name
                ctrl_df['whether_test'] = 0
                self.ctrl_heading_info_df = pd.concat(
                    [self.ctrl_heading_info_df, ctrl_df], axis=0)
                print(f'  Added {len(ctrl_df)} control rows')
            else:
                print('  No control data to add (this may be the problem!)')
        else:
            # No filter - process both
            if len(test_df) > 0:
                test_df['data_name'] = data_name
                test_df['whether_test'] = 1
                self.test_heading_info_df = pd.concat(
                    [self.test_heading_info_df, test_df], axis=0)
                print(f'  Added {len(test_df)} test rows')

            if len(ctrl_df) > 0:
                ctrl_df['data_name'] = data_name
                ctrl_df['whether_test'] = 0
                self.ctrl_heading_info_df = pd.concat(
                    [self.ctrl_heading_info_df, ctrl_df], axis=0)
                print(f'  Added {len(ctrl_df)} control rows')

        # not sure what the following is for
        # self.test_heading_info_df = pd.concat(
        #     [self.test_heading_info_df, self.test_heading_info_df], axis=0)
        # self.ctrl_heading_info_df = pd.concat(
        #     [self.ctrl_heading_info_df, self.ctrl_heading_info_df], axis=0)

    def get_test_and_ctrl_heading_info_df_across_sessions2(self,
                                                           ref_point_mode='distance',
                                                           ref_point_value=-150,
                                                           curv_traj_window_before_stop=[
                                                               -25, 0],
                                                           heading_info_df_exists_ok=True,
                                                           combd_heading_df_x_sessions_exists_ok=True,
                                                           stops_near_ff_df_exists_ok=True,
                                                           save_data=True,
                                                           filter_heading_info_df_across_refs=False,
                                                           use_stored_data_only=False,
                                                           **kwargs):
        """
        Override parent method to support test_or_control_filter for agents.
        This prevents errors when only one type of data exists.
        """

        save_data = False if use_stored_data_only else save_data

        # Call the agent-specific version with filter support
        self.get_test_and_ctrl_heading_info_df_across_sessions(
            num_datasets_to_collect=self.num_datasets_to_collect,
            ref_point_mode=ref_point_mode,
            ref_point_value=ref_point_value,
            curv_traj_window_before_stop=curv_traj_window_before_stop,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            save_data=save_data,
            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
            use_stored_data_only=use_stored_data_only,
            **kwargs
        )

        # Double-check column structure before returning (safety net)
        if len(self.test_heading_info_df) > 0 and len(self.ctrl_heading_info_df) == 0:
            self.ctrl_heading_info_df = pd.DataFrame(
                columns=self.test_heading_info_df.columns)
        elif len(self.ctrl_heading_info_df) > 0 and len(self.test_heading_info_df) == 0:
            self.test_heading_info_df = pd.DataFrame(
                columns=self.ctrl_heading_info_df.columns)

    def make_or_retrieve_all_ref_pooled_median_info(self, 
                                                    num_datasets_to_collect=1,
                                                    **kwargs):
        """
        Make or retrieve pooled median info across all reference points.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to parent method.

        Returns
        -------
        pd.DataFrame
            All reference pooled median info, optionally filtered by test_or_control.
        """

        self.num_datasets_to_collect = num_datasets_to_collect

        try:
            # Enable plotting processing - it's now robust to handle missing columns
            if 'process_info_for_plotting' not in kwargs:
                kwargs['process_info_for_plotting'] = True
                
            kwargs['exists_ok'] = False # to make sure to use the right num_datasets_to_collect
            kwargs['pooled_median_info_exists_ok'] = False # to make sure to use the right num_datasets_to_collect

            self.all_ref_pooled_median_info = super(
            ).make_or_retrieve_all_ref_pooled_median_info(
                **kwargs)
            self.all_ref_pooled_median_info['monkey_name'] = 'agent'

            # Filter by test_or_control if specified
            # Only filter if the column exists (it might not exist if data was already filtered at retrieval)
            if self.test_or_control_filter is not None:
                if 'test_or_control' in self.all_ref_pooled_median_info.columns:
                    self.all_ref_pooled_median_info = self.all_ref_pooled_median_info[
                        self.all_ref_pooled_median_info['test_or_control'] == self.test_or_control_filter
                    ].copy()
                    print(
                        f'Filtered to only {self.test_or_control_filter} data. Shape: {self.all_ref_pooled_median_info.shape}')
                else:
                    print(
                        'Note: test_or_control column not found. Data was already filtered at retrieval level.')
        finally:
            pass

        return self.all_ref_pooled_median_info

    def make_or_retrieve_pooled_perc_info(self, use_stored_data_only=False,
                                          exists_ok=True, stops_near_ff_df_exists_ok=True,
                                          heading_info_df_exists_ok=True, ref_point_mode='distance',
                                          ref_point_value=-50, verbose=False, save_data=True,
                                          filter_heading_info_df_across_refs=False, 
                                          num_datasets_to_collect=1,
                                          **kwargs):
        """
        Make or retrieve pooled percentage info with single-condition support.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            Pooled percentage info.
        """

        save_data = False if use_stored_data_only else save_data

        self.num_datasets_to_collect = num_datasets_to_collect

        try:
            if exists_ok & exists(self.pooled_perc_info_path):
                self.pooled_perc_info = pd.read_csv(self.pooled_perc_info_path).drop(
                    ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
            else:
                self.get_test_and_ctrl_heading_info_df_across_sessions2(
                    ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                    heading_info_df_exists_ok=heading_info_df_exists_ok,
                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                    filter_heading_info_df_across_refs=filter_heading_info_df_across_refs,
                    use_stored_data_only=use_stored_data_only)

                # Enable single condition support when filtering
                allow_single = self.test_or_control_filter is not None

                self.pooled_perc_info = make_variations_utils.make_pooled_perc_info_from_test_and_ctrl_heading_info_df(
                    self.test_heading_info_df,
                    self.ctrl_heading_info_df,
                    verbose=verbose,
                    allow_single_condition=allow_single,
                    )

                if save_data:
                    self.pooled_perc_info.to_csv(self.pooled_perc_info_path)
                    print('Stored new pooled_perc_info in ',
                        self.pooled_perc_info_path)

            self.pooled_perc_info['monkey_name'] = 'agent'
            self.pooled_perc_info['opt_arc_type'] = self.opt_arc_type
        finally:
            # Clean up instance variable
            pass

        return self.pooled_perc_info
