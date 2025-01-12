
from planning_analysis.plan_factors import test_vs_control_utils, test_vs_control_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, process_variations_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_utils
from planning_analysis import ml_methods_utils, ml_methods_class
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class _VariationsBase():

    x_columns = ['time_when_stop_ff_last_seen_rel_to_stop',
                'left_eye_stop_ff_time_perc',
                'right_eye_stop_ff_time_perc',
                'left_eye_stop_ff_time_perc_10',
                'right_eye_stop_ff_time_perc_10',
                'LDy_25%',
                'LDy_50%',
                'LDy_75%',
                'LDz_25%',
                'LDz_50%',
                'LDz_75%',
                'RDy_25%',
                'RDy_50%',
                'RDy_75%',
                'RDz_25%',
                'monkey_speed_25%',
                'monkey_speed_50%',
                'monkey_speed_75%',
                'monkey_dw_25%',
                'monkey_dw_50%',
                'monkey_dw_75%',
                # 'stop_ff_angle_when_stop_ff_last_seen',
                # 'stop_ff_distance_when_stop_ff_last_seen',
                # 'traj_curv_when_stop_ff_last_seen',
                    ]

    stop_ff_cluster_columns = ['stop_ff_cluster_100_size',
                                'stop_ff_cluster_100_EARLIEST_APPEAR_ff_angle',
                                'stop_ff_cluster_100_EARLIEST_APPEAR_latest_vis_time',
                                'stop_ff_cluster_100_EARLIEST_APPEAR_visible_duration_after_stop',
                                'stop_ff_cluster_100_EARLIEST_APPEAR_visible_duration_before_stop',
                                'stop_ff_cluster_100_LAST_DISP_earliest_vis_time',
                                'stop_ff_cluster_100_LAST_DISP_ff_angle',
                                'stop_ff_cluster_100_LAST_DISP_visible_duration_after_stop',
                                'stop_ff_cluster_100_LAST_DISP_visible_duration_before_stop',
                                'stop_ff_cluster_100_LONGEST_VIS_earliest_vis_time',
                                'stop_ff_cluster_100_LONGEST_VIS_ff_angle',
                                'stop_ff_cluster_100_LONGEST_VIS_latest_vis_time',
                                'stop_ff_cluster_100_LONGEST_VIS_visible_duration_after_stop',
                                'stop_ff_cluster_100_LONGEST_VIS_visible_duration_before_stop',
                                'stop_ff_cluster_100_combd_min_ff_angle',
                                'stop_ff_cluster_100_combd_max_ff_angle',
                                'stop_ff_cluster_100_combd_median_ff_angle',
                                'stop_ff_cluster_100_combd_median_ff_distance',
                                'stop_ff_cluster_100_combd_earliest_vis_time',
                                'stop_ff_cluster_100_combd_latest_vis_time',
                                'stop_ff_cluster_100_combd_visible_duration',
                                'stop_ff_cluster_100_combd_earliest_vis_time_after_stop',
                                'stop_ff_cluster_100_combd_latest_vis_time_before_stop',
                                #'stop_ff_cluster_100_EARLIEST_APPEAR_earliest_vis_time',
                                #'stop_ff_cluster_100_LAST_DISP_latest_vis_time',                            
                                ]

    curv_columns = ['ref_curv_of_traj',
                    'curv_mean',
                    'curv_std',
                    'curv_min',
                    'curv_25%',
                    'curv_50%',
                    'curv_75%',
                    'curv_max']
    
    
    def __init__(self,
                 optimal_arc_type='norm_opt_arc', # options are: norm_opt_arc, opt_arc_stop_first_vis_bdry, opt_arc_stop_closest,
                 ): 

        self.optimal_arc_type = optimal_arc_type

    def make_key_paths(self):
        self.stop_and_alt_data_comparison_path = os.path.join(self.combd_stop_and_alt_folder_path, 'data_comparison')
        self.all_perc_info_path = os.path.join(self.stop_and_alt_data_comparison_path, f'{self.optimal_arc_type}/all_perc_info.csv')
        self.all_median_info_folder_path = os.path.join(self.stop_and_alt_data_comparison_path, f'{self.optimal_arc_type}/all_median_info')
        self.overall_median_info_path = os.path.join(self.stop_and_alt_data_comparison_path, f'{self.optimal_arc_type}/overall_median_info.csv')
        show_planning_class.ShowPlanning.get_combd_info_folder_paths(self)

        self.stop_and_alt_lr_df_path = os.path.join(self.combd_stop_and_alt_folder_path, f'ml_results/lr_variations/{self.optimal_arc_type}/all_stop_and_alt_lr_df.csv')
        self.stop_and_alt_lr_pred_ff_df_path = os.path.join(self.combd_stop_and_alt_folder_path, f'ml_results/lr_variations/{self.optimal_arc_type}/all_stop_and_alt_lr_pred_ff_df.csv')
        os.makedirs(os.path.dirname(self.stop_and_alt_lr_df_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.stop_and_alt_lr_pred_ff_df_path), exist_ok=True)


    # note that the method below is only used for monkey; for agent, the method is defined in the agent class
    def get_test_and_ctrl_heading_info_df_across_sessions(self, ref_point_mode='distance', ref_point_value=-150, 
                                                          curv_traj_window_before_stop=[-50, 0],
                                                          heading_info_df_exists_ok=True, combd_heading_df_x_sessions_exists_ok=True, stops_near_ff_df_exists_ok=True, save_data=True):
        self.sp = show_planning_class.ShowPlanning(monkey_name=self.monkey_name,
                                                   optimal_arc_type=self.optimal_arc_type)
        self.test_heading_info_df, self.ctrl_heading_info_df = self.sp.make_or_retrieve_combd_heading_df_x_sessions_from_both_test_and_control(ref_point_mode, ref_point_value,
                                                                            curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                                            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
                                                                            show_printed_output=True, heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                                            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data)
        
    def make_or_retrieve_overall_median_info(self, 
                                            exists_ok=True, 
                                            all_median_info_exists_ok=True, 
                                            ref_point_params_based_on_mode=None, 
                                            list_of_curv_traj_window_before_stop=[[-50, 0]],
                                            save_data=True, 
                                            combd_heading_df_x_sessions_exists_ok=True, 
                                            stops_near_ff_df_exists_ok=True, 
                                            heading_info_df_exists_ok=True,
                                            process_info_for_plotting=True):

        if exists_ok & exists(self.overall_median_info_path):
            self.overall_median_info = pd.read_csv(self.overall_median_info_path).drop(columns=['Unnamed: 0'])
            print('Successfully retrieved overall_median_info from ', self.overall_median_info_path)
        else:
            if ref_point_params_based_on_mode is None:
                ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
            
            self.overall_median_info = pd.DataFrame([])
            for curv_traj_window_before_stop in list_of_curv_traj_window_before_stop:
                temp_overall_median_info = make_variations_utils.make_variations_df_across_ref_point_values(self.make_all_median_info,
                                                                            ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                            monkey_name=self.monkey_name,
                                                                            variation_func_kwargs={'all_median_info_exists_ok': all_median_info_exists_ok,
                                                                                                   'curv_traj_window_before_stop': curv_traj_window_before_stop,
                                                                                                'save_data': save_data,
                                                                                                'combd_heading_df_x_sessions_exists_ok': combd_heading_df_x_sessions_exists_ok,
                                                                                                'stops_near_ff_df_exists_ok': stops_near_ff_df_exists_ok,
                                                                                                'heading_info_df_exists_ok': heading_info_df_exists_ok,
                                                                                                },
                                                                            path_to_save=None,
                                                                            )
                temp_overall_median_info['curv_traj_window_before_stop'] = str(curv_traj_window_before_stop)
                self.overall_median_info = pd.concat([self.overall_median_info, temp_overall_median_info], axis=0)

        self.overall_median_info.reset_index(drop=True, inplace=True)    
        self.overall_median_info['monkey_name'] = self.monkey_name
        self.overall_median_info['optimal_arc_type'] = self.optimal_arc_type
        self.overall_median_info.to_csv(self.overall_median_info_path)
        print(f'Saved overall_median_info_path to {self.overall_median_info_path}')
        if process_info_for_plotting:
            self.process_overall_median_info_to_plot_heading_and_curv()

        return self.overall_median_info
    

    def make_or_retrieve_all_perc_info(self, exists_ok=True, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                       ref_point_mode='distance', ref_point_value=-50, verbose=False, save_data=True, process_info_for_plotting=True):
        # These two parameters (ref_point_mode, ref_point_value) are actually not important here as long as the corresponding data can be successfully retrieved,
        # since the results are the same regardless
        
        if exists_ok & exists(self.all_perc_info_path):
            self.all_perc_info = pd.read_csv(self.all_perc_info_path)
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                                                   heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok)
            self.all_perc_info = make_variations_utils.make_all_perc_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df, 
                                                            self.ctrl_heading_info_df, verbose=verbose)
            
            if save_data:
                self.all_perc_info.to_csv(self.all_perc_info_path)
            print('Stored new all_perc_info in ', self.all_perc_info_path)

        self.all_perc_info['monkey_name'] = self.monkey_name
        self.all_perc_info['optimal_arc_type'] = self.optimal_arc_type

        if process_info_for_plotting:
            self.process_all_perc_info_to_plot_direction()

        return self.all_perc_info


    def process_overall_median_info_to_plot_heading_and_curv(self):
        self.all_median_info_heading  = process_variations_utils.make_new_df_for_plotly_comparison(self.overall_median_info)
        self.all_median_info_curv = self.all_median_info_heading.copy()
        self.all_median_info_curv['sample_size'] = self.all_median_info_curv['sample_size_for_curv']


    def process_all_perc_info_to_plot_direction(self):
        self.all_perc_info_new = process_variations_utils.make_new_df_for_plotly_comparison(self.all_perc_info, 
                                                                 match_rows_based_on_ref_columns_only=False)


    def make_or_retrieve_all_stop_and_alt_lr_df(self, ref_point_params_based_on_mode=None, exists_ok=True):   
        df_path = self.stop_and_alt_lr_df_path
        if exists_ok:
            if exists(df_path):
                self.all_stop_and_alt_lr_df = pd.read_csv(df_path)
                print('Successfully retrieved all_stop_and_alt_lr_df from ', df_path)
                return self.all_stop_and_alt_lr_df
            else:
                print('all_stop_and_alt_lr_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_stop_and_alt_lr_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_stop_and_alt_lr_df,
                                                                    ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                    monkey_name=self.monkey_name,
                                                                    path_to_save=df_path,
                                                                    )
        return self.all_stop_and_alt_lr_df
    

    def make_or_retrieve_all_stop_and_alt_lr_pred_ff_df(self, ref_point_params_based_on_mode=None, exists_ok=True):
        df_path = self.stop_and_alt_lr_pred_ff_df_path
        if exists_ok:
            if exists(df_path):
                self.all_stop_and_alt_lr_pred_ff_df = pd.read_csv(df_path)
                print('Successfully retrieved all_stop_and_alt_lr_pred_ff_df from ', df_path)
                return self.all_stop_and_alt_lr_pred_ff_df
            else:
                print('all_stop_and_alt_lr_pred_ff_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_stop_and_alt_lr_pred_ff_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_stop_and_alt_lr_df,
                                                                    ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                    variation_func_kwargs={'to_predict_ff': True},
                                                                    monkey_name=self.monkey_name,
                                                                    path_to_save=df_path,
                                                                    )
        return self.all_stop_and_alt_lr_pred_ff_df

    def make_or_retrieve_all_stop_and_alt_clf_df(self, ref_point_params_based_on_mode=None, exists_ok=True, ):
        df_path = os.path.join(self.combd_stop_and_alt_folder_path, 'ml_results/clf_variations/all_stop_and_alt_clf_df')
        if exists_ok:
            if exists(df_path):
                self.all_stop_and_alt_clf_df = pd.read_csv(df_path)
                print('Successfully retrieved all_stop_and_alt_clf_df from ', df_path)
                return self.all_stop_and_alt_clf_df
            else:
                print('all_stop_and_alt_clf_df does not exist. Will recreate it.')
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        self.all_stop_and_alt_clf_df = make_variations_utils.make_variations_df_across_ref_point_values(self.make_stop_and_alt_clf_df,
                                                                    ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                    monkey_name=self.monkey_name,
                                                                    path_to_save=df_path,
                                                                    )
        return self.all_stop_and_alt_clf_df


    def make_all_median_info(self, ref_point_mode='time after stop ff visible', 
                             ref_point_value=0.1, 
                             curv_traj_window_before_stop=[-50, 0],
                             all_median_info_exists_ok=True, 
                             combd_heading_df_x_sessions_exists_ok=True, 
                             stops_near_ff_df_exists_ok=True, 
                             heading_info_df_exists_ok=True, 
                             verbose=False, save_data=True):

        df_name = find_stops_near_ff_utils.find_diff_in_curv_df_name(ref_point_mode, ref_point_value, curv_traj_window_before_stop)
        df_path = os.path.join(self.all_median_info_folder_path, df_name)
        if all_median_info_exists_ok & exists(df_path):
            self.all_median_info = pd.read_csv(df_path)
            print('Successfully retrieved all_median_info from ', df_path)
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value, 
                                                                   curv_traj_window_before_stop=curv_traj_window_before_stop,
                                                    heading_info_df_exists_ok=heading_info_df_exists_ok,
                                                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data,
                                                    combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok)
            self.all_median_info = make_variations_utils.make_all_median_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df, 
                                                            self.ctrl_heading_info_df, verbose=verbose)
            self.all_median_info['ref_point_mode'] = ref_point_mode
            self.all_median_info['ref_point_value'] = ref_point_value
            time_calibration = {'ref_point_mode': ref_point_mode, 'ref_point_value': ref_point_value, 'monkey_name': self.monkey_name}
            self.all_median_info.attrs.update(time_calibration)
            os.makedirs(self.all_median_info_folder_path, exist_ok=True)
            self.all_median_info.to_csv(df_path)
            print('Stored new all_median_info in ', self.all_median_info_folder_path)
        return self.all_median_info



    def make_stop_and_alt_lr_df(self, ref_point_mode, ref_point_value, to_predict_ff=False,
                                keep_monkey_info_choices = [True],
                                key_for_split_choices=['ff_seen', 'cluster_seen'],
                                whether_filter_info_choices=[True],
                                whether_even_out_distribution_choices=[True, False],
                                whether_test_alt_ff_flash_after_stop_choices=['flexible'],
                                whether_limit_stop_ff_cluster_50_size_choices=[False],
                                ctrl_flash_compared_to_test_choices=['flexible'],
                                max_curv_range_choices=[200]):
        
        print('to_predict_ff:', to_predict_ff)
        use_lr_func = self.use_lr_to_predict_monkey_info if not to_predict_ff else self.use_lr_to_predict_ff_info
        
        make_regrouped_info_kwargs = dict(key_for_split_choices=key_for_split_choices,
                                        whether_filter_info_choices=whether_filter_info_choices,
                                        whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                        whether_test_alt_ff_flash_after_stop_choices=whether_test_alt_ff_flash_after_stop_choices,
                                        whether_limit_stop_ff_cluster_50_size_choices=whether_limit_stop_ff_cluster_50_size_choices,
                                        ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                        max_curv_range_choices=max_curv_range_choices)

        self.stop_and_alt_lr_df = self._make_stop_and_alt_variations_df(ref_point_mode, 
                                                                        ref_point_value, 
                                                                        use_lr_func,
                                                                        to_predict_ff=to_predict_ff,
                                                                        keep_monkey_info_choices=keep_monkey_info_choices,
                                                                        make_regrouped_info_kwargs=make_regrouped_info_kwargs)
                                                                        
        
        return self.stop_and_alt_lr_df
    

    def make_stop_and_alt_clf_df(self, 
                                 ref_point_mode, 
                                 ref_point_value,
                                keep_monkey_info_choices = [True],
                                key_for_split_choices=['ff_seen', 'cluster_seen'],
                                whether_filter_info_choices=[True],
                                whether_even_out_distribution_choices=[True, False],
                                whether_test_alt_ff_flash_after_stop_choices=['flexible'],
                                whether_limit_stop_ff_cluster_50_size_choices=[False],
                                ctrl_flash_compared_to_test_choices=['flexible'],
                                max_curv_range_choices=[200],
                                agg_regrouped_info_kwargs={}):
        
        use_clf_func = self.use_clf_to_predict_monkey_info 
        
        make_regrouped_info_kwargs = dict(key_for_split_choices=key_for_split_choices,
                                        whether_filter_info_choices=whether_filter_info_choices,
                                        whether_even_out_distribution_choices=whether_even_out_distribution_choices,
                                        whether_test_alt_ff_flash_after_stop_choices=whether_test_alt_ff_flash_after_stop_choices,
                                        whether_limit_stop_ff_cluster_50_size_choices=whether_limit_stop_ff_cluster_50_size_choices,
                                        ctrl_flash_compared_to_test_choices=ctrl_flash_compared_to_test_choices,
                                        max_curv_range_choices=max_curv_range_choices)

        self.stop_and_alt_clf_df = self._make_stop_and_alt_variations_df(ref_point_mode, 
                                                                        ref_point_value, 
                                                                        use_clf_func,
                                                                        agg_regrouped_info_kwargs=agg_regrouped_info_kwargs,
                                                                        keep_monkey_info_choices=keep_monkey_info_choices,
                                                                        make_regrouped_info_kwargs=make_regrouped_info_kwargs,
                                                                        )
                                                                        
        
        return self.stop_and_alt_clf_df

    def quickly_get_plan_x_and_y_control_and_test_data(self, ref_point_mode, ref_point_value, to_predict_ff=False, keep_monkey_info=False, for_classification=False):
        self.get_plan_x_and_plan_y_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)
        self.process_combd_plan_x_tc_and_plan_y_tc()
        self.further_process_combd_plan_x_tc(to_predict_ff, for_classification=for_classification)
        if keep_monkey_info is False:
            self.combd_plan_x_tc = plan_factors_utils.delete_monkey_info_in_plan_x(self.combd_plan_x_tc)
        self.plan_xy_test, self.plan_xy_ctrl = plan_factors_utils.make_plan_xy_test_and_plan_xy_ctrl(self.combd_plan_x_tc, self.combd_plan_y_tc)
        return
    
    def _make_stop_and_alt_variations_df(self, ref_point_mode, ref_point_value, 
                                        agg_regrouped_info_func,
                                        agg_regrouped_info_kwargs={},
                                        to_predict_ff=False,
                                        keep_monkey_info_choices = [True],
                                        make_regrouped_info_kwargs={}):
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
    
        df = pd.DataFrame()
        for keep_monkey_info in keep_monkey_info_choices:
            
            print('keep_monkey_info:', keep_monkey_info)
            self.quickly_get_plan_x_and_y_control_and_test_data(ref_point_mode, ref_point_value, to_predict_ff=to_predict_ff, 
                                                                keep_monkey_info=keep_monkey_info)
            print('Have successfully run get_plan_x_and_plan_y_across_sessions.')

            temp_df = make_variations_utils.make_regrouped_info(self.plan_xy_test, 
                                                                self.plan_xy_ctrl, 
                                                                agg_regrouped_info_func,
                                                                agg_regrouped_info_kwargs=agg_regrouped_info_kwargs,
                                                                **make_regrouped_info_kwargs)
            
            
            temp_df['keep_monkey_info'] = keep_monkey_info
            df = pd.concat([df, temp_df], axis=0)
        df.reset_index(drop=True, inplace=True) 
        return df

    def use_clf_to_predict_monkey_info(self, plan_xy_test, plan_xy_ctrl, **agg_regrouped_info_kwargs):

        method_kwargs = dict(y_columns_of_interest=['dir_from_stop_ff_to_stop'],
                            add_ref_interaction_choices=[True],                      
                            clusters_to_keep_choices=['none', 'stop_ff_cluster_100_PLUS_alt_ff_cluster_100'],
                            clusters_for_interaction_choices=['none', 'stop_ff_cluster_100'],
                            use_combd_features_for_cluster_only_choices=[False],
                            max_features_to_save=None,
                            ref_columns_only_choices=[True, False])
        method_kwargs.update(agg_regrouped_info_kwargs)

        self.ml_inst = ml_methods_class.MlMethods()
        
        regrouped_info = self._use_a_method_on_test_and_ctrl_data_data_respectively(plan_xy_test, plan_xy_ctrl, 
                                     self.ml_inst.try_different_combinations_for_classification,
                                     method_kwargs=method_kwargs,
                                    )
        
        return regrouped_info
        

    def use_lr_to_predict_monkey_info(self, plan_xy_test, plan_xy_ctrl):

        method_kwargs = dict(y_columns_of_interest=[
                                'curvature_of_traj_before_stop',
                                'ref_d_heading_of_traj',
                                'dev_d_angle_from_null',
                                'diff_in_abs',   
                                'dir_from_stop_ff_to_stop', # this one is classification though
                                # 'dir_from_stop_ff_same_side',
                                # 'diff'
                                ], 
                            clusters_for_interaction_choices=[
                                    'stop_ff_cluster_100',
                                    #'alt_ff_cluster_100',

                                    #'stop_ff_cluster_200',
                                    #'alt_ff_cluster_200',
                                    # 'stop_ff_cluster_300',
                                    # 'stop_ff_ang_cluster_20',
                                    ],
                            clusters_to_keep_choices=[
                                                    #'stop_ff_cluster_100_PLUS_stop_ff_cluster_200_PLUS_alt_ff_cluster_100_PLUS_alt_ff_cluster_200',
                                                    #'stop_ff_cluster_100_PLUS_stop_ff_cluster_200_PLUS_alt_ff_cluster_100_PLUS_alt_ff_cluster_300',            
                                                    'stop_ff_cluster_100_PLUS_alt_ff_cluster_100_PLUS_alt_ff_cluster_300',
                                                    #'stop_ff_cluster_100_PLUS_alt_ff_cluster_100',
                                                    ],
                            max_features_to_save=None,
                            use_combd_features_for_cluster_only_choices=[False],
                            )
        self.ml_inst = ml_methods_class.MlMethods()
        
        regrouped_info = self._use_a_method_on_test_and_ctrl_data_data_respectively(plan_xy_test, plan_xy_ctrl, 
                                     self.ml_inst.try_different_combinations_for_linear_regressions,
                                     method_kwargs=method_kwargs,
                                    )
        return regrouped_info


    def use_lr_to_predict_ff_info(self, plan_xy_test, plan_xy_ctrl):
        method_kwargs = dict(y_columns_of_interest=['alt_ff_angle_at_ref'], 
                                    clusters_to_keep_choices=['stop_ff_cluster_100_PLUS_stop_ff_cluster_300'], 
                                    clusters_for_interaction_choices=[], 
                                    max_features_to_save=None,
                                    use_combd_features_for_cluster_only_choices=[False],)

        self.ml_inst = ml_methods_class.MlMethods()

        regrouped_info = self._use_a_method_on_test_and_ctrl_data_data_respectively(plan_xy_test, plan_xy_ctrl, 
                                     self.ml_inst.try_different_combinations_for_linear_regressions,
                                     method_kwargs=method_kwargs,
                                    )    
        return regrouped_info


    def separate_plan_xy_test_and_plan_xy_ctrl(self):
        self.plan_x_test = self.plan_xy_test[self.combd_plan_x_tc.columns].copy()
        self.plan_y_test = self.plan_xy_test[self.combd_plan_y_tc.columns].copy()
        self.plan_x_ctrl = self.plan_xy_ctrl[self.combd_plan_x_tc.columns].copy()
        self.plan_y_ctrl = self.plan_xy_ctrl[self.combd_plan_y_tc.columns].copy()
    

    def process_both_heading_info_df(self):
        self.test_heading_info_df = plan_factors_utils.process_heading_info_df(self.test_heading_info_df)
        self.ctrl_heading_info_df = plan_factors_utils.process_heading_info_df(self.ctrl_heading_info_df)


    def filter_both_heading_info_df(self, **kwargs):
        self.test_heading_info_df, self.ctrl_heading_info_df = test_vs_control_utils.filter_both_df(self.test_heading_info_df, self.ctrl_heading_info_df, **kwargs)

    def process_combd_plan_x_tc_and_plan_y_tc(self):
        test_vs_control_utils.process_combd_plan_x_and_y_combd(self.combd_plan_x_tc, self.combd_plan_y_tc, curv_columns=self.curv_columns)
        self.ref_columns = [column for column in self.combd_plan_x_tc.columns if ('ref' in column) & ('stop_ff' in column)]
        # note that it will include ref_d_heading_of_traj

        # drop columns with NA in self.combd_plan_x_tc and print them
        columns_with_null_info = self.combd_plan_x_tc.isnull().sum(axis=0)[self.combd_plan_x_tc.isnull().sum(axis=0) > 0]
        if len(columns_with_null_info) > 0:
            print('Columns with nulls are dropped:')
            print(columns_with_null_info)
        self.combd_plan_x_tc.dropna(axis=1, inplace=True)

        # Also drop the columns that can't be put into x_var
        for column in ['data_name', 'stop_point_index']:
            if column in self.combd_plan_x_tc.columns:
                self.combd_plan_x_tc.drop(columns=[column], inplace=True)


    def further_process_combd_plan_x_tc(self, to_predict_ff, for_classification=False):
        if to_predict_ff:
            self.combd_plan_x_tc = plan_factors_utils.process_plan_x_to_predict_ff_info(self.combd_plan_x_tc, self.combd_plan_y_tc)
        else:
            self.combd_plan_x_tc = plan_factors_utils.process_plan_x_to_predict_monkey_info(self.combd_plan_x_tc, for_classification=for_classification)


    def streamline_preparing_for_ml(self, 
                                    y_var_column, 
                                    ref_columns_only=False,
                                    cluster_to_keep='all',
                                    cluster_for_interaction='none',
                                    add_ref_interaction=True,
                                    winsorize_angle_features=True, 
                                    using_lasso=True,
                                    use_combd_features_for_cluster_only=False,
                                    for_classification=False):
            
        self.separate_plan_xy_test_and_plan_xy_ctrl()
            
        if self.test_or_control == 'test':
            self.plan_x = self.plan_x_test.copy()
            self.plan_y = self.plan_y_test.copy()
        else:
            self.plan_x = self.plan_x_ctrl.copy()
            self.plan_y = self.plan_y_ctrl.copy()
        
        print('test_or_control:', self.test_or_control)

        self.x_var_df, self.y_var_df = ml_methods_utils.streamline_preparing_for_ml(self.plan_x, 
                                                                                    self.plan_y,
                                                                                    y_var_column,
                                                                                    ref_columns_only=ref_columns_only,
                                                                                    cluster_to_keep=cluster_to_keep,
                                                                                    cluster_for_interaction=cluster_for_interaction, 
                                                                                    add_ref_interaction=add_ref_interaction,
                                                                                    winsorize_angle_features=winsorize_angle_features, 
                                                                                    using_lasso=using_lasso, 
                                                                                    ensure_stop_ff_at_front=False,
                                                                                    use_combd_features_for_cluster_only=use_combd_features_for_cluster_only,
                                                                                    for_classification=for_classification)


    def _use_a_method_on_test_and_ctrl_data_data_respectively(self, 
                                                            plan_xy_test, 
                                                            plan_xy_ctrl, 
                                                            method,
                                                            method_kwargs={}):
        self.plan_xy_test = plan_xy_test.copy()
        self.plan_xy_ctrl = plan_xy_ctrl.copy()
        regrouped_info = pd.DataFrame()
        
        for test_or_control in ['control', 'test']:
            print('test_or_control:', test_or_control)
            self.test_or_control = test_or_control

            df = method(self, **method_kwargs)

            df['test_or_control'] = test_or_control
            if test_or_control == 'control':
                print('control')
            regrouped_info = pd.concat([regrouped_info, df], axis=0)
        return regrouped_info