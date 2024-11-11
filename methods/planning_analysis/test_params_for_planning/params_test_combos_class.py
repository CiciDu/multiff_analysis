import sys
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.test_params_for_planning import params_utils
from visualization.dash_tools import dash_prep_class
import pandas as pd
import numpy as np
from scipy import stats





class ParamsTestCombos(dash_prep_class.DashCartesianPreparation):

    remove_i_o_modify_rows_with_big_ff_angles = True



    # curv_of_traj_lower_end_based_on_mode = {'time': np.arange(-1.5, 0, 0.25), 
    #                                 'distance': np.arange(-150, 0, 25)}
    
    # curv_of_traj_upper_end_based_on_mode = {'time': np.arange(0.1, 1.5, 0.25),
    #                                         'distance': np.arange(10, 150, 25)}
    
    # ref_point_info = {'time': {'min': -1.5,
    #                                                     'max': 0.1,
    #                                                     'step': 0.25,
    #                                                     'values': None,
    #                                                     'marks': None},
    #                                 'distance': {'min': -150,
    #                                                         'max': 0,
    #                                                         'step': 25,
    #                                                         'values': None,
    #                                                         'marks': None}
    #                                 }
    


    curv_of_traj_lower_end_based_on_mode = {'time': np.arange(-1.9, 0, 0.2), 
                                    'distance': np.arange(-190, 0, 20)}
    
    curv_of_traj_upper_end_based_on_mode = {'time': np.arange(0.1, 2.1, 0.2),
                                            'distance': np.arange(10, 210, 20)}
    
    ref_point_info = {'time': {'min': -1.9,
                                'max': 0.1,
                                'step': 0.2,
                                'values': None,
                                'marks': None},
                    'distance': {'min': -190,
                                'max': 0,
                                'step': 20,
                                'values': None,
                                'marks': None}
                    }


        

    def __init__(self, 
                 raw_data_folder_path=None):
        
        super().__init__(raw_data_folder_path=raw_data_folder_path)
        
        self.tested_combo_df = pd.DataFrame()
        self.stop_point_index = None
        self.ref_point_info = params_utils.add_values_and_marks_to_ref_point_info(self.ref_point_info)
        self.ref_point_params_based_on_mode = {'time': self.ref_point_info['time']['values'], 
                                              'distance': self.ref_point_info['distance']['values']}

    def generate_all_combo_df(self):
        self.all_combo_df = params_utils.generate_possible_combos_for_planning(curv_of_traj_lower_end_based_on_mode=self.curv_of_traj_lower_end_based_on_mode,
                                    curv_of_traj_upper_end_based_on_mode=self.curv_of_traj_upper_end_based_on_mode,
                                    ref_point_value_based_on_mode=self.ref_point_params_based_on_mode)
        self.all_combo_df = params_utils.add_columns_of_dummy_variables_to_all_combos_df(self.all_combo_df, column_names=['use_curvature_to_ff_center'])
        self.all_combo_short = self.all_combo_df.copy()
        self.all_combo_df = params_utils.add_columns_of_dummy_variables_to_all_combos_df(self.all_combo_df, column_names=['truncate_curv_of_traj_by_time_of_capture', 'eliminate_outliers'])
        self.all_combo_df[['sample_size', 'curv_r', 'shuffled_curv_r_mean', 'shuffled_curv_r_std',
                            'heading_r', 'shuffled_heading_r_mean', 'shuffled_heading_r_std']] = None
        


    def test_all_set_of_hyperparameters(self, sample_size):
        for i in range(self.all_combo_short.shape[0]):
            print(i, 'out of', self.all_combo_short.shape[0])
            self.test_one_set_of_hyperparameters(sample_size=sample_size, position_index=i)
        return self.tested_combo_df



    # ==================================== Helper Functions ==================================== #
        
    def print_current_info(self):
        print('=============================================')
        print('eliminate_outliers:', self.overall_params['eliminate_outliers'], '; truncate_curv_of_traj_by_time_of_capture:', self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'], 
              '\n', 'ref_point_mode:', self.ref_point_params['ref_point_mode'], '; ref_point_value:', self.ref_point_params['ref_point_value'], 
              '\n', 'curv_of_traj_mode:', self.curv_of_traj_params['curv_of_traj_mode'], '; window_for_curv_of_traj:', self.curv_of_traj_params['window_for_curv_of_traj'])



    def get_a_set_of_hyperparameters(self, position_index=None):
        if position_index is None:
            self.row = self.all_combo_short.sample(n=1).iloc[0]
        else:
            self.row = self.all_combo_short.iloc[position_index]
        self.ref_point_params['ref_point_mode'] = self.row['ref_point_mode']
        self.ref_point_params['ref_point_value'] = self.row['ref_point_value']
        self.curv_of_traj_params['curv_of_traj_mode'] = self.row['curv_of_traj_mode']
        self.curv_of_traj_lower_end = self.row['curv_of_traj_lower_end']
        self.curv_of_traj_upper_end = self.row['curv_of_traj_upper_end']
        self.curv_of_traj_params['window_for_curv_of_traj'] = [self.curv_of_traj_lower_end, self.curv_of_traj_upper_end]
        self.overall_params['use_curvature_to_ff_center'] = self.row['use_curvature_to_ff_center']
        
        # self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = self.row['truncate_curv_of_traj_by_time_of_capture']
        # self.overall_params['eliminate_outliers'] = self.row['eliminate_outliers']
       
        if self.overall_params['use_curvature_to_ff_center']:
            self.overall_params['remove_i_o_modify_rows_with_big_ff_angles'] = True
        else:
            self.overall_params['remove_i_o_modify_rows_with_big_ff_angles'] = False

        self.current_main_hyperparameters_info = (self.all_combo_df['ref_point_mode'] == self.row['ref_point_mode']) &\
                        (self.all_combo_df['curv_of_traj_mode'] == self.row['curv_of_traj_mode']) &\
                        (self.all_combo_df['ref_point_value'] == self.row['ref_point_value']) &\
                        (self.all_combo_df['curv_of_traj_lower_end'] == self.row['curv_of_traj_lower_end']) &\
                        (self.all_combo_df['curv_of_traj_upper_end'] == self.row['curv_of_traj_upper_end']) &\
                        (self.all_combo_df['use_curvature_to_ff_center'] == self.row['use_curvature_to_ff_center'])
    


    def test_one_set_of_hyperparameters(self, sample_size, position_index=None):

        self.get_a_set_of_hyperparameters(position_index=position_index)
        self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = False
        self.overall_params['eliminate_outliers'] = False

        self.streamline_organizing_info(ref_point_mode=self.ref_point_params['ref_point_mode'], 
                                        ref_point_value=self.ref_point_params['ref_point_value'], 
                                        curv_of_traj_mode=self.curv_of_traj_params['curv_of_traj_mode'], 
                                        window_for_curv_of_traj=self.curv_of_traj_params['window_for_curv_of_traj'], 
                                        truncate_curv_of_traj_by_time_of_capture=self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'], 
                                        eliminate_outliers=self.overall_params['eliminate_outliers'],
                                        remove_i_o_modify_rows_with_big_ff_angles=self.overall_params['remove_i_o_modify_rows_with_big_ff_angles'],
                                        use_curvature_to_ff_center=self.overall_params['use_curvature_to_ff_center'])
        self.all_combo_df.loc[self.current_main_hyperparameters_info, ['sample_size']] = sample_size
                
        self.calculate_curv_r_and_heading_r(sample_size)

        self.overall_params['eliminate_outliers'] = True
        self._rerun_after_changing_eliminate_outliers()
        self.calculate_curv_r_and_heading_r(sample_size)

        self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture'] = True
        self.overall_params['eliminate_outliers'] = False
        self._rerun_after_changing_curv_of_traj_params()
        self.calculate_curv_r_and_heading_r(sample_size)

        self.overall_params['eliminate_outliers'] = True
        self._rerun_after_changing_eliminate_outliers()
        self.calculate_curv_r_and_heading_r(sample_size)

        self.tested_combo_df = pd.concat([self.tested_combo_df, self.all_combo_df.loc[self.current_main_hyperparameters_info, :]], axis=0)




    def calculate_curv_r_and_heading_r(self, sample_size):
        self.current_full_hyperparameters_info = self.current_main_hyperparameters_info &\
                                                (self.all_combo_df['truncate_curv_of_traj_by_time_of_capture'] == self.curv_of_traj_params['truncate_curv_of_traj_by_time_of_capture']) &\
                                                (self.all_combo_df['eliminate_outliers'] == self.overall_params['eliminate_outliers'])         
        self.calculate_curv_r(sample_size)
        self.calculate_heading_r(sample_size)
        #self.print_current_info()



    def calculate_curv_r(self, sample_size=100):
        traj_curv_counted = np.array(self.traj_curv_counted)
        alt_curv_counted = np.array(self.alt_curv_counted)
        slope, intercept, r_value, p_value, std_err = stats.linregress(alt_curv_counted, traj_curv_counted)

        all_r_values, all_p_values = params_utils.generate_distribution_of_correlation_after_shuffling_alt_ff_curv(self.alt_ff_counted_df, self.stop_ff_counted_df, self.curv_of_traj_counted, self.overall_params['use_curvature_to_ff_center'], sample_size)
        self.all_combo_df.loc[self.current_full_hyperparameters_info, ['curv_r', 'shuffled_curv_r_mean', 'shuffled_curv_r_std']] = r_value, np.mean(all_r_values), np.std(all_r_values)



    def calculate_heading_r(self, sample_size=100):
        d_heading_alt = self.alt_ff_counted_df['d_heading_of_arc'].values.copy()
        d_heading_stop = self.stop_ff_counted_df['d_heading_of_arc'].values.copy()
        d_heading_of_traj = self.alt_ff_counted_df['d_heading_of_traj'].values.copy()

        rel_heading_traj = d_heading_of_traj - d_heading_stop
        rel_heading_alt = d_heading_alt - d_heading_stop
        rel_heading_traj = find_stops_near_ff_utils.confine_angle_to_within_one_pie(rel_heading_traj)
        rel_heading_alt = find_stops_near_ff_utils.confine_angle_to_within_one_pie(rel_heading_alt)
        slope, intercept, r_value, p_value, std_err = stats.linregress(rel_heading_alt, rel_heading_traj)

        all_r_values, all_p_values = params_utils.generate_distribution_of_correlation_after_shuffling_d_heading(d_heading_alt, d_heading_stop, d_heading_of_traj, sample_size)
        self.all_combo_df.loc[self.current_full_hyperparameters_info, ['heading_r', 'shuffled_heading_r_mean', 'shuffled_heading_r_std']] = r_value, np.mean(all_r_values), np.std(all_r_values)











