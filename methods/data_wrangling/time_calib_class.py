from data_wrangling import specific_utils, further_processing_class, general_utils, retrieve_raw_data, time_calib_utils
from pattern_discovery import pattern_by_trials, organize_patterns_and_features, monkey_landing_in_ff
from visualization import plot_behaviors_utils
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from decision_making_analysis.GUAT import GUAT_utils
from data_wrangling import specific_utils, process_monkey_information
from pattern_discovery import pattern_by_points

import os
import os.path
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import seaborn as sns


class TimeCalibration(further_processing_class.FurtherProcessing):
    def __init__(self, raw_data_folder_path=None):
        super().__init__(raw_data_folder_path=raw_data_folder_path)

    def prepare_data(self):
        self.retrieve_or_make_monkey_data()

    def get_ff_capture_time_from_smr_and_neural_data(self):
        self.neural_event_time = pd.read_csv(os.path.join(self.time_calibration_folder_path, 'neural_event_time.txt'))
        self.channel_signal_output, self.marker_list, smr_sampling_rate = retrieve_raw_data.extract_smr_data(self.raw_data_folder_path)
        if ('Schro' in self.raw_data_folder_path) & ('data_0410' in self.raw_data_folder_path):
            self.neural_events_start_time = self.neural_event_time.loc[self.neural_event_time['label'] == 4, 'time'].values[0]
            self.smr_markers_start_time = self.marker_list[0]['values'][self.marker_list[0]['labels'] == 4][0]
        else:
            self.neural_events_start_time = self.neural_event_time.loc[self.neural_event_time['label'] == 1, 'time'].values[0]
            self.smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(self.raw_data_folder_path,
                                                                                            exists_ok=False)

        self.neural_t_raw = self.neural_event_time.loc[self.neural_event_time['label'] == 4, 'time'].values
        self.smr_t_raw = self.marker_list[0]['values'][self.marker_list[0]['labels'] == 4]
        self.txt_t = self.ff_caught_T_sorted.copy()

    def make_adjusted_ff_caught_times_df(self):
        if not hasattr(self, 'neural_t'):
            self.get_ff_capture_time_from_smr_and_neural_data()
        self.ff_caught_times_df = time_calib_utils.make_adjusted_ff_caught_times_df(self.neural_t_raw, self.smr_t_raw, self.txt_t, 
                                                                       self.neural_events_start_time, self.smr_markers_start_time)
    
    def separate_ff_caught_times_df(self):
        self.txt_and_smr_columns = [col for col in self.ff_caught_times_df.columns if ('neural' not in col)]
        self.smr_and_neural_columns = [col for col in self.ff_caught_times_df.columns if ('txt' not in col)]
        self.txt_and_neural_columns = [col for col in self.ff_caught_times_df.columns if ('smr' not in col)]

        self.txt_and_smr = self.ff_caught_times_df[['txt_t', 'smr_t', 'diff_txt_smr', 'diff_txt_smr', 'diff_txt_smr_2']].dropna(axis=0)
        self.smr_and_neural = self.ff_caught_times_df[['txt_t', 'smr_t', 'neural_t', 'neural_t_2', 'diff_neural_smr', 'diff_neural_2_smr_2']].dropna(axis=0)
        self.txt_and_neural = self.ff_caught_times_df[['txt_t', 'neural_t', 'neural_t_2', 'diff_txt_neural', 'diff_txt_neural_2', 'diff_txt_neural_3', 'diff_txt_neural_4', 'diff_txt_neural_raw']].dropna(axis=0)

    def compare_txt_and_smr_with_boxplot(self):
        # make a long df for plotting
        self.long_txt_smr_df = self.ff_caught_times_df[['diff_txt_smr_raw', 'diff_txt_smr', 'diff_txt_smr_2']].melt()
        self.long_txt_smr_df.columns = ['whether_smr_adjusted', 'diff_in_time_between_txt_and_smr']
        self.long_txt_smr_df['whether smr adjusted'] = 'txt - smr raw'
        self.long_txt_smr_df.loc[self.long_txt_smr_df['whether_smr_adjusted'] == 'diff_txt_smr', 'whether smr adjusted'] = 'adj by 1st txt t'
        self.long_txt_smr_df.loc[self.long_txt_smr_df['whether_smr_adjusted'] == 'diff_txt_smr_2', 'whether smr adjusted'] = 'adj by median of diff_t'

        # make a boxplot of the differences in capture time
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=self.long_txt_smr_df, x='diff_in_time_between_txt_and_smr', hue='whether smr adjusted')
        plt.title('txt capture time - smr capture time')
        # hide the title of the legend
        plt.gca().get_legend().set_title('')
        # make the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    def compare_txt_and_smr_with_scatterplot(self, remove_outliers=True):

        time_axis = self.ff_caught_times_df['txt_t'].values
        ax, stat_df = time_calib_utils.get_linear_regression(time_axis, self.ff_caught_times_df['diff_txt_smr_raw'], remove_outliers=remove_outliers,
                                                        label='txt - smr raw')
        ax, temp_stat_df1 = time_calib_utils.get_linear_regression(time_axis, self.ff_caught_times_df['diff_txt_smr'], remove_outliers=remove_outliers,
                                                        ax=ax, color='green', label='adj by 1st txt t')
        ax, temp_stat_df2 = time_calib_utils.get_linear_regression(time_axis, self.ff_caught_times_df['diff_txt_smr_2'], remove_outliers=remove_outliers,
                                                        ax=ax, color='orange', label='adj by median of diff_t')
        
        stat_df = pd.concat([stat_df, temp_stat_df1, temp_stat_df2], axis=0).reset_index(drop=True)
        print(stat_df)

        plt.plot(time_axis, np.zeros(len(time_axis)), c='red')
        title_str = 'txt capture time - smr capture time'
        title_str = title_str + ' (outliers removed in linreg_model)' if remove_outliers else title_str
        plt.title(title_str)
        plt.legend()
        plt.show()

    def compare_txt_and_neural_with_scatterplot(self, remove_outliers=True):
        time_axis = self.txt_and_neural['txt_t'].values
        ax, stat_df = time_calib_utils.get_linear_regression(time_axis, self.txt_and_neural['diff_txt_neural'], remove_outliers=remove_outliers,
                                                        label='adj by 1st txt t')
        ax, temp_stat_df1 = time_calib_utils.get_linear_regression(time_axis, self.txt_and_neural['diff_txt_neural_2'], remove_outliers=remove_outliers,
                                                        ax=ax, color='green', label='adj by label=1, median of diff_t')
        ax, temp_stat_df2 = time_calib_utils.get_linear_regression(time_axis, self.txt_and_neural['diff_txt_neural_3'], remove_outliers=remove_outliers,
                                                        ax=ax, color='orange', label='adj by label=1, 1st txt t')
        ax, temp_stat_df3 = time_calib_utils.get_linear_regression(time_axis, self.txt_and_neural['diff_txt_neural_4'], remove_outliers=remove_outliers,
                                                        ax=ax, color='purple', label='adj by only label=1')
    
        stat_df = pd.concat([stat_df, temp_stat_df1, temp_stat_df2, temp_stat_df3], axis=0).reset_index(drop=True)
        print(stat_df)
        
        plt.plot(time_axis, np.zeros(len(time_axis)), c='red')
        title_str = 'txt capture time - neural capture time'
        title_str = title_str + ' (outliers removed in linreg_model)' if remove_outliers else title_str
        plt.title(title_str)
        plt.legend()
        plt.show()


    def compare_neural_and_smr_with_scatterplot(self, remove_outliers=True):
        time_axis = self.smr_and_neural['txt_t'].values
        ax, stat_df = time_calib_utils.get_linear_regression(time_axis, self.smr_and_neural['diff_neural_smr'], remove_outliers=remove_outliers,
                                                        label='adj by 1st txt t')
        ax, temp_stat_df1 = time_calib_utils.get_linear_regression(time_axis, self.smr_and_neural['diff_neural_2_smr_2'], remove_outliers=remove_outliers,
                                                        ax=ax, color='green', label='adj by label=1')
    
        stat_df = pd.concat([stat_df, temp_stat_df1], axis=0).reset_index(drop=True)
        print(stat_df)
        
        plt.plot(time_axis, np.zeros(len(time_axis)), c='red')
        title_str = 'neural capture time - smr capture time'
        title_str = title_str + ' (outliers removed in linreg_model)' if remove_outliers else title_str
        plt.title(title_str)
        plt.legend()
        plt.show()

