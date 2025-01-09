from data_wrangling import specific_utils, further_processing_class, general_utils, retrieve_raw_data, time_offset_utils
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


class TimeOffsetClass(further_processing_class.FurtherProcessing):
    def __init__(self, raw_data_folder_path=None):
        super().__init__(raw_data_folder_path=raw_data_folder_path)

    def prepare_data(self):

        self.retrieve_or_make_monkey_data()


    def get_ff_capture_time_from_smr_and_neural_data(self):
        self.neural_offset_df = pd.read_csv(os.path.join(self.metadata_folder_path, 'neural_time_offset.txt'))
        self.Channel_signal_output, self.marker_list, smr_sampling_rate = retrieve_raw_data.extract_smr_data(self.raw_data_folder_path)
        if ('Schro' in self.raw_data_folder_path) & ('data_0410' in self.raw_data_folder_path):
            self.neural_events_start_time = self.neural_offset_df.loc[self.neural_offset_df['label'] == 4, 'time'].values[0]
            self.smr_markers_start_time = self.marker_list[0]['values'][self.marker_list[0]['labels'] == 4][0]
        else:
            self.neural_events_start_time = self.neural_offset_df.loc[self.neural_offset_df['label'] == 1, 'time'].values[0]
            self.smr_markers_start_time, smr_markers_end_time = time_offset_utils.find_smr_markers_start_and_end_time(self.raw_data_folder_path,
                                                                                            exists_ok=False)

        self.neural_t = self.neural_offset_df.loc[self.neural_offset_df['label'] == 4, 'time'].values
        self.smr_t = self.marker_list[0]['values'][self.marker_list[0]['labels'] == 4]
        self.txt_t = self.ff_caught_T_sorted.copy()


    def make_ff_caught_times_df(self):
        if not hasattr(self, 'neural_t'):
            self.get_ff_capture_time_from_smr_and_neural_data()
        self.ff_caught_times_df = time_offset_utils.make_ff_caught_times_df(self.neural_t, self.smr_t, self.txt_t, 
                                                                       self.neural_events_start_time, self.smr_markers_start_time)
    
    def separate_ff_caught_times_df(self):
        self.txt_and_smr_columns = [col for col in self.ff_caught_times_df.columns if ('neural' not in col)]
        self.smr_and_neural_columns = [col for col in self.ff_caught_times_df.columns if ('txt' not in col)]
        self.txt_and_neural_columns = [col for col in self.ff_caught_times_df.columns if ('smr' not in col)]

        self.txt_and_smr = self.ff_caught_times_df[['txt_t', 'txt_t_adj', 'closest_smr_t_to_txt_t', 'diff_txt_smr_closest', 'diff_txt_adj_smr_closest', 'diff_txt_adj_2_smr_closest']].dropna(axis=0)
        self.smr_and_neural = self.ff_caught_times_df[['smr_t', 'neural_t_adj', 'neural_t_adj_2', 'diff_neural_adj_smr', 'diff_neural_adj_2_smr']].dropna(axis=0)
        self.txt_and_neural = self.ff_caught_times_df[['txt_t_adj', 'neural_t_adj', 'neural_t_adj_2', 'diff_txt_adj_neural_adj', 'diff_txt_adj_neural_adj_2', 'diff_txt_adj_2_neural_adj']].dropna(axis=0)

    def compare_txt_and_smr_with_boxplot(self):
        # make a long df for plotting
        self.long_txt_smr_df = self.ff_caught_times_df[['diff_txt_adj_smr_closest', 'diff_txt_smr_closest', 'diff_txt_adj_2_smr_closest']].melt()
        self.long_txt_smr_df.columns = ['whether_txt_adjusted', 'diff_in_time_between_txt_and_smr']
        self.long_txt_smr_df['whether txt adjusted'] = 'txt - closest smr'
        self.long_txt_smr_df.loc[self.long_txt_smr_df['whether_txt_adjusted'] == 'diff_txt_adj_smr_closest', 'whether txt adjusted'] = 'adjusted txt - closest smr'
        self.long_txt_smr_df.loc[self.long_txt_smr_df['whether_txt_adjusted'] == 'diff_txt_adj_2_smr_closest', 'whether txt adjusted'] = 'adjusted txt_2 - closest smr'

        # make a boxplot of the differences in capture time
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=self.long_txt_smr_df, x='diff_in_time_between_txt_and_smr', hue='whether txt adjusted')
        plt.title('txt capture time - closest smr capture time')
        # hide the title of the legend
        plt.gca().get_legend().set_title('')
        # make the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    def compare_txt_and_smr_with_scatterplot(self):
        # make a scatter plot of the differences in capture time
        num_rows = len(self.ff_caught_times_df)
        plt.figure(figsize=(6, 4))
        plt.scatter(np.arange(num_rows), self.ff_caught_times_df['diff_txt_smr_closest'], s=5, c='blue', label='txt - closest smr')
        plt.scatter(np.arange(num_rows), self.ff_caught_times_df['diff_txt_adj_smr_closest'], s=5, c='orange', label='adjusted txt - closest smr')
        plt.scatter(np.arange(num_rows), self.ff_caught_times_df['diff_txt_adj_2_smr_closest'], s=5, c='green', label='adjusted txt_2 - closest smr')
        plt.plot(np.arange(num_rows), np.zeros(num_rows))
        plt.title('txt capture time - closest smr capture time')
        plt.legend()
        plt.show()


    def compare_txt_and_neural_with_scatterplot(self):
        # make a scatter plot of the differences in capture time
        num_rows = len(self.txt_and_neural)
        plt.figure(figsize=(6, 4))
        plt.scatter(np.arange(num_rows), self.txt_and_neural['diff_txt_adj_neural_adj'], s=5, c='blue', label='txt adjusted - neural adjusted by label==4')
        plt.scatter(np.arange(num_rows), self.txt_and_neural['diff_txt_adj_2_neural_adj'], s=5, c='orange', label='txt adjusted - neural adjusted by label==1')
        plt.plot(np.arange(num_rows), np.zeros(num_rows))
        plt.title('txt adjusted capture time - neural adjusted capture time')
        plt.legend()
        plt.show()


    def compare_smr_and_neural_with_scatterplot(self):
        # make a scatter plot of the differences in capture time
        num_rows = len(self.smr_and_neural)
        plt.figure(figsize=(6, 4))
        plt.scatter(np.arange(num_rows), self.smr_and_neural['diff_neural_adj_smr'], s=5, c='blue', label='neural - smr adjusted by label==4')
        plt.scatter(np.arange(num_rows), self.smr_and_neural['diff_neural_adj_2_smr'], s=5, c='orange', label='neural - smr adjusted by label==1')
        plt.plot(np.arange(num_rows), np.zeros(num_rows))
        plt.title('neural capture time - smr capture time')
        plt.legend()
        plt.show()

