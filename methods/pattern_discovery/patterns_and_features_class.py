import sys
from data_wrangling import basic_func, base_processing_class, combine_info_utils, monkey_data_classes
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, organize_patterns_and_features, monkey_landing_in_ff
from visualization import animation_func, animation_utils, plot_trials, plot_behaviors_utils, plot_statistics, plot_change_over_time
from data_wrangling import base_processing_class

import os
import os.path
import pandas as pd
import numpy as np
import matplotlib
from functools import partial
from matplotlib import rc, animation
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from functools import partial



plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class PatternsAndFeatures():

    raw_data_dir_name = 'all_monkey_data/raw_monkey_data/individual_monkey_data'

    pattern_order = ['ff_capture_rate', 'stop_success_rate',
                        'two_in_a_row', 'waste_cluster_around_target', 'visible_before_last_one', 'disappear_latest', 
                        'give_up_after_trying', 'try_a_few_times', 'ignore_sudden_flash']

    feature_order = ['t', 't_last_vis', 'd_last_vis', 'abs_angle_last_vis',
                    'num_stops', 'num_stops_since_last_vis', 'num_stops_near_target']

    def __init__(self, monkey_name='monkey_Bruno'):
        self.monkey_name = monkey_name
        self.combd_patterns_and_features_folder_path = f"all_monkey_data/patterns_and_features/{self.monkey_name}/combined_data"


    def combine_or_retrieve_patterns_and_features(self, exists_ok=True, save_data=True, verbose=True):

        if exists_ok:
            try:
                self._retrieve_combined_patterns_and_features()
                print('Successfully retrieved combined_patterns_and_features')
                return
            except FileNotFoundError:
                print('Failed to retrieve combined_patterns_and_features. Will make them anew.')
                pass

        if not verbose:
            with basic_func.suppress_stdout():
                self._combine_patterns_and_features(exists_ok=exists_ok, save_data=save_data)
        else:
            self._combine_patterns_and_features(exists_ok=exists_ok, save_data=save_data)

        return
    
    def _combine_patterns_and_features(self, exists_ok=True, save_data=True):
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.raw_data_dir_name, self.monkey_name)

        self.combd_pattern_frequencies = pd.DataFrame()
        self.combd_feature_statistics = pd.DataFrame()
        self.combd_all_trial_features = pd.DataFrame()
        self.combd_scatter_around_target_df = pd.DataFrame()
        
        for index, row in self.sessions_df_for_one_monkey.iterrows():
            if row['finished'] is True:
                continue
        
            data_name = row['data_name']
            raw_data_folder_path = os.path.join(self.raw_data_dir_name, row['monkey_name'], data_name)
            print(raw_data_folder_path)
            self.data_item = monkey_data_classes.ProcessMonkeyData(raw_data_folder_path=raw_data_folder_path)
            self.data_item.make_df_related_to_patterns_and_features(exists_ok=exists_ok)
            print('Successfully made df related to patterns and features for ', data_name)

            self.data_item.pattern_frequencies['data_name'] = data_name
            self.data_item.feature_statistics['data_name'] = data_name
            self.data_item.all_trial_features['data_name'] = data_name
            self.data_item.scatter_around_target_df['data_name'] = data_name

            self.combd_pattern_frequencies = pd.concat([self.combd_pattern_frequencies, self.data_item.pattern_frequencies], axis=0).reset_index(drop=True)
            self.combd_feature_statistics = pd.concat([self.combd_feature_statistics, self.data_item.feature_statistics], axis=0).reset_index(drop=True)
            self.combd_all_trial_features = pd.concat([self.combd_all_trial_features, self.data_item.all_trial_features], axis=0).reset_index(drop=True)
            self.combd_scatter_around_target_df = pd.concat([self.combd_scatter_around_target_df, self.data_item.scatter_around_target_df], axis=0).reset_index(drop=True)

        organize_patterns_and_features.add_dates_and_sessions(self.combd_pattern_frequencies)
        organize_patterns_and_features.add_dates_and_sessions(self.combd_feature_statistics)
        organize_patterns_and_features.add_dates_and_sessions(self.combd_all_trial_features)
        organize_patterns_and_features.add_dates_and_sessions(self.combd_scatter_around_target_df)

        self.agg_pattern_frequencies = self._make_agg_pattern_frequency()
        self.agg_feature_statistics = organize_patterns_and_features.make_feature_statistics(self.combd_all_trial_features.drop(
                                                                                                        columns=['data_name', 'data', 'Date']), data_folder_name = None)

        if save_data:
            os.makedirs(self.combd_patterns_and_features_folder_path, exist_ok=True)
            self.combd_pattern_frequencies.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_pattern_frequencies.csv'))
            self.combd_feature_statistics.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_feature_statistics.csv'))
            self.combd_all_trial_features.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_all_trial_features.csv'))
            self.agg_pattern_frequencies.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_pattern_frequencies.csv'))
            self.agg_feature_statistics.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_feature_statistics.csv'))
            self.combd_scatter_around_target_df.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_scatter_around_target_df.csv'))

    # If only wanting to make combd_scatter_around_target_df
    def make_combd_scatter_around_target_df(self, exists_ok=True, save_data=True):
        df_path = os.path.join(self.combd_patterns_and_features_folder_path, 'combd_scatter_around_target_df.csv')
        if exists_ok & exists(df_path):
            self.combd_scatter_around_target_df = pd.read_csv(df_path).drop(columns='Unnamed: 0')
            return

        self.combd_scatter_around_target_df = pd.DataFrame()
        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.raw_data_dir_name, self.monkey_name)
        for index, row in self.sessions_df_for_one_monkey.iterrows():
            if row['finished'] is True:
                continue
            data_name = row['data_name']
            print('Processing data: ', data_name)
            raw_data_folder_path = os.path.join(self.raw_data_dir_name, row['monkey_name'], data_name)
            self.data_item = monkey_data_classes.ProcessMonkeyData(raw_data_folder_path=raw_data_folder_path)
            self.data_item.retrieve_or_make_monkey_data(exists_ok=True)
            self.data_item.make_or_retrieve_ff_dataframe(exists_ok=True)
            self.data_item.make_or_retrieve_scatter_around_target_df(exists_ok=True)
            self.scatter_around_target_df = self.data_item.scatter_around_target_df
            self.scatter_around_target_df['data_name'] = data_name
            self.combd_scatter_around_target_df = pd.concat([self.combd_scatter_around_target_df, self.scatter_around_target_df], axis=0).reset_index(drop=True)

        organize_patterns_and_features.add_dates_and_sessions(self.combd_scatter_around_target_df)
        if save_data:
            os.makedirs(self.combd_patterns_and_features_folder_path, exist_ok=True)
            self.combd_scatter_around_target_df.to_csv(df_path)
        return

    
    def _retrieve_combined_patterns_and_features(self):
        self.combd_pattern_frequencies = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_pattern_frequencies.csv')).drop(columns='Unnamed: 0')
        self.combd_feature_statistics = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_feature_statistics.csv')).drop(columns='Unnamed: 0')
        self.combd_all_trial_features = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_all_trial_features.csv')).drop(columns='Unnamed: 0')
        self.agg_pattern_frequencies = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_pattern_frequencies.csv')).drop(columns='Unnamed: 0')
        self.agg_feature_statistics = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_feature_statistics.csv')).drop(columns='Unnamed: 0')
        self.combd_scatter_around_target_df = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'combd_scatter_around_target_df.csv')).drop(columns='Unnamed: 0')

        # the line below is used because when the df was saved, 'Percentage' column was not in it.
        self.combd_pattern_frequencies['Percentage'] = self.combd_pattern_frequencies['Rate']*100
        return

    def _make_agg_pattern_frequency(self):
        self.agg_pattern_frequencies = self.combd_pattern_frequencies[['Item', 'Group', 'Label', 'Frequency', 'N_total']].groupby(['Item', 'Group', 'Label']).sum().reset_index()
        self.agg_pattern_frequencies['Rate'] = self.agg_pattern_frequencies['Frequency']/self.agg_pattern_frequencies['N_total']
        self.agg_pattern_frequencies['Percentage'] = self.agg_pattern_frequencies['Rate']*100
        return self.agg_pattern_frequencies
    

    def plot_feature_statistics(self, hue=None):
        plot_statistics.plot_feature_statistics(self.agg_feature_statistics, monkey_name=self.monkey_name, hue=hue)
        plot_statistics.plot_feature_statistics(self.combd_feature_statistics, monkey_name=self.monkey_name, hue=hue)


    def plot_pattern_frequencies(self, hue=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1 = plot_statistics.plot_pattern_frequencies(self.agg_pattern_frequencies, monkey_name=self.monkey_name, ax=ax1, return_ax=True, hue=hue)
        ax2 = plot_statistics.plot_pattern_frequencies(self.combd_pattern_frequencies, monkey_name=self.monkey_name, ax=ax2, return_ax=True, hue=hue)
        plt.show()


    def plot_the_changes_in_pattern_frequencies_over_time(self, multiple_monkeys=False):
        plot_change_over_time.plot_the_changes_over_time_in_long_df(self.combd_pattern_frequencies, x="Session", y="Rate", 
                                                multiple_monkeys=multiple_monkeys, monkey_name='monkey_Bruno',
                                                category_order=self.pattern_order)
        

    def plot_the_changes_in_feature_statistics_over_time(self, multiple_monkeys=False):
        plot_change_over_time.plot_the_changes_over_time_in_long_df(self.combd_feature_statistics, x="Session", y="Median", title_column="Label for median", 
                                                                          multiple_monkeys=multiple_monkeys, monkey_name=self.monkey_name, category_order=self.feature_order)
        plot_change_over_time.plot_the_changes_over_time_in_long_df(self.combd_feature_statistics, x="Session", y="Mean", title_column="Label for mean", 
                                                                         multiple_monkeys=multiple_monkeys, monkey_name=self.monkey_name, category_order=self.feature_order)


    def plot_the_changes_in_scatter_around_target_over_time(self, y_columns=None):
        if y_columns is None:
            y_columns = ['distance_mean', 'distance_std', 'distance_25%', 'distance_50%',
                        'distance_75%', 'distance_iqr', 'angle_mean', 'angle_std', 'angle_25%',
                        'angle_50%', 'angle_75%', 'angle_iqr', 'abs_angle_mean',
                        'abs_angle_std', 'abs_angle_25%', 'abs_angle_50%', 'abs_angle_75%',
                        'abs_angle_iqr', 'Q1_perc', 'Q2_perc', 'Q3_perc', 'Q4_perc']
        plot_change_over_time.plot_the_changes_over_time_in_wide_df(self.combd_scatter_around_target_df, x="Session", 
                                                                y_columns=y_columns,
                                                                monkey_name=self.monkey_name, 
                                                                )
