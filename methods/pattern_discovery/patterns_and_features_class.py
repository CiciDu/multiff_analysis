import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from data_wrangling import basic_func, base_processing_class, combine_info_utils, monkey_data_classes
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, organize_patterns_and_features
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

    dir_name = 'all_monkey_data/raw_monkey_data/individual_monkey_data'

    pattern_order = ['ff_capture_rate', 'stop_success_rate',
                        'two_in_a_row', 'waste_cluster_around_target', 'visible_before_last_one', 'disappear_latest', 
                        'give_up_after_trying', 'try_a_few_times', 'ignore_sudden_flash']

    feature_order = ['t', 't_last_visible', 'd_last_visible', 'abs_angle_last_visible',
                    'num_stops', 'num_stops_since_last_visible', 'num_stops_near_target']

    def __init__(self, monkey_name='monkey_Bruno'):
        self.monkey_name = monkey_name
        self.combd_patterns_and_features_folder_path = "all_monkey_data/patterns_and_features/combined_data"

    def combine_patterns_and_features(self, exists_ok=True, save_data=True, verbose=True):

        if exists_ok:
            try:
                self._retrieve_combined_patterns_and_features()
                return
            except FileNotFoundError:
                pass

        # suppress printed output just for this function
        original_stdout = sys.stdout
        if not verbose:
            sys.stdout = open(os.devnull, 'w')

        self.sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(self.dir_name, self.monkey_name)

        self.concat_pattern_frequencies = pd.DataFrame()
        self.concat_feature_statistics = pd.DataFrame()
        self.concat_all_trial_features = pd.DataFrame()
        for index, row in self.sessions_df_for_one_monkey.iterrows():
            if row['finished'] is True:
                continue
        
            data_name = row['data_name']
            raw_data_folder_path = os.path.join(self.dir_name, row['monkey_name'], data_name)
            print(raw_data_folder_path)
            self.data_item = monkey_data_classes.ProcessMonkeyData(raw_data_folder_path=raw_data_folder_path)
            self.data_item.make_df_related_to_patterns_and_features(exists_ok=exists_ok)

            self.data_item.pattern_frequencies['data_name'] = data_name
            self.data_item.feature_statistics['data_name'] = data_name
            self.data_item.all_trial_features['data_name'] = data_name

            self.concat_pattern_frequencies = pd.concat([self.concat_pattern_frequencies, self.data_item.pattern_frequencies], axis=0).reset_index(drop=True)
            self.concat_feature_statistics = pd.concat([self.concat_feature_statistics, self.data_item.feature_statistics], axis=0).reset_index(drop=True)
            self.concat_all_trial_features = pd.concat([self.concat_all_trial_features, self.data_item.all_trial_features], axis=0).reset_index(drop=True)

        self.concat_pattern_frequencies = organize_patterns_and_features.add_dates_and_sessions(self.concat_pattern_frequencies)
        self.concat_feature_statistics = organize_patterns_and_features.add_dates_and_sessions(self.concat_feature_statistics)
        self.concat_all_trial_features = organize_patterns_and_features.add_dates_and_sessions(self.concat_all_trial_features)

        self.agg_pattern_frequencies = self._make_agg_pattern_frequency()
        self.agg_feature_statistics = organize_patterns_and_features.make_feature_statistics(self.concat_all_trial_features.drop(columns='data_name'), data_folder_name = None)

        if save_data:
            os.makedirs(self.combd_patterns_and_features_folder_path, exist_ok=True)
            self.concat_pattern_frequencies.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'concat_pattern_frequencies.csv'))
            self.concat_feature_statistics.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'concat_feature_statistics.csv'))
            self.concat_all_trial_features.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'concat_all_trial_features.csv'))
            self.agg_pattern_frequencies.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_pattern_frequencies.csv'))
            self.agg_feature_statistics.to_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_feature_statistics.csv'))

        sys.stdout = original_stdout

        return
    
    def _retrieve_combined_patterns_and_features(self):
        self.concat_pattern_frequencies = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'concat_pattern_frequencies.csv')).drop(columns='Unnamed: 0')
        self.concat_feature_statistics = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'concat_feature_statistics.csv')).drop(columns='Unnamed: 0')
        self.concat_all_trial_features = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'concat_all_trial_features.csv')).drop(columns='Unnamed: 0')
        self.agg_pattern_frequencies = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_pattern_frequencies.csv')).drop(columns='Unnamed: 0')
        self.agg_feature_statistics = pd.read_csv(os.path.join(self.combd_patterns_and_features_folder_path, 'agg_feature_statistics.csv')).drop(columns='Unnamed: 0')
        return


    def _make_agg_pattern_frequency(self):
        self.agg_pattern_frequencies = self.concat_pattern_frequencies.drop(columns='data_name').groupby(['Item', 'Group', 'Label']).sum().reset_index()
        self.agg_pattern_frequencies['Rate'] = self.agg_pattern_frequencies['Frequency']/self.agg_pattern_frequencies['N_total']
        self.agg_pattern_frequencies['Percentage'] = self.agg_pattern_frequencies['Rate']*100
        return self.agg_pattern_frequencies
    

    def plot_feature_statistics(self, hue=None):
        plot_statistics.plot_feature_statistics(self.agg_feature_statistics, monkey_name=self.monkey_name, hue=hue)
        plot_statistics.plot_feature_statistics(self.concat_feature_statistics, monkey_name=self.monkey_name, hue=hue)


    def plot_pattern_frequencies(self, hue=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1 = plot_statistics.plot_pattern_frequencies(self.agg_pattern_frequencies, monkey_name=self.monkey_name, ax=ax1, return_ax=True, hue=hue)
        ax2 = plot_statistics.plot_pattern_frequencies(self.concat_pattern_frequencies, monkey_name=self.monkey_name, ax=ax2, return_ax=True, hue=hue)
        plt.show()


    def plot_the_changes_in_pattern_frequencies_over_time(self):
        plot_change_over_time.plot_the_changes_over_time(self.concat_pattern_frequencies, x="Session", y="Rate", 
                                                monkey_name='monkey_Bruno',
                                                category_order=self.pattern_order)
        

    def plot_the_changes_in_feature_statistics_over_time(self):
        plot_change_over_time.plot_the_changes_over_time(self.concat_feature_statistics, x="Session", y="Median", title_column="Label for median", 
                                                                          monkey_name=self.monkey_name, category_order=self.feature_order)
        plot_change_over_time.plot_the_changes_over_time(self.concat_feature_statistics, x="Session", y="Mean", title_column="Label for mean", 
                                                                          monkey_name=self.monkey_name, category_order=self.feature_order)
