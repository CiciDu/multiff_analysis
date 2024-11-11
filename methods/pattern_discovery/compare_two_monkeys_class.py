import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
from data_wrangling import basic_func, base_processing_class, combine_info_utils, monkey_data_classes
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, organize_patterns_and_features, patterns_and_features_class
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


class CompareTwoMonkeys():

    def __init__(self):

        self.bruno = patterns_and_features_class.PatternsAndFeatures(monkey_name='monkey_Bruno')
        self.bruno.combine_patterns_and_features(verbose=False)

        self.schro = patterns_and_features_class.PatternsAndFeatures(monkey_name='monkey_Schro')
        self.schro.combine_patterns_and_features(verbose=False)

        self.monkey_name = ''

        self.combine_df()


    def combine_df(self, df_names=['agg_pattern_frequencies', 'agg_feature_statistics', 'concat_pattern_frequencies', 'concat_feature_statistics', 'concat_all_trial_features']):
        for df_name in df_names:
            bruno_df = getattr(self.bruno, df_name)
            schro_df = getattr(self.schro, df_name)
            bruno_df['Monkey'] = 'Bruno'
            schro_df['Monkey'] = 'Schro'
            setattr(self, df_name, pd.concat([bruno_df, schro_df], axis=0).reset_index(drop=True))


    def plot_feature_statistics(self):
        patterns_and_features_class.PatternsAndFeatures.plot_feature_statistics(self, hue='Monkey')


    def plot_pattern_frequencies(self):
        patterns_and_features_class.PatternsAndFeatures.plot_pattern_frequencies(self, hue='Monkey')


    def plot_the_changes_in_pattern_frequencies_over_time(self):
        plot_change_over_time.plot_the_changes_over_time_for_two_monkeys(self.concat_pattern_frequencies, x="Session", y="Rate", 
                                                category_order=patterns_and_features_class.PatternsAndFeatures.pattern_order)    

    def plot_the_changes_in_feature_statistics_over_time(self):
        plot_change_over_time.plot_the_changes_over_time_for_two_monkeys(self.concat_feature_statistics, x="Session", y="Mean", title_column="Label for mean", 
                                                category_order=patterns_and_features_class.PatternsAndFeatures.feature_order)   
        plot_change_over_time.plot_the_changes_over_time_for_two_monkeys(self.concat_feature_statistics, x="Session", y="Median", title_column="Label for median", 
                                                category_order=patterns_and_features_class.PatternsAndFeatures.feature_order)                                                    
        
