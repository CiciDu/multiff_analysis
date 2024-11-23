import sys
from data_wrangling import basic_func, process_raw_data, base_processing_class, further_processing_class, monkey_data_classes
from pattern_discovery import pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features


import os
import os.path
import pandas as pd
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



class ProcessMonkeyData(further_processing_class.FurtherProcessing):


    def __init__(self, raw_data_folder_path):
        super().__init__()

        if not exists(raw_data_folder_path):
            raise ValueError(f"raw_data_folder_path {raw_data_folder_path} does not exist.")
        self.extract_info_from_raw_data_folder_path(raw_data_folder_path)
        
        self.cluster_around_target_indices = None



    def save_important_files(self, exists_ok=True, monkey_information_exists_ok=True):
        self.retrieve_or_make_monkey_data(exists_ok=monkey_information_exists_ok)
        
        self.make_or_retrieve_ff_dataframe(exists_ok=exists_ok)
        
        self.make_or_retrieve_target_closest(exists_ok=exists_ok)

        self.make_or_retrieve_target_angle_smallest(exists_ok=exists_ok)
        
        self.find_patterns()

        self.make_df_related_to_patterns_and_features(exists_ok=exists_ok)

        

    def make_df_related_to_patterns_and_features(self, exists_ok=True):
        self.make_or_retrieve_all_trial_patterns(exists_ok=exists_ok)
        
        self.make_or_retrieve_pattern_frequencies(exists_ok=exists_ok)
        
        self.make_or_retrieve_all_trial_features(exists_ok=exists_ok)
        
        self.make_or_retrieve_feature_statistics(exists_ok=exists_ok)


    def make_or_retrieve_target_closest(self, exists_ok=False): # these may need to be run again if they're to be used
        filepath = os.path.join(self.patterns_and_features_data_folder_path, 'target_closest.csv')
        if exists(filepath) & exists_ok:
            self.target_closest = np.genfromtxt(filepath, delimiter=',').astype('int')
            print("Retrieved target_closest")
        else: 
            self.target_closest = pattern_by_points.make_target_closest(self.ff_dataframe, self.max_point_index, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made target_closest")


    def make_or_retrieve_target_angle_smallest(self, exists_ok=False):
        filepath = self.patterns_and_features_data_folder_path + '/target_angle_smallest.csv'
        if exists(filepath) & exists_ok:
            self.target_angle_smallest = np.genfromtxt(filepath, delimiter=',').astype('int')
            print("Retrieved target_angle_smallest")
        else:
            # make target_angle_smallest:
            # 2 means target is has the smallest absolute angle at that point (visible or in memory)
            # 1 means the target does not have the smallest absolute angle. In the subset of 1:
                # 1 means both the target and a non-target are visible or in memory (which we call present)
                # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
                # -1 means both the target and other ff are neither visible or in memory
            self.target_angle_smallest = pattern_by_points.make_target_angle_smallest(self.ff_dataframe, self.max_point_index, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made target_angle_smallest")


    def make_distance_dataframe(self):
        corresponding_t = np.array(self.monkey_information['monkey_t'])[:self.max_point_index+1]
        distance_dataframe = process_raw_data.turn_array_into_df(self, array=self.target_closest, corresponding_t=corresponding_t)
        self.distance_dataframe = distance_dataframe
        self.trial_vs_distance = distance_dataframe[['num', 'trial']].groupby('trial').max()

    def make_angle_dataframe(self):
        corresponding_t = np.array(self.monkey_information['monkey_t'])[:self.max_point_index+1]
        angle_dataframe = process_raw_data.turn_array_into_df(self, array=self.target_angle_smallest, corresponding_t=corresponding_t)
        self.angle_dataframe = angle_dataframe
        self.trial_vs_angle = angle_dataframe[['num', 'trial']].groupby('trial').max()



    def make_bins(self, window_width=1, bin_width=0.25):
        self.window_width = window_width
        self.bin_width = bin_width
        self.min_time = self.monkey_information['monkey_t'].min()
        self.max_time = self.monkey_information['monkey_t'].max()
        self.time_bins = np.arange(self.min_time, self.max_time, self.bin_width) 
        self.monkey_information.loc[:, 'bin'] = np.digitize(self.monkey_information['monkey_t'].values, self.time_bins)-1
        self.continuous_bins = pd.DataFrame({'bin': range(len(self.time_bins))})
        self.binned_features = self.continuous_bins.copy()


    def get_convolution_pattern(self):
        num_bins_in_window = int(self.window_width/self.bin_width)
        if num_bins_in_window % 2 == 0:
            num_bins_in_window += 1
            self.window_width = num_bins_in_window * self.bin_width
        print("True window width: ", self.window_width)
        self.convolve_pattern = np.ones(num_bins_in_window)



    def add_monkey_info(self):
        monkey_information = self.monkey_information.copy()

        # Dummy variable of turning left or right - 0 means left and 1 means right
        monkey_information['turning_right'] = 0
        monkey_information.loc[monkey_information['monkey_dw'] < 0, 'turning_right'] = 1

        # add time_box to monkey_information
        # because for neural data, the first bin starts after the first edge, similarly we will let the first bin of 
        # monkey data (bin number = 0) start after the first edge
        
        rebinned_monkey_info = monkey_information.groupby('bin').mean().reset_index(drop=False)
        rebinned_monkey_info['num_stops'] = monkey_information.groupby('bin').sum()['monkey_speeddummy']


        # add num_caught_ff to rebinned_monkey_info
        catching_target_bins = np.digitize(self.ff_caught_T_sorted, self.time_bins)-1
        catching_target_bins_unique, counts = np.unique(catching_target_bins, return_counts=True)
        rebinned_monkey_info['num_caught_ff'] = 0
        rebinned_monkey_info.loc[catching_target_bins_unique, 'num_caught_ff'] = counts

        for column in ['gaze_monkey_view_x', 'gaze_monkey_view_y', 'gaze_world_x', 'gaze_world_y']:
            rebinned_monkey_info.loc[:,column] = np.clip(rebinned_monkey_info.loc[:,column], -1000, 1000)

        columns_of_interest = ['bin', 'LDy', 'LDz', 'RDy', 'RDz', 'gaze_monkey_view_x', 'gaze_monkey_view_y', 'gaze_world_x', 'gaze_world_y', 
            'monkey_speed', 'monkey_angles', 'monkey_dw', 'monkey_speeddummy', 'monkey_ddw', 'monkey_ddv', 'num_stops', 'num_caught_ff']

        rebinned_monkey_info_essential = rebinned_monkey_info[columns_of_interest].copy()

        self.get_convolution_pattern()
        num_stops_convolved = np.convolve(rebinned_monkey_info_essential['num_stops'], self.convolve_pattern, 'same')
        num_caught_ff_convolved = np.convolve(rebinned_monkey_info_essential['num_caught_ff'], self.convolve_pattern, 'same')
        rebinned_monkey_info_essential['stop_rate'] = num_stops_convolved/self.window_width
        rebinned_monkey_info_essential['stop_success_rate'] = num_caught_ff_convolved/num_stops_convolved    

        self.binned_features = self.binned_features.merge(rebinned_monkey_info_essential, how='left', on='bin')
        self.binned_features = self.binned_features.fillna(method='ffill').reset_index(drop=True)
        self.binned_features = self.binned_features.fillna(method='bfill').reset_index(drop=True)



    def add_ff_info(self):
        ff_dataframe = self.ff_dataframe.copy()
        ff_dataframe['bin'] = np.digitize(ff_dataframe.time, self.time_bins)-1
        ff_dataframe['point_index'] = ff_dataframe['point_index'].astype(int)
        ff_dataframe['monkey_angle'] = self.monkey_information.loc[:, 'monkey_angles'].values[ff_dataframe.loc[:, 'point_index'].values]


        # get some summary statistics from ff_dataframe to use as features for CCA
        # count of visible and in-memory ff
        ff_dataframe_sub = ff_dataframe[['bin', 'ff_index']]
        ff_dataframe_unique_ff = ff_dataframe_sub.groupby('bin').nunique().reset_index(drop=False)
        ff_dataframe_unique_ff.rename(columns={'ff_index': 'num_alive_ff'}, inplace=True)
        self.binned_features = self.binned_features.merge(ff_dataframe_unique_ff, how='left', on='bin')


        # count of visible ff
        ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1]
        ff_dataframe_unique_visible_ff = ff_dataframe_visible[['bin', 'ff_index']]
        ff_dataframe_unique_visible_ff = ff_dataframe_unique_visible_ff.groupby('bin').nunique().reset_index(drop=False)
        ff_dataframe_unique_visible_ff.rename(columns={'ff_index': 'num_visible_ff'}, inplace=True)
        self.binned_features = self.binned_features.merge(ff_dataframe_unique_visible_ff, how='left', on='bin')

        ## The below is currently not used because the whole column is 1
        # ff_dataframe_unique_visible_ff['any_ff_visible'] = 0
        # ff_dataframe_unique_visible_ff.loc[ff_dataframe_unique_visible_ff['num_visible_ff'] > 0, 'any_ff_visible'] = 1

        # min_ff_info
        #min_ff_info = ff_dataframe[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary', 'memory']]
        min_ff_info = ff_dataframe[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
        min_ff_info = min_ff_info.groupby('bin').min().reset_index(drop=False)
        min_ff_info.rename(columns={'ff_distance': 'min_ff_distance',
                                    'ff_angle': 'min_abs_ff_angle',
                                    'ff_angle_boundary': 'min_abs_ff_angle_boundary'}, inplace=True)
                                    #'memory': 'min_ff_memory'}, inplace=True) # memory is currently not used bc the whole column is 100
        self.binned_features = self.binned_features.merge(min_ff_info, how='left', on='bin')


        # min_visible_ff_info
        min_visible_ff_info = ff_dataframe_visible[['bin', 'ff_distance', 'ff_angle', 'ff_angle_boundary']]
        min_visible_ff_info = min_visible_ff_info.groupby('bin').min().reset_index(drop=False)
        min_visible_ff_info.rename(columns={'ff_distance': 'min_visible_ff_distance',
                                            'ff_angle': 'min_abs_visible_ff_angle', 
                                            'ff_angle_boundary': 'min_abs_visible_ff_angle_boundary'}, inplace=True)
        self.binned_features = self.binned_features.merge(min_visible_ff_info, how='left', on='bin')

        # mark bins where ff is caught
        self.binned_features['catching_ff'] = 0
        catching_target_bins = np.digitize(self.ff_caught_T_sorted, self.time_bins)-1
        self.binned_features.loc[self.binned_features['bin'].isin(catching_target_bins), 'catching_ff'] = 1


        # dummy variable of whether any ff is visible
        # only add it if the ratio of bins with visible ff is between 10% and 90%, since otherwise it might not be so meaningful
        any_ff_visible = (self.binned_features['num_visible_ff'] > 0).astype(int)
        if (any_ff_visible.sum()/len(self.binned_features) > 0.1) and (any_ff_visible.sum()/len(self.binned_features) < 0.9):
            self.binned_features['any_ff_visible'] = self.binned_features['num_visible_ff'] > 0  
        self.binned_features = self.binned_features.fillna(method='ffill').reset_index(drop=True)
        self.binned_features = self.binned_features.fillna(method='bfill').reset_index(drop=True)


    def create_target_df(self):
        # Create self.target_df
        self.target_df = self.monkey_information[['bin', 'monkey_t', 'monkey_x', 'monkey_y', 'monkey_angles']].copy()
        self.target_df.rename(columns={'monkey_angles': 'monkey_angle', 'monkey_t': 'time'}, inplace=True)
        self.target_df['point_index'] = self.target_df.index
        self.target_df['target_index'] = np.searchsorted(self.ff_caught_T_sorted, self.target_df['time'])
        self.target_df['target_x'] = self.ff_real_position_sorted[self.target_df['target_index'].values, 0]
        self.target_df['target_y'] = self.ff_real_position_sorted[self.target_df['target_index'].values, 1]
        # find nearby_alive_ff_indices (a.k.a target cluster)
        self.nearby_alive_ff_indices = cluster_analysis.find_target_clusters(self.ff_real_position_sorted, self.ff_caught_T_sorted, \
                                                                                        self.ff_life_sorted, max_distance=50)

    def add_target_last_seen_info(self):
        # Add target-last-seen info to self.target_df
        self.target_df = organize_patterns_and_features.add_target_last_seen_info_to_target_df(self.target_df, self.ff_dataframe, self.nearby_alive_ff_indices, use_target_cluster=False, include_frozen_info=True)
        self.target_df['point_index'] = self.target_df['point_index'].astype(int)

        # condense the information
        target_average_info = self.target_df[['bin', 'target_last_seen_time', 'target_last_seen_distance', 'target_last_seen_angle', 'target_last_seen_angle_to_boundary',\
                                            'target_last_seen_distance_frozen', 'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen']].copy()
        
        target_average_info = target_average_info.groupby('bin').mean().reset_index(drop=False)                                      
        self.binned_features = self.binned_features.merge(target_average_info, how='left', on='bin')


        # Add target_has_disappeared_for_last_time_dummy
        self.target_df['target_has_disappeared_for_last_time_dummy'] = 0
        # for each target
        for target in range(len(self.ff_caught_T_sorted)):
        # fine the time it disappears for the last time
            target_info = self.ff_dataframe[(self.ff_dataframe['ff_index']==target) & (self.ff_dataframe['visible']==1)]
            target_last_visible_time = target_info['time'].max()
            # then mark the period between that time and the capture time as 1
            self.target_df.loc[(self.target_df['target_index']==target) & (self.target_df['time'] > target_last_visible_time), 'target_has_disappeared_for_last_time_dummy'] = 1
        
        target_min_info = self.target_df[['bin', 'target_has_disappeared_for_last_time_dummy']].groupby('bin').min().reset_index(drop=False)                                   
        target_min_info.rename(columns={'target_has_disappeared_for_last_time_dummy': 'min_target_has_disappeared_for_last_time_dummy'}, inplace=True)   
        self.binned_features = self.binned_features.merge(target_min_info, how='left', on='bin')


    def add_target_cluster_last_seen_info(self):
        # Add some target last seen info to self.target_df
        # But first make sure there's no duplicate
        if 'target_cluster_last_seen_time' not in self.target_df.columns: # to avoid duplicate column
            # add target-cluster-last-seen info to self.target_df
            self.target_df = organize_patterns_and_features.add_target_last_seen_info_to_target_df(self.target_df, self.ff_dataframe, self.nearby_alive_ff_indices, use_target_cluster=True, include_frozen_info=True)
            self.target_df['point_index'] = self.target_df['point_index'].astype(int)
            self.target_df.rename(columns={'target_last_seen_time': 'target_cluster_last_seen_time', 
                                            'target_last_seen_distance': 'target_cluster_last_seen_distance', 
                                            'target_last_seen_angle': 'target_cluster_last_seen_angle', 
                                            'target_last_seen_angle_to_boundary': 'target_cluster_last_seen_angle_to_boundary',
                                            'target_last_seen_distance_frozen': 'target_cluster_last_seen_distance_frozen', 
                                            'target_last_seen_angle_frozen': 'target_cluster_last_seen_angle_frozen', 
                                            'target_last_seen_angle_to_boundary_frozen': 'target_cluster_last_seen_angle_to_boundary_frozen',
                                        }, inplace=True)
        
        target_average_info = self.target_df[['bin','target_cluster_last_seen_time', 'target_cluster_last_seen_distance', 'target_cluster_last_seen_angle', 'target_cluster_last_seen_angle_to_boundary',\
                                                  'target_cluster_last_seen_distance_frozen', 'target_cluster_last_seen_angle_frozen', 'target_cluster_last_seen_angle_to_boundary_frozen',]].copy()  
        target_average_info = target_average_info.groupby('bin').mean().reset_index(drop=False)    
        self.binned_features = self.binned_features.merge(target_average_info, how='left', on='bin')

        self.target_df['target_cluster_has_disappeared_for_last_time_dummy'] = 0
        for target in range(len(self.ff_caught_T_sorted)):
            target_cluster_indices = self.nearby_alive_ff_indices[target] 
            target_info = self.ff_dataframe[(self.ff_dataframe['ff_index'].isin(target_cluster_indices)) & (self.ff_dataframe['visible']==1)]
            target_last_visible_time = target_info['time'].max()
            # then mark the period between that time and the capture time as 1
            self.target_df.loc[(self.target_df['target_index']==target) & (self.target_df['time'] > target_last_visible_time), 'target_cluster_has_disappeared_for_last_time_dummy'] = 1

        target_min_info = self.target_df[['bin', 'target_cluster_has_disappeared_for_last_time_dummy']].groupby('bin').min().reset_index(drop=False)                                   
        target_min_info.rename(columns={'target_cluster_has_disappeared_for_last_time_dummy': 'min_target_cluster_has_disappeared_for_last_time_dummy'}, inplace=True)   
        self.binned_features = self.binned_features.merge(target_min_info, how='left', on='bin')



    def add_target_info(self):
        # calculate target_distance, target_angle, and target_angle_to_boundary based on self.target_df['target_x'], self.target_df['target_y'] in self.target_df
        target_distance = np.sqrt((self.target_df['target_x'] - self.target_df['monkey_x'])**2 + (self.target_df['target_y'] - self.target_df['monkey_y'])**2)
        target_angle = basic_func.calculate_angles_to_ff_centers(ff_x=self.target_df['target_x'], ff_y=self.target_df['target_y'], mx=self.target_df['monkey_x'], my=self.target_df['monkey_y'], m_angle=self.target_df['monkey_angle'])
        self.target_df['target_distance'] = target_distance
        self.target_df['target_angle'] = target_angle
        self.target_df['target_angle_to_boundary'] = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=target_angle, distances_to_ff=target_distance)

        # find time_since_last_capture; since self.target_df contains more time than self.ff_caught_T_sorted, we need to add a value to the end of self.ff_caught_T_sorted
        if self.target_df.target_index.unique().max() >= len(self.ff_caught_T_sorted)-1:
            num_exceeding_target = self.target_df.target_index.unique().max() - (len(self.ff_caught_T_sorted)-1)
            self.ff_caught_T_sorted_temp = np.concatenate((self.ff_caught_T_sorted, np.repeat(self.target_df.time.max(), num_exceeding_target)))
        else:
            self.ff_caught_T_sorted_temp = self.ff_caught_T_sorted.copy()
        last_target_caught_time = self.ff_caught_T_sorted_temp[self.target_df['target_index']-1]
        last_target_caught_time[0] = 0
        self.target_df['time_since_last_capture'] = self.target_df['time'] - last_target_caught_time


        target_average_info = self.target_df[['bin', 'target_distance', 'target_angle', 'target_angle_to_boundary', 'time_since_last_capture']].copy()
        target_average_info = target_average_info.groupby('bin').mean().reset_index(drop=False)    
        self.binned_features = self.binned_features.merge(target_average_info, how='left', on='bin')


        self.target_df[['target_visible_dummy']] = 1
        ff_dataframe_temp = self.ff_dataframe.copy()
        ff_dataframe_temp = ff_dataframe_temp[ff_dataframe_temp['ff_index']==ff_dataframe_temp['target_index']]
        ff_dataframe_temp = ff_dataframe_temp[ff_dataframe_temp['visible']==1]
        target_visible_point_index = ff_dataframe_temp.point_index.values
        self.target_df.loc[self.target_df['point_index'].isin(target_visible_point_index), 'target_visible_dummy'] = 0
        target_max_info = self.target_df[['bin', 'target_visible_dummy']].groupby('bin').max().reset_index(drop=False)                                   
        target_max_info.rename(columns={'target_visible_dummy': 'max_target_visible_dummy'}, inplace=True)   
        self.binned_features = self.binned_features.merge(target_max_info, how='left', on='bin')


    def add_target_cluster_info(self):

        # dummy variable of target cluster being visible (over the full span of the bin)
        if 'target_cluster_last_seen_time' not in self.target_df.columns: # to avoid duplicate column
            # add target-cluster-last-seen info to self.target_df
            self.target_df = organize_patterns_and_features.add_target_last_seen_info_to_target_df(self.target_df, self.ff_dataframe, self.nearby_alive_ff_indices, use_target_cluster=True, include_frozen_info=True)
            self.target_df.rename(columns={'target_last_seen_time': 'target_cluster_last_seen_time'}, inplace=True)
        self.target_df[['target_cluster_visible_dummy']] = 1
        self.target_df.loc[self.target_df['target_cluster_last_seen_time'] > 0, 'target_cluster_visible_dummy'] = 0
        
        # Add to self.target_df dummy variable of being in the last duration of seeing the target cluster
        self.target_df['while_last_seeing_target_cluster'] = 0
        for target in self.target_df.target_index.unique():
            target_subset = self.target_df[self.target_df.target_index == target]
            if len(target_subset) > 0:
                dif = np.diff(target_subset['target_cluster_visible_dummy'])
                becoming_visible_points = np.where(dif == 1)[0]
                if len(becoming_visible_points) > 0:
                    starting_index = becoming_visible_points[-1]+1
                elif target_subset['target_cluster_visible_dummy'].iloc[0] == 1: # the target has been visible throughout the duration of the trial
                    starting_index = 0
                else: # the target has not become visible throughout the duration of the trial
                    continue
                stop_being_visible_points = np.where(dif == -1)[0]
                if len(stop_being_visible_points) > 0:
                    ending_index = stop_being_visible_points[-1]+1
                    if ending_index < starting_index:
                        ending_index = len(target_subset)
                else:
                    ending_index = len(target_subset)
                self.target_df.loc[target_subset.iloc[starting_index:ending_index].index, 'while_last_seeing_target_cluster'] = 1
                

        target_max_info = self.target_df[['bin', 'target_cluster_visible_dummy', 'while_last_seeing_target_cluster']].groupby('bin').max().reset_index(drop=False)                                   
        target_max_info.rename(columns={'target_cluster_visible_dummy': 'max_target_cluster_visible_dummy',
                                        'while_last_seeing_target_cluster': 'max_while_last_seeing_target_cluster'}, inplace=True)   
        self.binned_features = self.binned_features.merge(target_max_info, how='left', on='bin')

    def add_all_target_info(self):
        self.create_target_df()
        self.add_target_info()
        self.add_target_last_seen_info()
        self.add_target_cluster_last_seen_info()
        self.add_target_cluster_info()


    def add_pattern_info_base_on_points(self):
        pattern_df = self.monkey_information[['bin']].copy()
        pattern_df['point_index'] = pattern_df.index
        
        pattern_df['try_a_few_times_indice_dummy'] = 0
        pattern_df.loc[self.try_a_few_times_indices_for_anim, 'try_a_few_times_indice_dummy'] = 1
        pattern_df['give_up_after_trying_indice_dummy'] = 0
        pattern_df.loc[self.GUAT_point_indices_for_anim, 'give_up_after_trying_indice_dummy'] = 1
        pattern_df['ignore_sudden_flas_indice_dummy'] = 0
        pattern_df.loc[self.ignore_sudden_flash_indices_for_anim, 'ignore_sudden_flas_indice_dummy'] = 1

        pattern_df_condensed = pattern_df[['bin', 'try_a_few_times_indice_dummy', 'give_up_after_trying_indice_dummy',
                                        'ignore_sudden_flas_indice_dummy']].copy()
        pattern_df_condensed = pattern_df_condensed.groupby('bin').max().reset_index(drop=False) 
        self.binned_features = self.binned_features.merge(pattern_df_condensed, how='left', on='bin')
        self.binned_features = self.binned_features.fillna(method='ffill').reset_index(drop=True)
        self.binned_features = self.binned_features.fillna(method='bfill').reset_index(drop=True)



    def add_pattern_info_based_on_trials(self):
        # add the category info based on trials
        self.all_trial_patterns['trial_end_time'] = self.ff_caught_T_sorted[:len(self.all_trial_patterns)]
        self.all_trial_patterns['trial_start_time'] = self.all_trial_patterns['trial_end_time'].shift(1)
        self.all_trial_patterns.loc[0, 'trial_start_time'] = 0

        # find the centers of time_bins, which is the average of every two points
        bin_midlines = (self.time_bins[:-1] + self.time_bins[1:])/2
        bin_midlines = bin_midlines[bin_midlines < self.ff_caught_T_sorted[-1]]
        bin_midlines = pd.DataFrame(bin_midlines, columns=['bin_midline'])
        bin_midlines['trial'] = np.searchsorted(self.ff_caught_T_sorted, bin_midlines['bin_midline'])
        self.all_trial_patterns['trial'] = self.all_trial_patterns.index
        bin_midlines = bin_midlines.merge(self.all_trial_patterns, on='trial', how='left')
        bin_midlines['bin'] = bin_midlines.index
        self.binned_features = self.binned_features.merge(bin_midlines[['bin', 'two_in_a_row', 'visible_before_last_one',
            'disappear_latest', 'ignore_sudden_flash', 'try_a_few_times',
            'give_up_after_trying', 'cluster_around_target',
            'waste_cluster_around_target']], on = 'bin', how='left')
        
        self.binned_features = self.binned_features.fillna(method='ffill').reset_index(drop=True)
        self.binned_features = self.binned_features.fillna(method='bfill').reset_index(drop=True)


    def combine_everything(self, window_width=1, bin_width=0.25):
        self.make_bins(window_width=window_width, bin_width=bin_width)
        self.add_monkey_info()    
        self.add_all_target_info()   
        self.add_pattern_info_base_on_points()
        self.add_pattern_info_based_on_trials()
        return self.binned_features
        



