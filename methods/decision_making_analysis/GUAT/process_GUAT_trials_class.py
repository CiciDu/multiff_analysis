import sys
from decision_making_analysis.GUAT import GUAT_and_TAFT, GUAT_utils
from pattern_discovery import cluster_analysis
from visualization import plot_trials


import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd




plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)







class ProcessGUATtrials:

    plotting_kwargs = {'player': 'monkey',
                        'show_stops': True,
                        'show_believed_target_positions': True,
                        'show_reward_boundary': True,
                        'show_scale_bar': True,
                        'hitting_arena_edge_ok': True,
                        'trial_too_short_ok': True,
                        'show_connect_path_ff': True,
                        'vary_color_for_connecting_path_ff': True,
                        'show_connect_path_ff_memory': True,
                        'show_alive_fireflies': False,
                        'show_visible_fireflies': True,
                        'show_in_memory_fireflies': True,
                        'connect_path_ff_max_distance': 400}

        

    def __init__(self, give_up_after_trying_info_bundle, PlotTrials_args, max_distance_to_stop_for_GUAT_target=50, max_allowed_time_since_last_vis=2.5) -> None:
        
        self.give_up_after_trying_info_bundle = give_up_after_trying_info_bundle
        self.give_up_after_trying_trials, self.GUAT_point_indices_for_anim, self.GUAT_indices_df, self.GUAT_trials_df = give_up_after_trying_info_bundle
        self.give_up_after_trying_indices = self.GUAT_indices_df['point_index'].values
        self.GUAT_cluster_index = np.unique(self.GUAT_indices_df['cluster_index'].values)
        
        self.PlotTrials_args = PlotTrials_args
        self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, \
                self.cluster_around_target_indices, self.ff_caught_T_new = self.PlotTrials_args

        self.max_distance_to_stop_for_GUAT_target = max_distance_to_stop_for_GUAT_target 
        self.max_allowed_time_since_last_vis = max_allowed_time_since_last_vis


    def find_GUAT_ff_aimed_at_from_manual_anno(self, manual_anno):
        self.manual_anno = manual_anno
        GUAT_ff_aimed_at_from_manual_anno = []
        unique_stopping_clusters = np.unique(self.GUAT_indices_df['cluster_index'].values) # actually this should be unnecessary, since cluster should go from 0 to n continously
        # for each cluster
        for i in range(len(unique_stopping_clusters)):
            cluster = unique_stopping_clusters[i]
            relevant_info = np.where(self.GUAT_indices_df['cluster_index'].values == cluster)[0]
            relevant_indices = self.GUAT_indices_df['point_index'].values[relevant_info]            
            ff_aimed_at = GUAT_and_TAFT.find_ff_aimed_at_through_manual_annotation(relevant_indices, self.monkey_information, self.manual_anno)  
            GUAT_ff_aimed_at_from_manual_anno.append(ff_aimed_at)   
        self.GUAT_ff_aimed_at_from_manual_anno = np.array(GUAT_ff_aimed_at_from_manual_anno, dtype=object) 


    def find_possible_objects_of_pursuit_in_GUAT(self):
        self.GUAT_w_ff_df, self.GUAT_expanded_trials_df = GUAT_utils.get_GUAT_w_ff_df(self.GUAT_indices_df,
                                                                            self.GUAT_trials_df,
                                                                            self.ff_dataframe,
                                                                            self.monkey_information,
                                                                            self.ff_real_position_sorted,                                                                            
                                                                            max_distance_to_stop_for_GUAT_target=self.max_distance_to_stop_for_GUAT_target,
                                                                            max_allowed_time_since_last_vis=self.max_allowed_time_since_last_vis)


    def make_GUAT_plot(self, trial, ff_near_stops, relevant_indices, additional_kwargs=None):
        

        # change default figsize
        plt.rcParams['figure.figsize'] = [10, 10]
        num_trials = 2

        plotting_kwargs_temp = self.plotting_kwargs.copy()

        if len(ff_near_stops) > 0:
            plotting_kwargs_temp['indices_of_ff_to_mark'] = ff_near_stops

        if additional_kwargs is not None:
            for key, value in additional_kwargs.items():
                plotting_kwargs_temp[key] = value

        duration = [self.ff_caught_T_new[trial-num_trials], self.ff_caught_T_new[trial]]
        returned_info = plot_trials.PlotTrials(duration, 
                    *self.PlotTrials_args,
                    **plotting_kwargs_temp,
                    currentTrial = trial,
                    num_trials = num_trials,                   
                    )
        
        R = returned_info['R']
        axes = returned_info['axes']

        if R is not None:
            # plot circles around relevant_indices with radius of self.max_distance_to_stop_for_GUAT_target
            temp_cum_mx, temp_cum_my = np.array(self.monkey_information['monkey_x'].loc[relevant_indices]), np.array(self.monkey_information['monkey_y'].loc[relevant_indices])
            if len(temp_cum_mx) > 0:
                temp_cum_mxy_rotated = np.matmul(R, np.stack((temp_cum_mx, temp_cum_my)))
                for i in range(len(temp_cum_mx)):
                    circle = plt.Circle((temp_cum_mxy_rotated[0, i], temp_cum_mxy_rotated[1, i]), self.max_distance_to_stop_for_GUAT_target, facecolor='yellow', edgecolor='red', alpha=0.01, zorder=1)
                    axes.add_patch(circle)

        plt.show()
        


    def check_GUAT_object_with_manual_anno(self, verbose=True):
        
        clusters_w_o_ff_aimed_at = []
        clusters_w_o_ff_near_stops = []
        clusters_w_o_matching_ff_near_stops = []
        clusters_w_matching_ff_near_stops = []

        for i in range(len(self.GUAT_expanded_trials_df.cluster_index.values)):
            counter = self.GUAT_expanded_trials_df.cluster_index.values[i]
            relevant_info = np.where(self.GUAT_indices_df['cluster_index'].values == counter)[0]
            relevant_indices = self.GUAT_indices_df['point_index'].values[relevant_info]

            trial = self.GUAT_expanded_trials_df.trial_index.values[i]
            ff_near_stops = self.GUAT_expanded_trials_df['nearby_alive_ff_indices'].iloc[i]
            ff_aimed_at = self.GUAT_ff_aimed_at_from_manual_anno[i]

            if len(ff_near_stops) > 0:
                # do they match?
                if len(ff_aimed_at) > 0:
                    flag = False
                    if len(ff_aimed_at) > 0:
                        for ff in ff_aimed_at:
                            if ff in ff_near_stops:
                                if verbose:
                                    print("Trial %d: Matched! Aimed at ff %d, which is near stops!" % (trial, ff))
                                flag = True
                                clusters_w_matching_ff_near_stops.append(counter)
                    if not flag:
                        if verbose:
                            print("Trial %d: no matching ff near stops!" %(trial))
                        clusters_w_o_matching_ff_near_stops.append(counter)

                else:
                    if verbose:
                        print("Trial %d: no ff aimed at!" %(trial))
                    clusters_w_o_ff_aimed_at.append(counter)
            else:
                if verbose:
                    print("Trial %d: no ff near stops!" %(trial))
                clusters_w_o_ff_near_stops.append(counter)


        self.clusters_w_matching_ff_near_stops = np.array(clusters_w_matching_ff_near_stops)
        self.clusters_w_o_matching_ff_near_stops = np.array(clusters_w_o_matching_ff_near_stops)
        self.clusters_w_o_ff_aimed_at = np.array(clusters_w_o_ff_aimed_at)
        self.clusters_w_o_ff_near_stops = np.array(clusters_w_o_ff_near_stops)
        

    def inspect_clusters_w_o_matching_ff(self, verbose=True, make_plots=True, max_plots=5):
        # or one can use this to check other categories too
        plot_counter = 0
        for cluster in self.clusters_w_o_matching_ff_near_stops:
            if plot_counter >= max_plots:
                break
            plot_counter += 1
            cluster_index = np.where(self.GUAT_expanded_trials_df.cluster_index.values == cluster)[0][0]
            relevant_info = np.where(self.GUAT_indices_df['cluster_index'].values == cluster)[0]
            relevant_indices = self.GUAT_indices_df['point_index'].values[relevant_info]
            trial = self.GUAT_indices_df['trial'].values[relevant_info][0]
            ff_near_stops = self.GUAT_expanded_trials_df['nearby_alive_ff_indices'].iloc[cluster_index] 
            ff_aimed_at = np.array(self.GUAT_ff_aimed_at_from_manual_anno[cluster_index])
            if verbose:
                print("ff_aimed_at", ff_aimed_at)
            ff_aimed_at = np.unique(ff_aimed_at[ff_aimed_at > -1])
            if make_plots:
                additional_kwargs = {'show_connect_path_ff_specific_indices': np.unique(np.concatenate([ff_near_stops, ff_aimed_at])),
                                    'indices_of_ff_to_mark_2nd_kind': ff_aimed_at}
                self.make_GUAT_plot(trial, ff_near_stops, relevant_indices, additional_kwargs=additional_kwargs)


    def make_GUAT_cluster_df(self):
        ff_indices_of_each_cluster = self.GUAT_w_ff_df['nearby_alive_ff_indices'].values
        GUAT_last_stop_time = self.GUAT_w_ff_df['last_stop_time'].values

        self.GUAT_cluster_df = cluster_analysis.find_ff_cluster_last_vis_df(ff_indices_of_each_cluster, GUAT_last_stop_time, ff_dataframe=self.ff_dataframe, cluster_identifiers=self.GUAT_w_ff_df['cluster_index'].values)

        self.GUAT_cluster_df.rename(columns={'cluster_identifier': 'cluster_index'}, inplace=True)
        
        self.GUAT_cluster_df = self.GUAT_cluster_df.merge(self.GUAT_w_ff_df[['cluster_index', 'first_stop_time', 'second_stop_time', 'last_stop_time', 'first_stop_point_index',
                                                                                    'second_stop_point_index', 'last_stop_point_index', 'target_index', 'num_stops']],
                                                          on='cluster_index', how='left')

        # to prepare for free selection 
        self.GUAT_cluster_df['latest_visible_time_before_last_stop'] = self.GUAT_cluster_df['last_stop_time'] - self.GUAT_cluster_df['time_since_last_vis']

        # sort by last_stop_time (note that the order in GUAT_cluster_df will henceforward be different from other variables)
        self.GUAT_cluster_df.sort_values(by='last_stop_time', inplace=True)
        


