import sys
from decision_making_analysis.decision_making import decision_making_utils, plot_decision_making
from decision_making_analysis import free_selection
from data_wrangling import basic_func
from visualization import plot_trials, plot_behaviors_utils
from pattern_discovery import pattern_by_points


import os
from math import pi
import os.path
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from numpy import linalg as LA
import seaborn as sns




plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)




class ProcessCategoryData:

    def __init__(self, PlotTrials_args, ff_flash_sorted,
                 sort_1_trials, sort_1_name, sort_1_df,
                 sort_2_trials, sort_2_name, sort_2_df, 
                 sort_1_ff_indices=None, sort_2_ff_indices=None,
                 sort_1_ff_positions=None, sort_2_ff_positions=None,
                 sort_1_trials_ending_time=None, sort_2_trials_ending_time=None,
                 sort_1_time_for_predicting_ff=None, sort_2_time_for_predicting_ff=None):
        
        # sort_1_trials_ending_time & sort_2_trials_ending_time: these don't have to be the real ending time, but just a time where a trial is *considered* over


        self.ff_flash_sorted = ff_flash_sorted
        self.PlotTrials_args = PlotTrials_args
        (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, 
                                  self.cluster_around_target_indices, self.ff_caught_T_new) = PlotTrials_args

        self.sort_1_name = sort_1_name
        self.sort_2_name = sort_2_name


        self.sort_1_trials = sort_1_trials
        self.sort_2_trials = sort_2_trials


        self.sort_1_df = sort_1_df
        self.sort_2_df = sort_2_df


        self.sort_1_ff_indices = sort_1_ff_indices
        if self.sort_1_ff_indices is None:
            self.sort_1_ff_indices = self.sort_1_trials


        self.sort_2_ff_indices = sort_2_ff_indices
        if self.sort_2_ff_indices is None:
            self.sort_2_ff_indices = self.sort_2_trials


        self.sort_1_ff_positions = sort_1_ff_positions
        if self.sort_1_ff_positions is None:
            self.sort_1_ff_positions = self.ff_real_position_sorted[self.sort_1_ff_indices]
            
        self.sort_2_ff_positions = sort_2_ff_positions
        if self.sort_2_ff_positions is None:
            self.sort_2_ff_positions = self.ff_real_position_sorted[self.sort_2_ff_indices]


        self.sort_1_trials_ending_time  = sort_1_trials_ending_time
        if self.sort_1_trials_ending_time is None:
            self.sort_1_trials_ending_time = self.ff_caught_T_new[self.sort_1_trials]

        self.sort_2_trials_ending_time  = sort_2_trials_ending_time
        if self.sort_2_trials_ending_time is None:
            self.sort_2_trials_ending_time = self.ff_caught_T_new[self.sort_2_trials]


        self.sort_1_time_for_predicting_ff = sort_1_time_for_predicting_ff
        if self.sort_1_time_for_predicting_ff is None:
            self.sort_1_time_for_predicting_ff = self.ff_caught_T_new[self.sort_1_trials-1]

        self.sort_2_time_for_predicting_ff = sort_2_time_for_predicting_ff
        if self.sort_2_time_for_predicting_ff is None:
            self.sort_2_time_for_predicting_ff = self.ff_caught_T_new[self.sort_2_trials-1]



        self.temp_plotting_kwargs = {'player': 'monkey',
                                    'show_alive_fireflies': False,
                                    'show_visible_fireflies': True,
                                    'show_in_memory_fireflies': True,
                                    'show_stops': True,
                                    'show_believed_target_positions': True,
                                    'show_reward_boundary': True,
                                    'show_scale_bar': True,
                                    'hitting_arena_edge_ok': True,
                                    'trial_too_short_ok': True,
                                    'show_legend': True,
                                    'vary_color_for_connecting_path_ff': True,
                                    'show_null_agent_trajectory': True,
                                    'minimal_margin': 50,
                                    'show_points_when_ff_stop_being_visible': False,
                                    'show_path_when_target_visible': True,
                                    'truncate_part_before_crossing_arena_edge': False}


    def clean_out_cross_boundary_trials(self, min_time_no_crossing_boundary=2.5):

        # Eliminate the cases where during the min_time_no_crossing_boundary up to catching the target, the monkey has crossed boundary
        crossing_boundary_trials = decision_making_utils.find_crossing_boundary_trials(self.sort_1_trials, self.sort_1_trials_ending_time, self.monkey_information, min_time_no_crossing_boundary)
        print("There are", str(len(crossing_boundary_trials)), "trials in", self.sort_1_name, "out of", str(len(self.sort_1_trials)), "that the monkey crossed the boundary before catching the firefly.")
        # find the difference between crossing_boundary_trials and self.sort_1_df.ff_index
        remaining_sort_1_trials = np.setdiff1d(self.sort_1_trials, crossing_boundary_trials)
        new_indices = np.where(np.in1d(self.sort_1_df['target_index'], remaining_sort_1_trials))[0]
        self.sort_1_df = self.sort_1_df.iloc[new_indices]
        self.sort_1_trials = self.sort_1_trials[new_indices]
        self.sort_1_ff_indices = self.sort_1_ff_indices[new_indices]
        self.sort_1_ff_positions = self.sort_1_ff_positions[new_indices]
        self.sort_1_trials_ending_time = self.sort_1_trials_ending_time[new_indices]
        self.sort_1_time_for_predicting_ff = self.sort_1_time_for_predicting_ff[new_indices]

     

        # for those that are not in visible_before_last_one trials, similarly eliminate crossing-boundary cases
        crossing_boundary_trials_2 = decision_making_utils.find_crossing_boundary_trials(self.sort_2_trials, self.sort_2_trials_ending_time, self.monkey_information, min_time_no_crossing_boundary)
        print("There are", str(len(crossing_boundary_trials_2)), "trials in", self.sort_2_name, "out of", str(len(self.sort_2_trials)), "that the monkey crossed the boundary before catching the firefly.")
        # find the difference between crossing_boundary_trials and self.sort_1_df.ff_index
        remaining_sort_2_trials = np.setdiff1d(self.sort_2_trials, crossing_boundary_trials_2)
        new_indices = np.where(np.in1d(self.sort_2_df['target_index'], remaining_sort_2_trials))[0]
        self.sort_2_df = self.sort_2_df.iloc[new_indices]
        self.sort_2_trials = self.sort_2_trials[new_indices]
        self.sort_2_ff_indices = self.sort_2_ff_indices[new_indices]
        self.sort_2_ff_positions = self.sort_2_ff_positions[new_indices]
        self.sort_2_trials_ending_time = self.sort_2_trials_ending_time[new_indices]
        self.sort_2_time_for_predicting_ff = self.sort_2_time_for_predicting_ff[new_indices]



    def clean_out_trials_where_target_cluster_was_not_seen_for_a_long_time_before_capture(self, max_not_seen_time=3):
        # Eliminate the cases where during the max_not_seen_time up to catching the target, the monkey has not been visible to the monkey
        original_length = len(self.sort_1_df)
        new_indices = np.where(self.sort_1_df['time_since_last_vis'] < max_not_seen_time)[0]
        new_length = len(self.sort_1_df)
        self.sort_1_df = self.sort_1_df.iloc[new_indices]
        self.sort_1_trials = self.sort_1_trials[new_indices]
        self.sort_1_ff_indices = self.sort_1_ff_indices[new_indices]
        self.sort_1_ff_positions = self.sort_1_ff_positions[new_indices]
        self.sort_1_trials_ending_time = self.sort_1_trials_ending_time[new_indices]
        self.sort_1_time_for_predicting_ff = self.sort_1_time_for_predicting_ff[new_indices]
        print(new_length, "out of", original_length, "left of ", self.sort_1_name, "remained after cleaning out trials where target cluster was not seen for a long time before capture.")


        original_length = len(self.sort_2_df)
        new_indices = np.where(self.sort_2_df['time_since_last_vis'] < max_not_seen_time)[0]
        new_length = len(self.sort_2_df)
        self.sort_2_df = self.sort_2_df.iloc[new_indices]
        self.sort_2_trials = self.sort_2_trials[new_indices]
        self.sort_2_ff_indices = self.sort_2_ff_indices[new_indices]
        self.sort_2_ff_positions = self.sort_2_ff_positions[new_indices]
        self.sort_2_trials_ending_time = self.sort_2_trials_ending_time[new_indices]
        self.sort_2_time_for_predicting_ff = self.sort_2_time_for_predicting_ff[new_indices]
        print(new_length, "out of", original_length, "left of ", self.sort_2_name, "remained after cleaning out trials where target cluster was not seen for a long time before capture.")
                    


    def make_polar_plot_of_target_last_seen_positions(self): 

        min_sample_size = min(self.sort_1_df.shape[0], self.sort_2_df.shape[0])
        self.sort_1_df_sample = self.sort_1_df.sample(n=min_sample_size, replace=False)
        self.sort_2_df_sample = self.sort_2_df.sample(n=min_sample_size, replace=False)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax = plot_behaviors_utils.set_polar_background_for_plotting(ax, 400, color_visible_area_in_background=True)
        ax.scatter(self.sort_1_df_sample['last_vis_ang'], self.sort_1_df_sample['last_vis_dist'], c="green", alpha=0.7, zorder=2, s=5, marker='o') # originally it was s=15
        # sample from it so the size is the same as target_cluster_info
        ax.scatter(self.sort_2_df_sample['last_vis_ang'], self.sort_2_df_sample['last_vis_dist'], c="red", alpha=0.4, zorder=2, s=5, marker='o') # originally it was s=15
        plt.title("Firefly Last Seen Positions", fontsize=17)
        plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
        plt.show()
            

    

    def make_histograms_of_target_last_seen_attributes(self):
        sns.set_style(style="darkgrid")


        variable_of_interest = "time_since_last_vis"
        if (variable_of_interest in self.sort_1_df.columns) & (variable_of_interest in self.sort_2_df.columns):
            fig, axes = plt.subplots()
            sns.histplot(data = self.sort_1_df[variable_of_interest], kde = False, alpha = 0.4, color = "green", binwidth = 0.1, stat="probability")
            sns.histplot(data = self.sort_2_df[variable_of_interest], kde = False, alpha = 0.4, color = "blue", binwidth = 0.1, stat="probability")
            axes.set_title("Time Since Firefly Last Visible", fontsize=17)
            axes.set_xlim([0, 6])
            axes.set_xlabel("Time (s)", fontsize = 15)
            plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
            plt.show()
            

        variable_of_interest = "abs_last_vis_ang"
        if (variable_of_interest in self.sort_1_df.columns) & (variable_of_interest in self.sort_2_df.columns):
            fig, axes = plt.subplots()
            sns.histplot(data = self.sort_1_df[variable_of_interest], kde = False, binwidth=0.02, alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
            sns.histplot(data = self.sort_2_df[variable_of_interest], kde = False,  binwidth=0.02, alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
            axes.set_title("Abs Angle of Firefly Last Visible", fontsize=17)
            axes.set_xlabel("Angle (rad)", fontsize=15)
            axes.set_xticks(np.arange(0.0, 0.9, 0.2))
            axes.set_xticks(axes.get_xticks())
            axes.set_xticklabels(np.arange(0.0, 0.9, 0.2).round(1))
            axes.set_xlim(0, 0.7)
            plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
            plt.show()
            

        variable_of_interest = "abs_last_vis_ang_to_bndry"
        if (variable_of_interest in self.sort_1_df.columns) & (variable_of_interest in self.sort_2_df.columns):
            fig, axes = plt.subplots()
            sns.histplot(data = self.sort_1_df[variable_of_interest], kde = False, binwidth=0.02, alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
            sns.histplot(data = self.sort_2_df[variable_of_interest], kde = False,  binwidth=0.02, alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
            axes.set_title("Abs Angle to Boundary of Firefly Last Visible", fontsize=17)
            axes.set_xlabel("Angle (rad)", fontsize=15)
            axes.set_xticks(np.arange(0.0, 0.9, 0.2))
            axes.set_xticks(axes.get_xticks())
            axes.set_xticklabels(np.arange(0.0, 0.9, 0.2).round(1))
            axes.set_xlim(0, 0.7)
            plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
            plt.show()
            


        variable_of_interest = "last_vis_dist"
        if (variable_of_interest in self.sort_1_df.columns) & (variable_of_interest in self.sort_2_df.columns):
            fig, axes = plt.subplots()
            sns.histplot(data = self.sort_1_df[variable_of_interest], kde = False, alpha = 0.3,  color = "green", binwidth=10, stat="probability",  edgecolor='grey')
            sns.histplot(data = self.sort_2_df[variable_of_interest], kde = False, alpha = 0.3,  color = "blue", binwidth=10, stat="probability",  edgecolor='grey')
            axes.set_xlim(0, 400)
            axes.set_title("Distance of Firefly Last Visible", fontsize=17)
            axes.set_xlabel("Distance", fontsize=15)
            xticklabels=axes.get_xticks().tolist()
            xticklabels = [str(int(label)) for label in xticklabels]
            xticklabels[-1]='400+'
            axes.set_xticks(axes.get_xticks())
            axes.set_xticklabels(xticklabels)
            plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
            plt.show()
            


 

    def make_histogram_of_distances_from_previous_targets(self):
        # distance between sort_1 targets from previous targets
        valid_indices_1 = np.where(self.sort_1_trials >= 1)[0]
        self.sort_1_to_prev_target_distances = LA.norm(self.sort_1_ff_positions[valid_indices_1] - self.ff_real_position_sorted[self.sort_1_trials-1][valid_indices_1], axis=1)

        # distance between non-sort_1 targets from previous targets
        valid_indices_2 = np.where(self.sort_2_trials > 0)[0]
        self.sort_2_to_prev_target_distances = LA.norm(self.sort_2_ff_positions[valid_indices_2] - self.ff_real_position_sorted[self.sort_2_trials-1][valid_indices_2], axis=1)


        fig, axes = plt.subplots()
        sns.histplot(data = self.sort_1_to_prev_target_distances, kde = False, alpha = 0.3,  color = "green", binwidth=10, stat="probability",  edgecolor='grey')
        sns.histplot(data = self.sort_2_to_prev_target_distances, kde = False, alpha = 0.3,  color = "blue", binwidth=10, stat="probability",  edgecolor='grey')
        axes.set_xlim(0, 400)
        axes.set_title("Distance from previous target", fontsize=17)
        axes.set_xlabel("Distance", fontsize=15)
        xticklabels=axes.get_xticks().tolist()
        xticklabels = [str(int(label)) for label in xticklabels]
        xticklabels[-1]='400+'
        axes.set_xticks(axes.get_xticks())
        axes.set_xticklabels(xticklabels)
        plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
        plt.show()
        




    def make_polar_plot_of_positions_from_previous_targets(self):
        sns.set_style(style="white")

        min_sample_size = min(self.sort_1_df.shape[0], self.sort_2_df.shape[0])
        self.sort_1_sample_indices = np.random.choice(self.sort_1_df.shape[0], min_sample_size, replace=False)
        self.sort_2_sample_indices = np.random.choice(self.sort_2_df.shape[0], min_sample_size, replace=False)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax = plot_behaviors_utils.set_polar_background_for_plotting(ax, 400, color_visible_area_in_background=True)


        prev_target_caught_T = self.ff_caught_T_new[self.sort_1_trials-1][self.sort_1_sample_indices]
        target_distances, target_angles = decision_making_utils.get_distance_and_angle_from_previous_target(self.sort_1_ff_positions[self.sort_1_sample_indices], prev_target_caught_T, self.monkey_information)
        ax.scatter(target_angles, target_distances, c="green", alpha=0.7, zorder=2, s=5, marker='o') # originally it was s=15

        prev_target_caught_T = self.ff_caught_T_new[self.sort_2_trials-1][self.sort_2_sample_indices]
        target_distances, target_angles = decision_making_utils.get_distance_and_angle_from_previous_target(self.sort_2_ff_positions[self.sort_2_sample_indices], prev_target_caught_T, self.monkey_information)
        ax.scatter(target_angles, target_distances, c="red", alpha=0.7, zorder=2, s=5, marker='o') # originally it was s=15
        plt.legend(labels=[self.sort_1_name, self.sort_2_name], fontsize=11, loc="upper right")
        plt.title("Positions From Previous Targets", fontsize=17) 
        plt.show()
            




    def plot_trajectories(self, trials, **kwargs):

        temp_plotting_kwargs = self.temp_plotting_kwargs
        for key, value in kwargs.items():
            temp_plotting_kwargs[key] = value
        

        if isinstance(trials, int):
            trials = [trials]

        ## By monkey
        # plot the chunks
        for trial in trials:
            with basic_func.initiate_plot(7, 7, 100):
                duration = [self.ff_caught_T_new[trial-1]-2, self.ff_caught_T_new[trial]+0.01]
                print("duration", duration)

                temp_plotting_kwargs['null_agent_starting_time'] = self.ff_caught_T_new[trial-1]
                for i in range(1):
                    fig = plt.figure()
                    returned_info = plot_trials.PlotTrials(duration,
                            *self.PlotTrials_args,
                            **temp_plotting_kwargs,
                            fig=fig)

                    axes = returned_info['axes']
                    axes.set_aspect('equal')
                    axes.set_title('Trial ' + str(trial), fontsize=17)
                    plt.show()
                    



    def plot_distributions_of_visible_ff_and_in_memory_ff(self):

        # Compare box plots (or violin plots) of two cases:
        # After catching previous ff, how many ff are visible? How many are in memory?

        num_visible_ff, num_in_memory_ff = pattern_by_points.find_number_of_visible_or_in_memory_ff_at_beginning_of_trials(self.sort_1_trials-1, self.ff_caught_T_new, self.ff_dataframe)
        num_visible_ff_else, num_in_memory_ff_else = pattern_by_points.find_number_of_visible_or_in_memory_ff_at_beginning_of_trials(self.sort_2_trials-1, self.ff_caught_T_new, self.ff_dataframe)


        fig = plt.figure(figsize=(8, 6))
        num_ff_df = pd.DataFrame({
            "num_visible_ff": np.concatenate([num_visible_ff, num_visible_ff_else]),
            "type": [self.sort_1_name] * len(num_visible_ff) + [self.sort_2_name] * len(num_visible_ff_else)
        })
        # set a beautiful sns style without grid
        sns.set_style(style="white")
        sns.histplot(data=num_ff_df, x="num_visible_ff", hue="type", multiple="dodge", shrink=.8, stat="probability", binwidth=1, common_norm=False, discrete=True)
        # plot a legend
        plt.xlabel("Number of visible FF", fontsize=15)
        plt.legend(title=None, labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
        plt.show()
        


        fig = plt.figure(figsize=(8, 6))
        num_ff_df = pd.DataFrame({
            "num_in_memory_ff": np.concatenate([num_in_memory_ff, num_in_memory_ff_else]),
            "type": [self.sort_1_name] * len(num_in_memory_ff) + [self.sort_2_name] * len(num_in_memory_ff_else)
        })
        sns.histplot(data=num_ff_df, x="num_in_memory_ff", hue="type", multiple="dodge", shrink=.8, stat="probability", binwidth=1, common_norm=False, discrete=True)
        plt.xlabel("Number of FF in memory", fontsize=15)
        plt.legend(title=None, labels=[self.sort_1_name, self.sort_2_name], fontsize=13, loc="upper right")
        plt.show()
        
   



    def make_and_visualize_free_selection_predictions_using_trained_model(self, trained_model, use_sort_1=True, use_sort_2=False, max_plot_to_make=2,
                                                            sort_1_select_trials=None, sort_2_select_trials=None):
        # Note: select_sort_1_trials or select_sort_2_trials will override max_plot_to_make
        
        if use_sort_1:
            print("Predictions on free selection trials using the trained model: ", self.sort_1_name)
            self.sort_1_inputs, self.sort_1_labels, self.sort_1_y_pred = free_selection.make_free_selection_predictions_using_trained_model(trained_model, self.sort_1_ff_indices, self.sort_1_trials, self.ff_dataframe, 
                                                                                                                                            self.ff_real_position_sorted, self.ff_caught_T_new, self.monkey_information, time_of_evaluation=self.sort_1_time_for_predicting_ff)
             
            if sort_1_select_trials is not None:
            # find corresponding indices of select_trials in trials
                selected_cases = np.where(np.isin(self.sort_1_trials, sort_1_select_trials))[0]             
                                                                                                                                                              
            plot_decision_making.make_polar_plots_for_decision_making(ff_inputs=self.sort_1_inputs,
                                                                    labels=self.sort_1_labels,
                                                                    y_pred=self.sort_1_y_pred, 
                                                                    trials=self.sort_1_trials, # this is only for naming the title
                                                                    max_plot_to_make=max_plot_to_make,
                                                                    selected_cases=selected_cases)

        
        if use_sort_2:
            print("Predictions on free selection trials using the trained model: ", self.sort_2_name)
            self.sort_2_inputs, self.sort_2_labels, self.sort_2_y_pred = free_selection.make_free_selection_predictions_using_trained_model(trained_model, self.sort_2_ff_indices, self.sort_2_trials, self.ff_dataframe, self.ff_real_position_sorted, 
                                                                                         self.ff_caught_T_new, self.monkey_information, time_of_evaluation=self.sort_2_time_for_predicting_ff)

            if sort_2_select_trials is not None:
                selected_cases = np.where(np.isin(self.sort_2_trials, sort_2_select_trials))[0]
            plot_decision_making.make_polar_plots_for_decision_making(ff_inputs=self.sort_2_inputs,
                                                                    labels=self.sort_2_labels,
                                                                    y_pred=self.sort_2_y_pred, 
                                                                    trials=self.sort_2_trials, # this is only for naming the title
                                                                    max_plot_to_make=max_plot_to_make,
                                                                    selected_cases=selected_cases)




    def inspect_special_cases(self, weird_trials):
        try:
            num_trials = 2
            # plot the chunks

            #for trial in weird_trials:
            for trial in weird_trials:
                on_duration = self.ff_flash_sorted[trial][-1]
                # calculate ff_distance and ff_angle during the last flashing-on duration of the target
                cum_iloc_indices, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(on_duration, self.monkey_information)
                distances_to_ff = LA.norm(np.stack([cum_mx, cum_my], axis=1)-self.ff_real_position_sorted[trial], axis = 1)
                angles_to_ff = basic_func.calculate_angles_to_ff_centers(ff_x=self.ff_real_position_sorted[trial, 0], ff_y=self.ff_real_position_sorted[trial, 1], mx=cum_mx, my=cum_my, m_angle=cum_angle)
                angles_to_boundaries = basic_func.calculate_angles_to_ff_boundaries(angles_to_ff=angles_to_ff, distances_to_ff=distances_to_ff)


                print("distances_to_ff: \n", distances_to_ff)
                print(" ")
                print("angles_to_ff: \n", angles_to_ff*180/pi)

                corresponding_row = np.where(self.sort_1_trials==trial)[0][0]
                # also get the corresponding part in self.sort_1_inputs
                print("sort_1 input: row", corresponding_row, 
                        ", real label:", self.sort_1_labels[corresponding_row], "\n", 
                        ", predicted label", self.sort_1_y_pred[corresponding_row], "\n",
                        self.sort_1_inputs[corresponding_row].reshape(-1,3))


                with basic_func.initiate_plot(7, 7, 100):
                    duration = [self.ff_caught_T_new[trial]-3, self.ff_caught_T_new[trial]+0.01]
                    print("duration", duration)

                    for i in range(1):
                        fig = plt.figure()
                        returned_info = plot_trials.PlotTrials(duration,
                                *self.PlotTrials_args,
                                **self.temp_plotting_kwargs,
                                null_agent_starting_time = self.ff_caught_T_new[trial-1],
                                show_points_when_ff_stop_being_visible = False,
                                show_path_when_target_visible=True,
                                truncate_part_before_crossing_arena_edge=False,
                                fig=fig)
                        R = returned_info['R']
                        axes = returned_info['axes']

                        # draw a circle around the target
                        target_pos = self.ff_real_position_sorted[trial]
                        target_pos = np.matmul(R, target_pos.reshape(2,1))
                        x0, y0 = 0, 0
                        circle = plt.Circle((target_pos[0]-x0, target_pos[1]-y0), 400, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
                        axes.add_patch(circle)

                        # plot out the cum_mx and cum_my where the target flashes on for the last duration
                        cum_xy_rotated_2 = np.matmul(R, np.stack([cum_mx, cum_my], axis=1).T)
                        axes.scatter(cum_xy_rotated_2[0], cum_xy_rotated_2[1], c='blue', s=20,alpha=0.4, zorder=4)

                        # axes.set_xlim(-1000, 1000)
                        # axes.set_ylim(-1000, 1000)

                        axes.set_aspect('equal')
                        axes.set_title('Trial ' + str(trial), fontsize=17)
                        plt.show()
                        
        except AttributeError:
            print("Error! Please run the method free_selection.make_free_selection_predictions_using_trained_model first!")




    

