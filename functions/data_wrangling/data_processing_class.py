from multiff_analysis.functions.data_wrangling import basic_func, process_raw_data, make_ff_dataframe, find_patterns, analyze_patterns_and_features
from multiff_analysis.functions.data_visualization import animation_func, plot_behaviors

import os
import math
from math import pi
import re
import os.path
import neo
import pandas as pd
import numpy as np
import matplotlib
from functools import partial
from matplotlib import rc, animation
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists


class BaseProcessing:
    def __init__(self):
        pass



    def find_patterns(self):
        self.n_ff_in_a_row = find_patterns.n_ff_in_a_row_func(self.ff_believed_position_sorted, distance_between_ff = 50)
        self.two_in_a_row = np.where(self.n_ff_in_a_row==2)[0]
        self.two_in_a_row_simul, self.two_in_a_row_non_simul = find_patterns.whether_current_and_last_targets_are_captured_simultaneously(self.two_in_a_row, self.ff_caught_T_sorted)
        self.three_in_a_row= np.where(self.n_ff_in_a_row==3)[0]
        self.four_in_a_row= np.where(self.n_ff_in_a_row==4)[0]
        self.on_before_last_one_trials = find_patterns.on_before_last_one_func(self.ff_flash_end_sorted, self.ff_caught_T_sorted, self.caught_ff_num)
        self.on_before_last_one_simul, self.on_before_last_one_non_simul = find_patterns.whether_current_and_last_targets_are_captured_simultaneously(self.on_before_last_one_trials, self.ff_caught_T_sorted)
        self.visible_before_last_one_trials = find_patterns.visible_before_last_one_func(self.ff_dataframe)
        self.used_cluster = np.intersect1d(self.two_in_a_row_non_simul, self.visible_before_last_one_trials)
        self.disappear_latest_trials = find_patterns.disappear_latest_func(self.ff_dataframe)
        self.cluster_around_target_trials, self.cluster_around_target_indices, self.cluster_around_target_positions = find_patterns.cluster_around_target_func(self.ff_dataframe, self.caught_ff_num, self.ff_caught_T_sorted, self.ff_real_position_sorted, max_time_apart = 1.25)
        self.waste_cluster_around_target_trials = np.intersect1d(self.cluster_around_target_trials+1, np.where(self.n_ff_in_a_row == 1)[0])
        self.ignore_sudden_flash_trials, self.ignore_sudden_flash_indices, self.ignore_sudden_flash_indices_for_anim, self.ignored_ff_target_pairs = find_patterns.ignore_sudden_flash_func(self.ff_dataframe, self.ff_real_position_sorted, self.max_point_index, max_ff_distance_from_monkey = 50)
        self.ignore_sudden_flash_time = self.monkey_information['monkey_t'][self.ignore_sudden_flash_indices]
        self.try_a_few_times_trials, self.try_a_few_times_indices, self.try_a_few_times_indices_for_anim = find_patterns.try_a_few_times_func(self.ff_caught_T_sorted, self.monkey_information, self.ff_believed_position_sorted, self.player, self.max_point_index)
        self.give_up_after_trying_trials, self.give_up_after_trying_indices, self.give_up_after_trying_indices_for_anim = find_patterns.give_up_after_trying_func(self.ff_caught_T_sorted, self.monkey_information, self.ff_believed_position_sorted, self.player, self.max_point_index)

        self.all_categories = {'visible_before_last_one': self.visible_before_last_one_trials,
                                'disappear_latest': self.disappear_latest_trials,
                                'two_in_a_row': self.two_in_a_row,
                                'waste_cluster_around_target': self.waste_cluster_around_target_trials,
                                'try_a_few_times': self.try_a_few_times_trials,
                                'give_up_after_trying': self.give_up_after_trying_trials,
                                'ignore_sudden_flash': self.ignore_sudden_flash_trials,
                                
                                # additional categories:
                                'two_in_a_row_simul': self.two_in_a_row_simul,
                                'two_in_a_row_non_simul': self.two_in_a_row_non_simul,
                                'used_cluster': self.used_cluster,}


    def try_retrieving_df(self, df_name, exists_ok=True, data_folder_name_for_retrieval=None):
        if data_folder_name_for_retrieval is None:
            data_folder_name_for_retrieval = self.data_folder_name
        csv_name = df_name + '.csv'
        filepath = os.path.join(data_folder_name_for_retrieval, csv_name)
        if exists(filepath) & exists_ok:
            df_of_interest = pd.read_csv(filepath).drop(["Unnamed: 0"], axis=1)
            print("retrieved ", df_name)        
        else:
            df_of_interest = None
        return df_of_interest 



    def make_or_retrieve_ff_dataframe(self, exists_ok=True):
        self.ff_dataframe = self.try_retrieving_df(df_name='ff_dataframe', exists_ok=exists_ok, data_folder_name_for_retrieval=None)
        if self.ff_dataframe is not None:
            self.ff_dataframe = make_ff_dataframe.furnish_ff_dataframe(self.ff_dataframe, self.ff_real_position_sorted, self.monkey_information,
                                                     self.ff_caught_T_sorted, self.ff_life_sorted)

        if self.ff_dataframe is None: 
            ff_dataframe_args = (self.monkey_information, self.ff_caught_T_sorted, self.ff_flash_sorted,  self.ff_real_position_sorted, self.ff_life_sorted)
            ff_dataframe_kargs = {"max_distance": 400, "data_folder_name": self.data_folder_name, "num_missed_index": 0}
            if self.player != 'monkey':
                ff_dataframe_kargs['obs_ff_indices_in_ff_dataframe'] = self.obs_ff_indices_in_ff_dataframe
            self.ff_dataframe = make_ff_dataframe.make_ff_dataframe_func(*ff_dataframe_args, **ff_dataframe_kargs, player = self.player)
            print("made ff_dataframe")
            
        self.ff_dataframe.memory = self.ff_dataframe.memory.astype('int')
        if len(self.ff_dataframe) > 0:
            self.min_point_index, self.max_point_index = np.min(np.array(self.ff_dataframe['point_index'])), np.max(np.array(self.ff_dataframe['point_index']))
        else:
            self.min_point_index, self.max_point_index = 0, 0
        self.caught_ff_num = len(self.ff_caught_T_sorted)




    def make_or_retrieve_all_trial_patterns(self, exists_ok=True, data_folder_name_for_retrieval=None):
        self.all_trial_patterns = self.try_retrieving_df(df_name='all_trial_patterns', exists_ok=exists_ok, data_folder_name_for_retrieval=data_folder_name_for_retrieval)

        if self.all_trial_patterns is None:       
            self.all_trial_patterns = analyze_patterns_and_features.make_all_trial_patterns(self.caught_ff_num, self.n_ff_in_a_row, self.visible_before_last_one_trials, 
                                                          self.disappear_latest_trials, self.ignore_sudden_flash_trials,
                                                          self.try_a_few_times_trials, self.give_up_after_trying_trials, 
                                                          self.cluster_around_target_trials, self.cluster_around_target_indices, 
                                                          self.waste_cluster_around_target_trials, data_folder_name = self.data_folder_name)
            print("made all_trial_patterns")




    def make_or_retrieve_pattern_frequencies(self, exists_ok=True, data_folder_name_for_retrieval=None):
        self.pattern_frequencies = self.try_retrieving_df(df_name='pattern_frequencies', exists_ok=exists_ok, data_folder_name_for_retrieval=data_folder_name_for_retrieval)

        if self.pattern_frequencies is None:                
            self.pattern_frequencies = analyze_patterns_and_features.make_pattern_frequencies(self.all_trial_patterns, self.ff_caught_T_sorted, self.monkey_information, data_folder_name = self.data_folder_name)
            print("made pattern_frequencies")




    def make_or_retrieve_all_trial_features(self, exists_ok=True, data_folder_name_for_retrieval=None):
        self.all_trial_features = self.try_retrieving_df(df_name='all_trial_features', exists_ok=exists_ok, data_folder_name_for_retrieval=data_folder_name_for_retrieval)

        if self.all_trial_features is None:                
            self.all_trial_features = analyze_patterns_and_features.make_all_trial_features(self.ff_dataframe, self.monkey_information, self.ff_caught_T_sorted, self.cluster_around_target_indices,\
                                                              self.ff_believed_position_sorted, data_folder_name = self.data_folder_name)
            print("made ff_dataframe")



    def make_or_retrieve_feature_statistics(self, exists_ok=True, data_folder_name_for_retrieval=None):
        self.feature_statistics = self.try_retrieving_df(df_name='feature_statistics', exists_ok=exists_ok, data_folder_name_for_retrieval=data_folder_name_for_retrieval)

        if self.feature_statistics is None:                
            self.feature_statistics = analyze_patterns_and_features.make_feature_statistics(self.all_trial_features, data_folder_name = self.data_folder_name)
            print("made feature_statistics")





    def make_info_of_monkey(self):
        self.info_of_monkey = {"monkey_information": self.monkey_information,
                                "ff_dataframe": self.ff_dataframe,
                                "ff_caught_T_sorted": self.ff_caught_T_sorted,
                                "ff_real_position_sorted": self.ff_real_position_sorted,
                                "ff_believed_position_sorted": self.ff_believed_position_sorted,
                                "ff_life_sorted": self.ff_life_sorted,
                                "ff_flash_sorted": self.ff_flash_sorted,
                                "ff_flash_end_sorted": self.ff_flash_end_sorted,
                                "cluster_around_target_indices": self.cluster_around_target_indices}


    def PlotTrials_args(self, classic_plot_kwargs=None, combined_plot_kwargs=None, animation_plot_kwargs=None, all_category_kwargs=None):
        self.plot_behaviors.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_sorted)
        
        if classic_plot_kwargs is not None:
            self.classic_plot_kwargs = classic_plot_kwargs
        
        if combined_plot_kwargs is not None:
            self.combined_plot_kwargs = combined_plot_kwargs
        
        if animation_plot_kwargs is not None:
            self.animation_plot_kwargs = animation_plot_kwargs
            self.all_category_animation_kwargs = plot_behaviors.customize_kwargs_by_category(self.animation_plot_kwargs, images_dir=None)
        
        if all_category_kwargs is not None:
            self.all_category_kwargs = all_category_kwargs
        elif classic_plot_kwargs is not None:
            self.all_category_kwargs = plot_behaviors.customize_kwargs_by_category(classic_plot_kwargs, images_dir=None)


    def plot_trials_from_a_category(self, category_name, max_trial_to_plot, additional_kwargs=None, images_dir=None, using_subplots=False):
        category = self.all_categories[category_name]
        plot_behaviors.plot_trials_from_a_category(category, category_name, max_trial_to_plot, self.plot_behaviors.PlotTrials_args, self.all_category_kwargs, 
                                    self.ff_caught_T_sorted, additional_kwargs=additional_kwargs, images_dir=images_dir, using_subplots=using_subplots)



    def set_animation_parameters(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None, k=3, static_plot_on_the_left = False): 
        # Among currentTrial, num_trials, duration, either currentTrial and num_trials must be specified, or duration must be specified
        currentTrial, num_trials, duration = basic_func.find_currentTrial_or_num_trials_or_duration(self.ff_caught_T_sorted, currentTrial, num_trials, duration)
        
        while duration[1] - duration[0] < 0.1:
            num_trials = num_trials+1
            duration = [self.ff_caught_T_sorted[currentTrial-num_trials], self.ff_caught_T_sorted[currentTrial]]
        
        if static_plot_on_the_left:
            self.fig = plt.figure(figsize=(14.5, 7))
            plot_behaviors.PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_sorted)
 
            if animation_plot_kwargs is None:
                animation_plot_kwargs = self.animation_plot_kwargs

            _, ax1, _, _ = plot_behaviors.PlotTrials(duration, 
                                        fig=self.fig,
                                        *plot_behaviors.PlotTrials_args,
                                        **animation_plot_kwargs,
                                        currentTrial = currentTrial,
                                        num_trials = num_trials,            
                                        )
            self.fig.add_axes(ax1)
            self.ax = self.fig.add_subplot(1, 2, 2)
            self.fig.tight_layout()
        
        else:
            self.fig, self.ax = plt.subplots()
        
        self.currentTrial = currentTrial
        self.num_trials = num_trials
        self.k = k
        self.num_frames, self.anim_monkey_info, self.flash_on_ff_dict, self.alive_ff_dict, self.believed_ff_dict, self.new_num_trials, self.ff_dataframe_anim \
                    = animation_func.prepare_for_animation(self.ff_dataframe, self.ff_caught_T_sorted, self.ff_life_sorted, self.ff_believed_position_sorted, 
                      self.ff_real_position_sorted, self.ff_flash_sorted, self.monkey_information, k = k, currentTrial=currentTrial, num_trials=num_trials, duration=duration)
        while self.num_frames > 150:
            self.k = self.k + 1
            self.num_frames, self.anim_monkey_info, self.flash_on_ff_dict, self.alive_ff_dict, self.believed_ff_dict, self.new_num_trials, self.ff_dataframe_anim \
                    = animation_func.prepare_for_animation(self.ff_dataframe, self.ff_caught_T_sorted, self.ff_life_sorted, self.ff_believed_position_sorted, 
                      self.ff_real_position_sorted, self.ff_flash_sorted, self.monkey_information, k = self.k, currentTrial=currentTrial, num_trials=num_trials, duration=duration)
        print("Number of frames for the animation is:", self.num_frames)




    def save_animation_with_plots_from_a_category(self, category_name, max_trial_to_plot, sampling_frame_ratio = 3, 
                                                  num_trials=3, additional_kwargs=None, exists_ok=True):
        k = sampling_frame_ratio
        category = self.all_categories[category_name]
        self.video_dir = os.path.join('patterns', category_name, self.monkey_name)
        animation_plot_kwargs = self.all_category_animation_kwargs[category_name]
        animation_plot_kwargs['images_dir'] = None # instead, video_dir is used later

        if additional_kwargs is not None:
            for key, value in additional_kwargs.items():
                animation_plot_kwargs[key] = value

        if len(category) > 0:
            with basic_func.initiate_plot(10,10,100):
                for currentTrial in category[:max_trial_to_plot]:
                    file_name = f"{self.data_name}_{currentTrial-num_trials+1}-{currentTrial}.mp4"
                    if exists_ok:
                        video_path_name = os.path.join(self.video_dir, file_name)
                        if exists(os.path.join(self.video_dir, file_name)):
                            print("Animation for the current trial already exists at", video_path_name, "Moving on to the next trial.")
                            continue
                    self.make_animation_with_plots(currentTrial=currentTrial, num_trials=num_trials, animation_plot_kwargs=animation_plot_kwargs, save_video=True, file_name=file_name)


    def save_animation_of_chunks(self, df_with_chunks, monkey_information, chunk_numbers = range(10), sampling_frame_ratio = 3, additional_kwargs=None, exists_ok=True):
        k = sampling_frame_ratio
        self.video_dir = os.path.join('chunks', self.monkey_name, self.data_name)
        animation_plot_kwargs = self.animation_plot_kwargs
        animation_plot_kwargs['images_dir'] = None

        if additional_kwargs is not None:
            for key, value in additional_kwargs.items():
                animation_plot_kwargs[key] = value


        with basic_func.initiate_plot(10,10,100):
            for chunk in chunk_numbers:
                chunk_df = df_with_chunks[df_with_chunks['chunk'] == chunk]
                duration_points = [chunk_df['point_index'].min(), chunk_df['point_index'].max()]
                duration = [monkey_information['monkey_t'][duration_points[0]], monkey_information['monkey_t'][duration_points[0]]+10]

                file_name = f"chunk{chunk}_{round(duration[0], 1)}s-{round(duration[1], 1)}s.mp4"
                if exists_ok:
                    video_path_name = os.path.join(self.video_dir, file_name)
                    if exists(os.path.join(self.video_dir, file_name)):
                        print("Animation for the current trial already exists at", video_path_name, "Moving on to the next trial.")
                        continue
                self.make_animation_with_plots(duration=duration, animation_plot_kwargs=animation_plot_kwargs, save_video=True, file_name=file_name)





    def make_animation_with_plots(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None, save_video=False, file_name=None):
        if file_name is None:
            if currentTrial is not None:
                file_name = f"{self.data_name}_{currentTrial-num_trials+1}-{currentTrial}.mp4"
            else:
                file_name = f"{self.data_name}_animation.mp4"
        self.set_animation_parameters(currentTrial=currentTrial, num_trials=num_trials, duration=duration, animation_plot_kwargs=animation_plot_kwargs, k=3, static_plot_on_the_left=True)
        self.make_animation(margin=100, save_video=save_video, video_dir=self.video_dir, plot_eye_position=True,
                                    file_name=file_name)



    def make_animation(self, margin=400, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True, plot_flash_on_ff=False):
        animate_func = partial(animation_func.animate, ax=self.ax, anim_monkey_info=self.anim_monkey_info, ff_dataframe_anim=self.ff_dataframe_anim, ff_real_position_sorted=self.ff_real_position_sorted, \
                                flash_on_ff_dict=self.flash_on_ff_dict, alive_ff_dict=self.alive_ff_dict, believed_ff_dict=self.believed_ff_dict, margin=margin,
                                plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits, plot_flash_on_ff=plot_flash_on_ff)
        self.anim = animation.FuncAnimation(self.fig, animate_func, frames=self.num_frames, interval=250*self.k, repeat=True) 

        if self.player == 'agent':
            writervideo = animation.FFMpegWriter(fps=int(4/self.k)) #the real life speed, since dt = 0.25
        else:    
            writervideo = animation.FFMpegWriter(fps=int(62/self.k))
        
        if save_video:
            if video_dir is None:
                video_dir = self.log_dir
            if file_name is None:
                file_name = f"{self.currentTrial-self.num_trials+1}-{self.currentTrial}.mp4"
            os.makedirs(video_dir, exist_ok = True)
            video_path_name = f"{video_dir}/{file_name}"
            self.anim.save(video_path_name, writer=writervideo)
            print("Animation is saved at:", video_path_name)

        # save animation as gif
        # self.anim.save(f"{self.log_dir}/agent_animation.gif", writer='imagemagick', fps=int(62/self.k)) #SB3
        # self.anim.save(f"{self.log_dir}/agent_animation.mp4", writer=writervideo)





    def make_animation_with_annotation(self, margin=400, save_video=True, video_dir=None, file_name=None, plot_eye_position=False, set_xy_limits=True):
        self.annotation_info = animation_func.make_annotation_info(self.caught_ff_num+1, self.max_point_index, self.n_ff_in_a_row, self.visible_before_last_one_trials, self.disappear_latest_trials, \
                                                self.ignore_sudden_flash_indices, self.give_up_after_trying_indices, self.try_a_few_times_indices)
        animate_annotated_func = partial(animation_func.animate_annotated, ax=self.ax, anim_monkey_info=self.anim_monkey_info, margin=margin, ff_dataframe_anim=self.ff_dataframe_anim, ff_real_position_sorted=self.ff_real_position_sorted, \
                                          flash_on_ff_dict=self.flash_on_ff_dict, alive_ff_dict=self.alive_ff_dict, believed_ff_dict=self.believed_ff_dict, ff_caught_T_sorted=self.ff_caught_T_sorted, 
                                          annotation_info=self.annotation_info, plot_eye_position=plot_eye_position, set_xy_limits=set_xy_limits)
        self.anim_annotated = animation.FuncAnimation(self.fig, animate_annotated_func, frames=self.num_frames, interval=250*self.k, repeat=True) 
        if self.player == 'agent':
            writervideo = animation.FFMpegWriter(fps=int(4/self.k)) #the real life speed, since dt = 0.25
        else:    
            writervideo = animation.FFMpegWriter(fps=int(62/self.k))        

        if save_video:
            if video_dir is None:
                video_dir = self.log_dir
            if file_name is None:
                file_name = f"annotated_{self.currentTrial-self.num_trials+1}-{self.currentTrial}.mp4"
            os.makedirs(video_dir, exist_ok = True)
            video_path_name = f"{video_dir}/{file_name}.mp4"
            if not exists(video_path_name):
                self.anim_annotated.save(video_path_name, writer=writervideo)







class ProcessMonkeyData(BaseProcessing):
    def __init__(self, raw_data_folder_name):
        if not exists(raw_data_folder_name):
            raise ValueError(f"raw_data_folder_name {raw_data_folder_name} does not exist.")
        self.raw_data_folder_name = raw_data_folder_name
        self.data_folder_name = raw_data_folder_name + '/monkey_patterns'
        self.log_dir = self.data_folder_name
        self.monkey_name = self.log_dir.split('/')[2]
        self.data_name = self.log_dir.split('/')[3]
        if not exists(self.data_folder_name):
            os.makedirs(self.data_folder_name, exist_ok = True)
        self.player = 'monkey'


    def save_important_files(self, exists_ok=True, monkey_information_exists_ok=True):
        self.retrieve_monkey_data(exists_ok = monkey_information_exists_ok)
        
        self.make_or_retrieve_ff_dataframe(exists_ok)
        
        self.make_or_retrieve_target_closest(exists_ok)

        self.make_or_retrieve_target_angle_smallest(exists_ok)
        
        self.find_patterns()

        self.make_or_retrieve_all_trial_patterns(exists_ok)
        
        self.make_or_retrieve_pattern_frequencies(exists_ok)
        
        self.make_or_retrieve_all_trial_features(exists_ok)
        
        self.make_or_retrieve_feature_statistics(exists_ok)
        

    def retrieve_monkey_data(self, exists_ok=True):
        self.accurate_start_time, self.accurate_end_time = process_raw_data.find_start_and_accurate_end_time(self.raw_data_folder_name)
        self.monkey_information, self.ff_caught_T_sorted, self.ff_index_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.ff_life_sorted, self.ff_flash_sorted, \
                self.ff_flash_end_sorted = process_raw_data.log_extractor(raw_data_folder_name = self.raw_data_folder_name).extract_data(monkey_information_exists_OK=exists_ok)   
        filepath = os.path.join(self.raw_data_folder_name, 'monkey_information.csv')
        if not exists(filepath):
            monkey_information_small = self.monkey_information[['monkey_t', 'monkey_x', 'monkey_y', 'monkey_speed', 'monkey_angles', 'monkey_dw', 'LDy', 'LDz', 'RDy', 'RDz']]
            monkey_information_small.to_csv(filepath)
        print("retrieved monkey data")



    def make_or_retrieve_target_closest(self, exists_ok=True):
        filepath = self.data_folder_name + '/target_closest.csv'
        if exists(filepath) & exists_ok:
            self.target_closest = np.genfromtxt(filepath, delimiter=',').astype('int')
            print("retrieved target_closest")
        else: 
            self.target_closest = find_patterns.make_target_closest(self.ff_dataframe, self.max_point_index, data_folder_name=self.data_folder_name)
            print("made target_closest")


    def make_or_retrieve_target_angle_smallest(self, exists_ok=True):
        filepath = self.data_folder_name + '/target_angle_smallest.csv'
        if exists(filepath) & exists_ok:
            self.target_angle_smallest = np.genfromtxt(filepath, delimiter=',').astype('int')
            print("retrieved target_angle_smallest")
        else:
            # make target_angle_smallest:
            # 2 means target is has the smallest absolute angle at that point (visible or in memory)
            # 1 means the target does not have the smallest absolute angle. In the subset of 1:
                # 1 means both the target and a non-target are visible or in memory (which we call present)
                # 0 means the target is neither visible or in memory, but there is at least one other ff visible or in memory
                # -1 means both the target and other ff are neither visible or in memory
            self.target_angle_smallest = find_patterns.make_target_angle_smallest(self.ff_dataframe, self.max_point_index, data_folder_name=self.data_folder_name)
            print("made target_angle_smallest")




    def make_distance_dataframe(self):
        corresponding_t = np.array(self.monkey_information['monkey_t'])[:self.max_point_index+1]
        distance_dataframe = turn_array_into_df(self, array=self.target_closest, corresponding_t=corresponding_t)
        self.distance_dataframe = distance_dataframe
        self.trial_vs_distance = distance_dataframe[['num', 'trial']].groupby('trial').max()

    def make_angle_dataframe(self):
        corresponding_t = np.array(self.monkey_information['monkey_t'])[:self.max_point_index+1]
        angle_dataframe = turn_array_into_df(self, array=self.target_angle_smallest, corresponding_t=corresponding_t)
        self.angle_dataframe = angle_dataframe
        self.trial_vs_angle = angle_dataframe[['num', 'trial']].groupby('trial').max()



def turn_array_into_df(self, array, corresponding_t):
    info_num = [int(num) for num in array]
    info_dict = {'t': corresponding_t.tolist(), 'num': info_num}
    info_dataframe = pd.DataFrame(info_dict)
    info_dataframe['trial']=np.digitize(corresponding_t, self.ff_caught_T_sorted)
    return info_dataframe