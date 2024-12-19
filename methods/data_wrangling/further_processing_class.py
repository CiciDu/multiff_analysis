from data_wrangling import basic_func, base_processing_class, monkey_data_classes
from pattern_discovery import pattern_by_trials, organize_patterns_and_features, monkey_landing_in_ff
from visualization import animation_func, animation_utils, plot_trials, plot_behaviors_utils
from decision_making_analysis.compare_GUAT_and_TAFT import find_GUAT_or_TAFT_trials
from decision_making_analysis.GUAT import GUAT_utils

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


class FurtherProcessing(base_processing_class.BaseProcessing):
    def __init__(self):
        super().__init__()


    def prepare_to_find_patterns_and_features(self, find_patterns=True):
        self.retrieve_or_make_monkey_data(already_made_ok=True)
        self.make_or_retrieve_ff_dataframe(already_made_ok=True, exists_ok=True)
        if find_patterns:
            self.find_patterns()

    def get_monkey_data(self, already_retrieved_ok=True, include_ff_dataframe=True, include_GUAT_data=False,
                        include_TAFT_data=False):
        
        if (already_retrieved_ok is False) | (not hasattr(self, 'monkey_information')):
            self.data_item = monkey_data_classes.ProcessMonkeyData(raw_data_folder_path=self.raw_data_folder_path)
            self.data_item.retrieve_or_make_monkey_data()
            self.monkey_information = self.data_item.monkey_information
            self.ff_life_sorted = self.data_item.ff_life_sorted
            self.ff_real_position_sorted = self.data_item.ff_real_position_sorted
            self.ff_believed_position_sorted = self.data_item.ff_believed_position_sorted
            self.ff_caught_T_new = self.data_item.ff_caught_T_new

        if include_ff_dataframe:
            if (already_retrieved_ok is False) | (not hasattr(self, 'ff_dataframe')):
                # if self.data_item is not called yet
                if not hasattr(self, 'data_item'):
                    self.data_item = monkey_data_classes.ProcessMonkeyData(raw_data_folder_path=self.raw_data_folder_path)
                    self.data_item.retrieve_or_make_monkey_data()
                if (already_retrieved_ok is False) | (not hasattr(self, 'ff_dataframe')):
                    self.data_item.make_or_retrieve_ff_dataframe(num_missed_index=0, exists_ok=True)
                    self.ff_dataframe = self.data_item.ff_dataframe
                    self.ff_dataframe_visible = self.ff_dataframe.loc[self.ff_dataframe['visible']==1].copy()


    def find_patterns(self):
        self.n_ff_in_a_row = pattern_by_trials.n_ff_in_a_row_func(self.ff_believed_position_sorted, distance_between_ff = 50)
        self.two_in_a_row = np.where(self.n_ff_in_a_row==2)[0]
        self.two_in_a_row_simul, self.two_in_a_row_non_simul = pattern_by_trials.whether_current_and_last_targets_are_captured_simultaneously(self.two_in_a_row, self.ff_caught_T_new)
        self.three_in_a_row= np.where(self.n_ff_in_a_row==3)[0]
        self.four_in_a_row= np.where(self.n_ff_in_a_row==4)[0]
        self.on_before_last_one_trials = pattern_by_trials.on_before_last_one_func(self.ff_flash_end_sorted, self.ff_caught_T_new, self.caught_ff_num)
        self.on_before_last_one_simul, self.on_before_last_one_non_simul = pattern_by_trials.whether_current_and_last_targets_are_captured_simultaneously(self.on_before_last_one_trials, self.ff_caught_T_new)
        self.visible_before_last_one_trials = pattern_by_trials.visible_before_last_one_func(self.ff_dataframe)
        self.used_cluster = np.intersect1d(self.two_in_a_row_non_simul, self.visible_before_last_one_trials)
        self.disappear_latest_trials = pattern_by_trials.disappear_latest_func(self.ff_dataframe)
        self.cluster_around_target_trials, self.used_cluster_trials, self.cluster_around_target_indices, self.cluster_around_target_positions = pattern_by_trials.cluster_around_target_func(self.ff_dataframe, self.caught_ff_num, self.ff_caught_T_new, self.ff_real_position_sorted)
        # take out trials in cluster_around_target_trials but not in used_cluster_trials
        self.waste_cluster_around_target_trials = np.setdiff1d(self.cluster_around_target_trials, self.used_cluster_trials)
        self.ignore_sudden_flash_trials, self.sudden_flash_trials, self.ignore_sudden_flash_indices, self.ignore_sudden_flash_indices_for_anim, self.ignored_ff_target_pairs = pattern_by_trials.ignore_sudden_flash_func(self.ff_dataframe, self.max_point_index, max_ff_distance_from_monkey = 50)
        self.ignore_sudden_flash_time = self.monkey_information['monkey_t'][self.ignore_sudden_flash_indices]
        self.get_give_up_after_trying_info()
        self.get_try_a_few_times_info()

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


    def get_try_a_few_times_info(self):

        self.try_a_few_times_trials, self.TAFT_indices_df, self.TAFT_trials_df, self.try_a_few_times_indices_for_anim = find_GUAT_or_TAFT_trials.try_a_few_times_func(self.monkey_information, self.ff_caught_T_new,  self.ff_real_position_sorted, max_point_index=self.max_point_index)


    def get_give_up_after_trying_info(self):

        self.give_up_after_trying_trials, self.GUAT_indices_df, self.GUAT_trials_df, self.GUAT_point_indices_for_anim, self.GUAT_w_ff_df = find_GUAT_or_TAFT_trials.give_up_after_trying_func(self.ff_dataframe, self.monkey_information, self.ff_caught_T_new, self.ff_real_position_sorted, max_point_index=self.max_point_index)
        self.give_up_after_trying_indices = self.GUAT_indices_df['point_index'].values
        self.give_up_after_trying_info_bundle = (self.give_up_after_trying_trials, self.GUAT_point_indices_for_anim, self.GUAT_indices_df, self.GUAT_trials_df)                                                                                                                                                                                                                                           


    def make_or_retrieve_all_trial_patterns(self, exists_ok=True):
        self.all_trial_patterns = self.try_retrieving_df(df_name='all_trial_patterns', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.all_trial_patterns is None:       
            if getattr(self, 'n_ff_in_a_row', None) is None:
                self.prepare_to_find_patterns_and_features()
            self.all_trial_patterns = self.make_all_trial_patterns()
            print("made all_trial_patterns")


    def make_or_retrieve_pattern_frequencies(self, exists_ok=True):
        self.pattern_frequencies = self.try_retrieving_df(df_name='pattern_frequencies', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)
        self.make_one_stop_w_ff_df()
        self.one_stop_w_ff_frequency = len(self.one_stop_w_ff_df)
        if getattr(self, 'GUAT_w_ff_df', None) is None:
            self.get_give_up_after_trying_info()
        self.GUAT_w_ff_frequency = len(self.GUAT_w_ff_df)
        

        if self.pattern_frequencies is None:
            if getattr(self, 'monkey_information', None) is None:
                self.retrieve_or_make_monkey_data(already_made_ok=True)
            self.pattern_frequencies = organize_patterns_and_features.make_pattern_frequencies(self.all_trial_patterns, self.ff_caught_T_new, self.monkey_information, 
                                                                                               self.GUAT_w_ff_frequency, self.one_stop_w_ff_frequency,
                                                                                               data_folder_name = self.patterns_and_features_data_folder_path)
            print("made pattern_frequencies")



    def make_or_retrieve_all_trial_features(self, exists_ok=True):
        self.all_trial_features = self.try_retrieving_df(df_name='all_trial_features', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.all_trial_features is None:   
            if getattr(self, 'cluster_around_target_indices', None) is None:
                self.prepare_to_find_patterns_and_features()             
            self.all_trial_features = organize_patterns_and_features.make_all_trial_features(self.ff_dataframe, self.monkey_information, self.ff_caught_T_new, self.cluster_around_target_indices,\
                                                              self.ff_real_position_sorted, self.ff_believed_position_sorted, data_folder_name = self.patterns_and_features_data_folder_path)
            print("made all_trial_features")



    def make_or_retrieve_feature_statistics(self, exists_ok=True):
        self.feature_statistics = self.try_retrieving_df(df_name='feature_statistics', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.feature_statistics is None:                
            self.feature_statistics = organize_patterns_and_features.make_feature_statistics(self.all_trial_features, data_folder_name=self.patterns_and_features_data_folder_path)
            print("made feature_statistics")


    def make_or_retrieve_scatter_around_target_df(self, exists_ok=True):
        self.scatter_around_target_df = self.try_retrieving_df(df_name='scatter_around_target_df', exists_ok=exists_ok, data_folder_name_for_retrieval=self.patterns_and_features_data_folder_path)

        if self.scatter_around_target_df is None:
            self.scatter_around_target_df = monkey_landing_in_ff.make_scatter_around_target_df(self.monkey_information, 
                                                                                                self.closest_stop_to_capture_df, 
                                                                                                self.ff_real_position_sorted,
                                                                                                data_folder_name=self.patterns_and_features_data_folder_path)
            print("made scatter_around_target_df")

    def make_info_of_monkey(self):
        self.info_of_monkey = {"monkey_information": self.monkey_information,
                                "ff_dataframe": self.ff_dataframe,
                                "ff_caught_T_new": self.ff_caught_T_new,
                                "ff_real_position_sorted": self.ff_real_position_sorted,
                                "ff_believed_position_sorted": self.ff_believed_position_sorted,
                                "ff_life_sorted": self.ff_life_sorted,
                                "ff_flash_sorted": self.ff_flash_sorted,
                                "ff_flash_end_sorted": self.ff_flash_end_sorted,
                                "cluster_around_target_indices": self.cluster_around_target_indices}


    def make_PlotTrials_kargs(self, classic_plot_kwargs=None, combined_plot_kwargs=None, animation_plot_kwargs=None, all_category_kwargs=None):

        if classic_plot_kwargs is not None:
            self.classic_plot_kwargs = classic_plot_kwargs
        
        if combined_plot_kwargs is not None:
            self.combined_plot_kwargs = combined_plot_kwargs
        
        if animation_plot_kwargs is not None:
            self.animation_plot_kwargs = animation_plot_kwargs
            self.all_category_animation_kwargs = plot_behaviors_utils.customize_kwargs_by_category(self.animation_plot_kwargs, images_dir=None)
        
        if all_category_kwargs is not None:
            self.all_category_kwargs = all_category_kwargs
        elif classic_plot_kwargs is not None:
            self.all_category_kwargs = plot_behaviors_utils.customize_kwargs_by_category(classic_plot_kwargs, images_dir=None)


    def plot_trials_from_a_category(self, category_name, max_trial_to_plot, trials=None, additional_kwargs=None, images_dir=None, using_subplots=False, figsize=(10, 10)):
        category = self.all_categories[category_name]
        plot_behaviors_utils.plot_trials_from_a_category(category, category_name, max_trial_to_plot, self.PlotTrials_args, self.all_category_kwargs, 
                                    self.ff_caught_T_new, trials=trials, additional_kwargs=additional_kwargs, images_dir=images_dir, using_subplots=using_subplots, figsize=figsize)



    def set_animation_parameters(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None, k=3, static_plot_on_the_left = False, max_num_frames=150, max_duration=30, min_duration=1, rotated=True): 
        # Among currentTrial, num_trials, duration, either currentTrial and num_trials must be specified, or duration must be specified
        currentTrial, num_trials, duration = basic_func.find_currentTrial_or_num_trials_or_duration(self.ff_caught_T_new, currentTrial, num_trials, duration)
        
        # if the duration is too short, then increase the number of trials
        while duration[1] - duration[0] < 0.1:
            num_trials = num_trials+1
            duration = [self.ff_caught_T_new[currentTrial-num_trials], self.ff_caught_T_new[currentTrial]]
        
        if static_plot_on_the_left:
            self.fig, self.ax = self._make_static_plot_on_the_left(currentTrial=currentTrial, num_trials=num_trials, duration=duration, animation_plot_kwargs=animation_plot_kwargs)
        else:
            self.fig, self.ax = plt.subplots()
        
        self.currentTrial = currentTrial
        self.num_trials = num_trials
        self.duration = duration
        self.k = k

        self._call_prepare_for_animation_func(currentTrial=currentTrial, num_trials=num_trials, duration=duration, k=k, max_num_frames=max_num_frames,
                                              max_duration=max_duration, min_duration=min_duration, rotated=rotated)


    def _call_prepare_for_animation_func(self, currentTrial=None, num_trials=None, duration=None, k=1, max_num_frames=None,
                                         max_duration=30, min_duration=1, rotated=True):
        self.num_frames, self.anim_monkey_info, self.flash_on_ff_dict, self.alive_ff_dict, self.believed_ff_dict, self.new_num_trials, self.ff_dataframe_anim \
                = animation_utils.prepare_for_animation(
                self.ff_dataframe, self.ff_caught_T_new, self.ff_life_sorted, self.ff_believed_position_sorted, 
                self.ff_real_position_sorted, self.ff_flash_sorted, self.monkey_information, k=k, currentTrial=currentTrial, num_trials=num_trials, duration=duration,
                max_duration=max_duration, min_duration=min_duration, rotated=rotated)
        print("Number of frames is:", self.num_frames)

        # if the number of frames is too large, then reduce k so that the number of frames is reduced
        if max_num_frames is not None:
            while self.num_frames > 150:
                self.k = self.k + 1
                self.num_frames, self.anim_monkey_info, self.flash_on_ff_dict, self.alive_ff_dict, self.believed_ff_dict, self.new_num_trials, self.ff_dataframe_anim \
                        = animation_utils.prepare_for_animation(self.ff_dataframe, self.ff_caught_T_new, self.ff_life_sorted, self.ff_believed_position_sorted, 
                        self.ff_real_position_sorted, self.ff_flash_sorted, self.monkey_information, k = self.k, currentTrial=currentTrial, num_trials=num_trials, 
                        duration=duration)
        print("Number of frames for the animation is:", self.num_frames)


    def _make_static_plot_on_the_left(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None):
        self.fig = plt.figure(figsize=(14.5, 7))
        PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted, self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_new)

        if animation_plot_kwargs is None:
            animation_plot_kwargs = self.animation_plot_kwargs

        ax1 = self.fig.add_subplot(1, 2, 1)
        returned_info = plot_trials.PlotTrials(duration, 
                                    fig=self.fig,
                                    axes=ax1,
                                    *PlotTrials_args,
                                    **animation_plot_kwargs,
                                    currentTrial = currentTrial,
                                    num_trials = num_trials,            
                                    )
        ax1 = returned_info['axes']
        self.fig.add_axes(ax1)
        self.ax = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout()

        return self.fig, self.ax


    def call_animation_function(self, with_annotation=False, 
                                margin=100, dt=0.0165, plot_time_index=False, fps=None, 
                                save_video=True, video_dir=None, file_name=None, 
                                **animate_kwargs):
        # dt is to be used to determine the fps(frame per second) of the animation

        # if the key "margin" is in animate_kwargs, then put it into animate_kwargs
        # if not, then put the default value (500) into animate_kwargs
        if 'margin' not in animate_kwargs.keys():
            animate_kwargs['margin'] = margin
            
        if with_annotation:
            self.annotation_info = animation_utils.make_annotation_info(self.caught_ff_num+1, self.max_point_index, self.n_ff_in_a_row, self.visible_before_last_one_trials, self.disappear_latest_trials, \
                                                                       self.ignore_sudden_flash_indices, self.GUAT_indices_df['point_index'].values, self.try_a_few_times_indices)
            animate_func = partial(animation_func.animate_annotated, ax=self.ax, anim_monkey_info=self.anim_monkey_info, ff_dataframe_anim=self.ff_dataframe_anim, \
                                   flash_on_ff_dict=self.flash_on_ff_dict, alive_ff_dict=self.alive_ff_dict, believed_ff_dict=self.believed_ff_dict, ff_caught_T_new=self.ff_caught_T_new, annotation_info=self.annotation_info,
                                   plot_time_index=plot_time_index, **animate_kwargs)
        else:
            animate_func = partial(animation_func.animate, ax=self.ax, anim_monkey_info=self.anim_monkey_info, ff_dataframe_anim=self.ff_dataframe_anim,\
                                    flash_on_ff_dict=self.flash_on_ff_dict, alive_ff_dict=self.alive_ff_dict, believed_ff_dict=self.believed_ff_dict, plot_time_index=plot_time_index, **animate_kwargs)
        self.anim = animation.FuncAnimation(self.fig, animate_func, frames=self.num_frames, interval=dt*1000*self.k, repeat=True) 

        if save_video:
            self._save_animation(fps, video_dir, file_name)



    def _save_animation(self, fps, video_dir, file_name):
        if fps is None:
            if self.player == 'agent':
                fps = int(4/self.k) #the real life speed
            else:
                fps = int(62/self.k)

        if video_dir is None:
            video_dir = self.processed_data_folder_path

        if file_name is None:
            try:
                file_name=self.agent_id + f'__{self.currentTrial-self.num_trials+1}-{self.currentTrial}.mp4'
            except TypeError:
                file_name=self.agent_id + f'__{self.duration[0]}s_to_{self.duration[1]}s.mp4'

        os.makedirs(video_dir, exist_ok = True)
        video_path_name = f"{video_dir}/{file_name}"

        print("Saving animation as:", video_path_name)
        writervideo = animation.FFMpegWriter(fps=fps)
        self.anim.save(video_path_name, writer=writervideo)
        print("Animation is saved at:", video_path_name)
        
        # save animation as gif
        # self.anim.save(f"{self.processed_data_folder_path}/agent_animation.gif", writer='imagemagick', fps=int(62/self.k)) #SB3
        # self.anim.save(f"{self.processed_data_folder_path}/agent_animation.mp4", writer=writervideo)


    def make_animation(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None, 
                       save_video=False, video_dir=None, file_name=None, 
                       dt=0.0165, k=3, with_annotation=False, plot_time_index=False, 
                       static_plot_on_the_left=True, margin=100, max_num_frames=150, max_duration=30, min_duration=1,
                       fps=None, rotated=True, **animate_kwargs):
        if file_name is None:
            if currentTrial is not None:
                file_name = f"{self.data_name}_{currentTrial-num_trials+1}-{currentTrial}.mp4"
            elif duration is not None:
                file_name = f"{self.data_name}_{duration[0]}s-{duration[1]}s.mp4"
            else:
                file_name = f"{self.data_name}_animation.mp4"
        if save_video:
            if video_dir is None:
                video_dir = 'videos'
                print("No video was set, so the default path -- 'videos' -- is used.")
        
        animate_kwargs['margin'] = margin
        self.set_animation_parameters(currentTrial=currentTrial, num_trials=num_trials, duration=duration, animation_plot_kwargs=animation_plot_kwargs, k=k, rotated=rotated,
                                      static_plot_on_the_left=static_plot_on_the_left, max_num_frames=max_num_frames, max_duration=max_duration, min_duration=min_duration)
        self.call_animation_function(save_video=save_video, video_dir=video_dir, plot_time_index=plot_time_index,
                                    file_name=file_name, dt=dt, with_annotation=with_annotation, fps=fps, **animate_kwargs)





    def make_animation_from_a_category(self, category_name, max_trial_to_plot, sampling_frame_ratio = 3, max_duration=30, min_duration=1,
                                                  num_trials=3, save_video=True, video_dir=None, additional_kwargs=None, animation_exists_ok=True,
                                                  with_annotation=False, dt=0.0165, static_plot_on_the_left=True, max_num_frames=None, **animate_kwargs):
        '''
        Create animations for a given category of trials.
        '''
        
        # note: if save_video is True, but video_dir is None, then video_dir is set to be the same as self.video_dir eventually
        k = sampling_frame_ratio
        category = self.all_categories[category_name]

        self.video_dir = os.path.join('patterns', category_name, self.monkey_name)
        video_dir = video_dir if video_dir is not None else self.video_dir

        animation_plot_kwargs = self.all_category_animation_kwargs[category_name]
        animation_plot_kwargs['images_dir'] = None # instead, video_dir is used later

        if additional_kwargs is not None:
            animation_plot_kwargs.update(additional_kwargs)

        if len(category) == 0:
            print(f"No trials found for category: {category_name}")
            return 

        with basic_func.initiate_plot(10,10,100):
            for currentTrial in category[:max_trial_to_plot]:
                file_name = f"{self.data_name}_{currentTrial-num_trials+1}-{currentTrial}.mp4"
                video_path_name = os.path.join(self.video_dir, file_name)
                
                if animation_exists_ok and exists(video_path_name):
                    print(f"Animation for trial {currentTrial} already exists at {video_path_name}. Skipping.")
                    continue
                
                self.make_animation(
                    currentTrial=currentTrial,
                    num_trials=num_trials,
                    animation_plot_kwargs=animation_plot_kwargs,
                    static_plot_on_the_left=static_plot_on_the_left,
                    save_video=save_video,
                    video_dir=video_dir,
                    file_name=file_name,
                    with_annotation=with_annotation,
                    dt=dt,
                    max_num_frames=max_num_frames,
                    max_duration=max_duration,
                    min_duration=min_duration,
                    **animate_kwargs
                )

    def make_animation_of_chunks(self, df_with_chunks, monkey_information, chunk_numbers = range(10), sampling_frame_ratio = 3, additional_kwargs=None, exists_ok=True, 
                                 max_duration=30, min_duration=1, dt=0.016, save_video=True, static_plot_on_the_left=True, max_num_frames=None, **animate_kwargs):
        k = sampling_frame_ratio
        self.video_dir = os.path.join('chunks', self.monkey_name, self.data_name)
        animation_plot_kwargs = self.animation_plot_kwargs
        animation_plot_kwargs['images_dir'] = None

        if additional_kwargs is not None:
            animation_plot_kwargs.update(additional_kwargs)

        with basic_func.initiate_plot(10,10,100):
            for chunk in chunk_numbers:
                chunk_df = df_with_chunks[df_with_chunks['chunk'] == chunk]
                duration_points = [chunk_df['point_index'].min(), chunk_df['point_index'].max()]
                duration = [monkey_information['monkey_t'][duration_points[0]], monkey_information['monkey_t'][duration_points[0]]+10]

                file_name = f"chunk{chunk}_{round(duration[0], 1)}s-{round(duration[1], 1)}s.mp4"
                video_path_name = os.path.join(self.video_dir, file_name)
                if exists_ok & exists(os.path.join(self.video_dir, file_name)):
                    print("Animation for the current chunk already exists at", video_path_name, "Moving on to the next trial.")
                    continue
                self.make_animation(duration=duration, animation_plot_kwargs=animation_plot_kwargs, save_video=save_video, file_name=file_name, dt=dt, max_duration=max_duration, min_duration=min_duration,
                                    video_dir=self.video_dir, static_plot_on_the_left=static_plot_on_the_left, max_num_frames=max_num_frames, **animate_kwargs)


    def make_all_trial_patterns(self):
        zero_array = np.zeros(self.caught_ff_num + 1, dtype=int)

        multiple_in_a_row = np.where(self.n_ff_in_a_row >= 2)[0]
        # multiple_in_a_row_all means it also includes the first ff that's caught in a cluster
        multiple_in_a_row_all = np.union1d(multiple_in_a_row, multiple_in_a_row - 1)
        multiple_in_a_row2 = zero_array.copy()
        multiple_in_a_row_all2 = zero_array.copy()
        multiple_in_a_row2[multiple_in_a_row] = 1
        multiple_in_a_row_all2[multiple_in_a_row_all] = 1

        two_in_a_row = np.where(self.n_ff_in_a_row == 2)[0]
        three_in_a_row = np.where(self.n_ff_in_a_row == 3)[0]
        four_in_a_row = np.where(self.n_ff_in_a_row == 4)[0]

        two_in_a_row2 = zero_array.copy()
        if len(two_in_a_row) > 0:
            two_in_a_row2[two_in_a_row] = 1

        three_in_a_row2 = zero_array.copy()
        if len(three_in_a_row) > 0:
            three_in_a_row2[three_in_a_row] = 1

        four_in_a_row2 = zero_array.copy()
        if len(four_in_a_row) > 0:
            four_in_a_row2[four_in_a_row] = 1

        one_in_a_row = np.where(self.n_ff_in_a_row < 2)[0]
        one_in_a_row2 = zero_array.copy()
        if len(one_in_a_row) > 0:
            one_in_a_row2[one_in_a_row] = 1

        visible_before_last_one2 = zero_array.copy()
        if len(self.visible_before_last_one_trials) > 0:
            visible_before_last_one2[self.visible_before_last_one_trials] = 1

        disappear_latest2 = zero_array.copy()
        if len(self.disappear_latest_trials) > 0:
            disappear_latest2[self.disappear_latest_trials] = 1

        sudden_flash_trials2 = zero_array.copy()
        if len(self.sudden_flash_trials) > 0:
            sudden_flash_trials2[self.sudden_flash_trials] = 1

        ignore_sudden_flash2 = zero_array.copy()
        if len(self.ignore_sudden_flash_trials) > 0:
            ignore_sudden_flash2[self.ignore_sudden_flash_trials] = 1

        try_a_few_times2 = zero_array.copy()
        if len(self.try_a_few_times_trials) > 0:
            try_a_few_times2[self.try_a_few_times_trials] = 1

        give_up_after_trying2 = zero_array.copy()
        if len(self.give_up_after_trying_trials) > 0:
            give_up_after_trying2[self.give_up_after_trying_trials] = 1

        cluster_around_target2 = zero_array.copy()
        if len(self.cluster_around_target_trials) > 0:
            cluster_around_target2[self.cluster_around_target_trials] = 1

        use_cluster2 = zero_array.copy()
        if len(self.used_cluster_trials) > 0:
            use_cluster2[self.used_cluster_trials] = 1

        waste_cluster_around_target2 = zero_array.copy()
        if len(self.waste_cluster_around_target_trials) > 0:
            waste_cluster_around_target2[self.waste_cluster_around_target_trials] = 1

        all_trial_patterns_dict = {
            # bool
            'two_in_a_row': two_in_a_row2,
            'three_in_a_row': three_in_a_row2,
            'four_in_a_row': four_in_a_row2,
            'one_in_a_row': one_in_a_row2,
            'multiple_in_a_row': multiple_in_a_row2,
            'multiple_in_a_row_all': multiple_in_a_row_all2,
            'visible_before_last_one': visible_before_last_one2,
            'disappear_latest': disappear_latest2,
            'sudden_flash': sudden_flash_trials2,
            'ignore_sudden_flash': ignore_sudden_flash2,
            'try_a_few_times': try_a_few_times2,
            'give_up_after_trying': give_up_after_trying2,
            'cluster_around_target': cluster_around_target2,
            'use_cluster': use_cluster2,
            'waste_cluster_around_target': waste_cluster_around_target2
        }

        for key, value in all_trial_patterns_dict.items():
            all_trial_patterns_dict[key] = value[:-1]

        self.all_trial_patterns = pd.DataFrame(all_trial_patterns_dict)

        if self.patterns_and_features_data_folder_path:
            basic_func.save_df_to_csv(self.all_trial_patterns, 'all_trial_patterns', self.patterns_and_features_data_folder_path)

        return self.all_trial_patterns

    def make_one_stop_w_ff_df(self):   
        self.prepare_to_find_patterns_and_features(find_patterns=False)

        self.one_stop_df = GUAT_utils.streamline_getting_one_stop_df(self.monkey_information, self.ff_dataframe, self.ff_caught_T_new)
        self.one_stop_w_ff_df = GUAT_utils.make_one_stop_w_ff_df(self.one_stop_df)