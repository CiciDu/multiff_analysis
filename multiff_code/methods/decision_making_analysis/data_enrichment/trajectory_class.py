import os
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from os.path import exists

# ----------------------------------------------------------------------
# Internal module imports
# ----------------------------------------------------------------------
from data_wrangling import base_processing_class
from null_behaviors import curvature_utils
from decision_making_analysis.data_enrichment import trajectory_utils
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data
from decision_making_analysis.event_detection import get_miss_to_switch_data
from decision_making_analysis.data_compilation import miss_events_class
from decision_making_analysis.ff_data_acquisition import cluster_replacement_utils
from decision_making_analysis.event_detection import detect_rsw_and_rcap
from null_behaviors import curv_of_traj_utils


class TrajectoryDataClass():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def streamline_process_to_collect_traj_data_only(self, point_index_all, curv_of_traj_data_df_exists_ok=True):
        """Collect only trajectory-related data for specified point indices."""
        self.time_range_of_trajectory = self.gc_kwargs['time_range_of_trajectory']
        self.trajectory_features = self.gc_kwargs['trajectory_features']
        self.point_index_all = point_index_all

        self.get_monkey_data(include_ff_dataframe=False,
                             include_rsw_data=False)
        self.make_or_retrieve_curv_of_traj_df(
            exists_ok=curv_of_traj_data_df_exists_ok)
        self.add_curv_of_traj_info_to_monkey_information(column_exists_ok=True)

        self.time_all = self.monkey_information.loc[self.point_index_all, 'time'].values
        self.get_trajectory_and_stop_info_for_machine_learning()

        relevant_curv_of_traj = trajectory_utils.find_trajectory_arc_info(
            self.point_index_all,
            self.curv_of_traj_df,
            ff_caught_T_new=self.ff_caught_T_new,
            monkey_information=self.monkey_information,
        )

        self.relevant_curv_of_traj_df = pd.DataFrame({
            'point_index': self.point_index_all,
            'curv_of_traj': relevant_curv_of_traj,
        })

        self.traj_data_df, self.traj_data_feature_names = trajectory_utils.combine_trajectory_and_stop_info_and_curvature_info(
            self.traj_points_df, self.traj_stops_df, self.relevant_curv_of_traj_df
        )
        self.traj_data_df['point_index'] = self.point_index_all

    def collect_trajectory_data_to_be_combined_across_sessions(
        self,
        time_range_of_trajectory=(-1, 1),
        num_time_points_for_trajectory=10,
        time_range_of_trajectory_to_plot=None,
        num_time_points_for_trajectory_to_plot=10,
        trajectory_features=('monkey_distance', 'monkey_angle_to_origin'),
        **kwargs,
    ):

        self.time_range_of_trajectory = time_range_of_trajectory
        self.trajectory_features = trajectory_features
        self.gc_kwargs.update({
            'num_time_points_for_trajectory': num_time_points_for_trajectory,
            'time_range_of_trajectory_to_plot': time_range_of_trajectory_to_plot,
            'num_time_points_for_trajectory_to_plot': num_time_points_for_trajectory_to_plot,
        })

        self.get_trajectory_and_stop_info_for_machine_learning()

        # --- Compute curvature per trajectory arc ---
        relevant_curv_of_traj = trajectory_utils.find_trajectory_arc_info(
            self.point_index_all, self.curv_of_traj_df,
            ff_caught_T_new=self.ff_caught_T_new,
            monkey_information=self.monkey_information,
        )
        self.relevant_curv_of_traj_df = pd.DataFrame({
            'point_index': self.point_index_all,
            'curv_of_traj': relevant_curv_of_traj,
        })

        # --- Combine trajectory + curvature ---
        self.traj_data_df, self.traj_data_feature_names = trajectory_utils.combine_trajectory_and_stop_info_and_curvature_info(
            self.traj_points_df, self.traj_stops_df, self.relevant_curv_of_traj_df
        )
        self.traj_data_df['point_index'] = self.point_index_all

    def get_trajectory_and_stop_info_for_machine_learning(self):
        """Generate trajectory and stop information matrices for ML models."""
        traj_points, traj_feature_names = trajectory_utils.generate_trajectory_position_data(
            self.time_all,
            self.monkey_information,
            time_range_of_trajectory=self.time_range_of_trajectory,
            num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'],
            trajectory_features=self.trajectory_features,
        )

        traj_stops, stop_feature_names = trajectory_utils.generate_stops_info(
            self.time_all,
            self.monkey_information,
            time_range_of_trajectory=self.time_range_of_trajectory,
            num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'],
        )

        self.traj_points_df = pd.DataFrame(
            traj_points, columns=traj_feature_names)
        self.traj_stops_df = pd.DataFrame(
            traj_stops, columns=stop_feature_names)
        self.traj_points_df['point_index'] = self.point_index_all
        self.traj_stops_df['point_index'] = self.point_index_all
