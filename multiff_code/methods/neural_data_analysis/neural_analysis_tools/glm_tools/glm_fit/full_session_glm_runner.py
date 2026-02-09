# =========================
# Standard library
# =========================
import os
import math
import json

# =========================
# Third-party
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# MultiFF imports (USED ONLY)
# =========================
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_utils,
    pn_aligned_by_event,
)

from neural_data_analysis.design_kits.design_by_segment import (
    rebin_segments,
    temporal_feats,
    create_pn_design_df,
)

from neural_data_analysis.topic_based_neural_analysis.full_session import (
    create_full_session_design,
    selected_pn_design_features,
    selected_stop_design_features,
    create_best_arc_design,
    select_fs_features,
)

from neural_data_analysis.design_kits.design_around_event import (
    event_binning,
)

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    get_stops_utils,
    prepare_stop_design,
    collect_stop_data,
)

from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import (
    glm_runner,
)

from decision_making_analysis.data_compilation import (
    miss_events_class,
)

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import assemble_stop_design


class FullSessionGLMRunner:

    def __init__(
        self,
        raw_data_folder_path="all_monkey_data/raw_monkey_data/monkey_Bruno/data_0330",
        bin_width=0.04,
        planning_data_by_point_exists_ok=True,
        reuse_cached=True,
        save_cached=True,
        cache_dir=None,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width

        self.planning_data_by_point_exists_ok = planning_data_by_point_exists_ok
        self.reuse_cached = reuse_cached
        self.save_cached = save_cached
        self._user_cache_dir = cache_dir

        self.pn = None
        self.global_bins_2d = None

        self.fs_design_df = None
        self.pn_design_df_sub = None
        self.stop_design_df_sub = None

        self.merged_design_df = None
        self.merged_meta_groups = None

        self.df_X = None
        self.df_Y = None

        self.pipeline = None
        self.cache_paths = None

    # =====================
    # Public API
    # =====================
    def run(self, show_plots: bool = False, hyperparam_tuning: bool = True):
        self.assemble_merged_designs()
        # self._fit_glm()
        self._fit_glm_sel_cols(show_plots=show_plots, hyperparam_tuning=hyperparam_tuning)
        print('done')  # preserve original behavior


    def assemble_merged_designs(self):
        if self._try_load_cached():
            return
        self._build_full_session_and_pn_design()
        self._build_best_arc_design()
        self._build_stop_design()
        self._merge_designs()
        self._save_cached()

    # =====================
    # Cache helpers
    # =====================
    def _get_cache_dir(self):
        # Prefer user-specified directory; otherwise place in planning_and_neural/full_session/cache
        if self._user_cache_dir is not None:
            self.cache_dir = self._user_cache_dir
        else:
            self.cache_dir = os.path.join(
                self.pn.planning_and_neural_folder_path, 'design_df', 'full_session')
        os.makedirs(self.cache_dir, exist_ok=True)
        return self.cache_dir

    @staticmethod
    def _dt_tag(bin_width):
        # e.g., 0.04 -> "0p04"
        return str(bin_width).replace('.', 'p')

    def _get_cache_paths(self):
        if self.cache_paths is not None:
            return self.cache_paths
        self._get_cache_dir()
        tag = self._dt_tag(self.bin_width)
        paths = {
            'merged_design': os.path.join(self.cache_dir, f'merged_design_{tag}.csv'),
            'binned_spikes': os.path.join(self.cache_dir, f'binned_spikes_{tag}.csv'),
            'bin_df': os.path.join(self.cache_dir, f'bin_df_{tag}.csv'),
            'merged_meta': os.path.join(self.cache_dir, f'merged_meta_{tag}.json'),
        }
        self.cache_paths = paths
        return paths

    def _try_load_cached(self):

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        paths = self._get_cache_paths()
        if not self.reuse_cached:
            return False
        want = ['merged_design', 'binned_spikes', 'bin_df', 'merged_meta']
        if not all(os.path.exists(paths[k]) for k in want):
            print(f"Missing cache files: {want}. Will build from scratch.")
            return False
        # Load
        print(f"Loading cached merged_design_df from {paths['merged_design']}")
        self.merged_design_df = pd.read_csv(paths['merged_design'])
        print(f"Loading cached binned_spikes from {paths['binned_spikes']}")
        self.binned_spikes = pd.read_csv(paths['binned_spikes'])
        print(f"Loading cached bin_df from {paths['bin_df']}")
        self.bin_df = pd.read_csv(paths['bin_df'])
        print(f"Loading cached merged_meta_groups from {paths['merged_meta']}")
        with open(paths['merged_meta'], 'r') as f:
            self.merged_meta_groups = json.load(f)
        # Set df_X/df_Y
        self.df_X = self.merged_design_df.copy()
        self.df_Y = self.binned_spikes.copy()
        self.global_bins_2d = self.bin_df[['bin_left', 'bin_right']].values

        assert len(self.df_X) == len(self.df_Y) == len(self.bin_df)
        assert np.all(self.df_X['bin'].values == self.bin_df['new_bin'].values)

        self.pn._make_spikes_df()

        return True

    def _save_cached(self):
        if not self.save_cached:
            return
        paths = self._get_cache_paths()
        if self.merged_design_df is not None:
            self.merged_design_df.to_csv(paths['merged_design'], index=False)
            print(f"Saved merged_design_df to {paths['merged_design']}")
        if getattr(self, 'binned_spikes', None) is not None:
            self.binned_spikes.to_csv(paths['binned_spikes'], index=False)
            print(f"Saved binned_spikes to {paths['binned_spikes']}")
        if getattr(self, 'bin_df', None) is not None:
            self.bin_df.to_csv(paths['bin_df'], index=False)
            print(f"Saved bin_df to {paths['bin_df']}")
        if getattr(self, 'merged_meta_groups', None) is not None:
            with open(paths['merged_meta'], 'w') as f:
                json.dump(self.merged_meta_groups, f)
            print(f"Saved merged_meta_groups to {paths['merged_meta']}")

    # =====================
    # Stage 1
    # =====================
    def _build_full_session_and_pn_design(self):

        pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        pn.prep_data_to_analyze_planning(
            planning_data_by_point_exists_ok=self.planning_data_by_point_exists_ok
        )

        pn.rebin_data_in_new_segments(
            cur_or_nxt='cur',
            first_or_last='first',
            time_limit_to_count_sighting=2,
            start_t_rel_event=0,
            end_t_rel_event=1.5,
            rebinned_max_x_lag_number=2,
        )

        pn.make_or_retrieve_ff_dataframe()
        pn.monkey_information = pn_utils.add_ff_visible_or_in_memory_info_by_point(
            pn.monkey_information, pn.ff_dataframe
        )

        # new_seg_info here is basically the whole session
        new_seg_info_fs = pd.DataFrame({
            'new_segment': 0,
            'new_seg_start_time': max(0, pn.ff_caught_T_sorted.min() - 1),
            'new_seg_end_time': pn.ff_caught_T_sorted.max(),
            'new_seg_duration':
                pn.ff_caught_T_sorted.max()
                - max(0, pn.ff_caught_T_sorted.min() - 1),
        }, index=[0])

        rebinned_monkey_data, self.global_bins_2d = rebin_segments.rebin_all_segments_local_bins(
            pn.monkey_information,
            new_seg_info_fs,
            bin_width=self.bin_width,
            respect_old_segment=False,
            add_bin_edges=True,
        )

        trial_ids = np.repeat(0, len(rebinned_monkey_data))
        rebinned_monkey_data = temporal_feats.add_stop_and_capture_columns(
            rebinned_monkey_data, trial_ids, pn.ff_caught_T_new
        )

        fs_design_df, _, meta = (
            create_full_session_design
            .get_initial_full_session_design_df(
                rebinned_monkey_data, self.bin_width, trial_ids
            )
        )

        fs_design_df['bin'] = rebinned_monkey_data['new_bin']

        self.pn = pn
        self.fs_design_df = fs_design_df
        self.fs_meta_groups = meta['groups']

        # PN design
        pn_df, _ = rebin_segments.rebin_all_segments_global_bins(
            pn.planning_data_by_point,
            pn.new_seg_info,
            bins_2d=self.global_bins_2d,
            how='mean',
            respect_old_segment=True,
            require_full_bin=True,
            add_bin_edges=True,
            add_support_duration=True,
        )

        pn_df = temporal_feats.add_stop_and_capture_columns(
            pn_df, pn_df['new_segment'], pn.ff_caught_T_new
        )

        self.pn_design_df, _, self.pn_meta = create_pn_design_df.get_initial_design_df(
            pn_df, self.bin_width, pn_df['new_segment'])

        self.pn_design_df_sub = self.pn_design_df[
            selected_pn_design_features.pn_design_predictors
        ].copy()
        self.pn_design_df_sub['bin'] = pn_df['new_bin']
        self.pn_design_df_sub['in_pn_window'] = 1
        self.pn_meta_groups = self.pn_meta['groups']

    def _build_best_arc_design(self):
        self.mec = miss_events_class.MissEventsClass(
            raw_data_folder_path=self.raw_data_folder_path, time_range_of_trajectory=[-0.5, 2.5], num_time_points_for_trajectory=10)
        self.mec.get_monkey_data(
            already_retrieved_ok=True, include_ff_dataframe=True)
        self.mec.ff_dataframe = self.mec.ff_dataframe[abs(
            self.mec.ff_dataframe['ff_angle_boundary']) <= math.pi/4]
        self.mec.ff_dataframe = self.mec.ff_dataframe[self.mec.ff_dataframe['time_since_last_vis'] <= 2.5]
        self.mec.make_curvature_df([-25, 25], curv_of_traj_mode='distance')
        self.mec.eliminate_crossing_boundary_cases(
            n_seconds_after_crossing_boundary=0.5)
        self.mec.make_or_retrieve_best_arc_df()

        best_arc_df_sub = self.mec.best_arc_df[[
            'ff_index', 'ff_distance', 'ff_angle', 'opt_arc_curv', 'opt_arc_length', 'curv_diff', 'abs_curv_diff']].copy()
        # add 'ba_' prefix to the column names
        best_arc_df_sub.columns = ['best_arc_' +
                                   c for c in best_arc_df_sub.columns]
        best_arc_df_sub['point_index'] = self.mec.best_arc_df['point_index']
        best_arc_df_sub['time'] = self.mec.best_arc_df['time']

        rebinned_best_arc_df, best_arc_bin_edges = rebin_segments.rebin_all_segments_global_bins_pick_point(
            best_arc_df_sub, new_seg_info, bins_2d=self.global_bins_2d, respect_old_segment=False,
            add_bin_edges=True,
        )

        self.best_arc_design_df, self.best_arc_ff_meta0, self.best_arc_ff_meta = create_best_arc_design.get_best_arc_design_df(
            rebinned_best_arc_df, self.bin_width)
        self.best_arc_design_df['bin'] = rebinned_best_arc_df['new_bin']
        self.best_arc_meta_groups = self.best_arc_ff_meta['groups']

        self.best_arc_design_df_sub = self.best_arc_design_df.drop(columns=[
                                                                   'const'], errors='ignore')
        self.best_arc_design_df_sub['having_best_arc_ff'] = 1

    def _check_meta_groups(self):

        if len(self.merged_meta_groups) != (
            len(self.fs_meta_groups)
            + len(self.best_arc_meta_groups)
            + len(self.pn_meta_groups)
            + len(self.stop_meta_groups)
        ):
            print(
                f'Meta groups length mismatch: {len(self.merged_meta_groups)} != {len(self.fs_meta_groups) + len(self.best_arc_meta_groups) + len(self.pn_meta_groups) + len(self.stop_meta_groups)}')
            print(f'fs_meta_groups: {self.fs_meta_groups}')
            print(f'best_arc_meta_groups: {self.best_arc_meta_groups}')
            print(f'pn_meta_groups: {self.pn_meta_groups}')
            print(f'stop_meta_groups: {self.stop_meta_groups}')
            print(f'merged_meta_groups: {self.merged_meta_groups}')

        missing = [k for k in self.merged_meta_groups.keys()
                   if k not in self.df_X.columns]
        if len(missing) > 0:
            print(f'Meta groups contain missing predictors: {missing}')
            print(f'df_X columns: {self.df_X.columns}')
            print(f'merged_meta_groups keys: {self.merged_meta_groups.keys()}')

    # =====================
    # Stage 3
    # =====================
    def _merge_designs(self):

        # Otherwise compute and then save
        self.merged_design_df = create_full_session_design.merge_design_blocks(
            self.fs_design_df,
            self.best_arc_design_df_sub,
            self.pn_design_df_sub,
            self.stop_design_df_sub,
        )

        # Always (re)compose meta groups from the components we just built
        self.merged_meta_groups = {
            **self.fs_meta_groups,
            **self.best_arc_meta_groups,
            **self.pn_meta_groups,
            **self.stop_meta_groups,
        }

        spike_counts, cluster_ids = event_binning.bin_spikes_by_cluster(
            self.pn.spikes_df,
            self.global_bins_2d,
            time_col='time',
            cluster_col='cluster',
        )

        self.binned_spikes = (
            pd.DataFrame(spike_counts, columns=cluster_ids)
            .reset_index(drop=True)
        )

        # Assign X/Y and build bin_df
        self.df_X = self.merged_design_df.copy()
        self.df_Y = self.binned_spikes.copy()
        self._make_bin_df()

        self._check_meta_groups()

    def _make_bin_df(self):
        self.bin_df = self.df_X[['bin']].rename(columns={'bin': 'new_bin'})
        self.bin_df['new_segment'] = 0
        self.bin_df['bin_left'] = self.global_bins_2d[:, 0]
        self.bin_df['bin_right'] = self.global_bins_2d[:, 1]
        # assert that self.bin_df['bin'] is monotonically increasing
        assert self.bin_df['new_bin'].is_monotonic_increasing, "bin_df['bin'] is not monotonically increasing"

    # =====================
    # Stage 4
    # =====================
    def _fit_glm(self):

        output_root = os.path.join(
            self.pn.planning_and_neural_folder_path,
            'full_session'
        )

        self.pipeline = glm_runner.GLMPipeline(
            spikes_df=self.pn.spikes_df,
            bin_df=self.bin_df,
            df_X=self.df_X,
            df_Y=self.df_Y,
            meta_groups=self.merged_meta_groups,
            bin_width=self.bin_width,
            output_root=output_root,
            cv_splitter='blocked_time_buffered'
        )

        self.pipeline.run()
        self.pipeline.plot_comparisons()

    def _fit_glm_sel_cols(self, show_plots: bool = False, hyperparam_tuning: bool = True):
        df_X = self.merged_design_df[select_fs_features.ALL_REGRESSORS].copy()
        df_Y = self.binned_spikes.copy()

        output_root = os.path.join(
            self.pn.planning_and_neural_folder_path, 'full_session_sel_cols')

        self.pipeline_sel_cols = glm_runner.GLMPipeline(
            spikes_df=self.pn.spikes_df,
            bin_df=self.bin_df,
            df_X=df_X,
            df_Y=df_Y,
            meta_groups=self.merged_meta_groups,
            bin_width=self.bin_width,
            output_root=output_root,
            cv_splitter='blocked_time_buffered'
        )

        self.pipeline_sel_cols.run(
            glm_results_exists_ok=False, pruned_columns_exists_ok=True,
            show_plots=show_plots,
            hyperparam_tuning=hyperparam_tuning,
        )
        self.pipeline_sel_cols.plot_comparisons()

    def _build_stop_design(self):

        pn, stop_binned_spikes, stop_binned_feats, offset_log, stop_meta_used, stop_meta_groups = assemble_stop_design.assemble_stop_design_func(
            self.raw_data_folder_path,
            self.bin_width,
            self.global_bins_2d,
        )

        self.stop_design_df_sub = stop_binned_feats[
            selected_stop_design_features.stop_design_predictors
        ].copy()
        self.stop_design_df_sub['bin'] = stop_meta_used['global_bin']
        self.stop_design_df_sub['in_stop_window'] = 1
        self.stop_meta_groups = stop_meta_groups
