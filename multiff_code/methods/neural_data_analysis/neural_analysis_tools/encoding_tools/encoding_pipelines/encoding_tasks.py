"""
Concrete encoding task classes.

Each class owns data loading + feature construction for one paradigm.
No model fitting lives here.

Usage
-----
    task = PNEncodingTask(raw_data_folder_path)
    task.collect_data()
    # task.binned_feats, task.binned_spikes now available
"""

from __future__ import annotations

import os

from neural_data_analysis.design_kits.design_by_segment import (
    spike_history,
    spatial_feats,
    temporal_feats,
    create_pn_design_df,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoder_gam_helper,
    encode_stops_utils,
    encode_fs_utils,
    encode_pn_utils,
    encoding_design_utils,
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
    decode_stops_design,
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils

from .base_encoding_task import BaseEncodingTask


# ---------------------------------------------------------------------------
# PNEncodingTask
# ---------------------------------------------------------------------------

class PNEncodingTask(BaseEncodingTask):
    """Planning-and-neural (PN) encoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, encoder_prs=None, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width, encoder_prs=encoder_prs)
        self.var_categories = encoder_gam_helper.PN_ENCODING_VAR_CATEGORIES
        print("var_categories:", self.var_categories)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_to_collect_data(self):
        print("[PNEncodingTask] Computing design matrices from scratch")

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )
        self.pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=True)
        self.pn.rebin_data_in_new_segments(
            cur_or_nxt="cur",
            first_or_last="first",
            time_limit_to_count_sighting=2,
            start_t_rel_event=0,
            end_t_rel_event=1.5,
            rebinned_max_x_lag_number=2,
        )
        self.pn.global_bins_2d = self.pn.local_bin_edges

    def _make_binned_feats(self):
        rebinned_y_var = self.pn.rebinned_y_var.copy()
        trial_ids = rebinned_y_var["new_segment"]

        rebinned_y_var = temporal_feats.add_stop_and_capture_columns(
            rebinned_y_var, trial_ids, self.pn.ff_caught_T_new
        )
        rebinned_y_var = spatial_feats.add_transition_columns(
            rebinned_y_var,
            rebinned_y_var["new_segment"].values,
            stems=("cur_vis", "nxt_vis"),
            inplace=False,
        )
        self.rebinned_y_var = rebinned_y_var.copy()

        cluster_cols = [c for c in self.pn.rebinned_x_var.columns if c.startswith("cluster_")]
        self.binned_spikes = self.pn.rebinned_x_var[cluster_cols]
        self.binned_spikes.columns = (
            self.binned_spikes.columns.str.replace("cluster_", "").astype(int)
        )

        design_kwargs = self._encoding_design_kwargs()

        linear_vars = (
            encoding_design_utils.DEFAULT_TUNING_VARS_NO_WRAP
            + ["time_since_target_last_seen", "cur_ff_distance", "nxt_ff_distance",
               "cur_ff_angle", "nxt_ff_angle"]
        )

        self.binned_feats, self.temporal_meta, self.tuning_meta, self.raw_behavioral_feats = (
            encode_pn_utils.build_pn_encoding_design(
                rebinned_y_var,
                self.pn.monkey_information,
                global_bins_2d=self.pn.global_bins_2d,
                bin_width=self.pn.bin_width,
                ff_caught_T_new=self.pn.ff_caught_T_new,
                linear_vars=linear_vars,
                **design_kwargs,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[PNEncodingTask] Using cached design matrices")
            return

        self._prepare_to_collect_data()
        self._make_binned_feats()

        data = self.pn.rebinned_y_var.copy()

        cluster_cols = [c for c in self.pn.rebinned_x_var.columns if c.startswith("cluster_")]
        self.binned_spikes = self.pn.rebinned_x_var[cluster_cols]
        self.binned_spikes.columns = (
            self.binned_spikes.columns.str.replace("cluster_", "").astype(int)
        )
        self.binned_spikes = self.binned_spikes.reset_index(drop=True)

        self.trial_ids = data["new_segment"]
        self.bin_df = create_pn_design_df.make_bin_df_for_pn(
            self.pn.rebinned_x_var, self.pn.local_bin_edges
        )
        self.binned_feats = self.binned_feats.reset_index(drop=True)
        if self.raw_behavioral_feats is not None:
            self.raw_behavioral_feats = self.raw_behavioral_feats.reset_index(drop=True)

        self._finalize_collect_data()

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            "encoding_outputs/pn_encoder_outputs",
        )
        
    def get_gam_results_subdir(self):
        return "pn_gam_results"


# ---------------------------------------------------------------------------
# FSEncodingTask
# ---------------------------------------------------------------------------

class FSEncodingTask(BaseEncodingTask):
    """Full-session (FS) encoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, encoder_prs=None, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width, encoder_prs=encoder_prs)
        self.var_categories = encoder_gam_helper.FS_ENCODING_VAR_CATEGORIES
        print("var_categories:")
        for k, v in self.var_categories.items():
            print(f"  {k}: {v}")

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[FSEncodingTask] Using cached design matrices")
            return

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        print("[FSEncodingTask] Computing design matrices from scratch")
        design_kwargs = self._encoding_design_kwargs()

        (
            self.pn,
            self.binned_spikes,
            self.binned_feats,
            self.meta_df_used,
            self.temporal_meta,
            self.tuning_meta,
            self.raw_behavioral_feats,
        ) = encode_fs_utils.build_fs_encoding_design(
            self.pn, bin_width=self.bin_width, **design_kwargs
        )

        self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self._finalize_collect_data()

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            "encoding_outputs/fs_encoder_outputs",
        )
        
    def get_gam_results_subdir(self):
        return "fs_gam_results"


# ---------------------------------------------------------------------------
# StopEncodingTask
# ---------------------------------------------------------------------------

class StopEncodingTask(BaseEncodingTask):
    """Stop-event encoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, encoder_prs=None, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width, encoder_prs=encoder_prs)
        self.var_categories = encoder_gam_helper.STOP_ENCODING_VAR_CATEGORIES
        print("var_categories:")
        for k, v in self.var_categories.items():
            print(f"  {k}: {v}")

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[StopEncodingTask] Using cached design matrices")
            return

        self.pn, datasets, new_seg_info, events_with_stats = (
            decode_stops_design.prepare_stop_design_inputs(
                self.raw_data_folder_path, self.bin_width
            )
        )

        print("[StopEncodingTask] Computing design matrices from scratch")
        design_kwargs = self._encoding_design_kwargs()

        (
            self.pn,
            self.binned_spikes,
            self.binned_feats,
            self.meta_df_used,
            self.temporal_meta,
            self.tuning_meta,
            self.raw_behavioral_feats,
        ) = encode_stops_utils.build_stop_encoding_design(
            self.pn, datasets, new_seg_info, events_with_stats, self.bin_width, **design_kwargs
        )

        self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self._finalize_collect_data()

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            "encoding_outputs/stop_encoder_outputs",
        )
        
    def get_gam_results_subdir(self):
        return "stop_gam_results"


# ---------------------------------------------------------------------------
# VisEncodingTask
# ---------------------------------------------------------------------------

class VisEncodingTask(BaseEncodingTask):
    """Firefly-visibility (Vis) encoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, encoder_prs=None, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width, encoder_prs=encoder_prs)
        self.var_categories = encoder_gam_helper.VIS_ENCODING_VAR_CATEGORIES
        print("var_categories:")
        for k, v in self.var_categories.items():
            print(f"  {k}: {v}")

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[VisEncodingTask] Using cached design matrices")
            return

        self.pn, datasets, _ = collect_stop_data.collect_stop_data_func(
            self.raw_data_folder_path, bin_width=self.bin_width
        )
        self.pn.make_or_retrieve_ff_dataframe()

        new_seg_info, events_with_stats = decode_vis_utils.prepare_new_seg_info(
            self.pn.ff_dataframe, self.pn.bin_width
        )

        print("[VisEncodingTask] Computing design matrices from scratch")
        design_kwargs = self._encoding_design_kwargs()

        self.ff_on_df, self.group_on_df = decode_vis_utils.extract_ff_visibility_tables_fast(
            self.pn.ff_dataframe
        )

        (
            self.pn,
            self.binned_spikes,
            self.binned_feats,
            self.meta_df_used,
            self.temporal_meta,
            self.tuning_meta,
            self.raw_behavioral_feats,
        ) = encoding_design_utils.build_vis_encoding_design(
            self.pn,
            datasets,
            new_seg_info,
            events_with_stats,
            self.ff_on_df,
            self.group_on_df,
            self.bin_width,
            add_stop_cluster_features=False,
            add_retry_features=False,
            **design_kwargs,
        )

        self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self._finalize_collect_data()

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            "encoding_outputs/vis_encoder_outputs",
        )

    def get_gam_results_subdir(self):
        return "vis_gam_results"