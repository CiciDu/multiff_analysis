"""
Concrete decoding task classes.

Each owns data loading + feature construction for one paradigm.
No model fitting lives here.

Usage
-----
    task = StopDecodingTask(raw_data_folder_path)
    task.collect_data()
    # task.binned_spikes, task.feats_to_decode now available
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from neural_data_analysis.design_kits.design_around_event import event_binning
from neural_data_analysis.design_kits.design_by_segment import (
    spike_history,
    temporal_feats,
    create_pn_design_df,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decoding_design_utils,
    decode_pn_utils,
    decode_fs_utils,
    rebinned_alignment,
    detrend_neural_data,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers.decoding_design_utils import (
    VIS_DECODING_VAR_CATEGORIES,
    STOP_DECODING_VAR_CATEGORIES,
    PN_DECODING_VAR_CATEGORIES,
    FS_DECODING_VAR_CATEGORIES,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
    decode_stops_design,
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import (
    decode_vis_design,
    decode_vis_utils,
)

from .base_decoding_task import BaseDecodingTask


DEFAULT_STOP_CANONCORR_VARS = [
    "speed", "ang_speed", "accel",
    "time_rel_to_event_start", "cluster_progress_c",
]
DEFAULT_FS_CANONCORR_VARS = ["speed", "ang_speed", "accel"]


# ---------------------------------------------------------------------------
# StopDecodingTask
# ---------------------------------------------------------------------------

class StopDecodingTask(BaseDecodingTask):
    """Stop-event decoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width,
                         var_categories=STOP_DECODING_VAR_CATEGORIES, **kwargs)

    def get_target_df(self) -> pd.DataFrame:
        if self.feats_to_decode is None:
            self.collect_data(exists_ok=True)
        target = self.feats_to_decode.copy()
        return decoding_design_utils.truncate_columns_to_percentiles(
            target, ["time_since_prev_event"]
        )

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        found = [v for v in DEFAULT_STOP_CANONCORR_VARS if v in y_df.columns]
        return found if found else list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[StopDecodingTask] Using cached design matrices")
            return

        print("[StopDecodingTask] Computing design matrices from scratch")

        self.pn, datasets, new_seg_info, events_with_stats = (
            decode_stops_design.prepare_stop_design_inputs(
                self.raw_data_folder_path, self.bin_width
            )
        )

        self.get_processed_spike_rates()

        rebinned_spike_rates, binned_feats, _, self.meta_df_used = (
            decode_stops_design.build_stop_design_decoding(
                new_seg_info=new_seg_info,
                events_with_stats=events_with_stats,
                monkey_information=self.pn.monkey_information,
                ff_dataframe=self.pn.ff_dataframe,
                spikes_df=self.pn.spikes_df,
                processed_spike_rates=self.processed_spike_rates,
                bin_dt=self.bin_width,
                datasets=datasets,
            )
        )

        self.feats_to_decode = decode_stops_design.scale_binned_feats(binned_feats)
        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(
            self.feats_to_decode
        )

        id_cols = {"new_segment", "new_bin", "new_seg_start_time", "new_seg_end_time", "new_seg_duration"}
        cluster_cols = [c for c in rebinned_spike_rates.columns if c not in id_cols]
        self.binned_spikes = rebinned_spike_rates[cluster_cols].copy()

        if self.use_spike_history:
            self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self.reduce_binned_spikes()
        self._save_design_matrices()
        self.clean_var_categories()

    def _get_save_dir(self):
        return self._get_save_dir_common("stop_decoder_outputs")

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {k: save_dir / f"{k}.pkl"
                for k in ["binned_spikes", "feats_to_decode", "meta_df_used", "bin_df"]}

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "feats_to_decode": self.feats_to_decode,
            "meta_df_used": self.meta_df_used,
            "bin_df": self.bin_df,
        }

    def _get_design_matrix_key_to_attr(self):
        return {k: k for k in ["binned_spikes", "feats_to_decode", "meta_df_used", "bin_df"]}


# ---------------------------------------------------------------------------
# VisDecodingTask
# ---------------------------------------------------------------------------

class VisDecodingTask(BaseDecodingTask):
    """Firefly-visibility decoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, datasets=None, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width,
                         var_categories=VIS_DECODING_VAR_CATEGORIES, **kwargs)
        self.datasets = datasets

    def get_target_df(self) -> pd.DataFrame:
        if self.feats_to_decode is None:
            self.collect_data(exists_ok=True)
        return self.feats_to_decode.copy()

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[VisDecodingTask] Using cached design matrices")
            return

        self.pn = collect_stop_data.init_pn_to_collect_stop_data(
            self.raw_data_folder_path, bin_width=0.04
        )
        self.pn.make_or_retrieve_ff_dataframe()

        print("[VisDecodingTask] Computing design matrices from scratch")

        new_seg_info, events_with_stats = decode_vis_utils.prepare_new_seg_info(
            self.pn.ff_dataframe, self.pn.bin_width
        )

        self.get_processed_spike_rates()

        rebinned_spike_rates, binned_feats, _, self.meta_df_used = (
            decode_vis_design.build_vis_design_decoding(
                new_seg_info,
                events_with_stats,
                self.pn.monkey_information,
                self.pn.ff_dataframe,
                self.pn.spikes_df,
                processed_spike_rates=self.processed_spike_rates,
                datasets=self.datasets,
                bin_dt=self.pn.bin_width,
                add_ff_visible_info=True,
                add_retries_info=False,
                drop_bad_neurons=self.drop_bad_neurons,
            )
        )

        if "global_burst_id" not in self.meta_df_used.columns:
            self.meta_df_used = self.meta_df_used.merge(
                new_seg_info[["event_id", "global_burst_id"]],
                on="event_id", how="left",
            )

        feats_to_decode, _ = event_binning.selective_zscore(binned_feats)
        feats_to_decode = sm.add_constant(feats_to_decode, has_constant="add")
        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(feats_to_decode)

        id_cols = {"new_segment", "new_bin", "new_seg_start_time", "new_seg_end_time", "new_seg_duration"}
        cluster_cols = [c for c in rebinned_spike_rates.columns if c not in id_cols]
        self.binned_spikes = rebinned_spike_rates[cluster_cols].copy()

        if self.use_spike_history:
            self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self.reduce_binned_spikes()
        self._save_design_matrices()
        self.clean_var_categories()

    def _get_save_dir(self):
        return self._get_save_dir_common("vis_decoder_outputs")

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {k: save_dir / f"{k}.pkl"
                for k in ["binned_spikes", "feats_to_decode", "meta_df_used", "bin_df"]}

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "feats_to_decode": self.feats_to_decode,
            "meta_df_used": self.meta_df_used,
            "bin_df": self.bin_df,
        }

    def _get_design_matrix_key_to_attr(self):
        return {k: k for k in ["binned_spikes", "feats_to_decode", "meta_df_used", "bin_df"]}


# ---------------------------------------------------------------------------
# PNDecodingTask
# ---------------------------------------------------------------------------

class PNDecodingTask(BaseDecodingTask):
    """Planning-and-neural (PN) decoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width,
                         var_categories=PN_DECODING_VAR_CATEGORIES, **kwargs)

    def get_target_df(self) -> pd.DataFrame:
        if self.feats_to_decode is None:
            self.collect_data(exists_ok=True)
        return self.feats_to_decode.copy()

    def _get_groups(self):
        return self.trial_ids

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[PNDecodingTask] Using cached design matrices")
            return

        print("[PNDecodingTask] Computing design matrices from scratch")

        pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )
        pn.prep_data_to_analyze_planning(planning_data_by_point_exists_ok=True)
        pn.rebin_data_in_new_segments(
            cur_or_nxt="cur",
            first_or_last="first",
            time_limit_to_count_sighting=2,
            start_t_rel_event=0,
            end_t_rel_event=1.5,
            rebinned_max_x_lag_number=2,
        )

        data = pn.rebinned_y_var.copy()
        self.meta_df_used = (
            pn.rebinned_y_var[["new_segment", "bin_in_new_seg"]]
            .rename(columns={"new_segment": "event_id", "bin_in_new_seg": "k_within_seg"})
        )
        trial_ids = data["new_segment"]

        if "t_center" in data.columns:
            data["time"] = data["t_center"]

        self.feats_to_decode = temporal_feats.add_stop_and_capture_columns(
            data, trial_ids, pn.ff_caught_T_new
        )

        self.pn = pn
        self.get_processed_spike_rates()
        self._get_binned_spikes_pn()

        if self.use_spike_history:
            self.bin_df = create_pn_design_df.make_bin_df_for_pn(
                pn.rebinned_x_var, pn.local_bin_edges
            )

        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(
            self.feats_to_decode
        )

        detrend_dict = {}
        if "t_center" in pn.rebinned_y_var.columns:
            detrend_dict["time"] = np.asarray(pn.rebinned_y_var["t_center"], dtype=float)
        if "new_segment" in pn.rebinned_y_var.columns:
            detrend_dict["trial_index"] = np.asarray(pn.rebinned_y_var["new_segment"], dtype=float)
        self.detrend_covariates = pd.DataFrame(detrend_dict) if detrend_dict else None

        self.trial_ids = trial_ids

        self.reduce_binned_spikes()
        self._save_design_matrices()
        self.clean_var_categories()

    def _get_binned_spikes_pn(self):
        self.get_processed_spike_rates()
        if self.processed_spike_rates is not None:
            self.binned_spikes = decode_pn_utils.rebin_processed_spike_rates(
                self.processed_spike_rates,
                self.pn.new_seg_info,
                self.pn.local_bin_edges,
                self.pn.rebinned_y_var,
            )
        else:
            self.binned_spikes = decode_pn_utils.rebinned_x_var_to_binned_spike_rates_hz(
                self.pn.rebinned_x_var,
                self.bin_width,
                drop_bad_neurons=self.drop_bad_neurons,
            )

    def _get_save_dir(self):
        return self._get_save_dir_common("pn_decoder_outputs")

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {k: save_dir / f"{k}.pkl"
                for k in ["binned_spikes", "feats_to_decode", "meta_df_used",
                           "trial_ids", "detrend_covariates", "bin_df"]}

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "feats_to_decode": self.feats_to_decode,
            "meta_df_used": self.meta_df_used,
            "trial_ids": self.trial_ids,
            "detrend_covariates": self.detrend_covariates,
            "bin_df": self.bin_df,
        }

    def _get_design_matrix_key_to_attr(self):
        return {k: k for k in ["binned_spikes", "feats_to_decode", "meta_df_used",
                                "trial_ids", "detrend_covariates", "bin_df"]}


# ---------------------------------------------------------------------------
# FSDecodingTask
# ---------------------------------------------------------------------------

class FSDecodingTask(BaseDecodingTask):
    """Full-session decoding task."""

    def __init__(self, raw_data_folder_path, bin_width=0.04, **kwargs):
        super().__init__(raw_data_folder_path, bin_width=bin_width,
                         var_categories=FS_DECODING_VAR_CATEGORIES, **kwargs)

    def get_target_df(self) -> pd.DataFrame:
        if self.feats_to_decode is None:
            self.collect_data(exists_ok=True)
        target = self.feats_to_decode.copy()
        if "time_since_prev_event" in target.columns:
            target = decoding_design_utils.truncate_columns_to_percentiles(
                target, ["time_since_prev_event"]
            )
        return target

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        found = [v for v in DEFAULT_FS_CANONCORR_VARS if v in y_df.columns]
        return found if found else list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[FSDecodingTask] Using cached design matrices")
            return

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        binned_feats, self.meta_df_used, self.bins_2d, self.pos, self.new_seg_info = (
            decode_fs_utils.build_fs_design_decoding(self.pn)
        )

        if self.use_spike_history:
            self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self.feats_to_decode = decode_stops_design.scale_binned_feats(binned_feats)
        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(
            self.feats_to_decode
        )

        self._get_binned_spikes_fs()

        self.reduce_binned_spikes()
        self._save_design_matrices()
        self.clean_var_categories()

    def _get_binned_spikes_fs(self):
        self.get_processed_spike_rates()

        new_seg_for_rebin = self.new_seg_info.copy()
        if "new_segment" not in new_seg_for_rebin.columns:
            new_seg_for_rebin["new_segment"] = np.arange(len(new_seg_for_rebin))
        merge_keys = decode_stops_design._prepare_merge_keys(
            self.meta_df_used, new_seg_for_rebin
        )

        if self.processed_spike_rates is not None:
            self.binned_spikes = rebinned_alignment.rebin_then_align_spike_rates(
                self.processed_spike_rates,
                new_seg_for_rebin,
                self.bins_2d,
                merge_keys,
            )
        else:
            binned_counts, _ = encoding_design_utils.bin_spikes_for_event_windows(
                self.pn.spikes_df,
                self.bins_2d,
                self.pos,
                time_col="time",
                cluster_col="cluster",
            )
            self.binned_spikes = (binned_counts / self.bin_width).copy()
            if self.drop_bad_neurons:
                self.binned_spikes = detrend_neural_data.drop_nonstationary_neurons(
                    self.binned_spikes
                )

    def _get_save_dir(self):
        return self._get_save_dir_common("fs_decoder_outputs")

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {k: save_dir / f"{k}.pkl"
                for k in ["binned_spikes", "feats_to_decode", "meta_df_used", "bin_df"]}

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "feats_to_decode": self.feats_to_decode,
            "meta_df_used": self.meta_df_used,
            "bin_df": self.bin_df,
        }

    def _get_design_matrix_key_to_attr(self):
        return {k: k for k in ["binned_spikes", "feats_to_decode", "meta_df_used", "bin_df"]}
