from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)

from neural_data_analysis.design_kits.design_by_segment import spike_history

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseDecodingRunner,
)

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_design_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers.decoding_design_utils import (
    FS_DECODING_VAR_CATEGORIES,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import detrend_neural_data

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import rebinned_alignment
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decode_fs_utils


DEFAULT_FS_CANONCORR_VARS = [
    "speed",
    "ang_speed",
    "accel",
]


class FSDecodingRunner(BaseDecodingRunner):
    """
    Full session decoding self. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width: float = 0.04,
        var_categories=None,
        **kwargs,
    ):
        super().__init__(bin_width=bin_width, **kwargs)
        self.raw_data_folder_path = raw_data_folder_path
        self.var_categories = var_categories if var_categories is not None else FS_DECODING_VAR_CATEGORIES


        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

    # ------------------------------------------------------------------
    # BaseDecodingRunner override points
    # ------------------------------------------------------------------
    def get_target_df(self) -> pd.DataFrame:
        if getattr(self, 'feats_to_decode', None) is None:
            self.collect_data(exists_ok=True)
        self.target_df = self.feats_to_decode.copy()
        if 'time_since_prev_event' in self.target_df.columns:
            self.target_df = decoding_design_utils.truncate_columns_to_percentiles(self.target_df, ['time_since_prev_event'])
        return self.target_df

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        vars_found = [v for v in DEFAULT_FS_CANONCORR_VARS if v in y_df.columns]
        return vars_found if len(vars_found) > 0 else list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "feats_to_decode"

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_data(self, exists_ok: bool = True):
        if exists_ok and self._load_design_matrices():
            self.clean_var_categories()
            print("[FSDecodingRunner] Using cached design matrices")
            return
        
        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )


        # if not hasattr(self.pn, 'ff_dataframe'):
        #     self.pn.make_or_retrieve_ff_dataframe()

        # self.pn.monkey_information = decode_fs_utils.add_time_since_capture_or_stop(self.pn.monkey_information)
        # self.add_columns_related_to_time_since_ff_visible()

        # (
        #     self.bins_2d,
        #     _meta,
        #     binned_feats,
        #     _exposure,
        #     _used_bins,
        #     _mask_used,
        #     self.pos,
        #     self.meta_df_used,
        # ) = encoding_design_utils.bin_event_windows_core(
        #     new_seg_info=self.new_seg_info,
        #     monkey_information=pn.monkey_information,
        #     bin_dt=self.bin_width,
        #     verbose=True,
        #     tile_window=True,
        #     agg_cols=decoding_design_utils.ONE_FF_STYLE_DECODING_COLS + ['time_since_prev_stop', 'time_since_prev_capture', 
        #                                                                  'time_since_prev_ff_visible', 'time_since_global_burst_start'],
        # )
        
        # binned_feats = self.add_columns_related_to_ff_visibility(binned_feats)
        (
            binned_feats,
            self.meta_df_used,
            self.bins_2d,
            self.pos,
            self.new_seg_info,
        ) = decode_fs_utils.build_fs_design_decoding(self.pn)

        if self.use_spike_history:
            self.bin_df = spike_history.make_bin_df_from_meta_df(self.meta_df_used)

        self.feats_to_decode = decode_stops_design.scale_binned_feats(
            binned_feats,
        )

        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(self.feats_to_decode)

        self.get_binned_spikes()

        self.reduce_binned_spikes()
        self._save_design_matrices()
        self.clean_var_categories()





    def build_fs_design_decoding(self):
        pass


    def get_binned_spikes(self):
        self.get_processed_spike_rates()

        new_seg_for_rebin = self.new_seg_info.copy()
        if 'new_segment' not in new_seg_for_rebin.columns:
            new_seg_for_rebin['new_segment'] = np.arange(len(new_seg_for_rebin))
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
            binned_counts, _cluster_ids = encoding_design_utils.bin_spikes_for_event_windows(
                self.pn.spikes_df,
                self.bins_2d,
                self.pos,
                time_col='time',
                cluster_col='cluster',
            )
            self.binned_spikes = (binned_counts / self.bin_width).copy()
            if self.drop_bad_neurons:
                self.binned_spikes = detrend_neural_data.drop_nonstationary_neurons(
                    self.binned_spikes
                )


    # ------------------------------------------------------------------
    # Caching utilities (BaseDecodingRunner interface)
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return self._get_save_dir_common("fs_decoder_outputs")

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            "binned_spikes": save_dir / "binned_spikes.pkl",
            "feats_to_decode": save_dir / "feats_to_decode.pkl",
            "meta_df_used": save_dir / "meta_df_used.pkl",
            "bin_df": save_dir / "bin_df.pkl",
        }

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "feats_to_decode": self.feats_to_decode,
            "meta_df_used": self.meta_df_used,
            "bin_df": self.bin_df,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            "binned_spikes": "binned_spikes",
            "feats_to_decode": "feats_to_decode",
            "meta_df_used": "meta_df_used",
            "bin_df": "bin_df",
        }
