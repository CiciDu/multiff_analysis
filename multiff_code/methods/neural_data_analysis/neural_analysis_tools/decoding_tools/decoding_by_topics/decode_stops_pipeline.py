from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseDecodingRunner,
)

DEFAULT_STOP_CANONCORR_VARS = [
    "speed",
    "ang_speed",
    "accel",
    "time_rel_to_event_start",
    "cluster_progress_c",
]


class StopDecodingRunner(BaseDecodingRunner):
    """
    Stop decoding runner. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width: float = 0.04,
    ):
        super().__init__(bin_width=bin_width)
        self.raw_data_folder_path = raw_data_folder_path
        

        self.meta_df_used = None
        self.stop_feats_to_decode = None
        self.binned_spikes = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

    # ------------------------------------------------------------------
    # BaseDecodingRunner override points
    # ------------------------------------------------------------------
    def _get_target_df(self) -> pd.DataFrame:
        return self.stop_feats_to_decode

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        vars_found = [v for v in DEFAULT_STOP_CANONCORR_VARS if v in y_df.columns]
        return vars_found if len(vars_found) > 0 else list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "stop_feats_to_decode"

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_data(self, exists_ok: bool = True):
        if exists_ok and self._load_design_matrices():
            print("[StopDecodingRunner] Using cached design matrices")
            return

        print("[StopDecodingRunner] Computing design matrices from scratch")
        (
            self.pn,
            self.binned_spikes,
            self.stop_feats_to_decode,
            _offset_log,
            self.meta_df_used,
            _stop_meta_groups,
        ) = decode_stops_design.assemble_stop_decoding_design(
            self.raw_data_folder_path,
            self.bin_width,
        )

        self._save_design_matrices()

    # ------------------------------------------------------------------
    # Caching utilities (BaseDecodingRunner interface)
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            "decoding_outputs/stop_decoder_outputs",
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            "binned_spikes": save_dir / "binned_spikes.pkl",
            "stop_feats_to_decode": save_dir / "stop_feats_to_decode.pkl",
            "meta_df_used": save_dir / "meta_df_used.pkl",
        }

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "stop_feats_to_decode": self.stop_feats_to_decode,
            "meta_df_used": self.meta_df_used,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            "binned_spikes": "binned_spikes",
            "stop_feats_to_decode": "stop_feats_to_decode",
            "meta_df_used": "meta_df_used",
        }
