from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseDecodingRunner,
)

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_design_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers.decoding_design_utils import (
    STOP_DECODING_VAR_CATEGORIES,
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
        var_categories=None,
    ):
        super().__init__(bin_width=bin_width)
        self.raw_data_folder_path = raw_data_folder_path
        self.var_categories = var_categories if var_categories is not None else STOP_DECODING_VAR_CATEGORIES

        self.meta_df_used = None
        self.feats_to_decode = None
        self.binned_spikes = None

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
        self.target_df = decoding_design_utils.truncate_columns_to_percentiles(self.target_df, ['time_since_prev_event'])
        return self.target_df

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
        return "feats_to_decode"

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
            self.feats_to_decode,
            _offset_log,
            self.meta_df_used,
        ) = decode_stops_design.assemble_stop_decoding_design(
            self.raw_data_folder_path,
            self.bin_width,
        )

        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(self.feats_to_decode)
 
        self.reduce_binned_spikes()
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
            "feats_to_decode": save_dir / "feats_to_decode.pkl",
            "meta_df_used": save_dir / "meta_df_used.pkl",
        }

    def _get_design_matrix_data(self):
        return {
            "binned_spikes": self.binned_spikes,
            "feats_to_decode": self.feats_to_decode,
            "meta_df_used": self.meta_df_used,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            "binned_spikes": "binned_spikes",
            "feats_to_decode": "feats_to_decode",
            "meta_df_used": "meta_df_used",
        }
