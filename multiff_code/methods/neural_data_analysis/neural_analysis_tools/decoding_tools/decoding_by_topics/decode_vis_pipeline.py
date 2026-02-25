import os
from pathlib import Path
from typing import List

# Third-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Neuroscience specific imports
from neural_data_analysis.design_kits.design_around_event import event_binning
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
    decode_stops_design
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseOneFFStyleDecodingRunner,
)


class FFVisDecodingRunner(BaseOneFFStyleDecodingRunner):
    """
    FF visibility decoding runner. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        t_max=0.20,
    ):
        super().__init__(bin_width=bin_width)
        self.raw_data_folder_path = raw_data_folder_path
        self.t_max = t_max

        # will be filled during setup
        self.datasets = None
        self.comparisons = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width
        )

    # ------------------------------------------------------------------
    # BaseOneFFStyleDecodingRunner override points
    # ------------------------------------------------------------------
    def _get_target_df(self) -> pd.DataFrame:
        return self.vis_feats_to_decode

    def _get_groups(self):
        return self.meta_used["event_id"].values

    def _get_neural_matrix(self, use_spike_history=None) -> np.ndarray:
        return np.asarray(self.spike_data_w_history, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "vis_feats_to_decode"

    def _collect_data(self, exists_ok=True):
        """
        Collect and prepare data for decoding.

        Args:
            exists_ok: If True, load cached design matrices if they exist.
        """
        # Try to load cached design matrices
        if exists_ok and self._load_design_matrices():
            print('[_collect_data] Using cached design matrices')
        else:
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(self.raw_data_folder_path, bin_width=0.04)
            self.pn.make_or_retrieve_ff_dataframe()

            print('[_collect_data] Computing design matrices from scratch')
            (
                self.spike_data_w_history,
                self.vis_feats_to_decode,
                self.meta_used,
            ) = self._prepare_design_matrices()

            # Save the computed design matrices for future use
            self._save_design_matrices()

    def _prepare_design_matrices(self):
        new_seg_info, events_with_stats = decode_vis_utils.prepare_new_seg_info(
            self.pn.ff_dataframe,
            self.pn.bin_width,
        )

        (
            binned_spikes,
            binned_feats,
            offset_log,
            meta_used,
            meta_groups,
        ) = decode_stops_design.build_stop_design_decoding(
            new_seg_info,
            events_with_stats,
            self.pn.monkey_information,
            self.pn.spikes_df,
            self.pn.ff_dataframe,
            datasets=self.datasets,
            bin_dt=self.pn.bin_width,
            add_ff_visible_info=True,
            add_retries_info=False,
        )

        if 'global_burst_id' not in meta_used.columns:
            meta_used = meta_used.merge(
                new_seg_info[['event_id', 'global_burst_id']],
                on='event_id',
                how='left',
            )

        vis_feats_to_decode, scaled_cols = event_binning.selective_zscore(
            binned_feats
        )
        vis_feats_to_decode = sm.add_constant(
            vis_feats_to_decode,
            has_constant='add',
        )

        bin_df = spike_history.make_bin_df_from_stop_meta(meta_used)

        (
            spike_data_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=binned_spikes,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        return spike_data_w_history, vis_feats_to_decode, meta_used

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'decoding_outputs/vis_decoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            'spike_data_w_history': save_dir / 'spike_data_w_history.pkl',
            'vis_feats_to_decode': save_dir / 'vis_feats_to_decode.pkl',
            'meta_used': save_dir / 'meta_used.pkl',
        }

    def _get_design_matrix_data(self):
        return {
            'spike_data_w_history': self.spike_data_w_history,
            'vis_feats_to_decode': self.vis_feats_to_decode,
            'meta_used': self.meta_used,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            'spike_data_w_history': 'spike_data_w_history',
            'vis_feats_to_decode': 'vis_feats_to_decode',
            'meta_used': 'meta_used',
        }

