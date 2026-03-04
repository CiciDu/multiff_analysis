import os
from pathlib import Path
from typing import List, Optional, Sequence

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
    BaseDecodingRunner,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding


class FFVisDecodingRunner(BaseDecodingRunner):
    """
    FF visibility decoding runner. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        
    ):
        super().__init__(bin_width=bin_width)
        self.raw_data_folder_path = raw_data_folder_path
        

        # will be filled during setup
        self.datasets = None
        self.comparisons = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width
        )

    # ------------------------------------------------------------------
    # BaseDecodingRunner override points
    # ------------------------------------------------------------------
    def _get_target_df(self) -> pd.DataFrame:
        return self.vis_feats_to_decode

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.vis_binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "vis_feats_to_decode"

    # ------------------------------------------------------------------
    # Plotting helpers (one-FF-style outputs)
    # ------------------------------------------------------------------
    def plot_canoncorr_coefficients(self, **plot_kwargs):
        block = self.stats.get("canoncorr")
        if block is None:
            raise ValueError("No canoncorr results found. Run compute_canoncorr() first.")
        plot_one_ff_decoding.plot_canoncorr_coefficients(block, **plot_kwargs)

    def plot_decoder_parity(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_one_ff_decoding.plot_decoder_parity(block, varnames=varnames, **plot_kwargs)

    def plot_decoder_correlation_bars(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_one_ff_decoding.plot_decoder_correlation_bars(block, varnames=varnames, **plot_kwargs)


    def collect_data(self, exists_ok=True):
        """
        Collect and prepare data for decoding.

        Args:
            exists_ok: If True, load cached design matrices if they exist.
        """
        # Try to load cached design matrices
        if exists_ok and self._load_design_matrices():
            print('[collect_data] Using cached design matrices')
        else:
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(self.raw_data_folder_path, bin_width=0.04)
            self.pn.make_or_retrieve_ff_dataframe()

            print('[collect_data] Computing design matrices from scratch')
            (
                self.vis_binned_spikes,
                self.vis_feats_to_decode,
                self.meta_df_used,
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
            meta_df_used,
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

        if 'global_burst_id' not in meta_df_used.columns:
            meta_df_used = meta_df_used.merge(
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
  
        self.vis_binned_spikes = binned_spikes

        return binned_spikes, vis_feats_to_decode, meta_df_used

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'decoding_outputs/vis_decoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            'vis_binned_spikes': save_dir / 'vis_binned_spikes.pkl',
            'vis_feats_to_decode': save_dir / 'vis_feats_to_decode.pkl',
            'meta_df_used': save_dir / 'meta_df_used.pkl',
        }

    def _get_design_matrix_data(self):
        return {
            'vis_binned_spikes': self.vis_binned_spikes,
            'vis_feats_to_decode': self.vis_feats_to_decode,
            'meta_df_used': self.meta_df_used,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            'vis_binned_spikes': 'vis_binned_spikes',
            'vis_feats_to_decode': 'vis_feats_to_decode',
            'meta_df_used': 'meta_df_used',
        }

