import os
from pathlib import Path
from typing import List

# Third-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Neuroscience specific imports
from neural_data_analysis.design_kits.design_around_event import event_binning
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import (
    decode_vis_design,
    decode_vis_utils,
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseDecodingRunner,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_design_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers.decoding_design_utils import (
    VIS_DECODING_VAR_CATEGORIES,
)


class FFVisDecodingRunner(BaseDecodingRunner):
    """
    FF visibility decoding runner. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        var_categories=None,
    ):
        super().__init__(bin_width=bin_width)
        self.raw_data_folder_path = raw_data_folder_path
        self.var_categories = var_categories if var_categories is not None else VIS_DECODING_VAR_CATEGORIES

        # will be filled during setup
        self.datasets = None
        self.comparisons = None
        

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width
        )

    # ------------------------------------------------------------------
    # BaseDecodingRunner override points
    # ------------------------------------------------------------------
    def get_target_df(self) -> pd.DataFrame:
        if getattr(self, 'feats_to_decode', None) is None:
            self.collect_data(exists_ok=True)
        self.target_df = self.feats_to_decode.copy()
        return self.target_df

    def _get_groups(self):
        return self.meta_df_used["event_id"].values

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "feats_to_decode"


    def collect_data(self, exists_ok=True, detrend_spikes: bool = True):
        """
        Collect and prepare data for decoding.

        Args:
            exists_ok: If True, load cached design matrices if they exist.
        """
        self.detrend_spikes = detrend_spikes
        # Try to load cached design matrices
        if exists_ok and self._load_design_matrices():
            print('[collect_data] Using cached design matrices')
        else:
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(self.raw_data_folder_path, bin_width=0.04)
            self.pn.make_or_retrieve_ff_dataframe()

            print('[collect_data] Computing design matrices from scratch')
            (
                self.binned_spikes,
                self.feats_to_decode,
                self.meta_df_used,
            ) = self._prepare_design_matrices(detrend_spikes=detrend_spikes)

            self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(self.feats_to_decode)

            # Save the computed design matrices for future use
            self.reduce_binned_spikes()
            self._save_design_matrices()

    def _prepare_design_matrices(self, detrend_spikes: bool = True):
        new_seg_info, events_with_stats = decode_vis_utils.prepare_new_seg_info(
            self.pn.ff_dataframe,
            self.pn.bin_width,
        )

        (
            rebinned_spike_rates,
            binned_feats,
            offset_log,
            meta_df_used,
        ) = decode_vis_design.build_vis_design_decoding(
            new_seg_info,
            events_with_stats,
            self.pn.monkey_information,
            self.pn.spikes_df,
            self.pn.ff_dataframe,
            datasets=self.datasets,
            bin_dt=self.pn.bin_width,
            add_ff_visible_info=True,
            add_retries_info=False,
            detrend_spikes=detrend_spikes,
        )

        if 'global_burst_id' not in meta_df_used.columns:
            meta_df_used = meta_df_used.merge(
                new_seg_info[['event_id', 'global_burst_id']],
                on='event_id',
                how='left',
            )

        feats_to_decode, scaled_cols = event_binning.selective_zscore(
            binned_feats
        )
        feats_to_decode = sm.add_constant(
            feats_to_decode,
            has_constant='add',
        )

        _id_cols = {'new_segment', 'new_bin', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration'}
        _cluster_cols = [c for c in rebinned_spike_rates.columns if c not in _id_cols]
        self.binned_spikes = rebinned_spike_rates[_cluster_cols].copy()

        return rebinned_spike_rates, feats_to_decode, meta_df_used

    def _get_save_dir(self):
        sub = "detrended" if getattr(self, "detrend_spikes", True) else "raw"
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            "decoding_outputs/vis_decoder_outputs",
            sub,
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            'binned_spikes': save_dir / 'binned_spikes.pkl',
            'feats_to_decode': save_dir / 'feats_to_decode.pkl',
            'meta_df_used': save_dir / 'meta_df_used.pkl',
        }

    def _get_design_matrix_data(self):
        return {
            'binned_spikes': self.binned_spikes,
            'feats_to_decode': self.feats_to_decode,
            'meta_df_used': self.meta_df_used,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            'binned_spikes': 'binned_spikes',
            'feats_to_decode': 'feats_to_decode',
            'meta_df_used': 'meta_df_used',
        }

