from pathlib import Path
from typing import List

# Third-party imports
import numpy as np
import pandas as pd

# Decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseDecodingRunner,
)

# PN-specific imports
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_aligned_by_event,
)
from neural_data_analysis.design_kits.design_by_segment import (
    temporal_feats,
)

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_design_utils, decode_pn_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers.decoding_design_utils import (
    PN_DECODING_VAR_CATEGORIES,
)


class PNDecodingRunner(BaseDecodingRunner):
    """
    PN decoding runner. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        var_categories=None,
        **kwargs,
    ):
        super().__init__(bin_width=bin_width, **kwargs)
        self.raw_data_folder_path = raw_data_folder_path
        self.var_categories = var_categories if var_categories is not None else PN_DECODING_VAR_CATEGORIES

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
                        
        self.var_categories = decoding_design_utils.add_other_category_from_df(self.var_categories, self.target_df)

        return self.target_df

    def _get_groups(self):
        return self.trial_ids

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "feats_to_decode"

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_data(self, exists_ok=True):

        if exists_ok and self._load_design_matrices():
            print('[PNDecodingRunner] Using cached design matrices')
            return

        print('[PNDecodingRunner] Computing design matrices from scratch')

        pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        pn.prep_data_to_analyze_planning(
            planning_data_by_point_exists_ok=True
        )

        pn.rebin_data_in_new_segments(
            cur_or_nxt='cur',
            first_or_last='first',
            time_limit_to_count_sighting=2,
            start_t_rel_event=0,
            end_t_rel_event=1.5,
            rebinned_max_x_lag_number=2,
        )

        data = pn.rebinned_y_var.copy()
        self.meta_df_used = pn.rebinned_y_var[['new_segment', 'bin_in_new_seg']].rename(columns={'new_segment': 'event_id', 'bin_in_new_seg': 'k_within_seg'})
        trial_ids = data['new_segment']

        if 't_center' in data.columns:
            data['time'] = data['t_center']

        self.feats_to_decode = temporal_feats.add_stop_and_capture_columns(
            data,
            trial_ids,
            pn.ff_caught_T_new,
        )

        # Ensure processed spikes are computed from this freshly prepared pn object.
        self.pn = pn
        self.get_processed_spike_rates()
        self.get_binned_spikes()
        
        self.feats_to_decode = decoding_design_utils.clean_binary_and_drop_constant(self.feats_to_decode)

        # detrend covariates for optional multi-covariate detrending
        detrend_dict = {}
        if 't_center' in pn.rebinned_y_var.columns:
            detrend_dict['time'] = np.asarray(pn.rebinned_y_var['t_center'], dtype=float)
        if 'new_segment' in pn.rebinned_y_var.columns:
            detrend_dict['trial_index'] = np.asarray(pn.rebinned_y_var['new_segment'], dtype=float)
        self.detrend_covariates = pd.DataFrame(detrend_dict) if detrend_dict else None

        self.trial_ids = trial_ids
        
        self.reduce_binned_spikes()
        self._save_design_matrices()
        self.clean_var_categories()

    # ------------------------------------------------------------------
    # Caching utilities (BaseDecodingRunner interface)
    # ------------------------------------------------------------------
    
        
    def get_binned_spikes(self):
        self.get_processed_spike_rates()

        if self.processed_spike_rates is not None:
            self.binned_spikes = decode_pn_utils.rebin_processed_spike_rates(self.processed_spike_rates, self.pn.new_seg_info, self.pn.local_bin_edges, self.pn.rebinned_y_var)
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
        return {
            'binned_spikes': save_dir / 'binned_spikes.pkl',
            'feats_to_decode': save_dir / 'feats_to_decode.pkl',
            'meta_df_used': save_dir / 'meta_df_used.pkl',
            'trial_ids': save_dir / 'trial_ids.pkl',
            'detrend_covariates': save_dir / 'detrend_covariates.pkl',
        }

    def _get_design_matrix_data(self):
        return {
            'binned_spikes': self.binned_spikes,
            'feats_to_decode': self.feats_to_decode,
            'meta_df_used': self.meta_df_used,
            'trial_ids': self.trial_ids,
            'detrend_covariates': self.detrend_covariates,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            'binned_spikes': 'binned_spikes',
            'feats_to_decode': 'feats_to_decode',
            'meta_df_used': 'meta_df_used',
            'trial_ids': 'trial_ids',
            'detrend_covariates': 'detrend_covariates',
        }
