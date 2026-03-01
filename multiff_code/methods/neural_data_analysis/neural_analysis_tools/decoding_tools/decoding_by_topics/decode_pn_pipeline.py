import os
from pathlib import Path
from typing import List, Optional, Sequence

# Third-party imports
import numpy as np
import pandas as pd

# Decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics.base_decoding_runner import (
    BaseOneFFStyleDecodingRunner,
)

# PN-specific imports
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_aligned_by_event,
)
from neural_data_analysis.design_kits.design_by_segment import (
    spike_history,
    temporal_feats,
    create_pn_design_df,
)


class PNDecodingRunner(BaseOneFFStyleDecodingRunner):
    """
    PN decoding runner. CV model-spec decoding via run(); one-FF style
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

        # Filled during setup
        self.behav_df = None
        self.trial_ids = None
        self.pn_binned_spikes = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width
        )

    # ------------------------------------------------------------------
    # BaseOneFFStyleDecodingRunner override points
    # ------------------------------------------------------------------
    def _get_target_df(self) -> pd.DataFrame:
        return self.behav_df

    def _get_groups(self):
        return self.trial_ids

    def _get_neural_matrix(self) -> np.ndarray:
        return np.asarray(self.pn_binned_spikes, dtype=float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        return list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _target_df_error_msg(self) -> str:
        return "pn_feats_to_decode"

    # ------------------------------------------------------------------
    # Plotting helpers (one-FF-style outputs)
    # ------------------------------------------------------------------
    def plot_canoncorr_coefficients(self, **plot_kwargs):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("canoncorr")
        if block is None:
            raise ValueError("No canoncorr results found. Run compute_canoncorr() first.")
        plot_decode_stops.plot_canoncorr_coefficients(block, **plot_kwargs)

    def plot_decoder_parity(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decode_stops.plot_decoder_parity(block, varnames=varnames, **plot_kwargs)

    def plot_decoder_correlation_bars(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decode_stops.plot_decoder_correlation_bars(block, varnames=varnames, **plot_kwargs)

    def plot_single_trial_decoding_panel(
        self,
        *,
        trial_indices: Optional[Sequence[int]] = None,
        n_trials: int = 6,
        **plot_kwargs,
    ):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decode_stops.plot_single_trial_decoding_panel(
            block,
            trial_indices=trial_indices,
            n_trials=n_trials,
            **plot_kwargs,
        )

    def plot_all_decoding_results(
        self,
        *,
        parity_varnames: Optional[Sequence[str]] = None,
        bar_varnames: Optional[Sequence[str]] = None,
        trial_indices: Optional[Sequence[int]] = None,
        n_trials: int = 6,
    ):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        plot_decode_stops.plot_all_decoding_results(
            canoncorr_block=self.stats.get("canoncorr"),
            readout_block=self.stats.get("lineardecoder"),
            parity_varnames=parity_varnames,
            bar_varnames=bar_varnames,
            trial_indices=trial_indices,
            n_trials=n_trials,
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self, exists_ok=True):
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
        trial_ids = data['new_segment']

        pn_feats_to_decode = temporal_feats.add_stop_and_capture_columns(
            data,
            trial_ids,
            pn.ff_caught_T_new,
        )

        cluster_cols = [
            c for c in pn.rebinned_x_var.columns
            if c.startswith('cluster_')
        ]

        df_Y = pn.rebinned_x_var[cluster_cols]
        df_Y.columns = (
            df_Y.columns
            .str.replace('cluster_', '')
            .astype(int)
        )

        # bin_df = create_pn_design_df.make_bin_df_for_pn(
        #     pn.rebinned_x_var,
        #     pn.bin_edges,
        # )


        self.pn = pn
        self.behav_df = pn_feats_to_decode
        self.trial_ids = trial_ids
        self.pn_binned_spikes = df_Y
        self._save_design_matrices()

    # ------------------------------------------------------------------
    # Caching utilities (BaseOneFFStyleDecodingRunner interface)
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'decoding_outputs/pn_decoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            'pn_binned_spikes': save_dir / 'pn_binned_spikes.pkl',
            'pn_feats_to_decode': save_dir / 'pn_feats_to_decode.pkl',
            'trial_ids': save_dir / 'trial_ids.pkl',
        }

    def _get_design_matrix_data(self):
        return {
            'pn_binned_spikes': self.pn_binned_spikes,
            'pn_feats_to_decode': self.behav_df,
            'trial_ids': self.trial_ids,
        }

    def _get_design_matrix_key_to_attr(self):
        return {
            'pn_binned_spikes': 'pn_binned_spikes',
            'pn_feats_to_decode': 'behav_df',
            'trial_ids': 'trial_ids',
        }
