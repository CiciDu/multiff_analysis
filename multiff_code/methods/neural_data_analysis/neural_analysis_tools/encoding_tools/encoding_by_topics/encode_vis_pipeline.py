import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
import statsmodels.api as sm

# Neuroscience specific imports
from neural_data_analysis.design_kits.design_around_event import event_binning
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encode_stops_gam_helper,
    encode_stops_utils,
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
    decode_stops_design
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics.base_encoding_runner import (
    BaseEncodingRunner,
)


class FFVisEncodingRunner(BaseEncodingRunner):
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
        self.binned_spikes = None
        self._bin_df = None
        self._X_hist_enc = None
        self._colnames_enc = None
        self._spike_cols_enc = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

        self._gam_analysis_helper = None

    def _get_gam_analysis_helper(self):
        if self._gam_analysis_helper is None:
            self._gam_analysis_helper = (
                encode_stops_gam_helper.SimpleEncodingGAMAnalysisHelper(
                    self,
                    var_categories=encode_stops_gam_helper.VIS_VAR_CATEGORIES,
                    gam_results_subdir="vis_gam_results",
                )
            )
        return self._gam_analysis_helper

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
                self.binned_feats_sc,
                self.meta_used,
                self.binned_spikes,
            ) = self._prepare_design_matrices()

            # Build encoding design (behavioral + spike history) for GAM modeling
            bin_df = spike_history.make_bin_df_from_stop_meta(self.meta_used)
            feats_reset = self.binned_feats_sc.reset_index(drop=True)
            if len(feats_reset) == len(bin_df):
                (
                    _,
                    _,
                    self._colnames_enc,
                    _,
                    self._X_hist_enc,
                ) = spike_history.build_design_with_spike_history_from_bins(
                    spikes_df=self.pn.spikes_df,
                    bin_df=bin_df,
                    X_pruned=feats_reset,
                    meta_groups={},
                    dt=self.bin_width,
                    t_max=self.t_max,
                    return_X_hist=True,
                )
                self._bin_df = bin_df
                self._spike_cols_enc = sorted(self._colnames_enc.keys())
            else:
                self._bin_df = None
                self._colnames_enc = None
                self._X_hist_enc = None
                self._spike_cols_enc = None

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
        ) = decode_stops_design.build_stop_design(
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

        binned_feats_sc, scaled_cols = event_binning.selective_zscore(
            binned_feats
        )
        binned_feats_sc = sm.add_constant(
            binned_feats_sc,
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

        return spike_data_w_history, binned_feats_sc, meta_used, binned_spikes

    def _ensure_encoding_spike_history(self):
        """Build encoding spike history if not yet available (e.g. after cache load)."""
        if self._X_hist_enc is not None:
            return
        if self.binned_feats_sc is None or self.binned_spikes is None:
            raise RuntimeError("Run _collect_data first.")
        self._collect_data(exists_ok=False)

    def get_design_for_unit(self, unit_idx: int) -> pd.DataFrame:
        """Build design matrix with spike-history regressors for the given target neuron."""
        self._ensure_encoding_spike_history()
        if self._spike_cols_enc is None:
            raise RuntimeError(
                "Encoding spike history not available; design/bin row count may mismatch."
            )
        if unit_idx < 0 or unit_idx >= len(self._spike_cols_enc):
            raise IndexError(f"unit_idx {unit_idx} out of range [0, {len(self._spike_cols_enc)})")
        target_col = self._spike_cols_enc[unit_idx]
        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=self.binned_feats_sc.reset_index(drop=True),
            colnames=self._colnames_enc,
            X_hist=self._X_hist_enc,
            target_col=target_col,
            include_self=True,
            cross_neurons=None,
            meta_groups=None,
        )
        const_cols_to_drop = [
            c for c in design_df.columns
            if c != "const" and design_df[c].nunique() <= 1
        ]
        if const_cols_to_drop:
            design_df = design_df.drop(columns=const_cols_to_drop)
        return design_df


    def get_gam_groups(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
    ):
        self._ensure_encoding_spike_history()
        design_df = self.get_design_for_unit(unit_idx)
        return encode_stops_utils.build_simple_gam_groups(
            design_df,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
        )

    def get_gam_save_paths(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
        ensure_dirs: bool = True,
    ) -> dict:
        base = Path(self._get_save_dir()) / "vis_gam_results"
        if ensure_dirs:
            base.mkdir(parents=True, exist_ok=True)
        lambda_config = dict(lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p)
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(lambda_config=lambda_config)
        outdir = base / f"neuron_{unit_idx}"
        if ensure_dirs:
            (outdir / "fit_results").mkdir(parents=True, exist_ok=True)
            (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
        return {
            "base": base,
            "outdir": outdir,
            "lambda_config": lambda_config,
            "lam_suffix": lam_suffix,
            "fit_save_path": str(outdir / "fit_results" / f"{lam_suffix}.pkl"),
            "cv_save_path": str(outdir / "cv_var_explained" / f"{lam_suffix}.pkl"),
        }

    @property
    def num_neurons(self) -> int:
        if self.binned_spikes is None:
            return 0
        return self.binned_spikes.shape[1]

    def _gam_results_subdir(self) -> str:
        return "vis_gam_results"

    # ------------------------------------------------------------------
    # GAM analysis (category contributions, penalty tuning, backward elimination)
    # ------------------------------------------------------------------

    def run_category_variance_contributions(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        buffer_samples: int = 20,
        category_names: Optional[List[str]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ) -> Dict:
        """
        Run leave-one-category-out CV analysis for Vis encoding.
        """
        return self._get_gam_analysis_helper().run_category_variance_contributions(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            n_folds=n_folds,
            buffer_samples=buffer_samples,
            category_names=category_names,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
        )

    def run_penalty_tuning(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        lam_grid: Optional[Dict[str, List[float]]] = None,
        group_name_map: Optional[Dict[str, List[str]]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ) -> Dict:
        """
        Run grid search over group penalties for Vis encoding.
        """
        return self._get_gam_analysis_helper().run_penalty_tuning(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            n_folds=n_folds,
            lam_grid=lam_grid,
            group_name_map=group_name_map,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
        )

    def run_backward_elimination(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
        n_folds: int = 10,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
    ) -> Dict:
        """
        Run backward elimination over Vis GAM groups.
        """
        return self._get_gam_analysis_helper().run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
        )

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'encoding_outputs/vis_encoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            "spike_data_w_history": save_dir / "spike_data_w_history.pkl",
            "binned_feats_sc": save_dir / "binned_feats_sc.pkl",
            "meta_used": save_dir / "meta_used.pkl",
            "binned_spikes": save_dir / "binned_spikes.pkl",
            "bin_df": save_dir / "bin_df.pkl",
            "colnames_enc": save_dir / "colnames_enc.pkl",
            "X_hist_enc": save_dir / "X_hist_enc.pkl",
        }

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths["spike_data_w_history"].parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            "spike_data_w_history": self.spike_data_w_history,
            "binned_feats_sc": self.binned_feats_sc,
            "meta_used": self.meta_used,
            "binned_spikes": self.binned_spikes,
            "bin_df": self._bin_df,
            "colnames_enc": self._colnames_enc,
            "X_hist_enc": self._X_hist_enc,
        }
        for key, data in data_to_save.items():
            if key not in paths or data is None:
                continue
            try:
                with open(paths[key], 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[_save_design_matrices] Saved {key} to {paths[key]}')
            except Exception as e:
                print(
                    f'[_save_design_matrices] WARNING: could not save {key}: {type(e).__name__}: {e}')

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()
        required = ["spike_data_w_history", "binned_feats_sc", "meta_used", "binned_spikes"]
        if not all(paths[k].exists() for k in required):
            return False
        try:
            with open(paths["spike_data_w_history"], "rb") as f:
                self.spike_data_w_history = pickle.load(f)
            with open(paths["binned_feats_sc"], "rb") as f:
                self.binned_feats_sc = pickle.load(f)
            with open(paths["meta_used"], "rb") as f:
                self.meta_used = pickle.load(f)
            with open(paths["binned_spikes"], "rb") as f:
                self.binned_spikes = pickle.load(f)
            if paths["bin_df"].exists():
                with open(paths["bin_df"], "rb") as f:
                    self._bin_df = pickle.load(f)
                with open(paths["colnames_enc"], "rb") as f:
                    self._colnames_enc = pickle.load(f)
                with open(paths["X_hist_enc"], "rb") as f:
                    self._X_hist_enc = pickle.load(f)
                self._spike_cols_enc = sorted(self._colnames_enc.keys())
            else:
                self._bin_df = None
                self._colnames_enc = None
                self._X_hist_enc = None
                self._spike_cols_enc = None
            print("[FFVisEncodingRunner] Loaded cached design matrices")
            return True
        except Exception as e:
            print(
                f"[FFVisEncodingRunner] WARNING: could not load design matrices: "
                f"{type(e).__name__}: {e}"
            )
            return False
