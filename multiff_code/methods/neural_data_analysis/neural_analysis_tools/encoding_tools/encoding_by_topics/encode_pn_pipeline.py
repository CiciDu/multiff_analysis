import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd

# PN-specific imports
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_aligned_by_event
)
from neural_data_analysis.design_kits.design_by_segment import (
    spike_history,
    temporal_feats,
    create_pn_design_df,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encode_stops_gam_helper,
    encode_stops_utils,
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics.base_encoding_runner import (
    BaseEncodingRunner,
)


class PNEncodingRunner(BaseEncodingRunner):
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
        self.design_df = None
        self.trial_ids = None
        self.spike_data_w_history = None
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
                    var_categories=encode_stops_gam_helper.PN_VAR_CATEGORIES,
                    gam_results_subdir="pn_gam_results",
                )
            )
        return self._gam_analysis_helper

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            print('[PNEncodingRunner] Using cached design matrices')
            return

        print('[PNEncodingRunner] Computing design matrices from scratch')

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
        dt = pn.bin_width

        data = temporal_feats.add_stop_and_capture_columns(
            data,
            trial_ids,
            pn.ff_caught_T_new,
        )

        design_df, meta0, meta = (
            create_pn_design_df.get_initial_design_df(
                data,
                dt,
                trial_ids,
            )
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

        bin_df = create_pn_design_df.make_bin_df_for_pn(
            pn.rebinned_x_var,
            pn.bin_edges,
        )

        (
            spike_data_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=pn.spikes_df,
            bin_df=bin_df,
            X_pruned=df_Y,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        self.pn = pn
        self.design_df = design_df
        self.trial_ids = trial_ids
        self.spike_data_w_history = spike_data_w_history
        self.binned_spikes = df_Y.reset_index(drop=True)

        # Build encoding design (behavioral + spike history) for GAM modeling
        design_reset = design_df.reset_index(drop=True)
        if len(design_reset) == len(bin_df):
            (
                _,
                _,
                self._colnames_enc,
                _,
                self._X_hist_enc,
            ) = spike_history.build_design_with_spike_history_from_bins(
                spikes_df=pn.spikes_df,
                bin_df=bin_df,
                X_pruned=design_reset,
                meta_groups={},
                dt=self.bin_width,
                t_max=self.t_max,
                return_X_hist=True,
            )
            self._bin_df = bin_df
            self._spike_cols_enc = sorted(self._colnames_enc.keys())
        else:
            self._colnames_enc = None
            self._X_hist_enc = None
            self._bin_df = None
            self._spike_cols_enc = None

        self._save_design_matrices()

    def _ensure_encoding_spike_history(self):
        """Build encoding spike history if not yet available (e.g. after cache load)."""
        if self._X_hist_enc is not None:
            return
        if self.design_df is None or self.binned_spikes is None:
            raise RuntimeError("Run _collect_data first.")
        self._collect_data(exists_ok=False)

    def get_design_for_unit(self, unit_idx: int) -> pd.DataFrame:
        """Build design matrix with spike-history regressors for the given target neuron."""
        self._ensure_encoding_spike_history()
        if self._spike_cols_enc is None:
            raise RuntimeError("Encoding spike history not available; design/bin row count may mismatch.")
        if unit_idx < 0 or unit_idx >= len(self._spike_cols_enc):
            raise IndexError(f"unit_idx {unit_idx} out of range [0, {len(self._spike_cols_enc)})")
        target_col = self._spike_cols_enc[unit_idx]
        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=self.design_df.reset_index(drop=True),
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

    def get_binned_spikes(self) -> pd.DataFrame:
        return self.binned_spikes

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
        base = Path(self._get_save_dir()) / "pn_gam_results"
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
        return "pn_gam_results"

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
        Run leave-one-category-out CV analysis for PN encoding.
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
        Run grid search over group penalties for PN encoding.
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
        Run backward elimination over PN GAM groups.
        """
        return self._get_gam_analysis_helper().run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
        )

    # ------------------------------------------------------------------
    # Caching utilities
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'encoding_outputs/pn_encoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            "spike_data_w_history": save_dir / "spike_data_w_history.pkl",
            "design_df": save_dir / "design_df.pkl",
            "trial_ids": save_dir / "trial_ids.pkl",
            "binned_spikes": save_dir / "binned_spikes.pkl",
            "bin_df": save_dir / "bin_df.pkl",
            "colnames_enc": save_dir / "colnames_enc.pkl",
            "X_hist_enc": save_dir / "X_hist_enc.pkl",
        }

    def _save_design_matrices(self):
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_design_matrix_paths()
        data_to_save = {
            "spike_data_w_history": self.spike_data_w_history,
            "design_df": self.design_df,
            "trial_ids": self.trial_ids,
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
                print(f'[PNEncodingRunner] Saved {key} → {paths[key]}')
            except Exception as e:
                print(
                    f'[PNEncodingRunner] WARNING: could not save {key}: '
                    f'{type(e).__name__}: {e}'
                )

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()
        required = ["spike_data_w_history", "design_df", "trial_ids", "binned_spikes"]
        if not all(paths[k].exists() for k in required):
            return False
        try:
            with open(paths["spike_data_w_history"], "rb") as f:
                self.spike_data_w_history = pickle.load(f)
            with open(paths["design_df"], "rb") as f:
                self.design_df = pickle.load(f)
            with open(paths["trial_ids"], "rb") as f:
                self.trial_ids = pickle.load(f)
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
            print("[PNEncodingRunner] Loaded cached design matrices")
            return True
        except Exception as e:
            print(
                f"[PNEncodingRunner] WARNING: could not load design matrices: "
                f"{type(e).__name__}: {e}"
            )
            return False
