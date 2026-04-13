"""Base class for encoding tasks (data + feature construction ONLY)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoding_design_utils,
    multiff_encoding_params,
)
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
)


class BaseEncodingTask:
    """
    Owns all data-loading, alignment, and feature construction for one session.

    Subclasses must implement:
        collect_data(exists_ok)
        _get_save_dir()             → str   (session-level output root)

    Provides (after collect_data):
        binned_feats        DataFrame  (n_bins × n_features)
        binned_spikes       DataFrame  (n_bins × n_neurons)
        temporal_meta       dict
        tuning_meta         dict
        structured_meta_groups  dict
        bin_df              DataFrame
        X_hist              ndarray    (spike-history design)
        spk_colnames        dict
        meta_df_used        DataFrame  (optional)
        trial_ids           Series     (optional)
        binrange_dict       dict
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width: float = 0.04,
        encoder_prs=None,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width

        self.num_neurons = neural_data_processing.find_num_neurons(
            self.raw_data_folder_path
        )

        self.encoder_prs = (
            encoder_prs
            if encoder_prs is not None
            else multiff_encoding_params.default_prs()
        )
        self.binrange_dict = self.encoder_prs.binrange

        # Filled by collect_data / _load_design_matrices
        self.binned_feats: Optional[pd.DataFrame] = None
        self.binned_spikes: Optional[pd.DataFrame] = None
        self.temporal_meta: Optional[dict] = None
        self.tuning_meta: Optional[dict] = None
        self.structured_meta_groups: Optional[dict] = None
        self.hist_meta: Optional[dict] = None
        self.bin_df: Optional[pd.DataFrame] = None
        self.X_hist = None
        self.spk_colnames: Optional[dict] = None
        self.spike_cols: Optional[list] = None
        self.meta_df_used: Optional[pd.DataFrame] = None
        self.trial_ids = None

        # Lazily initialised pn object (subclasses may replace this)
        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def collect_data(self, exists_ok: bool = True) -> None:
        """Load or compute binned_feats, binned_spikes, and all metadata."""
        raise NotImplementedError

    def _get_save_dir(self) -> str:
        """Session-level output root (used for caching design matrices)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Feature access helpers
    # ------------------------------------------------------------------

    def get_feats(self, mode: str = "raw") -> pd.DataFrame:
        """
        Return feature matrix.

        Parameters
        ----------
        mode : {'raw', 'spline'}
            'raw'    → self.binned_feats as-is
            'spline' → appends spike-history columns (calls
                       _prepare_spike_history_components first)
        """
        if self.binned_feats is None:
            raise RuntimeError("Call collect_data() first.")
        if mode == "raw":
            return self.binned_feats
        if mode == "spline":
            self._prepare_spike_history_components()
            return self.binned_feats  # X_hist lives separately; callers may combine
        raise ValueError(f"Unknown mode {mode!r}. Use 'raw' or 'spline'.")

    def get_binned_data(self):
        """Return (binned_feats, binned_spikes) tuple."""
        return self.binned_feats, self.binned_spikes

    # ------------------------------------------------------------------
    # CV grouping (used by models that need GroupKFold)
    # ------------------------------------------------------------------

    def get_cv_groups_for_design(self, design_df: pd.DataFrame):
        """
        Per-sample group labels aligned with design_df rows.

        Priority: new_segment column in design → trial_ids attr →
                  meta_df_used event_id / new_segment.
        Returns None if no grouping is available.
        """
        n = len(design_df)
        if n == 0:
            return None
        if "new_segment" in design_df.columns:
            return np.asarray(design_df["new_segment"].to_numpy())
        trial_ids = getattr(self, "trial_ids", None)
        if trial_ids is not None:
            arr = (
                trial_ids.reset_index(drop=True).to_numpy()
                if hasattr(trial_ids, "reset_index")
                else np.asarray(trial_ids)
            )
            if len(arr) == n:
                return arr
        meta = getattr(self, "meta_df_used", None)
        if meta is not None and len(meta) == n:
            for col in ("event_id", "new_segment"):
                if col in meta.columns:
                    return np.asarray(meta[col].to_numpy())
        return None

    # ------------------------------------------------------------------
    # Design-matrix construction helpers
    # ------------------------------------------------------------------

    def _encoding_design_kwargs(self) -> Dict:
        mode = getattr(self.encoder_prs, "tuning_feature_mode", None)
        return {
            "use_boxcar": bool(mode in ("boxcar_only", "raw_plus_boxcar")),
            "tuning_feature_mode": mode,
            "binrange_dict": self.binrange_dict,
            "n_basis": getattr(self.encoder_prs, "default_n_basis", 20),
            "t_min": -getattr(self.encoder_prs, "pre_event", 0.3),
            "t_max": getattr(self.encoder_prs, "post_event", 0.3),
            "tuning_n_bins": getattr(self.encoder_prs, "tuning_n_bins", 10),
        }

    def _prepare_spike_history_components(self):
        """Compute X_hist and spk_colnames (idempotent)."""
        if hasattr(self, "X_hist") and self.X_hist is not None:
            return

        if not hasattr(self.pn, "spikes_df"):
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(
                self.raw_data_folder_path, bin_width=0.04
            )

        n_basis_hist = getattr(self.encoder_prs, "default_n_basis", 20)
        spike_hist_t_max = self.binrange_dict["spike_hist"][1]

        _, self._basis, self.spk_colnames, _, self.X_hist = (
            spike_history.build_design_with_spike_history_from_bins(
                spikes_df=self.pn.spikes_df,
                bin_df=self.bin_df,
                X_pruned=self.binned_feats,
                meta_groups={},
                dt=self.bin_width,
                t_max=spike_hist_t_max,
                returnX_hist=True,
                n_basis=n_basis_hist,
            )
        )

    def make_hist_meta_for_unit(self, unit_idx: int):
        target_col = list(self.spk_colnames.keys())[unit_idx]
        spike_hist_t_max = self.binrange_dict["spike_hist"][1]
        n_basis_hist = getattr(self.encoder_prs, "default_n_basis", 20)

        self.hist_meta = encoding_design_utils.build_hist_meta_from_colnames(
            colnames=self.spk_colnames,
            target_col=target_col,
            dt=self.bin_width,
            t_max=spike_hist_t_max,
            n_basis=n_basis_hist,
        )

    def _make_structured_meta_groups(self):
        self.structured_meta_groups = encoding_design_utils.build_structured_meta_groups(
            hist_meta=self.hist_meta,
            temporal_meta=self.temporal_meta,
            tuning_meta=self.tuning_meta,
        )

    def reduce_binned_feats(self):
        pass  # hook for subclasses

    # ------------------------------------------------------------------
    # var_categories cleanup (called at end of collect_data)
    # ------------------------------------------------------------------

    def clean_var_categories(self):
        from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
            encoder_gam_helper,
        )

        self.var_categories = encoding_design_utils.build_clean_var_categories_from_meta(
            self.var_categories, self.structured_meta_groups
        )
        self.var_categories = encoder_gam_helper.add_category_unassigned_vars_for_encoding(
            self.var_categories, self.structured_meta_groups
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_design_matrix_paths(self) -> Dict[str, Path]:
        save_dir = Path(self._get_save_dir()) / "encoding_design"
        keys = [
            "binned_feats", "binned_spikes", "binrange_dict", "bin_df",
            "temporal_meta", "tuning_meta", "X_hist", "spk_colnames",
            "meta_df_used", "trial_ids",
        ]
        return {k: save_dir / f"{k}.pkl" for k in keys}

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths["binned_feats"].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "binned_feats": self.binned_feats,
            "binned_spikes": self.binned_spikes,
            "binrange_dict": self.binrange_dict,
            "temporal_meta": self.temporal_meta,
            "tuning_meta": self.tuning_meta,
            "bin_df": getattr(self, "bin_df", None),
            "X_hist": getattr(self, "X_hist", None),
            "spk_colnames": getattr(self, "spk_colnames", None),
            "meta_df_used": getattr(self, "meta_df_used", None),
            "trial_ids": getattr(self, "trial_ids", None),
        }

        for key, data in data_to_save.items():
            if data is None:
                continue
            try:
                with open(paths[key], "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{type(self).__name__}] Saved {key} → {paths[key]}")
            except Exception as e:
                print(
                    f"[{type(self).__name__}] WARNING: could not save {key}: "
                    f"{type(e).__name__}: {e}"
                )

    def _load_design_matrices(self) -> bool:
        paths = self._get_design_matrix_paths()
        required = ["binned_feats", "binned_spikes"]
        if not all(paths[k].exists() for k in required):
            return False

        try:
            with open(paths["binned_feats"], "rb") as f:
                self.binned_feats = pickle.load(f)
            with open(paths["binned_spikes"], "rb") as f:
                self.binned_spikes = pickle.load(f)
            with open(paths["binrange_dict"], "rb") as f:
                self.binrange_dict = pickle.load(f)
            with open(paths["temporal_meta"], "rb") as f:
                self.temporal_meta = pickle.load(f)
            with open(paths["tuning_meta"], "rb") as f:
                self.tuning_meta = pickle.load(f)
            with open(paths["bin_df"], "rb") as f:
                self.bin_df = pickle.load(f)
            with open(paths["X_hist"], "rb") as f:
                self.X_hist = pickle.load(f)
            with open(paths["spk_colnames"], "rb") as f:
                self.spk_colnames = pickle.load(f)
                self.spike_cols = list(self.spk_colnames.keys())
            if paths["meta_df_used"].exists():
                with open(paths["meta_df_used"], "rb") as f:
                    self.meta_df_used = pickle.load(f)
            if paths["trial_ids"].exists():
                with open(paths["trial_ids"], "rb") as f:
                    self.trial_ids = pickle.load(f)

            print(f"[{type(self).__name__}] Loaded cached design matrices from: {paths['binned_feats']}")

            self.structured_meta_groups = encoding_design_utils.build_structured_meta_groups(
                hist_meta=self.hist_meta,
                temporal_meta=self.temporal_meta,
                tuning_meta=self.tuning_meta,
            )
            return True

        except Exception as e:
            print(f"[{type(self).__name__}] WARNING: could not load design matrices: {e}")
            return False
