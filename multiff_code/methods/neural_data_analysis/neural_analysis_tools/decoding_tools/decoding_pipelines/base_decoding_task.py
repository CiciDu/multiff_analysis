"""Base class for decoding tasks (data + feature construction ONLY)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import os 

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decoding_design_utils,
    smooth_neural_data,
)
from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event


class BaseDecodingTask:
    """
    Owns all data-loading, alignment, and feature construction for one session.

    Subclasses must implement:
        collect_data(exists_ok)
        _get_save_dir()         → str
        get_target_df()         → DataFrame   (features to decode)
        _get_groups()           → array-like  (per-bin group labels for CV)
        _get_neural_matrix()    → ndarray     (n_bins × n_neurons)
        _default_canoncorr_varnames() → List[str]
        _default_readout_varnames()   → List[str]

    Provides (after collect_data):
        binned_spikes       DataFrame  (n_bins × n_neurons)
        feats_to_decode     DataFrame  (n_bins × n_features)
        meta_df_used        DataFrame
        trial_ids           Series     (optional, PN only)
        bin_df              DataFrame  (optional, spike-history only)
        var_categories      dict
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width: float = 0.04,
        smoothing_width: Optional[int] = None,
        detrend_spikes: bool = False,
        drop_bad_neurons: bool = True,
        pca_n_components: Optional[int] = None,
        pca_fit_presmooth_width: int = 0,
        use_spike_history: bool = False,
        spike_history_mode: str = "lags",
        spike_history_t_max: float = 0.4,
        spike_history_n_basis: int = 10,
        spike_history_n_lags: int = 8,
        var_categories=None,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width

        self.smooth_spikes = smoothing_width is not None
        self.smoothing_width = smoothing_width
        self.detrend_spikes = detrend_spikes
        self.drop_bad_neurons = drop_bad_neurons

        self.pca_n_components = pca_n_components
        self.pca_fit_presmooth_width = int(pca_fit_presmooth_width)
        self.pca_ = None

        self.use_spike_history = use_spike_history
        self.spike_history_mode = spike_history_mode
        self.spike_history_t_max = spike_history_t_max
        self.spike_history_n_basis = spike_history_n_basis
        self.spike_history_n_lags = spike_history_n_lags

        # Filled by collect_data
        self.feats_to_decode: Optional[pd.DataFrame] = None
        self.binned_spikes: Optional[pd.DataFrame] = None
        self.meta_df_used: Optional[pd.DataFrame] = None
        self.trial_ids = None
        self.bin_df: Optional[pd.DataFrame] = None
        self.detrend_covariates: Optional[pd.DataFrame] = None
        self.spike_history_features = None
        self.spike_history_features_df: Optional[pd.DataFrame] = None
        self.processed_spike_rates = None

        self.var_categories = var_categories or {}

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def collect_data(self, exists_ok: bool = True) -> None:
        raise NotImplementedError

    def _get_save_dir(self) -> str:
        raise NotImplementedError

    def get_target_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def _get_groups(self):
        raise NotImplementedError

    def _get_neural_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def _default_canoncorr_varnames(self) -> List[str]:
        raise NotImplementedError

    def _default_readout_varnames(self) -> List[str]:
        raise NotImplementedError

    def _target_df_error_msg(self) -> str:
        return "feats_to_decode"

    # ------------------------------------------------------------------
    # Neural matrix helpers (smoothing, PCA) — used by DecodingModel
    # ------------------------------------------------------------------

    def _neural_ncols_after_pca(self) -> int:
        raw = np.asarray(self._get_neural_matrix(), dtype=float)
        return int(self.pca_n_components) if self.pca_n_components is not None else int(raw.shape[1])

    def _smooth_raw_neural(self, raw: np.ndarray, width: int,
                           trial_idx: Optional[np.ndarray] = None) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if int(width) <= 0:
            return raw
        if trial_idx is None:
            return smooth_neural_data.smooth_signal(raw, int(width))
        return smooth_neural_data.smooth_signal(raw, int(width), trial_idx=trial_idx)

    def _ensure_pca_fitted(self) -> None:
        if self.pca_n_components is None or self.pca_ is not None:
            return
        from sklearn.decomposition import PCA
        raw = np.asarray(self._get_neural_matrix(), dtype=float)
        groups = self._get_groups()
        trial_idx = np.asarray(groups) if groups is not None else None
        w = int(self.pca_fit_presmooth_width or 0)
        raw_fit = self._smooth_raw_neural(raw, w, trial_idx=trial_idx)
        self.pca_ = PCA(n_components=self.pca_n_components, random_state=0)
        self.pca_.fit(raw_fit)
        explained = self.pca_.explained_variance_ratio_.sum()
        print(f"[{type(self).__name__}] PCA: {self.pca_n_components} components, "
              f"cumulative variance = {explained:.3f}")

    def _neural_matrix_after_pca(self, *, smooth_width: int = 0,
                                  trial_idx: Optional[np.ndarray] = None) -> np.ndarray:
        raw = np.asarray(self._get_neural_matrix(), dtype=float)
        raw_sm = self._smooth_raw_neural(raw, int(smooth_width), trial_idx=trial_idx)
        if self.pca_n_components is None:
            return raw_sm
        self._ensure_pca_fitted()
        return np.asarray(self.pca_.transform(raw_sm), dtype=float)

    def _ensure_spike_history_features_materialized(self):
        if self.spike_history_features is not None:
            return
        from neural_data_analysis.design_kits.design_by_segment import spike_history as spike_history_kit
        if self.bin_df is None:
            raise RuntimeError("bin_df required for spike history — set use_spike_history=True before collect_data.")
        pn = getattr(self, "pn", None)
        if pn is None or getattr(pn, "spikes_df", None) is None:
            raise RuntimeError("pn.spikes_df required for spike history.")
        self.spike_history_features = spike_history_kit.compute_spike_history_designs(
            spikes_df=pn.spikes_df,
            bin_df=self.bin_df,
            mode=self.spike_history_mode,
            t_max=self.spike_history_t_max,
            n_basis=self.spike_history_n_basis,
            n_lags=self.spike_history_n_lags,
        )

    def _get_neural_matrix_for_decoding(self, *, neural_smooth_width: int,
                                         trial_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Smooth → optional PCA → optional spike history."""
        raw = np.asarray(self._get_neural_matrix(), dtype=float)
        raw_sm = self._smooth_raw_neural(raw, int(neural_smooth_width), trial_idx=trial_idx)
        if self.pca_n_components is None:
            Xn = raw_sm
        else:
            self._ensure_pca_fitted()
            Xn = np.asarray(self.pca_.transform(raw_sm), dtype=float)
        if not self.use_spike_history:
            return Xn
        self._ensure_spike_history_features_materialized()
        Xh = np.asarray(self.spike_history_features, dtype=float)
        if Xh.shape[1] == 0:
            return Xn
        return np.hstack([Xn, Xh])

    # ------------------------------------------------------------------
    # Spike rate processing
    # ------------------------------------------------------------------

    def get_processed_spike_rates(self):
        if self.processed_spike_rates is not None:
            return
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
            decoding_design_utils,
        )
        # Smooth/detrend full-session binned rates from spikes_df — not binned_spikes,
        # which is only filled after collect_data builds event-aligned matrices.
        pn = getattr(self, "pn", None)
        spikes_df = getattr(pn, "spikes_df", None) if pn is not None else None
        if spikes_df is None:
            self.processed_spike_rates = None
            return
        sw = self.smoothing_width if self.smoothing_width is not None else 10
        self.processed_spike_rates = decoding_design_utils.get_processed_spike_rates(
            spikes_df,
            smooth_spikes=self.smooth_spikes,
            detrend_spikes=self.detrend_spikes,
            smoothing_width=int(sw),
            drop_bad_neurons=self.drop_bad_neurons,
            fs_bin_width=None,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def reduce_binned_spikes(self):
        pass  # hook for subclasses

    def _get_numeric_target_df(self) -> pd.DataFrame:
        y_df = self.get_target_df().copy()
        keep = [
            c for c in y_df.columns
            if c != "const" and (
                pd.api.types.is_numeric_dtype(y_df[c])
                or pd.api.types.is_bool_dtype(y_df[c])
            )
        ]
        return y_df[keep].astype(float)

    def clean_var_categories(self):
        self.var_categories = decoding_design_utils._build_clean_var_categories(
            self.feats_to_decode, self.var_categories
        )
        self.var_categories = decoding_design_utils.add_category_unassigned_vars_for_encoding_for_decoding(
            self.var_categories, self.feats_to_decode
        )

    def _get_save_dir_common(self, subdir: str) -> str:
        if getattr(self, "smooth_spikes", False):
            sub = os.path.join("smooth", f"width_{int(self.smoothing_width)}")
        elif getattr(self, "detrend_spikes", False):
            sub = "detrended"
        else:
            sub = "raw"

        if getattr(self, "pca_n_components", None) is not None:
            sub = sub + f"_pca{self.pca_n_components}"

        parts = [
            self.pn.planning_and_neural_folder_path,
            "decoding_outputs",
            subdir,
        ]
        if bool(getattr(self, "use_spike_history", False)):
            parts.append(self._spike_history_save_subdir())
        else:
            parts.append("no_spike_history")
        parts.append(sub)
        self.save_dir = os.path.join(*parts)
        return self.save_dir

    def _spike_history_save_subdir(self) -> str:
        bw = f"{float(self.bin_width):g}".replace(".", "p").replace("-", "m")
        mode = str(getattr(self, "spike_history_mode", "lags")).replace(os.sep, "_")
        if os.altsep:
            mode = mode.replace(os.altsep, "_")
        t_max = float(getattr(self, "spike_history_t_max", 0.4))
        tm = f"{t_max:g}".replace(".", "p").replace("-", "m")
        nb = int(getattr(self, "spike_history_n_basis", 10))
        if mode == "lags":
            nl = int(getattr(self, "spike_history_n_lags", 8))
            return f"sh_{mode}_nl{nl}_nb{nb}_tm{tm}_bw{bw}"
        return f"sh_{mode}_nb{nb}_tm{tm}_bw{bw}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_design_matrix_paths(self) -> Dict[str, Path]:
        raise NotImplementedError

    def _get_design_matrix_data(self) -> Dict:
        raise NotImplementedError

    def _get_design_matrix_key_to_attr(self) -> Dict[str, str]:
        raise NotImplementedError

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        data = self._get_design_matrix_data()
        list(paths.values())[0].parent.mkdir(parents=True, exist_ok=True)
        for key, val in data.items():
            if val is None or key not in paths:
                continue
            try:
                with open(paths[key], "wb") as f:
                    pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{type(self).__name__}] Saved {key} → {paths[key]}")
            except Exception as e:
                print(f"[{type(self).__name__}] WARNING: could not save {key}: {e}")

    def _load_design_matrices(self) -> bool:
        paths = self._get_design_matrix_paths()
        key_to_attr = self._get_design_matrix_key_to_attr()
        required = ["binned_spikes", "feats_to_decode"]
        if not all(paths[k].exists() for k in required if k in paths):
            return False
        try:
            for key, attr in key_to_attr.items():
                p = paths.get(key)
                if p is None or not p.exists():
                    continue
                with open(p, "rb") as f:
                    setattr(self, attr, pickle.load(f))
            print(f"[{type(self).__name__}] Loaded cached design matrices")
            return True
        except Exception as e:
            print(f"[{type(self).__name__}] WARNING: could not load design matrices: {e}")
            return False

