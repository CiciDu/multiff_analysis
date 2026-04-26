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

        self.encoder_prs = (
            encoder_prs
            if encoder_prs is not None
            else multiff_encoding_params.default_prs()
        )
        self.binrange_dict = self.encoder_prs.binrange
        self.use_neural_coupling = False  # set by EncodingRunner at construction

        # Filled by collect_data / _load_design_matrices
        self.binned_feats: Optional[pd.DataFrame] = None
        self.binned_spikes: Optional[pd.DataFrame] = None
        self.raw_behavioral_feats: Optional[pd.DataFrame] = None
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

    def get_effective_num_neurons(self) -> int:
        """Neuron count valid for both spike matrix and spike-history metadata."""
        if self.binned_spikes is None:
            raise RuntimeError("Call collect_data() first.")
        n_spikes = int(self.binned_spikes.shape[1])
        n_spk_cols = len(self.spk_colnames) if self.spk_colnames is not None else n_spikes
        return min(n_spikes, n_spk_cols)

    def get_raw_behavioral_feats(self) -> pd.DataFrame:
        """Return raw behavioral features (no splines, no spike history).

        These are the same variables that decoding tasks expose as
        ``feats_to_decode``: kinematics and other continuous covariates
        captured *before* spline/boxcar expansion in the encoding design
        builder.  Use this as the RNN input when you want the model to
        receive unparameterised behavioural signals.
        """
        if self.raw_behavioral_feats is None:
            raise RuntimeError(
                "raw_behavioral_feats is not available.  "
                "Call collect_data() first."
            )
        return self.raw_behavioral_feats

    def _finalize_collect_data(self) -> None:
        """Shared tail called at the end of every subclass collect_data.

        Runs feature reduction, spike-history construction, meta-group
        assembly, persistence, and var-category cleanup — in that order.
        Subclasses must have already populated ``binned_feats``,
        ``binned_spikes``, and ``raw_behavioral_feats`` before calling
        this.
        """
        self.reduce_binned_feats()
        self._prepare_spike_history_components()
        self._make_structured_meta_groups()
        self._save_design_matrices()
        self.clean_var_categories()

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

    def get_max_temporal_dependency(self, verbose: bool = True) -> Dict:
        """
        Summarise the maximum temporal dependency of the encoding model.

        Reads from (in decreasing priority):
        1. ``hist_meta['basis_info']`` lags – exact lags after get_design_for_unit
        2. ``temporal_meta['basis_info']`` lags – exact lags after collect_data
        3. ``binrange_dict`` / ``encoder_prs`` – always available as fallback

        Returns
        -------
        dict with keys:
            spike_hist_window, event_windows, max_causal_s, max_acausal_s,
            overall_max_s, overall_max_bins
        """
        sh_min: Optional[float] = None
        sh_max: Optional[float] = None

        hist_meta = getattr(self, 'hist_meta', None) or {}
        for info in (hist_meta.get('basis_info') or {}).values():
            lags = info.get('lags')
            if lags is not None and len(lags):
                sh_min = float(lags[0])
                sh_max = float(lags[-1])
                break

        if sh_max is None:
            sh_range = self.binrange_dict.get('spike_hist')
            if sh_range is not None:
                sh_min, sh_max = float(sh_range[0]), float(sh_range[1])

        ev_min: Optional[float] = None
        ev_max: Optional[float] = None

        temporal_meta = getattr(self, 'temporal_meta', None) or {}
        basis_info_tm = temporal_meta.get('basis_info') or {}
        first_tm_info = next(iter(basis_info_tm.values()), None)
        if first_tm_info is not None:
            lags = first_tm_info.get('lags')
            if lags is not None and len(lags):
                ev_min = float(lags[0])
                ev_max = float(lags[-1])

        if ev_min is None:
            for key in ('t_stop', 't_event'):
                br = self.binrange_dict.get(key)
                if br is not None:
                    ev_min, ev_max = float(br[0]), float(br[1])
                    break

        if ev_min is None:
            ev_min = -float(getattr(self.encoder_prs, 'pre_event', 0.3))
            ev_max = float(getattr(self.encoder_prs, 'post_event', 0.3))

        event_groups = temporal_meta.get('groups') or {}
        event_windows: Dict[str, tuple] = {
            name: (ev_min, ev_max) for name in event_groups
        }
        if not event_windows and ev_min is not None:
            event_windows['(default_event)'] = (ev_min, ev_max)

        causal_candidates: List[float] = []
        acausal_candidates: List[float] = [0.0]
        if sh_max is not None:
            causal_candidates.append(abs(sh_max))
        for t0, t1 in event_windows.values():
            if t0 is not None:
                causal_candidates.append(abs(t0))
            if t1 is not None:
                acausal_candidates.append(abs(t1))

        max_causal = max(causal_candidates) if causal_candidates else 0.0
        max_acausal = max(acausal_candidates)
        overall = max(max_causal, max_acausal)
        overall_bins = int(round(overall / self.bin_width))

        result = {
            'spike_hist_window': (sh_min, sh_max),
            'event_windows': event_windows,
            'max_causal_s': max_causal,
            'max_acausal_s': max_acausal,
            'overall_max_s': overall,
            'overall_max_bins': overall_bins,
        }

        if verbose:
            fmt = lambda t: f"{t * 1000:.1f} ms" if t is not None else "N/A"
            print("── Temporal dependency summary ──")
            print(f"  Spike history :  {fmt(sh_min)} → {fmt(sh_max)}  (causal lookback)")
            for name, (t0, t1) in event_windows.items():
                print(f"  Event [{name:>20s}]:  {fmt(t0)} → {fmt(t1)}")
            print(f"  Max causal    :  {fmt(max_causal)}")
            print(f"  Max acausal   :  {fmt(max_acausal)}")
            print(f"  Overall max   :  {fmt(overall)}  ({overall_bins} bins @ {self.bin_width*1000:.0f} ms)")

        return result

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
            "binned_feats", "binned_spikes", "raw_behavioral_feats",
            "binrange_dict", "bin_df",
            "temporal_meta", "tuning_meta", "spk_colnames",
            "meta_df_used", "trial_ids",
        ]
        return {k: save_dir / f"{k}.pkl" for k in keys}

    def load_cached_spk_colnames(self) -> bool:
        """Load cached spike-history column metadata when available."""
        if self.spk_colnames is not None:
            return True

        path = self._get_design_matrix_paths()["spk_colnames"]
        if not path.exists():
            return False

        try:
            with open(path, "rb") as f:
                self.spk_colnames = pickle.load(f)
            if self.spk_colnames is None:
                return False
            self.spike_cols = list(self.spk_colnames.keys())
            return True
        except Exception as e:
            print(
                f"[{type(self).__name__}] WARNING: could not load cached "
                f"spk_colnames: {type(e).__name__}: {e}"
            )
            return False

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths["binned_feats"].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "binned_feats": self.binned_feats,
            "binned_spikes": self.binned_spikes,
            "raw_behavioral_feats": getattr(self, "raw_behavioral_feats", None),
            "binrange_dict": self.binrange_dict,
            "temporal_meta": self.temporal_meta,
            "tuning_meta": self.tuning_meta,
            "bin_df": getattr(self, "bin_df", None),
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
        if not paths["raw_behavioral_feats"].exists():
            print(
                f"[{type(self).__name__}] Cached design matrices are missing "
                "raw_behavioral_feats; recomputing from source data."
            )
            return False

        try:
            with open(paths["binned_feats"], "rb") as f:
                self.binned_feats = pickle.load(f)
            with open(paths["binned_spikes"], "rb") as f:
                self.binned_spikes = pickle.load(f)
            with open(paths["raw_behavioral_feats"], "rb") as f:
                self.raw_behavioral_feats = pickle.load(f)
            if self.raw_behavioral_feats is None:
                print(
                    f"[{type(self).__name__}] Cached raw_behavioral_feats is empty; "
                    "recomputing from source data."
                )
                return False
            with open(paths["binrange_dict"], "rb") as f:
                self.binrange_dict = pickle.load(f)
            with open(paths["temporal_meta"], "rb") as f:
                self.temporal_meta = pickle.load(f)
            with open(paths["tuning_meta"], "rb") as f:
                self.tuning_meta = pickle.load(f)
            with open(paths["bin_df"], "rb") as f:
                self.bin_df = pickle.load(f)
            if not self.load_cached_spk_colnames():
                self.spk_colnames = None
                self.spike_cols = None
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