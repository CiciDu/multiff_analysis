"""
Encoding model classes (pure modeling — no data loading).

Provided models
---------------
PGAMModel   Poisson GAM with penalty tuning, backward elimination, category
            contributions.  Delegates to the existing one_ff_gam pipeline.

RNNModel    GRU + Poisson NLL, with GroupKFold cross-validation.

Both share the interface:

    model.fit(task, unit_idx)       → fit result
    model.crossval(task, unit_idx)  → cross-val result dict

Usage
-----
    from encoding_tasks import PNEncodingTask
    from encoding_models import PGAMModel, RNNModel

    task = PNEncodingTask(raw_data_folder_path)
    runner = EncodingRunner(task, PGAMModel())
    runner.crossval(unit_idx=0)

    runner2 = EncodingRunner(task, RNNModel())
    runner2.crossval(unit_idx=0)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoder_gam_helper,
    encoding_design_utils,
    linear_model_utils,
    process_encode_design,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    gam_variance_explained,
    one_ff_gam_fit,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    FitResult,
    GroupSpec,
)


# ===========================================================================
# Shared base
# ===========================================================================

class BaseEncodingModel:
    """
    Base class for encoding models.

    Subclasses must implement fit() and crossval().

    Also provides ANOVA and LM analyses operating on
    task.get_raw_behavioral_feats() — the same raw behavioral variables
    that decoding tasks expose as feats_to_decode.  These are encoding
    questions (behavioral variable → neural response) so they belong here.
    """

    def fit(self, task, unit_idx: int, **kwargs):
        raise NotImplementedError

    def crossval(self, task, unit_idx: int, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers shared by ANOVA and LM
    # ------------------------------------------------------------------

    def _behavioral_feats(self, task) -> "pd.DataFrame":
        """Return the raw behavioral feature matrix for ANOVA/LM.

        Uses task.get_raw_behavioral_feats() when available (encoding
        tasks), falling back to task.feats_to_decode (decoding tasks).
        """
        if hasattr(task, "get_raw_behavioral_feats"):
            try:
                return task.get_raw_behavioral_feats()
            except RuntimeError:
                pass
        if hasattr(task, "feats_to_decode") and task.feats_to_decode is not None:
            return task.feats_to_decode
        raise RuntimeError("No behavioral feature matrix available. Call collect_data() first.")

    def find_categorical_vars(
        self, task, max_unique: int = 20, min_unique: int = 2
    ) -> List[str]:
        """Discover integer-valued low-cardinality columns suitable for ANOVA/LM.

        A column is included when it is integer-like (or 0/1 float), has
        between min_unique and max_unique distinct values, and is not a
        spline/basis column.

        Note: event_vars (e.g. ff_on, ff_off) are intentionally NOT excluded —
        they are raw binary indicators in raw_behavioral_feats and are exactly
        the kind of categorical variable ANOVA should test.  Only columns that
        are already basis-expanded (identified by a trailing digit suffix like
        _0, _1, _12 or listed in temporal_meta/tuning_meta groups) are skipped.
        """
        feats = self._behavioral_feats(task)

        # Collect spline/basis column names from meta when available.
        # raw_behavioral_feats never contains these, but guard for robustness.
        basis_cols: set = set()
        for meta_attr in ("temporal_meta", "tuning_meta"):
            meta = getattr(task, meta_attr, None) or {}
            for col_list in (meta.get("groups") or {}).values():
                basis_cols.update(col_list)

        _basis_suffix = re.compile(r"_\d+$")
        categorical_cols: List[str] = []
        for col in feats.columns:
            if col in basis_cols or _basis_suffix.search(col):
                continue
            series = feats[col]
            notnull = series.dropna()
            if len(notnull) == 0:
                continue
            is_int_like = pd.api.types.is_integer_dtype(series) or (
                pd.api.types.is_float_dtype(series)
                and bool((notnull == notnull.round()).all())
            )
            if not is_int_like:
                continue
            if min_unique <= int(series.nunique(dropna=True)) <= max_unique:
                categorical_cols.append(col)
        return categorical_cols

    def _results_cache_path(self, task, label: str):
        try:
            return Path(task._get_save_dir()) / f"{label}_results.pkl"
        except Exception:
            return None

    def _load_results_cache(self, task, cache_path, label: str):
        import pickle as _pkl
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                results = _pkl.load(f)
            print(f"[{type(self).__name__}] {label.upper()}: loaded from cache ({cache_path})")
            return results
        except Exception as e:
            print(f"[{type(self).__name__}] {label.upper()}: cache load failed ({e}) — re-running.")
            return None

    def _save_results_cache(self, task, results, cache_path, label: str):
        import pickle as _pkl
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                _pkl.dump(results, f, protocol=_pkl.HIGHEST_PROTOCOL)
            print(f"[{type(self).__name__}] {label.upper()}: saved to cache ({cache_path})")
        except Exception as e:
            print(f"[{type(self).__name__}] {label.upper()}: cache save failed ({e})")

    def _resolve_lm_covariate_cols(
        self, task,
        categorical_cols: List[str],
        continuous_cols: Optional[List[str]],
        covariates_cache: Optional[str],
    ) -> List[str]:
        import json
        feats = self._behavioral_feats(task)
        cache_path = (
            Path(covariates_cache) if covariates_cache is not None
            else (Path(task._get_save_dir()) / "lm_covariates_cache.json"
                  if hasattr(task, "_get_save_dir") else None)
        )
        if cache_path is not None and cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass

        exclude = set(categorical_cols) | set(continuous_cols or [])
        candidates = [c for c in feats.columns if c not in exclude]
        candidate_df = feats[candidates]
        reduced_df = process_encode_design.reduce_encoding_design(
            candidate_df, corr_threshold_for_lags=0.95, vif_threshold=20
        )
        covariate_cols = list(reduced_df.columns)

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(covariate_cols, f)
            except Exception:
                pass
        return covariate_cols

    # ------------------------------------------------------------------
    # ANOVA
    # ------------------------------------------------------------------

    def run_anova_for_categorical_vars(
        self, task, unit_idx: int,
        categorical_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> "pd.DataFrame":
        """One-way ANOVA of spike counts across levels of each categorical variable."""
        task.collect_data(exists_ok=True)
        feats = self._behavioral_feats(task)
        if categorical_cols is None:
            categorical_cols = self.find_categorical_vars(task)
        y = np.asarray(task.binned_spikes.iloc[:, unit_idx].to_numpy(copy=True), dtype=float).ravel()
        result_df = linear_model_utils.anova_spike_counts_for_columns(
            y=y, binned_feats=feats, categorical_cols=categorical_cols, alpha=alpha,
        )
        if verbose:
            linear_model_utils.print_anova_single_unit(result_df, unit_idx, alpha)
        return result_df

    def run_anova_all_neurons(
        self, task,
        categorical_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        verbose: bool = True,
        results_cache: Optional[str] = None,
        force_rerun: bool = False,
    ) -> Dict[int, "pd.DataFrame"]:
        """ANOVA across all neurons; results cached to disk."""
        cache_path = (
            None if results_cache == ""
            else Path(results_cache) if results_cache is not None
            else self._results_cache_path(task, "anova")
        )
        if not force_rerun and cache_path is not None:
            cached = self._load_results_cache(task, cache_path, "anova")
            if cached is not None:
                if verbose:
                    linear_model_utils.print_anova_all_neurons(cached, alpha)
                return cached

        task.collect_data(exists_ok=True)
        cols = categorical_cols or getattr(task, "categorical_vars", None) or                self.find_categorical_vars(task)
        if not cols:
            print(f"[{type(self).__name__}] No categorical vars — ANOVA skipped.")
            return {}

        results = {
            unit_idx: self.run_anova_for_categorical_vars(
                task, unit_idx=unit_idx, categorical_cols=cols, alpha=alpha, verbose=False,
            )
            for unit_idx in range(task.binned_spikes.shape[1])
        }
        self._save_results_cache(task, results, cache_path, "anova")
        if verbose:
            linear_model_utils.print_anova_all_neurons(results, alpha)
        return results

    def plot_anova_results(
        self, task,
        anova_results: Dict[int, "pd.DataFrame"],
        alpha: float = 0.05,
        figsize=None,
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        if not anova_results:
            print(f"[{type(self).__name__}] plot_anova_results: empty results.")
            return plt.figure()
        if title is None:
            title = f"ANOVA — {type(task).__name__} ({len(anova_results)} neurons, α={alpha})"
        return linear_model_utils.plot_anova_results(
            anova_results=anova_results, alpha=alpha, figsize=figsize, title=title,
        )

    # ------------------------------------------------------------------
    # LM (partial F-tests)
    # ------------------------------------------------------------------

    def run_lm_for_categorical_vars(
        self, task, unit_idx: int,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        include_all_feats: bool = True,
        covariates_cache: Optional[str] = None,
        alpha: float = 0.05,
        verbose: bool = True,
        _covariate_cols: Optional[List[str]] = None,
    ) -> "pd.DataFrame":
        """Partial F-test (LM) for each categorical variable in one neuron."""
        task.collect_data(exists_ok=True)
        feats = self._behavioral_feats(task)
        if categorical_cols is None:
            categorical_cols = self.find_categorical_vars(task)

        covariate_cols = _covariate_cols
        if covariate_cols is None and include_all_feats:
            covariate_cols = self._resolve_lm_covariate_cols(
                task, categorical_cols=categorical_cols,
                continuous_cols=continuous_cols, covariates_cache=covariates_cache,
            )

        y = np.asarray(task.binned_spikes.iloc[:, unit_idx].to_numpy(copy=True), dtype=float).ravel()
        result_df = linear_model_utils.lm_spike_counts_for_columns(
            y=y, binned_feats=feats,
            categorical_cols=categorical_cols, continuous_cols=continuous_cols,
            covariate_cols=covariate_cols, alpha=alpha,
        )
        if verbose:
            linear_model_utils.print_lm_single_unit(result_df, unit_idx, alpha)
        return result_df

    def run_lm_all_neurons(
        self, task,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        include_all_feats: bool = True,
        covariates_cache: Optional[str] = None,
        alpha: float = 0.05,
        verbose: bool = True,
        results_cache: Optional[str] = None,
        force_rerun: bool = False,
    ) -> Dict[int, "pd.DataFrame"]:
        """LM partial F-tests across all neurons."""
        cache_path = (
            None if results_cache == ""
            else Path(results_cache) if results_cache is not None
            else self._results_cache_path(task, "lm")
        )
        if not force_rerun and cache_path is not None:
            cached = self._load_results_cache(task, cache_path, "lm")
            if cached is not None:
                if verbose:
                    linear_model_utils.print_lm_all_neurons(cached, alpha)
                return cached

        task.collect_data(exists_ok=True)
        cols = categorical_cols or getattr(task, "categorical_vars", None) or                self.find_categorical_vars(task)
        if not cols:
            print(f"[{type(self).__name__}] No categorical vars — LM skipped.")
            return {}

        resolved_covariates = None
        if include_all_feats:
            resolved_covariates = self._resolve_lm_covariate_cols(
                task, categorical_cols=cols, continuous_cols=continuous_cols,
                covariates_cache=covariates_cache,
            )

        results = {
            unit_idx: self.run_lm_for_categorical_vars(
                task, unit_idx=unit_idx, categorical_cols=cols,
                continuous_cols=continuous_cols, include_all_feats=False,
                _covariate_cols=resolved_covariates, alpha=alpha, verbose=False,
            )
            for unit_idx in range(task.binned_spikes.shape[1])
        }
        self._save_results_cache(task, results, cache_path, "lm")
        if verbose:
            linear_model_utils.print_lm_all_neurons(results, alpha)
        return results

    def plot_lm_vs_anova_results(
        self, task,
        anova_results: Dict[int, "pd.DataFrame"],
        lm_results: Dict[int, "pd.DataFrame"],
        alpha: float = 0.05,
        figsize=None,
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        if not anova_results or not lm_results:
            print(f"[{type(self).__name__}] plot_lm_vs_anova_results: empty results.")
            return plt.figure()
        if title is None:
            title = f"ANOVA vs LM — {type(task).__name__} ({len(anova_results)} neurons, α={alpha})"
        return linear_model_utils.plot_lm_vs_anova(
            anova_results=anova_results, lm_results=lm_results,
            alpha=alpha, figsize=figsize, title=title,
        )


# ===========================================================================
# PGAMModel
# ===========================================================================

DEFAULT_LAMBDA_CONFIG = {"lam_f": 100.0, "lam_g": 10.0, "lam_h": 10.0, "lam_p": 10.0}

DEFAULT_LAM_GRID = {
    "lam_f": [50, 100, 200],
    "lam_g": [50, 100, 200],
    "lam_h": [5, 10, 30],
    "lam_p": [10],
}


class PGAMModel(BaseEncodingModel):
    """
    Poisson GAM model.

    All GAM-specific logic (design construction per unit, group penalties,
    backward elimination, category contributions) lives here.

    Parameters
    ----------
    lambda_config : dict, optional
        Penalty values {lam_f, lam_g, lam_h, lam_p}.
    lam_grid : dict, optional
        Grid for penalty tuning.
    cv_mode : str, optional
        Cross-validation strategy ('blocked_time_buffered', 'group_kfold', …).
    """

    def __init__(
        self,
        lambda_config: Optional[Dict] = None,
        lam_grid: Optional[Dict] = None,
        cv_mode: Optional[str] = "blocked_time_buffered",
    ):
        self.lambda_config = lambda_config or dict(DEFAULT_LAMBDA_CONFIG)
        self.lam_grid = lam_grid or dict(DEFAULT_LAM_GRID)
        self.cv_mode = cv_mode

    # ------------------------------------------------------------------
    # Per-unit design helpers (depend on task data)
    # ------------------------------------------------------------------

    def _get_design_for_unit(self, task, unit_idx: int):
        """Build design_df + gam_groups for one unit from task state."""
        task._prepare_spike_history_components()

        if unit_idx < 0 or unit_idx >= len(task.spk_colnames):
            raise IndexError(
                f"unit_idx {unit_idx} out of range [0, {len(task.spk_colnames)})"
            )
        target_col = list(task.spk_colnames.keys())[unit_idx]
        cross_neurons = (
            [c for c in task.spk_colnames.keys() if c != target_col]
            if task.use_neural_coupling
            else None
        )

        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=task.binned_feats.reset_index(drop=True),
            colnames=task.spk_colnames,
            X_hist=task.X_hist,
            target_col=target_col,
            include_self=True,
            cross_neurons=cross_neurons,
            meta_groups=None,
        )

        # Drop constant columns except 'const'
        const_cols = [
            c for c in design_df.columns
            if c != "const" and design_df[c].nunique() <= 1
        ]
        design_df = design_df.drop(columns=const_cols)

        task.make_hist_meta_for_unit(unit_idx)
        task._make_structured_meta_groups()

        if task.use_neural_coupling:
            hist_groups = (task.hist_meta or {}).get("groups", {})
            task.var_categories["coupling_vars"] = sorted(
                g for g in hist_groups if str(g).startswith("cpl_")
            )

        gam_groups = encoding_design_utils.build_gam_groups_from_meta(
            task.structured_meta_groups,
            lam_f=self.lambda_config["lam_f"],
            lam_g=self.lambda_config["lam_g"],
            lam_h=self.lambda_config["lam_h"],
            lam_p=self.lambda_config["lam_p"],
        )
        encoding_design_utils._validate_design_columns(design_df, gam_groups)

        return design_df, gam_groups

    @staticmethod
    def _get_effective_num_neurons(task) -> int:
        """Number of neurons valid across binned_spikes and spike-history design."""
        if hasattr(task, "get_effective_num_neurons"):
            return int(task.get_effective_num_neurons())
        n_spikes = int(task.binned_spikes.shape[1])
        n_spk_cols = len(task.spk_colnames) if getattr(task, "spk_colnames", None) else n_spikes
        return min(n_spikes, n_spk_cols)

    # ------------------------------------------------------------------
    # Save paths
    # ------------------------------------------------------------------

    @staticmethod
    def _coupling_subdir(use_neural_coupling: bool) -> str:
        return "coupling" if use_neural_coupling else "no_coupling"

    def get_gam_save_paths(
        self,
        task,
        unit_idx: int,
        *,
        gam_results_subdir: str,
        ensure_dirs: bool = True,
        cv_mode: Optional[str] = None,
        n_folds: Optional[int] = 10,
    ) -> dict:
        coupling_subdir = self._coupling_subdir(task.use_neural_coupling)
        base = Path(task._get_save_dir()) / gam_results_subdir / coupling_subdir

        if ensure_dirs:
            base.mkdir(parents=True, exist_ok=True)

        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(lambda_config=self.lambda_config)
        outdir = base / f"neuron_{unit_idx}"
        if ensure_dirs:
            (outdir / "fit_results").mkdir(parents=True, exist_ok=True)
            (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        cv_dir = outdir / "cv_var_explained"
        if effective_cv:
            cv_dir = cv_dir / str(effective_cv)
            if ensure_dirs:
                cv_dir.mkdir(parents=True, exist_ok=True)

        cv_filename_suffix = lam_suffix
        if effective_cv and n_folds is not None:
            cv_filename_suffix = f"{lam_suffix}_nfolds{n_folds}"

        return {
            "base": base,
            "outdir": outdir,
            "lambda_config": self.lambda_config,
            "lam_suffix": lam_suffix,
            "fit_save_path": str(outdir / "fit_results" / f"{lam_suffix}.pkl"),
            "cv_save_path": str(cv_dir / f"{cv_filename_suffix}.pkl"),
        }

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def fit(
        self,
        task,
        unit_idx: int,
        *,
        gam_results_subdir: str = "gam_results",
        l1_groups: Optional[List[GroupSpec]] = None,
        l1_smooth_eps: float = 1e-6,
        max_iter: int = 1000,
        tol: float = 1e-6,
        optimizer: str = "L-BFGS-B",
        verbose: bool = True,
        save_path: Optional[str] = None,
        save_design: bool = False,
        save_metadata: Optional[Dict] = None,
        load_if_exists: bool = True,
    ) -> FitResult:
        """Fit Poisson GAM for one unit."""
        task.collect_data(exists_ok=True)

        design_df, gam_groups = self._get_design_for_unit(task, unit_idx)

        if len(design_df) != len(task.binned_spikes):
            raise ValueError(
                f"design / binned_spikes row mismatch: "
                f"{len(design_df)} vs {len(task.binned_spikes)}"
            )

        y = np.asarray(task.binned_spikes.iloc[:, unit_idx].to_numpy(copy=True), dtype=float).ravel()

        if save_path is None:
            paths = self.get_gam_save_paths(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
            )
            save_path = paths["fit_save_path"]

        return one_ff_gam_fit.fit_poisson_gam(
            design_df=design_df,
            y=y,
            groups=gam_groups,
            l1_groups=l1_groups,
            l1_smooth_eps=l1_smooth_eps,
            max_iter=max_iter,
            tol=tol,
            optimizer=optimizer,
            verbose=verbose,
            save_path=save_path,
            save_design=save_design,
            save_metadata=save_metadata,
            load_if_exists=load_if_exists,
        )

    def crossval(
        self,
        task,
        unit_idx: int,
        *,
        gam_results_subdir: str = "gam_results",
        n_folds: int = 5,
        random_state: int = 0,
        fit_kwargs: Optional[Dict] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        save_metadata: Optional[Dict] = None,
        verbose: bool = True,
        dt: Optional[float] = None,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
        cv_groups=None,
    ) -> Dict:
        """Cross-validated variance explained for one unit."""
        task.collect_data(exists_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode

        if save_path is None:
            paths = self.get_gam_save_paths(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
                cv_mode=effective_cv,
                n_folds=n_folds,
            )
            save_path = paths["cv_save_path"]

        maybe_loaded = gam_variance_explained.maybe_load_saved_crossval(
            save_path=save_path, load_if_exists=load_if_exists, verbose=verbose
        )
        if maybe_loaded is not None:
            if verbose:
                print(f"[PGAMModel] Loaded cached CV results from: {save_path}")
            return maybe_loaded

        design_df, gam_groups = self._get_design_for_unit(task, unit_idx)

        if len(design_df) != len(task.binned_spikes):
            raise ValueError(
                f"design / binned_spikes row mismatch: "
                f"{len(design_df)} vs {len(task.binned_spikes)}"
            )

        y = np.asarray(task.binned_spikes.iloc[:, unit_idx].to_numpy(copy=True), dtype=float).ravel()
        dt = dt if dt is not None else float(task.bin_width)
        fit_kwargs = fit_kwargs or {}

        if effective_cv == "group_kfold" and cv_groups is None:
            cv_groups = task.get_cv_groups_for_design(design_df)
        if effective_cv == "group_kfold" and cv_groups is None:
            raise ValueError(
                "cv_mode='group_kfold' requires per-sample group labels."
            )

        meta = dict(save_metadata or {}, unit_idx=unit_idx)
        return gam_variance_explained._crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=design_df,
            y=y,
            groups=gam_groups,
            dt=dt,
            n_folds=n_folds,
            random_state=random_state,
            fit_kwargs=fit_kwargs,
            save_path=save_path,
            load_if_exists=load_if_exists,
            save_metadata=meta,
            verbose=verbose,
            cv_mode=effective_cv,
            buffer_samples=buffer_samples,
            cv_groups=cv_groups,
        )

    def crossval_all_neurons(
        self,
        task,
        *,
        gam_results_subdir: str = "gam_results",
        n_folds: int = 5,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        fit_kwargs: Optional[Dict] = None,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
        plot_cdf: bool = True,
        log_x: bool = False,
    ) -> Optional[List[float]]:
        """Run crossval for every neuron in the session."""
        task.collect_data(exists_ok=True)
        task._prepare_spike_history_components()
        effective_cv = cv_mode if cv_mode is not None else self.cv_mode

        if retrieve_only:
            if not task._load_design_matrices():
                print('No cached data found')
                return None

        n_neurons = self._get_effective_num_neurons(task)
        if unit_indices is None:
            unit_indices = list(range(n_neurons))

        if fit_kwargs is None:
            fit_kwargs = dict(
                l1_groups=[], max_iter=1000, tol=1e-6, verbose=False, save_path=None
            )

        all_r2 = []
        for unit_idx in unit_indices:
            paths = self.get_gam_save_paths(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
                cv_mode=effective_cv,
                n_folds=n_folds,
            )
            res = self.crossval(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
                n_folds=n_folds,
                fit_kwargs=fit_kwargs,
                save_path=paths["cv_save_path"],
                load_if_exists=load_if_exists,
                retrieve_only=retrieve_only,
                cv_mode=effective_cv,
                buffer_samples=buffer_samples,
                verbose=verbose,
            )
            if verbose:
                print(res["mean_classical_r2"], res["mean_pseudo_r2"])
            all_r2.append(res["mean_pseudo_r2"])

        if plot_cdf:
            gam_variance_explained.plot_variance_explained_cdf(all_r2, log_x=log_x)

        return all_r2

    # ------------------------------------------------------------------
    # GAM analysis helpers (category contributions, penalty tuning, etc.)
    # ------------------------------------------------------------------

    def _get_analysis_helper(self, task):
        """Return a GAM analysis helper bound to (task, model)."""
        return encoder_gam_helper.BaseEncodingGAMAnalysisHelper(task, self)

    def run_category_variance_contributions(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 5,
        buffer_samples: int = 20,
        category_names: Optional[List[str]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_analysis_helper(task).run_category_variance_contributions(
            unit_idx=unit_idx,
            lambda_config=self.lambda_config,
            n_folds=n_folds,
            buffer_samples=buffer_samples,
            category_names=category_names,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
            cv_mode=cv_mode,
        )

    def run_penalty_tuning(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 5,
        group_name_map: Optional[Dict] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_analysis_helper(task).run_penalty_tuning(
            unit_idx=unit_idx,
            lambda_config=self.lambda_config,
            n_folds=n_folds,
            group_name_map=group_name_map,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
            cv_mode=cv_mode,
        )

    def run_backward_elimination(
        self,
        task,
        unit_idx: int,
        *,
        alpha: float = 0.05,
        n_folds: int = 10,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> Dict:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_analysis_helper(task).run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=self.lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
        )

    def crossval_tuning_curve_coef(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 10,
        random_state: int = 0,
        fit_kwargs: Optional[Dict] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        save_metadata: Optional[Dict] = None,
        verbose: bool = True,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
        cv_groups=None,
    ) -> Dict:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_analysis_helper(task).crossval_tuning_curve_coef(
            unit_idx=unit_idx,
            lambda_config=self.lambda_config,
            n_folds=n_folds,
            random_state=random_state,
            fit_kwargs=fit_kwargs,
            save_path=save_path,
            load_if_exists=load_if_exists,
            save_metadata=save_metadata,
            verbose=verbose,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            cv_groups=cv_groups,
        )

    def try_load_variance_explained_for_all_neurons(
        self,
        task,
        *,
        n_folds: int = 5,
        load_if_exists: bool = True,
        verbose: bool = True,
        gam_results_subdir: str = "gam_results",
    ) -> Optional[List[float]]:
        """Try to load cached CV results for every neuron without recomputing."""
        all_r2 = []
        task.collect_data(exists_ok=True)
        task._prepare_spike_history_components()
        n_neurons = self._get_effective_num_neurons(task)
        for unit_idx in range(n_neurons):
            try:
                paths = self.get_gam_save_paths(
                    task,
                    unit_idx,
                    gam_results_subdir=gam_results_subdir,
                    cv_mode=self.cv_mode,
                    n_folds=n_folds,
                )
                maybe = gam_variance_explained.maybe_load_saved_crossval(
                    save_path=paths["cv_save_path"],
                    load_if_exists=load_if_exists,
                    verbose=verbose,
                )
                if maybe is None:
                    return None
                all_r2.append(maybe["mean_pseudo_r2"])
            except Exception as e:
                if verbose:
                    print(f"[PGAMModel] try_load unit {unit_idx} failed: {e}")
                return None
        return all_r2

    def run_full_analysis(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 5,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        verbose: bool = True,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> None:
        """Category contributions + backward elimination for one unit."""
        if verbose:
            print(f"[PGAMModel] Full analysis for unit {unit_idx}")
        try:
            self.run_category_variance_contributions(
                task, unit_idx,
                n_folds=n_folds,
                buffer_samples=buffer_samples,
                load_if_exists=load_if_exists,
                cv_mode=cv_mode,
            )
            self.run_backward_elimination(
                task, unit_idx,
                n_folds=backward_n_folds,
                alpha=alpha,
                load_if_exists=load_if_exists,
                cv_mode=cv_mode,
                buffer_samples=buffer_samples,
            )
        except Exception as e:
            if verbose:
                print(f"  [WARN] unit {unit_idx}: {e}")

    def run_full_analysis_all_neurons(
        self,
        task,
        *,
        unit_indices: Optional[List[int]] = None,
        n_folds: int = 5,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        verbose: bool = True,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> None:
        task.collect_data(exists_ok=True)
        task._prepare_spike_history_components()
        n_neurons = self._get_effective_num_neurons(task)
        if unit_indices is None:
            unit_indices = list(range(n_neurons))
        for unit_idx in unit_indices:
            self.run_full_analysis(
                task, unit_idx,
                n_folds=n_folds,
                backward_n_folds=backward_n_folds,
                alpha=alpha,
                load_if_exists=load_if_exists,
                verbose=verbose,
                cv_mode=cv_mode,
                buffer_samples=buffer_samples,
            )


# ===========================================================================
# RNNModel  (GRU + Poisson NLL)
# ===========================================================================

import pickle as _pickle  # local alias to avoid shadowing


class _PoissonGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.exp(self.readout(h))  # positive rate, shape (B, T, 1)


class RNNModel(BaseEncodingModel):
    """
    GRU-based neural encoding model with Poisson likelihood.

    Reads from task:
        binned_feats    (n_bins × n_features)  — taken as-is, no splines
        binned_spikes   (n_bins × n_neurons)
        get_cv_groups_for_design(...)           — required for sequence splitting

    Parameters
    ----------
    hidden_dim : int
        GRU hidden size.
    n_epochs : int
        Training epochs per fold / fit call.
    lr : float
        Adam learning rate.
    device : str
        'cpu' or 'cuda'.
    verbose : bool
    rnn_results_subdir : str
        Sub-directory under task._get_save_dir() for cached CV results.
        Defaults to 'rnn_results'.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_epochs: int = 100,
        lr: float = 3e-4,
        device: str = "cpu",
        verbose: bool = True,
        rnn_results_subdir: str = "rnn_results",
        cv_mode: str = "blocked_time_buffered",
        buffer_samples: int = 20,
        use_raw_feats: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self.rnn_results_subdir = rnn_results_subdir
        self.cv_mode = cv_mode
        self.buffer_samples = buffer_samples
        self.use_raw_feats = use_raw_feats

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def _get_groups(self, task) -> np.ndarray:
        groups = task.get_cv_groups_for_design(task.binned_feats)
        if groups is None:
            raise ValueError(
                "RNNModel requires per-trial grouping (e.g. new_segment column)."
            )
        return np.asarray(groups)

    def _get_X(self, task) -> np.ndarray:
        """Return the feature matrix fed to the RNN.

        When ``use_raw_feats=True`` (default), uses
        ``task.get_raw_behavioral_feats()`` — raw kinematics with no splines
        and no spike history, matching what decoding tasks expose as
        ``feats_to_decode``.  Set ``use_raw_feats=False`` to fall back to
        the full spline-expanded ``task.binned_feats``.
        """
        if self.use_raw_feats:
            return task.get_raw_behavioral_feats().to_numpy()
        return task.binned_feats.to_numpy()

    def _make_sequences(self, X: np.ndarray, y: np.ndarray,
                        groups: np.ndarray, idx_set: np.ndarray):
        """Build list of (X_trial, y_trial) arrays for rows in idx_set only."""
        X_list, y_list = [], []
        sub_groups = groups[idx_set]
        for g in np.unique(sub_groups):
            mask = idx_set[sub_groups == g]   # indices into full array
            X_list.append(X[mask])
            y_list.append(y[mask, None])      # shape (T, 1)
        return X_list, y_list

    def _to_tensors(self, X_list, y_list):
        def t(a, dt):
            return torch.tensor(np.asarray(a, dtype=np.float32), dtype=dt, device=self.device)
        return (
            [t(x, torch.float32) for x in X_list],
            [t(y, torch.float32) for y in y_list],
        )

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_nll(rate, y, eps: float = 1e-8):
        return torch.mean(rate - y * torch.log(rate + eps))

    def _train(self, model: _PoissonGRU, X_list, y_list) -> List[float]:
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        epoch_losses = []
        model.train()
        for epoch in range(self.n_epochs):
            total = 0.0
            for x, y in zip(X_list, y_list):
                rate = model(x.unsqueeze(0))        # (1, T, 1)
                loss = self._poisson_nll(rate, y.unsqueeze(0))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            epoch_losses.append(total)
            if self.verbose:
                print(f"[RNNModel] epoch {epoch:>3d}  loss {total:.4f}")
        return epoch_losses

    @torch.no_grad()
    def _evaluate(self, model: _PoissonGRU, X_list, y_list):
        """
        Returns (pseudo_r2, classical_r2).

        pseudo_r2   : Poisson log-likelihood ratio vs constant-rate null
        classical_r2: 1 - SS_res/SS_tot on predicted rate vs spike counts
        """
        model.eval()
        y_all = torch.cat(y_list, dim=0)
        mean_rate = torch.mean(y_all)

        all_y, all_pred = [], []
        ll_model = ll_null = 0.0

        for x, y in zip(X_list, y_list):
            rate = model(x.unsqueeze(0)).squeeze(0)  # (T, 1)
            ll_model += torch.sum(y * torch.log(rate + 1e-8) - rate).item()
            ll_null  += torch.sum(y * torch.log(mean_rate + 1e-8) - mean_rate).item()
            all_y.append(y.cpu().numpy())
            all_pred.append(rate.cpu().numpy())

        pseudo_r2 = float(1.0 - ll_model / ll_null) if ll_null != 0.0 else float("nan")

        y_np    = np.concatenate(all_y).ravel()
        pred_np = np.concatenate(all_pred).ravel()
        ss_res  = np.sum((y_np - pred_np) ** 2)
        ss_tot  = np.sum((y_np - y_np.mean()) ** 2)
        classical_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return pseudo_r2, classical_r2

    # ------------------------------------------------------------------
    # Save paths
    # ------------------------------------------------------------------

    def _cv_save_path(self, task, unit_idx: int, *, n_folds: int,
                      cv_mode: Optional[str] = None) -> Path:
        subdir = getattr(self, "_gam_results_subdir", self.rnn_results_subdir)
        base = Path(task._get_save_dir()) / subdir / f"neuron_{unit_idx}"
        base.mkdir(parents=True, exist_ok=True)
        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        return base / f"rnn_cv_{effective_cv}_nfolds{n_folds}_hidden{self.hidden_dim}_epochs{self.n_epochs}.pkl"

    def _fit_save_path(self, task, unit_idx: int) -> Path:
        subdir = getattr(self, "_gam_results_subdir", self.rnn_results_subdir)
        base = Path(task._get_save_dir()) / subdir / f"neuron_{unit_idx}"
        base.mkdir(parents=True, exist_ok=True)
        return base / f"rnn_fit_hidden{self.hidden_dim}_epochs{self.n_epochs}.pt"

    def _full_analysis_dir(self, task) -> Path:
        subdir = getattr(self, "_gam_results_subdir", self.rnn_results_subdir)
        outdir = Path(task._get_save_dir()) / subdir / "full_analysis"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    @staticmethod
    def _alpha_suffix(alpha: float) -> str:
        return str(alpha).replace(".", "p")

    @staticmethod
    def _save_dataframe(df: "pd.DataFrame", path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def _save_figure(fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    @staticmethod
    def _significant_variables(result_df: "pd.DataFrame") -> List[str]:
        if result_df is None or result_df.empty or "significant" not in result_df.columns:
            return []
        return result_df.loc[result_df["significant"], "variable"].tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        task,
        unit_idx: int,
        *,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        **kwargs,
    ) -> _PoissonGRU:
        """Train on all data and return the fitted model. Optionally cache weights."""
        task.collect_data(exists_ok=True)

        if save_path is None:
            save_path = self._fit_save_path(task, unit_idx)
        else:
            save_path = Path(save_path)

        if load_if_exists and save_path.exists():
            if self.verbose:
                print(f"[RNNModel] Loading cached model from {save_path}")
            X = self._get_X(task)
            input_dim = X.shape[1]
            model = _PoissonGRU(input_dim, self.hidden_dim).to(self.device)
            model.load_state_dict(torch.load(save_path, map_location=self.device))
            return model

        groups = self._get_groups(task)
        X = self._get_X(task)
        y = task.binned_spikes.iloc[:, unit_idx].to_numpy(copy=True)
        all_idx = np.arange(len(X))
        X_list, y_list = self._make_sequences(X, y, groups, all_idx)
        X_t, y_t = self._to_tensors(X_list, y_list)

        model = _PoissonGRU(X_t[0].shape[1], self.hidden_dim).to(self.device)
        self._train(model, X_t, y_t)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            print(f"[RNNModel] Saved model weights to {save_path}")

        return model

    def crossval(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 5,
        cv_mode: Optional[str] = None,
        buffer_samples: Optional[int] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Cross-validation using the same fold strategy as PGAMModel.

        cv_mode options: 'blocked_time_buffered', 'blocked_time',
                         'group_kfold', 'kfold'

        Returns
        -------
        dict with keys:
            mean_pseudo_r2  float
            all_folds       List[float]
            epoch_losses    List[List[float]]
            cv_mode         str
        """
        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
            _build_folds,
        )

        task.collect_data(exists_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        effective_buf = buffer_samples if buffer_samples is not None else self.buffer_samples

        if save_path is None:
            save_path = self._cv_save_path(task, unit_idx, n_folds=n_folds, cv_mode=effective_cv)
        else:
            save_path = Path(save_path)
            
        print(f'save_path: {save_path}')

        can_load = (load_if_exists or retrieve_only) and save_path.exists()
        if can_load:
            if self.verbose:
                print(f"[RNNModel] Loading cached CV results from {save_path}")
            with save_path.open("rb") as f:
                return _pickle.load(f)
        if retrieve_only:
            raise FileNotFoundError(f"No cached CV result found at: {save_path}")

        groups = self._get_groups(task)
        X = self._get_X(task)
        y = task.binned_spikes.iloc[:, unit_idx].to_numpy(copy=True)
        n = len(X)

        groups_for_folds = groups if effective_cv == "group_kfold" else None
        splits = _build_folds(
            n,
            n_splits=n_folds,
            groups=groups_for_folds,
            cv_splitter=effective_cv,
            random_state=0,
            buffer_samples=effective_buf,
        )

        scores: List[float] = []
        classical_scores: List[float] = []
        epoch_losses: List[List[float]] = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx  = np.asarray(test_idx,  dtype=int)
            if self.verbose:
                print(f"[RNNModel] Fold {fold + 1}/{n_folds}  "
                      f"train={len(train_idx)} test={len(test_idx)}")

            X_tr, y_tr = self._to_tensors(*self._make_sequences(X, y, groups, train_idx))
            X_te, y_te = self._to_tensors(*self._make_sequences(X, y, groups, test_idx))

            if not X_tr or not X_te:
                if self.verbose:
                    print(f"[RNNModel] Fold {fold + 1}: empty split, skipping")
                scores.append(float("nan"))
                classical_scores.append(float("nan"))
                epoch_losses.append([])
                continue

            model = _PoissonGRU(X_tr[0].shape[1], self.hidden_dim).to(self.device)
            losses = self._train(model, X_tr, y_tr)
            epoch_losses.append(losses)

            pseudo_r2, classical_r2 = self._evaluate(model, X_te, y_te)
            if self.verbose:
                print(f"[RNNModel] Fold {fold + 1}  "
                      f"pseudo-R² {pseudo_r2:.4f}  classical-R² {classical_r2:.4f}")
            scores.append(pseudo_r2)
            classical_scores.append(classical_r2)

        result = {
            "mean_pseudo_r2": float(np.nanmean(scores)),
            "mean_classical_r2": float(np.nanmean(classical_scores)),
            "fold_pseudo_r2": scores,
            "fold_classical_r2": classical_scores,
            "epoch_losses": epoch_losses,
            "unit_idx": unit_idx,
            "n_folds": n_folds,
            "cv_mode": effective_cv,
            "hidden_dim": self.hidden_dim,
            "n_epochs": self.n_epochs,
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            _pickle.dump(result, f, protocol=_pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(f"[RNNModel] Saved CV results to {save_path}")

        return result

    def crossval_all_neurons(
        self,
        task,
        *,
        n_folds: int = 5,
        cv_mode: Optional[str] = None,
        buffer_samples: Optional[int] = None,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
        plot_cdf: bool = True,
        log_x: bool = False,
        **kwargs,
    ) -> List[float]:
        """Run crossval for every neuron; return list of mean pseudo-R²."""
        task.collect_data(exists_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        effective_buf = buffer_samples if buffer_samples is not None else self.buffer_samples

        n_neurons = task.binned_spikes.shape[1]
        if unit_indices is None:
            unit_indices = list(range(n_neurons))

        all_r2 = []
        for unit_idx in unit_indices:
            if verbose:
                print(f"[RNNModel] crossval unit {unit_idx}/{n_neurons - 1}")
            res = self.crossval(
                task, unit_idx,
                n_folds=n_folds,
                cv_mode=effective_cv,
                buffer_samples=effective_buf,
                load_if_exists=load_if_exists,
                retrieve_only=retrieve_only,
            )
            all_r2.append(res["mean_pseudo_r2"])
            if verbose:
                print(f"  pseudo-R² = {res['mean_pseudo_r2']:.4f}")

        if plot_cdf:
            try:
                gam_variance_explained.plot_variance_explained_cdf(all_r2, log_x=log_x)
            except Exception:
                pass

        return all_r2

    def try_load_variance_explained_for_all_neurons(
        self,
        task,
        *,
        n_folds: int = 5,
        cv_mode: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Optional[List[float]]:
        """Try to load cached CV results for every neuron without recomputing."""
        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        n_neurons = task.binned_spikes.shape[1]
        all_r2 = []
        for unit_idx in range(n_neurons):
            save_path = self._cv_save_path(task, unit_idx, n_folds=n_folds, cv_mode=effective_cv)
            if not save_path.exists():
                if verbose:
                    print(f"[RNNModel] No cached result for unit {unit_idx}")
                return None
            try:
                with save_path.open("rb") as f:
                    res = _pickle.load(f)
                all_r2.append(res["mean_pseudo_r2"])
            except Exception as e:
                if verbose:
                    print(f"[RNNModel] Failed to load unit {unit_idx}: {e}")
                return None
        return all_r2

    def run_full_analysis(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 5,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        verbose: bool = True,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> Dict:
        """Run a per-neuron RNN analysis bundle.

        For RNNs, "full analysis" means:
        1. ensure cross-validated model performance is available,
        2. run one-neuron ANOVA over categorical behavioral variables,
        3. run one-neuron LM partial-F tests controlling for other features,
        4. save CSV summaries under the RNN output directory.
        """
        import json

        task.collect_data(exists_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        effective_buf = buffer_samples if buffer_samples is not None else self.buffer_samples
        alpha_suffix = self._alpha_suffix(alpha)
        outdir = self._full_analysis_dir(task) / f"neuron_{unit_idx}"
        outdir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[RNNModel] Full analysis for unit {unit_idx}")

        cv_result = self.crossval(
            task,
            unit_idx,
            n_folds=n_folds,
            cv_mode=effective_cv,
            buffer_samples=effective_buf,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
            verbose=verbose,
        )

        categorical_cols = self.find_categorical_vars(task)
        if not categorical_cols:
            if verbose:
                print("[RNNModel] No categorical vars found; skipping ANOVA/LM.")
            return {"crossval": cv_result, "anova": None, "lm": None, "paths": {}}

        anova_df = self.run_anova_for_categorical_vars(
            task,
            unit_idx=unit_idx,
            categorical_cols=categorical_cols,
            alpha=alpha,
            verbose=verbose,
        )
        lm_df = self.run_lm_for_categorical_vars(
            task,
            unit_idx=unit_idx,
            categorical_cols=categorical_cols,
            include_all_feats=True,
            alpha=alpha,
            verbose=verbose,
        )

        anova_path = outdir / f"anova_alpha{alpha_suffix}.csv"
        lm_path = outdir / f"lm_alpha{alpha_suffix}.csv"
        summary_path = outdir / f"summary_alpha{alpha_suffix}.json"
        self._save_dataframe(anova_df, anova_path)
        self._save_dataframe(lm_df, lm_path)

        summary = {
            "unit_idx": unit_idx,
            "n_folds": n_folds,
            "cv_mode": effective_cv,
            "buffer_samples": effective_buf,
            "mean_pseudo_r2": cv_result.get("mean_pseudo_r2"),
            "mean_classical_r2": cv_result.get("mean_classical_r2"),
            "categorical_vars": categorical_cols,
            "anova_significant": self._significant_variables(anova_df),
            "lm_significant": self._significant_variables(lm_df),
            "anova_csv": str(anova_path),
            "lm_csv": str(lm_path),
        }
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        return {
            "crossval": cv_result,
            "anova": anova_df,
            "lm": lm_df,
            "paths": {
                "anova_csv": str(anova_path),
                "lm_csv": str(lm_path),
                "summary_json": str(summary_path),
            },
        }

    def run_full_analysis_all_neurons(
        self,
        task,
        *,
        unit_indices: Optional[List[int]] = None,
        n_folds: int = 5,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        verbose: bool = True,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> Dict:
        """Run the all-neuron RNN analysis bundle and save summary figures."""
        import matplotlib.pyplot as plt

        task.collect_data(exists_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode
        effective_buf = buffer_samples if buffer_samples is not None else self.buffer_samples
        alpha_suffix = self._alpha_suffix(alpha)
        outdir = self._full_analysis_dir(task)

        n_neurons = task.binned_spikes.shape[1]
        if unit_indices is None:
            unit_indices = list(range(n_neurons))

        crossval_scores = self.crossval_all_neurons(
            task,
            n_folds=n_folds,
            cv_mode=effective_cv,
            buffer_samples=effective_buf,
            load_if_exists=load_if_exists,
            unit_indices=unit_indices,
            verbose=verbose,
            plot_cdf=False,
        )

        categorical_cols = self.find_categorical_vars(task)
        if not categorical_cols:
            if verbose:
                print("[RNNModel] No categorical vars found; skipping ANOVA/LM.")
            return {
                "crossval_scores": crossval_scores,
                "anova": {},
                "lm": {},
                "paths": {},
            }

        if unit_indices == list(range(n_neurons)):
            anova_results = self.run_anova_all_neurons(
                task,
                categorical_cols=categorical_cols,
                alpha=alpha,
                verbose=verbose,
                force_rerun=not load_if_exists,
            )
            lm_results = self.run_lm_all_neurons(
                task,
                categorical_cols=categorical_cols,
                include_all_feats=True,
                alpha=alpha,
                verbose=verbose,
                force_rerun=not load_if_exists,
            )
        else:
            resolved_covariates = self._resolve_lm_covariate_cols(
                task,
                categorical_cols=categorical_cols,
                continuous_cols=None,
                covariates_cache=None,
            )
            anova_results = {
                unit_idx: self.run_anova_for_categorical_vars(
                    task,
                    unit_idx=unit_idx,
                    categorical_cols=categorical_cols,
                    alpha=alpha,
                    verbose=False,
                )
                for unit_idx in unit_indices
            }
            lm_results = {
                unit_idx: self.run_lm_for_categorical_vars(
                    task,
                    unit_idx=unit_idx,
                    categorical_cols=categorical_cols,
                    include_all_feats=False,
                    _covariate_cols=resolved_covariates,
                    alpha=alpha,
                    verbose=False,
                )
                for unit_idx in unit_indices
            }
            if verbose:
                linear_model_utils.print_anova_all_neurons(anova_results, alpha)
                linear_model_utils.print_lm_all_neurons(lm_results, alpha)

        anova_fig = self.plot_anova_results(
            task,
            anova_results,
            alpha=alpha,
            title=f"RNN ANOVA - {type(task).__name__}",
        )
        compare_fig = self.plot_lm_vs_anova_results(
            task,
            anova_results,
            lm_results,
            alpha=alpha,
            title=f"RNN ANOVA vs LM - {type(task).__name__}",
        )

        anova_fig_path = outdir / f"anova_alpha{alpha_suffix}.png"
        compare_fig_path = outdir / f"lm_vs_anova_alpha{alpha_suffix}.png"
        self._save_figure(anova_fig, anova_fig_path)
        self._save_figure(compare_fig, compare_fig_path)
        plt.close(anova_fig)
        plt.close(compare_fig)

        return {
            "crossval_scores": crossval_scores,
            "anova": anova_results,
            "lm": lm_results,
            "paths": {
                "anova_plot": str(anova_fig_path),
                "lm_vs_anova_plot": str(compare_fig_path),
            },
        }