"""
Decoding model classes (pure modeling — no data loading).

Provided models
---------------
OneFFStyleModel     CCA + linear population readout.
CVDecodingModel     Cross-validated model-spec decoding (CatBoost, ridge, etc.)
                    Cross-validated decoding with model selection.

Both share the interface:
    model.run(task, ...)    → result

Usage
-----
    from decoding_tasks import StopTask
    from decoding_models import OneFFStyleModel, CVDecodingModel
    from decoding_runner import DecodingRunner

    runner = DecodingRunner(StopTask(path), OneFFStyleModel())
    runner.run()

    runner = DecodingRunner(StopTask(path), CVDecodingModel())
    runner.run()
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_pipelines import one_ff_style_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    plot_decoding_utils,
    decoding_model_specs,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    linear_model_utils,
    process_encode_design,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
    show_decoding_results,
    plot_decoding_predictions,
)


# ===========================================================================
# Shared base
# ===========================================================================

class BaseDecodingModel:
    def run(self, task, **kwargs):
        raise NotImplementedError

    def fit(self, task, **kwargs):
        return self.run(task, **kwargs)

    @staticmethod
    def _resolve_decoding_save_dir(task, save_dir=None) -> Path:
        """
        Resolve a decoding save_dir with a consistent spike-history component.
        """
        base = Path(task._get_save_dir()) if save_dir is None else Path(save_dir)
        required_component = (
            task._spike_history_save_subdir()
            if bool(getattr(task, "use_spike_history", False))
            else "no_spike_history"
        )
        parts = set(base.parts)
        has_required = required_component in parts
        has_any_spike_history_component = ("no_spike_history" in parts) or any(
            p.startswith("sh_") for p in parts
        )
        if has_required or has_any_spike_history_component:
            return base
        return base / required_component


# ===========================================================================
# Shared save/load helpers
# ===========================================================================

def _path_with_cv_params(save_path, *, cv_mode, n_splits, buffer_samples):
    if save_path is None:
        return None
    p = Path(save_path)
    return p.parent / f"{p.stem}_cv{cv_mode}_n{n_splits}_buf{buffer_samples}{p.suffix}"


def _maybe_load(save_path, load_if_exists, label, verbose, *,
                cv_mode=None, n_splits=None, buffer_samples=None):
    if not load_if_exists or save_path is None:
        return None
    p = (
        _path_with_cv_params(save_path, cv_mode=cv_mode, n_splits=n_splits,
                              buffer_samples=buffer_samples)
        if cv_mode is not None else Path(save_path)
    )
    if not p.exists():
        return None
    with p.open("rb") as f:
        obj = pickle.load(f)
    if verbose:
        print(f"[{label}] loaded: {p}")
    return obj


def _save(save_path, result, label, verbose, *,
          cv_mode=None, n_splits=None, buffer_samples=None):
    if save_path is None:
        return
    p = (
        _path_with_cv_params(save_path, cv_mode=cv_mode, n_splits=n_splits,
                              buffer_samples=buffer_samples)
        if cv_mode is not None else Path(save_path)
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"[{label}] saved: {p}")


# ===========================================================================
# OneFFStyleModel  (CCA + linear readout)
# ===========================================================================

class OneFFStyleModel(BaseDecodingModel):
    """
    One-FF-style population decoding: CCA + per-variable linear readout.

    Parameters
    ----------
    readout_n_splits       : int
    readout_cv_mode        : str
    readout_buffer_samples : int
    fit_kernelwidth        : bool
    candidate_widths       : sequence of int
    fixed_width            : int
    filtwidth              : int   smoothing width for CCA step
    """

    def __init__(
        self,
        readout_n_splits: int = 5,
        readout_cv_mode: str = "blocked_time_buffered",
        readout_buffer_samples: int = 20,
        fit_kernelwidth: bool = True,
        candidate_widths: Sequence[int] = tuple(range(1, 21)),
        fixed_width: int = 25,
        filtwidth: int = 5,
    ):
        self.readout_n_splits = readout_n_splits
        self.readout_cv_mode = readout_cv_mode
        self.readout_buffer_samples = readout_buffer_samples
        self.fit_kernelwidth = fit_kernelwidth
        self.candidate_widths = list(candidate_widths)
        self.fixed_width = fixed_width
        self.filtwidth = filtwidth
        self.stats: Dict = {}

    def run(
        self,
        task,
        *,
        save_dir=None,
        canoncorr_varnames: Optional[Sequence[str]] = None,
        readout_varnames: Optional[Sequence[str]] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        task.collect_data(exists_ok=True)

        save_dir = self._resolve_decoding_save_dir(task, save_dir) / "one_ff_style"
        save_dir.mkdir(parents=True, exist_ok=True)

        canoncorr = self.compute_canoncorr(
            task, varnames=canoncorr_varnames,
            save_path=str(save_dir / "canoncorr.pkl"),
            load_if_exists=load_if_exists, verbose=verbose,
        )
        readout = self.regress_popreadout(
            task, varnames=readout_varnames,
            save_path=str(save_dir / "lineardecoder.pkl"),
            load_if_exists=load_if_exists, verbose=verbose,
        )
        print("[OneFFStyleModel] Finished.")
        return {"canoncorr": canoncorr, "readout": readout, "stats": self.stats}

    # ------------------------------------------------------------------
    # CCA
    # ------------------------------------------------------------------

    def compute_canoncorr(
        self, task, *,
        varnames: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        loaded = _maybe_load(save_path, load_if_exists, "canoncorr", verbose)
        if loaded is not None:
            self.stats["canoncorr"] = loaded
            return loaded

        task.collect_data(exists_ok=True)
        y_df = task._get_numeric_target_df()

        if varnames is None:
            varnames = task._default_canoncorr_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if not varnames:
            raise ValueError(f"No valid canoncorr variables in {task._target_df_error_msg()}.")

        if verbose:
            print(f"[canoncorr] vars: {varnames}")

        x_task = y_df[varnames].to_numpy(dtype=float)
        groups = task._get_groups()
        trial_idx = np.asarray(groups) if groups is not None else None
        y_neural = task._get_neural_matrix_for_decoding(
            neural_smooth_width=int(self.filtwidth), trial_idx=trial_idx
        )
        n_smooth = task._neural_ncols_after_pca() if getattr(task, "use_spike_history", False) else None

        out = one_ff_style_utils.compute_canoncorr_block(
            x_task=x_task, y_neural=y_neural,
            dt=float(task.bin_width), filtwidth=0, neural_cols_to_smooth=n_smooth,
        )
        out["vars"] = list(varnames)
        self.stats["canoncorr"] = out
        _save(save_path, out, "canoncorr", verbose)
        return out

    # ------------------------------------------------------------------
    # Linear readout
    # ------------------------------------------------------------------

    def regress_popreadout(
        self, task, *,
        varnames: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        save_predictions: bool = False,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        loaded = _maybe_load(
            save_path, load_if_exists, "lineardecoder", verbose,
            cv_mode=self.readout_cv_mode, n_splits=self.readout_n_splits,
            buffer_samples=self.readout_buffer_samples,
        )
        if loaded is not None:
            self.stats["lineardecoder"] = loaded
            return loaded

        task.collect_data(exists_ok=True)
        y_df = task._get_numeric_target_df()

        if varnames is None:
            varnames = task._default_readout_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if not varnames:
            raise ValueError("No valid readout variables found.")

        groups = task._get_groups()
        trial_idx = np.asarray(groups) if groups is not None else None
        unique_groups, lengths = one_ff_style_utils.build_group_lengths(
            groups if groups is not None else []
        )
        n_smooth = task._neural_ncols_after_pca() if getattr(task, "use_spike_history", False) else None

        out: Dict = {}
        decodertype = "lineardecoder"

        for v in varnames:
            if v not in y_df.columns:
                continue
            x_true = y_df[v].to_numpy(dtype=float)
            n = len(x_true)

            if self.fit_kernelwidth:
                groups_for_folds = np.asarray(groups) if self.readout_cv_mode == "group_kfold" else None
                splits = cv_decoding._build_folds(
                    n, n_splits=self.readout_n_splits, groups=groups_for_folds,
                    cv_splitter=self.readout_cv_mode, buffer_samples=self.readout_buffer_samples,
                )
                pred = np.full(n, np.nan, dtype=float)
                widths_per_fold, wts_per_fold, fold_tuning_info = [], [], []

                for train_idx, test_idx in splits:
                    train_groups = np.asarray(groups)[train_idx] if groups is not None else None
                    _, lengths_train = one_ff_style_utils.build_group_lengths(
                        train_groups if train_groups is not None else []
                    )
                    best = one_ff_style_utils.tune_linear_decoder_cv(
                        y_neural=task._get_neural_matrix_for_decoding(
                            neural_smooth_width=0, trial_idx=trial_idx)[train_idx],
                        x_true=x_true[train_idx], lengths=lengths_train,
                        candidate_widths=self.candidate_widths,
                        n_splits=self.readout_n_splits, cv_mode=self.readout_cv_mode,
                        buffer_samples=self.readout_buffer_samples, neural_cols_to_smooth=n_smooth,
                    )
                    best_width = int(best["width"])
                    widths_per_fold.append(best_width)
                    wts_per_fold.append(np.asarray(best.get("wts")))
                    fold_tuning_info.append(best.get("width_scores", {}))

                    X_tr = task._get_neural_matrix_for_decoding(
                        neural_smooth_width=best_width, trial_idx=trial_idx)[train_idx]
                    X_te = task._get_neural_matrix_for_decoding(
                        neural_smooth_width=best_width, trial_idx=trial_idx)[test_idx]
                    try:
                        coef, *_ = np.linalg.lstsq(X_tr, x_true[train_idx], rcond=None)
                    except Exception:
                        coef = np.asarray(best.get("wts")).reshape(-1)
                    pred[test_idx] = X_te.dot(np.asarray(coef).reshape(-1))

                rep_width = int(np.round(float(np.nanmedian(widths_per_fold)))) if widths_per_fold else self.fixed_width
                entry = {
                    "bestfiltwidth": rep_width, "candidate_widths": self.candidate_widths,
                    "wts": wts_per_fold, "corr": one_ff_style_utils.safe_corr(x_true, pred),
                    "fold_tuning_info": fold_tuning_info,
                }
            else:
                neural = task._get_neural_matrix_for_decoding(
                    neural_smooth_width=self.fixed_width, trial_idx=trial_idx)
                best = one_ff_style_utils.fit_linear_decoder_cv(
                    y_neural=neural, x_true=x_true, lengths=lengths,
                    width=self.fixed_width, n_splits=self.readout_n_splits,
                    cv_mode=self.readout_cv_mode, buffer_samples=self.readout_buffer_samples,
                    neural_cols_to_smooth=n_smooth,
                )
                pred = best["pred"]
                entry = {
                    "bestfiltwidth": self.fixed_width, "candidate_widths": [self.fixed_width],
                    "wts": best["wts"], "corr": one_ff_style_utils.safe_corr(x_true, pred),
                }

            if save_predictions:
                entry["true"] = x_true
                entry["pred"] = pred
                entry["trials"] = {
                    "true": one_ff_style_utils.split_by_lengths(x_true, lengths),
                    "pred": one_ff_style_utils.split_by_lengths(pred, lengths),
                }
            out[v] = entry

        out["_cv_config"] = {
            "cv_mode": self.readout_cv_mode,
            "n_splits": self.readout_n_splits,
            "buffer_samples": self.readout_buffer_samples,
        }
        self.stats[decodertype] = out
        _save(save_path, out, decodertype, verbose,
              cv_mode=self.readout_cv_mode, n_splits=self.readout_n_splits,
              buffer_samples=self.readout_buffer_samples)
        return out

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_canoncorr_coefficients(self, **kwargs):
        block = self.stats.get("canoncorr")
        if block is None:
            raise ValueError("Run compute_canoncorr first.")
        plot_one_ff_decoding.plot_canoncorr_coefficients(block, **kwargs)

    def plot_decoder_parity(self, *, varnames=None, **kwargs):
        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("Run regress_popreadout first.")
        plot_one_ff_decoding.plot_decoder_parity(block, varnames=varnames, **kwargs)

    def plot_decoder_correlation_bars(self, *, varnames=None, **kwargs):
        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("Run regress_popreadout first.")
        plot_one_ff_decoding.plot_decoder_correlation_bars(block, varnames=varnames, **kwargs)

    def plot_all_decoding_results(self, *, parity_varnames=None, bar_varnames=None,
                                  trial_indices=None, n_trials=6):
        plot_decoding_utils.plot_all_decoding_results(
            canoncorr_block=self.stats.get("canoncorr"),
            readout_block=self.stats.get("lineardecoder"),
            parity_varnames=parity_varnames, bar_varnames=bar_varnames,
            trial_indices=trial_indices, n_trials=n_trials,
        )

    def extract_corr_df(self, block_key: str = "lineardecoder") -> pd.DataFrame:
        block = self.stats.get(block_key)
        if block is None:
            raise ValueError(f'No block "{block_key}" in stats.')
        rows = [
            {"variable": k, "corr": float(v.get("corr", np.nan))}
            for k, v in block.items()
            if isinstance(v, dict) and "corr" in v
        ]
        if not rows:
            raise ValueError("No decoder correlations found.")
        return (
            pd.DataFrame(rows)
            .sort_values("corr", key=lambda x: np.nan_to_num(x, nan=-np.inf), ascending=False)
            .reset_index(drop=True)
        )


# ===========================================================================
# CVDecodingModel  (model-spec CV decoding + ANOVA + LM)
# ===========================================================================

class CVDecodingModel(BaseDecodingModel):
    """
    Cross-validated model-spec decoding (CatBoost, ridge, logistic, etc.).
    Also owns ANOVA, LM, and feature-score extraction — all modeling operations
    that read task.feats_to_decode and task.binned_spikes.

    Parameters
    ----------
    cv_mode        : str   'group_kfold', 'blocked_time_buffered', …
    model_specs    : dict  mapping name → spec (defaults to decoding_model_specs.MODEL_SPECS)
    """

    def __init__(
        self,
        cv_mode: str = "group_kfold",
        model_specs: Optional[Dict] = None,
        verbose: bool = True,
    ):
        self.cv_mode = cv_mode
        self.model_specs = model_specs  # None → resolved at run time
        self.verbose = verbose

        # State set after run()
        self.all_results: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.stats: Dict = {}

    def _name(self) -> str:
        return type(self).__name__

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(
        self,
        task,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
        fit_kernelwidth: bool = True,
        candidate_widths: Sequence[int] = tuple(range(1, 6, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 5,
        cv_mode: Optional[str] = None,
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity: int = 1,
        use_detrend_inside_cv: bool = False,
    ) -> pd.DataFrame:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        self.all_results = self.run_cv_decoding(
            task,
            n_splits=n_splits, save_dir=save_dir,
            design_matrices_exists_ok=design_matrices_exists_ok,
            model_specs=model_specs, shuffle_mode=shuffle_mode,
            fit_kernelwidth=fit_kernelwidth, candidate_widths=candidate_widths,
            fixed_width=fixed_width, inner_cv_splits=inner_cv_splits,
            cv_mode=cv_mode, load_if_exists=load_if_exists,
            load_existing_only=load_existing_only,
            cv_decoding_verbosity=cv_decoding_verbosity,
            use_detrend_inside_cv=use_detrend_inside_cv,
        )
        if cv_mode is not None and "cv_mode" in self.all_results.columns:
            self.results_df = self.all_results[self.all_results["cv_mode"] == cv_mode]
        else:
            self.results_df = self.all_results.copy()
        return self.all_results

    def run_cv_decoding(
        self,
        task,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
        fit_kernelwidth: bool = False,
        candidate_widths: Sequence[int] = tuple(range(1, 6, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 5,
        cv_mode: Optional[str] = None,
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity: int = 1,
        use_detrend_inside_cv: bool = False,
    ) -> pd.DataFrame:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        specs = model_specs or self.model_specs or decoding_model_specs.MODEL_SPECS

        if not load_existing_only:
            task.collect_data(exists_ok=design_matrices_exists_ok)

        if getattr(task, "use_spike_history", False):
            task._ensure_spike_history_features_materialized()

        save_dir = self._resolve_decoding_save_dir(task, save_dir)
        if shuffle_mode and shuffle_mode != "none":
            save_dir = Path(save_dir) / f"shuffle_{shuffle_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)

        self.cv_decoding_out_of_models = {}
        all_results = []
        for model_name, spec in specs.items():
            print(f"[CVDecodingModel] Running model: {model_name}")
            results_df = self._run_single_model_cv(
                task,
                model_name=model_name, spec=spec, save_dir=save_dir,
                n_splits=n_splits, shuffle_mode=shuffle_mode,
                fit_kernelwidth=fit_kernelwidth, candidate_widths=candidate_widths,
                fixed_width=fixed_width, inner_cv_splits=inner_cv_splits,
                cv_mode=cv_mode, load_if_exists=load_if_exists,
                load_existing_only=load_existing_only,
                verbosity=cv_decoding_verbosity,
                use_detrend_inside_cv=use_detrend_inside_cv,
            )
            results_df["model_name"] = model_name
            all_results.append(results_df)

        self.all_results = pd.concat(all_results, ignore_index=True)
        return self.all_results

    # ------------------------------------------------------------------
    # Per-model dispatcher
    # ------------------------------------------------------------------
    

    # print('hehehehehehehhe)

    def _run_single_model_cv(
        self, task, *,
        model_name, spec, save_dir, n_splits, shuffle_mode,
        fit_kernelwidth, candidate_widths, fixed_width, inner_cv_splits,
        cv_mode, load_if_exists, load_existing_only, verbosity,
        use_detrend_inside_cv: bool = False,
    ):
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        detrend_covariates = self._get_detrend_covariates(task) if use_detrend_inside_cv else None
        groups = task._get_groups()
        detrend_per_block = spec.get("detrend_per_block", True)
        block_cv_modes = ("blocked_time_buffered", "blocked_time", "group_kfold")
        use_per_block = (
            detrend_covariates is not None and groups is not None
            and cv_mode in block_cv_modes and detrend_per_block
        )
        if detrend_covariates is None:
            detrend_suffix = ""
        elif use_per_block:
            detrend_suffix = "_detrend_perblock"
        else:
            detrend_suffix = "_detrend"

        model_save_path = self._cv_decoding_model_pickle_path(
            save_dir, model_name, shuffle_mode,
            fit_kernelwidth=fit_kernelwidth, n_splits=n_splits,
            inner_cv_splits=inner_cv_splits, fixed_width=fixed_width,
            cv_mode=cv_mode, detrend_suffix=detrend_suffix, include_shuffle_in_stem=True,
        )
        print("model_save_path: ", model_save_path)

        loaded = self._maybe_load_cv_decoding_result(
            save_dir=save_dir, model_name=model_name, shuffle_mode=shuffle_mode,
            fit_kernelwidth=fit_kernelwidth, n_splits=n_splits,
            inner_cv_splits=inner_cv_splits, fixed_width=fixed_width,
            cv_mode=cv_mode, detrend_suffix=detrend_suffix,
            load_if_exists=load_if_exists, verbose=True,
        )
        if loaded is not None:
            print(f"[{self._name()}] {model_name}: loaded cached results")
            self.stats[f"cv_decoding_{model_name}"] = loaded
            self.cv_decoding_out_of_models[model_name] = loaded
            results_df = loaded["results_df"]
            if "model_name" not in results_df.columns:
                results_df["model_name"] = model_name
            if "shuffle_mode" not in results_df.columns:
                results_df["shuffle_mode"] = "none" if shuffle_mode is None else shuffle_mode
            return results_df

        if load_existing_only:
            print("Failed to load existing results, and load_existing_only=True, skipping.")
            return pd.DataFrame()

        config = cv_decoding.DecodingRunConfig(
            regression_model_class=spec.get("regression_model_class"),
            regression_model_kwargs=spec.get("regression_model_kwargs", {}),
            classification_model_class=spec.get("classification_model_class"),
            classification_model_kwargs=spec.get("classification_model_kwargs", {}),
            use_early_stopping=False,
            detrend_per_block=spec.get("detrend_per_block", True),
        )

        if fit_kernelwidth:
            results_df = self.run_nested_kernelwidth_cv(
                task, model_name=model_name, config=config,
                n_splits=n_splits, candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits, cv_mode=cv_mode,
                model_save_path=model_save_path, verbosity=verbosity,
                shuffle_mode=shuffle_mode, detrend_covariates=detrend_covariates,
            )
        else:
            result_save_dir = Path(save_dir) / f"fixed_kernelwidth_{fixed_width}"
            results_df = self._run_fixed_width_cv(
                task, model_name=model_name, config=config, save_dir=result_save_dir,
                n_splits=n_splits, fixed_width=fixed_width, cv_mode=cv_mode,
                shuffle_mode=shuffle_mode, load_existing_only=load_existing_only,
                model_save_path=model_save_path, verbosity=verbosity,
                detrend_covariates=detrend_covariates,
            )

        results_df = self._ensure_cv_decoding_columns(
            results_df, model_name=model_name, config=config, n_splits=n_splits,
            shuffle_mode=shuffle_mode, cv_mode=cv_mode,
            buffer_samples=None, context_label="pooled",
        )
        return results_df

    def _run_fixed_width_cv(
        self, task, *,
        model_name, config, save_dir, n_splits, fixed_width,
        cv_mode, shuffle_mode, load_existing_only, model_save_path, verbosity,
        detrend_covariates=None,
    ):
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        groups = task._get_groups()
        trial_idx = np.asarray(groups) if groups is not None else None
        X = task._get_neural_matrix_for_decoding(
            neural_smooth_width=int(fixed_width) if int(fixed_width) > 0 else 0,
            trial_idx=trial_idx,
        )
        results_df = cv_decoding.run_cv_decoding(
            X=X, y_df=task.get_target_df(), behav_features=None,
            groups=task._get_groups(), n_splits=n_splits, config=config,
            context_label="pooled", save_dir=save_dir, model_name=model_name,
            shuffle_mode=shuffle_mode, load_existing_only=load_existing_only,
            verbosity=verbosity, detrend_covariates=detrend_covariates, cv_mode=cv_mode,
        )
        results_df["kernelwidth"] = fixed_width
        results_df["cv_mode"] = cv_mode

        self.cv_decoding_out_of_models[model_name] = {
            "results_df": results_df,
            "_cv_config": {
                "fit_kernelwidth": False, "fixed_width": fixed_width,
                "n_splits": n_splits, "cv_mode": cv_mode,
                "use_detrend_inside_cv": detrend_covariates is not None,
            },
        }
        self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]

        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            with model_save_path.open("wb") as f:
                pickle.dump(self.stats[f"cv_decoding_{model_name}"], f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity:
                print(f"Fixed-width CV results saved: {model_save_path}")
        return results_df

    def run_nested_kernelwidth_cv(
        self, task, *,
        model_name, model_save_path, config, n_splits, candidate_widths,
        inner_cv_splits, cv_mode, verbosity, shuffle_mode, detrend_covariates=None,
    ):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import _build_folds

        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        y_df = task.get_target_df()
        groups = task._get_groups()
        X_full = task._get_neural_matrix_for_decoding(neural_smooth_width=0)
        n_samples = len(X_full)

        outer_splits = _build_folds(
            n_samples, n_splits=n_splits, groups=groups,
            cv_splitter=cv_mode, random_state=0,
        )

        outer_results, fold_tuning_info, width_results_all_folds = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            groups_train = groups[train_idx] if groups is not None else None
            groups_test = groups[test_idx] if groups is not None else None

            def _slice_cov(cov, idx):
                if cov is None:
                    return None
                return cov.iloc[idx] if isinstance(cov, pd.DataFrame) else np.asarray(cov)[idx]

            best_width, width_scores, width_results = self._run_inner_kernelwidth_search(
                task, train_idx=train_idx, y_df=y_df.iloc[train_idx],
                groups=groups_train, config=config, candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits, verbosity=verbosity,
                shuffle_mode=shuffle_mode, cv_mode=cv_mode,
                detrend_covariates=_slice_cov(detrend_covariates, train_idx),
            )

            fold_result = self._evaluate_outer_fold(
                task,
                X_train=X_full[train_idx], y_train=y_df.iloc[train_idx],
                X_test=X_full[test_idx], y_test=y_df.iloc[test_idx],
                best_width_by_feature=best_width, config=config,
                shuffle_mode=shuffle_mode, groups_train=groups_train, groups_test=groups_test,
                detrend_cov_train=_slice_cov(detrend_covariates, train_idx),
                detrend_cov_test=_slice_cov(detrend_covariates, test_idx),
                fold_train_idx=train_idx, fold_test_idx=test_idx,
            )

            fold_result["fold"] = fold_idx
            if isinstance(best_width, dict) and "behav_feature" in fold_result.columns:
                fold_result["kernelwidth"] = fold_result["behav_feature"].map(best_width).astype(float)
            else:
                fold_result["kernelwidth"] = best_width

            outer_results.append(fold_result)
            fold_tuning_info.append(width_scores)
            for w, df_w in width_results.items():
                if df_w is None or df_w.empty:
                    continue
                df_tmp = df_w.copy()
                df_tmp["fold"] = fold_idx
                df_tmp["kernelwidth"] = int(w)
                width_results_all_folds.append(df_tmp)

        results_df = pd.concat(outer_results, ignore_index=True)
        results_df["model_name"] = model_name
        results_df["cv_mode"] = cv_mode
        results_df_all_filt = (
            pd.concat(width_results_all_folds, ignore_index=True)
            if width_results_all_folds else pd.DataFrame()
        )

        self.cv_decoding_out_of_models[model_name] = {
            "results_df": results_df, "results_df_all_filt": results_df_all_filt,
            "fold_tuning_info": fold_tuning_info,
            "_cv_config": {
                "fit_kernelwidth": True, "candidate_widths": list(candidate_widths),
                "n_splits": n_splits, "inner_cv_splits": inner_cv_splits, "cv_mode": cv_mode,
                "use_detrend_inside_cv": detrend_covariates is not None,
            },
        }
        self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]

        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            with model_save_path.open("wb") as f:
                pickle.dump(self.stats[f"cv_decoding_{model_name}"], f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity:
                print(f"Nested CV results saved: {model_save_path}")
        return results_df

    def _run_inner_kernelwidth_search(
        self, task, *,
        train_idx, y_df, groups, config, candidate_widths,
        inner_cv_splits, verbosity, shuffle_mode, cv_mode=None, detrend_covariates=None,
    ):
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        train_idx = np.asarray(train_idx, dtype=int)
        groups_full = task._get_groups()
        trial_full = np.asarray(groups_full) if groups_full is not None else None

        best_width, best_score, width_scores, width_results = {}, {}, {}, {}

        for width in candidate_widths:
            X_smooth = task._get_neural_matrix_for_decoding(
                neural_smooth_width=int(width), trial_idx=trial_full,
            )[train_idx]

            inner_df = cv_decoding.run_cv_decoding(
                cv_mode=cv_mode, X=X_smooth, y_df=y_df, behav_features=None,
                groups=groups, n_splits=inner_cv_splits, config=config,
                context_label="pooled", save_dir=None, model_name=None,
                shuffle_mode=shuffle_mode, verbosity=verbosity,
                detrend_covariates=detrend_covariates,
            )

            try:
                inner_df_copy = inner_df.copy() if inner_df is not None else pd.DataFrame()
            except Exception:
                inner_df_copy = pd.DataFrame()
            if not inner_df_copy.empty:
                inner_df_copy["kernelwidth"] = int(width)
            width_results[int(width)] = inner_df_copy

            if inner_df is None or inner_df.empty or "mode" not in inner_df.columns:
                width_scores[int(width)] = float("nan")
                continue

            n = len(inner_df)
            r_cv = inner_df["r_cv"].to_numpy(dtype=float) if "r_cv" in inner_df.columns else np.full(n, np.nan)
            auc_mean = inner_df["auc_mean"].to_numpy(dtype=float) if "auc_mean" in inner_df.columns else np.full(n, np.nan)
            metric_scores = np.where(inner_df["mode"].to_numpy() == "regression", r_cv, auc_mean)
            width_scores[int(width)] = float(np.nanmean(metric_scores))

            for _, row in inner_df.iterrows():
                feat = row.get("behav_feature")
                if feat is None:
                    continue
                score = row.get("r_cv" if row.get("mode") == "regression" else "auc_mean", np.nan)
                if np.isfinite(score) and score > best_score.get(feat, -np.inf):
                    best_score[feat] = float(score)
                    best_width[feat] = int(width)

        return best_width, width_scores, width_results

    def _evaluate_outer_fold(
        self, task, *,
        X_train, y_train, X_test, y_test,
        best_width_by_feature, config, shuffle_mode="none",
        groups_train=None, groups_test=None,
        detrend_cov_train=None, detrend_cov_test=None,
        fold_train_idx=None, fold_test_idx=None,
    ):
        return self._train_test_single_fold(
            task, X_train, y_train, X_test, y_test, config,
            shuffle_mode=shuffle_mode, groups_train=groups_train, groups_test=groups_test,
            kernelwidth_by_feature=best_width_by_feature,
            detrend_cov_train=detrend_cov_train, detrend_cov_test=detrend_cov_test,
            fold_train_idx=fold_train_idx, fold_test_idx=fold_test_idx,
        )

    def _train_test_single_fold(
        self, task,
        X_train, y_train_df, X_test, y_test_df, config, *,
        shuffle_mode="none", groups_train=None, groups_test=None,
        kernelwidth_by_feature=None, detrend_cov_train=None, detrend_cov_test=None,
        fold_train_idx=None, fold_test_idx=None,
    ):
        from catboost import CatBoostRegressor, CatBoostError
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.utils.multiclass import type_of_target
        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
            infer_decoding_type, _shuffle_y_for_fold, _maybe_detrend_neural,
        )

        rng = np.random.default_rng(0)
        buffer_samples = getattr(config, "buffer_samples", 20)
        results = []

        for col in y_train_df.columns:
            y_train = y_train_df[col].to_numpy()
            y_test = y_test_df[col].to_numpy()
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)

            if kernelwidth_by_feature is not None and groups_train is not None and groups_test is not None:
                width = kernelwidth_by_feature.get(col)
                if width is not None:
                    groups_full = task._get_groups()
                    trial_full = np.asarray(groups_full) if groups_full is not None else None
                    if task.pca_n_components is not None and fold_train_idx is not None:
                        X_train_sm = task._get_neural_matrix_for_decoding(
                            neural_smooth_width=int(width), trial_idx=trial_full,
                        )[fold_train_idx]
                        X_test_sm = task._get_neural_matrix_for_decoding(
                            neural_smooth_width=int(width), trial_idx=trial_full,
                        )[fold_test_idx]
                    else:
                        X_train_sm = task._smooth_raw_neural(X_train, int(width), trial_idx=groups_train)
                        X_test_sm = task._smooth_raw_neural(X_test, int(width), trial_idx=groups_test)
                else:
                    X_train_sm = X_train
                    X_test_sm = X_test
            else:
                X_train_sm = X_train
                X_test_sm = X_test

            X_tr = X_train_sm[train_valid]
            y_tr = y_train[train_valid].copy()
            g_tr = groups_train[train_valid] if groups_train is not None else None
            X_te = X_test_sm[test_valid]
            y_te = y_test[test_valid]

            if len(y_tr) == 0 or len(y_te) == 0:
                continue

            y_tr = _shuffle_y_for_fold(y_tr, g_tr, shuffle_mode, rng, buffer_samples)
            if np.unique(y_tr).size <= 1:
                continue

            mode = infer_decoding_type(y_tr)

            def _slice_cov(cov, mask):
                if cov is None:
                    return None
                return cov.iloc[mask].values if isinstance(cov, pd.DataFrame) else np.asarray(cov)[mask]

            X_tr, X_te = _maybe_detrend_neural(
                X_tr, X_te,
                _slice_cov(detrend_cov_train, train_valid),
                _slice_cov(detrend_cov_test, test_valid),
                degree=getattr(config, "detrend_degree", 1),
            )

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            if mode == "regression":
                model_class = config.regression_model_class or CatBoostRegressor
                model_kwargs = config.regression_model_kwargs or {"verbose": False}
                mdl = model_class(**model_kwargs)
                try:
                    mdl.fit(X_tr, y_tr)
                except CatBoostError as e:
                    if "All train targets are equal" in str(e) or "All targets are equal" in str(e):
                        continue
                    raise
                y_pred = mdl.predict(X_te)
                row = {
                    "behav_feature": col, "mode": "regression",
                    "r2_cv": r2_score(y_te, y_pred),
                    "rmse_cv": float(np.sqrt(mean_squared_error(y_te, y_pred))),
                    "r_cv": float(np.corrcoef(y_te, y_pred)[0, 1]),
                    "n_samples": int(len(y_te)),
                }
            else:
                try:
                    target_type = type_of_target(y_tr)
                except Exception:
                    target_type = "unknown"
                if target_type == "continuous":
                    continue
                le = LabelEncoder()
                y_tr_enc = le.fit_transform(y_tr)
                seen = set(le.classes_)
                te_mask = np.array([y in seen for y in y_te])
                if not np.any(te_mask):
                    continue
                X_te_f = X_te[te_mask]
                y_te_f = y_te[te_mask]
                y_te_enc = le.transform(y_te_f)
                if np.unique(y_te_enc).size < 2:
                    continue
                model_class = config.classification_model_class or LogisticRegression
                mdl = model_class(**(config.classification_model_kwargs or {}))
                try:
                    mdl.fit(X_tr, y_tr_enc)
                except CatBoostError as e:
                    if "All train targets are equal" in str(e) or "All targets are equal" in str(e):
                        continue
                    raise
                y_pred_proba = mdl.predict_proba(X_te_f)
                try:
                    auc = float(roc_auc_score(y_te_enc, y_pred_proba[:, 1]))
                except Exception:
                    auc = np.nan
                row = {
                    "behav_feature": col, "mode": "classification",
                    "auc_mean": auc, "auc_std": np.nan,
                    "pr_mean": np.nan, "pr_std": np.nan,
                    "n_samples": int(len(y_te_f)),
                }
            results.append(row)

        return pd.DataFrame(results) if results else pd.DataFrame()

    # ------------------------------------------------------------------
    # Path helpers (static — no task or model state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _cv_shuffle_stem_for_pickle(shuffle_mode, *, include_shuffle_in_stem):
        if not include_shuffle_in_stem:
            return ""
        sm = "none" if shuffle_mode is None else str(shuffle_mode)
        if sm == "none":
            return ""
        safe = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in sm)
        return f"_shuffle_{safe}"

    @staticmethod
    def _cv_decoding_model_pickle_path(
        save_dir, model_name, shuffle_mode, *,
        fit_kernelwidth, n_splits, inner_cv_splits, fixed_width,
        cv_mode, detrend_suffix, include_shuffle_in_stem=True,
    ) -> Path:
        save_dir = Path(save_dir)
        shuff = CVDecodingModel._cv_shuffle_stem_for_pickle(
            shuffle_mode, include_shuffle_in_stem=include_shuffle_in_stem
        )
        if fit_kernelwidth:
            return (
                save_dir / "cv_decoding" / "cvnested"
                / f"{model_name}_n{n_splits}_inner{inner_cv_splits}_{cv_mode}{detrend_suffix}{shuff}.pkl"
            )
        return (
            save_dir / "cv_decoding" / "fixed_width"
            / f"{model_name}_width{fixed_width}_n{n_splits}_{cv_mode}{detrend_suffix}{shuff}.pkl"
        )

    @staticmethod
    def _maybe_load_cv_decoding_result(
        *, save_dir, model_name, shuffle_mode, fit_kernelwidth, n_splits,
        inner_cv_splits, fixed_width, cv_mode, detrend_suffix, load_if_exists, verbose,
    ):
        if not load_if_exists:
            return None
        primary = CVDecodingModel._cv_decoding_model_pickle_path(
            save_dir, model_name, shuffle_mode,
            fit_kernelwidth=fit_kernelwidth, n_splits=n_splits,
            inner_cv_splits=inner_cv_splits, fixed_width=fixed_width,
            cv_mode=cv_mode, detrend_suffix=detrend_suffix, include_shuffle_in_stem=True,
        )
        candidates = [primary]
        sm = "none" if shuffle_mode is None else str(shuffle_mode)
        if sm != "none":
            legacy = CVDecodingModel._cv_decoding_model_pickle_path(
                save_dir, model_name, shuffle_mode,
                fit_kernelwidth=fit_kernelwidth, n_splits=n_splits,
                inner_cv_splits=inner_cv_splits, fixed_width=fixed_width,
                cv_mode=cv_mode, detrend_suffix=detrend_suffix, include_shuffle_in_stem=False,
            )
            if legacy.resolve() != primary.resolve():
                candidates.append(legacy)
        for p in candidates:
            if not p.exists():
                continue
            with p.open("rb") as f:
                obj = pickle.load(f)
            if verbose:
                print(f"cv decoding results loaded: {p}")
            return obj
        print("Model save path does not exist (tried): ", candidates)
        return None

    @staticmethod
    def _ensure_cv_decoding_columns(
        df, *, model_name, config, n_splits, shuffle_mode, cv_mode, buffer_samples, context_label,
    ):
        if df is None or df.empty:
            df = pd.DataFrame()
        df = df.copy()
        df["model_name"] = model_name
        df["n_splits"] = n_splits
        df["shuffle_mode"] = "none" if shuffle_mode is None else shuffle_mode
        df["cv_mode"] = cv_mode
        df["buffer_samples"] = buffer_samples
        df["context"] = context_label
        if "kernelwidth" not in df.columns:
            df["kernelwidth"] = np.nan
        if "n_samples" not in df.columns:
            df["n_samples"] = np.nan
        for col in ("r2_cv", "rmse_cv", "r_cv"):
            if col not in df.columns:
                df[col] = np.nan
        for col in ("auc_mean", "auc_std", "pr_mean", "pr_std",
                    "n_total_folds", "n_valid_folds", "n_skipped_folds"):
            if col not in df.columns:
                df[col] = np.nan
        return df

    @staticmethod
    def _get_detrend_covariates(task):
        target_df = task.get_target_df()
        if "time" in target_df.columns:
            return target_df[["time"]].copy()
        return getattr(task, "detrend_covariates", None)

    # ------------------------------------------------------------------
    # Feature analysis
    # ------------------------------------------------------------------

    def find_true_vs_pred_cv_for_feature(
        self, task, feature: str, *,
        config=None, model_name: str = "ridge_strong",
        model_specs: Optional[Mapping] = None,
        n_splits: int = 5, cv_mode: Optional[str] = None,
        design_matrices_exists_ok: bool = True,
    ):
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        specs = model_specs or decoding_model_specs.MODEL_SPECS
        if config is None:
            spec = specs[model_name]
            config = cv_decoding.DecodingRunConfig(
                regression_model_class=spec.get("regression_model_class"),
                regression_model_kwargs=spec.get("regression_model_kwargs", {}),
                classification_model_class=spec.get("classification_model_class"),
                classification_model_kwargs=spec.get("classification_model_kwargs", {}),
                use_early_stopping=False,
                detrend_per_block=spec.get("detrend_per_block", True),
            )

        task.collect_data(exists_ok=design_matrices_exists_ok)
        X = np.asarray(task._get_neural_matrix_for_decoding(neural_smooth_width=0), dtype=float)
        y_df = task.get_target_df()
        if feature not in y_df.columns:
            raise KeyError(f"feature={feature!r} not in target columns.")
        y = y_df[feature].to_numpy().ravel()
        groups = task._get_groups()
        if groups is None:
            groups = np.zeros(len(X), dtype=int)

        X, groups, _ = cv_decoding._prepare_inputs(X, groups, 0)
        X_ok, y_ok, g_ok, _ = cv_decoding.filter_valid_rows(X, y, groups, None)
        if g_ok is None:
            g_ok = np.zeros(len(y_ok), dtype=int)

        return plot_decoding_predictions.get_cv_predictions(
            X_ok, y_ok, g_ok,
            plot_decoding_predictions.ConfigRegressionEstimator, config,
            n_splits=n_splits, cv_mode=cv_mode, model_name=model_name,
        )

    def extract_regression_feature_scores_df(
        self, task,
        row_selector_fn=None,
        regression_metric: str = "r_cv",
        classification_metric: str = "auc_mean",
    ) -> pd.DataFrame:
        if self.results_df is None:
            raise RuntimeError("Run run_cv_decoding first.")
        if row_selector_fn is None:
            row_selector_fn = show_decoding_results._select_rows
        results_df = plot_decoding_utils.add_score_column(
            self.results_df, regression_metric, classification_metric
        )
        selected = row_selector_fn(results_df)
        rel = selected[selected["mode"] == "regression"].copy()
        if rel.empty:
            raise ValueError("No regression rows found.")
        df = (
            rel[["behav_feature", "score"]]
            .rename(columns={"behav_feature": "variable"})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        self.regression_feature_scores_df = df.copy()
        return df

    def plot_fold_tuning_info(self, task):
        if self.results_df is None:
            raise RuntimeError("Run run_cv_decoding first.")
        for model in self.results_df["model_name"].unique():
            fig, ax, summary = plot_decoding_utils.plot_fold_tuning_info(
                self.stats[f"cv_decoding_{model}"]["fold_tuning_info"],
                shade="sem", show_folds=True,
            )
            print(summary["best_width"], summary["best_mean_cv"])
            plt.show()

    # ------------------------------------------------------------------
    # ANOVA / LM — moved to BaseEncodingModel
    # ------------------------------------------------------------------
    # ANOVA and LM are encoding questions (behavioral variable → neural
    # response) and now live in BaseEncodingModel / EncodingRunner.
    # For backward compatibility, thin wrappers are provided in
    # DecodingRunner that call linear_model_utils directly.
    #
    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _results_cache_path(self, task, label: str) -> Optional[Path]:
        try:
            return self._resolve_decoding_save_dir(task, None) / f"{label}_results.pkl"
        except Exception:
            return None

    def _load_results_cache(self, task, cache_path: Path, label: str):
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
            print(f"[{self._name()}] {label.upper()}: loaded from cache ({cache_path})")
            return results
        except Exception as e:
            print(f"[{self._name()}] {label.upper()}: cache load failed ({e}) — re-running.")
            return None

    def _save_results_cache(self, task, results, cache_path: Path, label: str):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[{self._name()}] {label.upper()}: saved to cache ({cache_path})")
        except Exception as e:
            print(f"[{self._name()}] {label.upper()}: cache save failed ({e})")

