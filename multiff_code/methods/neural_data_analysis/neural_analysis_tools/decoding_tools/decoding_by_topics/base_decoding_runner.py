"""Base class for decoding runners with shared one-FF-style (CCA + linear readout) and CV decoding logic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decode_stops_utils, plot_decoding_utils
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import one_ff_style_decoding_runner


class BaseDecodingRunner(one_ff_style_decoding_runner.OneFFStyleDecodingRunner):
    """
    Base class for decoding runners that support one-FF-style population decoding
    (CCA + linear readout). Subclasses must implement:
    - _get_target_df(): feature dataframe to decode
    - _get_groups(): group labels for CV (e.g. event_id)
    - _get_neural_matrix(): neural data array
    - _default_canoncorr_varnames(): default vars for CCA
    - _default_readout_varnames(): default vars for linear readout
    """

    def __init__(self, bin_width: float = 0.04):
        self.bin_width = bin_width
        self.stats: Dict = {}

    # ------------------------------------------------------------------
    # Abstract / override points (subclasses must implement)
    # ------------------------------------------------------------------
    def _get_target_df(self) -> pd.DataFrame:
        """Feature dataframe to decode (e.g. stop_feats_to_decode, behav_df, vis_feats_to_decode)."""
        raise NotImplementedError

    def _get_groups(self):
        """Group labels for CV (e.g. event_id or trial_ids)."""
        raise NotImplementedError

    def _get_neural_matrix(self) -> np.ndarray:
        """Neural data matrix (samples x neurons)."""
        raise NotImplementedError

    def _default_canoncorr_varnames(self) -> List[str]:
        """Default variable names for canoncorr."""
        raise NotImplementedError

    def _default_readout_varnames(self) -> List[str]:
        """Default variable names for linear readout."""
        raise NotImplementedError

    def _get_save_dir(self) -> str:
        """Base save directory for outputs."""
        raise NotImplementedError

    def _runner_name(self) -> str:
        """Name for log messages. Override in subclass."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # CV decoding (run / run_cv_decoding)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
        fit_kernelwidth: bool = False,
        candidate_widths: Sequence[int] = tuple(range(1, 21, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 3,
        cv_mode: str = "blocked_time_buffered",  # can be 'blocked_time_buffered', 'blocked_time', 'group_kfold'
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity=1,
        
    ) -> pd.DataFrame:
        
        """Run CV decoding. Delegates to run_cv_decoding."""
        self.all_results = self.run_cv_decoding(
            n_splits=n_splits,
            save_dir=save_dir,
            design_matrices_exists_ok=design_matrices_exists_ok,
            model_specs=model_specs,
            shuffle_mode=shuffle_mode,
            fit_kernelwidth=fit_kernelwidth,
            candidate_widths=candidate_widths,
            fixed_width=fixed_width,
            inner_cv_splits=inner_cv_splits,
            cv_mode=cv_mode,
            load_if_exists=load_if_exists,
            load_existing_only=load_existing_only,
            cv_decoding_verbosity=cv_decoding_verbosity,
        )
        
        if cv_mode is not None and 'cv_mode' in self.all_results.columns:
            self.results_df = self.all_results[self.all_results['cv_mode'] == cv_mode] 
        else:
            self.results_df = self.all_results.copy()
            
        return self.results_df


    def _ensure_cv_decoding_columns(
        self,
        df: pd.DataFrame,
        *,
        model_name: str,
        config: cv_decoding.DecodingRunConfig,
        n_splits: int,
        shuffle_mode: str,
        cv_mode: Optional[str],
        buffer_samples: Optional[int],
        context_label: Optional[str],
    ) -> pd.DataFrame:
        """Normalize a results DataFrame to include the columns emitted by cv_decoding.run_cv_decoding.

        Adds missing columns with NaN or sensible defaults so concatenation across branches is safe.
        """
        if df is None or df.empty:
            # create empty DataFrame with minimal columns
            df = pd.DataFrame()

        df = df.copy()

        # Basic identifying columns
        df["model_name"] = model_name
        df["n_splits"] = n_splits
        df["shuffle_mode"] = shuffle_mode
        df["cv_mode"] = config.cv_mode if cv_mode is None else cv_mode
        df["buffer_samples"] = buffer_samples
        df["context"] = context_label

        # Filter width used for smoothing (may be set earlier for nested CV folds)
        if "kernelwidth" not in df.columns:
            df["kernelwidth"] = np.nan

        # Ensure n_samples exists (if not present, try to infer or set NaN)
        if "n_samples" not in df.columns:
            df["n_samples"] = np.nan

        # Regression metric columns
        for col in ("r2_cv", "rmse_cv", "r_cv"):
            if col not in df.columns:
                df[col] = np.nan

        # Classification metric columns
        for col in ("auc_mean", "auc_std", "pr_mean", "pr_std", "n_total_folds", "n_valid_folds", "n_skipped_folds"):
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _train_test_single_fold(
        self,
        X_train: np.ndarray,
        y_train_df: pd.DataFrame,
        X_test: np.ndarray,
        y_test_df: pd.DataFrame,
        config: cv_decoding.DecodingRunConfig,
    ) -> Dict:
        """Train models on training data and evaluate on test data.
        
        Returns a dict with one row per target variable, containing predictions and metrics.
        """
        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
            infer_decoding_type,
        )
        
        results = []
        
        for col in y_train_df.columns:
            y_train = y_train_df[col].to_numpy()
            y_test = y_test_df[col].to_numpy()
            
            # Filter invalid rows (NaN/Inf)
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)

            X_tr = X_train[train_valid]
            y_tr = y_train[train_valid]
            X_te = X_test[test_valid]
            y_te = y_test[test_valid]

            if len(y_tr) == 0 or len(y_te) == 0:
                continue

            # Skip if no variance in training labels
            if np.unique(y_tr).size <= 1:
                continue

            # Infer decoding mode from the finite training labels
            mode = infer_decoding_type(y_tr)
            
            # Scale features
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            if mode == 'regression':
                model_class = config.regression_model_class or CatBoostRegressor
                model_kwargs = config.regression_model_kwargs or dict(verbose=False)
                model = model_class(**model_kwargs)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                
                row = {
                    "behav_feature": col,
                    "mode": "regression",
                    "r2_cv": r2_score(y_te, y_pred),
                    "rmse_cv": np.sqrt(mean_squared_error(y_te, y_pred)),
                    "r_cv": np.corrcoef(y_te, y_pred)[0, 1],
                    "n_total_folds": 1,
                    "n_valid_folds": 1,
                    "n_skipped_folds": 0,
                    "n_samples": int(len(y_te)),
                }
            else:  # classification
                model_class = config.classification_model_class or LogisticRegression
                model_kwargs = config.classification_model_kwargs or {}
                model = model_class(**model_kwargs)
                
                # Ensure at least 2 classes in training and test
                if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
                    continue
                
                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_te)
                
                try:
                    auc = float(roc_auc_score(y_te, y_pred_proba[:, 1]))
                except Exception:
                    auc = np.nan

                # Align keys with cv_decoding.run_cv_decoding classification output
                row = {
                    "behav_feature": col,
                    "mode": "classification",
                    "auc_mean": auc,
                    "auc_std": np.nan,
                    "pr_mean": np.nan,
                    "pr_std": np.nan,
                    "n_total_folds": 1,
                    "n_valid_folds": 1,
                    "n_skipped_folds": 0,
                    "n_samples": int(len(y_te)),
                }
            
            results.append(row)
        
        # Return as DataFrame for consistency with run_cv_decoding output
        return pd.DataFrame(results) if results else pd.DataFrame()

    # ------------------------------------------------------------------
    # Caching utilities (optional override)
    # ------------------------------------------------------------------
    def _get_design_matrix_paths(self) -> Mapping[str, Path]:
        """Paths for cached design matrices. Override in subclass."""
        raise NotImplementedError

    def _get_design_matrix_data(self) -> Mapping[str, Any]:
        """Data to save: key -> value. Override in subclass."""
        raise NotImplementedError

    def _get_design_matrix_key_to_attr(self) -> Mapping[str, str]:
        """Map path key -> attribute name for loading. Override in subclass."""
        raise NotImplementedError

    def _save_design_matrices(self) -> None:
        """Save design matrices to disk. Uses _get_design_matrix_paths and _get_design_matrix_data."""
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_design_matrix_paths()
        data = self._get_design_matrix_data()
        name = self._runner_name()
        for key, path in paths.items():
            if key not in data:
                continue
            try:
                with open(path, "wb") as f:
                    pickle.dump(data[key], f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{name}] Saved {key} -> {path}")
            except Exception as e:
                print(f"[{name}] WARNING save {key}: {type(e).__name__}: {e}")

    def _load_design_matrices(self) -> bool:
        """Load design matrices from disk. Returns True if successful."""
        paths = self._get_design_matrix_paths()
        key_to_attr = self._get_design_matrix_key_to_attr()
        if not all(paths[k].exists() for k in key_to_attr if k in paths):
            return False
        name = self._runner_name()
        try:
            for key, attr in key_to_attr.items():
                if key not in paths:
                    continue
                with open(paths[key], "rb") as f:
                    setattr(self, attr, pickle.load(f))
            print(f"[{name}] Loaded cached design matrices")
            return True
        except Exception as e:
            print(f"[{name}] WARNING load matrices: {type(e).__name__}: {e}")
            return False


    # ============================================================
    # Cross-Validated Decoding (Refactored, Copy-Paste Ready)
    # ============================================================

    def run_cv_decoding(
        self,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
        fit_kernelwidth: bool = False,
        candidate_widths: Sequence[int] = tuple(range(1, 21, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 3,
        cv_mode: str = "blocked_time_buffered",
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity=1,
    ) -> pd.DataFrame:
        """
        Run cross-validated model-spec decoding with optional nested CV
        for kernel width tuning.
        """

        # --------------------------------------------------------
        # Prepare environment
        # --------------------------------------------------------
        self.model_specs = (
            model_specs
            if model_specs is not None
            else pn_decoding_model_specs.MODEL_SPECS
        )

        self._collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            save_dir = self._get_save_dir()

        if shuffle_mode != "none":
            save_dir = Path(save_dir) / f"shuffle_{shuffle_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = []
        self.cv_decoding_out_of_models = {}

        # --------------------------------------------------------
        # Loop over models
        # --------------------------------------------------------
        for model_name, spec in self.model_specs.items():

            results_df = self._run_single_model_cv(
                model_name=model_name,
                spec=spec,
                save_dir=save_dir,
                n_splits=n_splits,
                shuffle_mode=shuffle_mode,
                fit_kernelwidth=fit_kernelwidth,
                candidate_widths=candidate_widths,
                fixed_width=fixed_width,
                inner_cv_splits=inner_cv_splits,
                cv_mode=cv_mode,
                load_if_exists=load_if_exists,
                load_existing_only=load_existing_only,
                verbosity=cv_decoding_verbosity,
            )

            self.all_results.append(results_df)

        self.all_results = pd.concat(self.all_results, ignore_index=True)
        return self.all_results


        # ============================================================
        # Per-Model Dispatcher
        # ============================================================

    def _run_single_model_cv(
        self,
        *,
        model_name,
        spec,
        save_dir,
        n_splits,
        shuffle_mode,
        fit_kernelwidth,
        candidate_widths,
        fixed_width,
        inner_cv_splits,
        cv_mode,
        load_if_exists,
        load_existing_only,
        verbosity,
    ):

        model_save_path = Path(save_dir) / f"cv_decoding_{model_name}.pkl"

        # --------------------------------------------------------
        # Try loading cached result
        # --------------------------------------------------------
        loaded = self._maybe_load_cv_decoding_result(
            str(model_save_path),
            load_if_exists,
            f"cv_decoding_{model_name}",
            verbose=True,
            fit_kernelwidth=fit_kernelwidth,
            n_splits=n_splits,
            inner_cv_splits=inner_cv_splits if fit_kernelwidth else None,
            cv_mode=cv_mode,
        )

        if loaded is not None:
            print(f"[{self._runner_name()}] {model_name}: loaded cached results")

            self.stats[f"cv_decoding_{model_name}"] = loaded
            self.cv_decoding_out_of_models[model_name] = loaded

            results_df = loaded["results_df"]
            results_df["model_name"] = model_name
            return results_df

        # --------------------------------------------------------
        # Build decoding config
        # --------------------------------------------------------
        config = cv_decoding.DecodingRunConfig(
            regression_model_class=spec.get("regression_model_class"),
            regression_model_kwargs=spec.get("regression_model_kwargs", {}),
            classification_model_class=spec.get("classification_model_class"),
            classification_model_kwargs=spec.get("classification_model_kwargs", {}),
            use_early_stopping=False,
            cv_mode=cv_mode,
        )

        # --------------------------------------------------------
        # Branch: nested CV or fixed width
        # --------------------------------------------------------
        if fit_kernelwidth:
            results_df = self._run_nested_kernelwidth_cv(
                model_name=model_name,
                config=config,
                model_save_path=model_save_path,
                n_splits=n_splits,
                candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits,
                fixed_width=fixed_width,
                cv_mode=cv_mode,
                load_existing_only=load_existing_only,
                verbosity=verbosity,
            )
        else:
            results_df = self._run_fixed_width_cv(
                model_name=model_name,
                config=config,
                model_save_path=model_save_path,
                n_splits=n_splits,
                fixed_width=fixed_width,
                cv_mode=cv_mode,
                shuffle_mode=shuffle_mode,
                load_existing_only=load_existing_only,
                verbosity=verbosity,
            )

        return results_df


    # ============================================================
    # Fixed Width CV
    # ============================================================

    def _run_fixed_width_cv(
        self,
        *,
        model_name,
        config,
        model_save_path,
        n_splits,
        fixed_width,
        cv_mode,
        shuffle_mode,
        load_existing_only,
        verbosity,
    ):

        X = self._get_neural_matrix()
        if fixed_width > 0:
            X = decode_stops_utils.smooth_signal(X, int(fixed_width))

        results_df = cv_decoding.run_cv_decoding(
            X=X,
            y_df=self._get_target_df(),
            behav_features=None,
            groups=self._get_groups(),
            n_splits=n_splits,
            config=config,
            context_label="pooled",
            save_dir=None,
            model_name=model_name,
            shuffle_mode=shuffle_mode,
            load_existing_only=load_existing_only,
            verbosity=verbosity,
        )

        results_df["kernelwidth"] = fixed_width

        self.cv_decoding_out_of_models[model_name] = {
            "results_df": results_df,
            "_cv_config": {
                "fit_kernelwidth": False,
                "fixed_width": fixed_width,
                "n_splits": n_splits,
                "cv_mode": cv_mode,
            },
        }

        self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]

        return results_df


    # ============================================================
    # Nested Kernelwidth CV
    # ============================================================

    def _run_nested_kernelwidth_cv(
        self,
        *,
        model_name,
        config,
        model_save_path,
        n_splits,
        candidate_widths,
        inner_cv_splits,
        fixed_width,
        cv_mode,
        load_existing_only,
        verbosity,
    ):

        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
            _build_folds,
        )

        X = self._get_neural_matrix()
        y_df = self._get_target_df()
        groups = self._get_groups()

        outer_splits = _build_folds(
            len(X),
            n_splits=n_splits,
            groups=groups,
            cv_splitter=cv_mode,
            random_state=0,
        )

        outer_results = []
        fold_tuning_info = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):

            best_width, width_scores = self._run_inner_kernelwidth_search(
                X=X[train_idx],
                y_df=y_df.iloc[train_idx],
                groups=groups[train_idx] if groups is not None else None,
                config=config,
                candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits,
                load_existing_only=load_existing_only,
                verbosity=verbosity,
            )

            fold_result = self._evaluate_outer_fold(
                X_train=X[train_idx],
                y_train=y_df.iloc[train_idx],
                X_test=X[test_idx],
                y_test=y_df.iloc[test_idx],
                best_width=best_width,
                config=config,
            )

            fold_result["fold"] = fold_idx
            fold_result["kernelwidth"] = best_width

            outer_results.append(fold_result)
            fold_tuning_info.append(width_scores)

        results_df = pd.concat(outer_results, ignore_index=True)

        self.cv_decoding_out_of_models[model_name] = {
            "results_df": results_df,
            "fold_tuning_info": fold_tuning_info,
            "_cv_config": {
                "fit_kernelwidth": True,
                "candidate_widths": list(candidate_widths),
                "n_splits": n_splits,
                "inner_cv_splits": inner_cv_splits,
                "cv_mode": cv_mode,
            },
        }

        self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]

        return results_df


    # ============================================================
    # Inner Kernelwidth Search
    # ============================================================
    def _run_inner_kernelwidth_search(
        self,
        *,
        X,
        y_df,
        groups,
        config,
        candidate_widths,
        inner_cv_splits,
        load_existing_only,
        verbosity,
    ):

        best_width = None
        best_score = -np.inf
        width_scores = {}

        for width in candidate_widths:

            X_smooth = decode_stops_utils.smooth_signal(X, int(width))

            inner_df = cv_decoding.run_cv_decoding(
                X=X_smooth,
                y_df=y_df,
                behav_features=None,
                groups=groups,
                n_splits=inner_cv_splits,
                config=config,
                context_label="pooled",
                save_dir=None,
                model_name=None,
                shuffle_mode="none",
                load_existing_only=load_existing_only,
                verbosity=verbosity,
            )

            metric_scores = np.where(
                inner_df["mode"] == "regression",
                inner_df["r_cv"],
                inner_df["auc_mean"],
            )

            mean_score = float(np.nanmean(metric_scores))
            width_scores[int(width)] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_width = int(width)

        return best_width, width_scores


    # ============================================================
    # Outer Fold Evaluation
    # ============================================================

    def _evaluate_outer_fold(
        self,
        *,
        X_train,
        y_train,
        X_test,
        y_test,
        best_width,
        config,
    ):

        X_train = decode_stops_utils.smooth_signal(X_train, int(best_width))
        X_test = decode_stops_utils.smooth_signal(X_test, int(best_width))

        return self._train_test_single_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
        )