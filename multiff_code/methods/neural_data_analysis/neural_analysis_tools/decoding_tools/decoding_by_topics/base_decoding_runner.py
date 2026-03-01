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
    decode_stops_utils,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs,
)


class BaseOneFFStyleDecodingRunner:
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

    def _get_neural_matrix(self, use_spike_history: Optional[bool] = None) -> np.ndarray:
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
        cv_mode: str = "group_kfold",
        load_if_exists: bool = True,
    ) -> pd.DataFrame:
        """Run CV decoding. Delegates to run_cv_decoding."""
        return self.run_cv_decoding(
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
        )

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
        cv_mode: str = "group_kfold",
        load_if_exists: bool = True,
    ) -> pd.DataFrame:
        """Run cross-validated model-spec decoding with nested CV for hyperparameter tuning.
        
        If fit_kernelwidth=True, uses nested cross-validation:
        - Outer CV loop: evaluates final model performance (n_splits folds)
        - Inner CV loop: tunes kernel width on training data only (inner_cv_splits folds)
        
        This avoids data leakage when selecting hyperparameters.
        
        Parameters
        ----------
        fit_kernelwidth : bool, default False
            If True, tune smoothing kernel width using nested CV. If False, use fixed_width.
        candidate_widths : Sequence[int]
            Candidate smoothing widths to try if fit_kernelwidth=True.
        fixed_width : int
            Fixed smoothing width to use if fit_kernelwidth=False.
        inner_cv_splits : int
            Number of CV splits for inner loop (hyperparameter tuning). Only used if fit_kernelwidth=True.
        cv_mode : str, default 'group_kfold'
            Cross-validation mode: 'group_kfold', 'blocked_time_buffered', 'blocked_time', or default (shuffled KFold).
        load_if_exists : bool, default True
            If True, load cached results if they exist with matching parameters.
        """
        self.model_specs = (
            model_specs if model_specs is not None else pn_decoding_model_specs.MODEL_SPECS
        )
        self._collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            save_dir = self._get_save_dir()
        if shuffle_mode != "none":
            save_dir = Path(save_dir) / f"shuffle_{shuffle_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = []
        self.cv_decoding_out_of_models = {}

        for model_name, spec in self.model_specs.items():
            # Try to load pre-computed results
            model_save_path = Path(save_dir) / f"cv_decoding_{model_name}.pkl"
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
                self.stats[f"cv_decoding_{model_name}"] = loaded
                self.cv_decoding_out_of_models[model_name] = loaded  # Store by model name
                self.results_df = loaded["results_df"]
                self.results_df["model_name"] = model_name
                self.all_results.append(self.results_df)
                print(f"[{self._runner_name()}] {model_name}: loaded cached results")
                continue
            
            config = cv_decoding.DecodingRunConfig(
                regression_model_class=spec.get("regression_model_class", None),
                regression_model_kwargs=spec.get("regression_model_kwargs", {}),
                classification_model_class=spec.get("classification_model_class", None),
                classification_model_kwargs=spec.get("classification_model_kwargs", {}),
                use_early_stopping=False,
                cv_mode=cv_mode,
            )
            print(f"[{self._runner_name()}] model_name: {model_name}")
            
            if fit_kernelwidth:
                # Nested CV: tune kernel width in inner loop, test on outer loop
                print(f"[{self._runner_name()}] {model_name}: using nested CV with inner_cv_splits={inner_cv_splits}")
                
                X = self._get_neural_matrix()
                y_df = self._get_target_df()
                groups = self._get_groups()
                
                # Get train/test splits for outer CV
                from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
                    _build_folds,
                )
                outer_splits = _build_folds(
                    len(X),
                    n_splits=n_splits,
                    groups=groups,
                    cv_splitter=cv_mode,
                    random_state=0,
                )
                
                outer_results = []
                all_best_widths = []
                all_best_inner_corrs = []
                all_width_scores = []
                all_width_inner_results = {int(w): [] for w in candidate_widths}
                
                for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
                    print(f"[{self._runner_name()}] {model_name}: outer fold {fold_idx + 1}/{n_splits}")
                    
                    # Inner CV: tune kernel width on training fold
                    X_train, y_train = X[train_idx], y_df.iloc[train_idx]
                    groups_train = groups[train_idx] if groups is not None else None
                    
                    best_width = fixed_width
                    best_inner_corr = -np.inf
                    width_scores = {}  # Track score for each width
                    width_inner_results = {}  # Store full inner CV results for each width
                    
                    # Run inner CV for all candidate widths to find best and save all results
                    for width in candidate_widths:
                        X_train_smooth = decode_stops_utils.smooth_signal(X_train, int(width))
                        inner_results_df = cv_decoding.run_cv_decoding(
                            X=X_train_smooth,
                            y_df=y_train,
                            behav_features=None,
                            groups=groups_train,
                            n_splits=inner_cv_splits,
                            config=config,
                            context_label="pooled",
                            save_dir=None,
                            model_name=model_name,
                            shuffle_mode="none",
                        )
                        
                        if "mode" not in inner_results_df.columns:
                            raise ValueError("Expected 'mode' column in inner CV results for tuning, but not found.")

                        if "r_cv" not in inner_results_df.columns or "auc_mean" not in inner_results_df.columns:
                            raise ValueError("Expected 'r_cv' and 'auc_mean' columns in inner CV results for tuning, but not found.")

                        metric_scores = np.where(
                            inner_results_df["mode"] == "regression",
                            inner_results_df["r_cv"],
                            inner_results_df["auc_mean"],
                        )
                        mean_score = float(np.nanmean(metric_scores))
                        width_scores[int(width)] = mean_score
                        width_inner_results[int(width)] = inner_results_df  # Store full results as DataFrame
                        
                        if mean_score > best_inner_corr:
                            best_inner_corr = mean_score
                            best_width = width
                    
                    print(f"[{self._runner_name()}] {model_name}: fold {fold_idx + 1}: best tuned width = {best_width}")
                    
                    # Collect results across all folds
                    all_best_widths.append(int(best_width))
                    all_best_inner_corrs.append(float(best_inner_corr))
                    all_width_scores.append(width_scores)
                    for width in candidate_widths:
                        all_width_inner_results[int(width)].append(width_inner_results[int(width)])
                    
                    # Only evaluate the best width on the outer test fold
                    X_test, y_test = X[test_idx], y_df.iloc[test_idx]
                    X_train_smooth = decode_stops_utils.smooth_signal(X_train, int(best_width))
                    X_test_smooth = decode_stops_utils.smooth_signal(X_test, int(best_width))
                    
                    # Train and evaluate single model on train/test split with best width
                    fold_results = self._train_test_single_fold(
                        X_train_smooth,
                        y_train,
                        X_test_smooth,
                        y_test,
                        config,
                    )
                    fold_results["fold"] = fold_idx
                    fold_results["kernelwidth"] = best_width
                    outer_results.append(fold_results)
                
                # Concatenate DataFrames per width across all folds
                fold_tuning_info = {
                    "best_width": all_best_widths,
                    "best_inner_corr": all_best_inner_corrs,
                    "width_scores": all_width_scores,
                    "width_inner_results": pd.concat([
                        pd.concat(dfs, ignore_index=True).assign(width=int(w))
                        for w, dfs in all_width_inner_results.items()
                    ], ignore_index=True),
                }
                
                self.results_df = pd.concat(outer_results, ignore_index=True)
                
                # Store detailed results in stats
                self.cv_decoding_out_of_models[model_name] = {
                    "results_df": self.results_df,
                    "fold_tuning_info": fold_tuning_info,
                    "_cv_config": {
                        "fit_kernelwidth": True,
                        "candidate_widths": list(candidate_widths),
                        "n_splits": n_splits,
                        "inner_cv_splits": inner_cv_splits,
                        "fixed_width": fixed_width,
                        "cv_mode": cv_mode,
                    },
                }
                self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]
                
                # Save to disk
                self._save_cv_decoding_result(
                    str(model_save_path),
                    self.cv_decoding_out_of_models[model_name],
                    f"cv_decoding_{model_name}",
                    verbose=True,
                    fit_kernelwidth=True,
                    n_splits=n_splits,
                    inner_cv_splits=inner_cv_splits,
                    cv_mode=cv_mode,
                )
            else:
                # Use fixed smoothing width (standard CV, single loop)
                neural_X = decode_stops_utils.smooth_signal(
                    self._get_neural_matrix(), int(fixed_width)
                ) if fixed_width > 0 else self._get_neural_matrix()
                
                self.results_df = cv_decoding.run_cv_decoding(
                    X=neural_X,
                    y_df=self._get_target_df(),
                    behav_features=None,
                    groups=self._get_groups(),
                    n_splits=n_splits,
                    config=config,
                    context_label="pooled",
                    save_dir=save_dir,
                    model_name=model_name,
                    shuffle_mode=shuffle_mode,
                )
                
                # Store results config in stats
                self.cv_decoding_out_of_models[model_name] = {
                    "results_df": self.results_df,
                    "_cv_config": {
                        "fit_kernelwidth": False,
                        "fixed_width": fixed_width,
                        "n_splits": n_splits,
                        "cv_mode": cv_mode,
                    },
                }
                self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]
                
                # Save to disk
                self._save_cv_decoding_result(
                    str(model_save_path),
                    self.cv_decoding_out_of_models[model_name],
                    f"cv_decoding_{model_name}",
                    verbose=True,
                    fit_kernelwidth=False,
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                )
            
            self.results_df["model_name"] = model_name
            self.all_results.append(self.results_df)

        return pd.concat(self.all_results, ignore_index=True)

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
            
            # Skip if no variance
            if np.unique(y_train).size <= 1:
                continue
            
            mode = infer_decoding_type(y_train)
            
            # Filter invalid rows (NaN/Inf)
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)
            
            X_tr = X_train[train_valid]
            y_tr = y_train[train_valid]
            X_te = X_test[test_valid]
            y_te = y_test[test_valid]
            
            if len(y_tr) == 0 or len(y_te) == 0:
                continue
            
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
                    "r2": r2_score(y_te, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
                    "corr": np.corrcoef(y_te, y_pred)[0, 1],
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
                    auc = roc_auc_score(y_te, y_pred_proba[:, 1])
                except Exception:
                    auc = np.nan
                
                row = {
                    "behav_feature": col,
                    "mode": "classification",
                    "auc": auc,
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

    # ------------------------------------------------------------------
    # Shared one-FF-style logic
    # ------------------------------------------------------------------
    def _get_numeric_target_df(self) -> pd.DataFrame:
        """Filter target df to numeric columns (exclude const)."""
        y_df = self._get_target_df().copy()
        keep_cols = []
        for c in y_df.columns:
            if c == "const":
                continue
            if pd.api.types.is_numeric_dtype(y_df[c]) or pd.api.types.is_bool_dtype(y_df[c]):
                keep_cols.append(c)
        return y_df[keep_cols].astype(float)

    def run_one_ff_style(
        self,
        *,
        design_matrices_exists_ok: bool = True,
        save_dir=None,
        canoncorr_varnames: Optional[Sequence[str]] = None,
        readout_varnames: Optional[Sequence[str]] = None,
        readout_n_splits: int = 5,
        readout_cv_mode: str = "blocked_time_buffered",
        readout_buffer_samples: int = 20,
        fit_kernelwidth: bool = True,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """Run one-FF-style decoding: CCA + linear readout. Requires _collect_data implemented."""
        self._collect_data(exists_ok=design_matrices_exists_ok)
        if save_dir is None:
            save_dir = Path(self._get_save_dir()) / "one_ff_style"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        canoncorr = self.compute_canoncorr(
            varnames=canoncorr_varnames,
            save_path=str(save_dir / "canoncorr.pkl"),
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        readout = self.regress_popreadout(
            varnames=readout_varnames,
            n_splits=readout_n_splits,
            cv_mode=readout_cv_mode,
            buffer_samples=readout_buffer_samples,
            fit_kernelwidth=fit_kernelwidth,
            save_path=str(save_dir / "lineardecoder.pkl"),
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        print('Finished running one-FF-style decoding')
        return {"canoncorr": canoncorr, "readout": readout, "stats": self.stats}

    @staticmethod
    def _path_with_cv_decoding_params(
        save_path: Optional[str],
        *,
        fit_kernelwidth: bool,
        n_splits: int,
        inner_cv_splits: Optional[int] = None,
        cv_mode: str = "group_kfold",
    ) -> Optional[Path]:
        """Derive save path that encodes CV decoding config so different configs don't collide."""
        if save_path is None:
            return None
        p = Path(save_path)
        stem, suf = p.stem, p.suffix
        if fit_kernelwidth:
            return p.parent / f"{stem}_cvnested_n{n_splits}_inner{inner_cv_splits}_{cv_mode}{suf}"
        else:
            return p.parent / f"{stem}_cvfixed_n{n_splits}_{cv_mode}{suf}"

    @staticmethod
    def _maybe_load_cv_decoding_result(
        save_path: Optional[str],
        load_if_exists: bool,
        label: str,
        verbose: bool,
        *,
        fit_kernelwidth: bool,
        n_splits: int,
        inner_cv_splits: Optional[int] = None,
        cv_mode: str = "group_kfold",
    ):
        """Load CV decoding result if it exists and load_if_exists=True."""
        if (not load_if_exists) or save_path is None:
            return None
        p = BaseOneFFStyleDecodingRunner._path_with_cv_decoding_params(
            save_path,
            fit_kernelwidth=fit_kernelwidth,
            n_splits=n_splits,
            inner_cv_splits=inner_cv_splits,
            cv_mode=cv_mode,
        )
        if not p.exists():
            return None
        with p.open("rb") as f:
            obj = pickle.load(f)
        if verbose:
            print(f"[{label}] loaded: {p}")
        return obj

    @staticmethod
    def _save_cv_decoding_result(
        save_path: Optional[str],
        result,
        label: str,
        verbose: bool,
        *,
        fit_kernelwidth: bool,
        n_splits: int,
        inner_cv_splits: Optional[int] = None,
        cv_mode: str = "group_kfold",
    ):
        """Save CV decoding result to disk with parameter-encoded filename."""
        if save_path is None:
            return
        p = BaseOneFFStyleDecodingRunner._path_with_cv_decoding_params(
            save_path,
            fit_kernelwidth=fit_kernelwidth,
            n_splits=n_splits,
            inner_cv_splits=inner_cv_splits,
            cv_mode=cv_mode,
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"[{label}] saved: {p}")

    @staticmethod
    def _path_with_cv_params(
        save_path: Optional[str],
        *,
        cv_mode: str,
        n_splits: int,
        buffer_samples: int,
    ) -> Optional[Path]:
        """Derive save path that encodes cv config so different configs don't collide."""
        if save_path is None:
            return None
        p = Path(save_path)
        stem, suf = p.stem, p.suffix
        return p.parent / f"{stem}_cv{cv_mode}_n{n_splits}_buf{buffer_samples}{suf}"

    @staticmethod
    def _maybe_load_result(
        save_path: Optional[str],
        load_if_exists: bool,
        label: str,
        verbose: bool,
        *,
        cv_mode: Optional[str] = None,
        n_splits: Optional[int] = None,
        buffer_samples: Optional[int] = None,
    ):
        if (not load_if_exists) or save_path is None:
            return None
        if cv_mode is not None and n_splits is not None and buffer_samples is not None:
            p = BaseOneFFStyleDecodingRunner._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            p = Path(save_path)
        if not p.exists():
            return None
        with p.open("rb") as f:
            obj = pickle.load(f)
        if verbose:
            print(f"[{label}] loaded: {p}")
        return obj

    @staticmethod
    def _save_result(
        save_path: Optional[str],
        result,
        label: str,
        verbose: bool,
        *,
        cv_mode: Optional[str] = None,
        n_splits: Optional[int] = None,
        buffer_samples: Optional[int] = None,
    ):
        if save_path is None:
            return
        if cv_mode is not None and n_splits is not None and buffer_samples is not None:
            p = BaseOneFFStyleDecodingRunner._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"[{label}] saved: {p}")

    def _target_df_error_msg(self) -> str:
        """Override in subclass for clearer error messages."""
        return "target features dataframe"

    def compute_canoncorr(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        use_spike_history: Optional[bool] = None,
        filtwidth: int = 5,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        loaded = self._maybe_load_result(save_path, load_if_exists, "canoncorr", verbose)
        if loaded is not None:
            self.stats["canoncorr"] = loaded
            return loaded

        self._collect_data(exists_ok=True)
        y_df = self._get_numeric_target_df()
        if varnames is None:
            varnames = self._default_canoncorr_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if len(varnames) == 0:
            raise ValueError(
                f"No valid canoncorr variables found in {self._target_df_error_msg()}."
            )

        if verbose:
            print(f"[canoncorr] vars: {varnames}")
        x_task = y_df[varnames].to_numpy(dtype=float)
        y_neural = self._get_neural_matrix(use_spike_history=use_spike_history)

        out = decode_stops_utils.compute_canoncorr_block(
            x_task=x_task,
            y_neural=y_neural,
            dt=float(self.bin_width),
            filtwidth=int(filtwidth),
        )
        out["vars"] = list(varnames)
        self.stats["canoncorr"] = out
        self._save_result(save_path, out, "canoncorr", verbose)
        return out

    def regress_popreadout(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        use_spike_history: Optional[bool] = None,
        fit_kernelwidth: bool = True,
        candidate_widths: Sequence[int] = tuple(range(1, 21, 1)),
        fixed_width: int = 25,
        n_splits: int = 5,
        cv_mode: str = "blocked_time_buffered",
        buffer_samples: int = 20,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        save_predictions: bool = False,
        verbose: bool = True,
    ) -> Dict:
        decodertype = "lineardecoder"
        loaded = self._maybe_load_result(
            save_path,
            load_if_exists,
            decodertype,
            verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        if loaded is not None:
            self.stats[decodertype] = loaded
            if verbose:
                print(f"[{decodertype}] loaded from cache")
            return loaded

        if verbose:
            print(f"[{decodertype}] computing (no cached result found)")
        self._collect_data(exists_ok=True)
        y_df = self._get_numeric_target_df()
        if varnames is None:
            varnames = self._default_readout_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if len(varnames) == 0:
            raise ValueError(
                f"No valid readout variables found in {self._target_df_error_msg()}."
            )

        neural = self._get_neural_matrix(use_spike_history=use_spike_history)
        groups = self._get_groups()
        groups = np.asarray(groups)
        _, lengths = decode_stops_utils.build_group_lengths(groups)

        out: Dict = {}
        for v in varnames:
            if verbose:
                print(f"[{decodertype}] fitting {v} (CV n_splits={n_splits}, cv_mode={cv_mode})")
            x_true = y_df[v].to_numpy(dtype=float)
            x_true[np.isnan(x_true)] = 0.0

            if fit_kernelwidth:
                best = decode_stops_utils.tune_linear_decoder_cv(
                    y_neural=neural,
                    x_true=x_true,
                    lengths=lengths,
                    candidate_widths=candidate_widths,
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                )
            else:
                best = decode_stops_utils.fit_linear_decoder_cv(
                    y_neural=neural,
                    x_true=x_true,
                    lengths=lengths,
                    width=int(fixed_width),
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                )

            pred = best["pred"]
            widths_used = (
                list(candidate_widths) if fit_kernelwidth else [int(fixed_width)]
            )
            entry = {
                "bestfiltwidth": int(best["width"]),
                "candidate_widths": widths_used,
                "wts": best["wts"],
                "corr": decode_stops_utils.safe_corr(x_true, pred),
            }
            if save_predictions:
                entry["true"] = x_true
                entry["pred"] = pred
                entry["trials"] = {
                    "true": decode_stops_utils.split_by_lengths(x_true, lengths),
                    "pred": decode_stops_utils.split_by_lengths(pred, lengths),
                }
            out[v] = entry

        out["_cv_config"] = {
            "cv_mode": cv_mode,
            "n_splits": n_splits,
            "buffer_samples": buffer_samples,
        }
        self.stats[decodertype] = out
        self._save_result(
            save_path,
            out,
            decodertype,
            verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        return out
