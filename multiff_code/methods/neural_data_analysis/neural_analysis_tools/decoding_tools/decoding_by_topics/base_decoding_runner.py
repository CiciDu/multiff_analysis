"""Base class for decoding runners with shared one-FF-style (CCA + linear readout) and CV decoding logic."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_design_utils
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    plot_decoding_utils
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
    plot_decoding_predictions,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_model_specs, smooth_neural_data
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import one_ff_style_decoding_runner

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import show_decoding_results
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import process_encode_design
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import linear_model_utils



class BaseDecodingRunner(one_ff_style_decoding_runner.OneFFStyleDecodingRunner):
    """
    Base class for decoding runners that support one-FF-style population decoding
    (CCA + linear readout). Subclasses must implement:
    - get_target_df(): feature dataframe to decode
    - _get_groups(): group labels for CV (e.g. event_id)
    - _get_neural_matrix(): neural data array
    - _default_canoncorr_varnames(): default vars for CCA
    - _default_readout_varnames(): default vars for linear readout
    """

    def __init__(
        self,
        bin_width: float = 0.04,
        smoothing_width: int | None = None,
        detrend_spikes: bool = False,
        drop_bad_neurons: bool = True,
        cv_mode: str = "group_kfold",
        pca_n_components: int | None = None,
    ):
        
        self.bin_width = bin_width
        self.stats: Dict = {}
        self.smooth_spikes = True if smoothing_width is not None else False
        self.smoothing_width = smoothing_width
        self.detrend_spikes = detrend_spikes
        self.drop_bad_neurons = drop_bad_neurons

        # Cross-validation strategy (e.g. 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold')
        self.cv_mode = cv_mode

        # Optional PCA dimensionality reduction applied to neural data before decoding.
        # PCA is fit once on the full neural matrix and cached in self.pca_.
        self.pca_n_components = pca_n_components
        self.pca_ = None
        
        # Filled during setup
        self.feats_to_decode = None
        self.trial_ids = None
        self.binned_spikes = None
        self.detrend_covariates = None  # DataFrame with time, trial_index, etc.


    # ------------------------------------------------------------------
    # Abstract / override points (subclasses must implement)
    # ------------------------------------------------------------------
    def get_target_df(self) -> pd.DataFrame:
        """Feature dataframe to decode (e.g. feats_to_decode, behav_df, feats_to_decode)."""
        raise NotImplementedError

    def _get_groups(self):
        """Group labels for CV (e.g. event_id or trial_ids)."""
        raise NotImplementedError

    def get_detrend_covariates(self):
        """Return DataFrame with time, trial_index, etc. for detrending.
        Override to add more columns (e.g. from rebinned_y_var)."""
        if getattr(self, 'target_df', None) is None:
            self.get_target_df()
        self.detrend_covariates = self.target_df[['time']].copy()
        
        # ## disbable detrending for now
        # self.detrend_covariates = None
        return self.detrend_covariates

    def _get_neural_matrix(self) -> np.ndarray:
        """Neural data matrix (samples x neurons)."""
        raise NotImplementedError

    def _get_neural_matrix_for_decoding(self) -> np.ndarray:
        """Return neural matrix, optionally reduced via PCA.

        PCA is fit once on the full matrix and cached in ``self.pca_``.  When
        ``pca_n_components`` is *None* the raw matrix is returned unchanged.
        """
        X = self._get_neural_matrix()
        if self.pca_n_components is None:
            return X

        from sklearn.decomposition import PCA

        if self.pca_ is None:
            self.pca_ = PCA(n_components=self.pca_n_components, random_state=0)
            X = self.pca_.fit_transform(X)
            explained = self.pca_.explained_variance_ratio_.sum()
            print(
                f"[{self._runner_name()}] PCA fitted: {self.pca_n_components} components, "
                f"cumulative explained variance = {explained:.3f}"
            )
        else:
            X = self.pca_.transform(X)
        return X

    def _default_canoncorr_varnames(self) -> List[str]:
        """Default variable names for canoncorr."""
        raise NotImplementedError

    def _default_readout_varnames(self) -> List[str]:
        """Default variable names for linear readout."""
        raise NotImplementedError

    def _get_save_dir(self) -> str:
        """Base save directory for outputs."""
        raise NotImplementedError

    def _get_save_dir_common(self, decoder_outputs_dirname: str) -> str:
        """
        Common save dir builder for decoding runners.

        Subfolder rule:
        - if smoothing is enabled: smooth/width_{smoothing_width}
        - elif detrending is enabled: detrended
        - else: raw
        """
        if getattr(self, "smooth_spikes", False):
            sub = os.path.join("smooth", f"width_{int(self.smoothing_width)}")
        elif getattr(self, "detrend_spikes", False):
            sub = "detrended"
        else:
            sub = "raw"

        if getattr(self, "pca_n_components", None) is not None:
            sub = sub + f"_pca{self.pca_n_components}"

        self.save_dir = os.path.join(
            self.pn.planning_and_neural_folder_path,
            "decoding_outputs",
            decoder_outputs_dirname,
            sub,
        )

        return self.save_dir

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
        fit_kernelwidth: bool = True,
        candidate_widths: Sequence[int] = tuple(range(1, 6, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 5,
        cv_mode: Optional[str] = None,  # 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold'
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity=1,
        use_detrend_inside_cv: bool = False,
    ) -> pd.DataFrame:
    
        if save_dir is None:
            save_dir = Path(self._get_save_dir())

        # Resolve cv_mode: use argument if provided, otherwise default to instance setting
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        
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
            load_if_exists=load_if_exists,
            load_existing_only=load_existing_only,
            cv_decoding_verbosity=cv_decoding_verbosity,
            use_detrend_inside_cv=use_detrend_inside_cv,
            cv_mode=cv_mode,
        )
        
        if cv_mode is not None and 'cv_mode' in self.all_results.columns:
            self.results_df = self.all_results[self.all_results['cv_mode'] == cv_mode]
        else:
            self.results_df = self.all_results.copy()

    def find_true_vs_pred_cv_for_feature(
        self,
        feature: str,
        *,
        config: Optional[cv_decoding.DecodingRunConfig] = None,
        model_name: str = "ridge_strong",
        model_specs: Optional[Mapping[str, Any]] = None,
        n_splits: int = 5,
        cv_mode: Optional[str] = None,
        design_matrices_exists_ok: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run :func:`plot_decoding_predictions.get_cv_predictions` on this runner's data, then optionally plot.

        Uses :class:`plot_decoding_predictions.ConfigRegressionEstimator` as ``runner_class`` and
        :func:`plot_decoding_predictions.plot_true_vs_pred` when ``show_plot`` is True.
        """
        cv_mode_resolved = cv_mode if cv_mode is not None else self.cv_mode
        specs = (
            model_specs
            if model_specs is not None
            else decoding_model_specs.MODEL_SPECS
        )
        if config is None:
            if model_name not in specs:
                raise KeyError(
                    f"model_name={model_name!r} not in model_specs; keys={list(specs)!r}"
                )
            spec = specs[model_name]
            config = cv_decoding.DecodingRunConfig(
                regression_model_class=spec.get("regression_model_class"),
                regression_model_kwargs=spec.get("regression_model_kwargs", {}),
                classification_model_class=spec.get("classification_model_class"),
                classification_model_kwargs=spec.get("classification_model_kwargs", {}),
                use_early_stopping=False,
                detrend_per_block=spec.get("detrend_per_block", True),
            )

        self.collect_data(exists_ok=design_matrices_exists_ok)

        X = np.asarray(self._get_neural_matrix_for_decoding(), dtype=float)
        y_df = self.get_target_df()
        if feature not in y_df.columns:
            raise KeyError(
                f"feature={feature!r} not in target columns: {list(y_df.columns)!r}"
            )
        y = y_df[feature].to_numpy().ravel()

        groups = self._get_groups()
        if groups is None:
            groups = np.zeros(len(X), dtype=int)
        X, groups, _ = cv_decoding._prepare_inputs(X, groups, 0)
        X_ok, y_ok, g_ok, _ = cv_decoding.filter_valid_rows(X, y, groups, None)

        mode = cv_decoding.infer_decoding_type(y_ok)
        if mode == "skip":
            raise ValueError(
                f"feature={feature!r} has insufficient / non-finite values for decoding"
            )
        if mode != "regression":
            raise ValueError(
                "find_true_vs_pred_cv_for_feature only supports regression targets; "
                f"got mode={mode!r}"
            )
        if g_ok is None:
            g_ok = np.zeros(len(y_ok), dtype=int)

        y_true, y_pred, fold_ids = plot_decoding_predictions.get_cv_predictions(
            X_ok,
            y_ok,
            g_ok,
            plot_decoding_predictions.ConfigRegressionEstimator,
            config,
            n_splits=n_splits,
            cv_mode=cv_mode_resolved,
            model_name=model_name,
        )

        return y_true, y_pred, fold_ids

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
        shuffle_mode_value = "none" if shuffle_mode is None else shuffle_mode

        # Basic identifying columns
        df["model_name"] = model_name
        df["n_splits"] = n_splits
        df["shuffle_mode"] = shuffle_mode_value
        df["cv_mode"] = cv_mode
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
        *,
        shuffle_mode: str = "none",
        groups_train=None,
        groups_test=None,
        kernelwidth_by_feature=None,
        detrend_cov_train=None,
        detrend_cov_test=None,
    ) -> Dict:
        """Train models on training data and evaluate on test data.
        
        Returns a dict with one row per target variable, containing predictions and metrics.
        """
        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
            infer_decoding_type,
            _shuffle_y_for_fold,
            _maybe_detrend_neural,
        )

        rng = np.random.default_rng(0)
        buffer_samples = getattr(config, 'buffer_samples', 20)
        
        results = []
        
        for col in y_train_df.columns:
            y_train = y_train_df[col].to_numpy()
            y_test = y_test_df[col].to_numpy()
            
            # Filter invalid rows (NaN/Inf)
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)

            # Optionally smooth neural data with a feature-specific kernel width
            if kernelwidth_by_feature is not None and groups_train is not None and groups_test is not None:
                width = kernelwidth_by_feature.get(col, None)
                if width is not None:
                    X_train_sm = smooth_neural_data.smooth_signal(
                        X_train,
                        int(width),
                        trial_idx=groups_train,
                    )
                    X_test_sm = smooth_neural_data.smooth_signal(
                        X_test,
                        int(width),
                        trial_idx=groups_test,
                    )
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

            # Skip if no variance in training labels
            if np.unique(y_tr).size <= 1:
                continue

            # Infer decoding mode from the finite training labels
            mode = infer_decoding_type(y_tr)
            
            cov_tr = (
                detrend_cov_train.iloc[train_valid].values
                if isinstance(detrend_cov_train, pd.DataFrame)
                else (np.asarray(detrend_cov_train)[train_valid] if detrend_cov_train is not None else None)
            )
            cov_te = (
                detrend_cov_test.iloc[test_valid].values
                if isinstance(detrend_cov_test, pd.DataFrame)
                else (np.asarray(detrend_cov_test)[test_valid] if detrend_cov_test is not None else None)
            )
            detrend_degree = getattr(config, 'detrend_degree', 1)
            X_tr, X_te = _maybe_detrend_neural(
                X_tr, X_te,
                cov_tr,
                cov_te,
                degree=detrend_degree,
            )
            
            # Scale features
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            if mode == 'regression':
                model_class = config.regression_model_class or CatBoostRegressor
                model_kwargs = config.regression_model_kwargs or dict(verbose=False)
                model = model_class(**model_kwargs)
                try:
                    model.fit(X_tr, y_tr)
                except CatBoostError as e:
                    if "All train targets are equal" in str(e) or "All targets are equal" in str(e):
                        continue
                    raise
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
                # Guard: sklearn classifiers require discrete labels. If y_tr has
                # non-integer floats (e.g. 0.5, 1.2) type_of_target returns
                # 'continuous' and LogisticRegression raises. Encode to 0,1,...
                # and filter test to seen classes only.
                try:
                    target_type = type_of_target(y_tr)
                except Exception:
                    target_type = "unknown"
                if target_type == "continuous":
                    continue

                le = LabelEncoder()
                y_tr_enc = le.fit_transform(y_tr)
                # Filter test to rows with classes seen in training
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
                model_kwargs = config.classification_model_kwargs or {}
                model = model_class(**model_kwargs)

                try:
                    model.fit(X_tr, y_tr_enc)
                except CatBoostError as e:
                    if "All train targets are equal" in str(e) or "All targets are equal" in str(e):
                        continue
                    raise
                y_pred_proba = model.predict_proba(X_te_f)

                try:
                    auc = float(roc_auc_score(y_te_enc, y_pred_proba[:, 1]))
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
                    # "n_total_folds": 1,
                    # "n_valid_folds": 1,
                    # "n_skipped_folds": 0,
                    "n_samples": int(len(y_te_f)),
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
            print(f"[{name}] Loaded cached design matrices from: {paths}")
            return True
        except Exception as e:
            print(f"[{name}] WARNING load matrices: {type(e).__name__}: {e}")
            return False


    # ============================================================
    # Cross-Validated Decoding (Refactored, Copy-Paste Ready)
    # ============================================================
    
    def reduce_binned_spikes(self, corr_threshold_for_lags_of_a_feature=0.98, 
                         vif_threshold=20, 
                         verbose=True):

        self.binned_spikes = process_encode_design.reduce_encoding_design(self.binned_spikes, 
                         corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature, 
                         vif_threshold=vif_threshold, 
                         verbose=verbose)


    def run_cv_decoding(
        self,
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
        cv_decoding_verbosity=1,
        use_detrend_inside_cv: bool = False,
    ) -> pd.DataFrame:
        """
        Run cross-validated model-spec decoding with optional nested CV
        for kernel width tuning.
        """

        # Resolve cv_mode: use argument if provided, otherwise default to instance setting
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode

        # --------------------------------------------------------
        # Prepare environment
        # --------------------------------------------------------
        self.model_specs = (
            model_specs
            if model_specs is not None
            else decoding_model_specs.MODEL_SPECS
        )

        if not load_existing_only:
            self.collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            save_dir = self._get_save_dir()

        if (shuffle_mode is not None) and (shuffle_mode != 'none'):
            save_dir = Path(save_dir) / f"shuffle_{shuffle_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = []
        self.cv_decoding_out_of_models = {}

        # --------------------------------------------------------
        # Loop over models
        # --------------------------------------------------------
        for model_name, spec in self.model_specs.items():

            print('Running model: ', model_name)

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
                use_detrend_inside_cv=use_detrend_inside_cv,
            )
            results_df['model_name'] = model_name

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
        use_detrend_inside_cv: bool = False,
    ):

        # Resolve cv_mode: use argument if provided, otherwise default to instance setting
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        detrend_covariates = self.get_detrend_covariates() if use_detrend_inside_cv else None
        groups = self._get_groups()
        detrend_per_block = spec.get("detrend_per_block", True)
        block_cv_modes = ("blocked_time_buffered", "blocked_time", "group_kfold")
        use_per_block = (
            detrend_covariates is not None
            and groups is not None
            and cv_mode in block_cv_modes
            and detrend_per_block
        )
        if detrend_covariates is None:
            detrend_suffix = ""
        elif use_per_block:
            detrend_suffix = "_detrend_perblock"
        else:
            detrend_suffix = "_detrend"
        model_save_path = self._cv_decoding_model_pickle_path(
            save_dir,
            model_name,
            shuffle_mode,
            fit_kernelwidth=fit_kernelwidth,
            n_splits=n_splits,
            inner_cv_splits=inner_cv_splits,
            fixed_width=fixed_width,
            cv_mode=cv_mode,
            detrend_suffix=detrend_suffix,
            include_shuffle_in_stem=True,
        )

        print('model_save_path: ', model_save_path)

        loaded = self._maybe_load_cv_decoding_result(
            save_dir=save_dir,
            model_name=model_name,
            shuffle_mode=shuffle_mode,
            fit_kernelwidth=fit_kernelwidth,
            n_splits=n_splits,
            inner_cv_splits=inner_cv_splits,
            fixed_width=fixed_width,
            cv_mode=cv_mode,
            detrend_suffix=detrend_suffix,
            load_if_exists=load_if_exists,
            verbose=True,
        )

        if loaded is not None:
            print(f"[{self._runner_name()}] {model_name}: loaded cached results")

            self.stats[f"cv_decoding_{model_name}"] = loaded
            self.cv_decoding_out_of_models[model_name] = loaded

            results_df = loaded["results_df"]
            shuffle_mode_value = "none" if shuffle_mode is None else shuffle_mode
            if "model_name" not in results_df.columns:
                results_df["model_name"] = model_name
            if "shuffle_mode" not in results_df.columns:
                results_df["shuffle_mode"] = shuffle_mode_value

            return results_df

        if load_existing_only:
            print('Failed to load existing results, and load_existing_only=True, so skipping computation.')
            return pd.DataFrame()

        # --------------------------------------------------------
        # Build decoding config
        # --------------------------------------------------------
        config = cv_decoding.DecodingRunConfig(
            regression_model_class=spec.get("regression_model_class"),
            regression_model_kwargs=spec.get("regression_model_kwargs", {}),
            classification_model_class=spec.get("classification_model_class"),
            classification_model_kwargs=spec.get("classification_model_kwargs", {}),
            use_early_stopping=False,
            detrend_per_block=spec.get("detrend_per_block", True),
        )

        # --------------------------------------------------------
        # Branch: nested CV or fixed width
        # --------------------------------------------------------
        if fit_kernelwidth:
            results_df = self.run_nested_kernelwidth_cv(
                model_name=model_name,
                config=config,
                n_splits=n_splits,
                candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits,
                cv_mode=cv_mode,
                model_save_path=model_save_path,
                verbosity=verbosity,
                shuffle_mode=shuffle_mode,
                detrend_covariates=detrend_covariates,
            )
        else:
            result_save_dir = Path(save_dir) / f"fixed_kernelwidth_{fixed_width}"
            results_df = self._run_fixed_width_cv(
                model_name=model_name,
                config=config,
                save_dir=result_save_dir,
                n_splits=n_splits,
                fixed_width=fixed_width,
                cv_mode=cv_mode,
                shuffle_mode=shuffle_mode,
                load_existing_only=load_existing_only,
                model_save_path=model_save_path,
                verbosity=verbosity,
                detrend_covariates=detrend_covariates,
            )


        # Ensure the results DataFrame contains the expected columns emitted by cv_decoding
        results_df = self._ensure_cv_decoding_columns(
            results_df,
            model_name=model_name,
            config=config,
            n_splits=n_splits,
            shuffle_mode=shuffle_mode,
            cv_mode=cv_mode,
            buffer_samples=None,
            context_label="pooled",
        )
        
        return results_df


    # ============================================================
    # Fixed Width CV
    # ============================================================

    @staticmethod
    def _cv_shuffle_stem_for_pickle(
        shuffle_mode: Optional[str], *, include_shuffle_in_stem: bool
    ) -> str:
        if not include_shuffle_in_stem:
            return ""
        sm = "none" if shuffle_mode is None else str(shuffle_mode)
        if sm == "none":
            return ""
        safe = "".join(
            ch if (ch.isalnum() or ch in "-._") else "_" for ch in sm
        )
        return f"_shuffle_{safe}"

    @staticmethod
    def _cv_decoding_model_pickle_path(
        save_dir: Path | str,
        model_name: str,
        shuffle_mode: Optional[str],
        *,
        fit_kernelwidth: bool,
        n_splits: int,
        inner_cv_splits: int,
        fixed_width: int,
        cv_mode: str,
        detrend_suffix: str,
        include_shuffle_in_stem: bool = True,
    ) -> Path:
        """Canonical path for the per-model CV decoding pickle (matches load/save)."""
        save_dir = Path(save_dir)
        shuff = BaseDecodingRunner._cv_shuffle_stem_for_pickle(
            shuffle_mode, include_shuffle_in_stem=include_shuffle_in_stem
        )
        if fit_kernelwidth:
            return (
                save_dir
                / "cv_decoding"
                / "cvnested"
                / f"{model_name}_n{n_splits}_inner{inner_cv_splits}_{cv_mode}{detrend_suffix}{shuff}.pkl"
            )
        return (
            save_dir
            / "cv_decoding"
            / "fixed_width"
            / f"{model_name}_width{fixed_width}_n{n_splits}_{cv_mode}{detrend_suffix}{shuff}.pkl"
        )

    @staticmethod
    def _maybe_load_cv_decoding_result(
        *,
        save_dir: Path | str,
        model_name: str,
        shuffle_mode: Optional[str],
        fit_kernelwidth: bool,
        n_splits: int,
        inner_cv_splits: int,
        fixed_width: int,
        cv_mode: str,
        detrend_suffix: str,
        load_if_exists: bool,
        verbose: bool,
    ) -> Optional[Dict[str, Any]]:
        """Load CV decoding pickle for this model and shuffle_mode (and CV layout), if present."""
        if not load_if_exists:
            return None

        shuffle_mode_resolved = "none" if shuffle_mode is None else str(shuffle_mode)

        primary = BaseDecodingRunner._cv_decoding_model_pickle_path(
            save_dir,
            model_name,
            shuffle_mode,
            fit_kernelwidth=fit_kernelwidth,
            n_splits=n_splits,
            inner_cv_splits=inner_cv_splits,
            fixed_width=fixed_width,
            cv_mode=cv_mode,
            detrend_suffix=detrend_suffix,
            include_shuffle_in_stem=True,
        )
        candidates: List[Path] = [primary]
        if shuffle_mode_resolved != "none":
            legacy = BaseDecodingRunner._cv_decoding_model_pickle_path(
                save_dir,
                model_name,
                shuffle_mode,
                fit_kernelwidth=fit_kernelwidth,
                n_splits=n_splits,
                inner_cv_splits=inner_cv_splits,
                fixed_width=fixed_width,
                cv_mode=cv_mode,
                detrend_suffix=detrend_suffix,
                include_shuffle_in_stem=False,
            )
            if legacy.resolve() != primary.resolve():
                candidates.append(legacy)

        for model_save_path in candidates:
            if not model_save_path.exists():
                continue
            with model_save_path.open("rb") as f:
                obj = pickle.load(f)
            if verbose:
                print(
                    f"cv decoding results loaded: {model_save_path} "
                    f"(model_name={model_name!r}, shuffle_mode={shuffle_mode_resolved!r})"
                )
            return obj

        print("Model save path does not exist (tried): ", candidates)
        return None



    def _run_fixed_width_cv(
        self,
        *,
        model_name,
        config,
        save_dir,
        n_splits,
        fixed_width,
        cv_mode,
        shuffle_mode,
        load_existing_only,
        model_save_path,
        verbosity,
        detrend_covariates=None,
    ):
        # Resolve cv_mode: use argument if provided, otherwise default to instance setting
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode

        X = self._get_neural_matrix_for_decoding()
        groups = self._get_groups()
        if fixed_width > 0:
            X = smooth_neural_data.smooth_signal(
                X,
                int(fixed_width),
                trial_idx=groups,
            )

        results_df = cv_decoding.run_cv_decoding(
            X=X,
            y_df=self.get_target_df(),
            behav_features=None,
            groups=self._get_groups(),
            n_splits=n_splits,
            config=config,
            context_label="pooled",
            save_dir=save_dir,
            model_name=model_name,
            shuffle_mode=shuffle_mode,
            load_existing_only=load_existing_only,
            verbosity=verbosity,
            detrend_covariates=detrend_covariates,
            cv_mode=cv_mode,
        )

        results_df["kernelwidth"] = fixed_width
        results_df["cv_mode"] = cv_mode

        self.cv_decoding_out_of_models[model_name] = {
            "results_df": results_df,
            "_cv_config": {
                "fit_kernelwidth": False,
                "fixed_width": fixed_width,
                "n_splits": n_splits,
                "cv_mode": cv_mode,
                "use_detrend_inside_cv": detrend_covariates is not None,
                "detrend_degree": getattr(config, "detrend_degree", 1),
                "detrend_per_block": getattr(config, "detrend_per_block", True),
            },
        }

        self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]

        # save self.stats[f"cv_decoding_{model_name}"] to model_save_path
        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            with model_save_path.open("wb") as f:
                pickle.dump(self.stats[f"cv_decoding_{model_name}"], f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity:
                print(f"Nested CV results saved: {model_save_path}")


        return results_df


    # ============================================================
    # Nested Kernelwidth CV
    # ============================================================

    def run_nested_kernelwidth_cv(
        self,
        *,
        model_name,
        model_save_path,
        config,
        n_splits,
        candidate_widths,
        inner_cv_splits,
        cv_mode,
        verbosity,
        shuffle_mode,
        detrend_covariates=None,
    ):

        from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
            _build_folds,
        )

        # Resolve cv_mode: use argument if provided, otherwise default to instance setting
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode

        X = self._get_neural_matrix_for_decoding()
        y_df = self.get_target_df()
        groups = self._get_groups()
        # detrend_covariates passed from caller (None when use_detrend_inside_cv=True)

        outer_splits = _build_folds(
            len(X),
            n_splits=n_splits,
            groups=groups,
            cv_splitter=cv_mode,
            random_state=0,
        )

        outer_results = []
        fold_tuning_info = []
        width_results_all_folds = []

        groups = self._get_groups()

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):

            # Slice CV groups (for folds) and trial_idx (for smoothing) separately
            groups_train = groups[train_idx] if groups is not None else None
            groups_test = groups[test_idx] if groups is not None else None

            best_width, width_scores, width_results = self._run_inner_kernelwidth_search(
                X=X[train_idx],
                y_df=y_df.iloc[train_idx],
                groups=groups_train,
                config=config,
                candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits,
                verbosity=verbosity,
                shuffle_mode=shuffle_mode,
                detrend_covariates=detrend_covariates.iloc[train_idx] if isinstance(detrend_covariates, pd.DataFrame) else (detrend_covariates[train_idx] if detrend_covariates is not None else None),
                cv_mode=cv_mode,
            )

            fold_result = self._evaluate_outer_fold(
                X_train=X[train_idx],
                y_train=y_df.iloc[train_idx],
                X_test=X[test_idx],
                y_test=y_df.iloc[test_idx],
                best_width_by_feature=best_width,
                config=config,
                shuffle_mode=shuffle_mode,
                groups_train=groups_train,
                groups_test=groups_test,
                detrend_cov_train=detrend_covariates.iloc[train_idx] if isinstance(detrend_covariates, pd.DataFrame) else (detrend_covariates[train_idx] if detrend_covariates is not None else None),
                detrend_cov_test=detrend_covariates.iloc[test_idx] if isinstance(detrend_covariates, pd.DataFrame) else (detrend_covariates[test_idx] if detrend_covariates is not None else None),
            )

            fold_result["fold"] = fold_idx
            # Map per-feature best widths into the results rows
            if isinstance(best_width, dict) and "behav_feature" in fold_result.columns:
                fold_result["kernelwidth"] = fold_result["behav_feature"].map(best_width).astype(float)
            else:
                fold_result["kernelwidth"] = best_width

            outer_results.append(fold_result)
            fold_tuning_info.append(width_scores)
            # collect per-width inner-CV results for this outer fold
            for w, df_w in width_results.items():
                if df_w is None or df_w.empty:
                    continue
                df_tmp = df_w.copy()
                df_tmp["fold"] = fold_idx
                df_tmp["kernelwidth"] = int(w)
                width_results_all_folds.append(df_tmp)

        results_df = pd.concat(outer_results, ignore_index=True)
        results_df['model_name'] = model_name
        results_df["cv_mode"] = cv_mode

        # combined per-width inner-CV results across folds
        if width_results_all_folds:
            results_df_all_filt = pd.concat(width_results_all_folds, ignore_index=True)
        else:
            results_df_all_filt = pd.DataFrame()

        self.cv_decoding_out_of_models[model_name] = {
            "results_df": results_df,
            "results_df_all_filt": results_df_all_filt,
            "fold_tuning_info": fold_tuning_info,
            "_cv_config": {
                "fit_kernelwidth": True,
                "candidate_widths": list(candidate_widths),
                "n_splits": n_splits,
                "inner_cv_splits": inner_cv_splits,
                "cv_mode": cv_mode,
                "use_detrend_inside_cv": detrend_covariates is not None,
                "detrend_degree": getattr(config, "detrend_degree", 1),
                "detrend_per_block": getattr(config, "detrend_per_block", True),
            },
        }

        self.stats[f"cv_decoding_{model_name}"] = self.cv_decoding_out_of_models[model_name]

        # save self.stats[f"cv_decoding_{model_name}"] to model_save_path
        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            with model_save_path.open("wb") as f:
                pickle.dump(self.stats[f"cv_decoding_{model_name}"], f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity:
                print(f"Nested CV results saved: {model_save_path}")

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
        verbosity,
        shuffle_mode,
        detrend_covariates=None,
        cv_mode=None,
    ):

        # Track best width per feature (behav_feature)
        best_width = {}
        best_score = {}
        width_scores = {}
        width_results = {}
        
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode

        for width in candidate_widths:

            X_smooth = smooth_neural_data.smooth_signal(
                X,
                int(width),
                trial_idx=groups,
            )

            inner_df = cv_decoding.run_cv_decoding(
                cv_mode=cv_mode,
                X=X_smooth,
                y_df=y_df,
                behav_features=None,
                groups=groups,
                n_splits=inner_cv_splits,
                config=config,
                context_label="pooled",
                save_dir=None,
                model_name=None,
                shuffle_mode=shuffle_mode,
                verbosity=verbosity,
                detrend_covariates=detrend_covariates,
            )

            # store full per-width inner-CV result
            try:
                inner_df_copy = inner_df.copy() if inner_df is not None else pd.DataFrame()
            except Exception:
                inner_df_copy = pd.DataFrame()
            if not inner_df_copy.empty:
                inner_df_copy["kernelwidth"] = int(width)
            width_results[int(width)] = inner_df_copy

            # Overall width score (mean across all features for plotting)
            if inner_df is None or inner_df.empty or "mode" not in inner_df.columns:
                width_scores[int(width)] = float("nan")
                continue

            n = len(inner_df)
            r_cv = (
                inner_df["r_cv"].to_numpy(dtype=float)
                if "r_cv" in inner_df.columns
                else np.full(n, np.nan)
            )
            auc_mean = (
                inner_df["auc_mean"].to_numpy(dtype=float)
                if "auc_mean" in inner_df.columns
                else np.full(n, np.nan)
            )
            metric_scores = np.where(
                inner_df["mode"].to_numpy() == "regression",
                r_cv,
                auc_mean,
            )

            mean_score = float(np.nanmean(metric_scores))
            width_scores[int(width)] = mean_score

            # Per-feature best widths
            for _, row in inner_df.iterrows():
                feat = row.get("behav_feature", None)
                if feat is None:
                    continue
                mode = row.get("mode")
                if mode == "regression":
                    score = row.get("r_cv", np.nan)
                else:
                    score = row.get("auc_mean", np.nan)
                if not np.isfinite(score):
                    continue
                prev_best = best_score.get(feat, -np.inf)
                if score > prev_best:
                    best_score[feat] = float(score)
                    best_width[feat] = int(width)

        return best_width, width_scores, width_results


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
        best_width_by_feature,
        config,
        shuffle_mode: str = "none",
        groups_train=None,
        groups_test=None,
        detrend_cov_train=None,
        detrend_cov_test=None,
    ):

        return self._train_test_single_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            shuffle_mode=shuffle_mode,
            groups_train=groups_train,
            groups_test=groups_test,
            kernelwidth_by_feature=best_width_by_feature,
            detrend_cov_train=detrend_cov_train,
            detrend_cov_test=detrend_cov_test,
        )
    

    def plot_fold_tuning_info(self):
        for model in self.results_df['model_name'].unique():
            fig, ax, summary = plot_decoding_utils.plot_fold_tuning_info(self.stats[f'cv_decoding_{model}']['fold_tuning_info'], shade='sem', show_folds=True)
            print(summary['best_width'], summary['best_mean_cv'])
            plt.show()

    def extract_regression_feature_scores_df(self, row_selector_fn=None, regression_metric='r_cv', classification_metric='auc_mean') -> pd.DataFrame:
        '''
        Extract regression decoding scores from results_df.

        Returns
        -------
        df : pandas.DataFrame
            Columns:
            - variable
            - score
        '''
        if row_selector_fn is None:
            row_selector_fn = show_decoding_results._select_rows

        self.results_df = plot_decoding_utils.add_score_column(self.results_df, regression_metric, classification_metric)

        selected_df = row_selector_fn(self.results_df)

        rel_df = selected_df[selected_df['mode'] == 'regression'].copy()

        if rel_df.empty:
            raise ValueError('No regression rows found.')

        df = (
            rel_df[['behav_feature', 'score']]
            .rename(columns={'behav_feature': 'variable'})
            .sort_values('score', ascending=False)
            .reset_index(drop=True)
        )

        self.regression_feature_scores_df = df.copy()

    def get_processed_spike_rates(self):
        self.processed_spike_rates = decoding_design_utils.get_processed_spike_rates(
            self.pn.spikes_df,
            smooth_spikes=self.smooth_spikes,
            detrend_spikes=self.detrend_spikes,
            smoothing_width=self.smoothing_width,
            drop_bad_neurons=self.drop_bad_neurons,
        )

    # ------------------------------------------------------------------
    # Categorical variable discovery and ANOVA
    #
    # NOTE: ANOVA is conceptually an *encoding* analysis — it asks whether
    # neural activity (spike counts) is modulated by a behavioral variable,
    # i.e. the variable → neuron direction.  It lives here in the decoding
    # runner purely for convenience: the runner already holds both
    # ``feats_to_decode`` (behavioral features) and ``binned_spikes``
    # (neural activity) in the same data-collection pipeline, so there is
    # no need to duplicate that infrastructure in the encoding runner just
    # to run a quick significance screen.  Think of these methods as a
    # lightweight sanity-check / pre-filter that can be called before or
    # after a full decoding run without touching the encoding codebase.
    # ------------------------------------------------------------------

    def find_categorical_vars_in_binned_feats(
        self,
        max_unique: int = 20,
        min_unique: int = 2,
    ) -> List[str]:
        """
        Return column names in ``feats_to_decode`` that are categorical
        (integer-valued with few distinct levels) but are *not* event variables
        and *not* spline / basis-function columns.

        A column qualifies when ALL of the following hold:
        - Not listed in ``var_categories['event_vars']`` (if present)
        - Not a known basis column (from ``temporal_meta`` / ``tuning_meta`` groups, if present)
        - Name does not match the ``_{digits}$`` suffix pattern used for basis columns
        - Its non-null values are all integers (or integer-valued floats)
        - The number of unique values is between ``min_unique`` and ``max_unique``

        Parameters
        ----------
        max_unique : int
            Upper bound on distinct values (inclusive).
        min_unique : int
            Lower bound on distinct values (inclusive, usually 2).

        Returns
        -------
        List[str]
            Ordered list of qualifying column names.
        """
        import re

        feats_df = self.feats_to_decode
        if feats_df is None:
            raise RuntimeError(
                "feats_to_decode is None; call collect_data() first."
            )

        event_vars: set = set(
            getattr(self, "var_categories", {}).get("event_vars", [])
        )

        # Exact spline/basis column names from temporal and tuning meta groups
        basis_cols: set = set()
        for meta_attr in ("temporal_meta", "tuning_meta"):
            meta = getattr(self, meta_attr, None) or {}
            for col_list in (meta.get("groups") or {}).values():
                basis_cols.update(col_list)

        # Regex fallback: basis columns end with _{int} (e.g. stop_0, v_19)
        _basis_suffix = re.compile(r"_\d+$")

        categorical_cols: List[str] = []
        for col in feats_df.columns:
            if col in event_vars:
                continue
            if col in basis_cols:
                continue
            if _basis_suffix.search(col):
                continue

            series = feats_df[col]
            notnull = series.dropna()
            if len(notnull) == 0:
                continue

            is_int_like = pd.api.types.is_integer_dtype(series) or (
                pd.api.types.is_float_dtype(series)
                and bool((notnull == notnull.round()).all())
            )
            if not is_int_like:
                continue

            n_unique = int(series.nunique(dropna=True))
            if min_unique <= n_unique <= max_unique:
                categorical_cols.append(col)

        return categorical_cols

    def run_anova_for_categorical_vars(
        self,
        unit_idx: int,
        categorical_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        One-way ANOVA of spike counts grouped by each categorical variable.

        Although ANOVA is an encoding-direction test (variable → neuron), it
        is implemented here because the decoding runner already has both
        ``feats_to_decode`` and ``binned_spikes`` readily available.

        Delegates per-column computation to
        ``linear_model_utils.anova_spike_counts_for_columns``.

        Parameters
        ----------
        unit_idx : int
            Column index of the target neuron in ``binned_spikes``.
        categorical_cols : list of str, optional
            Columns to test.  Defaults to the output of
            ``find_categorical_vars_in_binned_feats()``.
        alpha : float
            Significance threshold used for the ``significant`` flag.
        verbose : bool
            Print a summary table when ``True``.

        Returns
        -------
        pd.DataFrame
            One row per variable; columns:
            ``variable``, ``n_categories``, ``F``, ``p_value``, ``significant``
        """
        if self.feats_to_decode is None or self.binned_spikes is None:
            self.collect_data(exists_ok=True)

        if categorical_cols is None:
            categorical_cols = self.find_categorical_vars_in_binned_feats()

        y = np.asarray(
            self.binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float
        ).ravel()

        result_df = linear_model_utils.anova_spike_counts_for_columns(
            y=y,
            binned_feats=self.feats_to_decode,
            categorical_cols=categorical_cols,
            alpha=alpha,
        )

        if verbose:
            linear_model_utils.print_anova_single_unit(result_df, unit_idx, alpha)

        return result_df

    def run_anova_all_neurons(
        self,
        categorical_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        verbose: bool = True,
        results_cache: Optional[str] = None,
        force_rerun: bool = False,
    ) -> Dict[int, pd.DataFrame]:
        """
        Run one-way ANOVA for categorical variables across all neurons.

        See ``run_anova_for_categorical_vars`` for the rationale of why an
        encoding-direction ANOVA lives inside the decoding runner.

        Calls ``run_anova_for_categorical_vars`` for each neuron and collects
        the results.  ``collect_data`` is called implicitly when needed.

        Parameters
        ----------
        categorical_cols : list of str, optional
            Columns to test.  Defaults to ``self.categorical_vars`` (if set) or
            the result of ``find_categorical_vars_in_binned_feats()``.
        alpha : float
            Significance threshold forwarded to each per-neuron call.
        verbose : bool
            Print per-neuron summaries.
        results_cache : str, optional
            Path to a pickle file for saving / loading the full results dict.
            Defaults to ``<save_dir>/anova_results.pkl`` (next to
            ``feats_to_decode.pkl``).  Pass ``""`` to disable caching.
        force_rerun : bool
            When ``True``, ignore an existing cache and recompute from scratch,
            then overwrite the cache with the new results.

        Returns
        -------
        Dict[int, pd.DataFrame]
            Mapping from ``unit_idx`` to its ANOVA result DataFrame.
        """
        # ── resolve cache path ────────────────────────────────────────────
        if results_cache == "":
            cache_path: Optional[Path] = None
        elif results_cache is not None:
            cache_path = Path(results_cache)
        else:
            cache_path = self._results_cache_path("anova")

        if not force_rerun and cache_path is not None:
            cached = self._load_results_cache(cache_path, "anova")
            if cached is not None:
                if verbose:
                    linear_model_utils.print_anova_all_neurons(cached, alpha)
                return cached

        # ── run ───────────────────────────────────────────────────────────
        self.collect_data(exists_ok=True)

        cols = categorical_cols
        if cols is None:
            cols = getattr(self, 'categorical_vars', None)
        if cols is None:
            cols = self.find_categorical_vars_in_binned_feats()

        if not cols:
            print(f'[{self._runner_name()}] No categorical vars to test — ANOVA skipped.')
            return {}

        n_neurons = self.binned_spikes.shape[1]
        results: Dict[int, pd.DataFrame] = {}
        for unit_idx in range(n_neurons):
            # Pass verbose=False here; the all-neuron summary below replaces
            # the per-neuron printouts when verbose=True.
            results[unit_idx] = self.run_anova_for_categorical_vars(
                unit_idx=unit_idx,
                categorical_cols=cols,
                alpha=alpha,
                verbose=False,
            )

        if cache_path is not None:
            self._save_results_cache(results, cache_path, "anova")

        if verbose:
            linear_model_utils.print_anova_all_neurons(results, alpha)

        return results

    def plot_anova_results(
        self,
        anova_results: Dict[int, pd.DataFrame],
        alpha: float = 0.05,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualise the output of ``run_anova_all_neurons``.

        Delegates all logic to
        ``linear_model_utils.plot_anova_results``; see that function for full
        parameter and return-value documentation.  The only runner-specific
        behaviour is supplying a default ``title`` that includes the runner
        name and dataset dimensions.
        """
        if not anova_results:
            print(f"[{self._runner_name()}] plot_anova_results: empty results dict.")
            return plt.figure()

        if title is None:
            n_neurons = len(anova_results)
            n_vars    = len(next(iter(anova_results.values())))
            title     = (
                f"ANOVA  —  {self._runner_name()}  "
                f"({n_neurons} neurons, {n_vars} vars, α={alpha})"
            )

        return linear_model_utils.plot_anova_results(
            anova_results=anova_results,
            alpha=alpha,
            figsize=figsize,
            title=title,
        )

    # ------------------------------------------------------------------
    # LM (OLS with partial F-tests per categorical variable)
    #
    # These methods mirror the ANOVA block above.  The key difference:
    # ANOVA tests each variable in isolation (marginal effects), while
    # the LM tests each variable while controlling for all others
    # simultaneously (partial / Type-II effects).  Running both and
    # comparing via plot_lm_vs_anova_results reveals confounds:
    #
    #   ANOVA sig, LM not  →  apparent effect was confounded by another var
    #   Both sig            →  effect is robust to other variables
    #   LM sig, ANOVA not  →  effect was suppressed in isolation (rare)
    # ------------------------------------------------------------------

    def _default_covariates_cache_path(self) -> Optional[Path]:
        """Return ``<save_dir>/lm_covariates_cache.json``, or None if unavailable."""
        try:
            return Path(self._get_save_dir()) / "lm_covariates_cache.json"
        except Exception:
            return None

    def _results_cache_path(self, label: str) -> Optional[Path]:
        """Return ``<save_dir>/<label>_results.pkl``, or None if unavailable."""
        try:
            return Path(self._get_save_dir()) / f"{label}_results.pkl"
        except Exception:
            return None

    def _load_results_cache(
        self, cache_path: Path, label: str
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """Load a pickled results dict.  Returns None on any failure."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as fh:
                results = pickle.load(fh)
            print(
                f"[{self._runner_name()}] {label.upper()}: loaded results from cache"
                f"  ({cache_path})"
            )
            return results
        except Exception as exc:
            print(
                f"[{self._runner_name()}] {label.upper()}: cache load failed"
                f" ({exc}) — re-running."
            )
            return None

    def _save_results_cache(
        self, results: Dict[int, pd.DataFrame], cache_path: Path, label: str
    ) -> None:
        """Pickle a results dict, creating parent directories as needed."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[{self._runner_name()}] {label.upper()}: saved results to cache"
                f"  ({cache_path})"
            )
        except Exception as exc:
            print(
                f"[{self._runner_name()}] {label.upper()}: cache save failed ({exc})"
            )

    def _resolve_lm_covariate_cols(
        self,
        categorical_cols: List[str],
        continuous_cols: Optional[List[str]],
        covariates_cache: Optional[str],
    ) -> List[str]:
        """
        Build the reduced covariate column list for ``include_all_feats`` LM calls.

        All columns of ``feats_to_decode`` not already in ``categorical_cols``
        or ``continuous_cols`` are candidates.
        ``process_encode_design.reduce_encoding_design`` is then applied
        (corr threshold 0.95, VIF threshold 20) to remove zero-variance,
        near-perfectly-correlated (e.g. adjacent spline bases), and high-VIF
        columns so the OLS design matrix stays well-conditioned.

        The result is always cached as a JSON list of column names so that
        subsequent calls skip the expensive reduction step.  The cache file
        lives next to ``feats_to_decode.pkl`` in ``_get_save_dir()`` unless
        an explicit ``covariates_cache`` path is provided.

        Parameters
        ----------
        categorical_cols : list of str
            Columns used as main effects (excluded from covariates).
        continuous_cols : list of str or None
            Columns used as reported continuous predictors (excluded).
        covariates_cache : str or None
            Override the default cache path.  Pass an empty string ``""`` to
            disable caching entirely.

        Returns
        -------
        list of str
            Reduced covariate column names.
        """
        import json

        # Resolve the effective cache path: explicit > auto-derived > disabled
        if covariates_cache == "":
            cache_path: Optional[Path] = None          # explicitly disabled
        elif covariates_cache is not None:
            cache_path = Path(covariates_cache)
        else:
            cache_path = self._default_covariates_cache_path()  # same dir as feats_to_decode

        if cache_path is not None and cache_path.exists():
            with open(cache_path) as fh:
                covariate_cols = json.load(fh)
            print(
                f"[{self._runner_name()}] LM covariates: loaded "
                f"{len(covariate_cols)} cols from cache  ({cache_path})"
            )
            return covariate_cols

        # Build the full joint design matrix so the reduction sees correlations
        # between covariates AND test-variable dummies, not covariates alone.
        exclude       = set(categorical_cols) | set(continuous_cols or [])
        candidate_cov = [c for c in self.feats_to_decode.columns if c not in exclude]

        test_dummies = pd.get_dummies(
            self.feats_to_decode[categorical_cols], drop_first=True, dtype=float
        )
        cont_df  = (
            self.feats_to_decode[list(continuous_cols)].astype(float)
            if continuous_cols else pd.DataFrame(index=self.feats_to_decode.index)
        )
        cov_df   = self.feats_to_decode[candidate_cov].astype(float)
        joint_df = pd.concat([test_dummies, cont_df, cov_df], axis=1)

        # Protected columns (test variables) must not be dropped.
        protected = set(test_dummies.columns) | set(continuous_cols or [])

        joint_reduced = process_encode_design.reduce_encoding_design(
            joint_df,
            corr_threshold_for_lags_of_a_feature=0.9,
            vif_threshold=5,
            verbose=False,
        )
        # Add back any protected columns the reduction may have dropped
        for col in protected:
            if col not in joint_reduced.columns and col in joint_df.columns:
                joint_reduced[col] = joint_df[col]

        # Extract surviving covariate column names (original, non-dummy names)
        covariate_cols = [c for c in candidate_cov if c in joint_reduced.columns]

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump(covariate_cols, fh)
            print(
                f"[{self._runner_name()}] LM covariates: saved "
                f"{len(covariate_cols)} cols to cache  ({cache_path})"
            )

        return covariate_cols

    def run_lm_for_categorical_vars(
        self,
        unit_idx: int,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        include_all_feats: bool = True,
        covariates_cache: Optional[str] = None,
        _covariate_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fit an OLS LM for one neuron and compute partial F-tests.

        All categorical variables are included simultaneously; each is tested
        via a drop-one partial F-test (Type-II SS) that controls for every
        other predictor.  Optional ``continuous_cols`` are included as
        covariates and also reported.

        Parameters
        ----------
        unit_idx : int
            Column index of the target neuron in ``binned_spikes``.
        categorical_cols : list of str, optional
            Columns to dummy-encode and test.  Defaults to the output of
            ``find_categorical_vars_in_binned_feats()``.
        continuous_cols : list of str, optional
            Continuous predictors included in the model and reported.
        include_all_feats : bool
            When ``True``, every remaining column of ``feats_to_decode``
            (after reduction via ``_resolve_lm_covariate_cols``) is added as
            an unreported covariate.
        covariates_cache : str, optional
            Path to a JSON file for saving / loading the reduced covariate
            column list.  Only used when ``include_all_feats=True``.
            On first call the list is computed and saved; on later calls it
            is loaded directly, skipping the expensive reduction step.
        alpha : float
            Significance threshold.
        verbose : bool
            Print a formatted result table when ``True``.

        Returns
        -------
        pd.DataFrame
            One row per variable; columns:
            ``variable``, ``n_params``, ``F``, ``p_value``, ``significant``
        """
        if self.feats_to_decode is None or self.binned_spikes is None:
            self.collect_data(exists_ok=True)

        if categorical_cols is None:
            categorical_cols = self.find_categorical_vars_in_binned_feats()

        # _covariate_cols is a private shortcut used by run_lm_all_neurons to
        # pass pre-resolved covariates directly, avoiding per-neuron recomputation.
        covariate_cols: Optional[List[str]] = _covariate_cols
        if covariate_cols is None and include_all_feats:
            covariate_cols = self._resolve_lm_covariate_cols(
                categorical_cols=categorical_cols,
                continuous_cols=continuous_cols,
                covariates_cache=covariates_cache,
            )

        y = np.asarray(
            self.binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float
        ).ravel()

        result_df = linear_model_utils.lm_spike_counts_for_columns(
            y=y,
            binned_feats=self.feats_to_decode,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            covariate_cols=covariate_cols,
            alpha=alpha,
        )

        if verbose:
            linear_model_utils.print_lm_single_unit(result_df, unit_idx, alpha)

        return result_df

    def run_lm_all_neurons(
        self,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        include_all_feats: bool = True,
        covariates_cache: Optional[str] = None,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """
        Run the partial-F LM for categorical variables across all neurons.

        Calls ``run_lm_for_categorical_vars`` for each neuron.
        ``collect_data`` is called implicitly when needed.

        When ``include_all_feats=True`` the covariate column list is resolved
        **once** via ``_resolve_lm_covariate_cols`` and reused for every
        neuron, so the expensive ``reduce_encoding_design`` call is not
        repeated N times.

        Parameters
        ----------
        categorical_cols : list of str, optional
            Columns to test.  Defaults to ``self.categorical_vars`` (if set)
            or ``find_categorical_vars_in_binned_feats()``.
        continuous_cols : list of str, optional
            Continuous covariates forwarded to each per-neuron call.
        include_all_feats : bool
            When ``True``, all remaining columns of ``feats_to_decode`` are
            added as unreported covariates.  See ``run_lm_for_categorical_vars``
            for details.
        covariates_cache : str, optional
            Path to a JSON file for saving / loading the reduced covariate
            column list.  Only used when ``include_all_feats=True``.
        alpha : float
            Significance threshold.
        verbose : bool
            Print an all-neuron summary grid when ``True``.

        Returns
        -------
        Dict[int, pd.DataFrame]
            Mapping from ``unit_idx`` to its LM result DataFrame.
        """
        self.collect_data(exists_ok=True)

        cols = categorical_cols
        if cols is None:
            cols = getattr(self, "categorical_vars", None)
        if cols is None:
            cols = self.find_categorical_vars_in_binned_feats()

        if not cols:
            print(f"[{self._runner_name()}] No categorical vars to test — LM skipped.")
            return {}

        # Resolve covariates once for all neurons when include_all_feats is set.
        resolved_covariate_cols: Optional[List[str]] = None
        if include_all_feats:
            resolved_covariate_cols = self._resolve_lm_covariate_cols(
                categorical_cols=cols,
                continuous_cols=continuous_cols,
                covariates_cache=covariates_cache,
            )

        n_neurons = self.binned_spikes.shape[1]
        results: Dict[int, pd.DataFrame] = {}
        for unit_idx in range(n_neurons):
            results[unit_idx] = self.run_lm_for_categorical_vars(
                unit_idx=unit_idx,
                categorical_cols=cols,
                continuous_cols=continuous_cols,
                include_all_feats=False,           # already resolved above
                _covariate_cols=resolved_covariate_cols,
                alpha=alpha,
                verbose=False,
            )

        if verbose:
            linear_model_utils.print_lm_all_neurons(results, alpha)

        return results

    def plot_lm_vs_anova_results(
        self,
        anova_results: Dict[int, pd.DataFrame],
        lm_results: Dict[int, pd.DataFrame],
        alpha: float = 0.05,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualise ANOVA vs LM consistency across all neurons.

        Delegates all logic to ``linear_model_utils.plot_lm_vs_anova``; see
        that function for full documentation.  Supplies a runner-specific
        default title.
        """
        if not anova_results or not lm_results:
            print(f"[{self._runner_name()}] plot_lm_vs_anova_results: empty results.")
            return plt.figure()

        if title is None:
            n_neurons = len(anova_results)
            title = (
                f"ANOVA vs LM  —  {self._runner_name()}  "
                f"({n_neurons} neurons, α={alpha})"
            )

        return linear_model_utils.plot_lm_vs_anova(
            anova_results=anova_results,
            lm_results=lm_results,
            alpha=alpha,
            figsize=figsize,
            title=title,
        )
        
    def clean_var_categories(self):
        self.var_categories = decoding_design_utils._build_clean_var_categories(self.feats_to_decode, self.var_categories)
