"""Base class for decoding runners with shared one-FF-style (CCA + linear readout) and CV decoding logic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decode_stops_utils, plot_decoding_utils
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import decoding_model_specs
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import one_ff_style_decoding_runner

from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import show_decoding_results
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import process_encode_design



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

    def __init__(self, bin_width: float = 0.04):
        self.bin_width = bin_width
        self.stats: Dict = {}

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
        fit_kernelwidth: bool = True,
        candidate_widths: Sequence[int] = tuple(range(2, 6, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 5,
        cv_mode: str = "blocked_time_buffered",  # can be 'blocked_time_buffered', 'blocked_time', 'group_kfold'
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity=1,
        use_detrend_inside_cv: bool = False,
    ) -> pd.DataFrame:

        if save_dir is None:
            save_dir = Path(self._get_save_dir())    
        
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
            use_detrend_inside_cv=use_detrend_inside_cv,
        )
        
        if cv_mode is not None and 'cv_mode' in self.all_results.columns:
            self.results_df = self.all_results[self.all_results['cv_mode'] == cv_mode] 
        else:
            self.results_df = self.all_results.copy()


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
        *,
        shuffle_mode: str = "none",
        groups_train=None,
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

            X_tr = X_train[train_valid]
            y_tr = y_train[train_valid].copy()
            g_tr = groups_train[train_valid] if groups_train is not None else None
            X_te = X_test[test_valid]
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
            print(f"[{name}] Loaded cached design matrices")
            return True
        except Exception as e:
            print(f"[{name}] WARNING load matrices: {type(e).__name__}: {e}")
            return False


    # ============================================================
    # Cross-Validated Decoding (Refactored, Copy-Paste Ready)
    # ============================================================
    
    def reduce_binned_spikes(self, corr_threshold_for_lags_of_a_feature=0.98, 
                         vif_threshold_for_initial_subset=20, 
                         vif_threshold=20, 
                         verbose=True):

        self.binned_spikes = process_encode_design.reduce_encoding_design(self.binned_spikes, 
                         corr_threshold_for_lags_of_a_feature=corr_threshold_for_lags_of_a_feature, 
                         vif_threshold_for_initial_subset=vif_threshold_for_initial_subset, 
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
        candidate_widths: Sequence[int] = tuple(range(2, 6, 1)),
        fixed_width: int = 25,
        inner_cv_splits: int = 5,
        cv_mode: str = "blocked_time_buffered",
        load_if_exists: bool = True,
        load_existing_only: bool = False,
        cv_decoding_verbosity=1,
        use_detrend_inside_cv: bool = False,
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
            else decoding_model_specs.MODEL_SPECS
        )

        if not load_existing_only:
            self.collect_data(exists_ok=design_matrices_exists_ok)

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
        if fit_kernelwidth:
            model_save_path = Path(save_dir) / "cv_decoding" / "cvnested" / f"{model_name}_n{n_splits}_inner{inner_cv_splits}_{cv_mode}{detrend_suffix}.pkl"
        else:
            model_save_path = Path(save_dir) / "cv_decoding" / "fixed_width" / f"{model_name}_width{fixed_width}_n{n_splits}_{cv_mode}{detrend_suffix}.pkl"

        print('model_save_path: ', model_save_path)

        loaded = self._maybe_load_cv_decoding_result(
            str(model_save_path),
            load_if_exists,
            verbose=True,
        )

        if loaded is not None:
            print(f"[{self._runner_name()}] {model_name}: loaded cached results")

            self.stats[f"cv_decoding_{model_name}"] = loaded
            self.cv_decoding_out_of_models[model_name] = loaded

            results_df = loaded["results_df"]
            results_df["model_name"] = model_name
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
            cv_mode=cv_mode,
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
    def _maybe_load_cv_decoding_result(
        model_save_path: Optional[str],
        load_if_exists: bool,
        verbose: bool,
    ):
        """Load CV decoding result if it exists and load_if_exists=True."""
        if not load_if_exists:
            return None
        
        if model_save_path is None:
            return None
        else:
            model_save_path = Path(model_save_path)

        if not model_save_path.exists():
            print('Model save path does not exist: ', model_save_path)
            return None
        with model_save_path.open("rb") as f:
            obj = pickle.load(f)
        if verbose:
            print(f"cv decoding results loaded: {model_save_path}")
        return obj



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

        X = self._get_neural_matrix()
        if fixed_width > 0:
            X = decode_stops_utils.smooth_signal(X, int(fixed_width))

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
        )

        results_df["kernelwidth"] = fixed_width

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

        X = self._get_neural_matrix()
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

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):

            best_width, width_scores, width_results = self._run_inner_kernelwidth_search(
                X=X[train_idx],
                y_df=y_df.iloc[train_idx],
                groups=groups[train_idx] if groups is not None else None,
                config=config,
                candidate_widths=candidate_widths,
                inner_cv_splits=inner_cv_splits,
                verbosity=verbosity,
                shuffle_mode=shuffle_mode,
                detrend_covariates=detrend_covariates.iloc[train_idx] if isinstance(detrend_covariates, pd.DataFrame) else (detrend_covariates[train_idx] if detrend_covariates is not None else None),
            )

            fold_result = self._evaluate_outer_fold(
                X_train=X[train_idx],
                y_train=y_df.iloc[train_idx],
                X_test=X[test_idx],
                y_test=y_df.iloc[test_idx],
                best_width=best_width,
                config=config,
                shuffle_mode=shuffle_mode,
                groups_train=groups[train_idx] if groups is not None else None,
                detrend_cov_train=detrend_covariates.iloc[train_idx] if isinstance(detrend_covariates, pd.DataFrame) else (detrend_covariates[train_idx] if detrend_covariates is not None else None),
                detrend_cov_test=detrend_covariates.iloc[test_idx] if isinstance(detrend_covariates, pd.DataFrame) else (detrend_covariates[test_idx] if detrend_covariates is not None else None),
            )

            fold_result["fold"] = fold_idx
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
    ):

        best_width = None
        best_score = -np.inf
        width_scores = {}
        width_results = {}

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
        best_width,
        config,
        shuffle_mode: str = "none",
        groups_train=None,
        detrend_cov_train=None,
        detrend_cov_test=None,
    ):

        X_train = decode_stops_utils.smooth_signal(X_train, int(best_width))
        X_test = decode_stops_utils.smooth_signal(X_test, int(best_width))

        return self._train_test_single_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            shuffle_mode=shuffle_mode,
            groups_train=groups_train,
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

        reg_df = selected_df[selected_df['mode'] == 'regression'].copy()

        if reg_df.empty:
            raise ValueError('No regression rows found.')

        df = (
            reg_df[['behav_feature', 'score']]
            .rename(columns={'behav_feature': 'variable'})
            .sort_values('score', ascending=False)
            .reset_index(drop=True)
        )

        self.regression_feature_scores_df = df.copy()
