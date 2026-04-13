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
    from encoding_tasks import PNTask
    from encoding_models import PGAMModel, RNNModel

    task = PNTask(raw_data_folder_path)
    runner = EncodingRunner(task, PGAMModel())
    runner.crossval(unit_idx=0)

    runner2 = EncodingRunner(task, RNNModel())
    runner2.crossval(unit_idx=0)
"""

from __future__ import annotations

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
    Minimal interface that every model must implement.

    Parameters passed to fit / crossval are always
    (task: BaseEncodingTask, unit_idx: int, **kwargs).
    """

    def fit(self, task, unit_idx: int, **kwargs):
        raise NotImplementedError

    def crossval(self, task, unit_idx: int, **kwargs):
        raise NotImplementedError


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

    def _get_design_for_unit(self, task, unit_idx: int, *, use_neural_coupling: bool = False):
        """Build design_df + gam_groups for one unit from task state."""
        task._prepare_spike_history_components()

        if unit_idx < 0 or unit_idx >= len(task.spk_colnames):
            raise IndexError(
                f"unit_idx {unit_idx} out of range [0, {len(task.spk_colnames)})"
            )
        target_col = list(task.spk_colnames.keys())[unit_idx]
        cross_neurons = (
            [c for c in task.spk_colnames.keys() if c != target_col]
            if use_neural_coupling
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

        if use_neural_coupling:
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
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
        n_folds: Optional[int] = 10,
    ) -> dict:
        coupling_subdir = self._coupling_subdir(use_neural_coupling)
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
        use_neural_coupling: bool = False,
    ) -> FitResult:
        """Fit Poisson GAM for one unit."""
        task.collect_data(exists_ok=True)

        design_df, gam_groups = self._get_design_for_unit(
            task, unit_idx, use_neural_coupling=use_neural_coupling
        )

        if len(design_df) != len(task.binned_spikes):
            raise ValueError(
                f"design / binned_spikes row mismatch: "
                f"{len(design_df)} vs {len(task.binned_spikes)}"
            )

        y = np.asarray(task.binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()

        if save_path is None:
            paths = self.get_gam_save_paths(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
                use_neural_coupling=use_neural_coupling,
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
        use_neural_coupling: bool = False,
    ) -> Dict:
        """Cross-validated variance explained for one unit."""
        task.collect_data(exists_ok=True)

        effective_cv = cv_mode if cv_mode is not None else self.cv_mode

        if save_path is None:
            paths = self.get_gam_save_paths(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
                use_neural_coupling=use_neural_coupling,
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

        design_df, gam_groups = self._get_design_for_unit(
            task, unit_idx, use_neural_coupling=use_neural_coupling
        )

        if len(design_df) != len(task.binned_spikes):
            raise ValueError(
                f"design / binned_spikes row mismatch: "
                f"{len(design_df)} vs {len(task.binned_spikes)}"
            )

        y = np.asarray(task.binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()
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
        load_only: bool = False,
        fit_kwargs: Optional[Dict] = None,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
        plot_cdf: bool = True,
        log_x: bool = False,
        use_neural_coupling: bool = False,
    ) -> Optional[List[float]]:
        """Run crossval for every neuron in the session."""
        task.collect_data(exists_ok=True)
        effective_cv = cv_mode if cv_mode is not None else self.cv_mode

        if unit_indices is None:
            unit_indices = list(range(task.num_neurons))

        if fit_kwargs is None:
            fit_kwargs = dict(
                l1_groups=[], max_iter=1000, tol=1e-6, verbose=False, save_path=None
            )

        all_r2 = []
        for unit_idx in unit_indices:
            paths = self.get_gam_save_paths(
                task, unit_idx,
                gam_results_subdir=gam_results_subdir,
                use_neural_coupling=use_neural_coupling,
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
                cv_mode=effective_cv,
                buffer_samples=buffer_samples,
                verbose=verbose,
                use_neural_coupling=use_neural_coupling,
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
        """Return a GAM analysis helper bound to task."""
        return encoder_gam_helper.BaseEncodingGAMAnalysisHelper(task)

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
        use_neural_coupling: bool = False,
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
            use_neural_coupling=use_neural_coupling,
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
        use_neural_coupling: bool = False,
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
            use_neural_coupling=use_neural_coupling,
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
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_analysis_helper(task).run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=self.lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
            use_neural_coupling=use_neural_coupling,
            cv_mode=cv_mode,
        )


# ===========================================================================
# RNNModel  (GRU + Poisson NLL)
# ===========================================================================

class _PoissonGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.exp(self.readout(h))  # positive rate


class RNNModel(BaseEncodingModel):
    """
    GRU-based neural encoding model with Poisson likelihood.

    Reads:
        task.binned_feats            (n_bins × n_features)
        task.binned_spikes           (n_bins × n_neurons)
        task.get_cv_groups_for_design(...)

    Does NOT use design_df, splines, or spike history — takes
    binned_feats as-is.

    Parameters
    ----------
    hidden_dim : int
    n_epochs   : int
    lr         : float
    device     : str   ('cpu' or 'cuda')
    verbose    : bool
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_epochs: int = 20,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_nll(rate, y, eps: float = 1e-8):
        return torch.mean(rate - y * torch.log(rate + eps))

    def _build_sequences(self, task, unit_idx: int):
        X = task.binned_feats.to_numpy()
        y = task.binned_spikes.iloc[:, unit_idx].to_numpy()

        groups = task.get_cv_groups_for_design(task.binned_feats)
        if groups is None:
            raise ValueError(
                "RNNModel requires per-trial grouping (e.g. new_segment column)."
            )

        X_list, y_list = [], []
        for g in np.unique(groups):
            idx = groups == g
            X_list.append(X[idx])
            y_list.append(y[idx, None])  # shape (T, 1)

        return X_list, y_list, groups

    def _to_tensors(self, X_list, y_list):
        to_t = lambda a, dtype: torch.tensor(a, dtype=dtype, device=self.device)
        return (
            [to_t(x, torch.float32) for x in X_list],
            [to_t(y, torch.float32) for y in y_list],
        )

    def _train(self, model: _PoissonGRU, X_list, y_list):
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for epoch in range(self.n_epochs):
            total = 0.0
            for x, y in zip(X_list, y_list):
                rate = model(x.unsqueeze(0))
                loss = self._poisson_nll(rate, y.unsqueeze(0))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            if self.verbose:
                print(f"[RNNModel] epoch {epoch:>3d}  loss {total:.4f}")

    @torch.no_grad()
    def _evaluate(self, model: _PoissonGRU, X_list, y_list) -> float:
        model.eval()
        y_all = torch.cat(y_list, dim=0)
        mean_rate = torch.mean(y_all)
        ll_model = ll_null = 0.0
        for x, y in zip(X_list, y_list):
            rate = model(x.unsqueeze(0))
            ll_model += torch.sum(y * torch.log(rate + 1e-8) - rate).item()
            ll_null += torch.sum(y * torch.log(mean_rate + 1e-8) - mean_rate).item()
        return float(1.0 - ll_model / ll_null) if ll_null != 0.0 else float("nan")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, task, unit_idx: int, **kwargs) -> _PoissonGRU:
        """Train on all data and return the fitted model."""
        task.collect_data(exists_ok=True)

        X_list, y_list, _ = self._build_sequences(task, unit_idx)
        X_list, y_list = self._to_tensors(X_list, y_list)

        model = _PoissonGRU(X_list[0].shape[1], self.hidden_dim).to(self.device)
        self._train(model, X_list, y_list)
        return model

    def crossval(
        self,
        task,
        unit_idx: int,
        *,
        n_folds: int = 5,
        **kwargs,
    ) -> Dict:
        """GroupKFold cross-validation; returns mean and per-fold pseudo-R²."""
        task.collect_data(exists_ok=True)

        X = task.binned_feats.to_numpy()
        y = task.binned_spikes.iloc[:, unit_idx].to_numpy()
        groups = task.get_cv_groups_for_design(task.binned_feats)
        if groups is None:
            raise ValueError("RNNModel crossval requires per-trial grouping.")

        gkf = GroupKFold(n_splits=n_folds)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            if self.verbose:
                print(f"[RNNModel] Fold {fold}")

            def _make_seq(idx_set):
                Xl, yl = [], []
                for g in np.unique(groups[idx_set]):
                    mask = groups == g
                    Xl.append(X[mask])
                    yl.append(y[mask, None])
                return Xl, yl

            X_tr, y_tr = self._to_tensors(*_make_seq(train_idx))
            X_te, y_te = self._to_tensors(*_make_seq(test_idx))

            model = _PoissonGRU(X_tr[0].shape[1], self.hidden_dim).to(self.device)
            self._train(model, X_tr, y_tr)

            r2 = self._evaluate(model, X_te, y_te)
            if self.verbose:
                print(f"[RNNModel] Fold {fold}  pseudo-R² {r2:.4f}")
            scores.append(r2)

        return {"mean_pseudo_r2": float(np.mean(scores)), "all_folds": scores}
