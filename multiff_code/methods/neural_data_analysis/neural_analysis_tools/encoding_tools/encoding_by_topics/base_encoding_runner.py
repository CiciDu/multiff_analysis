"""Base class for encoding runners with shared Poisson GAM modeling logic."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    gam_variance_explained,
    one_ff_gam_fit,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    FitResult,
    GroupSpec,
)


class BaseEncodingRunner:
    """
    Base class for encoding runners with Poisson GAM modeling.

    Subclasses must implement:
    - get_design_for_unit(unit_idx)
    - get_binned_spikes()
    - get_gam_groups(unit_idx, lam_f, lam_g, lam_h, lam_p)
    - get_gam_save_paths(unit_idx, ...)
    - num_neurons
    - _get_save_dir()
    - _collect_data(exists_ok)
    - _gam_results_subdir()  e.g. "stop_gam_results", "pn_gam_results"
    """

    def __init__(self, bin_width: float = 0.04):
        self.bin_width = bin_width

    def _gam_results_subdir(self) -> str:
        """Subdir under save_dir for GAM results (e.g. stop_gam_results)."""
        raise NotImplementedError

    def fit_poisson_gam(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
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
        self._collect_data(exists_ok=True)
        design_df = self.get_design_for_unit(unit_idx)
        binned_spikes = self.get_binned_spikes()
        n_rows = len(design_df)
        if n_rows != len(binned_spikes):
            raise ValueError(
                f"design and binned_spikes row count mismatch: {n_rows} vs {len(binned_spikes)}"
            )
        y = np.asarray(binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()

        groups, _ = self.get_gam_groups(
            unit_idx=unit_idx, lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p
        )
        if save_path is None:
            paths = self.get_gam_save_paths(
                unit_idx=unit_idx,
                lam_f=lam_f,
                lam_g=lam_g,
                lam_h=lam_h,
                lam_p=lam_p,
            )
            save_path = paths["fit_save_path"]

        return one_ff_gam_fit.fit_poisson_gam(
            design_df=design_df,
            y=y,
            groups=groups,
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

    def crossval_variance_explained(
        self,
        unit_idx: int,
        groups: List[GroupSpec],
        *,
        n_folds: int = 5,
        random_state: int = 0,
        fit_kwargs: Optional[Dict] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        save_metadata: Optional[Dict] = None,
        verbose: bool = True,
        dt: Optional[float] = None,
        cv_mode: Optional[str] = "blocked_time_buffered",
        buffer_samples: int = 20,
        cv_groups=None,
    ) -> Dict:
        """Run crossval_variance_explained for one unit."""
        maybe_loaded = gam_variance_explained.maybe_load_saved_crossval(
            save_path=save_path,
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        if maybe_loaded is not None:
            if verbose:
                print(f"Loaded cached cross-validation results from: {save_path}")
            return maybe_loaded

        self._collect_data(exists_ok=True)
        design_df = self.get_design_for_unit(unit_idx)
        binned_spikes = self.get_binned_spikes()
        n_rows = len(design_df)
        if n_rows != len(binned_spikes):
            raise ValueError(
                f"design and binned_spikes row count mismatch: {n_rows} vs {len(binned_spikes)}"
            )
        y = np.asarray(binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()
        if dt is None:
            dt = float(self.bin_width)
        if fit_kwargs is None:
            fit_kwargs = {}
        meta = dict(save_metadata or {}, unit_idx=unit_idx)
        return gam_variance_explained.crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=design_df,
            y=y,
            groups=groups,
            dt=dt,
            n_folds=n_folds,
            random_state=random_state,
            fit_kwargs=fit_kwargs,
            save_path=save_path,
            load_if_exists=load_if_exists,
            save_metadata=meta,
            verbose=verbose,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            cv_groups=cv_groups,
        )

    def crossval_variance_explained_all_neurons(
        self,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
        n_folds: int = 5,
        load_if_exists: bool = True,
        load_only: bool = False,
        fit_kwargs: Optional[Dict] = None,
        cv_mode: Optional[str] = "blocked_time_buffered",
        buffer_samples: int = 20,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
        plot_cdf: bool = True,
        log_x: bool = False,
    ) -> List[float]:
        """Run crossval_variance_explained for all neurons."""
        if load_if_exists:
            all_neuron_r2 = self._try_load_variance_explained_for_all_neurons(
                lam_f=lam_f,
                lam_g=lam_g,
                lam_h=lam_h,
                lam_p=lam_p,
                verbose=verbose,
            )
            if all_neuron_r2 is not None:
                if verbose:
                    print(
                        "Number of neurons with cached cross-validation results retrieved:",
                        len(all_neuron_r2),
                    )
                return all_neuron_r2
            elif verbose:
                print(
                    "No cached cross-validation variance explained results found for all neurons"
                )

        if load_only:
            if verbose:
                print("Load only mode is enabled, returning None")
            return None

        self._collect_data(exists_ok=True)
        binned_spikes = self.get_binned_spikes()
        n_neurons = binned_spikes.shape[1]
        if unit_indices is None:
            unit_indices = list(range(n_neurons))
        if fit_kwargs is None:
            fit_kwargs = dict(
                l1_groups=[],
                max_iter=1000,
                tol=1e-6,
                verbose=False,
                save_path=None,
            )

        paths = self.get_gam_save_paths(
            unit_idx=unit_indices[0],
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
        )
        base = paths["base"]
        lam_suffix = paths["lam_suffix"]

        all_neuron_r2 = []
        for unit_idx in unit_indices:
            groups, _ = self.get_gam_groups(
                unit_idx=unit_idx,
                lam_f=lam_f,
                lam_g=lam_g,
                lam_h=lam_h,
                lam_p=lam_p,
            )
            outdir = base / f"neuron_{unit_idx}"
            (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
            cv_save_path = str(outdir / "cv_var_explained" / f"{lam_suffix}.pkl")
            cv_res = self.crossval_variance_explained(
                unit_idx=unit_idx,
                groups=groups,
                n_folds=n_folds,
                fit_kwargs=fit_kwargs,
                save_path=cv_save_path,
                load_if_exists=load_if_exists,
                cv_mode=cv_mode,
                buffer_samples=buffer_samples,
                verbose=verbose,
            )
            if verbose:
                print(cv_res["mean_classical_r2"], cv_res["mean_pseudo_r2"])
            all_neuron_r2.append(cv_res["mean_pseudo_r2"])

        if plot_cdf:
            gam_variance_explained.plot_variance_explained_cdf(
                all_neuron_r2, log_x=log_x
            )
        return all_neuron_r2

    def run_category_contributions_and_penalty_tuning_all_neurons(
        self,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        buffer_samples: int = 20,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Run category contributions, penalty tuning, and backward elimination per neuron.

        Delegates to run_category_variance_contributions, run_penalty_tuning,
        and run_backward_elimination for each unit. Subclasses must implement these.
        """
        self._collect_data(exists_ok=True)
        n_neurons = self.num_neurons
        if unit_indices is None:
            unit_indices = list(range(n_neurons))

        lam_cfg = lambda_config or {}
        for unit_idx in unit_indices:
            if verbose:
                print(
                    f"  [neuron {unit_idx}] Running category contributions, "
                    "penalty tuning, backward elimination"
                )
            try:
                self.run_category_variance_contributions(
                    unit_idx,
                    lambda_config=lam_cfg,
                    n_folds=n_folds,
                    buffer_samples=buffer_samples,
                    load_if_exists=load_if_exists,
                )
                self.run_penalty_tuning(
                    unit_idx,
                    lambda_config=lam_cfg,
                    n_folds=n_folds,
                    load_if_exists=load_if_exists,
                )
                self.run_backward_elimination(
                    unit_idx,
                    lambda_config=lam_cfg,
                    n_folds=backward_n_folds,
                    alpha=alpha,
                    load_if_exists=load_if_exists,
                )
            except Exception as e:
                if verbose:
                    print(f"  [WARN] neuron {unit_idx}: {e}")

    def _try_load_variance_explained_for_all_neurons(
        self,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Optional[List[float]]:
        """Try to load variance explained for all neurons."""
        all_neuron_r2 = []
        for unit_idx in range(self.num_neurons):
            try:
                paths_i = self.get_gam_save_paths(
                    unit_idx=unit_idx,
                    lam_f=lam_f,
                    lam_g=lam_g,
                    lam_h=lam_h,
                    lam_p=lam_p,
                )
                base = paths_i["base"]
                outdir = base / f"neuron_{unit_idx}"
                (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
                lam_suffix = paths_i["lam_suffix"]
                cv_save_path = str(outdir / "cv_var_explained" / f"{lam_suffix}.pkl")

                maybe_loaded = gam_variance_explained.maybe_load_saved_crossval(
                    save_path=cv_save_path,
                    load_if_exists=load_if_exists,
                    verbose=verbose,
                )
                if maybe_loaded is not None:
                    if verbose:
                        print(
                            f"Loaded cached cross-validation results from: {cv_save_path}"
                        )
                    all_neuron_r2.append(maybe_loaded["mean_pseudo_r2"])
                else:
                    return None
            except Exception as e:
                if verbose:
                    print(f"Try to load variance explained for unit {unit_idx} failed: {e}")
                return None
        return all_neuron_r2
