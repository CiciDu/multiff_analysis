"""Base class for encoding runners with shared Poisson GAM modeling logic."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import os

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    gam_variance_explained,
    one_ff_gam_fit,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    FitResult,
    GroupSpec,
)


from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event

from neural_data_analysis.design_kits.design_by_segment import spike_history

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encode_stops_gam_helper,
    encode_stops_utils,
    encode_pn_utils
)

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing


class BaseEncodingRunner:
    """
    Base class for encoding runners with Poisson GAM modeling.

    Subclasses must implement:
    - get_design_for_unit(unit_idx)
    - get_gam_groups(unit_idx, lam_f, lam_g, lam_h, lam_p)
    - get_gam_save_paths(unit_idx, ...)
    - num_neurons
    - _get_save_dir()
    - collect_data(exists_ok)
    - get_gam_results_subdir()  e.g. "stop_gam_results", "pn_gam_results"
    """

    def __init__(self, raw_data_folder_path,
                 bin_width: float = 0.04,
                 encoder_prs=None,
                 ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width

        self.num_neurons = neural_data_processing.find_num_neurons(
            self.raw_data_folder_path)
        
        # Optional: multiff_encoding_params.StopParams for encoding design (temporal + tuning)
        self.encoder_prs = encoder_prs if encoder_prs is not None else multiff_encoding_params.default_prs()
        self.binrange_dict = self.encoder_prs.binrange

        # will be filled during setup
        self.binned_feats = None
        self.binned_spikes = None  # (n_bins x n_neurons) for Poisson GAM
        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)



    def get_gam_results_subdir(self) -> str:
        """Subdir under save_dir for GAM results (e.g. stop_gam_results)."""
        raise NotImplementedError

    def _encoding_design_kwargs(self) -> Dict:

        mode = getattr(self.encoder_prs, 'tuning_feature_mode', None)
        return {
            'use_boxcar': bool(
                mode in ('boxcar_only', 'raw_plus_boxcar')
            ),
            'tuning_feature_mode': mode,
            'binrange_dict': self.binrange_dict,
            'n_basis': getattr(self.encoder_prs, 'default_n_basis', 20),
            't_min': -getattr(self.encoder_prs, 'pre_event', 0.3),
            't_max': getattr(self.encoder_prs, 'post_event', 0.3),
            'tuning_n_bins': getattr(self.encoder_prs, 'tuning_n_bins', 10),
        }

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
        self.collect_data(exists_ok=True)
        self.get_design_for_unit(unit_idx)
        binned_spikes = self.binned_spikes
        n_rows = len(self.design_df)
        if n_rows != len(binned_spikes):
            raise ValueError(
                f"design and binned_spikes row count mismatch: {n_rows} vs {len(binned_spikes)}"
            )
        y = np.asarray(
            binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()

        groups = self.get_gam_groups(
            lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p
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
            design_df=self.design_df,
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
        # can be 'blocked_time_buffered', 'blocked_time', 'group_kfold'
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
                print(
                    f"Loaded cached cross-validation results from: {save_path}")
            return maybe_loaded

        self.collect_data(exists_ok=True)
        self.get_design_for_unit(unit_idx)
        binned_spikes = self.binned_spikes
        n_rows = len(design_df)
        if n_rows != len(binned_spikes):
            raise ValueError(
                f"design and binned_spikes row count mismatch: {n_rows} vs {len(binned_spikes)}"
            )
        y = np.asarray(
            binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()
        if dt is None:
            dt = float(self.bin_width)
        if fit_kwargs is None:
            fit_kwargs = {}
        meta = dict(save_metadata or {}, unit_idx=unit_idx)
        return gam_variance_explained.crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=self.design_df,
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
            all_neuron_r2 = self.try_load_variance_explained_for_all_neurons(
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

        self.collect_data(exists_ok=True)
        binned_spikes = self.binned_spikes
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
            groups = self.get_gam_groups(
                unit_idx=unit_idx,
                lam_f=lam_f,
                lam_g=lam_g,
                lam_h=lam_h,
                lam_p=lam_p,
            )
            outdir = base / f"neuron_{unit_idx}"
            (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
            cv_save_path = str(
                outdir / "cv_var_explained" / f"{lam_suffix}.pkl")
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
        self.collect_data(exists_ok=True)
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

    def try_load_variance_explained_for_all_neurons(
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
                cv_save_path = str(
                    outdir / "cv_var_explained" / f"{lam_suffix}.pkl")

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
                    print(
                        f"Try to load variance explained for unit {unit_idx} failed: {e}")
                return None
        return all_neuron_r2


    
    def _make_structured_meta_groups(self):
        # Restructure meta_groups into one_ff_gam-style categories
        self.structured_meta_groups = encoding_design_utils.build_structured_meta_groups(
            hist_meta=self.hist_meta,
            temporal_meta=self.temporal_meta,
            tuning_meta=self.tuning_meta,
        )


    def get_design_for_unit(self, unit_idx: int) -> pd.DataFrame:
        """
        Build design matrix with spike-history regressors for the given target neuron.

        Spike-history must match the target neuron when predicting its spikes.
        unit_idx maps to binned_spikes column order.
        """
        self._prepare_spike_history_components()
        if unit_idx < 0 or unit_idx >= len(self.spk_colnames):
            raise IndexError(
                f'unit_idx {unit_idx} out of range [0, {len(self.spk_colnames)})'
            )
        target_col = list(self.spk_colnames.keys())[unit_idx]

        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=self.binned_feats.reset_index(drop=True),
            colnames=self.spk_colnames,
            X_hist=self.X_hist,
            target_col=target_col,
            include_self=True,
            cross_neurons=None,
            meta_groups=None,
        )

        # Drop constant columns except 'const'
        const_cols_to_drop = [
            c for c in design_df.columns
            if c != 'const' and design_df[c].nunique() <= 1
        ]
        self.design_df = design_df.drop(columns=const_cols_to_drop)

        self.make_hist_meta_for_unit(unit_idx)
        self._make_structured_meta_groups()
        self.get_gam_groups()

        return


    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir()) / 'encoding_design'
        return {
            'binned_feats': save_dir / 'binned_feats.pkl',
            'binned_spikes': save_dir / 'binned_spikes.pkl',
            'binrange_dict': save_dir / 'binrange_dict.pkl',
            'structured_meta_groups': save_dir / 'structured_meta_groups.pkl',
            'bin_df': save_dir / 'bin_df.pkl',
            'temporal_meta': save_dir / 'temporal_meta.pkl',
            'tuning_meta': save_dir / 'tuning_meta.pkl',
            'X_hist': save_dir / 'X_hist.pkl',
            'spk_colnames': save_dir / 'spk_colnames.pkl',
        }


    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths['binned_feats'].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            'binned_feats': self.binned_feats,
            'binned_spikes': self.binned_spikes,
            'binrange_dict': self.binrange_dict,
            'structured_meta_groups': self.structured_meta_groups,
            'bin_df': getattr(self, 'bin_df', None),
            'X_hist': getattr(self, 'X_hist', None),
            'spk_colnames': getattr(self, 'spk_colnames', None),
        }

        for key, data in data_to_save.items():
            if key not in paths or data is None:
                continue
            try:
                with open(paths[key], 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[StopEncodingRunner] Saved {key} → {paths[key]}')
            except Exception as e:
                print(
                    f'[StopEncodingRunner] WARNING: could not save {key}: '
                    f'{type(e).__name__}: {e}'
                )

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()
        required = ['binned_feats',
                    'binned_spikes']
        if not all(paths[k].exists() for k in required):
            return False

        try:
            with open(paths['binned_feats'], 'rb') as f:
                self.binned_feats = pickle.load(f)

            with open(paths['binned_spikes'], 'rb') as f:
                self.binned_spikes = pickle.load(f)

            with open(paths['binrange_dict'], 'rb') as f:
                self.binrange_dict = pickle.load(f)

            # structured_meta_groups 
            with open(paths['structured_meta_groups'], 'rb') as f:
                self.structured_meta_groups = pickle.load(f)

            with open(paths['bin_df'], 'rb') as f:
                self.bin_df = pickle.load(f)

            with open(paths['X_hist'], 'rb') as f:
                self.X_hist = pickle.load(f)

            with open(paths['spk_colnames'], 'rb') as f:
                self.spk_colnames = pickle.load(f)
                self.spike_cols = list(self.spk_colnames.keys())

            print('Loaded cached design matrices')

            self.temporal_meta = self.structured_meta_groups['temporal']
            self.tuning_meta = self.structured_meta_groups['tuning']
            self.hist_meta = self.structured_meta_groups['hist']

            return True

        except Exception as e:
            print(
                f'WARNING: could not load design matrices: '
                f'{e}'
            )
            return False


    def get_gam_groups(
        self,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
    ):
        """
        Build GroupSpec list and lambda_config for a unit's design matrix.

        Call after collect_data(). Uses get_design_for_unit(unit_idx) and
        build_stop_gam_groups() with the same lambda roles: lam_f tuning,
        lam_g event, lam_h spike history, lam_p coupling.

        Parameters
        ----------
        unit_idx : int
            Column index for the target neuron.
        lam_f, lam_g, lam_h, lam_p : float
            Penalty parameters.

        Returns
        -------
        groups : List[GroupSpec]
        lambda_config : dict
            For generate_lambda_suffix(lambda_config=lambda_config).
        """
        self.groups = encoding_design_utils.build_gam_groups_from_meta(
            self.structured_meta_groups,
            self.design_df,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
        )
        return self.groups
    
    def _prepare_spike_history_components(self):
        """Compute spike-history components if not yet available (e.g. after cache load)."""
        if hasattr(self, 'X_hist') and self.X_hist is not None:
            return

        if not hasattr(self.pn, 'spikes_df'):
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(
                self.raw_data_folder_path, bin_width=0.04)

        n_basis_hist = getattr(self.encoder_prs, 'default_n_basis', 20)
        spike_hist_t_max = self.binrange_dict['spike_hist'][1]

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

    def make_hist_meta_for_unit(self, unit_idx):
        target_col = list(self.spk_colnames.keys())[unit_idx]
        spike_hist_t_max = self.binrange_dict['spike_hist'][1]
        n_basis_hist = getattr(self.encoder_prs, 'default_n_basis', 20)

        # Build histogram meta separately
        self.hist_meta = encoding_design_utils.build_hist_meta_from_colnames(
            colnames=self.spk_colnames,
            target_col=target_col,
            dt=self.bin_width,
            t_max=spike_hist_t_max,
            n_basis=n_basis_hist,
        )

    def get_gam_save_paths(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
        ensure_dirs: bool = True,
    ) -> dict:
        base = Path(self._get_save_dir()) / self.get_gam_results_subdir()
        if ensure_dirs:
            base.mkdir(parents=True, exist_ok=True)
        lambda_config = dict(lam_f=lam_f, lam_g=lam_g,
                             lam_h=lam_h, lam_p=lam_p)
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=lambda_config)
        outdir = base / f"neuron_{unit_idx}"
        if ensure_dirs:
            (outdir / "fit_results").mkdir(parents=True, exist_ok=True)
            (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
        return {
            "base": base,
            "outdir": outdir,
            "lambda_config": lambda_config,
            "lam_suffix": lam_suffix,
            "fit_save_path": str(outdir / "fit_results" / f"{lam_suffix}.pkl"),
            "cv_save_path": str(outdir / "cv_var_explained" / f"{lam_suffix}.pkl"),
        }


    def run_category_variance_contributions(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        buffer_samples: int = 20,
        category_names: Optional[List[str]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ) -> Dict:
        """
        Run leave-one-category-out CV analysis, mirroring one_ff_gam pipeline.
        """
        return self._get_gam_analysis_helper().run_category_variance_contributions(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            n_folds=n_folds,
            buffer_samples=buffer_samples,
            category_names=category_names,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
        )

    def run_penalty_tuning(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        lam_grid: Optional[Dict[str, List[float]]] = None,
        group_name_map: Optional[Dict[str, List[str]]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ) -> Dict:
        """
        Run grid search over group penalties, mirroring one_ff_gam pipeline.
        """
        return self._get_gam_analysis_helper().run_penalty_tuning(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            n_folds=n_folds,
            lam_grid=lam_grid,
            group_name_map=group_name_map,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
        )

    def run_backward_elimination(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
        n_folds: int = 10,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
    ) -> Dict:
        """
        Run backward elimination over stop GAM groups.

        Results are automatically saved to:
        - ``{save_dir}/{results_subdir}/neuron_{unit_idx}/backward_elimination/{lam_suffix}.pkl``
        - ``{save_dir}/{results_subdir}/neuron_{unit_idx}/backward_elimination/history.csv``
        """
        return self._get_gam_analysis_helper().run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
        )

    def crossval_tuning_curve_coef(
        self,
        unit_idx: int,
        *,
        n_folds: int = 10,
        random_state: int = 0,
        fit_kwargs: Optional[Dict] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        save_metadata: Optional[Dict] = None,
        verbose: bool = True,
        cv_mode: Optional[str] = 'blocked_time_buffered',
        buffer_samples: int = 20,
        cv_groups=None,
    ) -> Dict:
        """
        Perform cross-validation for tuning curve coefficients.

        Results are automatically saved to:
        - ``{save_dir}/{results_subdir}/neuron_{unit_idx}/cv_tuning_coef/{lam_suffix}.pkl``
        """

        return self._get_gam_analysis_helper().crossval_tuning_curve_coef(
            unit_idx=unit_idx,
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


    def _get_gam_analysis_helper(self):
        if not hasattr(self, '_gam_analysis_helper') or self._gam_analysis_helper is None:
            self._gam_analysis_helper = (
                encode_stops_gam_helper.BaseEncodingGAMAnalysisHelper(
                    self,
                    var_categories=self.var_categories,
                )
            )
        return self._gam_analysis_helper
