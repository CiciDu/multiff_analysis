"""Base class for encoding runners with shared Poisson GAM modeling logic."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
from pathlib import Path


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


from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoder_gam_helper
)

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing



DEFAULT_LAMBDA_CONFIG = {
    "lam_f": 100.0,
    "lam_g": 10.0,
    "lam_h": 10.0,
    "lam_p": 10.0,
}

DEFAULT_LAM_GRID = {
    "lam_f": [50, 100, 200],
    "lam_g": [50, 100, 200],
    "lam_h": [5, 10, 30],
    "lam_p": [10],
}


class BaseEncodingRunner:
    """
    Base class for encoding runners with Poisson GAM modeling.

    Subclasses must implement:
    - get_design_for_unit(unit_idx)
    - get_gam_groups(unit_idx)
    - get_gam_save_paths(unit_idx, ...)
    - num_neurons
    - _get_save_dir()
    - collect_data(exists_ok)
    - get_gam_results_subdir()  e.g. "stop_gam_results", "pn_gam_results"
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width: float = 0.04,
        encoder_prs=None,
        lambda_config: Optional[Dict[str, float]] = None,
        cv_mode: Optional[str] = "blocked_time_buffered",
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width

        self.num_neurons = neural_data_processing.find_num_neurons(
            self.raw_data_folder_path)

        # Optional: multiff_encoding_params.MultiFFParams for encoding design (temporal + tuning)
        self.encoder_prs = encoder_prs if encoder_prs is not None else multiff_encoding_params.default_prs()
        self.binrange_dict = self.encoder_prs.binrange

        # will be filled during setup
        self.binned_feats = None
        self.binned_spikes = None  # (n_bins x n_neurons) for Poisson GAM
        self.hist_meta = None
        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

        self.lam_grid = DEFAULT_LAM_GRID
        
        self.lambda_config = lambda_config if lambda_config is not None else DEFAULT_LAMBDA_CONFIG
        self.lam_f = self.lambda_config['lam_f']
        self.lam_g = self.lambda_config['lam_g']
        self.lam_h = self.lambda_config['lam_h']
        self.lam_p = self.lambda_config['lam_p']

        # Cross-validation strategy (e.g. 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold')
        self.cv_mode = cv_mode
        

    def get_gam_results_subdir(self) -> str:
        """Subdir under save_dir for GAM results (e.g. stop_gam_results)."""
        raise NotImplementedError

    @staticmethod
    def _coupling_subdir(use_neural_coupling: bool) -> str:
        """Return subfolder name: 'coupling' or 'no_coupling'."""
        return "coupling" if use_neural_coupling else "no_coupling"

    def get_cv_groups_for_design(self, design_df: pd.DataFrame):
        """
        Per-sample labels for GroupKFold, aligned with design_df rows.

        Tries, in order: ``new_segment`` on the design; ``trial_ids`` on the runner
        (same length as design); ``event_id`` / ``new_segment`` on ``meta_df_used``.
        """
        n = len(design_df)
        if n == 0:
            return None
        if "new_segment" in design_df.columns:
            return np.asarray(design_df["new_segment"].to_numpy())
        trial_ids = getattr(self, "trial_ids", None)
        if trial_ids is not None:
            if hasattr(trial_ids, "reset_index"):
                arr = trial_ids.reset_index(drop=True).to_numpy()
            else:
                arr = np.asarray(trial_ids)
            if len(arr) == n:
                return arr
        meta = getattr(self, "meta_df_used", None)
        if meta is not None and len(meta) == n:
            if "event_id" in meta.columns:
                return np.asarray(meta["event_id"].to_numpy())
            if "new_segment" in meta.columns:
                return np.asarray(meta["new_segment"].to_numpy())
        return None

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
        self.collect_data(exists_ok=True)
        self.get_design_for_unit(unit_idx, use_neural_coupling=use_neural_coupling)
        binned_spikes = self.binned_spikes
        n_rows = len(self.design_df)
        if n_rows != len(binned_spikes):
            raise ValueError(
                f"design and binned_spikes row count mismatch: {n_rows} vs {len(binned_spikes)}"
            )
        y = np.asarray(
            binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()

        if save_path is None:
            paths = self.get_gam_save_paths(
                unit_idx=unit_idx,
                use_neural_coupling=use_neural_coupling,
            )
            save_path = paths["fit_save_path"]

        return one_ff_gam_fit.fit_poisson_gam(
            design_df=self.design_df,
            y=y,
            groups=self.gam_groups,
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
        *,
        n_folds: int = 5,
        random_state: int = 0,
        fit_kwargs: Optional[Dict] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        save_metadata: Optional[Dict] = None,
        verbose: bool = True,
        dt: Optional[float] = None,
        # 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold'
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
        cv_groups=None,
        use_neural_coupling: bool = False,
    ) -> Dict:
        """Run crossval_variance_explained for one unit."""
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        if save_path is None:
            paths_i = self.get_gam_save_paths(
                unit_idx=unit_idx,
                use_neural_coupling=use_neural_coupling,
                cv_mode=cv_mode,
                n_folds=n_folds,
            )
            save_path = paths_i["cv_save_path"]

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
        self.get_design_for_unit(unit_idx, use_neural_coupling=use_neural_coupling)
        binned_spikes = self.binned_spikes
        n_rows = len(self.design_df)
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
        if cv_mode == "group_kfold" and cv_groups is None:
            cv_groups = self.get_cv_groups_for_design(self.design_df)
        if cv_mode == "group_kfold" and cv_groups is None:
            raise ValueError(
                "cv_mode='group_kfold' requires per-sample group labels. Pass cv_groups, "
                "or use a design with a 'new_segment' column, or set runner.trial_ids / "
                "runner.meta_df_used['event_id'] with one label per design row."
            )
        meta = dict(save_metadata or {}, unit_idx=unit_idx)
        return gam_variance_explained._crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=self.design_df,
            y=y,
            groups=self.gam_groups,
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
    ) -> List[float]:
        """Run crossval_variance_explained for all neurons."""
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        if load_if_exists:
            all_neuron_r2 = self.try_load_variance_explained_for_all_neurons(
                n_folds=n_folds,
                verbose=verbose,
                use_neural_coupling=use_neural_coupling,
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

        all_neuron_r2 = []

        for unit_idx in unit_indices:
            paths_i = self.get_gam_save_paths(
                unit_idx=unit_idx,
                use_neural_coupling=use_neural_coupling,
                cv_mode=cv_mode,
                n_folds=n_folds,
            )
            cv_save_path = paths_i["cv_save_path"]
            cv_res = self.crossval_variance_explained(
                unit_idx=unit_idx,
                n_folds=n_folds,
                fit_kwargs=fit_kwargs,
                save_path=cv_save_path,
                load_if_exists=load_if_exists,
                cv_mode=cv_mode,
                buffer_samples=buffer_samples,
                verbose=verbose,
                use_neural_coupling=use_neural_coupling,
            )
            if verbose:
                print(cv_res["mean_classical_r2"], cv_res["mean_pseudo_r2"])
            all_neuron_r2.append(cv_res["mean_pseudo_r2"])

        if plot_cdf:
            gam_variance_explained.plot_variance_explained_cdf(
                all_neuron_r2, log_x=log_x
            )
        return all_neuron_r2
    
    def temp_function(self, n_folds: int = 5):
        pass

    def run_category_contributions_etc_for_all_neurons(
        self,
        *,
        n_folds: int = 5,
        buffer_samples: int = 20,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> None:
        """
        Run category contributions, penalty tuning, and backward elimination per neuron.

        Delegates to run_category_variance_contributions, run_penalty_tuning,
        and run_backward_elimination for each unit. Subclasses must implement these.
        """
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        self.collect_data(exists_ok=True)
        n_neurons = self.num_neurons
        if unit_indices is None:
            unit_indices = list(range(n_neurons))

        for unit_idx in unit_indices:
            if verbose:
                print(
                    f"  [neuron {unit_idx}] Running category contributions, "
                    "penalty tuning, backward elimination"
                )
            try:
                self.run_category_variance_contributions(
                    unit_idx,
                    n_folds=n_folds,
                    buffer_samples=buffer_samples,
                    load_if_exists=load_if_exists,
                    use_neural_coupling=use_neural_coupling,
                    cv_mode=cv_mode,
                )

                # get tuning curve
                self.get_design_for_unit(
                    unit_idx,
                    use_neural_coupling=use_neural_coupling,
                )

                # self.run_penalty_tuning(
                #     unit_idx,
                #     n_folds=n_folds,
                #     load_if_exists=load_if_exists,
                # )
                # self.run_backward_elimination(
                #     unit_idx,
                #     n_folds=backward_n_folds,
                #     alpha=alpha,
                #     load_if_exists=load_if_exists,
                # )
            except Exception as e:
                if verbose:
                    print(f"  [WARN] neuron {unit_idx}: {e}")

    def try_load_variance_explained_for_all_neurons(
        self,
        *,
        n_folds: int = 5,
        load_if_exists: bool = True,
        verbose: bool = True,
        use_neural_coupling: bool = False,
    ) -> Optional[List[float]]:
        """Try to load variance explained for all neurons."""
        all_neuron_r2 = []
        for unit_idx in range(self.num_neurons):
            try:
                paths_i = self.get_gam_save_paths(
                    unit_idx=unit_idx,
                    use_neural_coupling=use_neural_coupling,
                    cv_mode=self.cv_mode,
                    n_folds=n_folds,
                )
                cv_save_path = paths_i["cv_save_path"]

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

    def get_design_for_unit(
        self,
        unit_idx: int,
        *,
        use_neural_coupling: bool = True,
    ) -> pd.DataFrame:
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

        cross_neurons = None
        if use_neural_coupling:
            cross_neurons = [c for c in self.spk_colnames.keys() if c != target_col]


        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=self.binned_feats.reset_index(drop=True),
            colnames=self.spk_colnames,
            X_hist=self.X_hist,
            target_col=target_col,
            include_self=True,
            cross_neurons=cross_neurons,
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

        if use_neural_coupling:
            # coupling groups are named cpl_* in build_hist_meta_from_colnames
            hist_groups = (self.hist_meta or {}).get('groups', {}) if hasattr(self, 'hist_meta') else {}
            self.var_categories['coupling_vars'] = sorted(
                [g for g in hist_groups.keys() if str(g).startswith('cpl_')]
            )
        
        self.gam_groups = self.get_gam_groups()

        return self.design_df
    

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir()) / 'encoding_design'
        return {
            'binned_feats': save_dir / 'binned_feats.pkl',
            'binned_spikes': save_dir / 'binned_spikes.pkl',
            'binrange_dict': save_dir / 'binrange_dict.pkl',
            'bin_df': save_dir / 'bin_df.pkl',
            'temporal_meta': save_dir / 'temporal_meta.pkl',
            'tuning_meta': save_dir / 'tuning_meta.pkl',
            'X_hist': save_dir / 'X_hist.pkl',
            'spk_colnames': save_dir / 'spk_colnames.pkl',
            'meta_df_used': save_dir / 'meta_df_used.pkl',
            'trial_ids': save_dir / 'trial_ids.pkl',
        }
   
    
    def reduce_binned_feats(self):
        #self.binned_feats = process_encode_design.reduce_encoding_design(self.binned_feats)
        pass



    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths['binned_feats'].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            'binned_feats': self.binned_feats,
            'binned_spikes': self.binned_spikes,
            'binrange_dict': self.binrange_dict,
            'temporal_meta': self.temporal_meta,
            'tuning_meta': self.tuning_meta,
            'bin_df': getattr(self, 'bin_df', None),
            'X_hist': getattr(self, 'X_hist', None),
            'spk_colnames': getattr(self, 'spk_colnames', None),
            'meta_df_used': getattr(self, 'meta_df_used', None),
            'trial_ids': getattr(self, 'trial_ids', None),
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
            with open(paths['temporal_meta'], 'rb') as f:
                self.temporal_meta = pickle.load(f)

            with open(paths['tuning_meta'], 'rb') as f:
                self.tuning_meta = pickle.load(f)

            with open(paths['bin_df'], 'rb') as f:
                self.bin_df = pickle.load(f)

            with open(paths['X_hist'], 'rb') as f:
                self.X_hist = pickle.load(f)

            with open(paths['spk_colnames'], 'rb') as f:
                self.spk_colnames = pickle.load(f)
                self.spike_cols = list(self.spk_colnames.keys())

            if paths['meta_df_used'].exists():
                with open(paths['meta_df_used'], 'rb') as f:
                    self.meta_df_used = pickle.load(f)

            if paths['trial_ids'].exists():
                with open(paths['trial_ids'], 'rb') as f:
                    self.trial_ids = pickle.load(f)

            print('Loaded cached design matrices from:', paths['binned_feats'])

            self.structured_meta_groups = encoding_design_utils.build_structured_meta_groups(
                hist_meta=self.hist_meta, temporal_meta=self.temporal_meta, tuning_meta=self.tuning_meta)
            
            return True

        except Exception as e:
            print(
                f'WARNING: could not load design matrices: '
                f'{e}'
            )
            return False

    def get_gam_groups(
        self,
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
        self.gam_groups = encoding_design_utils.build_gam_groups_from_meta(
            self.structured_meta_groups,
            lam_f=self.lambda_config['lam_f'],
            lam_g=self.lambda_config['lam_g'],
            lam_h=self.lambda_config['lam_h'],
            lam_p=self.lambda_config['lam_p'],
        )

        # -------------------------
        # Validate coverage
        # -------------------------
        if hasattr(self, 'design_df'):
            encoding_design_utils._validate_design_columns(self.design_df, self.gam_groups)  
            
        self.var_categories = encoder_gam_helper.add_other_category_from_groups(self.var_categories, self.gam_groups)
        return self.gam_groups

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

        # self.X_hist, self.spk_colnames, _ = spike_history.reduce_history_via_full_design(
        #     X_pruned=self.binned_feats,
        #     X_hist=self.X_hist,
        #     colnames=self.spk_colnames,
        #     meta_groups=None,
        #     spike_cols=list(self.spk_colnames.keys()),
        #     reduce_design_kwargs={},
        # )
        

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
        ensure_dirs: bool = True,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
        n_folds: Optional[int] = 10,
    ) -> dict:
        coupling_subdir = self._coupling_subdir(use_neural_coupling)
        base = Path(self._get_save_dir()) / self.get_gam_results_subdir() / coupling_subdir
        if ensure_dirs:
            base.mkdir(parents=True, exist_ok=True)
        lambda_config = dict(lam_f=self.lambda_config['lam_f'], lam_g=self.lambda_config['lam_g'],
                             lam_h=self.lambda_config['lam_h'], lam_p=self.lambda_config['lam_p'])
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=lambda_config)
        outdir = base / f"neuron_{unit_idx}"
        if ensure_dirs:
            (outdir / "fit_results").mkdir(parents=True, exist_ok=True)
            (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)

        # CV artifacts live under cv_var_explained/{cv_mode}/ when cv_mode is set.
        effective_cv_mode = (
            cv_mode if cv_mode is not None else getattr(self, "cv_mode", None)
        )
        cv_dir = outdir / "cv_var_explained"
        if effective_cv_mode:
            cv_dir = cv_dir / str(effective_cv_mode)
            if ensure_dirs:
                cv_dir.mkdir(parents=True, exist_ok=True)
        cv_filename_suffix = lam_suffix
        if effective_cv_mode and n_folds is not None:
            cv_filename_suffix = f"{lam_suffix}_nfolds{n_folds}"
        return {
            "base": base,
            "outdir": outdir,
            "lambda_config": lambda_config,
            "lam_suffix": lam_suffix,
            "fit_save_path": str(outdir / "fit_results" / f"{lam_suffix}.pkl"),
            "cv_save_path": str(cv_dir / f"{cv_filename_suffix}.pkl"),
        }

    def run_category_variance_contributions(
        self,
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
        """
        Run leave-one-category-out CV analysis, mirroring one_ff_gam pipeline.
        """
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_gam_analysis_helper().run_category_variance_contributions(
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
        unit_idx: int,
        *,
        n_folds: int = 5,
        group_name_map: Optional[Dict[str, List[str]]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        """
        Run grid search over group penalties, mirroring one_ff_gam pipeline.
        """
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_gam_analysis_helper().run_penalty_tuning(
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
        unit_idx: int,
        *,
        alpha: float = 0.05,
        n_folds: int = 10,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        """
        Run backward elimination over stop GAM groups.

        Results are automatically saved to:
        - ``{save_dir}/{results_subdir}/{coupling|no_coupling}/neuron_{unit_idx}/backward_elimination/{cv_mode}/{lam_suffix}.pkl``
        - ``{save_dir}/{results_subdir}/{coupling|no_coupling}/neuron_{unit_idx}/backward_elimination/{cv_mode}/history.csv``
        """
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
        return self._get_gam_analysis_helper().run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=self.lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
            use_neural_coupling=use_neural_coupling,
            cv_mode=cv_mode,
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
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
        cv_groups=None,
        use_neural_coupling: bool = False,
    ) -> Dict:
        """
        Perform cross-validation for tuning curve coefficients.
    
        Results are automatically saved to:
        - ``{save_dir}/{results_subdir}/{coupling|no_coupling}/neuron_{unit_idx}/cv_tuning_coef/{cv_mode}/{lam_suffix}.pkl``
        """
        cv_mode = cv_mode if cv_mode is not None else self.cv_mode
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
            use_neural_coupling=use_neural_coupling,
        )

    def _get_gam_analysis_helper(self):
        if not hasattr(self, '_gam_analysis_helper') or self._gam_analysis_helper is None:
            self._gam_analysis_helper = (
                encoder_gam_helper.BaseEncodingGAMAnalysisHelper(
                    self,
                    var_categories=self.var_categories,
                )
            )
        return self._gam_analysis_helper



    def get_max_temporal_dependency(self, verbose: bool = True) -> Dict:
        """
        Summarise the maximum temporal dependency of the encoding model.

        Reads from (in decreasing priority):
        1. ``hist_meta['basis_info']`` lags – exact lags after ``get_design_for_unit``
        2. ``temporal_meta['basis_info']`` lags – exact lags after ``collect_data``
        3. ``binrange_dict`` / ``encoder_prs`` – always available as fallback

        Returns
        -------
        dict
            spike_hist_window : (t_min, t_max) seconds – causal self-history window
            event_windows     : {event_name: (t_min, t_max)} – per-event kernel spans
            max_causal_s      : max lookback = max(|spike_hist_max|, |event_t_min|) s
            max_acausal_s     : max lookahead = max event t_max s  (0 if none)
            overall_max_s     : max(max_causal_s, max_acausal_s) s
            overall_max_bins  : overall_max_s / bin_width  (rounded to int)
        """
        # ── 1. Spike-history window ──────────────────────────────────────────
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

        # ── 2. Event-kernel windows ──────────────────────────────────────────
        ev_min_fallback: Optional[float] = None
        ev_max_fallback: Optional[float] = None

        temporal_meta = getattr(self, 'temporal_meta', None) or {}
        basis_info_tm = temporal_meta.get('basis_info') or {}
        first_tm_info = next(iter(basis_info_tm.values()), None)
        if first_tm_info is not None:
            lags = first_tm_info.get('lags')
            if lags is not None and len(lags):
                ev_min_fallback = float(lags[0])
                ev_max_fallback = float(lags[-1])

        if ev_min_fallback is None:
            for key in ('t_stop', 't_event'):
                br = self.binrange_dict.get(key)
                if br is not None:
                    ev_min_fallback = float(br[0])
                    ev_max_fallback = float(br[1])
                    break

        if ev_min_fallback is None:
            ev_min_fallback = -float(getattr(self.encoder_prs, 'pre_event', 0.3))
            ev_max_fallback = float(getattr(self.encoder_prs, 'post_event', 0.3))

        event_groups = temporal_meta.get('groups') or {}
        event_windows: Dict[str, tuple] = {
            name: (ev_min_fallback, ev_max_fallback)
            for name in event_groups
        }
        if not event_windows and ev_min_fallback is not None:
            event_windows['(default_event)'] = (ev_min_fallback, ev_max_fallback)

        # ── 3. Derive summary scalars ────────────────────────────────────────
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
            def _fmt(t: Optional[float]) -> str:
                return f"{t * 1000:.1f} ms" if t is not None else "N/A"

            print("── Temporal dependency summary ──")
            print(f"  Spike history :  {_fmt(sh_min)} → {_fmt(sh_max)}  (causal lookback)")
            for name, (t0, t1) in event_windows.items():
                print(f"  Event [{name:>20s}]:  {_fmt(t0)} → {_fmt(t1)}")
            print(f"  Max causal    :  {_fmt(max_causal)}")
            print(f"  Max acausal   :  {_fmt(max_acausal)}")
            print(f"  Overall max   :  {_fmt(overall)}  ({overall_bins} bins @ {self.bin_width*1000:.0f} ms)")

        return result