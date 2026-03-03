import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encode_stops_utils, encoding_design_utils

# Third-party imports
import numpy as np
import pandas as pd



from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    FitResult,
    GroupSpec,
)

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data
)

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encode_stops_gam_helper,
    encode_stops_utils,
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics.base_encoding_runner import (
    BaseEncodingRunner,
)


class StopEncodingRunner(BaseEncodingRunner):
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        t_max=0.20,
        encoder_prs=None,
    ):
        super().__init__(bin_width=bin_width)
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.t_max = t_max
        # Optional: multiff_encoding_params.StopParams for encoding design (temporal + tuning)
        self.encoder_prs = encoder_prs if encoder_prs is not None else multiff_encoding_params.default_prs()

        # will be filled during setup
        self.stop_meta_used = None
        self.binned_feats = None
        self.binned_spikes = None  # (n_bins x n_neurons) for Poisson GAM
        # Spike-history components for per-neuron design (computed on first use)
        self._X_hist = None
        self._colnames = None
        self._basis = None
        self._spike_cols = None
        # tuning bin ranges (from prs or estimated); inspect after run
        self.binrange_dict = self.encoder_prs.binrange

        self.structured_meta_groups = {}
        self._gam_analysis_helper = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

        self.num_neurons = neural_data_processing.find_num_neurons(
            self.raw_data_folder_path)

    def _get_gam_analysis_helper(self):
        if self._gam_analysis_helper is None:
            self._gam_analysis_helper = (
                encode_stops_gam_helper.StopEncodingGAMAnalysisHelper(self)
            )
        return self._gam_analysis_helper

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self, exists_ok=True,
                      # can be 'raw_only', 'boxcar_only', 'raw_plus_boxcar'
                      tuning_feature_mode='boxcar_only'
                      ):
        """
        Collect and prepare data for stop encoding.
        """
        if exists_ok and self._load_design_matrices():
            if not self._has_one_ff_like_terms(self.binned_feats):
                print(
                    '[StopEncodingRunner] Cached encoding design is stale '
                    '(missing one_ff-like temporal/spatial terms); recomputing'
                )
            else:
                print('[StopEncodingRunner] Using cached design matrices')
                return

        print('[StopEncodingRunner] Computing design matrices from scratch')
        design_kwargs = self._encoding_design_kwargs()
        design_kwargs['tuning_feature_mode'] = tuning_feature_mode

        (
            self.pn,
            self.binned_spikes,
            self.binned_feats,
            offset_log,
            self.stop_meta_used,
            temporal_meta,
            tuning_meta,
        ) = encode_stops_utils.build_stop_encoding_design(
            self.raw_data_folder_path,
            self.bin_width,
            **design_kwargs,
        )

        # Align row order with bin_df: spike_history uses make_bin_df_from_stop_meta
        # which sorts by (new_segment, new_bin) = (event_id, k_within_seg).
        # stop_binned_* are in meta pos order; reorder to match.
        sort_idx = self.stop_meta_used.sort_values(
            ['event_id', 'k_within_seg']
        ).index
        self.stop_meta_used = self.stop_meta_used.loc[sort_idx].reset_index(
            drop=True)
        self.binned_feats = self.binned_feats.loc[sort_idx].reset_index(
            drop=True)
        self.binned_spikes = self.binned_spikes.loc[sort_idx].reset_index(
            drop=True)

        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)
        n_basis_hist = getattr(self.encoder_prs, 'default_n_basis', 20)

        (
            _,
            self._basis,
            self._colnames,
            _,
            self._X_hist,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=self.binned_feats,
            meta_groups=None,
            dt=self.bin_width,
            t_max=self.t_max,
            return_X_hist=True,
            n_basis=n_basis_hist,
        )
        self._spike_cols = list(self._colnames.keys())

        # Restructure meta_groups into one_ff_gam-style categories
        self.structured_meta_groups = encoding_design_utils.build_structured_meta_groups(
            colnames=self._colnames,
            temporal_meta=temporal_meta,
            tuning_meta=tuning_meta,
            target_col=self._spike_cols[0] if self._spike_cols else None,
            dt=self.bin_width,
            t_max=self.t_max,
            n_basis=n_basis_hist,
        )

        self._save_design_matrices()


    def get_design_for_unit(self, unit_idx: int) -> pd.DataFrame:
        """
        Build design matrix with spike-history regressors for the given target neuron.

        Spike-history must match the target neuron when predicting its spikes.
        unit_idx maps to binned_spikes column order.
        """
        self._ensure_spike_history_components()
        if unit_idx < 0 or unit_idx >= len(self._spike_cols):
            raise IndexError(
                f'unit_idx {unit_idx} out of range [0, {len(self._spike_cols)})'
            )
        target_col = self._spike_cols[unit_idx]
        self._refresh_structured_meta_groups_for_unit(target_col)
        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=self.binned_feats.reset_index(drop=True),
            colnames=self._colnames,
            X_hist=self._X_hist,
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
        design_df = design_df.drop(columns=const_cols_to_drop)
        return design_df


    def _ensure_spike_history_components(self):
        """Compute spike-history components if not yet available (e.g. after cache load)."""
        if self._X_hist is not None:
            return
        if self.binned_feats is None or self.stop_meta_used is None:
            raise RuntimeError(
                'Run _collect_data first since self.binned_feats or self.stop_meta_used is None')
        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)

        if not hasattr(self.pn, 'spikes_df'):
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(
                self.raw_data_folder_path, bin_width=0.04)

        _, self._basis, self._colnames, _, self._X_hist = (
            spike_history.build_design_with_spike_history_from_bins(
                spikes_df=self.pn.spikes_df,
                bin_df=bin_df,
                X_pruned=self.binned_feats,
                meta_groups={},
                dt=self.bin_width,
                t_max=self.t_max,
                return_X_hist=True,
            )
        )
        self._spike_cols = list(self._colnames.keys())

    def _refresh_structured_meta_groups_for_unit(
        self, unit_ref: Union[int, str]
    ) -> None:
        """
        Rebuild structured_meta_groups for a target neuron.

        unit_ref can be either:
        - unit_idx (int): index into self._spike_cols
        - target_col (str): explicit spike column name, e.g. 'cluster_15'

        Ensures the first history group is spike_hist for the target neuron and
        each coupling group name cpl_K matches columns from cluster_K.
        """
        if isinstance(unit_ref, int):
            if self._spike_cols is None:
                raise RuntimeError('Spike columns not initialized')
            if unit_ref < 0 or unit_ref >= len(self._spike_cols):
                raise IndexError(
                    f'unit_idx {unit_ref} out of range [0, {len(self._spike_cols)})'
                )
            target_col = self._spike_cols[unit_ref]
        elif isinstance(unit_ref, str):
            target_col = unit_ref
        else:
            raise TypeError(
                f'unit_ref must be int (unit_idx) or str (target_col), got {type(unit_ref)}'
            )

        if self._colnames is None:
            raise RuntimeError('Spike history colnames not initialized')
        if target_col not in self._colnames:
            raise KeyError(
                f'target_col {target_col!r} not in spike-history colnames')

        self.structured_meta_groups = encoding_design_utils.build_structured_meta_groups(
            colnames=self._colnames,
            temporal_meta=self.structured_meta_groups.get('temporal', {}),
            tuning_meta=self.structured_meta_groups.get('tuning', {}),
            target_col=target_col,
            dt=self.bin_width,
            t_max=self.t_max,
            n_basis=self._basis.shape[1],
        )

        hist_groups = self.structured_meta_groups.get(
            'hist', {}).get('groups', {})
        for group_name, cols in hist_groups.items():
            if not group_name.startswith('cpl_') or not cols:
                continue
            suffix = group_name.split('cpl_', 1)[1]
            expected_prefix = f'cluster_{suffix}:'
            if any(not c.startswith(expected_prefix) for c in cols):
                raise ValueError(
                    f'Coupling group mismatch for {group_name}: '
                    f'expected columns from {expected_prefix}'
                )

    @staticmethod
    def _has_one_ff_like_terms(df: Optional[pd.DataFrame]) -> bool:
        """
        True when design contains one_ff-like temporal and spatial tuning terms.
        """
        if df is None or df.empty:
            return False
        cols = list(df.columns)
        has_temporal = any(c.startswith('rcos_') for c in cols)
        has_spatial_boxcar = any(':bin' in c for c in cols)
        return has_temporal and has_spatial_boxcar

    # ------------------------------------------------------------------
    # Caching utilities
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'encoding_outputs/stop_encoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir()) / 'encoding_design'
        return {
            'binned_feats': save_dir / 'binned_feats.pkl',
            'binned_spikes': save_dir / 'binned_spikes.pkl',
            'stop_meta_used': save_dir / 'stop_meta_used.pkl',
            'binrange_dict': save_dir / 'binrange_dict.pkl',
            'structured_meta_groups': save_dir / 'structured_meta_groups.pkl',
        }

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths['binned_feats'].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            'binned_feats': self.binned_feats,
            'binned_spikes': self.binned_spikes,
            'stop_meta_used': self.stop_meta_used,
            'binrange_dict': self.binrange_dict,
            'structured_meta_groups': self.structured_meta_groups,
        }

        for key in data_to_save:
            data = data_to_save[key]
            try:
                with open(paths[key], 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[StopEncodingRunner] Saved {key} → {paths[key]}')
            except Exception:
                print(
                    f'[StopEncodingRunner] WARNING: could not save {key}: '
                )

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()
        required = ['binned_feats',
                    'binned_spikes', 'stop_meta_used', 'structured_meta_groups']
        if not all(paths[k].exists() for k in required):
            return False

        try:
            with open(paths['binned_feats'], 'rb') as f:
                self.binned_feats = pickle.load(f)

            with open(paths['stop_meta_used'], 'rb') as f:
                self.stop_meta_used = pickle.load(f)

            with open(paths['binned_spikes'], 'rb') as f:
                self.binned_spikes = pickle.load(f)

            if paths['binrange_dict'].exists():
                with open(paths['binrange_dict'], 'rb') as f:
                    self.binrange_dict = pickle.load(f)
            else:
                self.binrange_dict = None

            if paths['structured_meta_groups'].exists():
                with open(paths['structured_meta_groups'], 'rb') as f:
                    self.structured_meta_groups = pickle.load(f)
            else:
                # Old cache without structured_meta_groups: use minimal structure
                self.structured_meta_groups = {
                    'tuning': {},
                    'temporal': {},
                    'hist': {},
                    'lambda_config': {},
                }

            self._X_hist = None
            self._colnames = None
            self._basis = None
            self._spike_cols = None

            # Ensure row order matches bin_df (event_id, k_within_seg) for spike_history
            sort_idx = self.stop_meta_used.sort_values(
                ['event_id', 'k_within_seg']
            ).index
            self.stop_meta_used = self.stop_meta_used.loc[sort_idx].reset_index(
                drop=True)
            self.binned_feats = self.binned_feats.loc[sort_idx].reset_index(
                drop=True)
            self.binned_spikes = self.binned_spikes.loc[sort_idx].reset_index(
                drop=True)

            print('[StopEncodingRunner] Loaded cached design matrices')
            return True

        except Exception as e:
            print(
                f'[StopEncodingRunner] WARNING: could not load design matrices: '
                f'{e}'
            )
            return False

    def get_gam_groups(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
    ):
        return self.get_stop_gam_groups(
            unit_idx=unit_idx,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
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
    ):
        return self.get_stop_gam_save_paths(
            unit_idx=unit_idx,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
            ensure_dirs=ensure_dirs,
        )

    def _gam_results_subdir(self) -> str:
        return "stop_gam_results"

    # ------------------------------------------------------------------
    # Poisson GAM fit (stop encoding: design_df + binned_spikes)
    # ------------------------------------------------------------------
    def fit_stop_poisson_gam(
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
        optimizer: str = 'L-BFGS-B',
        verbose: bool = True,
        save_path: Optional[str] = None,
        save_design: bool = False,
        save_metadata: Optional[Dict] = None,
        load_if_exists: bool = True,
    ) -> FitResult:
        """
        Fit a Poisson GAM for one unit using stop design and neural counts.

        Same as one_ff_gam_fit.fit_poisson_gam, but design matrix is
        built per-unit via get_design_for_unit(unit_idx) and the response
        is the neural data for the given unit from self.binned_spikes.

        Parameters
        ----------
        unit_idx : int
            Column index into self.binned_spikes (0 .. n_neurons - 1).
        groups : list of GroupSpec
            Penalty groups for the design columns (from get_stop_gam_groups(unit_idx)).
        lam_f, lam_g, lam_h, lam_p : float
            Penalty parameters; used to compute save_path when save_path is None.
        l1_groups, l1_smooth_eps, max_iter, tol, optimizer, verbose,
        save_path, save_design, save_metadata, load_if_exists
            Passed through to fit_poisson_gam.

        Returns
        -------
        FitResult
            Same as fit_poisson_gam.
        """
        return self.fit_poisson_gam(
            unit_idx,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
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

    def get_stop_gam_groups(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
    ):
        """
        Build GroupSpec list and lambda_config for a unit's design matrix.

        Call after _collect_data(). Uses get_design_for_unit(unit_idx) and
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
        if self.binned_feats is None:
            raise RuntimeError(
                'Run _collect_data first since self.binned_feats is None')
        design_df = self.get_design_for_unit(unit_idx)
        return encode_stops_utils.build_stop_gam_groups(
            design_df,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
        )

    def get_stop_gam_save_paths(
        self,
        unit_idx: int,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
        ensure_dirs: bool = True,
    ) -> Dict:
        """
        Get paths for stop GAM fit and crossval variance explained results.

        Call after _collect_data(exists_ok=True). Creates base stop_gam_results dir
        and neuron-specific fit_results/cv_var_explained subdirs.

        Parameters
        ----------
        unit_idx : int
            Neuron index (0 .. n_neurons - 1).
        lam_f, lam_g, lam_h, lam_p : float
            Penalty parameters for get_stop_gam_groups().
        ensure_dirs : bool
            If True, create base and subdirs (fit_results, cv_var_explained).

        Returns
        -------
        dict with keys:
            base : Path
                stop_gam_results directory.
            outdir : Path
                neuron_{unit_idx} directory.
            groups : List[GroupSpec]
            lambda_config : dict
            lam_suffix : str
            fit_save_path : str
            cv_save_path : str
        """

        self.base = Path(self._get_save_dir()) / 'stop_gam_results'
        if ensure_dirs:
            self.base.mkdir(parents=True, exist_ok=True)
        self.lambda_config = dict(
            lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p,
        )
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=self.lambda_config)
        self.outdir = self.base / f'neuron_{unit_idx}'
        if ensure_dirs:
            (self.outdir / 'fit_results').mkdir(parents=True, exist_ok=True)
            (self.outdir / 'cv_var_explained').mkdir(parents=True, exist_ok=True)
        self.fit_save_path = str(
            self.outdir / 'fit_results' / f'{lam_suffix}.pkl')
        self.cv_save_path = str(
            self.outdir / 'cv_var_explained' / f'{lam_suffix}.pkl')
        return {
            'base': self.base,
            'outdir': self.outdir,
            'lambda_config': self.lambda_config,
            'lam_suffix': lam_suffix,
            'fit_save_path': self.fit_save_path,
            'cv_save_path': self.cv_save_path,
        }

    def crossval_stop_variance_explained_all_neurons(
        self,
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
        n_folds: int = 5,
        load_if_exists: bool = True,
        load_only=False,
        fit_kwargs: Optional[Dict] = None,
        cv_mode: Optional[str] = 'blocked_time_buffered',
        buffer_samples: int = 20,
        unit_indices: Optional[List[int]] = None,
        verbose: bool = True,
        plot_cdf: bool = True,
        log_x: bool = False,
    ) -> List[float]:
        """
        Run crossval_stop_variance_explained for all neurons and collect mean pseudo R².

        Call after _collect_data(exists_ok=True). Uses get_stop_gam_save_paths to set
        up base, groups, and per-neuron cv_save_path.

        Parameters
        ----------
        lam_f, lam_g, lam_h, lam_p : float
            Penalty parameters for get_stop_gam_groups().
        n_folds : int
            Number of CV folds.
        load_if_exists : bool
            If True, load cached crossval results when available.
        fit_kwargs : dict, optional
            Passed to crossval_stop_variance_explained fit_kwargs.
        cv_mode : str
            Cross-validation mode (e.g. 'blocked_time_buffered').
        buffer_samples : int
            Buffer for blocked_time_buffered.
        unit_indices : list of int, optional
            Neuron indices to process. Default: all (0 .. n_neurons - 1).
        verbose : bool
            Print mean R² per neuron.
        plot_cdf : bool
            If True, call gam_variance_explained.plot_variance_explained_cdf().
        log_x : bool
            Passed to plot_variance_explained_cdf when plot_cdf=True.

        Returns
        -------
        all_neuron_r2 : List[float]
            mean_pseudo_r2 for each neuron.
        """
        return self.crossval_variance_explained_all_neurons(
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            load_only=load_only,
            fit_kwargs=fit_kwargs,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            unit_indices=unit_indices,
            verbose=verbose,
            plot_cdf=plot_cdf,
            log_x=log_x,
        )

    def try_load_variance_explained_for_all_neurons(
        self,
        *,
        load_if_exists: bool = True,
        verbose: bool = True,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
    ) -> Optional[List[float]]:
        """Try to load variance explained for all neurons."""
        return self._try_load_variance_explained_for_all_neurons(
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
            load_if_exists=load_if_exists,
            verbose=verbose,
        )

    def crossval_stop_variance_explained(
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
        cv_mode: Optional[str] = 'blocked_time_buffered',
        buffer_samples: int = 20,
        cv_groups=None,
    ) -> Dict:
        """
        Run crossval_variance_explained for one unit using stop design and neural data.

        Uses gam_variance_explained.crossval_variance_explained with
        design_df from get_design_for_unit(unit_idx) and y from self.binned_spikes.

        Parameters
        ----------
        unit_idx : int
            Column index into self.binned_spikes.
        groups : list of GroupSpec
            Same as for fit_stop_poisson_gam (e.g. from get_stop_gam_groups()).
        n_folds, random_state, fit_kwargs, save_path, load_if_exists,
        save_metadata, verbose
            Passed through to crossval_variance_explained.
        dt : float, optional
            Bin width for classical R² smoothing. Defaults to self.bin_width.
        cv_mode : str, optional
            Cross-validation mode. Same as cv_decoding: 'blocked_time_buffered',
            'blocked_time', 'group_kfold', or None (shuffled KFold).
        buffer_samples : int
            Buffer for 'blocked_time_buffered' (default: 0).
        cv_groups : array-like, optional
            Sample-level group labels for 'group_kfold'. Required when cv_mode='group_kfold'.

        Returns
        -------
        dict
            Same as crossval_variance_explained: fold_classical_r2, fold_pseudo_r2,
            mean_classical_r2, mean_pseudo_r2.
        """
        return self.crossval_variance_explained(
            unit_idx=unit_idx,
            groups=groups,
            n_folds=n_folds,
            random_state=random_state,
            fit_kwargs=fit_kwargs,
            save_path=save_path,
            load_if_exists=load_if_exists,
            save_metadata=save_metadata,
            verbose=verbose,
            dt=dt,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            cv_groups=cv_groups,
        )

    def crossval_stop_tuning_curve_coef(
        self,
        unit_idx: int,
        groups: List[GroupSpec],
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
        Perform n-fold (default 10) cross-validation for tuning curve coefficients.

        Fits model on training folds and extracts fit_result.coef for each fold
        to analyze stability and consistency of tuning curve estimates.

        Uses gam_variance_explained.crossval_variance_explained wrapper with
        design_df from get_design_for_unit(unit_idx) and y from self.binned_spikes.

        Parameters
        ----------
        unit_idx : int
            Column index into self.binned_spikes.
        groups : list of GroupSpec
            Penalty groups for the design matrix (e.g. from get_stop_gam_groups()).
        n_folds : int
            Number of cross-validation folds (default: 10).
        random_state : int
            Random seed for fold splitting.
        fit_kwargs : dict, optional
            Additional kwargs passed to fit_poisson_gam (max_iter, tol, l1_groups, etc).
        save_path : str, optional
            Path to save CV tuning curve results (fold coefficients, metadata).
        load_if_exists : bool
            If True, load cached results from save_path when available.
        save_metadata : dict, optional
            Additional metadata to save with results.
        verbose : bool
            Print progress and results.
        cv_mode : str, optional
            Cross-validation mode: 'blocked_time_buffered', 'blocked_time',
            'group_kfold', or None (shuffled KFold). Default: 'blocked_time_buffered'.
        buffer_samples : int
            Buffer for 'blocked_time_buffered' mode (default: 20).
        cv_groups : array-like, optional
            Sample-level group labels for 'group_kfold'. Required when cv_mode='group_kfold'.

        Returns
        -------
        dict with keys:
            fold_coef : list of np.ndarray
                Coefficients from fit_result.coef for each fold.
            fold_design_columns : list of str
                Column names from design_df (same for all folds).
            coef_shape : tuple
                Shape of each fold's coefficient array.
            mean_coef : np.ndarray
                Mean of fold_coef across folds.
            std_coef : np.ndarray
                Std dev of fold_coef across folds.
            unit_idx : int
                Target neuron index.
            cv_mode : str
                Cross-validation mode used.
            n_folds : int
                Number of folds.
            save_path : str
                Path where results were saved (if applicable).
        """

        self._refresh_structured_meta_groups_for_unit(unit_idx)

        if save_path is None:
            paths = self.get_stop_gam_save_paths(
                unit_idx=unit_idx,
                ensure_dirs=False,
            )
            outdir = paths['outdir']
            lam_suffix = paths['lam_suffix']
            cv_tuning_dir = outdir / 'cv_tuning_coef'
            cv_tuning_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(cv_tuning_dir / f'{lam_suffix}.pkl')

        # Try to load cached results
        if load_if_exists and os.path.exists(save_path):
            try:
                with open(save_path, 'rb') as f:
                    cached_result = pickle.load(f)
                if verbose:
                    print(
                        f'Loaded cached tuning curve CV results from: {save_path}')
                return cached_result
            except Exception as e:
                if verbose:
                    print(
                        f'Could not load cached results from {save_path}: {e}')

        # Get design matrix and response for the target neuron
        design_df = self.get_design_for_unit(unit_idx)
        n_rows = len(design_df)
        if n_rows != len(self.binned_spikes):
            raise ValueError(
                f'design and binned_spikes row count mismatch: '
                f'{n_rows} vs {len(self.binned_spikes)}'
            )
        y = np.asarray(
            self.binned_spikes.iloc[:, unit_idx].to_numpy(),
            dtype=float,
        ).ravel()

        if fit_kwargs is None:
            fit_kwargs = {'max_iter': 1000, 'tol': 1e-6, 'verbose': False}

        # Auto-generate save_path if not provided

        # Build cross-validation fold indices
        from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold

        if cv_mode is None:
            splitter = KFold(n_splits=n_folds, shuffle=True,
                             random_state=random_state)
        elif cv_mode == 'blocked_time':
            splitter = TimeSeriesSplit(n_splits=n_folds)
        elif cv_mode == 'blocked_time_buffered':
            # Time series split with buffer
            splitter = TimeSeriesSplit(n_splits=n_folds)
        elif cv_mode == 'group_kfold':
            if cv_groups is None:
                raise ValueError('cv_groups required for group_kfold mode')
            splitter = GroupKFold(n_splits=n_folds)
        else:
            raise ValueError(f'Unknown cv_mode: {cv_mode}')

        fold_coef_list = []
        fold_indices = []

        if cv_mode == 'group_kfold':
            fold_iter = splitter.split(y, groups=cv_groups)
        else:
            fold_iter = splitter.split(y)

        for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
            # Apply buffer for blocked_time_buffered
            if cv_mode == 'blocked_time_buffered' and buffer_samples > 0:
                # Remove samples near fold boundary from training set
                if len(test_idx) > 0:
                    min_test = test_idx.min()
                    train_idx = train_idx[train_idx <
                                          (min_test - buffer_samples)]

            # Fit on training fold
            X_train = design_df.iloc[train_idx, :]
            y_train = y[train_idx]

            if verbose:
                print(
                    f'Fold {fold_idx + 1}/{n_folds}: fitting {len(train_idx)} samples')

            fit_result = one_ff_gam_fit.fit_poisson_gam(
                design_df=X_train,
                y=y_train,
                groups=groups,
                save_path=None,
                save_design=False,
                load_if_exists=False,
                **fit_kwargs,
            )

            # Extract coefficient vector
            if hasattr(fit_result, 'coef') and fit_result.coef is not None:
                fold_coef_list.append(np.asarray(fit_result.coef).copy())
            else:
                raise RuntimeError(
                    f'Fold {fold_idx}: fit_result does not have valid coef attribute'
                )
            fold_indices.append((train_idx, test_idx))

        # Aggregate coefficient statistics
        fold_coef_array = np.array(fold_coef_list)  # (n_folds, n_coef)
        mean_coef = np.mean(fold_coef_array, axis=0)
        std_coef = np.std(fold_coef_array, axis=0)

        result = {
            'fold_coef': fold_coef_list,
            'fold_design_columns': list(design_df.columns),
            'coef_shape': fold_coef_array[0].shape if len(fold_coef_list) > 0 else None,
            'mean_coef': mean_coef,
            'std_coef': std_coef,
            'fold_indices': fold_indices,
            'unit_idx': unit_idx,
            'cv_mode': cv_mode,
            'n_folds': n_folds,
            'random_state': random_state,
        }

        if save_metadata is not None:
            result['metadata'] = save_metadata

        try:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f'Saved tuning curve CV results to: {save_path}')
        except Exception as e:
            print(
                f'WARNING: could not save tuning curve CV results to {save_path}: {e}')

        result['save_path'] = save_path

        if verbose:
            print(f'Completed {n_folds}-fold CV for unit {unit_idx}')
            print(f'  Mean coef shape: {mean_coef.shape}')
            print(
                f'  Coef mean ± std: {np.mean(mean_coef):.4f} ± {np.mean(std_coef):.4f}')

        return result

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
        - ``{save_dir}/stop_gam_results/neuron_{unit_idx}/backward_elimination/{lam_suffix}.pkl``
        - ``{save_dir}/stop_gam_results/neuron_{unit_idx}/backward_elimination/history.csv``
        """
        return self._get_gam_analysis_helper().run_backward_elimination(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            alpha=alpha,
            n_folds=n_folds,
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
        )
