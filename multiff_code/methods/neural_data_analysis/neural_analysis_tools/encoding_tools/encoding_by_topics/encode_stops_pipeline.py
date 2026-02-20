import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports
import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases

# Neuroscience-specific imports
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    encode_stops_design,
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis import (
    stop_parameters,
)
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit,
    gam_variance_explained,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    FitResult,
    GroupSpec,
)

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
    decode_stops_design
)

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
    encode_stops_utils,
)


def _build_structured_meta_groups(
    colnames: Dict[str, List[str]],
    temporal_meta: Optional[Dict[str, Any]],
    tuning_meta: Optional[Dict[str, Any]],
    *,
    dt: float,
    t_max: float,
    n_basis: int = 20,
) -> Dict[str, Any]:
    """
    Restructure flat meta_groups into one_ff_gam-style categories.

    Returns dict with 'tuning', 'temporal', 'hist', 'lambda_config' keys.
    """
    # Hist: build from colnames (all clusters), map to spike_hist / cpl_J for plot compatibility
    hist_groups: Dict[str, List[str]] = {}
    hist_basis_info: Dict[str, Dict] = {}
    if colnames:
        t_min = dt
        lags_hist, B_hist = glm_bases.raised_log_cosine_basis(
            n_basis=n_basis,
            t_min=t_min,
            t_max=t_max,
            dt=dt,
            log_spaced=True,
            hard_start_zero=True,
        )
        basis_info_entry = {'lags': lags_hist, 'basis': B_hist}
        neuron_order = sorted(colnames.keys())
        for i, neuron in enumerate(neuron_order):
            cols = colnames[neuron]
            if i == 0:
                hist_groups['spike_hist'] = cols
                hist_basis_info['spike_hist'] = basis_info_entry
            else:
                hist_groups[f'cpl_{i - 1}'] = cols
                hist_basis_info[f'cpl_{i - 1}'] = basis_info_entry

    hist_meta: Dict[str, Any] = {
        'groups': hist_groups,
        'basis_info': hist_basis_info,
    } if hist_groups else {}

    return {
        'tuning': tuning_meta if tuning_meta else {},
        'temporal': temporal_meta if temporal_meta else {},
        'hist': hist_meta,
        'lambda_config': {},
    }


class StopEncodingRunner:
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        t_max=0.20,
        stop_prs=None,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.t_max = t_max
        # Optional: stop_parameters.StopParams for encoding design (temporal + tuning)
        self.stop_prs = stop_prs if stop_prs is not None else stop_parameters.default_prs()

        # will be filled during setup
        self.stop_meta_used = None
        # deprecated: use get_design_for_unit(unit_idx)
        self.design_df_w_history = None
        self.stop_binned_feats = None
        self.stop_binned_spikes = None  # (n_bins x n_neurons) for Poisson GAM
        # Spike-history components for per-neuron design (computed on first use)
        self._X_hist = None
        self._colnames = None
        self._basis = None
        self._spike_cols = None
        # tuning bin ranges (from prs or estimated); inspect after run
        self.binrange_dict = self.stop_prs.binrange if self.stop_prs is not None else None
        
        self.structured_meta_groups = {}

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

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
            if not self._has_one_ff_like_terms(self.stop_binned_feats):
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
            self.stop_binned_spikes,
            self.stop_binned_feats,
            offset_log,
            self.stop_meta_used,
            flat_meta_groups,
            self.init_stop_binned_feats,
            self.binrange_dict,
            temporal_meta,
            tuning_meta,
        ) = encode_stops_design.assemble_stop_encoding_design(
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
        self.stop_binned_feats = self.stop_binned_feats.loc[sort_idx].reset_index(
            drop=True)
        self.stop_binned_spikes = self.stop_binned_spikes.loc[sort_idx].reset_index(
            drop=True)
        if hasattr(self, 'init_stop_binned_feats') and self.init_stop_binned_feats is not None:
            self.init_stop_binned_feats = self.init_stop_binned_feats.loc[sort_idx].reset_index(
                drop=True)

        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)
        n_basis_hist = getattr(self.stop_prs, 'default_n_basis', 20)

        (
            _,
            self._basis,
            self._colnames,
            _,
            self._X_hist,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=self.stop_binned_feats,
            meta_groups=flat_meta_groups,
            dt=self.bin_width,
            t_max=self.t_max,
            return_X_hist=True,
            n_basis=n_basis_hist,
        )
        self._spike_cols = list(self._colnames.keys())

        # Restructure meta_groups into one_ff_gam-style categories
        self.structured_meta_groups = _build_structured_meta_groups(
            colnames=self._colnames,
            temporal_meta=temporal_meta,
            tuning_meta=tuning_meta,
            dt=self.bin_width,
            t_max=self.t_max,
            n_basis=n_basis_hist,
        )
        # use get_design_for_unit(unit_idx) per neuron
        self.design_df_w_history = None

        self._save_design_matrices()

    def _ensure_spike_history_components(self):
        """Compute spike-history components if not yet available (e.g. after cache load)."""
        if self._X_hist is not None:
            return
        if self.stop_binned_feats is None or self.stop_meta_used is None:
            raise RuntimeError(
                'Run _collect_data first since self.stop_binned_feats or self.stop_meta_used is None')
        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)
        
        if not hasattr(self.pn, 'spikes_df'):
            self.pn = collect_stop_data.init_pn_to_collect_stop_data(self.raw_data_folder_path, bin_width=0.04)

        _, self._basis, self._colnames, _, self._X_hist = (
            spike_history.build_design_with_spike_history_from_bins(
                spikes_df=self.pn.spikes_df,
                bin_df=bin_df,
                X_pruned=self.stop_binned_feats,
                meta_groups={},
                dt=self.bin_width,
                t_max=self.t_max,
                return_X_hist=True,
            )
        )
        self._spike_cols = list(self._colnames.keys())

    def get_design_for_unit(self, unit_idx: int) -> pd.DataFrame:
        """
        Build design matrix with spike-history regressors for the given target neuron.

        Spike-history must match the target neuron when predicting its spikes.
        unit_idx maps to stop_binned_spikes column order.
        """
        self._ensure_spike_history_components()
        if unit_idx < 0 or unit_idx >= len(self._spike_cols):
            raise IndexError(
                f'unit_idx {unit_idx} out of range [0, {len(self._spike_cols)})'
            )
        target_col = self._spike_cols[unit_idx]
        design_df, _ = spike_history.add_spike_history_to_design(
            design_df=self.stop_binned_feats.reset_index(drop=True),
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

    def _encoding_design_kwargs(self) -> Dict:
        """
        Build kwargs for assemble_stop_encoding_design from stop_prs.
        """
        prs = self.stop_prs
        mode = getattr(prs, 'tuning_feature_mode', None)
        return {
            'use_tuning_design': bool(
                mode in ('boxcar_only', 'raw_plus_boxcar')
            ),
            'tuning_feature_mode': mode,
            'binrange_dict': self.binrange_dict,
            'n_basis': getattr(prs, 'default_n_basis', 20),
            't_min': -getattr(prs, 'pre_event', 0.3),
            't_max': getattr(prs, 'post_event', 0.3),
            'tuning_n_bins': getattr(prs, 'tuning_n_bins', 10),
        }

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
            'stop_binned_feats': save_dir / 'stop_binned_feats.pkl',
            'stop_binned_spikes': save_dir / 'stop_binned_spikes.pkl',
            'stop_meta_used': save_dir / 'stop_meta_used.pkl',
            'binrange_dict': save_dir / 'binrange_dict.pkl',
            'structured_meta_groups': save_dir / 'structured_meta_groups.pkl',
        }

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        paths['stop_binned_feats'].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            'stop_binned_feats': self.stop_binned_feats,
            'stop_binned_spikes': self.stop_binned_spikes,
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
            except Exception as e:
                print(
                    f'[StopEncodingRunner] WARNING: could not save {key}: '
                )

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()
        required = ['stop_binned_feats',
                    'stop_binned_spikes', 'stop_meta_used']
        if not all(paths[k].exists() for k in required):
            return False

        try:
            with open(paths['stop_binned_feats'], 'rb') as f:
                self.stop_binned_feats = pickle.load(f)

            with open(paths['stop_meta_used'], 'rb') as f:
                self.stop_meta_used = pickle.load(f)

            with open(paths['stop_binned_spikes'], 'rb') as f:
                self.stop_binned_spikes = pickle.load(f)

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

            self.design_df_w_history = None
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
            self.stop_binned_feats = self.stop_binned_feats.loc[sort_idx].reset_index(
                drop=True)
            self.stop_binned_spikes = self.stop_binned_spikes.loc[sort_idx].reset_index(
                drop=True)
            if hasattr(self, 'init_stop_binned_feats') and self.init_stop_binned_feats is not None:
                self.init_stop_binned_feats = self.init_stop_binned_feats.loc[sort_idx].reset_index(
                    drop=True)

            print('[StopEncodingRunner] Loaded cached design matrices')
            return True

        except Exception as e:
            print(
                f'[StopEncodingRunner] WARNING: could not load design matrices: '
                f'{e}'
            )
            return False

    # ------------------------------------------------------------------
    # Poisson GAM fit (stop encoding: design_df_w_history + stop_binned_spikes)
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
        is the neural data for the given unit from self.stop_binned_spikes.

        Parameters
        ----------
        unit_idx : int
            Column index into self.stop_binned_spikes (0 .. n_neurons - 1).
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
        if self.stop_binned_feats is None or self.stop_binned_spikes is None:
            self._collect_data(exists_ok=True)

        self.design_df_w_history = self.get_design_for_unit(unit_idx)
        n_rows = len(self.design_df_w_history)
        if n_rows != len(self.stop_binned_spikes):
            raise ValueError(
                f'design and stop_binned_spikes row count mismatch: '
                f'{n_rows} vs {len(self.stop_binned_spikes)}'
            )
        y = np.asarray(
            self.stop_binned_spikes.iloc[:, unit_idx].to_numpy(),
            dtype=float,
        ).ravel()
        
            
        groups, lambda_config = self.get_stop_gam_groups(unit_idx=unit_idx, lam_f=100.0, lam_g=10.0, lam_h=10.0, lam_p=10.0)

        if save_path is None:
            paths = self.get_stop_gam_save_paths(
                unit_idx=unit_idx,
                lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p,
            )
            save_path = paths['fit_save_path']
            

        return one_ff_gam_fit.fit_poisson_gam(
            design_df=self.design_df_w_history,
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
        if self.stop_binned_feats is None:
            raise RuntimeError(
                'Run _collect_data first since self.stop_binned_feats is None')
        self.design_df_w_history = self.get_design_for_unit(unit_idx)
        return encode_stops_utils.build_stop_gam_groups(
            self.design_df_w_history,
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
        load_only = False,
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

        if load_if_exists:
            all_neuron_r2 = self.try_load_variance_explained_for_all_neurons(
                lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p,
                verbose=verbose,
            )
            if all_neuron_r2 is not None:
                print('Number of neurons with cached cross-validation results retrieved: ',
                    len(all_neuron_r2))
                return all_neuron_r2
            else:
                print('No cached cross-validation variance explained results found for all neurons')

        if load_only:
            print('Load only mode is enabled, returning None')
            return None

        if self.stop_binned_feats is None or self.stop_binned_spikes is None:
            self._collect_data(exists_ok=True)
        n_neurons = self.stop_binned_spikes.shape[1]
        if unit_indices is None:
            unit_indices = list(range(n_neurons))
        if fit_kwargs is None:
            fit_kwargs = dict(
                l1_groups=[], max_iter=1000, tol=1e-6, verbose=False, save_path=None
            )

        paths = self.get_stop_gam_save_paths(
            unit_idx=unit_indices[0],
            lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p,
        )
        base = paths['base']
        lam_suffix = paths['lam_suffix']

        all_neuron_r2 = []
        for unit_idx in unit_indices:

            groups, lambda_config = self.get_stop_gam_groups(
                unit_idx=unit_idx,
                lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p,
            )
        
            outdir = base / f'neuron_{unit_idx}'
            (outdir / 'cv_var_explained').mkdir(parents=True, exist_ok=True)
            cv_save_path = str(
                outdir / 'cv_var_explained' / f'{lam_suffix}.pkl')
            cv_res = self.crossval_stop_variance_explained(
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
                print(cv_res['mean_classical_r2'], cv_res['mean_pseudo_r2'])
            all_neuron_r2.append(cv_res['mean_pseudo_r2'])
        if plot_cdf:
            gam_variance_explained.plot_variance_explained_cdf(
                all_neuron_r2, log_x=log_x
            )
        return all_neuron_r2

    def try_load_variance_explained_for_all_neurons(self,
                                                    *,
                                                    load_if_exists: bool = True,
                                                    verbose: bool = True,
                                                    lam_f: float = 100.0,
                                                    lam_g: float = 10.0,
                                                    lam_h: float = 10.0,
                                                    lam_p: float = 10.0,
                                                    ) -> List[float]:
        """
        Try to load variance explained for all neurons.
        """
        
        self.num_neurons = neural_data_processing.find_num_neurons(self.raw_data_folder_path)
        all_neuron_r2 = []
        for unit_idx in range(self.num_neurons):
            try:
                paths_i = self.get_stop_gam_save_paths(
                    unit_idx=unit_idx,
                    lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p,
                )
                base = paths_i['base']
                outdir = base / f'neuron_{unit_idx}'
                (outdir / 'cv_var_explained').mkdir(parents=True, exist_ok=True)
                lam_suffix = paths_i['lam_suffix']
                cv_save_path = str(
                    outdir / 'cv_var_explained' / f'{lam_suffix}.pkl')

                maybe_loaded = gam_variance_explained.maybe_load_saved_crossval(
                    save_path=cv_save_path,
                    load_if_exists=load_if_exists,
                    verbose=verbose,
                )
                if maybe_loaded is not None:
                    print(
                        f'Loaded cached cross-validation results from: {cv_save_path}')
                    all_neuron_r2.append(maybe_loaded['mean_pseudo_r2'])
                    unit_idx += 1
                else:
                    raise RuntimeError(
                        f'No cached cross-validation results found for {cv_save_path}')
            except Exception as e:
                print(
                    f'Try to load variance explained for unit {unit_idx} failed: {e}')
                return None

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
        design_df from get_design_for_unit(unit_idx) and y from self.stop_binned_spikes.

        Parameters
        ----------
        unit_idx : int
            Column index into self.stop_binned_spikes.
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

        # ------------------------------------------------------------------
        maybe_loaded = gam_variance_explained.maybe_load_saved_crossval(
            save_path=save_path,
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        if maybe_loaded is not None:
            print(f'Loaded cached cross-validation results from: {save_path}')
            return maybe_loaded
        else:
            print(f'No cached cross-validation results found for {save_path}')

        # Get design matrix for the target neuron
        self.design_df_w_history = self.get_design_for_unit(unit_idx)
        n_rows = len(self.design_df_w_history)
        if n_rows != len(self.stop_binned_spikes):
            raise ValueError(
                f'design and stop_binned_spikes row count mismatch: '
                f'{n_rows} vs {len(self.stop_binned_spikes)}'
            )
        y = np.asarray(
            self.stop_binned_spikes.iloc[:, unit_idx].to_numpy(),
            dtype=float,
        ).ravel()
        if dt is None:
            dt = float(self.bin_width)
        if fit_kwargs is None:
            fit_kwargs = {}
        meta = dict(save_metadata or {}, unit_idx=unit_idx)
        return gam_variance_explained.crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=self.design_df_w_history,
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
