import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Neuroscience-specific imports
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    assemble_stop_design,
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


# ---------------------------------------------------------------------------
# Stop GAM group specs (mirror one_ff_gam: lam_f tuning, lam_g event, lam_h hist, lam_p coupling)
# ---------------------------------------------------------------------------

# Behavioral column groups aligned with stop_design_for_decoding._build_feature_groups.
# (name, list of column names, vartype). Only columns present in design_df are used.
_STOP_BEHAVIORAL_GROUP_SPECS: List[tuple] = [
    # Kinematic (1D each)
    ('accel', ['accel'], '1D'),
    ('speed', ['speed'], '1D'),
    ('ang_speed', ['ang_speed'], '1D'),
    # One_ff_gam-style tuning (if present in design; from extra_agg_cols in encoding design)
    ('v', ['v'], '1D'),
    ('w', ['w'], '1D'),
    ('d', ['d'], '1D'),
    ('phi', ['phi'], '1D'),
    ('r_targ', ['r_targ'], '1D'),
    ('theta_targ', ['theta_targ'], '1D'),
    ('eye_ver', ['eye_ver'], '1D'),
    ('eye_hor', ['eye_hor'], '1D'),
    # Event design (event = temporal basis)
    ('basis', None, 'event'),   # cols: rcos_* without *captured; filled below
    ('basis*captured', None, 'event'),
    ('prepost', ['prepost'], '1D'),
    ('prepost*speed', ['prepost*speed'], '1D'),
    ('captured', ['captured'], '1D'),
    ('time_since_prev_event', ['time_since_prev_event'], '1D'),
    ('time_to_next_event', ['time_to_next_event'], '1D'),
    # Cluster
    ('cluster_flags', ['event_is_first_in_cluster'], '1D'),
    ('cluster_gaps', ['prev_gap_s_z', 'next_gap_s_z'], '1D'),
    ('cluster_duration', ['cluster_duration_s_z'], '1D'),
    ('cluster_progress', ['cluster_progress_c', 'cluster_progress_c2'], '1D'),
    ('cluster_rel_time', ['bin_t_from_cluster_start_s_z'], '1D'),
    ('cluster_n_events', ['log_n_events_in_cluster_z'], '1D'),
    ('is_clustered', ['is_clustered'], '1D'),
    ('cluster_event_t', ['event_t_from_cluster_start_s'], '1D'),
    # Firefly
    ('ff_visible', ['log1p_num_ff_visible', 'k_ff_visible'], '1D'),
    ('ff_in_memory', ['log1p_num_ff_in_memory', 'k_ff_in_memory'], '1D'),
    # Retries
    ('retries', [
        'rsw_first', 'rcap_first', 'rsw_middle', 'rcap_middle',
        'rsw_last', 'rcap_last', 'one_stop_miss', 'whether_in_retry_series', 'miss',
    ], 'event'),
    # Timing
    ('time_rel_to_event_start', ['time_rel_to_event_start'], '1D'),
]


def build_stop_gam_groups(
    design_df: pd.DataFrame,
    *,
    lam_f: float = 100.0,
    lam_g: float = 10.0,
    lam_h: float = 10.0,
    lam_p: float = 10.0,
):
    """
    Build GroupSpec list and lambda config for stop-encoding Poisson GAM, matching one_ff_gam.

    - Const (if present): unpenalized (0D).
    - Behavioral columns: grouped by semantic role; 1D/event and lam_f or lam_g.
    - Spike history: first neuron -> 'spike_hist' (lam_h), rest -> 'cpl_J' (lam_p).

    Parameters
    ----------
    design_df : DataFrame
        design_df_w_history (stop behavioral + spike-history columns).
    lam_f : float
        Penalty for tuning/firefly-style groups (1D smooth).
    lam_g : float
        Penalty for event-style groups (multi-column basis).
    lam_h : float
        Penalty for self spike history.
    lam_p : float
        Penalty for coupling (other neurons' history).

    Returns
    -------
    groups : List[GroupSpec]
        For use with fit_poisson_gam / fit_stop_poisson_gam.
    lambda_config : dict
        {'lam_f': lam_f, 'lam_g': lam_g, 'lam_h': lam_h, 'lam_p': lam_p} for generate_lambda_suffix.
    """
    cols_all = list(design_df.columns)
    # Spike-history columns: cluster_<id>:b0:<k>
    spike_hist_pattern = re.compile(r'^(cluster_\d+):b0:\d+$')
    spike_hist_cols_by_neuron: Dict[str, List[str]] = {}
    behavioral_candidates: List[str] = []
    for c in cols_all:
        m = spike_hist_pattern.match(c)
        if m:
            neuron = m.group(1)
            spike_hist_cols_by_neuron.setdefault(neuron, []).append(c)
        elif c == 'const':
            continue  # handled separately
        else:
            behavioral_candidates.append(c)

    groups: List[GroupSpec] = []

    # Const (unpenalized)
    if 'const' in cols_all:
        groups.append(GroupSpec('const', ['const'], '0D', 0.0))

    # Behavioral groups (mirror one_ff: lam_f for 1D, lam_g for event)
    def _cols_present(candidates) -> List[str]:
        if candidates is None:
            return []
        return [c for c in candidates if c in design_df.columns]

    for spec in _STOP_BEHAVIORAL_GROUP_SPECS:
        name, candidate_cols, vartype = spec
        if name in ('basis', 't_basis'):
            candidate_cols = [c for c in design_df.columns if c.startswith(
                'rcos_') and '*captured' not in c]
        elif name in ('basis*captured', 't_basis_captured'):
            candidate_cols = [c for c in design_df.columns if c.startswith(
                'rcos_') and c.endswith('*captured')]
        else:
            candidate_cols = _cols_present(candidate_cols)
        if not candidate_cols:
            continue
        lam = lam_g if vartype == 'event' else lam_f
        groups.append(GroupSpec(name, candidate_cols, vartype, lam))

    # One_ff-style tuning boxcar columns (var:bin0 .. var:binK from build_tuning_design_stop)
    tuning_boxcar_pattern = re.compile(r'^(\w+):bin\d+$')
    tuning_cols_by_var: Dict[str, List[str]] = {}
    for c in behavioral_candidates:
        if c in design_df.columns:
            m = tuning_boxcar_pattern.match(c)
            if m:
                tuning_cols_by_var.setdefault(m.group(1), []).append(c)
    for var, cols in tuning_cols_by_var.items():
        cols = sorted(cols, key=lambda s: int(s.split(':bin')[1]))
        groups.append(GroupSpec(f'{var}_boxcar', cols, '1D', lam_f))

    # Any behavioral column not in a named group -> single "other" group (event penalty)
    assigned = set()
    for g in groups:
        assigned.update(g.cols)
    other_cols = [c for c in behavioral_candidates if c not in assigned]
    if other_cols:
        groups.append(GroupSpec('t_other', other_cols, 'event', lam_g))

    # Spike history: first neuron = spike_hist (lam_h), rest = cpl_0, cpl_1, ... (lam_p)
    neuron_order = sorted(spike_hist_cols_by_neuron.keys())
    for i, neuron in enumerate(neuron_order):
        hist_cols = spike_hist_cols_by_neuron[neuron]
        hist_cols.sort(key=lambda s: int(s.split(':b0:')[1]))
        if i == 0:
            groups.append(GroupSpec('spike_hist', hist_cols, 'event', lam_h))
        else:
            groups.append(GroupSpec(f'cpl_{i - 1}', hist_cols, 'event', lam_p))

    lambda_config = {
        'lam_f': lam_f,
        'lam_g': lam_g,
        'lam_h': lam_h,
        'lam_p': lam_p,
    }
    return groups, lambda_config


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
        self.design_df_w_history = None
        self.stop_binned_feats = None
        self.stop_binned_spikes = None  # (n_bins x n_neurons) for Poisson GAM
        self.binrange_dict = self.stop_prs.binrange if self.stop_prs is not None else None # tuning bin ranges (from prs or estimated); inspect after run
        self.use_encoding_design = False  # set in _collect_data; used for cache paths

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self, exists_ok=True, use_encoding_design=True,
                      tuning_feature_mode='boxcar_only' # can be 'raw_only', 'boxcar_only', 'raw_plus_boxcar'
                      ):
        """
        Collect and prepare data for stop decoding/encoding.
        """
        self.use_encoding_design = use_encoding_design
        if exists_ok and self._load_design_matrices():
            if use_encoding_design and not self._has_one_ff_like_terms(self.stop_binned_feats):
                print(
                    '[StopEncodingRunner] Cached encoding design is stale '
                    '(missing one_ff-like temporal/spatial terms); recomputing'
                )
            else:
                print('[StopEncodingRunner] Using cached design matrices')
                return

        print('[StopEncodingRunner] Computing design matrices from scratch')
        if use_encoding_design:
            print(
                '[StopEncodingRunner] Using one_ff-style encoding design (build_stop_design_for_encoding)')

        design_kwargs = self._encoding_design_kwargs(use_encoding_design)
        design_kwargs['tuning_feature_mode'] = tuning_feature_mode

        (
            self.pn,
            self.stop_binned_spikes,
            self.stop_binned_feats,
            offset_log,
            self.stop_meta_used,
            stop_meta_groups,
            self.init_stop_binned_feats,
            self.binrange_dict,
        ) = assemble_stop_design.assemble_stop_design_func(
            self.raw_data_folder_path,
            self.bin_width,
            **design_kwargs,
        )

        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)

        (
            self.design_df_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=self.stop_binned_feats,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        self._save_design_matrices()

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

    def _encoding_design_kwargs(self, use_encoding_design: bool) -> Dict:
        """
        Build kwargs for assemble_stop_design_func from stop_prs.
        """
        prs = self.stop_prs
        mode = getattr(prs, 'tuning_feature_mode', None)
        return {
            'use_encoding_design': use_encoding_design,
            'use_tuning_design': bool(
                use_encoding_design
                and mode in ('boxcar_only', 'raw_plus_boxcar')
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
        save_dir = Path(self._get_save_dir())
        if getattr(self, 'use_encoding_design', False):
            save_dir = save_dir / 'encoding_design'
        return {
            'design_df_w_history': save_dir / 'design_df_w_history.pkl',
            'stop_binned_feats': save_dir / 'stop_binned_feats.pkl',
            'stop_binned_spikes': save_dir / 'stop_binned_spikes.pkl',
            'stop_meta_used': save_dir / 'stop_meta_used.pkl',
            'binrange_dict': save_dir / 'binrange_dict.pkl',
        }

    def _save_design_matrices(self):
        paths = self._get_design_matrix_paths()
        # Create the directory that contains the .pkl files (e.g. .../encoding_design when use_encoding_design)
        paths['design_df_w_history'].parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            'design_df_w_history': self.design_df_w_history,
            'stop_binned_feats': self.stop_binned_feats,
            'stop_binned_spikes': self.stop_binned_spikes,
            'stop_meta_used': self.stop_meta_used,
            'binrange_dict': self.binrange_dict,
        }
        # Only save binrange_dict when encoding design (may be None for decoding)
        keys_to_save = list(data_to_save.keys())
        if not getattr(self, 'use_encoding_design', False):
            keys_to_save = [k for k in keys_to_save if k != 'binrange_dict']

        for key in keys_to_save:
            data = data_to_save[key]
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
        required = ['design_df_w_history', 'stop_binned_feats', 'stop_binned_spikes', 'stop_meta_used']
        if not all(paths[k].exists() for k in required):
            return False

        try:
            with open(paths['design_df_w_history'], 'rb') as f:
                self.design_df_w_history = pickle.load(f)

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

            print('[StopEncodingRunner] Loaded cached design matrices')
            return True

        except Exception as e:
            print(
                f'[StopEncodingRunner] WARNING: could not load design matrices: '
                f'{type(e).__name__}: {e}'
            )
            return False

    # ------------------------------------------------------------------
    # Poisson GAM fit (stop encoding: design_df_w_history + stop_binned_spikes)
    # ------------------------------------------------------------------

    def fit_stop_poisson_gam(
        self,
        unit_idx: int,
        *,
        groups: List[GroupSpec],
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
        self.design_df_w_history and the response is the neural data for
        the given unit from self.stop_binned_spikes.

        Parameters
        ----------
        unit_idx : int
            Column index into self.stop_binned_spikes (0 .. n_neurons - 1).
        groups : list of GroupSpec
            Penalty groups for the design columns (must match design_df_w_history).
        l1_groups, l1_smooth_eps, max_iter, tol, optimizer, verbose,
        save_path, save_design, save_metadata, load_if_exists
            Passed through to fit_poisson_gam.

        Returns
        -------
        FitResult
            Same as fit_poisson_gam.
        """
        if self.design_df_w_history is None or self.stop_binned_spikes is None:
            raise RuntimeError(
                'Run _collect_data first (e.g. via run() or by calling _collect_data(exists_ok=True))'
            )
        n_rows = len(self.design_df_w_history)
        if n_rows != len(self.stop_binned_spikes):
            raise ValueError(
                f'design_df_w_history and stop_binned_spikes row count mismatch: '
                f'{n_rows} vs {len(self.stop_binned_spikes)}'
            )
        y = np.asarray(
            self.stop_binned_spikes.iloc[:, unit_idx].to_numpy(),
            dtype=float,
        ).ravel()
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
        *,
        lam_f: float = 100.0,
        lam_g: float = 10.0,
        lam_h: float = 10.0,
        lam_p: float = 10.0,
    ):
        """
        Build GroupSpec list and lambda_config for the current design_df_w_history (mirror one_ff_gam).

        Call after _collect_data(). Uses build_stop_gam_groups() with the same
        lambda roles: lam_f tuning, lam_g event, lam_h spike history, lam_p coupling.

        Returns
        -------
        groups : List[GroupSpec]
        lambda_config : dict
            For generate_lambda_suffix(lambda_config=lambda_config).
        """
        if self.design_df_w_history is None:
            raise RuntimeError('Run _collect_data first')
        return build_stop_gam_groups(
            self.design_df_w_history,
            lam_f=lam_f,
            lam_g=lam_g,
            lam_h=lam_h,
            lam_p=lam_p,
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
    ) -> Dict:
        """
        Run crossval_variance_explained for one unit using stop design and neural data.

        Uses gam_variance_explained.crossval_variance_explained with
        design_df=self.design_df_w_history and y from self.stop_binned_spikes.

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

        Returns
        -------
        dict
            Same as crossval_variance_explained: fold_classical_r2, fold_pseudo_r2,
            mean_classical_r2, mean_pseudo_r2.
        """
        if self.design_df_w_history is None or self.stop_binned_spikes is None:
            raise RuntimeError('Run _collect_data first')
        n_rows = len(self.design_df_w_history)
        if n_rows != len(self.stop_binned_spikes):
            raise ValueError(
                f'design_df_w_history and stop_binned_spikes row count mismatch: '
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
        )
