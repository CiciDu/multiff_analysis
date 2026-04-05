from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import copy
import pickle
import os

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    backward_elimination,
    gam_variance_explained,
    one_ff_gam_fit,
    penalty_tuning,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec,
)



DEFAULT_ENCODING_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w", "accel", "ang_accel"],
    "latent_vars": ["cur_ff_distance", "cur_ff_angle", "nxt_ff_distance", "nxt_ff_angle"], # originally ["r_targ", "theta_targ"] in one-ff
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["stop", "capture"],
    "spike_hist_vars": ["spike_hist"],
    "visibility_vars": ["cur_vis", "nxt_vis", "cur_in_memory", "nxt_in_memory"],
    "ff_count_vars": ["num_ff_visible", "num_ff_in_memory", "log1p_num_ff_visible", "log1p_num_ff_in_memory"],
    'session_time_vars': ["time"],
    'coupling_vars': [],
}

# All non-spike variables
DEFAULT_ENCODING_VAR_CATEGORIES['non_spike_vars'] = [
    var
    for category, vars_list in DEFAULT_ENCODING_VAR_CATEGORIES.items()
    if category != 'spike_hist_vars'
    for var in vars_list
]

FS_ENCODING_VAR_CATEGORIES = copy.deepcopy(DEFAULT_ENCODING_VAR_CATEGORIES)
FS_ENCODING_VAR_CATEGORIES['event_vars'].extend(['ff_on', 'ff_off', 'group_ff_on', 'group_ff_off',
                                                 ])
FS_ENCODING_VAR_CATEGORIES['visibility_vars'].extend(['time_since_prev_ff_visible', 'time_since_global_burst_start'])

STOP_ENCODING_VAR_CATEGORIES = copy.deepcopy(DEFAULT_ENCODING_VAR_CATEGORIES)
STOP_ENCODING_VAR_CATEGORIES.update({
    "cluster_vars": [
        "event_is_first_in_cluster", "gap_since_prev", "gap_till_next",
        "cluster_duration", "cluster_progress", "bin_t_from_cluster", "log_n_events", "event_t_from_cluster",
    ],
})
VIS_ENCODING_VAR_CATEGORIES = copy.deepcopy(STOP_ENCODING_VAR_CATEGORIES)
VIS_ENCODING_VAR_CATEGORIES['event_vars'].extend(["ff_on", "group_ff_on", "ff_off", "group_ff_off"])

PN_ENCODING_VAR_CATEGORIES = copy.deepcopy(DEFAULT_ENCODING_VAR_CATEGORIES)
PN_ENCODING_VAR_CATEGORIES.update({'event_vars': ["cur_ff_on", 'cur_ff_off']})



DEFAULT_LAMBDA_CONFIG = {
    "lam_f": 100.0,
    "lam_g": 10.0,
    "lam_h": 10.0,
    "lam_p": 10.0,
}

def add_other_category_from_groups(categories, groups):
    all_group_names = {g.name for g in groups}

    # exclude non_spike_vars from "assigned"
    assigned = set(
        v
        for cat, vars_list in categories.items()
        if cat != 'non_spike_vars'
        for v in vars_list
    )

    other_vars = sorted(all_group_names - assigned)

    categories = categories.copy()
    categories['other_vars'] = other_vars

    return categories


class BaseEncodingGAMAnalysisHelper:
    """
    Base helper for encoding GAM analyses (category contributions, penalty tuning, backward elimination).

    Subclasses override _unit_context and _neuron_outdir for pipeline-specific behavior.
    """

    def __init__(
        self,
        runner: Any,
        *,
        var_categories: Optional[Dict[str, List[str]]] = None,
    ):
        self.runner = runner
        self.var_categories = var_categories or DEFAULT_ENCODING_VAR_CATEGORIES
        self.gam_results_subdir = runner.get_gam_results_subdir()
        

        

    def _resolve_lambda_config(
        self,
        lambda_config: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        cfg = dict(DEFAULT_LAMBDA_CONFIG)
        if lambda_config:
            cfg.update(lambda_config)
        return cfg

    def _neuron_outdir(
        self,
        unit_idx: int,
        ensure_dirs: Optional[List[str]] = None,
        use_neural_coupling: bool = False,
    ) -> Path:
        paths = self.runner.get_gam_save_paths(
            unit_idx=unit_idx,
            use_neural_coupling=use_neural_coupling,
            ensure_dirs=True,
        )
        outdir = paths["outdir"]
        if ensure_dirs:
            for subdir in ensure_dirs:
                (outdir / subdir).mkdir(parents=True, exist_ok=True)
        return outdir


    def _get_structured_meta_groups(self) -> Dict:
        return getattr(self.runner, "structured_meta_groups", {}) or {}

    @staticmethod
    def _subset_design_and_groups(
        design_df: pd.DataFrame,
        groups: List[GroupSpec],
        keep_group_names: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[GroupSpec]]:
        keep_set = set(keep_group_names)
        groups_kept = [g for g in groups if g.name in keep_set]
        keep_cols = set()
        for g in groups_kept:
            keep_cols.update(g.cols)
        cols_in_order = [c for c in design_df.columns if (
            c in keep_cols) or (c == "const")]
        return design_df.loc[:, cols_in_order], groups_kept

    @staticmethod
    def _full_model_cv_path(cv_root: Path) -> Path:
        return cv_root / "full_model.pkl"

    @staticmethod
    def _category_cv_path(cv_root: Path, category_name: str) -> Path:
        return cv_root / f"leave_out_{category_name}.pkl"

    @staticmethod
    def _run_crossval(
        *,
        design_df: pd.DataFrame,
        y: np.ndarray,
        groups: List[GroupSpec],
        save_path: Path,
        dt: float,
        n_folds: int,
        buffer_samples: int,
        load_if_exists: bool,
        cv_mode: str = "blocked_time_buffered",
        cv_groups=None,
    ) -> Dict:
        return gam_variance_explained._crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=design_df,
            y=y,
            groups=groups,
            dt=dt,
            n_folds=n_folds,
            fit_kwargs={
                "l1_groups": [],
                "max_iter": 1000,
                "tol": 1e-6,
                "verbose": False,
                "save_path": None,
            },
            save_path=str(save_path),
            load_if_exists=load_if_exists,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            cv_groups=cv_groups,
        )

    @staticmethod
    def _expand_aliases_to_group_names(
        aliases: Sequence[str],
        available_group_names: Sequence[str],
    ) -> List[str]:
        available = set(available_group_names)
        selected = set()
        for alias in aliases:
            if alias in available:
                selected.add(alias)
            boxcar_name = f"{alias}_boxcar"
            if boxcar_name in available:
                selected.add(boxcar_name)
            if alias == "spike_hist" and "spike_hist" in available:
                selected.add("spike_hist")
            if alias == "coupling":
                selected.update([g for g in available if g.startswith("cpl_")])
        return sorted(selected)

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
        cv_mode: Optional[str] = None,
        use_neural_coupling: bool = False,
        verbose: bool = True,
    ) -> Dict:
        if category_names is None:
            category_names = list(self.var_categories.keys())
        else:
            unknown = [
                c for c in category_names if c not in self.var_categories]
            if unknown:
                raise ValueError(
                    f"Unknown category_names: {unknown}. "
                    f"Available: {list(self.var_categories.keys())}"
                )
        if not self.var_categories:
            raise ValueError(
                "var_categories cannot be empty for run_category_variance_contributions")

        resolved_cv_mode = (
            cv_mode
            if cv_mode is not None
            else getattr(self.runner, "cv_mode", "blocked_time_buffered")
        )
        paths = self.runner.get_gam_save_paths(
            unit_idx=unit_idx,
            use_neural_coupling=use_neural_coupling,
            ensure_dirs=True,
            cv_mode=resolved_cv_mode,
        )
        outdir = paths["outdir"]
        cv_root = Path(paths["cv_save_path"]).parent
        full_path = self._full_model_cv_path(cv_root)
        load_ok = load_if_exists or retrieve_only
        full_cv = gam_variance_explained.maybe_load_saved_crossval(
            save_path=full_path,
            load_if_exists=load_ok,
            verbose=verbose,
        )
        
        category_loads = {}
        for cat in category_names:
            cv = gam_variance_explained.maybe_load_saved_crossval(
                save_path=self._category_cv_path(cv_root, cat),
                load_if_exists=load_ok,
                verbose=verbose,
            )
            if cv is not None:
                category_loads[cat] = cv

        all_categories_missing = all(
            cat not in category_loads for cat in category_names
        )
        need_compute = full_cv is None or all_categories_missing
        if need_compute:
            if retrieve_only:
                if full_cv is None:
                    raise FileNotFoundError(
                        f"No saved full-model CV result found at: {full_path}"
                    )
                missing = [
                    c for c in category_names if c not in category_loads]
                if missing:
                    raise FileNotFoundError(
                        "No saved category CV result found for: " f"{missing}"
                    )
            design_df, y, groups, lambda_config = self._unit_context(
                unit_idx=unit_idx,
                lambda_config=lambda_config,
                use_neural_coupling=use_neural_coupling,
            )
            available_group_names = [g.name for g in groups]
        else:
            design_df = None
            y = None
            groups = None
            lam_cfg = self._resolve_lambda_config(lambda_config)
            available_group_names = []

        cv_groups_for_kfold = None
        if resolved_cv_mode == "group_kfold" and need_compute and design_df is not None:
            cv_groups_for_kfold = self.runner.get_cv_groups_for_design(design_df)
            if cv_groups_for_kfold is None:
                raise ValueError(
                    "cv_mode='group_kfold' requires per-sample group labels. "
                    "Ensure design has 'new_segment', or runner.trial_ids / "
                    "meta_df_used['event_id'] aligns row-wise with the design."
                )

        if full_cv is None:
            full_cv = self._run_crossval(
                design_df=design_df,
                y=y,
                groups=groups,
                save_path=full_path,
                dt=float(self.runner.bin_width),
                n_folds=n_folds,
                buffer_samples=buffer_samples,
                load_if_exists=load_if_exists,
                cv_mode=resolved_cv_mode,
                cv_groups=cv_groups_for_kfold,
            )

        contributions = {}
        categories_to_process = (
            category_names if need_compute else list(category_loads.keys())
        )
        for category_name in categories_to_process:
            category_aliases = self.var_categories[category_name]
            if category_name in category_loads:
                reduced_cv = category_loads[category_name]
            else:
                drop_group_names = self._expand_aliases_to_group_names(
                    aliases=category_aliases,
                    available_group_names=available_group_names,
                )
                design_cols = set(design_df.columns)
                for gname in drop_group_names:
                    g = next((x for x in groups if x.name == gname), None)
                    if g is not None:
                        cols_in_design = [c for c in g.cols if c in design_cols]
                        # if cols_in_design:
                        #     print('cols_in_design to drop for category:', category_name)
                        #     print(f"{gname}: {cols_in_design}")
                keep_group_names = [
                    gname for gname in available_group_names
                    if gname not in set(drop_group_names)
                ]
                reduced_df, reduced_groups = self._subset_design_and_groups(
                    design_df=design_df,
                    groups=groups,
                    keep_group_names=keep_group_names,
                )
                
                reduced_cv = self._run_crossval(
                    design_df=reduced_df,
                    y=y,
                    groups=reduced_groups,
                    save_path=self._category_cv_path(cv_root, category_name),
                    dt=float(self.runner.bin_width),
                    n_folds=n_folds,
                    buffer_samples=buffer_samples,
                    load_if_exists=load_if_exists,
                    cv_mode=resolved_cv_mode,
                    cv_groups=cv_groups_for_kfold,
                )
            contributions[category_name] = {
                "vars": category_aliases,
                "full_mean_classical_r2": full_cv["mean_classical_r2"],
                "full_mean_pseudo_r2": full_cv["mean_pseudo_r2"],
                "leave_out_mean_classical_r2": reduced_cv["mean_classical_r2"],
                "leave_out_mean_pseudo_r2": reduced_cv["mean_pseudo_r2"],
                "delta_classical_r2": (
                    full_cv["mean_classical_r2"] -
                    reduced_cv["mean_classical_r2"]
                ),
                "delta_pseudo_r2": (
                    full_cv["mean_pseudo_r2"] - reduced_cv["mean_pseudo_r2"]
                ),
            }

        contrib_df = pd.DataFrame.from_dict(contributions, orient="index")
        contrib_df.index.name = "category"
        contrib_csv = cv_root / "category_contributions.csv"
        contrib_df.to_csv(contrib_csv)
        return {
            "full_cv_result": full_cv,
            "category_contributions": contributions,
            "category_contributions_csv": contrib_csv,
            "outdir": outdir,
        }

    def run_penalty_tuning(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        group_name_map: Optional[Dict[str, List[str]]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        outdir = self._neuron_outdir(unit_idx, use_neural_coupling=use_neural_coupling)
        resolved_cv = (
            cv_mode if cv_mode is not None
            else getattr(self.runner, "cv_mode", None) or "blocked_time_buffered"
        )
        penalty_dir = outdir / "penalty_tuning" / str(resolved_cv)
        penalty_dir.mkdir(parents=True, exist_ok=True)
        save_path = penalty_dir / "penalty_tuning.pkl"
        if retrieve_only and not save_path.exists():
            raise FileNotFoundError(
                f"No tuning results file found at: {save_path}")
        can_load = (load_if_exists or retrieve_only) and save_path.exists()
        lam_grid = self.runner.lam_grid
        if can_load:
            best_lams, cv_results = penalty_tuning.tune_penalties(
                design_df=None,
                y=None,
                base_groups=None,
                l1_groups=[],
                lam_grid=lam_grid,
                group_name_map=group_name_map or {},
                n_folds=n_folds,
                save_path=str(save_path),
                load_if_exists=load_if_exists,
                retrieve_only=True,
                save_metadata={
                    "structured_meta_groups": self._get_structured_meta_groups()},
            )
            return {
                "best_lams": best_lams,
                "cv_results": cv_results,
                "save_path": save_path,
                "outdir": outdir,
            }
        design_df, y, groups, lambda_config = self._unit_context(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            use_neural_coupling=use_neural_coupling,
        )

        if group_name_map is None:
            lam_f_groups = [
                g.name for g in groups if g.vartype in ("1D", "1Dcirc", "2D") and g.name != "const"
            ]
            lam_g_groups = [
                g.name
                for g in groups
                if g.vartype == "event" and g.name != "spike_hist" and not g.name.startswith("cpl_")
            ]
            lam_h_groups = [g.name for g in groups if g.name == "spike_hist"]
            lam_p_groups = [
                g.name for g in groups if g.name.startswith("cpl_")]
            group_name_map = {
                "lam_f": lam_f_groups,
                "lam_g": lam_g_groups,
                "lam_h": lam_h_groups,
                "lam_p": lam_p_groups,
            }
        lam_grid = {k: v for k, v in lam_grid.items(
        ) if k in group_name_map and len(group_name_map[k]) > 0}
        group_name_map = {
            k: v for k, v in group_name_map.items() if k in lam_grid and len(v) > 0}
        if len(lam_grid) == 0:
            raise ValueError(
                "No valid lambda groups found for tuning in this design.")
        best_lams, cv_results = penalty_tuning.tune_penalties(
            design_df=design_df,
            y=y,
            base_groups=groups,
            l1_groups=[],
            lam_grid=lam_grid,
            group_name_map=group_name_map,
            n_folds=n_folds,
            save_path=str(save_path),
            load_if_exists=load_if_exists,
            retrieve_only=retrieve_only,
            save_metadata={
                "structured_meta_groups": self._get_structured_meta_groups()},
        )
        return {
            "best_lams": best_lams,
            "cv_results": cv_results,
            "save_path": save_path,
            "outdir": outdir,
        }

    def run_backward_elimination(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
        n_folds: int = 10,
        load_if_exists: bool = True,
        retrieve_only: bool = False,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        outdir = self._neuron_outdir(unit_idx, use_neural_coupling=use_neural_coupling)
        resolved_cv = (
            cv_mode if cv_mode is not None
            else getattr(self.runner, "cv_mode", None) or "blocked_time_buffered"
        )
        be_dir = outdir / "backward_elimination" / str(resolved_cv)
        be_dir.mkdir(parents=True, exist_ok=True)
        lam_cfg = self._resolve_lambda_config(lambda_config)
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=lam_cfg)
        save_path = be_dir / f"{lam_suffix}.pkl"
        if retrieve_only and not save_path.exists():
            raise FileNotFoundError(
                f"No backward elimination result found at: {save_path}")
        can_load = (load_if_exists or retrieve_only) and save_path.exists()
        if can_load:
            loaded = one_ff_gam_fit.load_elimination_results(str(save_path))
            kept = [
                GroupSpec(name=g["name"], cols=g["cols"],
                          vartype=g["vartype"], lam=g["lam"])
                for g in loaded.get("kept_groups", [])
            ]
            history = loaded.get("history", [])
            history_csv = be_dir / "history.csv"
            if history:
                pd.DataFrame(history).to_csv(history_csv, index=False)
            else:
                history_csv = None
            print(f"  Loaded backward elimination from {save_path}")
            return {
                "kept_groups": kept,
                "history": history,
                "history_csv": history_csv,
                "save_path": save_path,
                "outdir": outdir,
            }
        design_df, y, groups, lambda_config = self._unit_context(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            use_neural_coupling=use_neural_coupling,
        )
        if load_if_exists and save_path.exists():
            loaded = one_ff_gam_fit.load_elimination_results(str(save_path))
            kept = [
                GroupSpec(name=g["name"], cols=g["cols"],
                          vartype=g["vartype"], lam=g["lam"])
                for g in loaded.get("kept_groups", [])
            ]
            history = loaded.get("history", [])
            print(f"  Loaded backward elimination from {save_path}")
        else:
            kept, history = backward_elimination.backward_elimination_gam(
                design_df=design_df,
                y=y,
                groups=groups,
                alpha=alpha,
                n_folds=n_folds,
                verbose=True,
                save_path=str(save_path),
                load_if_exists=load_if_exists,
                save_metadata={
                    "structured_meta_groups": self._get_structured_meta_groups()},
            )
        history_csv = be_dir / "history.csv"
        if history:
            pd.DataFrame(history).to_csv(history_csv, index=False)
        else:
            history_csv = None
        return {
            "kept_groups": kept,
            "history": history,
            "history_csv": history_csv,
            "save_path": save_path,
            "outdir": outdir,
        }

    def _unit_context(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        use_neural_coupling: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[GroupSpec], Dict[str, float]]:
        if self.runner.binned_feats is None or self.runner.binned_spikes is None:
            self.runner.collect_data(exists_ok=True)
        self.runner.get_design_for_unit(unit_idx, use_neural_coupling=use_neural_coupling)
        binned_spikes = self.runner.binned_spikes
        y = np.asarray(
            binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()
        lam_cfg = self._resolve_lambda_config(lambda_config)
        groups = self.runner.get_gam_groups()
        return self.runner.design_df, y, groups, lam_cfg

    def crossval_tuning_curve_coef(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
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

        Automatically saves to:
        {save_dir}/{results_subdir}/{coupling|no_coupling}/neuron_{unit_idx}/cv_tuning_coef/{cv_mode}/{lam_suffix}.pkl
        """
        resolved_cv_mode = (
            cv_mode
            if cv_mode is not None
            else getattr(self.runner, "cv_mode", "blocked_time_buffered")
        )

        # -------------------------------------------------
        # Auto-generate save path
        # -------------------------------------------------
        if save_path is None:
            save_path = self._auto_build_cv_save_path(
                unit_idx,
                use_neural_coupling=use_neural_coupling,
                cv_mode=resolved_cv_mode,
            )

        design_df, y, groups, lambda_config = self._unit_context(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
            use_neural_coupling=use_neural_coupling,
        )
        # -------------------------------------------------
        # Load cached results if available
        # -------------------------------------------------
        if load_if_exists and save_path is not None and os.path.exists(save_path):
            try:
                with open(save_path, 'rb') as f:
                    cached = pickle.load(f)
                if verbose:
                    print(f'Loaded cached tuning curve CV results from: {save_path}')
                return cached
            except Exception as e:
                if verbose:
                    print(f'Could not load cached results from {save_path}: {e}')

        # -------------------------------------------------
        # Get design matrix and response
        # -------------------------------------------------
        self.runner.get_design_for_unit(unit_idx, use_neural_coupling=use_neural_coupling)
        design_df = self.runner.design_df.copy()
        y = self._extract_response_vector(unit_idx)

        if len(design_df) != len(y):
            raise ValueError(
                f'design and response row mismatch: {len(design_df)} vs {len(y)}'
            )

        if fit_kwargs is None:
            fit_kwargs = {'max_iter': 1000, 'tol': 1e-6, 'verbose': False}

        if resolved_cv_mode == 'group_kfold' and cv_groups is None:
            cv_groups = self.runner.get_cv_groups_for_design(design_df)
            if cv_groups is None:
                raise ValueError(
                    "group_kfold requires cv_groups or inferable trial/segment ids "
                    "(design 'new_segment', runner.trial_ids, or meta_df_used['event_id'])."
                )

        # -------------------------------------------------
        # Build splitter
        # -------------------------------------------------
        splitter = self._build_cv_splitter(
            y=y,
            n_folds=n_folds,
            random_state=random_state,
            cv_mode=resolved_cv_mode,
            cv_groups=cv_groups,
        )

        # -------------------------------------------------
        # Cross-validation loop
        # -------------------------------------------------
        fold_coef_list = []
        fold_indices = []

        if resolved_cv_mode == 'group_kfold':
            fold_iter = splitter.split(y, groups=cv_groups)
        else:
            fold_iter = splitter.split(y)

        for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):

            # Apply buffer for blocked_time_buffered
            if resolved_cv_mode == 'blocked_time_buffered' and buffer_samples > 0:
                if len(test_idx) > 0:
                    min_test = test_idx.min()
                    train_idx = train_idx[train_idx < (min_test - buffer_samples)]

            X_train = design_df.iloc[train_idx, :]
            y_train = y[train_idx]

            if verbose:
                print(
                    f'Fold {fold_idx + 1}/{n_folds}: fitting {len(train_idx)} samples'
                )

            fit_result = one_ff_gam_fit.fit_poisson_gam(
                design_df=X_train,
                y=y_train,
                groups=groups,
                save_path=None,
                save_design=False,
                load_if_exists=False,
                **fit_kwargs,
            )

            if not hasattr(fit_result, 'coef') or fit_result.coef is None:
                raise RuntimeError(
                    f'Fold {fold_idx}: fit_result does not contain valid coef'
                )

            fold_coef_list.append(np.asarray(fit_result.coef).copy())
            fold_indices.append((train_idx, test_idx))

        # -------------------------------------------------
        # Aggregate statistics
        # -------------------------------------------------
        mean_coef, std_coef, coef_shape = self._aggregate_fold_statistics(
            fold_coef_list
        )

        result = {
            'fold_coef': fold_coef_list,
            'fold_design_columns': list(design_df.columns),
            'coef_shape': coef_shape,
            'mean_coef': mean_coef,
            'std_coef': std_coef,
            'fold_indices': fold_indices,
            'unit_idx': unit_idx,
            'cv_mode': resolved_cv_mode,
            'n_folds': n_folds,
            'random_state': random_state,
            'save_path': save_path,
        }

        if save_metadata is not None:
            result['metadata'] = save_metadata

        # -------------------------------------------------
        # Save results
        # -------------------------------------------------
        if save_path is not None:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                if verbose:
                    print(f'Saved tuning curve CV results to: {save_path}')
            except Exception as e:
                print(
                    f'WARNING: could not save tuning curve CV results to {save_path}: {e}'
                )

        if verbose:
            print(f'Completed {n_folds}-fold CV for unit {unit_idx}')
            print(f'  Mean coef shape: {mean_coef.shape}')
            print(
                f'  Coef mean ± std: {np.mean(mean_coef):.4f} ± {np.mean(std_coef):.4f}'
            )

        return result


    # =====================================================
    # Private helpers
    # =====================================================

    def _build_cv_splitter(
        self,
        *,
        y: np.ndarray,
        n_folds: int,
        random_state: int,
        cv_mode: Optional[str],
        cv_groups,
    ):
        from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold

        if cv_mode is None:
            return KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state,
            )

        if cv_mode == 'blocked_time':
            return TimeSeriesSplit(n_splits=n_folds)

        if cv_mode == 'blocked_time_buffered':
            return TimeSeriesSplit(n_splits=n_folds)

        if cv_mode == 'group_kfold':
            if cv_groups is None:
                raise ValueError('cv_groups required for group_kfold mode')
            return GroupKFold(n_splits=n_folds)

        if cv_mode == 'kfold':
            return KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state,
            )

        raise ValueError(f'Unknown cv_mode: {cv_mode}')


    def _extract_response_vector(self, unit_idx: int) -> np.ndarray:
        y = np.asarray(
            self.runner.binned_spikes.iloc[:, unit_idx].to_numpy(),
            dtype=float,
        ).ravel()
        return y


    def _aggregate_fold_statistics(self, fold_coef_list):
        if len(fold_coef_list) == 0:
            return None, None, None

        fold_coef_array = np.array(fold_coef_list)
        mean_coef = np.mean(fold_coef_array, axis=0)
        std_coef = np.std(fold_coef_array, axis=0)
        coef_shape = fold_coef_array[0].shape

        return mean_coef, std_coef, coef_shape


    def _auto_build_cv_save_path(
        self,
        unit_idx: int,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
    ) -> str:
        paths = self.runner.get_gam_save_paths(
            unit_idx=unit_idx,
            use_neural_coupling=use_neural_coupling,
            ensure_dirs=False,
            cv_mode=cv_mode,
        )

        outdir = paths['outdir']
        lam_suffix = paths['lam_suffix']

        cv_tuning_dir = outdir / 'cv_tuning_coef'
        if cv_mode:
            cv_tuning_dir = cv_tuning_dir / str(cv_mode)
        cv_tuning_dir.mkdir(parents=True, exist_ok=True)

        return str(cv_tuning_dir / f'{lam_suffix}.pkl')