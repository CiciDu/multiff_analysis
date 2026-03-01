from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encode_stops_utils,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    backward_elimination,
    gam_variance_explained,
    one_ff_gam_fit,
    penalty_tuning,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    GroupSpec,
)


DEFAULT_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w"],
    "latent_vars": ["r_targ", "theta_targ"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["basis", "basis*captured", "prepost", "captured"],
    "spike_hist_vars": ["spike_hist"],
}

# PN encoding: column-based categories (match design column names/prefixes)
PN_VAR_CATEGORIES = {
    "visibility_vars": ["cur_vis", "nxt_vis", "cur_in_memory", "nxt_in_memory"],
    "spatial_vars": ["cur_ff_angle", "nxt_ff_angle", "cur_ff_distance", "nxt_ff_distance"],
    "kinematic_vars": ["speed", "ang_speed", "accel", "ang_accel"],
    "ff_count_vars": ["num_ff_visible", "num_ff_in_memory", "log1p_num_ff_visible", "log1p_num_ff_in_memory"],
    "event_vars": ["rcos_"],
    "spike_hist_vars": ["spike_hist"],
}

# Vis encoding: column-based categories (stop-style design with ff visibility)
VIS_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w", "speed", "ang_speed"],
    "latent_vars": ["r_targ", "theta_targ"],
    "position_vars": ["d", "phi"],
    "ff_visibility_vars": [
        "log1p_num_ff_visible", "k_ff_visible", "num_ff_visible",
        "log1p_num_ff_in_memory", "k_ff_in_memory", "num_ff_in_memory",
    ],
    "event_vars": ["rcos_", "prepost", "captured", "time_since_prev_event", "time_to_next_event"],
    "cluster_vars": [
        "event_is_first_in_cluster", "gap_since_prev", "gap_till_next",
        "cluster_duration", "cluster_progress", "bin_t_from_cluster", "log_n_events", "event_t_from_cluster",
    ],
    "spike_hist_vars": ["spike_hist"],
}

DEFAULT_LAMBDA_CONFIG = {
    "lam_f": 100.0,
    "lam_g": 10.0,
    "lam_h": 10.0,
    "lam_p": 10.0,
}


def _columns_matching_aliases(
    design_df: pd.DataFrame,
    groups: List[GroupSpec],
    aliases: Sequence[str],
) -> List[str]:
    """
    Find design columns matching aliases (column names, prefixes, or group names).

    - Exact match: c == alias
    - Prefix match: c.startswith(alias) for multi-char aliases (e.g. rcos_, cur_ff)
    - Group match: alias "spike_hist" or "coupling" -> columns from that group
    """
    cols_to_drop = set()
    available_group_names = {g.name for g in groups}
    group_cols = {g.name: set(g.cols) for g in groups}

    for alias in aliases:
        if alias == "spike_hist" and "spike_hist" in available_group_names:
            cols_to_drop.update(group_cols["spike_hist"])
        elif alias == "coupling":
            for gname in available_group_names:
                if gname.startswith("cpl_"):
                    cols_to_drop.update(group_cols[gname])
        else:
            for c in design_df.columns:
                if c == alias:
                    cols_to_drop.add(c)
                elif len(alias) > 1 and (c.startswith(alias + "_") or c.startswith(alias)):
                    cols_to_drop.add(c)
    return [c for c in design_df.columns if c in cols_to_drop]


def _subset_design_by_columns(
    design_df: pd.DataFrame,
    groups: List[GroupSpec],
    drop_aliases: Sequence[str],
    lam_f: float,
    lam_g: float,
    lam_h: float,
    lam_p: float,
) -> Tuple[pd.DataFrame, List[GroupSpec]]:
    """Drop columns matching aliases and rebuild groups with build_simple_gam_groups."""
    cols_to_drop = _columns_matching_aliases(design_df, groups, drop_aliases)
    keep_cols = [c for c in design_df.columns if c not in set(cols_to_drop)]
    if not keep_cols or (len(keep_cols) == 1 and keep_cols[0] == "const"):
        raise ValueError(
            f"Cannot leave out category: dropping {cols_to_drop} would remove all predictors"
        )
    reduced_df = design_df[keep_cols].copy()
    reduced_groups, _ = encode_stops_utils.build_simple_gam_groups(
        reduced_df, lam_f=lam_f, lam_g=lam_g, lam_h=lam_h, lam_p=lam_p
    )
    return reduced_df, reduced_groups


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
        gam_results_subdir: str = "gam_results",
        use_column_based_categories: bool = False,
    ):
        self.runner = runner
        self.var_categories = var_categories or {}
        self.gam_results_subdir = gam_results_subdir
        self.use_column_based_categories = use_column_based_categories

    def _resolve_lambda_config(
        self,
        lambda_config: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        cfg = dict(DEFAULT_LAMBDA_CONFIG)
        if lambda_config:
            cfg.update(lambda_config)
        return cfg

    def _neuron_outdir(self, unit_idx: int, ensure_dirs: Optional[List[str]] = None) -> Path:
        outdir = Path(self.runner._get_save_dir()) / self.gam_results_subdir / f"neuron_{unit_idx}"
        outdir.mkdir(parents=True, exist_ok=True)
        if ensure_dirs:
            for subdir in ensure_dirs:
                (outdir / subdir).mkdir(parents=True, exist_ok=True)
        return outdir

    def _unit_context(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[GroupSpec], Dict[str, float]]:
        """Override in subclasses. Returns (design_df, y, groups, lam_cfg)."""
        raise NotImplementedError

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
        cols_in_order = [c for c in design_df.columns if (c in keep_cols) or (c == "const")]
        return design_df.loc[:, cols_in_order], groups_kept

    @staticmethod
    def _full_model_cv_path(outdir: Path) -> Path:
        return outdir / "cv_var_explained" / "full_model.pkl"

    @staticmethod
    def _category_cv_path(outdir: Path, category_name: str) -> Path:
        return outdir / "cv_var_explained" / f"leave_out_{category_name}.pkl"

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
    ) -> Dict:
        return gam_variance_explained.crossval_variance_explained(
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
        cv_mode: str = "blocked_time_buffered",
    ) -> Dict:
        if category_names is None:
            category_names = list(self.var_categories.keys())
        else:
            unknown = [c for c in category_names if c not in self.var_categories]
            if unknown:
                raise ValueError(
                    f"Unknown category_names: {unknown}. "
                    f"Available: {list(self.var_categories.keys())}"
                )
        if not self.var_categories:
            raise ValueError("var_categories cannot be empty for run_category_variance_contributions")

        outdir = self._neuron_outdir(unit_idx, ensure_dirs=["cv_var_explained"])
        full_path = self._full_model_cv_path(outdir)
        load_ok = load_if_exists or retrieve_only
        full_cv = gam_variance_explained.maybe_load_saved_crossval(
            save_path=full_path,
            load_if_exists=load_ok,
            verbose=False,
        )
        category_loads = {}
        for cat in category_names:
            cv = gam_variance_explained.maybe_load_saved_crossval(
                save_path=self._category_cv_path(outdir, cat),
                load_if_exists=load_ok,
                verbose=False,
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
                missing = [c for c in category_names if c not in category_loads]
                if missing:
                    raise FileNotFoundError(
                        "No saved category CV result found for: " f"{missing}"
                    )
            design_df, y, groups, lam_cfg = self._unit_context(
                unit_idx=unit_idx,
                lambda_config=lambda_config,
            )
            available_group_names = [g.name for g in groups]
        else:
            design_df = None
            y = None
            groups = None
            lam_cfg = self._resolve_lambda_config(lambda_config)
            available_group_names = []

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
                cv_mode=cv_mode,
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
                if self.use_column_based_categories:
                    reduced_df, reduced_groups = _subset_design_by_columns(
                        design_df=design_df,
                        groups=groups,
                        drop_aliases=category_aliases,
                        lam_f=lam_cfg["lam_f"],
                        lam_g=lam_cfg["lam_g"],
                        lam_h=lam_cfg["lam_h"],
                        lam_p=lam_cfg["lam_p"],
                    )
                else:
                    drop_group_names = self._expand_aliases_to_group_names(
                        aliases=category_aliases,
                        available_group_names=available_group_names,
                    )
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
                    save_path=self._category_cv_path(outdir, category_name),
                    dt=float(self.runner.bin_width),
                    n_folds=n_folds,
                    buffer_samples=buffer_samples,
                    load_if_exists=load_if_exists,
                )
            contributions[category_name] = {
                "vars": category_aliases,
                "full_mean_classical_r2": full_cv["mean_classical_r2"],
                "full_mean_pseudo_r2": full_cv["mean_pseudo_r2"],
                "leave_out_mean_classical_r2": reduced_cv["mean_classical_r2"],
                "leave_out_mean_pseudo_r2": reduced_cv["mean_pseudo_r2"],
                "delta_classical_r2": (
                    full_cv["mean_classical_r2"] - reduced_cv["mean_classical_r2"]
                ),
                "delta_pseudo_r2": (
                    full_cv["mean_pseudo_r2"] - reduced_cv["mean_pseudo_r2"]
                ),
            }

        contrib_df = pd.DataFrame.from_dict(contributions, orient="index")
        contrib_df.index.name = "category"
        contrib_csv = outdir / "cv_var_explained" / "category_contributions.csv"
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
        lam_grid: Optional[Dict[str, List[float]]] = None,
        group_name_map: Optional[Dict[str, List[str]]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ) -> Dict:
        outdir = self._neuron_outdir(unit_idx)
        save_path = outdir / "penalty_tuning.pkl"
        if retrieve_only and not save_path.exists():
            raise FileNotFoundError(f"No tuning results file found at: {save_path}")
        can_load = (load_if_exists or retrieve_only) and save_path.exists()
        if can_load:
            lam_grid_load = lam_grid or {
                "lam_f": [10, 50, 100, 300],
                "lam_g": [1, 5, 10, 30],
                "lam_h": [1, 5, 10],
                "lam_p": [1, 5, 10],
            }
            best_lams, cv_results = penalty_tuning.tune_penalties(
                design_df=None,
                y=None,
                base_groups=None,
                l1_groups=[],
                lam_grid=lam_grid_load,
                group_name_map=group_name_map or {},
                n_folds=n_folds,
                save_path=str(save_path),
                load_if_exists=load_if_exists,
                retrieve_only=True,
                save_metadata={"structured_meta_groups": self._get_structured_meta_groups()},
            )
            return {
                "best_lams": best_lams,
                "cv_results": cv_results,
                "save_path": save_path,
                "outdir": outdir,
            }
        design_df, y, groups, _ = self._unit_context(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
        )
        if lam_grid is None:
            lam_grid = {
                "lam_f": [10, 50, 100, 300],
                "lam_g": [1, 5, 10, 30],
                "lam_h": [1, 5, 10],
                "lam_p": [1, 5, 10],
            }
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
            lam_p_groups = [g.name for g in groups if g.name.startswith("cpl_")]
            group_name_map = {
                "lam_f": lam_f_groups,
                "lam_g": lam_g_groups,
                "lam_h": lam_h_groups,
                "lam_p": lam_p_groups,
            }
        lam_grid = {k: v for k, v in lam_grid.items() if k in group_name_map and len(group_name_map[k]) > 0}
        group_name_map = {k: v for k, v in group_name_map.items() if k in lam_grid and len(v) > 0}
        if len(lam_grid) == 0:
            raise ValueError("No valid lambda groups found for tuning in this design.")
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
            save_metadata={"structured_meta_groups": self._get_structured_meta_groups()},
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
    ) -> Dict:
        outdir = self._neuron_outdir(unit_idx, ensure_dirs=["backward_elimination"])
        lam_cfg = self._resolve_lambda_config(lambda_config)
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(lambda_config=lam_cfg)
        save_path = outdir / "backward_elimination" / f"{lam_suffix}.pkl"
        if retrieve_only and not save_path.exists():
            raise FileNotFoundError(f"No backward elimination result found at: {save_path}")
        can_load = (load_if_exists or retrieve_only) and save_path.exists()
        if can_load:
            loaded = one_ff_gam_fit.load_elimination_results(str(save_path))
            kept = [
                GroupSpec(name=g["name"], cols=g["cols"], vartype=g["vartype"], lam=g["lam"])
                for g in loaded.get("kept_groups", [])
            ]
            history = loaded.get("history", [])
            history_csv = outdir / "backward_elimination" / "history.csv"
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
        design_df, y, groups, lam_cfg = self._unit_context(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
        )
        if load_if_exists and save_path.exists():
            loaded = one_ff_gam_fit.load_elimination_results(str(save_path))
            kept = [
                GroupSpec(name=g["name"], cols=g["cols"], vartype=g["vartype"], lam=g["lam"])
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
                save_metadata={"structured_meta_groups": self._get_structured_meta_groups()},
            )
        history_csv = outdir / "backward_elimination" / "history.csv"
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


class StopEncodingGAMAnalysisHelper(BaseEncodingGAMAnalysisHelper):
    """
    Shared helper for stop-encoding GAM analyses.

    Keeps category contribution, penalty tuning, and backward elimination out of
    the main pipeline class so encode_stops_pipeline stays focused on data/design.
    """

    def __init__(
        self,
        runner: Any,
        *,
        var_categories: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            runner,
            var_categories=var_categories or DEFAULT_VAR_CATEGORIES,
            gam_results_subdir="stop_gam_results",
            use_column_based_categories=False,
        )

    def _unit_context(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[GroupSpec], Dict[str, float]]:
        if self.runner.stop_binned_feats is None or self.runner.stop_binned_spikes is None:
            self.runner._collect_data(exists_ok=True)
        design_df = self.runner.get_design_for_unit(unit_idx)
        y = np.asarray(self.runner.stop_binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()
        lam_cfg = self._resolve_lambda_config(lambda_config)
        groups, _ = encode_stops_utils.build_stop_gam_groups(
            design_df,
            lam_f=lam_cfg["lam_f"],
            lam_g=lam_cfg["lam_g"],
            lam_h=lam_cfg["lam_h"],
            lam_p=lam_cfg["lam_p"],
        )
        return design_df, y, groups, lam_cfg


class SimpleEncodingGAMAnalysisHelper(BaseEncodingGAMAnalysisHelper):
    """
    GAM analysis helper for PN and Vis encoding (uses build_simple_gam_groups).

    Supports column-based var_categories for run_category_variance_contributions.
    """

    def __init__(
        self,
        runner: Any,
        *,
        var_categories: Optional[Dict[str, List[str]]] = None,
        gam_results_subdir: str = "gam_results",
    ):
        super().__init__(
            runner,
            var_categories=var_categories or {},
            gam_results_subdir=gam_results_subdir,
            use_column_based_categories=True,
        )

    def _unit_context(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[GroupSpec], Dict[str, float]]:
        self.runner._collect_data(exists_ok=True)
        design_df = self.runner.get_design_for_unit(unit_idx)
        binned_spikes = self.runner.get_binned_spikes()
        y = np.asarray(binned_spikes.iloc[:, unit_idx].to_numpy(), dtype=float).ravel()
        lam_cfg = self._resolve_lambda_config(lambda_config)
        groups, _ = self.runner.get_gam_groups(
            unit_idx,
            lam_f=lam_cfg["lam_f"],
            lam_g=lam_cfg["lam_g"],
            lam_h=lam_cfg["lam_h"],
            lam_p=lam_cfg["lam_p"],
        )
        return design_df, y, groups, lam_cfg
