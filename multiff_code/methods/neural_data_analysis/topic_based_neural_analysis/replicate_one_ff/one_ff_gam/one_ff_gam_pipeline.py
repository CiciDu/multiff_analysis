from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_parameters,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    backward_elimination,
    gam_variance_explained,
    one_ff_gam_design,
    one_ff_gam_fit,
    plot_gam_fit,
    penalty_tuning,
)


DEFAULT_ENCODING_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w"],
    "latent_vars": ["r_targ", "theta_targ"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["t_move", "t_targ", "t_stop", "t_rew"],
    "spike_hist_vars": ["spike_hist"],
}
DEFAULT_LAMBDA_CONFIG = {
    "lam_f": 100.0,
    "lam_g": 10.0,
    "lam_h": 10.0,
    "lam_p": 10.0,
}


class OneFFGAMRunner:
    """
    Pipeline wrapper for one-FF GAM analyses.

    This class consolidates the logic from:
    - jobs/one_ff/scripts/one_ff_my_gam_script.py
    - jobs/one_ff/scripts/one_ff_back_elim_script.py
    - jobs/one_ff/scripts/one_ff_pen_tune_script.py
    """

    def __init__(
        self,
        *,
        session_num: int = 0,
        var_categories: Optional[Dict[str, List[str]]] = None,
        selected_categories: Optional[List[str]] = None,
        selected_vars: Optional[List[str]] = None,
        output_root: str = "all_monkey_data/one_ff_data/my_gam_results",
    ):
        self.session_num = session_num
        self.var_categories = var_categories or DEFAULT_ENCODING_VAR_CATEGORIES
        self.selected_categories = selected_categories
        self.selected_vars = selected_vars
        self.output_root = Path(output_root)
        self.prs = one_ff_parameters.default_prs()
        self.design_builder = one_ff_gam_design.OneFFGAMDesignBuilder(
            session_num=self.session_num,
            prs=self.prs,
        )
        self._reset_unit_state()

    def _reset_unit_state(self) -> None:
        self.unit_idx: Optional[int] = None
        self.design_df = None
        self.y = None
        self.gam_groups = None
        self.structured_meta_groups = None
        self.data_obj = None
        self.outdir: Optional[Path] = None
        self.lam_suffix: Optional[str] = None

    def _build_design_for_unit(
        self,
        unit_idx: int,
        lambda_config: Optional[Dict[str, float]] = None,
    ):
        """
        Build design matrix/response/groups for one unit.

        Supports both old and new finalize_one_ff_gam_design signatures.
        """
        design_df, y, groups, structured_meta_groups, data_obj = (
            self.design_builder.build_unit_design(
                unit_idx=unit_idx,
                lambda_config=lambda_config,
            )
        )

        if self.selected_categories is not None or self.selected_vars is not None:
            design_df, groups, structured_meta_groups = (
                one_ff_gam_design.apply_variable_selection_and_update_meta(
                    design_df=design_df,
                    groups=groups,
                    structured_meta_groups=structured_meta_groups,
                    selected_categories=self.selected_categories,
                    selected_vars=self.selected_vars,
                    var_categories=self.var_categories,
                )
            )

        return design_df, y, groups, structured_meta_groups, data_obj

    def _unit_outdir(self, unit_idx: int) -> Path:
        return self.output_root / f"neuron_{unit_idx}"

    def _unit_context(self, unit_idx: int, ensure_dirs: Optional[List[str]] = None):
        outdir = self._unit_outdir(unit_idx)
        outdir.mkdir(parents=True, exist_ok=True)
        if ensure_dirs is not None:
            for subdir in ensure_dirs:
                (outdir / subdir).mkdir(parents=True, exist_ok=True)

        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=self.structured_meta_groups["lambda_config"]
        )
        return outdir, lam_suffix

    @staticmethod
    def _default_fit_kwargs() -> Dict:
        return {
            "l1_groups": [],
            "max_iter": 1000,
            "tol": 1e-6,
            "verbose": False,
            "save_path": None,
        }

    def _run_crossval(
        self,
        *,
        design_df: pd.DataFrame,
        y,
        groups,
        save_path,
        n_folds: int,
        buffer_samples: int,
        load_if_exists: bool = True,
    ):
        return gam_variance_explained._crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=design_df,
            y=y,
            groups=groups,
            dt=self.prs.dt,
            n_folds=n_folds,
            fit_kwargs=self._default_fit_kwargs(),
            save_path=save_path,
            load_if_exists=load_if_exists,
            cv_mode="blocked_time_buffered",
            buffer_samples=buffer_samples,
        )

    def prepare_unit(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        force_rebuild: bool = False,
    ) -> None:
        if (not force_rebuild) and (self.unit_idx == unit_idx) and (self.design_df is not None):
            return

        design_df, y, groups, structured_meta_groups, data_obj = self._build_design_for_unit(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
        )
        self.unit_idx = unit_idx
        self.design_df = design_df
        self.y = y
        self.gam_groups = groups
        self.structured_meta_groups = structured_meta_groups
        self.data_obj = data_obj
        self.outdir = None
        self.lam_suffix = None

    @staticmethod
    def _subset_design_and_groups(
        design_df: pd.DataFrame,
        groups,
        keep_group_names: List[str],
    ) -> Tuple[pd.DataFrame, List]:
        keep_set = set(keep_group_names)
        groups_kept = [g for g in groups if g.name in keep_set]

        keep_cols = []
        for g in groups_kept:
            keep_cols.extend(g.cols)
        keep_cols = set(keep_cols)

        cols_in_order = [c for c in design_df.columns if (c in keep_cols) or (c == "const")]
        return design_df.loc[:, cols_in_order], groups_kept

    @staticmethod
    def _full_model_cv_path(outdir: Path) -> Path:
        return outdir / "cv_var_explained" / "full_model.pkl"

    @staticmethod
    def _category_cv_path(outdir: Path, category_name: str) -> Path:
        return outdir / "cv_var_explained" / f"leave_out_{category_name}.pkl"

    def _compute_category_variance_contributions(
        self,
        *,
        design_df: pd.DataFrame,
        y,
        groups,
        outdir: Path,
        n_folds: int = 5,
        buffer_samples: int = 20,
        category_names: Optional[List[str]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ):
        all_group_names = [g.name for g in groups]
        (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)

        if category_names is None:
            category_names = list(self.var_categories.keys())
        else:
            unknown = [c for c in category_names if c not in self.var_categories]
            if unknown:
                raise ValueError(
                    f"Unknown category_names: {unknown}. "
                    f"Available: {list(self.var_categories.keys())}"
                )

        full_path = self._full_model_cv_path(outdir)
        if retrieve_only:
            full_cv = gam_variance_explained.maybe_load_saved_crossval(
                save_path=full_path,
                load_if_exists=True,
                verbose=False,
            )
            if full_cv is None:
                raise FileNotFoundError(f"No saved full-model CV result found at: {full_path}")
        else:
            full_cv = self._run_crossval(
                design_df=design_df,
                y=y,
                groups=groups,
                save_path=full_path,
                n_folds=n_folds,
                buffer_samples=buffer_samples,
                load_if_exists=load_if_exists,
            )

        contributions = {}
        for category_name in category_names:
            category_vars = self.var_categories[category_name]
            drop_set = set(category_vars)
            keep_group_names = [gname for gname in all_group_names if gname not in drop_set]
            reduced_df, reduced_groups = self._subset_design_and_groups(
                design_df=design_df,
                groups=groups,
                keep_group_names=keep_group_names,
            )

            loo_path = self._category_cv_path(outdir, category_name)
            if retrieve_only:
                reduced_cv = gam_variance_explained.maybe_load_saved_crossval(
                    save_path=loo_path,
                    load_if_exists=True,
                    verbose=False,
                )
                if reduced_cv is None:
                    raise FileNotFoundError(
                        f"No saved category CV result found for '{category_name}' at: {loo_path}"
                    )
            else:
                reduced_cv = self._run_crossval(
                    design_df=reduced_df,
                    y=y,
                    groups=reduced_groups,
                    save_path=loo_path,
                    n_folds=n_folds,
                    buffer_samples=buffer_samples,
                    load_if_exists=load_if_exists,
                )

            contributions[category_name] = {
                "vars": category_vars,
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

        return full_cv, contributions

    def run_my_gam(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        n_folds: int = 5,
        buffer_samples: int = 20,
        load_if_exists: bool = True,
    ) -> Dict:
        """
        Run fit + CV for one unit. If load_if_exists=True and both fit and CV
        results exist on disk, returns them without calling prepare_unit.
        """
        lam_cfg = lambda_config or DEFAULT_LAMBDA_CONFIG
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(lambda_config=lam_cfg)
        outdir = self._unit_outdir(unit_idx)
        fit_save_path = outdir / "fit_results" / f"{lam_suffix}.pkl"
        cv_save_path = outdir / "cv_var_explained" / f"{lam_suffix}.pkl"

        if load_if_exists and fit_save_path.exists() and cv_save_path.exists():
            loaded = one_ff_gam_fit.load_fit_results(str(fit_save_path))
            fit_res = loaded.get("fit_result")
            cv_res = gam_variance_explained.maybe_load_saved_crossval(
                save_path=str(cv_save_path),
                load_if_exists=True,
                verbose=False,
            )
            if fit_res is not None and cv_res is not None:
                self.outdir = outdir
                self.lam_suffix = lam_suffix
                return {
                    "fit_result": fit_res,
                    "cv_result": cv_res,
                    "outdir": outdir,
                }

        self.prepare_unit(unit_idx=unit_idx, lambda_config=lambda_config)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "fit_results").mkdir(exist_ok=True)
        (outdir / "cv_var_explained").mkdir(exist_ok=True)
        self.outdir = outdir
        self.lam_suffix = lam_suffix

        fit_res = one_ff_gam_fit.fit_poisson_gam(
            design_df=self.design_df,
            y=self.y,
            groups=self.gam_groups,
            l1_groups=[],
            tol=1e-6,
            verbose=True,
            save_path=str(fit_save_path),
            save_metadata={"structured_meta_groups": self.structured_meta_groups},
        )

        cv_res = self._run_crossval(
            design_df=self.design_df,
            y=self.y,
            groups=self.gam_groups,
            save_path=cv_save_path,
            n_folds=n_folds,
            buffer_samples=buffer_samples,
        )

        return {
            "fit_result": fit_res,
            "cv_result": cv_res,
            "outdir": outdir,
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
        Run or retrieve leave-one-category-out variance contribution analysis.
        """
        if category_names is None:
            category_names = list(self.var_categories.keys())
        else:
            unknown = [c for c in category_names if c not in self.var_categories]
            if unknown:
                raise ValueError(
                    f"Unknown category_names: {unknown}. "
                    f"Available: {list(self.var_categories.keys())}"
                )

        # Fast path: load saved CV results directly without preparing design/state.
        outdir = self._unit_outdir(unit_idx)
        (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
        full_path = self._full_model_cv_path(outdir)

        full_cv_loaded = None
        if retrieve_only or load_if_exists:
            full_cv_loaded = gam_variance_explained.maybe_load_saved_crossval(
                save_path=full_path,
                load_if_exists=True,
                verbose=False,
            )

        if full_cv_loaded is not None:
            category_contrib_loaded = {}
            missing_category = None
            for category_name in category_names:
                loo_path = self._category_cv_path(outdir, category_name)
                reduced_cv = gam_variance_explained.maybe_load_saved_crossval(
                    save_path=loo_path,
                    load_if_exists=True,
                    verbose=False,
                )
                if reduced_cv is None:
                    missing_category = category_name
                    break

                category_vars = self.var_categories[category_name]
                category_contrib_loaded[category_name] = {
                    "vars": category_vars,
                    "full_mean_classical_r2": full_cv_loaded["mean_classical_r2"],
                    "full_mean_pseudo_r2": full_cv_loaded["mean_pseudo_r2"],
                    "leave_out_mean_classical_r2": reduced_cv["mean_classical_r2"],
                    "leave_out_mean_pseudo_r2": reduced_cv["mean_pseudo_r2"],
                    "delta_classical_r2": (
                        full_cv_loaded["mean_classical_r2"] - reduced_cv["mean_classical_r2"]
                    ),
                    "delta_pseudo_r2": (
                        full_cv_loaded["mean_pseudo_r2"] - reduced_cv["mean_pseudo_r2"]
                    ),
                }

            if missing_category is None:
                contrib_df = pd.DataFrame.from_dict(
                    category_contrib_loaded, orient="index"
                )
                contrib_df.index.name = "category"
                contrib_csv = outdir / "cv_var_explained" / "category_contributions.csv"
                contrib_df.to_csv(contrib_csv)

                self.outdir = outdir
                return {
                    "full_cv_result": full_cv_loaded,
                    "category_contributions": category_contrib_loaded,
                    "category_contributions_csv": contrib_csv,
                    "outdir": outdir,
                }

            if retrieve_only:
                missing_path = self._category_cv_path(outdir, missing_category)
                raise FileNotFoundError(
                    f"No saved category CV result found for '{missing_category}' at: {missing_path}"
                )
        elif retrieve_only:
            raise FileNotFoundError(f"No saved full-model CV result found at: {full_path}")

        # Fallback: compute using prepared design/state.
        self.prepare_unit(unit_idx=unit_idx, lambda_config=lambda_config)
        self.outdir = outdir
        _, lam_suffix = self._unit_context(
            unit_idx=unit_idx,
            ensure_dirs=["cv_var_explained"],
        )
        self.lam_suffix = lam_suffix

        full_cv, category_contrib = self._compute_category_variance_contributions(
            design_df=self.design_df,
            y=self.y,
            groups=self.gam_groups,
            outdir=outdir,
            n_folds=n_folds,
            buffer_samples=buffer_samples,
            category_names=category_names,
            retrieve_only=retrieve_only,
            load_if_exists=load_if_exists,
        )

        contrib_df = pd.DataFrame.from_dict(category_contrib, orient="index")
        contrib_df.index.name = "category"
        contrib_csv = outdir / "cv_var_explained" / "category_contributions.csv"
        contrib_df.to_csv(contrib_csv)

        return {
            "full_cv_result": full_cv,
            "category_contributions": category_contrib,
            "category_contributions_csv": contrib_csv,
            "outdir": outdir,
        }

    @staticmethod
    def category_contributions_to_df(category_contributions: Dict) -> pd.DataFrame:
        """
        Convert category_contributions dict to a sorted DataFrame.
        """
        return plot_gam_fit.category_contributions_to_df(category_contributions)

    def plot_category_variance_contributions(
        self,
        result: Dict,
        *,
        sort_by: str = "delta_pseudo_r2",
        figsize: Tuple[int, int] = (8, 4),
    ) -> pd.DataFrame:
        """
        Plot category contribution results from run_category_variance_contributions().
        """
        return plot_gam_fit.plot_category_variance_contributions(
            result,
            sort_by=sort_by,
            figsize=figsize,
        )

    def load_category_variance_result(
        self,
        unit_idx: int,
        category_name: str,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Load a saved leave-one-category-out CV result for one category.
        """
        if category_name not in self.var_categories:
            raise ValueError(
                f"Unknown category_name '{category_name}'. "
                f"Available: {list(self.var_categories.keys())}"
            )

        self.prepare_unit(unit_idx=unit_idx, lambda_config=lambda_config)
        outdir, lam_suffix = self._unit_context(
            unit_idx=unit_idx,
            ensure_dirs=["cv_var_explained"],
        )
        self.outdir = outdir
        self.lam_suffix = lam_suffix

        path = self._category_cv_path(outdir, category_name)
        cv_result = gam_variance_explained.maybe_load_saved_crossval(
            save_path=path,
            load_if_exists=True,
            verbose=False,
        )
        if cv_result is None:
            raise FileNotFoundError(
                f"No saved category CV result found for '{category_name}' at: {path}"
            )
        return cv_result

    def run_backward_elimination(
        self,
        unit_idx: int,
        *,
        lambda_config: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
        n_folds: int = 10,
    ) -> Dict:
        """
        Run backward elimination for one unit.
        """
        # Fast path: load saved elimination results without preparing design.
        outdir = self._unit_outdir(unit_idx)
        (outdir / "backward_elimination").mkdir(parents=True, exist_ok=True)
        lambda_for_path = lambda_config or DEFAULT_LAMBDA_CONFIG
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=lambda_for_path
        )
        save_path = outdir / "backward_elimination" / f"{lam_suffix}.pkl"
        if save_path.exists():
            loaded = one_ff_gam_fit.load_elimination_results(str(save_path))
            kept = [
                one_ff_gam_fit.GroupSpec(
                    name=g["name"],
                    cols=g["cols"],
                    vartype=g["vartype"],
                    lam=g["lam"],
                )
                for g in loaded.get("kept_groups", [])
            ]
            history = loaded.get("history", [])
            history_csv = None
            if history:
                history_csv = outdir / "history.csv"
                pd.DataFrame(history).to_csv(history_csv, index=False)
            self.outdir = outdir
            self.lam_suffix = lam_suffix
            return {
                "kept_groups": kept,
                "history": history,
                "history_csv": history_csv,
                "save_path": save_path,
                "outdir": outdir,
            }

        self.prepare_unit(unit_idx=unit_idx, lambda_config=lambda_config)
        outdir, lam_suffix = self._unit_context(
            unit_idx=unit_idx,
            ensure_dirs=["backward_elimination"],
        )
        self.outdir = outdir
        self.lam_suffix = lam_suffix
        save_path = outdir / "backward_elimination" / f"{lam_suffix}.pkl"

        kept, history = backward_elimination.backward_elimination_gam(
            design_df=self.design_df,
            y=self.y,
            groups=self.gam_groups,
            alpha=alpha,
            n_folds=n_folds,
            verbose=True,
            save_path=str(save_path),
            save_metadata={"structured_meta_groups": self.structured_meta_groups},
        )

        history_csv = None
        if history:
            history_csv = outdir / "history.csv"
            pd.DataFrame(history).to_csv(history_csv, index=False)

        return {
            "kept_groups": kept,
            "history": history,
            "history_csv": history_csv,
            "save_path": save_path,
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
    ) -> Dict:
        """
        Run penalty tuning for one unit.
        """
        # Fast path: load saved tuning results without preparing design.
        outdir = self._unit_outdir(unit_idx)
        outdir.mkdir(parents=True, exist_ok=True)
        save_path = outdir / "penalty_tuning.pkl"
        if save_path.exists():
            loaded = penalty_tuning.load_tuning_results(str(save_path))
            self.outdir = outdir
            return {
                "best_lams": loaded.get("best_lams"),
                "cv_results": loaded.get("results"),
                "save_path": save_path,
                "outdir": outdir,
            }

        self.prepare_unit(unit_idx=unit_idx, lambda_config=lambda_config)

        if lam_grid is None:
            lam_grid = {
                "lam_f": [10, 50, 100, 300],
                "lam_g": [1, 5, 10, 30],
                "lam_h": [1, 5, 10],
            }

        if group_name_map is None:
            group_name_map = {
                "lam_f": list(self.structured_meta_groups["tuning"]["groups"].keys()),
                "lam_g": ["t_targ", "t_move", "t_rew"],
                "lam_h": ["spike_hist"],
            }

        outdir, lam_suffix = self._unit_context(unit_idx=unit_idx)
        self.outdir = outdir
        self.lam_suffix = lam_suffix
        save_path = outdir / "penalty_tuning.pkl"
        best_lams, cv_results = penalty_tuning.tune_penalties(
            design_df=self.design_df,
            y=self.y,
            base_groups=self.gam_groups,
            l1_groups=[],
            lam_grid=lam_grid,
            group_name_map=group_name_map,
            n_folds=n_folds,
            save_path=save_path,
            save_metadata={"structured_meta_groups": self.structured_meta_groups},
        )

        return {
            "best_lams": best_lams,
            "cv_results": cv_results,
            "save_path": save_path,
            "outdir": outdir,
        }

    def plot_tuning_heatmaps(
        self,
        *,
        unit_indices: Optional[List[int]] = None,
        lambda_config: Optional[Dict[str, float]] = None,
        var_list: Optional[List[str]] = None,
        tuned_criterion: str = "gain_range",
        gain_range_min: float = 1.5,
        save_dir: Optional[str] = None,
        **plot_kwargs,
    ) -> None:
        """
        Plot peak-normalized tuning heatmaps (MATLAB PlotSessions style).

        Loads fit results for each unit, extracts tuning curves, and calls
        plot_gam_fit.plot_tuning_heatmaps.

        Parameters
        ----------
        unit_indices : list of int, optional
            Neurons to include. If None, discovers from output_root (neuron_* with
            fit results) and falls back to design_builder.n_units.
        lambda_config : dict, optional
            Lambda config for fit path. Defaults to DEFAULT_LAMBDA_CONFIG.
        var_list : list of str, optional
            Variables to plot. If None, uses all tuning + temporal vars.
        tuned_criterion : str
            Passed to plot_tuning_heatmaps ('gain_range' or 'always').
        gain_range_min : float
            Passed to plot_tuning_heatmaps when tuned_criterion='gain_range'.
        save_dir : str or Path, optional
            Directory to save figures. If None, uses output_root / 'tuning_heatmaps'.
        **plot_kwargs
            Additional kwargs for plot_gam_fit.plot_tuning_heatmaps.
        """
        lam_cfg = lambda_config or DEFAULT_LAMBDA_CONFIG
        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(lambda_config=lam_cfg)

        if unit_indices is None:
            # Discover from output_root
            unit_indices = []
            for p in self.output_root.iterdir():
                if p.is_dir() and p.name.startswith("neuron_"):
                    try:
                        uid = int(p.name.split("_", 1)[1])
                    except ValueError:
                        continue
                    fit_path = p / "fit_results" / f"{lam_suffix}.pkl"
                    if fit_path.exists():
                        unit_indices.append(uid)
            unit_indices = sorted(unit_indices)
            if not unit_indices:
                # Fallback: use design to get n_units
                self.design_builder.prepare_shared_design()
                n_units = getattr(
                    self.design_builder.data_obj, "n_units", 0
                )
                unit_indices = list(range(n_units))

        units_data = []
        for unit_idx in unit_indices:
            outdir = self._unit_outdir(unit_idx)
            fit_path = outdir / "fit_results" / f"{lam_suffix}.pkl"
            if not fit_path.exists():
                continue
            try:
                loaded = one_ff_gam_fit.load_fit_results(str(fit_path))
            except Exception:
                continue
            fr = loaded.get("fit_result")
            meta = loaded.get("metadata", {})
            smg = meta.get("structured_meta_groups")
            if fr is None or smg is None:
                continue
            coef = getattr(fr, "coef", None) or (fr.get("coef") if isinstance(fr, dict) else None)
            if coef is None:
                continue
            design_df = loaded.get("design_df")
            col_names = (
                list(design_df.columns)
                if design_df is not None
                else list(coef.index)
                if hasattr(coef, "index")
                else []
            )
            units_data.append({
                "unit_idx": unit_idx,
                "beta": coef,
                "structured_meta_groups": smg,
                "col_names": col_names,
            })

        if not units_data:
            print("No unit fit results found for tuning heatmaps")
            return

        save_dir = save_dir or str(self.output_root / "tuning_heatmaps")
        plot_gam_fit.plot_tuning_heatmaps(
            units_data,
            var_list=var_list,
            tuned_criterion=tuned_criterion,
            gain_range_min=gain_range_min,
            save_dir=save_dir,
            **plot_kwargs,
        )
