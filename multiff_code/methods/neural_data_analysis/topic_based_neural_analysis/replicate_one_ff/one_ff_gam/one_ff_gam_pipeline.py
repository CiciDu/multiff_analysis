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
    penalty_tuning,
)


DEFAULT_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w"],
    "latent_vars": ["r_targ", "theta_targ"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["t_move", "t_targ", "t_stop", "t_rew"],
    "spike_hist_vars": ["spike_hist"],
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
        self.var_categories = var_categories or DEFAULT_VAR_CATEGORIES
        self.selected_categories = selected_categories
        self.selected_vars = selected_vars
        self.output_root = Path(output_root)
        self.prs = one_ff_parameters.default_prs()

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
            one_ff_gam_design.finalize_one_ff_gam_design(
                unit_idx=unit_idx,
                session_num=self.session_num,
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

    def _compute_category_variance_contributions(
        self,
        *,
        design_df: pd.DataFrame,
        y,
        groups,
        outdir: Path,
        n_folds: int = 5,
        buffer_samples: int = 20,
    ):
        all_group_names = [g.name for g in groups]
        (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)

        full_path = outdir / "cv_var_explained" / "full_model.pkl"
        full_cv = gam_variance_explained.crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=design_df,
            y=y,
            groups=groups,
            dt=self.prs.dt,
            n_folds=n_folds,
            fit_kwargs=dict(
                l1_groups=[],
                max_iter=1000,
                tol=1e-6,
                verbose=False,
                save_path=None,
            ),
            save_path=full_path,
            cv_mode="blocked_time_buffered",
            buffer_samples=buffer_samples,
        )

        contributions = {}
        for category_name, category_vars in self.var_categories.items():
            drop_set = set(category_vars)
            keep_group_names = [gname for gname in all_group_names if gname not in drop_set]
            reduced_df, reduced_groups = self._subset_design_and_groups(
                design_df=design_df,
                groups=groups,
                keep_group_names=keep_group_names,
            )

            loo_path = outdir / "cv_var_explained" / f"leave_out_{category_name}.pkl"
            reduced_cv = gam_variance_explained.crossval_variance_explained(
                fit_function=one_ff_gam_fit.fit_poisson_gam,
                design_df=reduced_df,
                y=y,
                groups=reduced_groups,
                dt=self.prs.dt,
                n_folds=n_folds,
                fit_kwargs=dict(
                    l1_groups=[],
                    max_iter=1000,
                    tol=1e-6,
                    verbose=False,
                    save_path=None,
                ),
                save_path=loo_path,
                cv_mode="blocked_time_buffered",
                buffer_samples=buffer_samples,
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
    ) -> Dict:
        """
        Run fit + CV + leave-one-category-out CV for one unit.
        """
        design_df, y, groups, structured_meta_groups, _ = self._build_design_for_unit(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
        )

        outdir = self._unit_outdir(unit_idx)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "fit_results").mkdir(parents=True, exist_ok=True)
        (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)

        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=structured_meta_groups["lambda_config"]
        )
        fit_save_path = outdir / "fit_results" / f"{lam_suffix}.pkl"

        fit_res = one_ff_gam_fit.fit_poisson_gam(
            design_df=design_df,
            y=y,
            groups=groups,
            l1_groups=[],
            tol=1e-6,
            verbose=True,
            save_path=str(fit_save_path),
            save_metadata={"structured_meta_groups": structured_meta_groups},
        )

        cv_save_path = outdir / "cv_var_explained" / f"{lam_suffix}.pkl"
        cv_res = gam_variance_explained.crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=design_df,
            y=y,
            groups=groups,
            dt=self.prs.dt,
            n_folds=n_folds,
            fit_kwargs=dict(
                l1_groups=[],
                max_iter=1000,
                tol=1e-6,
                verbose=False,
                save_path=None,
            ),
            save_path=cv_save_path,
            cv_mode="blocked_time_buffered",
            buffer_samples=buffer_samples,
        )

        full_cv, category_contrib = self._compute_category_variance_contributions(
            design_df=design_df,
            y=y,
            groups=groups,
            outdir=outdir,
            n_folds=n_folds,
            buffer_samples=buffer_samples,
        )

        contrib_df = pd.DataFrame.from_dict(category_contrib, orient="index")
        contrib_df.index.name = "category"
        contrib_csv = outdir / "cv_var_explained" / "category_contributions.csv"
        contrib_df.to_csv(contrib_csv)

        return {
            "fit_result": fit_res,
            "cv_result": cv_res,
            "full_cv_result": full_cv,
            "category_contributions": category_contrib,
            "category_contributions_csv": contrib_csv,
            "outdir": outdir,
        }

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
        design_df, y, groups, structured_meta_groups, _ = self._build_design_for_unit(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
        )

        outdir = self._unit_outdir(unit_idx)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "backward_elimination").mkdir(parents=True, exist_ok=True)

        lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
            lambda_config=structured_meta_groups["lambda_config"]
        )
        save_path = outdir / "backward_elimination" / f"{lam_suffix}.pkl"

        kept, history = backward_elimination.backward_elimination_gam(
            design_df=design_df,
            y=y,
            groups=groups,
            alpha=alpha,
            n_folds=n_folds,
            verbose=True,
            save_path=str(save_path),
            save_metadata={"structured_meta_groups": structured_meta_groups},
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
        design_df, y, groups, structured_meta_groups, _ = self._build_design_for_unit(
            unit_idx=unit_idx,
            lambda_config=lambda_config,
        )

        if lam_grid is None:
            lam_grid = {
                "lam_f": [10, 50, 100, 300],
                "lam_g": [1, 5, 10, 30],
                "lam_h": [1, 5, 10],
            }

        if group_name_map is None:
            group_name_map = {
                "lam_f": list(structured_meta_groups["tuning"]["groups"].keys()),
                "lam_g": ["t_targ", "t_move", "t_rew"],
                "lam_h": ["spike_hist"],
            }

        outdir = self._unit_outdir(unit_idx)
        outdir.mkdir(parents=True, exist_ok=True)

        save_path = outdir / "penalty_tuning.pkl"
        best_lams, cv_results = penalty_tuning.tune_penalties(
            design_df=design_df,
            y=y,
            base_groups=groups,
            l1_groups=[],
            lam_grid=lam_grid,
            group_name_map=group_name_map,
            n_folds=n_folds,
            save_path=save_path,
            save_metadata={"structured_meta_groups": structured_meta_groups},
        )

        return {
            "best_lams": best_lams,
            "cv_results": cv_results,
            "save_path": save_path,
            "outdir": outdir,
        }
