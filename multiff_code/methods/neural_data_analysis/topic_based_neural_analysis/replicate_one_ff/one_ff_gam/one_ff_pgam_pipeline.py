import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from neural_data_analysis.neural_analysis_tools.pgam_tools import pgam_class
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_pipeline,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_pgam_design,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_parameters import (
    default_prs,
)

DEFAULT_VAR_CATEGORIES = {
    "sensory_vars": ["v", "w"],
    "latent_vars": ["r_targ", "theta_targ"],
    "position_vars": ["d", "phi"],
    "eye_position_vars": ["eye_ver", "eye_hor"],
    "event_vars": ["t_move", "t_targ", "t_stop", "t_rew"],
    "spike_hist_vars": ["spike_hist"],
}


class OneFFPGAMRunner:
    """
    Pipeline wrapper for one-FF PGAM analyses.

    Consolidates logic from `jobs/one_ff/scripts/one_ff_pgam_script.py` into
    reusable class methods for notebook and job usage.
    """

    def __init__(
        self,
        *,
        session_num: int = 0,
        mat_path: str = "all_monkey_data/one_ff_data/sessions_python.mat",
        pgam_save_dir: str = "all_monkey_data/one_ff_data/pgam_results",
        category_output_root: Optional[str] = None,
        var_categories: Optional[Dict[str, List[str]]] = None,
        covariate_names: Optional[List[str]] = None,
        tuning_covariates: Optional[List[str]] = None,
        use_cyclic: Optional[set] = None,
        order: int = 4,
    ):
        self.session_num = session_num
        self.mat_path = mat_path
        self.pgam_save_dir = pgam_save_dir
        self.category_output_root = Path(
            category_output_root or f"{self.pgam_save_dir}/category_contrib"
        )
        self.var_categories = var_categories or DEFAULT_VAR_CATEGORIES
        self.prs = default_prs()

        self.covariate_names = covariate_names or [
            "v",
            "w",
            "d",
            "phi",
            "r_targ",
            "theta_targ",
            "eye_ver",
            "eye_hor",
        ]
        self.tuning_covariates = tuning_covariates or list(self.covariate_names)
        self.use_cyclic = use_cyclic if use_cyclic is not None else set()
        self.order = order

        self._reset_state()

    def _reset_state(self) -> None:
        # Shared/session-level
        self.data_obj = None
        self.binned_spikes_df = None
        self._shared_ready = False

        # Per-unit
        self.unit_idx: Optional[int] = None
        self.sm_handler = None
        self.pgam_runner = None

    def prepare_shared_data(self, *, force_rebuild: bool = False) -> None:
        """
        Build session-level data used by all units (covariates, spikes, events).
        """
        if self._shared_ready and not force_rebuild:
            return

        data_obj = one_ff_pipeline.OneFFSessionData(
            mat_path=self.mat_path,
            prs=self.prs,
            session_num=self.session_num,
        )
        data_obj.compute_covariates(self.covariate_names)
        data_obj.compute_spike_counts()
        data_obj.smooth_spikes()
        data_obj.compute_events()
        binned_spikes_df = data_obj.get_binned_spikes_df()

        self.data_obj = data_obj
        self.binned_spikes_df = binned_spikes_df
        self.unit_idx = None
        self.sm_handler = None
        self.pgam_runner = None
        self._shared_ready = True

    def _make_pgam_runner(self):
        if self.binned_spikes_df is None:
            raise RuntimeError("Shared data not prepared. Call prepare_shared_data() first.")
        return pgam_class.PGAMclass(
            x_var=self.binned_spikes_df,
            bin_width=self.prs.dt,
            save_dir=self.pgam_save_dir,
        )

    def prepare_unit(self, unit_idx: int, *, force_rebuild: bool = False) -> None:
        """
        Build and cache PGAM smooth handler + runner for a specific unit.
        """
        self.prepare_shared_data(force_rebuild=False)
        has_ready_runner_state = (
            self.pgam_runner is not None
            and self.sm_handler is not None
            and hasattr(self.pgam_runner, "trial_ids")
            and hasattr(self.pgam_runner, "train_trials")
            and hasattr(self.pgam_runner, "sm_handler")
        )
        if (not force_rebuild) and (self.unit_idx == unit_idx) and has_ready_runner_state:
            return

        sm_handler = one_ff_pgam_design.build_smooth_handler(
            data_obj=self.data_obj,
            unit_idx=unit_idx,
            covariate_names=self.covariate_names,
            tuning_covariates=self.tuning_covariates,
            use_cyclic=self.use_cyclic,
            order=self.order,
        )

        pgam_runner = self._make_pgam_runner()
        pgam_runner.sm_handler = sm_handler
        pgam_runner.trial_ids = self.data_obj.covariate_trial_ids
        pgam_runner.train_trials = pgam_runner.trial_ids % 3 != 1

        self.unit_idx = unit_idx
        self.sm_handler = sm_handler
        self.pgam_runner = pgam_runner

    def run_unit_pgam(
        self,
        unit_idx: int,
        *,
        n_splits: int = 5,
        filtwidth: int = 2,
        kernel_h_length: int = 100,
        load_if_exists: bool = True,
        load_only: bool = False,
    ) -> Dict:
        """
        Run PGAM for one unit (fit if needed) and run CV.

        Returns
        -------
        dict with keys:
            loaded_existing: bool
            cv_result: object returned by PGAMclass.run_pgam_cv
            save_dir: str
        """
        self.prepare_shared_data(force_rebuild=False)

        # Fast path: try loading existing results without building per-unit smooths.
        if load_if_exists:
            quick_runner = self._make_pgam_runner()
            try:
                quick_runner.load_pgam_results(unit_idx)
                cv_out = quick_runner.run_pgam_cv(
                    neural_cluster_number=unit_idx,
                    n_splits=n_splits,
                    filtwidth=filtwidth,
                    load_only=True,
                )
                self.unit_idx = unit_idx
                self.pgam_runner = quick_runner
                return {
                    "loaded_existing": True,
                    "cv_result": cv_out,
                    "save_dir": self.pgam_save_dir,
                }
            except FileNotFoundError:
                pass

        # Build per-unit objects and either load existing full model or fit a new one.
        self.prepare_unit(unit_idx=unit_idx, force_rebuild=False)
        if self.pgam_runner is None:
            raise RuntimeError("PGAM runner is not initialized.")

        loaded_existing = False
        if load_if_exists:
            try:
                self.pgam_runner.load_pgam_results(unit_idx)
                loaded_existing = True
            except FileNotFoundError:
                loaded_existing = False

        if (not loaded_existing) and (not load_only):
            self.pgam_runner.run_pgam(neural_cluster_number=unit_idx)
            self.pgam_runner.kernel_h_length = kernel_h_length
            self.pgam_runner.post_processing_results(neural_cluster_number=unit_idx)
            self.pgam_runner.save_results()

        cv_out = self.pgam_runner.run_pgam_cv(
            neural_cluster_number=unit_idx,
            n_splits=n_splits,
            filtwidth=filtwidth,
            load_only=load_only,
        )

        return {
            "loaded_existing": loaded_existing,
            "cv_result": cv_out,
            "save_dir": self.pgam_save_dir,
        }

    @staticmethod
    def _save_pickle(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _category_outdir(self, unit_idx: int) -> Path:
        outdir = self.category_output_root / f"neuron_{unit_idx}"
        (outdir / "cv_var_explained").mkdir(parents=True, exist_ok=True)
        return outdir

    @staticmethod
    def _full_model_cv_path(outdir: Path) -> Path:
        return outdir / "cv_var_explained" / "full_model.pkl"

    @staticmethod
    def _category_cv_path(outdir: Path, category_name: str) -> Path:
        return outdir / "cv_var_explained" / f"leave_out_{category_name}.pkl"

    @staticmethod
    def _resolve_drop_smooths(category_vars: List[str], available_smooths: List[str]) -> List[str]:
        available = set(available_smooths)
        drop = set()
        for var in category_vars:
            if var in available:
                drop.add(var)
            if var == "spike_hist":
                if "h_spike_history" in available:
                    drop.add("h_spike_history")
                continue

            candidates = [f"f_{var}", f"g_{var}", f"h_{var}"]
            for cand in candidates:
                if cand in available:
                    drop.add(cand)
        return sorted(drop)

    def _run_pgam_cv_with_smooth_subset(
        self,
        unit_idx: int,
        smooth_vars: List[str],
        *,
        n_splits: int = 5,
        filtwidth: int = 2,
        random_state: int = 0,
        cv_mode: str = "blocked_time_buffered",
        buffer_samples: int = 20,
        cv_groups=None,
    ) -> Dict:
        self.prepare_unit(unit_idx=unit_idx, force_rebuild=False)
        if self.pgam_runner is None:
            raise RuntimeError("PGAM runner is not initialized.")

        original_smooth_vars = list(self.pgam_runner.sm_handler.smooths_var)
        try:
            self.pgam_runner.sm_handler.smooths_var = list(smooth_vars)
            return self.pgam_runner._run_pgam_cv_compute_only(
                neural_cluster_number=unit_idx,
                n_splits=n_splits,
                filtwidth=filtwidth,
                random_state=random_state,
                cv_mode=cv_mode,
                buffer_samples=buffer_samples,
                cv_groups=cv_groups,
            )
        finally:
            self.pgam_runner.sm_handler.smooths_var = original_smooth_vars

    def run_category_variance_contributions(
        self,
        unit_idx: int,
        *,
        n_splits: int = 5,
        filtwidth: int = 2,
        random_state: int = 0,
        cv_mode: str = "blocked_time_buffered",
        buffer_samples: int = 20,
        cv_groups=None,
        category_names: Optional[List[str]] = None,
        retrieve_only: bool = False,
        load_if_exists: bool = True,
    ) -> Dict:
        """
        Run or retrieve leave-one-category-out PGAM variance contributions.
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

        # Fast path: load saved category-contribution CV results directly
        # without preparing per-unit design/smooth handler state.
        outdir = self._category_outdir(unit_idx)
        full_path = self._full_model_cv_path(outdir)

        full_cv_payload = None
        if retrieve_only or load_if_exists:
            full_cv_payload = self._load_pickle(full_path)

        if full_cv_payload is not None:
            full_cv_loaded = full_cv_payload["cv_result"]
            contributions_loaded = {}
            missing_category = None
            for category_name in category_names:
                loo_path = self._category_cv_path(outdir, category_name)
                loo_payload = self._load_pickle(loo_path)
                if loo_payload is None:
                    missing_category = category_name
                    break

                reduced_cv = loo_payload["cv_result"]
                category_vars = self.var_categories[category_name]
                contributions_loaded[category_name] = {
                    "vars": category_vars,
                    "dropped_smooths": loo_payload.get("dropped_smooths", []),
                    "full_mean_r2_eval": full_cv_loaded["mean_r2_eval"],
                    "leave_out_mean_r2_eval": reduced_cv["mean_r2_eval"],
                    "delta_r2_eval": (
                        full_cv_loaded["mean_r2_eval"] - reduced_cv["mean_r2_eval"]
                    ),
                }

            if missing_category is None:
                contrib_df = pd.DataFrame.from_dict(contributions_loaded, orient="index")
                contrib_df.index.name = "category"
                contrib_csv = outdir / "cv_var_explained" / "category_contributions.csv"
                contrib_df.to_csv(contrib_csv)
                return {
                    "full_cv_result": full_cv_loaded,
                    "category_contributions": contributions_loaded,
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

        # Fallback: compute missing results using prepared unit state.
        self.prepare_unit(unit_idx=unit_idx, force_rebuild=False)
        if self.pgam_runner is None:
            raise RuntimeError("PGAM runner is not initialized.")
        available_smooths = list(self.pgam_runner.sm_handler.smooths_var)

        full_cv = self._run_pgam_cv_with_smooth_subset(
            unit_idx=unit_idx,
            smooth_vars=available_smooths,
            n_splits=n_splits,
            filtwidth=filtwidth,
            random_state=random_state,
            cv_mode=cv_mode,
            buffer_samples=buffer_samples,
            cv_groups=cv_groups,
        )
        self._save_pickle(
            full_path,
            {
                "unit_idx": unit_idx,
                "kind": "full_model",
                "smooth_vars": available_smooths,
                "cv_result": full_cv,
            },
        )

        contributions = {}
        for category_name in category_names:
            category_vars = self.var_categories[category_name]
            drop_smooths = self._resolve_drop_smooths(category_vars, available_smooths)
            keep_smooths = [s for s in available_smooths if s not in set(drop_smooths)]
            loo_path = self._category_cv_path(outdir, category_name)

            loo_payload = None
            if load_if_exists or retrieve_only:
                loo_payload = self._load_pickle(loo_path)
            if retrieve_only:
                if loo_payload is None:
                    raise FileNotFoundError(
                        f"No saved category CV result found for '{category_name}' at: {loo_path}"
                    )
                reduced_cv = loo_payload["cv_result"]
            elif loo_payload is not None:
                reduced_cv = loo_payload["cv_result"]
            else:
                reduced_cv = self._run_pgam_cv_with_smooth_subset(
                    unit_idx=unit_idx,
                    smooth_vars=keep_smooths,
                    n_splits=n_splits,
                    filtwidth=filtwidth,
                    random_state=random_state,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                    cv_groups=cv_groups,
                )
                self._save_pickle(
                    loo_path,
                    {
                        "unit_idx": unit_idx,
                        "kind": "leave_one_category_out",
                        "category_name": category_name,
                        "dropped_smooths": drop_smooths,
                        "kept_smooths": keep_smooths,
                        "cv_result": reduced_cv,
                    },
                )

            contributions[category_name] = {
                "vars": category_vars,
                "dropped_smooths": drop_smooths,
                "full_mean_r2_eval": full_cv["mean_r2_eval"],
                "leave_out_mean_r2_eval": reduced_cv["mean_r2_eval"],
                "delta_r2_eval": full_cv["mean_r2_eval"] - reduced_cv["mean_r2_eval"],
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

    def load_category_variance_result(self, unit_idx: int, category_name: str) -> Dict:
        """
        Load a saved leave-one-category-out PGAM CV result for one category.
        """
        if category_name not in self.var_categories:
            raise ValueError(
                f"Unknown category_name '{category_name}'. "
                f"Available: {list(self.var_categories.keys())}"
            )
        outdir = self._category_outdir(unit_idx)
        path = self._category_cv_path(outdir, category_name)
        payload = self._load_pickle(path)
        if payload is None:
            raise FileNotFoundError(
                f"No saved category CV result found for '{category_name}' at: {path}"
            )
        return payload["cv_result"]
