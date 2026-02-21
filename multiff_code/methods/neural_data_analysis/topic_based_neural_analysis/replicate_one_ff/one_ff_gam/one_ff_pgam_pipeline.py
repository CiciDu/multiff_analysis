from typing import Dict, List, Optional

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
        covariate_names: Optional[List[str]] = None,
        tuning_covariates: Optional[List[str]] = None,
        use_cyclic: Optional[set] = None,
        order: int = 4,
    ):
        self.session_num = session_num
        self.mat_path = mat_path
        self.pgam_save_dir = pgam_save_dir
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
        if (not force_rebuild) and (self.unit_idx == unit_idx) and (self.pgam_runner is not None):
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
