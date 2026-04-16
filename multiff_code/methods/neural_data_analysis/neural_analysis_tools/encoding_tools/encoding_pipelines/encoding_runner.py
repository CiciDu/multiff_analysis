"""
EncodingRunner — lightweight orchestrator (Task × Model).

Usage
-----
    from encoding_tasks import PNEncodingTask, FSEncodingTask, StopEncodingTask, VisEncodingTask
    from encoding_models import PGAMModel, RNNModel
    from encoding_runner import EncodingRunner

    runner = EncodingRunner(PNEncodingTask(path), PGAMModel())
    runner.crossval(unit_idx=0)
    runner.run_category_variance_contributions(unit_idx=0)

    runner2 = EncodingRunner(PNEncodingTask(path), RNNModel())
    runner2.crossval(unit_idx=0)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .encoding_models import PGAMModel, BaseEncodingModel
from .base_encoding_task import BaseEncodingTask


class EncodingRunner:
    """
    Binds a (Task, Model) pair and exposes a flat call-site.

    All methods are pure pass-throughs — no logic lives here.
    External code reads task data attributes via __getattr__ fallthrough.

    Parameters
    ----------
    task               : BaseEncodingTask subclass
    model              : BaseEncodingModel subclass
    gam_results_subdir : str   sub-directory for GAM results under save dir.
                               Auto-inferred from task class name if not given.
    """

    def __init__(
        self,
        task: BaseEncodingTask,
        model: BaseEncodingModel,
        gam_results_subdir: Optional[str] = None,
    ):
        self.task = task
        self.model = model

        if gam_results_subdir is None:
            prefix = type(task).__name__.lower().replace("task", "")
            gam_results_subdir = f"{prefix}_gam_results" if prefix else "gam_results"
        self.gam_results_subdir = gam_results_subdir

        # Attach subdir to model so _TaskModelProxy can read it
        self.model._gam_results_subdir = gam_results_subdir

    # ------------------------------------------------------------------
    # Fallthrough: task data attributes available directly on runner
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Only called when normal lookup fails — routes to task
        return getattr(self.task, name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def collect_data(self, exists_ok: bool = True):
        self.task.collect_data(exists_ok=exists_ok)

    # ------------------------------------------------------------------
    # Core model interface
    # ------------------------------------------------------------------

    def fit(self, unit_idx: int, **kwargs):
        return self.model.fit(self.task, unit_idx, **kwargs)

    def crossval(self, unit_idx: int, **kwargs):
        kwargs.setdefault("gam_results_subdir", self.gam_results_subdir)
        return self.model.crossval(self.task, unit_idx, **kwargs)

    def crossval_all_neurons(self, **kwargs):
        kwargs.setdefault("gam_results_subdir", self.gam_results_subdir)
        return self.model.crossval_all_neurons(self.task, **kwargs)

    # ------------------------------------------------------------------
    # GAM pass-throughs
    # ------------------------------------------------------------------

    def get_gam_save_paths(self, unit_idx: int, **kwargs):
        if not isinstance(self.model, PGAMModel):
            return None
            # raise TypeError(
            #     "get_gam_save_paths is only available for PGAMModel, "
            #     f"got {type(self.model).__name__}."
            # )
        kwargs.setdefault("gam_results_subdir", self.gam_results_subdir)
        return self.model.get_gam_save_paths(self.task, unit_idx, **kwargs)


    def run_category_variance_contributions(self, unit_idx: int, **kwargs) -> Dict:
        return self.model.run_category_variance_contributions(self.task, unit_idx, **kwargs)

    def run_penalty_tuning(self, unit_idx: int, **kwargs) -> Dict:
        return self.model.run_penalty_tuning(self.task, unit_idx, **kwargs)

    def run_backward_elimination(self, unit_idx: int, **kwargs) -> Dict:
        return self.model.run_backward_elimination(self.task, unit_idx, **kwargs)

    def crossval_tuning_curve_coef(self, unit_idx: int, **kwargs) -> Dict:
        return self.model.crossval_tuning_curve_coef(self.task, unit_idx, **kwargs)

    def try_load_variance_explained_for_all_neurons(self, **kwargs):
        kwargs.setdefault("gam_results_subdir", self.gam_results_subdir)
        return self.model.try_load_variance_explained_for_all_neurons(self.task, **kwargs)

    def run_full_analysis(self, unit_idx: int, **kwargs) -> None:
        return self.model.run_full_analysis(self.task, unit_idx, **kwargs)

    def run_full_analysis_all_neurons(self, **kwargs) -> None:
        return self.model.run_full_analysis_all_neurons(self.task, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EncodingRunner(\n"
            f"  task  = {type(self.task).__name__}({self.task.raw_data_folder_path!r}),\n"
            f"  model = {type(self.model).__name__},\n"
            f"  subdir= {self.gam_results_subdir!r},\n"
            f")"
        )
        
    def get_gam_groups(self):
        from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils
        return encoding_design_utils.build_gam_groups_from_meta(
            self.task.structured_meta_groups,
            lam_f=self.model.lambda_config['lam_f'],
            lam_g=self.model.lambda_config['lam_g'],
            lam_h=self.model.lambda_config['lam_h'],
            lam_p=self.model.lambda_config['lam_p'],
        )