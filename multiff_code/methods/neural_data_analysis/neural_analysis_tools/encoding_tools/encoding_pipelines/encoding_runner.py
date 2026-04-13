"""
EncodingRunner — lightweight orchestrator (Task × Model).

Usage
-----
    from encoding_tasks import PNTask, FSTask, StopTask, VisTask
    from encoding_models import PGAMModel, RNNModel
    from encoding_runner import EncodingRunner

    # Swap tasks
    runner = EncodingRunner(PNTask(path), PGAMModel())
    runner.crossval(unit_idx=0)

    # Swap models — identical task, different model
    runner2 = EncodingRunner(PNTask(path), RNNModel())
    runner2.crossval(unit_idx=0)

The runner owns *no* data-loading and *no* model-fitting logic; it
simply binds a task to a model and exposes a clean call-site.

For PGAMModel the runner also exposes convenience pass-throughs to the
richer GAM analysis methods (category contributions, penalty tuning,
backward elimination).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .encoding_models import PGAMModel, BaseEncodingModel
from .base_encoding_task import BaseEncodingTask


class EncodingRunner:
    """
    Orchestrates a (Task, Model) pair.

    Parameters
    ----------
    task  : BaseEncodingTask subclass instance
    model : BaseEncodingModel subclass instance
    gam_results_subdir : str
        Sub-directory for GAM results under the task's save dir.
        Ignored for non-GAM models.
    """

    def __init__(
        self,
        task: BaseEncodingTask,
        model: BaseEncodingModel,
        gam_results_subdir: Optional[str] = None,
    ):
        self.task = task
        self.model = model

        # Infer sensible default subdir from task class name
        if gam_results_subdir is None:
            prefix = type(task).__name__.lower().replace("task", "")
            gam_results_subdir = f"{prefix}_gam_results" if prefix else "gam_results"
        self.gam_results_subdir = gam_results_subdir

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def run(self, unit_idx: int, **kwargs):
        """Fit model on all data for one unit."""
        return self.model.fit(self.task, unit_idx, **kwargs)

    def crossval(self, unit_idx: int, **kwargs):
        """Cross-validate model for one unit."""
        # Pass gam_results_subdir only if the model accepts it
        if isinstance(self.model, PGAMModel):
            kwargs.setdefault("gam_results_subdir", self.gam_results_subdir)
        return self.model.crossval(self.task, unit_idx, **kwargs)

    # ------------------------------------------------------------------
    # Convenience: all-neuron crossval
    # ------------------------------------------------------------------

    def crossval_all_neurons(self, **kwargs):
        """
        Cross-validate all neurons.  Only supported for PGAMModel right now;
        for other models call runner.crossval(unit_idx) in a loop.
        """
        if not isinstance(self.model, PGAMModel):
            n = self.task.num_neurons
            return [self.crossval(i, **kwargs) for i in range(n)]
        kwargs.setdefault("gam_results_subdir", self.gam_results_subdir)
        return self.model.crossval_all_neurons(self.task, **kwargs)

    # ------------------------------------------------------------------
    # GAM-specific pass-throughs (only valid when model is PGAMModel)
    # ------------------------------------------------------------------

    def _require_pgam(self, method_name: str):
        if not isinstance(self.model, PGAMModel):
            raise TypeError(
                f"{method_name} is only available when the model is PGAMModel, "
                f"got {type(self.model).__name__}."
            )

    def run_category_variance_contributions(self, unit_idx: int, **kwargs) -> Dict:
        self._require_pgam("run_category_variance_contributions")
        return self.model.run_category_variance_contributions(
            self.task, unit_idx, **kwargs
        )

    def run_penalty_tuning(self, unit_idx: int, **kwargs) -> Dict:
        self._require_pgam("run_penalty_tuning")
        return self.model.run_penalty_tuning(self.task, unit_idx, **kwargs)

    def run_backward_elimination(self, unit_idx: int, **kwargs) -> Dict:
        self._require_pgam("run_backward_elimination")
        return self.model.run_backward_elimination(self.task, unit_idx, **kwargs)

    def run_full_analysis(
        self,
        unit_idx: int,
        *,
        n_folds: int = 5,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        verbose: bool = True,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> None:
        """
        Category contributions + backward elimination for one unit.
        Convenience wrapper for PGAMModel.
        """
        self._require_pgam("run_full_analysis")
        if verbose:
            print(f"[EncodingRunner] Full analysis for unit {unit_idx}")
        try:
            self.run_category_variance_contributions(
                unit_idx,
                n_folds=n_folds,
                buffer_samples=buffer_samples,
                load_if_exists=load_if_exists,
                use_neural_coupling=use_neural_coupling,
                cv_mode=cv_mode,
            )
            self.run_backward_elimination(
                unit_idx,
                n_folds=backward_n_folds,
                alpha=alpha,
                load_if_exists=load_if_exists,
                use_neural_coupling=use_neural_coupling,
                cv_mode=cv_mode,
            )
        except Exception as e:
            if verbose:
                print(f"  [WARN] unit {unit_idx}: {e}")

    def run_full_analysis_all_neurons(
        self,
        *,
        unit_indices: Optional[List[int]] = None,
        n_folds: int = 5,
        backward_n_folds: int = 10,
        alpha: float = 0.05,
        load_if_exists: bool = True,
        verbose: bool = True,
        use_neural_coupling: bool = False,
        cv_mode: Optional[str] = None,
        buffer_samples: int = 20,
    ) -> None:
        self._require_pgam("run_full_analysis_all_neurons")
        self.task.collect_data(exists_ok=True)
        if unit_indices is None:
            unit_indices = list(range(self.task.num_neurons))
        for unit_idx in unit_indices:
            self.run_full_analysis(
                unit_idx,
                n_folds=n_folds,
                backward_n_folds=backward_n_folds,
                alpha=alpha,
                load_if_exists=load_if_exists,
                verbose=verbose,
                use_neural_coupling=use_neural_coupling,
                cv_mode=cv_mode,
                buffer_samples=buffer_samples,
            )

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
