"""
DecodingRunner — lightweight orchestrator (Task × Model).

Usage
-----
    from decoding_tasks import StopTask, PNTask
    from decoding_models import OneFFStyleModel, CVDecodingModel
    from decoding_runner import DecodingRunner

    # One-FF-style population decoding
    runner = DecodingRunner(StopTask(path), OneFFStyleModel())
    runner.run()

    # CV model-spec decoding
    runner = DecodingRunner(StopTask(path), CVDecodingModel())
    runner.run()

    # Swap task, keep model
    runner2 = DecodingRunner(PNTask(path), OneFFStyleModel())
    runner2.run()
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .decoding_models import BaseDecodingModel


class DecodingRunner:
    """
    Binds a (Task, Model) pair and exposes a flat call-site.

    All methods are pure pass-throughs — no logic lives here.
    External code reads task data attributes via __getattr__ fallthrough.
    """

    def __init__(self, task, model: BaseDecodingModel):
        self.task = task
        self.model = model

    # ------------------------------------------------------------------
    # Fallthrough: task data attributes available directly on runner
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.task, name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def collect_data(self, exists_ok: bool = True):
        self.task.collect_data(exists_ok=exists_ok)

    # ------------------------------------------------------------------
    # Core interface (both model types)
    # ------------------------------------------------------------------

    def run(self, **kwargs):
        return self.model.run(self.task, **kwargs)

    def fit(self, **kwargs):
        return self.run(**kwargs)

    # ------------------------------------------------------------------
    # OneFFStyleModel pass-throughs
    # ------------------------------------------------------------------

    def compute_canoncorr(self, **kwargs):
        return self.model.compute_canoncorr(self.task, **kwargs)

    def regress_popreadout(self, **kwargs):
        return self.model.regress_popreadout(self.task, **kwargs)

    def extract_corr_df(self, **kwargs):
        return self.model.extract_corr_df(**kwargs)

    def plot_canoncorr_coefficients(self, **kwargs):
        self.model.plot_canoncorr_coefficients(**kwargs)

    def plot_decoder_parity(self, **kwargs):
        self.model.plot_decoder_parity(**kwargs)

    def plot_decoder_correlation_bars(self, **kwargs):
        self.model.plot_decoder_correlation_bars(**kwargs)

    def plot_all_decoding_results(self, **kwargs):
        self.model.plot_all_decoding_results(**kwargs)

    # ------------------------------------------------------------------
    # CVDecodingModel pass-throughs
    # ------------------------------------------------------------------

    def run_cv_decoding(self, **kwargs):
        return self.model.run_cv_decoding(self.task, **kwargs)

    def run_nested_kernelwidth_cv(self, **kwargs):
        return self.model.run_nested_kernelwidth_cv(self.task, **kwargs)

    def find_true_vs_pred_cv_for_feature(self, feature: str, **kwargs):
        return self.model.find_true_vs_pred_cv_for_feature(self.task, feature, **kwargs)

    def extract_regression_feature_scores_df(self, **kwargs):
        return self.model.extract_regression_feature_scores_df(self.task, **kwargs)

    def plot_fold_tuning_info(self, **kwargs):
        self.model.plot_fold_tuning_info(self.task, **kwargs)

    def find_categorical_vars_in_binned_feats(self, **kwargs):
        return self.model.find_categorical_vars_in_binned_feats(self.task, **kwargs)

    def run_anova_for_categorical_vars(self, unit_idx: int, **kwargs):
        return self.model.run_anova_for_categorical_vars(self.task, unit_idx, **kwargs)

    def run_anova_all_neurons(self, **kwargs):
        return self.model.run_anova_all_neurons(self.task, **kwargs)

    def plot_anova_results(self, **kwargs):
        self.model.plot_anova_results(self.task, **kwargs)

    def run_lm_for_categorical_vars(self, unit_idx: int, **kwargs):
        return self.model.run_lm_for_categorical_vars(self.task, unit_idx, **kwargs)

    def run_lm_all_neurons(self, **kwargs):
        return self.model.run_lm_all_neurons(self.task, **kwargs)

    def plot_lm_vs_anova_results(self, **kwargs):
        return self.model.plot_lm_vs_anova_results(self.task, **kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DecodingRunner(\n"
            f"  task  = {type(self.task).__name__}({self.task.raw_data_folder_path!r}),\n"
            f"  model = {type(self.model).__name__},\n"
            f")"
        )
