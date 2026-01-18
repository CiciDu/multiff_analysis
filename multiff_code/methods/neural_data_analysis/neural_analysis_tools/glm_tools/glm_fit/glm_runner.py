import os
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt
from typing import Dict

from neural_data_analysis.design_kits.design_around_event import design_checks
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import glm_fit_utils
from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import general_glm_fit
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_plotting import compare_glm_fit


class GLMPipeline:
    """
    Pipeline for fitting and comparing Poisson GLMs for stop-related activity
    with behavioral covariates and spike-history terms.
    """

    def __init__(
        self,
        *,
        spikes_df,
        bin_df,
        df_X,
        df_Y,
        meta_groups,
        bin_width: float,
        output_root: str = None,
        spike_history_t_max: float = 0.20,
        cov_type: str = 'HC1',
        cv_splitter=None  # 'blocked_time_buffered'
    ):
        self.spikes_df = spikes_df
        self.bin_df = bin_df
        self.df_X = df_X
        self.df_Y = df_Y
        self.meta_groups = meta_groups
        self.bin_width = bin_width
        self.spike_history_t_max = spike_history_t_max
        self.cov_type = cov_type
        self.output_root = output_root
        self.cv_splitter = cv_splitter

        # Derived quantities
        exposure = np.repeat(self.bin_width, df_X.shape[0])
        self.offset_log = np.log(exposure)

        # Containers
        self.design_w_history = None
        self.reports: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Design helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prune_design(X, skip_vif=False):
        """Run VIF-based pruning."""
        X_pruned, vif_report = design_checks.check_design(X, skip_vif=skip_vif)
        return X_pruned, vif_report

    def load_or_prune_columns(self, X, cols_path, skip_vif=False, pruned_columns_exists_ok: bool = True):
        """
        Load selected columns from disk if they exist;
        otherwise prune and save.
        """
        X_out, vif_report = design_checks.load_or_compute_selected_cols(
            X, cols_path, skip_vif=skip_vif, exists_ok=pruned_columns_exists_ok)

        return X_out

    # ------------------------------------------------------------------
    # Spike-history design
    # ------------------------------------------------------------------

    def build_design_with_history(self):
        """
        Construct the full design matrix including spike-history terms.
        """
        (
            self.design_w_history,
            self.basis,
            self.colnames,
            self.meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.spikes_df,
            bin_df=self.bin_df,
            X_pruned=self.df_X,
            meta_groups=self.meta_groups,
            dt=self.bin_width,
            t_max=self.spike_history_t_max,
        )

        return self.design_w_history

    # ------------------------------------------------------------------
    # GLM fitting
    # ------------------------------------------------------------------

    def fit_glm(self, name, df_X, *, use_groups: bool, glm_results_exists_ok: bool = True):
        """
        Fit a Poisson GLM and store the report.
        """
        importlib.reload(glm_fit_utils)
        importlib.reload(general_glm_fit)

        subfolder_name = name.replace(' ', '_').replace('+', 'plus').lower()

        self.save_dir = os.path.join(self.output_root, 'glm_fit', subfolder_name)
        self.fig_dir = self.save_dir.replace('all_monkey_data/', 'figures/')
        print('Calling glm_mini_report with save_dir: ', self.save_dir)
        report = general_glm_fit.glm_mini_report(
            df_X=df_X,
            df_Y=self.df_Y,
            offset_log=self.offset_log,
            cov_type=self.cov_type,
            fast_mle=True,
            do_inference=True,
            make_plots=True,
            show_plots=True,
            fig_dir=self.fig_dir,
            meta_groups=self.meta_groups if use_groups else None,
            save_dir=self.save_dir,
            cv_splitter=self.cv_splitter,
            buffer_bins=250,
            exists_ok=glm_results_exists_ok
        )

        self.reports[name] = report
        return report

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_behavior_only(self,
                          glm_results_exists_ok=True,
                          pruned_columns_exists_ok=True):
        """Fit behavior-only GLM."""
        cols_path = os.path.join(
            self.output_root,
            'selected_predictors',
            'behavior_only.json',
        )

        self.X_behavior = self.load_or_prune_columns(
            self.df_X,
            cols_path,
            skip_vif=False,
            pruned_columns_exists_ok=pruned_columns_exists_ok,
        )

        self.fit_glm(
            name='Behavior only',
            df_X=self.X_behavior,
            use_groups=False,
            glm_results_exists_ok=glm_results_exists_ok,
        )

    def run_behavior_plus_history(self,
                                  glm_results_exists_ok=True,
                                  pruned_columns_exists_ok=True):
        """Fit GLM with behavior and spike-history covariates."""
        # Build spike-history design once
        self.build_design_with_history()

        cols_path = os.path.join(
            self.output_root,
            'selected_predictors',
            'behavior_plus_history.json',
        )

        self.X_beh_hist = self.load_or_prune_columns(
            self.design_w_history,
            cols_path,
            skip_vif=True,
            pruned_columns_exists_ok=pruned_columns_exists_ok,
        )

        self.fit_glm(
            name='Behavior + history',
            df_X=self.X_beh_hist,
            use_groups=True,
            glm_results_exists_ok=glm_results_exists_ok,
        )

    def run_history_only(self, glm_results_exists_ok=True, pruned_columns_exists_ok=True):
        """Fit GLM with spike-history covariates only."""
        all_history_cols = [
            c for c in self.design_w_history.columns
            if c.startswith('cluster_')
            and c not in self.df_X.columns
        ]

        cols_path = os.path.join(
            self.output_root,
            'selected_predictors',
            'history_only.json',
        )

        self.X_hist_only = self.load_or_prune_columns(
            self.design_w_history[all_history_cols],
            cols_path,
            skip_vif=True,
            pruned_columns_exists_ok=pruned_columns_exists_ok,
        )

        self.fit_glm(
            name='History only',
            df_X=self.X_hist_only,
            use_groups=True,
            glm_results_exists_ok=glm_results_exists_ok,
        )

    def run(self,
            glm_results_exists_ok=True,
            pruned_columns_exists_ok=True):
        """
        Run all three models:
        1) Behavior only
        2) Behavior + spike history
        3) Spike history only
        """
        self.run_behavior_only(glm_results_exists_ok=glm_results_exists_ok,
                               pruned_columns_exists_ok=pruned_columns_exists_ok)
        self.run_behavior_plus_history(
            glm_results_exists_ok=glm_results_exists_ok, pruned_columns_exists_ok=pruned_columns_exists_ok)
        self.run_history_only(glm_results_exists_ok=glm_results_exists_ok,
                              pruned_columns_exists_ok=pruned_columns_exists_ok)

        return self.reports

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------

    def plot_comparisons(self, save_dir=None):
        """
        Plot in-sample and CV comparisons across fitted models.
        """
        metrics_by_model = {
            name: report['metrics_df']
            for name, report in self.reports.items()
        }
        # In-sample comparison
        fig_insample, _ = compare_glm_fit.plot_insample_model_comparison(
            metrics_by_model,
            show=False,
        )
        
        # Cross-validated comparison
        compare_glm_fit.plot_cv_model_comparison(
            metrics_by_model,
            show=False,
        )
        fig_cv = plt.gcf()
        
        if save_dir is None and self.fig_dir is not None:
            save_dir = self.fig_dir

        if save_dir is not None:
            fig_insample.savefig(
                os.path.join(save_dir, 'insample.png'),
                dpi=150,
                bbox_inches='tight',
            )

            fig_cv.savefig(
                os.path.join(save_dir, 'cv.png'),
                dpi=150,
                bbox_inches='tight',
            )