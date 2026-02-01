# -*- coding: utf-8 -*-
# Poisson GLM per cluster with optional regularization + GroupKFold CV
# -------------------------------------------------------------------
# Highlights
# - Regularization via statsmodels.GLM.fit_regularized (L1/L2/Elastic-Net).
# - Hyper-parameter tuning (alpha × l1_wt) using GroupKFold or KFold.
# - Optional L1 "refit on support" to recover SE/p-values for inference.
# - NaN-robust FDR (penalized fits may lack SE/p).
# - Progress printing per cluster and best hyper-params per cluster.
#
# API compatibility
# - Public functions preserved: add_fdr, add_rate_ratios, term_population_tests,
#   fit_poisson_glm_per_cluster, glm_mini_report
# - New knobs are optional and default to your original (no penalty).
from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import glm_fit_utils
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_plotting import plot_spikes, plot_glm_fit

from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from scipy import stats as _stats
import pickle
import hashlib
import json
import os

# ---------- small helpers (place near the top of the file) ----------


def _make_zero_unit_row(cid, n, alpha_val, l1_wt_val, condX):
    """Return a metrics row dict for an all-zero unit with diagnostics/flags."""
    return {
        'cluster': cid, 'n_obs': n,
        'deviance': np.nan, 'null_deviance': np.nan,
        'llf': np.nan, 'llnull': np.nan,
        'deviance_explained': np.nan, 'mcfadden_R2': np.nan,
        'alpha': float(alpha_val), 'l1_wt': float(l1_wt_val),
        # ---- flags/diagnostics ----
        'skipped_zero_unit': True,
        'converged': False,
        'used_ridge_fallback': False,
        'convergence_message': 'all_zero_unit',
        'nonzeros': 0,
        'zero_frac': 1.0,
        'condX': float(condX),
        'offender': True,
    }


def _flag_and_update_metrics_row(mr, res, y, condX):
    """Augment a metrics-row dict (in-place) with convergence/diagnostics and offender flag."""
    used_ridge = bool(getattr(res, 'used_ridge_fallback', False)
                      or getattr(res, 'is_penalized', False))
    conv_msg = getattr(res, 'convergence_message', 'unknown')
    conv_bool = bool(getattr(res, 'converged', True))
    nz = int(np.sum(y > 0))
    zero_frac = float(np.mean(y == 0))

    mr.update({
        'converged': conv_bool,
        'used_ridge_fallback': used_ridge,
        'convergence_message': conv_msg,
        'nonzeros': nz,
        'zero_frac': zero_frac,
        'condX': float(condX),
    })
    mr['offender'] = (not mr['converged']) or mr['used_ridge_fallback'] or (
        mr['nonzeros'] < 3) or (mr['condX'] > 1e8)


def _compute_cv_loglik_and_deviance(y, X, off, folds, *, maxiter=25):
    """
    Compute grouped K-Fold cross-validated metrics for a Poisson GLM without regularization.
    Returns (cv_loglik_improvement, cv_deviance_explained) where:
      - cv_loglik_improvement = sum_over_folds[ ll(model) - ll(null) ] on held-out data
      - cv_deviance_explained = 1 - (sum dev_model) / (sum dev_null) on held-out data
    """
    fam = sm.families.Poisson()
    total_ll_model = 0.0
    total_ll_null = 0.0
    total_dev_model = 0.0
    total_dev_null = 0.0

    for tr_idx, te_idx in folds:
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        X_tr = X[tr_idx, :]
        X_te = X[te_idx, :]
        off_tr = None if off is None else off[tr_idx]
        off_te = None if off is None else off[te_idx]

        model = sm.GLM(y_tr, X_tr, family=fam, offset=off_tr)
        # Fast, unpenalized IRLS; avoid heavy cov computations
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            try:
                res_tr = glm_fit_utils.fit_with_fallback(
                    model, cov_type='nonrobust',
                    use_overdispersion_scale=False,
                    maxiter=maxiter, try_unpenalized_refit=False
                )
            except Exception:
                # last-resort trivial fit to avoid crashing CV (rare)
                res_tr = model.fit(maxiter=maxiter, disp=0)

        # Predict mean rate on held-out fold
        mu_te = res_tr.predict(exog=X_te, offset=off_te, linear=False)
        mu_te = np.clip(mu_te, 1e-12, np.inf)

        # Null model on training: intercept-only with offset
        # beta0_hat = log(sum(y_tr) / sum(exp(off_tr))) when offset present,
        # otherwise beta0_hat = log(mean(y_tr))
        if off_tr is None:
            rate_tr = float(np.mean(y_tr))
            mu_null_te = np.full_like(y_te, fill_value=max(rate_tr, 1e-12), dtype=float)
        else:
            sum_exp_off_tr = float(np.sum(np.exp(off_tr)))
            base_rate = float(np.sum(y_tr)) / max(sum_exp_off_tr, 1e-12)
            mu_null_te = base_rate * np.exp(off_te)
        mu_null_te = np.clip(mu_null_te, 1e-12, np.inf)

        # Accumulate held-out metrics (model vs null)
        ll_te_model = np.sum(fam.loglike_obs(y_te, mu_te))
        ll_te_null = np.sum(fam.loglike_obs(y_te, mu_null_te))
        dev_te_model = float(fam.deviance(y_te, mu_te))
        dev_te_null = float(fam.deviance(y_te, mu_null_te))

        total_ll_model += float(ll_te_model)
        total_ll_null += float(ll_te_null)
        total_dev_model += float(dev_te_model)
        total_dev_null += float(dev_te_null)

    cv_ll_improvement = float(total_ll_model - total_ll_null)
    cv_dev_explained = np.nan
    if np.isfinite(total_dev_null) and (total_dev_null > 0):
        cv_dev_explained = 1.0 - (float(total_dev_model) / float(total_dev_null))
    return cv_ll_improvement, cv_dev_explained


def record_cluster_outcomes(
    *, cid, y, n, res, alpha_val, l1_wt_val, feature_names, X, off, condX,
    results, coef_rows, metrics_rows, used_refit=False
):
    """Common post-fit bookkeeping for both MLE and CV paths."""
    llf, llnull, dev, dev0 = glm_fit_utils.metrics_from_result(res, y, X, off)
    # coefficients table (SE/p may be NaN for penalized fits)
    coef_rows.extend(
        glm_fit_utils.collect_coef_rows(feature_names, cid, res, alpha=0.0,
                                        l1_wt=0.0, used_refit=False)
    )

    # metrics table
    mr = glm_fit_utils.collect_metric_row(cid, n, llf, llnull, dev,
                                          dev0, alpha_val, l1_wt_val)
    _flag_and_update_metrics_row(mr, res, y, condX)
    metrics_rows.append(mr)
    # stash result
    results[cid] = res


# ---------- core fitting (refactored with MLE short-circuit) ----------
def fit_poisson_glm_per_cluster(
    df_X,
    df_Y,
    offset_log,
    cluster_ids=None,
    cov_type='HC1',
    regularization='none',
    alpha_grid=(0.1, 0.3, 1.0),
    l1_wt_grid=(1.0, 0.5, 0.0),
    n_splits=5,
    cv_metric='loglik',
    groups=None,
    refit_on_support=True,
    return_cv_tables=True,
    compute_cv_metrics: bool = True,
    *,
    cv_splitter=None,
    buffer_bins: int = 250,
    use_overdispersion_scale=False
):
    """Fit Poisson GLMs per cluster with optional Elastic-Net and CV."""
    feature_names, X, off, n, cluster_ids = glm_fit_utils._validate_shapes(
        df_X, df_Y, offset_log, cluster_ids)
    condX_once = float(np.linalg.cond(np.asarray(df_X, float)))

    # -------- MLE short-circuit (no tuning, no folds, no CV tables) --------
    no_tuning = (regularization == 'none' and tuple(alpha_grid)
                 == (0.0,) and tuple(l1_wt_grid) == (0.0,))
    if no_tuning:
        results, coef_rows, metrics_rows = {}, [], []
        # Build grouped folds for CV metrics if requested
        folds = None
        if compute_cv_metrics:
            folds =_build_folds(
                n, n_splits=n_splits, groups=groups, cv_splitter=cv_splitter, buffer_bins=buffer_bins)
        for i, cid in enumerate(cluster_ids, 1):
            print(
                f'Fitting cluster {i}/{len(cluster_ids)}: {cid} ...', flush=True)
            y = df_Y[cid].to_numpy()

            if np.all(y == 0):
                metrics_rows.append(_make_zero_unit_row(
                    cid, n, alpha_val=0.0, l1_wt_val=0.0, condX=condX_once))
                # add CV columns with NaN to keep schema consistent
                metrics_rows[-1]['cv_loglik_improvement'] = np.nan
                metrics_rows[-1]['cv_deviance_explained'] = np.nan
                continue

            # unpenalized single fit (with your fallback inside _fit_once)
            res = glm_fit_utils._fit_once(
                y=y, X=X, off=off, cov_type=cov_type,
                regularization='none', alpha=0.0, l1_wt=0.0,
                use_overdispersion_scale=use_overdispersion_scale,
                feature_names=feature_names
            )
            # Compute grouped K-Fold CV metrics for this cluster
            if compute_cv_metrics and folds is not None:
                try:
                    cv_ll_imp, cv_dev_expl = _compute_cv_loglik_and_deviance(
                        y, X, off, folds)
                except Exception:
                    cv_ll_imp, cv_dev_expl = np.nan, np.nan
            else:
                cv_ll_imp, cv_dev_expl = np.nan, np.nan
            record_cluster_outcomes(
                cid=cid, y=y, n=n, res=res,
                alpha_val=0.0, l1_wt_val=0.0,
                feature_names=feature_names, X=X, off=off, condX=condX_once,
                results=results, coef_rows=coef_rows, metrics_rows=metrics_rows, used_refit=False
            )
            # attach CV metrics to the latest metrics row
            metrics_rows[-1]['cv_loglik_improvement'] = cv_ll_imp
            metrics_rows[-1]['cv_deviance_explained'] = cv_dev_expl
            print('  done (MLE)', flush=True)

        return pd.Series(results).to_dict(), pd.DataFrame(coef_rows), pd.DataFrame(metrics_rows), pd.DataFrame()

    # ----------------------- regular path with tuning ------------------------
    folds =_build_folds(n, n_splits=n_splits,
                                       groups=groups, cv_splitter=cv_splitter, buffer_bins=buffer_bins)
    results, coef_rows, metrics_rows, cv_tables = {}, [], [], []

    for i, cid in enumerate(cluster_ids, 1):
        print(f'Fitting cluster {i}/{len(cluster_ids)}: {cid} ...', flush=True)
        y = df_Y[cid].to_numpy()

        if np.all(y == 0):
            metrics_rows.append(_make_zero_unit_row(
                cid, n, alpha_val=np.nan, l1_wt_val=np.nan, condX=condX_once))
            continue

        # Hyperparam search → best full fit
        best, cv_table = glm_fit_utils._hyperparam_search(
            y=y, X=X, off=off, folds=folds, cov_type=cov_type,
            regularization=regularization,
            alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid,
            cv_metric=cv_metric, return_table=True,
            use_overdispersion_scale=use_overdispersion_scale,
            feature_names=feature_names
        )
        res = best['res']

        # Optional L1 refit on support to recover SE/p
        res, used_refit = glm_fit_utils._refit_l1_support_if_needed(
            res, y, X, off, cov_type,
            regularization, best['alpha'], best['l1_wt'],
            refit_on_support, use_overdispersion_scale=use_overdispersion_scale
        )

        record_cluster_outcomes(
            cid=cid, y=y, n=n, res=res,
            alpha_val=best['alpha'], l1_wt_val=best['l1_wt'],
            feature_names=feature_names, X=X, off=off, condX=condX_once,
            results=results, coef_rows=coef_rows, metrics_rows=metrics_rows, used_refit=used_refit
        )

        if return_cv_tables:
            t = cv_table.copy()
            t['cluster'] = cid
            cv_tables.append(t)

        print(
            f'  done (best alpha={best["alpha"]}, l1_wt={best["l1_wt"]})', flush=True)

    coef_df = pd.DataFrame(coef_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    if return_cv_tables:
        cv_tables_df = pd.concat(cv_tables, ignore_index=True) if len(
            cv_tables) else pd.DataFrame()
        return pd.Series(results).to_dict(), coef_df, metrics_df, cv_tables_df
    return pd.Series(results).to_dict(), coef_df, metrics_df, pd.DataFrame()


def fit_poisson_glm_per_cluster_fast_mle(
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,
    offset_log,
    cluster_ids=None,
    cov_type: str = 'HC1',    # use 'nonrobust' for max speed
    maxiter: int = 100,
    show_progress: bool = False,
    n_splits: int = 5,
    groups=None,
    cv_splitter=None,
    buffer_bins: int = 250,
    compute_cv_metrics: bool = True,
):
    """
    Ultra-fast MLE per cluster: no CV/inference/plots.
    Returns (results_dict, coefs_df, metrics_df).
    """
    feature_names = list(df_X.columns)
    X = np.asarray(df_X, dtype=float, order='F')
    off = None if offset_log is None else np.asarray(offset_log, dtype=float)
    condX_once = float(np.linalg.cond(np.asarray(df_X, float)))
    eff_ids = list(df_Y.columns) if cluster_ids is None else list(cluster_ids)

    results, coef_rows, metrics_rows = {}, [], []
    fam = sm.families.Poisson()
    # Precompute grouped folds for CV metrics if requested
    n = X.shape[0]
    folds = None
    if compute_cv_metrics:
        folds =_build_folds(
            n, n_splits=n_splits, groups=groups, cv_splitter=cv_splitter, buffer_bins=buffer_bins)

    for i, cid in enumerate(eff_ids, 1):
        if show_progress:
            print(f'Fitting cluster {i}/{len(eff_ids)}: {cid} ...', flush=True)

        y = np.asarray(df_Y[cid], dtype=float)
        n_obs = y.shape[0]

        # All-zero unit → flagged metrics row, skip fit
        if np.all(y == 0):
            metrics_rows.append(_make_zero_unit_row(
                cid, n_obs, alpha_val=0.0, l1_wt_val=0.0, condX=condX_once))
            metrics_rows[-1]['cv_loglik_improvement'] = np.nan
            metrics_rows[-1]['cv_deviance_explained'] = np.nan
            if show_progress:
                print('  skipped (all-zero unit)', flush=True)
            continue

        # Unpenalized Newton/IRLS with robust fallback (tiny ridge), then (try) unpen refit
        model = sm.GLM(y, X, family=fam, offset=off)
        glm_fit_utils.attach_feature_names(model, feature_names)
        res = glm_fit_utils.fit_with_fallback(
            model, cov_type=cov_type, use_overdispersion_scale=False,
            maxiter=maxiter, try_unpenalized_refit=True
        )

        # Grouped K-Fold CV metrics for this cluster
        if compute_cv_metrics and folds is not None:
            try:
                cv_ll_imp, cv_dev_expl = _compute_cv_loglik_and_deviance(
                    y, X, off, folds)
            except Exception:
                cv_ll_imp, cv_dev_expl = np.nan, np.nan
        else:
            cv_ll_imp, cv_dev_expl = np.nan, np.nan

        # Bookkeeping identical to CV path (adds coef/metrics + flags)
        record_cluster_outcomes(
            cid=cid, y=y, n=n_obs, res=res,
            alpha_val=0.0, l1_wt_val=0.0,
            feature_names=feature_names, X=X, off=off, condX=condX_once,
            results=results, coef_rows=coef_rows, metrics_rows=metrics_rows, used_refit=False
        )
        metrics_rows[-1]['cv_loglik_improvement'] = cv_ll_imp
        metrics_rows[-1]['cv_deviance_explained'] = cv_dev_expl

        if show_progress:
            print('  done (fast MLE)', flush=True)

    coefs_df = pd.DataFrame.from_records(coef_rows)
    metrics_df = pd.DataFrame.from_records(metrics_rows)

    # ---------- safety net: enforce required columns ----------
    # We expect one coef row per (cluster, feature).
    if not coefs_df.empty:
        # Rebuild 'cluster' if missing (uses insertion order of results)
        if 'cluster' not in coefs_df.columns:
            fitted_ids = list(results.keys())  # insertion order preserved
            p = len(feature_names)
            expected_len = p * len(fitted_ids)
            if expected_len == len(coefs_df):
                coefs_df.insert(0, 'cluster', np.repeat(fitted_ids, p))
            else:
                # last-resort: at least avoid hard crash later
                coefs_df.insert(0, 'cluster', np.nan)
                print(
                    '[fast_mle] WARNING: could not reconstruct cluster column deterministically.')

        # Rebuild 'term' if missing (feature name per row)
        if 'term' not in coefs_df.columns:
            fitted_ids = list(results.keys())
            p = len(feature_names)
            expected_len = p * len(fitted_ids)
            if expected_len == len(coefs_df):
                coefs_df['term'] = np.tile(feature_names, len(fitted_ids))
            else:
                coefs_df['term'] = pd.RangeIndex(len(coefs_df)).astype(str)
                print(
                    '[fast_mle] WARNING: could not reconstruct term column deterministically.')

    return results, coefs_df, metrics_df


# ---------- slim helpers (top-level in this module) ----------

def _resolve_inputs(df_X, df_Y, feature_names, cluster_ids):
    feats = list(df_X.columns) if feature_names is None else list(
        feature_names)
    clust = list(df_Y.columns) if cluster_ids is None else list(cluster_ids)
    return feats, clust


def _fit_path(
    *, df_X, df_Y, offset_log, eff_clusters, cov_type,
    fast_mle, regularization, alpha_grid, l1_wt_grid,
    n_splits, cv_metric, groups, refit_on_support,
    cv_splitter, buffer_bins, use_overdispersion_scale, return_cv_tables,
    compute_cv_metrics,
    show_progress
):
    if fast_mle:
        results, coefs_df, metrics_df = fit_poisson_glm_per_cluster_fast_mle(
            df_X=df_X, df_Y=df_Y, offset_log=offset_log,
            cluster_ids=eff_clusters, cov_type=cov_type,
            maxiter=100, show_progress=show_progress,
            n_splits=n_splits, groups=groups, cv_splitter=cv_splitter, buffer_bins=buffer_bins,
            compute_cv_metrics=compute_cv_metrics,
        )
        return results, coefs_df, metrics_df, pd.DataFrame()

    no_tuning = (regularization == 'none'
                 and tuple(alpha_grid) == (0.0,)
                 and tuple(l1_wt_grid) == (0.0,))
    rcv = bool(return_cv_tables and not no_tuning)
    return fit_poisson_glm_per_cluster(
        df_X=df_X, df_Y=df_Y, offset_log=offset_log,
        cluster_ids=eff_clusters, cov_type=cov_type,
        regularization=regularization, alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid,
        n_splits=n_splits, cv_metric=cv_metric, groups=groups,
        refit_on_support=refit_on_support, cv_splitter=cv_splitter, buffer_bins=buffer_bins,
        use_overdispersion_scale=use_overdispersion_scale, return_cv_tables=rcv,
        compute_cv_metrics=compute_cv_metrics,
    )


def _run_inference(coefs_df, *, do_inference, make_plots, alpha, delta_for_rr, term_groups=None):
    """
    Robust inference step.
    - Ensures a usable 'p' column (compute from z or coef/se if missing).
    - Uses 'group' column (if present/constructed) for FDR grouping; falls back to global FDR otherwise.
    - add_rate_ratios only if 'coef' exists (adds 'se' as NaN if missing).
    """
    df = coefs_df.copy()

    # ---- ensure p-values exist (compute if needed) ----
    need_p = ('p' not in df.columns) or (
        not np.isfinite(df.get('p', np.nan)).any())
    if need_p:
        # try z → p
        z = df.get('z', None)
        if z is not None and np.isfinite(z).any():
            df['p'] = 2.0 * _stats.norm.sf(np.abs(z))
        else:
            # try coef/se → p
            if ('coef' in df.columns) and ('se' in df.columns):
                se = df['se'].to_numpy() if 'se' in df else np.full(
                    len(df), np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = df['coef'] / se
                df['z'] = z
                df['p'] = 2.0 * _stats.norm.sf(np.abs(z))
            else:
                # last resort: create NaN p's so add_fdr can still run
                df['p'] = np.nan

    # ---- build 'group' column for FDR grouping, keeping 'term' strictly as feature name ----
    has_term = ('term' in df.columns)
    if has_term:
        col_to_group = None
        if term_groups is not None:
            try:
                # term_groups expected as {group_label: [column_names...]}
                col_to_group = {c: t for t, cols in dict(
                    term_groups).items() for c in list(cols)}
            except Exception:
                col_to_group = None
        if col_to_group:
            df['group'] = df['term'].map(col_to_group).fillna(df['term'])
            print('Using group mapping from meta_groups', flush=True)
        else:
            df['group'] = glm_fit_utils.normalize_term_labels(df['term'])
    if do_inference:
        try:
            df = glm_fit_utils.add_fdr(df, alpha=alpha, by_term=(
                'group' in df.columns), group_col='group')
        except KeyError:
            # safety net: global FDR
            df = glm_fit_utils.add_fdr(df, alpha=alpha, by_term=False)

        # add rate ratios if possible
        if 'coef' in df.columns:
            if 'se' not in df.columns:
                df['se'] = np.nan
            df = glm_fit_utils.add_rate_ratios(df, delta=delta_for_rr)
        pop = glm_fit_utils.term_population_tests(
            df) if has_term else pd.DataFrame()
        return df, pop

    # not doing inference, but ensure FDR exists for plotting legends
    if make_plots and ('sig_FDR' not in df.columns):
        try:
            df = glm_fit_utils.add_fdr(df, alpha=alpha, by_term=(
                'group' in df.columns), group_col='group')
        except KeyError:
            df = glm_fit_utils.add_fdr(df, alpha=alpha, by_term=False)

    return df, pd.DataFrame()


def _pick_forest_term(feature_names, forest_term):
    if not feature_names:
        return None
    if forest_term is None:
        return feature_names[0]
    return forest_term if forest_term in feature_names else feature_names[0]


def _build_figs(coefs_df, metrics_df, *, feature_names, forest_term, forest_top_n, delta_for_rr, make_plots):
    # Currently, a lot of the plots are not useful for our purposes

    if not make_plots:
        return {}
    figs = {
        'coef_dists': plot_glm_fit.plot_coef_distributions(coefs_df),
        # 'model_quality': plot_glm_fit.plot_model_quality(metrics_df),
    }

    # ft = _pick_forest_term(feature_names, forest_term)
    # if ft is not None:
    #     figs['forest'] = plot_glm_fit.plot_forest_for_term(
    #         coefs_df, term=ft, top_n=forest_top_n)
    #     figs['rr_hist'] = plot_glm_fit.plot_rate_ratio_hist(
    #         coefs_df, term=ft, delta=delta_for_rr, bins=40, log=True, clip_q=0.995
    #     )

    # else:
    #     figs['forest'] = None
    #     figs['rr_hist'] = None
    return figs


def _save_outputs(save_dir, coefs_df, metrics_df, pop_tests, figs, *, make_plots, cv_tables_df=None, results=None, metadata: dict | None = None,
                  fig_dir=None, session_id=None):
    if save_dir is None:
        return
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    coefs_df.to_csv(p / 'coefs.csv', index=False)
    metrics_df.to_csv(p / 'metrics.csv', index=False)
    pop_tests.to_csv(p / 'population_tests.csv', index=False)
    # save metadata sidecar for robust retrieval
    if metadata is not None:
        try:
            with open(p / 'meta.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f'[save_outputs] WARNING: could not save meta.json: {type(e).__name__}: {e}')
    # save optional CV tables
    if cv_tables_df is not None and isinstance(cv_tables_df, pd.DataFrame) and not cv_tables_df.empty:
        cv_tables_df.to_csv(p / 'cv_tables.csv', index=False)
    # try to persist model results if provided (may be large)
    if results is not None:
        try:
            with open(p / 'results.pkl', 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'[save_outputs] Saved results.pkl to {p / "results.pkl"}')
        except Exception as e:
            print(f'[save_outputs] WARNING: could not save results.pkl: {type(e).__name__}: {e}')
    if make_plots and fig_dir is not None:
        save_figs(figs, fig_dir, session_id=session_id)
        


def save_figs(figs, fig_dir, session_id=None, dpi=150):
    fig_dir = Path(fig_dir)

    for name, fig in figs.items():
        if fig is None:
            continue

        if session_id is not None:
            out_dir = fig_dir / name
            fname = f'{session_id}.png'
        else:
            out_dir = fig_dir
            fname = f'{name}.png'

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / fname, dpi=dpi, bbox_inches='tight')


def _load_outputs(save_dir, *, expected_hash: str | None = None):
    """
    Best-effort loader for previously saved report artifacts.
    Returns tuple (loaded, payload) where:
      - loaded: bool indicating whether anything substantive was loaded
      - payload: dict mirroring glm_mini_report's return schema
    """
    if save_dir is None:
        return False, {}
    p = Path(save_dir)
    coefs_fp = p / 'coefs.csv'
    metrics_fp = p / 'metrics.csv'
    pop_fp = p / 'population_tests.csv'
    cv_fp = p / 'cv_tables.csv'
    res_fp = p / 'results.pkl'
    meta_fp = p / 'meta.json'

    loaded_any = False
    coefs_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    pop_df = pd.DataFrame()
    cv_df = pd.DataFrame()
    results = {}
    meta = {}

    # If an expected hash is provided, require meta.json to exist and match
    if expected_hash is not None:
        try:
            if meta_fp.exists():
                with open(meta_fp, 'r') as f:
                    meta = json.load(f)
                if meta.get('params_hash') != expected_hash:
                    return False, {}
            else:
                return False, {}
        except Exception as e:
            print(f'[load_outputs] WARNING: could not verify meta.json: {type(e).__name__}: {e}')
            return False, {}

    try:
        if coefs_fp.exists():
            coefs_df = pd.read_csv(coefs_fp)
            loaded_any = True
    except Exception as e:
        print(f'[load_outputs] WARNING: could not load {coefs_fp.name}: {type(e).__name__}: {e}')
    try:
        if metrics_fp.exists():
            metrics_df = pd.read_csv(metrics_fp)
            loaded_any = True
    except Exception as e:
        print(f'[load_outputs] WARNING: could not load {metrics_fp.name}: {type(e).__name__}: {e}')
    try:
        if pop_fp.exists():
            pop_df = pd.read_csv(pop_fp)
            loaded_any = True
    except Exception as e:
        print(f'[load_outputs] WARNING: could not load {pop_fp.name}: {type(e).__name__}: {e}')
    try:
        if cv_fp.exists():
            cv_df = pd.read_csv(cv_fp)
    except Exception as e:
        print(f'[load_outputs] WARNING: could not load {cv_fp.name}: {type(e).__name__}: {e}')
    try:
        if res_fp.exists():
            with open(res_fp, 'rb') as f:
                results = pickle.load(f)
    except Exception as e:
        print(f'[load_outputs] WARNING: could not load {res_fp.name}: {type(e).__name__}: {e}')

    offenders_df = _compute_offenders(metrics_df) if not metrics_df.empty else pd.DataFrame()
    # Figures are not reconstructed on load; provide empty dict
    figs = {}
    payload = {
        'results': results,
        'coefs_df': coefs_df,
        'metrics_df': metrics_df,
        'population_tests_df': pop_df,
        'figures': figs,
        'cv_tables_df': cv_df,
        'offenders_df': offenders_df,
        'meta': meta,
    }
    return loaded_any, payload


def _compute_params_hash_for_report(
    *,
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,
    offset_log,
    feature_names,
    eff_clusters,
    cov_type: str,
    fast_mle: bool,
    regularization: str,
    alpha_grid,
    l1_wt_grid,
    n_splits: int,
    cv_metric: str,
    refit_on_support: bool,
    cv_splitter,
    buffer_bins: int,
    use_overdispersion_scale: bool,
    return_cv_tables: bool,
    compute_cv_metrics: bool,
    make_plots: bool,
    do_inference: bool,
):
    """
    Build a stable parameters hash to guard cache retrieval.
    Uses shapes, selected features/clusters, and relevant knobs (not raw data).
    """
    n_samples = int(len(df_X))
    n_features = int(len(feature_names))
    n_clusters = int(len(eff_clusters))
    off_present = offset_log is not None
    off_stats = None
    if off_present:
        try:
            off_arr = np.asarray(offset_log, dtype=float).ravel()
            off_stats = {
                'n': int(off_arr.size),
                'sum': float(np.sum(off_arr)),
                'mean': float(np.mean(off_arr)),
                'std': float(np.std(off_arr)),
            }
        except Exception:
            off_stats = {'present': True}

    hash_payload = {
        'version': 1,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_clusters': n_clusters,
        'feature_names': list(feature_names),
        'cluster_ids': list(eff_clusters),
        'cov_type': cov_type,
        'fast_mle': bool(fast_mle),
        'regularization': str(regularization),
        'alpha_grid': list(alpha_grid) if alpha_grid is not None else None,
        'l1_wt_grid': list(l1_wt_grid) if l1_wt_grid is not None else None,
        'n_splits': int(n_splits),
        'cv_metric': str(cv_metric),
        'refit_on_support': bool(refit_on_support),
        'use_overdispersion_scale': bool(use_overdispersion_scale),
        'return_cv_tables': bool(return_cv_tables),
        'compute_cv_metrics': bool(compute_cv_metrics),
        'make_plots': bool(make_plots),
        'do_inference': bool(do_inference),
        'offset_stats': off_stats if off_present else None,
        'cv_splitter_class': None if cv_splitter is None else cv_splitter.__class__.__name__,
        'buffer_bins': int(buffer_bins),
    }
    params_hash = hashlib.sha1(
        json.dumps(hash_payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:10]
    metadata = {
        'params_hash': params_hash,
        'summary': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
        },
        'config': {k: hash_payload[k] for k in hash_payload if k not in ('feature_names', 'cluster_ids')},
    }
    return params_hash, metadata

def _show_or_close(figs, *, make_plots, show_plots):
    if not make_plots:
        return
    if show_plots:
        for fig in figs.values():
            if fig is not None:
                plt.show()
    else:
        for fig in figs.values():
            if fig is not None:
                plt.close(fig)


def _compute_offenders(metrics_df):
    m = metrics_df.copy()
    # guarantee columns exist so selection never KeyErrors
    defaults = {
        'skipped_zero_unit': False, 'converged': True, 'used_ridge_fallback': False,
        'nonzeros': np.nan, 'zero_frac': np.nan, 'condX': np.nan, 'convergence_message': ''
    }
    for k, v in defaults.items():
        if k not in m.columns:
            m[k] = v
    mask = m['skipped_zero_unit'] | (
        ~m['converged']) | m['used_ridge_fallback']
    cols = ['cluster', 'converged', 'used_ridge_fallback', 'skipped_zero_unit',
            'nonzeros', 'zero_frac', 'condX', 'deviance_explained', 'mcfadden_R2', 'convergence_message']
    cols = [c for c in cols if c in m.columns]
    return m.loc[mask, cols].copy()


# ---------- slim public wrapper ---------------------------------------------

def glm_mini_report(
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,
    offset_log,
    feature_names=None,
    cluster_ids=None,
    meta_groups: dict | None = None,
    alpha: float = 0.05,
    delta_for_rr: float = 1.0,
    forest_term=None,
    forest_top_n: int = 30,
    cov_type: str = 'HC1',
    show_plots: bool = True,
    save_dir=None,
    fig_dir=None,
    session_id=None,
    regularization: str = 'none',
    alpha_grid=(0.1, 0.3, 1.0),
    l1_wt_grid=(1.0, 0.5, 0.0),
    n_splits: int = 5,
    cv_metric: str = 'loglik',
    groups=None,
    refit_on_support: bool = True,
    cv_splitter=None,
    buffer_bins: int = 250,
    use_overdispersion_scale: bool = False,
    fast_mle: bool = False,
    make_plots: bool = True,
    do_inference: bool = True,
    return_cv_tables: bool = True,
    show_progress: bool = False,
    compute_cv_metrics: bool = True,
    exists_ok: bool = True,
):
    """Thin orchestration wrapper: fit → inference → figs → save/show → offenders."""
    feature_names, eff_clusters = _resolve_inputs(
        df_X, df_Y, feature_names, cluster_ids)

    # Prepare params hash and attempt to load cached results if allowed
    params_hash, metadata = _compute_params_hash_for_report(
        df_X=df_X, df_Y=df_Y, offset_log=offset_log,
        feature_names=feature_names, eff_clusters=eff_clusters,
        cov_type=cov_type, fast_mle=fast_mle, regularization=regularization,
        alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid, n_splits=n_splits,
        cv_metric=cv_metric, refit_on_support=refit_on_support,
        cv_splitter=cv_splitter, buffer_bins=buffer_bins, use_overdispersion_scale=use_overdispersion_scale,
        return_cv_tables=return_cv_tables, compute_cv_metrics=compute_cv_metrics,
        make_plots=make_plots, do_inference=do_inference,
    )
    if exists_ok and save_dir is not None:
        loaded, payload = _load_outputs(save_dir, expected_hash=params_hash)
        if loaded:
            print('[glm_mini_report] Loaded cached results from save_dir (exists_ok=True, hash match).')
            return payload
        else:
            print('[glm_mini_report] No cached results found, fitting from scratch.')
    else:
        print('[glm_mini_report] Not loading cached results because exists_ok=False or save_dir is None.')

    results, coefs_df, metrics_df, cv_tables_df = _fit_path(
        df_X=df_X, df_Y=df_Y, offset_log=offset_log,
        eff_clusters=eff_clusters, cov_type=cov_type,
        fast_mle=fast_mle, regularization=regularization,
        alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid,
        n_splits=n_splits, cv_metric=cv_metric, groups=groups,
        refit_on_support=refit_on_support, cv_splitter=cv_splitter, buffer_bins=buffer_bins,
        use_overdispersion_scale=use_overdispersion_scale,
        return_cv_tables=return_cv_tables, show_progress=show_progress,
        compute_cv_metrics=compute_cv_metrics
    )

    coefs_df, pop_tests = _run_inference(
        coefs_df, do_inference=do_inference, make_plots=make_plots,
        alpha=alpha, delta_for_rr=delta_for_rr, term_groups=meta_groups
    )

    try:
        figs = _build_figs(
            coefs_df, metrics_df,
            feature_names=feature_names,
            forest_term=forest_term,
            forest_top_n=forest_top_n,
            delta_for_rr=delta_for_rr,
            make_plots=make_plots,
        )
    except Exception as e:
        print(
            f'[glm_mini_report] WARNING: could not build figures: {type(e).__name__}: {e}')
        figs = {}

    # add hash to metadata before saving
    _save_outputs(
        save_dir, coefs_df, metrics_df, pop_tests, figs,
        make_plots=make_plots, cv_tables_df=cv_tables_df, results=results,
        metadata=metadata, fig_dir=fig_dir, session_id=session_id
    )
    _show_or_close(figs, make_plots=make_plots, show_plots=show_plots)

    # summary + offenders
    try:
        n_clusters = int(metrics_df['cluster'].nunique(
        )) if 'cluster' in metrics_df.columns else len(eff_clusters)
    except Exception:
        n_clusters = len(eff_clusters)
    if fast_mle:
        print(f'[glm_mini_report] Finished fast MLE on {n_clusters} clusters, '
              f'cov_type={cov_type}, inference={do_inference}, plots={make_plots}')

    offenders_df = _compute_offenders(metrics_df)
    print(f'[glm_mini_report] offenders: {len(offenders_df)}')

    return {
        'results': results,
        'coefs_df': coefs_df,
        'metrics_df': metrics_df,
        'population_tests_df': pop_tests,
        'figures': figs,
        'cv_tables_df': cv_tables_df,
        'offenders_df': offenders_df,
    }


def _build_folds(
    n,
    *,
    n_splits=5,
    groups=None,
    cv_splitter=None,
    random_state=0,
    buffer_bins=250 # for default: 5s/0.02s = 250 bins
):
    """
    Return a list of (train_idx, valid_idx) pairs.

    cv_splitter options:
      - 'blocked_time_buffered': contiguous time blocks with buffers on both sides
      - 'blocked_time': forward-chaining (past → future)
      - groups != None: GroupKFold
      - default: shuffled KFold
    """
    idx = np.arange(n)

    # -------- BLOCKED TIME + BUFFER (recommended) --------
    if cv_splitter == 'blocked_time_buffered':
        # contiguous blocks
        edges = np.linspace(0, n, n_splits + 1, dtype=int)
        folds = []

        for k in range(n_splits):
            test_start, test_end = edges[k], edges[k + 1]

            test_idx = idx[test_start:test_end]

            # buffer region in time
            buf_start = max(0, test_start - buffer_bins)
            buf_end   = min(n, test_end + buffer_bins)
            buffer_idx = idx[buf_start:buf_end]

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_mask[buffer_idx] = False

            train_idx = idx[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            folds.append((train_idx, test_idx))

        print('cv_splitter = blocked_time_buffered: Split into contiguous blocks with buffer region')
        return folds

    # -------- FORWARD-CHAINING (causal CV) --------
    if cv_splitter == 'blocked_time':
        bps = np.linspace(0, n, n_splits + 1, dtype=int)
        folds = []
        for k in range(1, len(bps)):
            start, stop = bps[k-1], bps[k]
            train = idx[:start]
            valid = idx[start:stop]
            if len(train) and len(valid):
                folds.append((train, valid))
        print('cv_splitter = blocked_time: Forward-chaining (past → future)')
        return folds

    # -------- GROUPED CV --------
    if groups is not None:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(idx, groups=groups))

    # -------- DEFAULT (NOT recommended for time series) --------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(idx))
