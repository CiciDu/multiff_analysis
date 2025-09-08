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
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_fit.glm_fit_utils import *
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold


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


def record_cluster_outcomes(
    *, cid, y, n, res, alpha_val, l1_wt_val, feature_names, X, off, condX,
    results, coef_rows, metrics_rows, used_refit=False
):
    """Common post-fit bookkeeping for both MLE and CV paths."""
    llf, llnull, dev, dev0 = metrics_from_result(res, y, X, off)
    # coefficients table (SE/p may be NaN for penalized fits)
    coef_rows.extend(
        collect_coef_rows(feature_names, cid, res, alpha=0.0,
                          l1_wt=0.0, used_refit=False)
    )

    # metrics table
    mr = collect_metric_row(cid, n, llf, llnull, dev,
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
    *,
    cv_splitter=None,
    use_overdispersion_scale=False
):
    """Fit Poisson GLMs per cluster with optional Elastic-Net and CV."""
    feature_names, X, off, n, cluster_ids = _validate_shapes(
        df_X, df_Y, offset_log, cluster_ids)
    condX_once = float(np.linalg.cond(np.asarray(df_X, float)))

    # -------- MLE short-circuit (no tuning, no folds, no CV tables) --------
    no_tuning = (regularization == 'none' and tuple(alpha_grid)
                 == (0.0,) and tuple(l1_wt_grid) == (0.0,))
    if no_tuning:
        results, coef_rows, metrics_rows = {}, [], []
        for i, cid in enumerate(cluster_ids, 1):
            print(
                f'Fitting cluster {i}/{len(cluster_ids)}: {cid} ...', flush=True)
            y = df_Y[cid].to_numpy()

            if np.all(y == 0):
                metrics_rows.append(_make_zero_unit_row(
                    cid, n, alpha_val=0.0, l1_wt_val=0.0, condX=condX_once))
                continue

            # unpenalized single fit (with your fallback inside _fit_once)
            res = _fit_once(
                y=y, X=X, off=off, cov_type=cov_type,
                regularization='none', alpha=0.0, l1_wt=0.0,
                use_overdispersion_scale=use_overdispersion_scale,
                feature_names=feature_names
            )
            record_cluster_outcomes(
                cid=cid, y=y, n=n, res=res,
                alpha_val=0.0, l1_wt_val=0.0,
                feature_names=feature_names, X=X, off=off, condX=condX_once,
                results=results, coef_rows=coef_rows, metrics_rows=metrics_rows, used_refit=False
            )
            print('  done (MLE)', flush=True)

        return pd.Series(results).to_dict(), pd.DataFrame(coef_rows), pd.DataFrame(metrics_rows), pd.DataFrame()

    # ----------------------- regular path with tuning ------------------------
    folds = _build_folds(n, n_splits=n_splits,
                         groups=groups, cv_splitter=cv_splitter)
    results, coef_rows, metrics_rows, cv_tables = {}, [], [], []

    for i, cid in enumerate(cluster_ids, 1):
        print(f'Fitting cluster {i}/{len(cluster_ids)}: {cid} ...', flush=True)
        y = df_Y[cid].to_numpy()

        if np.all(y == 0):
            metrics_rows.append(_make_zero_unit_row(
                cid, n, alpha_val=np.nan, l1_wt_val=np.nan, condX=condX_once))
            continue

        # Hyperparam search → best full fit
        best, cv_table = _hyperparam_search(
            y=y, X=X, off=off, folds=folds, cov_type=cov_type,
            regularization=regularization,
            alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid,
            cv_metric=cv_metric, return_table=True,
            use_overdispersion_scale=use_overdispersion_scale,
            feature_names=feature_names
        )
        res = best['res']

        # Optional L1 refit on support to recover SE/p
        res, used_refit = _refit_l1_support_if_needed(
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

    for i, cid in enumerate(eff_ids, 1):
        if show_progress:
            print(f'Fitting cluster {i}/{len(eff_ids)}: {cid} ...', flush=True)

        y = np.asarray(df_Y[cid], dtype=float)
        n = y.shape[0]

        # All-zero unit → flagged metrics row, skip fit
        if np.all(y == 0):
            metrics_rows.append(_make_zero_unit_row(
                cid, n, alpha_val=0.0, l1_wt_val=0.0, condX=condX_once))
            if show_progress:
                print('  skipped (all-zero unit)', flush=True)
            continue

        # Unpenalized Newton/IRLS with robust fallback (tiny ridge), then (try) unpen refit
        model = sm.GLM(y, X, family=fam, offset=off)
        attach_feature_names(model, feature_names)
        res = fit_with_fallback(
            model, cov_type=cov_type, use_overdispersion_scale=False,
            maxiter=maxiter, try_unpenalized_refit=True
        )

        # Bookkeeping identical to CV path (adds coef/metrics + flags)
        record_cluster_outcomes(
            cid=cid, y=y, n=n, res=res,
            alpha_val=0.0, l1_wt_val=0.0,
            feature_names=feature_names, X=X, off=off, condX=condX_once,
            results=results, coef_rows=coef_rows, metrics_rows=metrics_rows, used_refit=False
        )

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
    cv_splitter, use_overdispersion_scale, return_cv_tables,
    show_progress
):
    if fast_mle:
        results, coefs_df, metrics_df = fit_poisson_glm_per_cluster_fast_mle(
            df_X=df_X, df_Y=df_Y, offset_log=offset_log,
            cluster_ids=eff_clusters, cov_type=cov_type,
            maxiter=100, show_progress=show_progress,
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
        refit_on_support=refit_on_support, cv_splitter=cv_splitter,
        use_overdispersion_scale=use_overdispersion_scale, return_cv_tables=rcv,
    )


def _run_inference(coefs_df, *, do_inference, make_plots, alpha, delta_for_rr):
    """
    Robust inference step.
    - Ensures a usable 'p' column (compute from z or coef/se if missing).
    - If 'term' is absent, falls back to by_term=False for FDR.
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
            from scipy import stats as _stats
            df['p'] = 2.0 * _stats.norm.sf(np.abs(z))
        else:
            # try coef/se → p
            if ('coef' in df.columns) and ('se' in df.columns):
                se = df['se'].to_numpy() if 'se' in df else np.full(
                    len(df), np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = df['coef'] / se
                df['z'] = z
                from scipy import stats as _stats
                df['p'] = 2.0 * _stats.norm.sf(np.abs(z))
            else:
                # last resort: create NaN p's so add_fdr can still run
                df['p'] = np.nan

    # ---- choose by_term only if 'term' exists ----
    has_term = ('term' in df.columns)

    if do_inference:
        try:
            df = add_fdr(df, alpha=alpha, by_term=has_term)
        except KeyError:
            # safety net if helper still enforces 'term'
            df = add_fdr(df, alpha=alpha, by_term=False)

        # add rate ratios if possible
        if 'coef' in df.columns:
            if 'se' not in df.columns:
                df['se'] = np.nan
            df = add_rate_ratios(df, delta=delta_for_rr)
        pop = term_population_tests(df) if has_term else pd.DataFrame()
        return df, pop

    # not doing inference, but ensure FDR exists for plotting legends
    if make_plots and ('sig_FDR' not in df.columns):
        try:
            df = add_fdr(df, alpha=alpha, by_term=has_term)
        except KeyError:
            df = add_fdr(df, alpha=alpha, by_term=False)

    return df, pd.DataFrame()


def _pick_forest_term(feature_names, forest_term):
    if not feature_names:
        return None
    if forest_term is None:
        return feature_names[0]
    return forest_term if forest_term in feature_names else feature_names[0]


def _build_figs(coefs_df, metrics_df, *, feature_names, forest_term, forest_top_n, delta_for_rr, make_plots):
    if not make_plots:
        return {}
    figs = {
        'coef_dists': plot_glm_fit.plot_coef_distributions(coefs_df),
        'model_quality': plot_glm_fit.plot_model_quality(metrics_df),
    }
    ft = _pick_forest_term(feature_names, forest_term)
    if ft is not None:
        figs['forest'] = plot_glm_fit.plot_forest_for_term(
            coefs_df, term=ft, top_n=forest_top_n)
        figs['rr_hist'] = plot_glm_fit.plot_rate_ratio_hist(
            coefs_df, term=ft, delta=delta_for_rr)
    else:
        figs['forest'] = None
        figs['rr_hist'] = None
    return figs


def _save_outputs(save_dir, coefs_df, metrics_df, pop_tests, figs, *, make_plots):
    if save_dir is None:
        return
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    coefs_df.to_csv(p / 'coefs.csv', index=False)
    metrics_df.to_csv(p / 'metrics.csv', index=False)
    pop_tests.to_csv(p / 'population_tests.csv', index=False)
    if make_plots:
        for name, fig in figs.items():
            if fig is not None:
                fig.savefig(p / f'{name}.png', dpi=150, bbox_inches='tight')


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
    alpha: float = 0.05,
    delta_for_rr: float = 1.0,
    forest_term=None,
    forest_top_n: int = 30,
    cov_type: str = 'HC1',
    show_plots: bool = True,
    save_dir=None,
    regularization: str = 'none',
    alpha_grid=(0.1, 0.3, 1.0),
    l1_wt_grid=(1.0, 0.5, 0.0),
    n_splits: int = 5,
    cv_metric: str = 'loglik',
    groups=None,
    refit_on_support: bool = True,
    cv_splitter=None,
    use_overdispersion_scale: bool = False,
    fast_mle: bool = False,
    make_plots: bool = True,
    do_inference: bool = True,
    return_cv_tables: bool = True,
    show_progress: bool = False,
):
    """Thin orchestration wrapper: fit → inference → figs → save/show → offenders."""
    feature_names, eff_clusters = _resolve_inputs(
        df_X, df_Y, feature_names, cluster_ids)

    results, coefs_df, metrics_df, cv_tables_df = _fit_path(
        df_X=df_X, df_Y=df_Y, offset_log=offset_log,
        eff_clusters=eff_clusters, cov_type=cov_type,
        fast_mle=fast_mle, regularization=regularization,
        alpha_grid=alpha_grid, l1_wt_grid=l1_wt_grid,
        n_splits=n_splits, cv_metric=cv_metric, groups=groups,
        refit_on_support=refit_on_support, cv_splitter=cv_splitter,
        use_overdispersion_scale=use_overdispersion_scale,
        return_cv_tables=return_cv_tables, show_progress=show_progress
    )

    coefs_df, pop_tests = _run_inference(
        coefs_df, do_inference=do_inference, make_plots=make_plots,
        alpha=alpha, delta_for_rr=delta_for_rr
    )
    figs = _build_figs(
        coefs_df, metrics_df, feature_names=feature_names,
        forest_term=forest_term, forest_top_n=forest_top_n,
        delta_for_rr=delta_for_rr, make_plots=make_plots
    )
    _save_outputs(save_dir, coefs_df, metrics_df,
                  pop_tests, figs, make_plots=make_plots)
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
