import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.model_selection import GroupKFold
from typing import Optional

# ---------- helpers ----------


def _resolve_offset(offset_log, n_rows):
    if isinstance(offset_log, (pd.Series, pd.DataFrame)):
        off = pd.Series(np.asarray(offset_log).reshape(-1),
                        index=range(n_rows))
    else:
        off = pd.Series(np.asarray(offset_log).reshape(-1),
                        index=range(n_rows))
    if off.shape[0] != n_rows:
        raise ValueError('offset_log length does not match X rows.')
    if not np.all(np.isfinite(off)):
        raise ValueError('offset_log contains non-finite values.')
    return off


def ll_poisson(y, mu):
    # exact log-likelihood (includes log(y!))
    return poisson(mu).logpmf(y).sum()


# ---------- optional plotting helpers (headless-safe) ----------


def plot_cv_scores(scores: pd.DataFrame, title: Optional[str] = None):
    """
    Bar plot of McFadden's R^2 (CV) per unit. Saves to disk (no GUI).
    """
    title = title or 'Cross-validated Poisson GLM fits per neuron'
    scores_sorted = scores.sort_values('mcfadden_R2_cv', ascending=False)

    plt.figure(figsize=(9, 4))
    plt.bar(range(len(scores_sorted)), scores_sorted['mcfadden_R2_cv'])
    plt.xticks(range(len(scores_sorted)),
               scores_sorted['cluster'], rotation=90)
    plt.ylabel("McFadden's R² (CV)")
    plt.xlabel('Cluster / neuron')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return


def plot_pred_vs_obs(res, df_X, y, offset_log, outpath=None, title=None):
    # predict with guards
    mu = safe_predict_mu(res, df_X, offset_log, clip_eta=30.0)

    # align y to the same index if it’s a Series/DataFrame
    if hasattr(y, 'reindex'):
        # align to the exog alignment used inside safe_predict_mu
        exog = df_X.reindex(columns=list(res.model.exog_names), fill_value=0.0)
        y_use = y.reindex(exog.index).to_numpy(dtype=float)
    else:
        y_use = np.asarray(y, dtype=float)

    m = np.isfinite(mu) & np.isfinite(y_use)
    if not m.any():
        raise ValueError('No finite points to plot after prediction.')

    mu_f, y_f = mu[m], y_use[m]
    lim_max = max(1.0, float(np.nanmax([mu_f.max(), y_f.max()])))
    lim = (0.0, lim_max)

    plt.figure(figsize=(5, 5))
    plt.plot(mu_f, y_f, '.', alpha=0.5, markersize=4)
    plt.plot(lim, lim, lw=1)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel('Predicted mean (μ)')
    plt.ylabel('Observed count')
    if title:
        plt.title(title)
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.show()


def safe_predict_mu(res, df_X, offset_log, clip_eta=30.0):
    """
    Robust Poisson mean prediction μ = exp(η):
      - Align exog columns to what the model saw
      - Replace NaN/inf params with 0.0 (common after L1 refit-on-support)
      - Clip η to avoid exp overflow
    Returns a 1D float array (same row order as the aligned exog).
    """
    # 1) align design matrix to what the model used
    exog = df_X.copy()
    train_cols = list(res.model.exog_names)  # includes 'const' if used
    exog = exog.reindex(columns=train_cols, fill_value=0.0)

    # 2) align offset by index if possible
    if hasattr(offset_log, 'reindex'):
        off = offset_log.reindex(exog.index).to_numpy(dtype=float)
    else:
        off = np.asarray(offset_log, dtype=float).reshape(-1)
        if off.shape[0] != exog.shape[0]:
            raise ValueError("offset_log length doesn't match X rows.")

    # 3) robust linear predictor
    beta = np.asarray(res.params, dtype=float)
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

    eta = exog.to_numpy(dtype=float) @ beta + off
    eta = np.clip(eta, -clip_eta, clip_eta)

    # 4) μ
    return np.exp(eta)


def cv_score_per_cluster(
    df_X, df_Y, offset_log, groups=None, n_splits=5,
):
    """
    Cross-validated metrics per cluster. You can pass:
      - arrays: X (n x p), y (n x C), feature_names
      - dataframes: df_X (n x p), df_Y (n x C)
    'groups' should be a 1D array-like of length n for GroupKFold.
    Returns a DataFrame with columns: cluster, ll_test, ll_test_null, mcfadden_R2_cv
    """
    # resolve X and Y as DataFrames
    cluster_labels = df_Y.columns

    if df_X.shape[0] != df_Y.shape[0]:
        raise ValueError('X and Y have different numbers of rows.')

    n = df_X.shape[0]
    if groups is None:
        raise ValueError('groups (length n) is required for GroupKFold.')
    groups = np.asarray(groups).reshape(-1)
    if groups.shape[0] != n:
        raise ValueError('groups length does not match X/Y rows.')

    offset = _resolve_offset(offset_log, n)

    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    # Pre-build constant-added design for each split to keep column order consistent
    for j, clabel in enumerate(cluster_labels):
        ll_test, ll_test_null = [], []

        for train_idx, test_idx in gkf.split(df_X, groups=groups):
            Xt, Xv = df_X.iloc[train_idx], df_X.iloc[test_idx]
            yt, yv = df_Y.iloc[train_idx, j].to_numpy(
            ), df_Y.iloc[test_idx, j].to_numpy()
            ot, ov = offset.iloc[train_idx], offset.iloc[test_idx]

            # Full model
            model = sm.GLM(yt, Xt, family=sm.families.Poisson(), offset=ot)
            res = model.fit(method='newton', maxiter=100, disp=False)

            mu_v = res.predict(Xv, offset=ov)

            # Null model: intercept + offset only
            # empty -> const only after add_constant
            ones_train = pd.DataFrame(index=Xt.index)
            ones_test = pd.DataFrame(index=Xv.index)
            ones_train_c = sm.add_constant(ones_train, has_constant='add')
            ones_test_c = sm.add_constant(ones_test,  has_constant='add')

            model0 = sm.GLM(yt, ones_train_c,
                            family=sm.families.Poisson(), offset=ot)
            res0 = model0.fit(method='newton', maxiter=100, disp=False)
            mu0_v = res0.predict(ones_test_c, offset=ov)

            # Likelihoods
            ll_test.append(ll_poisson(yv, np.clip(mu_v, 1e-12, None)))
            ll_test_null.append(ll_poisson(yv, np.clip(mu0_v, 1e-12, None)))

        ll_mean = float(np.mean(ll_test))
        ll0_mean = float(np.mean(ll_test_null))
        mcfadden_R2 = 1 - (ll_mean / ll0_mean) if ll0_mean != 0 else np.nan

        scores.append({
            'cluster': clabel,
            'll_test': ll_mean,
            'll_test_null': ll0_mean,
            'mcfadden_R2_cv': mcfadden_R2,
        })

    return pd.DataFrame(scores)
