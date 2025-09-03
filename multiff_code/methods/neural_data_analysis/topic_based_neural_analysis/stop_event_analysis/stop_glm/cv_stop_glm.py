import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from scipy.stats import poisson

# ---------- helpers ----------
def _resolve_X_df(X, feature_names, df_X):
    if df_X is not None:
        X_df = df_X.copy()
    else:
        X = np.asarray(X)
        if feature_names is None:
            raise ValueError('feature_names must be provided when X is not a DataFrame.')
        X_df = pd.DataFrame(X, columns=list(feature_names))
    # add intercept later in fit/predict
    return X_df

def _resolve_y_df(y, df_Y):
    if df_Y is not None:
        # Expect shape (n_samples, n_clusters) or (n_samples,)
        if isinstance(df_Y, pd.Series):
            Y_df = df_Y.to_frame()
        else:
            Y_df = df_Y.copy()
        cluster_labels = list(Y_df.columns)
    else:
        y = np.asarray(y)
        if y.ndim == 1:
            Y_df = pd.DataFrame({'y': y})
            cluster_labels = ['y']
        elif y.ndim == 2:
            # name clusters 0..C-1
            cluster_labels = [str(j) for j in range(y.shape[1])]
            Y_df = pd.DataFrame(y, columns=cluster_labels)
        else:
            raise ValueError('y must be 1D or 2D.')
    return Y_df, cluster_labels

def _resolve_offset(offset_log, n_rows):
    if isinstance(offset_log, (pd.Series, pd.DataFrame)):
        off = pd.Series(np.asarray(offset_log).reshape(-1), index=range(n_rows))
    else:
        off = pd.Series(np.asarray(offset_log).reshape(-1), index=range(n_rows))
    if off.shape[0] != n_rows:
        raise ValueError('offset_log length does not match X rows.')
    if not np.all(np.isfinite(off)):
        raise ValueError('offset_log contains non-finite values.')
    return off

# ---------- core ----------
def fit_poisson_glm(X=None, y=None, offset_log=None, feature_names=None, df_X=None):
    """
    Fit a Poisson GLM with log link and an offset. Accepts either:
      - arrays X (n x p), y (n,), feature_names; or
      - df_X (DataFrame, n x p) and y (array-like or Series) of length n.
    Returns a statsmodels results object.
    """
    X_df = _resolve_X_df(X, feature_names, df_X)
    if isinstance(y, pd.Series):
        y_vec = y.to_numpy()
    else:
        y_vec = np.asarray(y).reshape(-1)
    if X_df.shape[0] != y_vec.shape[0]:
        raise ValueError('X and y have different numbers of rows.')

    offset = _resolve_offset(offset_log, X_df.shape[0])

    # add intercept
    X_df_const = sm.add_constant(X_df, has_constant='add')

    model = sm.GLM(y_vec, X_df_const, family=sm.families.Poisson(), offset=offset)
    return model.fit(method='newton', maxiter=100, disp=False)

def ll_poisson(y, mu):
    # exact log-likelihood (includes log(y!))
    return poisson(mu).logpmf(y).sum()

def cv_score_per_cluster(
    X=None, y=None, offset_log=None, groups=None, feature_names=None, n_splits=5,
    df_X=None, df_Y=None
):
    """
    Cross-validated metrics per cluster. You can pass:
      - arrays: X (n x p), y (n x C), feature_names
      - dataframes: df_X (n x p), df_Y (n x C)
    'groups' should be a 1D array-like of length n for GroupKFold.
    Returns a DataFrame with columns: cluster, ll_test, ll_test_null, mcfadden_R2_cv
    """
    # resolve X and Y as DataFrames
    X_df = _resolve_X_df(X, feature_names, df_X)
    Y_df, cluster_labels = _resolve_y_df(y, df_Y)

    if X_df.shape[0] != Y_df.shape[0]:
        raise ValueError('X and Y have different numbers of rows.')

    n = X_df.shape[0]
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

        for train_idx, test_idx in gkf.split(X_df, groups=groups):
            Xt, Xv = X_df.iloc[train_idx], X_df.iloc[test_idx]
            yt, yv = Y_df.iloc[train_idx, j].to_numpy(), Y_df.iloc[test_idx, j].to_numpy()
            ot, ov = offset.iloc[train_idx], offset.iloc[test_idx]

            # Full model
            Xt_c = sm.add_constant(Xt, has_constant='add')
            model = sm.GLM(yt, Xt_c, family=sm.families.Poisson(), offset=ot)
            res = model.fit(method='newton', maxiter=100, disp=False)

            Xv_c = sm.add_constant(Xv, has_constant='add')
            mu_v = res.predict(Xv_c, offset=ov)

            # Null model: intercept + offset only
            ones_train = pd.DataFrame(index=Xt.index)  # empty -> const only after add_constant
            ones_test  = pd.DataFrame(index=Xv.index)
            ones_train_c = sm.add_constant(ones_train, has_constant='add')
            ones_test_c  = sm.add_constant(ones_test,  has_constant='add')

            model0 = sm.GLM(yt, ones_train_c, family=sm.families.Poisson(), offset=ot)
            res0 = model0.fit(method='newton', maxiter=100, disp=False)
            mu0_v = res0.predict(ones_test_c, offset=ov)

            # Likelihoods
            ll_test.append(ll_poisson(yv, np.clip(mu_v, 1e-12, None)))
            ll_test_null.append(ll_poisson(yv, np.clip(mu0_v, 1e-12, None)))

        ll_mean  = float(np.mean(ll_test))
        ll0_mean = float(np.mean(ll_test_null))
        mcfadden_R2 = 1 - (ll_mean / ll0_mean) if ll0_mean != 0 else np.nan

        scores.append({
            'cluster': clabel,
            'll_test': ll_mean,
            'll_test_null': ll0_mean,
            'mcfadden_R2_cv': mcfadden_R2,
        })

    return pd.DataFrame(scores)


def glm_mini_report_batch(
    binned_feats: pd.DataFrame,
    binned_spikes: pd.DataFrame,
    groups,
    n_splits: int = 5,
    offset_col: str = 'offset_log',
):
    """
    Batch mini report over MANY units with CV — wrapper to match the old API feel.
    - binned_feats: DataFrame with predictors + 'offset_log'
    - binned_spikes: DataFrame of counts (n_bins x n_units)
    - groups: array-like group labels (e.g., window/session IDs) for GroupKFold
    Returns: DataFrame with per-unit CV log-likelihoods and McFadden's R^2.
    """
    if offset_col not in binned_feats.columns:
        raise ValueError(f'Offset column {offset_col!r} not found in binned_feats.')
    df_X = binned_feats.drop(columns=[offset_col])
    offset_log = binned_feats[offset_col]

    scores = cv_score_per_cluster(
        df_X=df_X,
        df_Y=binned_spikes,
        offset_log=offset_log,
        groups=groups,
        n_splits=n_splits
    )
    return scores


# ---------- optional plotting helpers (headless-safe) ----------
import matplotlib.pyplot as plt

def plot_cv_scores(scores: pd.DataFrame, outpath: str = 'glm_cv_scores.png', title: str | None = None):
    """
    Bar plot of McFadden's R^2 (CV) per unit. Saves to disk (no GUI).
    """
    title = title or 'Cross-validated Poisson GLM fits per neuron'
    scores_sorted = scores.sort_values('mcfadden_R2_cv', ascending=False)

    plt.figure(figsize=(9, 4))
    plt.bar(range(len(scores_sorted)), scores_sorted['mcfadden_R2_cv'])
    plt.xticks(range(len(scores_sorted)), scores_sorted['cluster'], rotation=90)
    plt.ylabel("McFadden's R² (CV)")
    plt.xlabel('Cluster / neuron')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig(outpath, dpi=200)
    # plt.close()
    return outpath

def plot_pred_vs_obs(res, df_X: pd.DataFrame, y: pd.Series, offset_log: pd.Series,
                     outpath: str = 'glm_pred_vs_obs.png', title: str | None = None):
    """
    Diagnostic: predicted vs observed counts for the provided (X, y, offset).
    """
    Xc = sm.add_constant(df_X, has_constant='add')
    mu = res.predict(Xc, offset=offset_log)
    title = title or 'Predicted vs observed (counts)'

    plt.figure(figsize=(5, 4))
    plt.scatter(mu, y, s=10, alpha=0.6)
    lim = [0, max(1.0, np.max([mu.max(), y.max()]))]
    plt.plot(lim, lim)  # y=x
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel('Predicted mean (μ)')
    plt.ylabel('Observed count')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig(outpath, dpi=200)
    # plt.close()
    return outpath


