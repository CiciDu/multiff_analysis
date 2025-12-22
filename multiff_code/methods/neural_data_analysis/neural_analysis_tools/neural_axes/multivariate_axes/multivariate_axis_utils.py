# multivariate_axis_utils.py
import numpy as np
from sklearn.model_selection import KFold
from numpy.linalg import svd
from sklearn.linear_model import Ridge, MultiTaskElasticNet


def fit_multitask_linear(X, Y, method='ridge', alpha=1.0, l1_ratio=0.5):
    """
    Fit multi-output linear model.

    Returns
    -------
    W : (n_neurons, n_targets)
    model : fitted sklearn estimator (or None for lstsq)
    """
    if method == 'lstsq':
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        return W, None

    if method == 'ridge':
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, Y)
        W = model.coef_.T
        return W, model

    if method == 'mt_elasticnet':
        model = MultiTaskElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=5000
        )
        model.fit(X, Y)
        W = model.coef_.T
        return W, model

    raise ValueError(f'Unknown method: {method}')


def reduced_rank_regression(X, Y, rank, alpha=1.0):
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, Y)
    W_full = ridge.coef_.T  # (neurons, targets)

    U, S, Vt = svd(W_full, full_matrices=False)
    W_rank = U[:, :rank]  # (neurons, rank)
    return W_rank, ridge



def build_interaction_terms(Y, interaction_pairs):
    """
    Append interaction columns to Y.

    Y : (n_samples, n_vars)
    interaction_pairs : list of (i, j)

    Returns
    -------
    Y_aug : (n_samples, n_vars + n_interactions)
    names  : list of column names (indices or strings)
    """
    interactions = []
    for i, j in interaction_pairs:
        interactions.append((Y[:, i] * Y[:, j])[:, None])

    if len(interactions) == 0:
        return Y

    return np.hstack([Y] + interactions)


def cross_validated_r2(X, Y, W, n_splits=5):
    """
    Cross-validated R^2 for multivariate projection.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    r2s = []

    for train, test in kf.split(X):
        W_hat = np.linalg.lstsq(X[train], Y[train], rcond=None)[0]
        Y_hat = X[test] @ W_hat
        ss_res = np.sum((Y[test] - Y_hat) ** 2)
        ss_tot = np.sum((Y[test] - Y[test].mean(axis=0)) ** 2)
        r2s.append(1 - ss_res / ss_tot)

    return np.mean(r2s)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def plot_peri_event_trajectories_3d(
    aligned_proj,
    labels,
    dims=(0, 1, 2),
    n_show=30,
    colors=('tab:blue', 'tab:orange'),
    labels_text=('Condition 0', 'Condition 1'),
    time=None,
    mark_event=True,
    alpha_single=0.15,
    lw_single=0.8,
    lw_mean=3.0,
    seed=0,
    title='Peri-event population trajectories',
):
    """
    Plot individual and mean peri-event population trajectories in a learned subspace.

    Parameters
    ----------
    aligned_proj : array, shape (n_events, n_timepoints, n_dims_total)
        Peri-event projections.
    labels : array-like, shape (n_events,)
        Binary or categorical labels for each event (must be two unique values).
    dims : tuple of int
        Subspace dimensions to plot (default: first 3).
    n_show : int
        Number of individual trajectories to show per condition.
    colors : tuple of str
        Colors for the two conditions.
    labels_text : tuple of str
        Legend labels for the two conditions.
    time : array-like or None
        Time vector for the peri-event window (optional; only used for start marker).
    mark_event : bool
        Whether to mark event onset (t=0) with a point on mean trajectory.
    alpha_single : float
        Alpha for individual trajectories.
    lw_single : float
        Line width for individual trajectories.
    lw_mean : float
        Line width for mean trajectories.
    seed : int
        Random seed for reproducible subsampling.
    title : str
        Plot title.
    """

    rng = np.random.default_rng(seed)

    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) != 2:
        raise ValueError('labels must have exactly two unique values')

    Z = aligned_proj[:, :, dims]  # (events, time, 3)

    Z0 = Z[labels == uniq[0]]
    Z1 = Z[labels == uniq[1]]

    n0 = min(n_show, Z0.shape[0])
    n1 = min(n_show, Z1.shape[0])

    idx0 = rng.choice(Z0.shape[0], n0, replace=False)
    idx1 = rng.choice(Z1.shape[0], n1, replace=False)

    mean0 = Z0.mean(axis=0)
    mean1 = Z1.mean(axis=0)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Individual trajectories
    for i in idx0:
        ax.plot(
            Z0[i, :, 0], Z0[i, :, 1], Z0[i, :, 2],
            color=colors[0], alpha=alpha_single, linewidth=lw_single
        )

    for i in idx1:
        ax.plot(
            Z1[i, :, 0], Z1[i, :, 1], Z1[i, :, 2],
            color=colors[1], alpha=alpha_single, linewidth=lw_single
        )

    # Mean trajectories
    ax.plot(
        mean0[:, 0], mean0[:, 1], mean0[:, 2],
        color=colors[0], linewidth=lw_mean, label=labels_text[0]
    )
    ax.plot(
        mean1[:, 0], mean1[:, 1], mean1[:, 2],
        color=colors[1], linewidth=lw_mean, label=labels_text[1]
    )

    # Mark event onset
    if mark_event:
        ax.scatter(
            mean0[0, 0], mean0[0, 1], mean0[0, 2],
            color=colors[0], s=60
        )
        ax.scatter(
            mean1[0, 0], mean1[0, 1], mean1[0, 2],
            color=colors[1], s=60
        )

    ax.set_xlabel('Subspace dim 1')
    ax.set_ylabel('Subspace dim 2')
    ax.set_zlabel('Subspace dim 3')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig, ax
