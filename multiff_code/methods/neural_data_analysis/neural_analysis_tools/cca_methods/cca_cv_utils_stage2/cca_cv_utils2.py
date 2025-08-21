import numpy as np
from sklearn.model_selection import GroupKFold, KFold
import rcca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def conduct_cca_cv(X1_unscaled, 
                   X2_unscaled,
                n_components=10,
                trial_ids=None,
                reg=1e-2,
                n_splits=5,
                shuffle=True,
                random_state=None):
    """
    Run CCA on full data (for weights/loadings) and compute cross-validated
    canonical correlations (held-out) to check generalization.

    - If `trial_ids` exists and has >= n_splits unique levels, uses GroupKFold by trial.
    - Otherwise uses KFold(shuffle=True).
    - Stores results under results['cv'].
    """


    # --- guard components ---
    n_components = min(n_components, len(X1_unscaled.columns), len(X2_unscaled.columns))

    scaler = StandardScaler()
    X1_sc, X2_sc = scaler.fit_transform(
        X1_unscaled), scaler.fit_transform(X2_unscaled)
        
    # --- cross-validation setup ---
    X1 = np.asarray(X1_sc, dtype=float)
    X2 = np.asarray(X2_sc, dtype=float)
    n = X1.shape[0]

    # Try GroupKFold by trial_ids if available and sensible
    use_groups = trial_ids is not None
    if use_groups:
        groups = np.asarray(trial_ids)
        n_groups = len(np.unique(groups))
        if n_groups >= max(2, n_splits):
            splitter = GroupKFold(n_splits=min(n_splits, n_groups))
            split_iter = splitter.split(np.arange(n), groups=groups)
        else:
            # fallback to KFold if not enough groups
            splitter = KFold(n_splits=min(n_splits, max(2, min(n_splits, n//5))),
                             shuffle=shuffle, random_state=random_state)
            split_iter = splitter.split(np.arange(n))
    else:
        splitter = KFold(n_splits=min(n_splits, max(2, min(n_splits, n//5))),
                         shuffle=shuffle, random_state=random_state)
        split_iter = splitter.split(np.arange(n))

    # --- helpers ---
    def _fit_cca(X1_tr, X2_tr):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_tr, X2_tr])
        return cca

    def _transform(cca, X1_te, X2_te):
        # rcca stores projection matrices in cca.ws
        Z1 = X1_te @ cca.ws[0]
        Z2 = X2_te @ cca.ws[1]
        return Z1, Z2

    def _corr_vec(A, B):
        # per-component Pearson corr
        num = np.sum((A - A.mean(0)) * (B - B.mean(0)), axis=0)
        den = np.sqrt(np.sum((A - A.mean(0))**2, axis=0) *
                      np.sum((B - B.mean(0))**2, axis=0))
        corr = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        # clip numerical wiggles
        return np.clip(corr, -1.0, 1.0)

    # --- CV loop ---
    test_corrs = []     # list of shape (n_components,) per fold
    train_corrs = []    # optional: to inspect overfitting
    for tr_idx, te_idx in split_iter:
        X1_tr, X2_tr = X1[tr_idx], X2[tr_idx]
        X1_te, X2_te = X1[te_idx], X2[te_idx]

        cca_cv = _fit_cca(X1_tr, X2_tr)
        Z1_tr, Z2_tr = _transform(cca_cv, X1_tr, X2_tr)
        Z1_te, Z2_te = _transform(cca_cv, X1_te, X2_te)

        train_corrs.append(_corr_vec(Z1_tr, Z2_tr))
        test_corrs.append(_corr_vec(Z1_te, Z2_te))

    train_corrs = np.vstack(train_corrs)  # (n_folds, n_components)
    test_corrs  = np.vstack(test_corrs)

    cv_summary = {
        "train_corrs_by_fold": train_corrs,
        "test_corrs_by_fold":  test_corrs,
        "test_corrs_mean":     test_corrs.mean(axis=0),
        "test_corrs_std":      test_corrs.std(axis=0),
        "avg_first_cc_test":   float(test_corrs[:, 0].mean()),
        "avg_first_cc_train":  float(train_corrs[:, 0].mean()),
        "n_splits":            getattr(splitter, "n_splits", None),
        "used_groups":         bool(use_groups),
    }

    # (Optional) quick printout
    m = cv_summary["test_corrs_mean"]
    s = cv_summary["test_corrs_std"]
    print(f"[CCA CV] Test canonical correlations (mean±sd) for first {len(m)} comps:")
    print("  " + "  ".join([f"{i+1}:{m[i]:.3f}±{s[i]:.3f}" for i in range(len(m))]))
    
    return cv_summary





def plot_cv_cca_overlay(cv_summary, title="CV canonical correlations",
                        show_train=True, jitter=0.06, alpha_points=0.35):
    test = np.asarray(cv_summary["test_corrs_by_fold"])   # (F, K)
    mean_ = np.asarray(cv_summary["test_corrs_mean"])
    std_  = np.asarray(cv_summary["test_corrs_std"])
    comps = np.arange(1, test.shape[1] + 1)

    plt.figure(figsize=(8,5))

    # --- overlay all folds as jittered points ---
    F = test.shape[0]
    for f in range(F):
        x = comps + np.random.uniform(-jitter, jitter, size=len(comps))
        plt.scatter(x, test[f], s=18, alpha=alpha_points, label=None)

    # --- mean ± SD band + mean line ---
    plt.fill_between(comps, mean_ - std_, mean_ + std_, alpha=0.2, label="Mean ± SD")
    plt.plot(comps, mean_, lw=2, marker="o", label="Mean (test)")

    # Optional: overlay training as hollow markers
    if show_train:
        train = np.asarray(cv_summary["train_corrs_by_fold"])
        for f in range(train.shape[0]):
            x = comps + np.random.uniform(-jitter, jitter, size=len(comps))
            plt.scatter(x, train[f], s=18, facecolors='none', edgecolors='black',
                        alpha=0.25, label=None)
        # also plot mean
        plt.plot(comps, train.mean(axis=0), lw=2, marker="o", label="Mean (train)", color='black')


    plt.xlabel("Canonical component")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.xticks(comps)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cv_cca_violin(cv_summary, title="CV canonical correlations (violin+points)",
                       jitter=0.05, alpha_points=0.35):
    test = np.asarray(cv_summary["test_corrs_by_fold"])   # (F, K)
    K = test.shape[1]
    comps = np.arange(1, K + 1)

    plt.figure(figsize=(8,5))
    parts = plt.violinplot([test[:,k] for k in range(K)], positions=comps,
                           showmeans=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_alpha(0.15)

    # scatter all folds
    for k in range(K):
        x = np.full(test.shape[0], comps[k]) + np.random.uniform(-jitter, jitter, size=test.shape[0])
        plt.scatter(x, test[:,k], s=18, alpha=alpha_points)

    plt.xlabel("Canonical component")
    plt.ylabel("Correlation (test)")
    plt.title(title)
    plt.xticks(comps)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_cv_cca_lines(cv_summary, title="CV canonical correlations (per-fold lines)"):
    test = np.asarray(cv_summary["test_corrs_by_fold"])
    comps = np.arange(1, test.shape[1] + 1)
    mean_ = np.asarray(cv_summary["test_corrs_mean"])

    plt.figure(figsize=(8,5))
    for f in range(test.shape[0]):
        plt.plot(comps, test[f], lw=1, alpha=0.4)
    plt.plot(comps, mean_, lw=2, marker="o", label="Mean (test)")
    plt.xlabel("Canonical component")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.xticks(comps)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()
