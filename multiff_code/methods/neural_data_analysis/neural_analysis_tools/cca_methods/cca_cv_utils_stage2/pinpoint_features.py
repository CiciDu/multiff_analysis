import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
import rcca


def cv_cca_perm_importance(
    X1, X2, feature_names, *,
    n_components=5, reg=1e-2, n_splits=5, random_state=0, groups=None,
    max_components_for_score=None, sanity_checks=True, tol=1e-6, verbose=True
):
    """
    Cross-validated permutation importance for CCA with sanity checks.
    Uses a rank-1 update when permuting each feature to avoid re-multiplying the full matrix.

    Returns:
      importance_df: per-feature drops (mean over folds) + summary columns, including:
          - drop_comp{k}: absolute drop in test corr for comp k
          - mean_drop_firstK: mean absolute drop over first K comps
          - pct_drop_comp{k}: percent drop (drop_comp{k} / baseline_mean{k})
          - pct_drop_firstK: mean percent drop over first K comps
          - frac_neg_drop_firstK: fraction of (feature, comp≤K) where permutation increased corr
          - frac_drop_gt_base_firstK: fraction where drop > baseline (perm corr < 0)
      diagnostics: dict with baseline means, stds, and sanity flags.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold
    import rcca

    rng = np.random.default_rng(random_state)
    n, p = X1.shape

    # splitter
    if groups is not None and len(np.unique(groups)) >= n_splits:
        split_iter = GroupKFold(n_splits=n_splits).split(
            np.arange(n), groups=groups)
    else:
        split_iter = KFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state).split(np.arange(n))

    # helpers
    def _fit(X1_tr, X2_tr):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_tr, X2_tr])
        return cca

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
        num = np.sum(A * B, axis=0)
        den = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(c, -1.0, 1.0)

    drops = np.zeros((p, n_components), dtype=float)
    counts = np.zeros((p,), dtype=int)

    # --- sanity trackers ---
    # baseline test corr per fold (K,)
    bases = []
    neg_drop_counts = np.zeros(
        (p, n_components), dtype=int)  # times cperm > base
    # times (base - cperm) > base
    drop_gt_base_counts = np.zeros((p, n_components), dtype=int)
    nonfinite_baseline_folds = []
    overlap_flagged_folds = []

    for fold_id, (tr_idx, te_idx) in enumerate(split_iter):
        # No overlap between train/test
        if sanity_checks and np.intersect1d(tr_idx, te_idx).size > 0:
            overlap_flagged_folds.append(fold_id)

        X1_tr, X2_tr = X1[tr_idx], X2[tr_idx]
        X1_te, X2_te = X1[te_idx], X2[te_idx]

        cca = _fit(X1_tr, X2_tr)
        ws0 = cca.ws[0]  # (p, K)
        ws1 = cca.ws[1]  # (q, K)

        # --- Precompute baseline Z1_te and Z2_te once per fold (rank-1 update uses these) ---
        Z1_base = X1_te @ ws0            # (n_test, K)
        Z2_te = X2_te @ ws1            # (n_test, K)

        base = _corr_cols(Z1_base, Z2_te)  # (K,)
        bases.append(base)

        if sanity_checks:
            if (not np.all(np.isfinite(base))) or np.any(np.abs(base) > 1 + 1e-8):
                nonfinite_baseline_folds.append(fold_id)
            if verbose and base[0] > 0.98:
                print(
                    f"[WARN] Fold {fold_id}: baseline first CC ~{base[0]:.3f} (check leakage/overfit)")

        # permute each feature on test, recompute corr using a rank-1 update
        for j in range(p):
            # For feature j, after shuffling X1_te[:, j] -> xj_perm:
            xj_perm = X1_te[:, j].copy()
            rng.shuffle(xj_perm)
            # (n_test, 1)
            delta = (xj_perm - X1_te[:, j])[:, None]

            # Rank-1 update: Z1_perm = Z1_base + delta * ws0[j, :]
            Z1_perm = Z1_base + delta * \
                ws0[j, :][None, :]          # (n_test, K)

            cperm = _corr_cols(Z1_perm, Z2_te)

            diff = base - cperm
            drops[j] += diff
            counts[j] += 1

            if sanity_checks:
                # permutation improved corr
                neg_drop_counts[j] += (cperm > base + tol)
                # drop larger than baseline
                drop_gt_base_counts[j] += (diff > base + tol)

    # average drop over folds
    drops = drops / np.maximum(1, counts)[:, None]

    # choose K for summaries
    K = n_components if max_components_for_score is None else min(
        max_components_for_score, n_components)
    mean_drop_firstK = drops[:, :K].mean(axis=1)

    # --- percent drops (needs baseline means) ---
    baseline_means = np.mean(
        np.vstack(bases), axis=0) if bases else np.zeros(n_components)
    eps = 1e-8
    # avoid divide-by-zero
    base_safe = np.maximum(eps, baseline_means)
    pct_drops = drops / base_safe[None, :]                # (p, K)
    pct_drop_firstK = pct_drops[:, :K].mean(axis=1)

    # package outputs
    drop_cols = {f"drop_comp{k+1}": drops[:, k] for k in range(n_components)}
    pct_cols = {
        f"pct_drop_comp{k+1}": pct_drops[:, k] for k in range(n_components)}

    importance_df = (pd.DataFrame({**drop_cols, **pct_cols})
                     .assign(feature=feature_names,
                             mean_drop_firstK=mean_drop_firstK,
                             pct_drop_firstK=pct_drop_firstK,
                             frac_neg_drop_firstK=(
                                 neg_drop_counts[:, :K].sum(1) / (counts * K + 1e-12)),
                             frac_drop_gt_base_firstK=(drop_gt_base_counts[:, :K].sum(1) / (counts * K + 1e-12)))
                     .sort_values("mean_drop_firstK", ascending=False)
                     .reset_index(drop=True))

    diagnostics = {
        "baseline_means": baseline_means,
        "baseline_stds": np.std(np.vstack(bases), axis=0) if bases else np.zeros(n_components),
        "overlap_flagged_folds": overlap_flagged_folds,
        "nonfinite_baseline_folds": nonfinite_baseline_folds,
        "notes": [
            "pct_drop_comp{k} = drop_comp{k} / baseline_mean{k}",
            "pct_drop_firstK = mean of pct_drop_comp1..K",
            "frac_neg_drop_firstK: fraction of (feature,component) in first K where permutation increased corr.",
            "frac_drop_gt_base_firstK: fraction where drop > baseline (implies perm corr < 0)."
        ],
    }

    if verbose:
        print("[SANITY] Mean baseline test correlations:",
              np.round(baseline_means, 3))
        if baseline_means[0] > 0.95:
            print(
                "[SANITY] First CC baseline unusually high on average; check grouping/leakage.")
        if overlap_flagged_folds:
            print(
                f"[SANITY] Overlapping train/test indices in folds: {overlap_flagged_folds}")
        if nonfinite_baseline_folds:
            print(
                f"[SANITY] Non-finite/|corr|>1 baseline in folds: {nonfinite_baseline_folds}")

    return importance_df, diagnostics


# One-shot runner for X1-only permutation importance (uses your cv_cca_perm_importance)


def cv_cca_leave1out_delta(
    X1_all, X2, feature_names, *,
    n_components=5, reg=1e-2, n_splits=5, random_state=0, groups=None,
    max_components_for_score=None,
    parallelize=True,            # NEW: on/off switch
    # number of workers if parallelize=True (-1 = all cores)
    n_jobs=-1,
    backend="loky",              # joblib backend
    blas_threads=1,              # cap BLAS threads inside workers to avoid oversubscription
):
    """
    Cross-validated leave-one-out importance for X1_all features.
    Parallelizes per-feature refits within each fold when `parallelize=True`.

    Returns a DataFrame with:
      - drop_comp{k}: absolute drop in test corr for component k
      - mean_drop_firstK: mean absolute drop over first K comps
      - pct_drop_comp{k}: percent drop (normalized by baseline mean corr for comp k)
      - pct_drop_firstK: mean of pct_drop_comp{1..K}
    """
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold
    import rcca

    # joblib (optional)
    try:
        from joblib import Parallel, delayed
        have_joblib = True
    except Exception:
        have_joblib = False

    # Decide whether to use parallel
    use_parallel = parallelize and have_joblib and (
        n_jobs is not None) and (n_jobs != 1)

    # Avoid BLAS oversubscription *only* if we parallelize
    if use_parallel and (blas_threads is not None):
        os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads))

    n, p = X1_all.shape
    if groups is not None and len(np.unique(groups)) >= n_splits:
        split = GroupKFold(n_splits=n_splits).split(
            np.arange(n), groups=groups)
    else:
        split = KFold(n_splits=n_splits, shuffle=True,
                      random_state=random_state).split(np.arange(n))

    def _fit(Xtr1, Xtr2):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([Xtr1, Xtr2])
        return cca

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
        num = np.sum(A * B, axis=0)
        den = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(c, -1.0, 1.0)

    # Per-feature drop storage
    drops = np.zeros((p, n_components), dtype=float)
    counts = np.zeros(p, dtype=int)
    bases = []  # baseline test correlations per fold

    # Worker: drop one feature (within a fold)
    def _drop_one_feature(j, X1_tr, X2_tr, X1_te, X2_te):
        keep = [k for k in range(X1_tr.shape[1]) if k != j]
        cca_lo = _fit(X1_tr[:, keep], X2_tr)
        Z1_te = X1_te[:, keep] @ cca_lo.ws[0]
        Z2_te = X2_te @ cca_lo.ws[1]
        return _corr_cols(Z1_te, Z2_te)

    for tr, te in split:
        X1_tr, X2_tr = X1_all[tr], X2[tr]
        X1_te, X2_te = X1_all[te], X2[te]

        # Baseline with all features
        cca_full = _fit(X1_tr, X2_tr)
        base = _corr_cols(
            X1_te @ cca_full.ws[0], X2_te @ cca_full.ws[1])  # (K,)
        bases.append(base)

        if use_parallel:
            # Parallel per-feature refits
            results = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
                delayed(_drop_one_feature)(j, X1_tr, X2_tr, X1_te, X2_te) for j in range(p)
            )
            results = np.asarray(results)  # (p, K)
            drops += (base[None, :] - results)
            counts += 1  # each feature measured once in this fold
        else:
            # Sequential loop
            for j in range(p):
                corr_lo = _drop_one_feature(j, X1_tr, X2_tr, X1_te, X2_te)
                drops[j] += (base - corr_lo)
                counts[j] += 1

    # Average over folds
    drops = drops / np.maximum(1, counts)[:, None]
    base_means = np.mean(
        np.vstack(bases), axis=0) if bases else np.zeros(n_components)

    # Summaries over first K components
    K = n_components if max_components_for_score is None else min(
        max_components_for_score, n_components)
    mean_drop_firstK = drops[:, :K].mean(axis=1)

    # Percent drops (safe divide by baseline means)
    eps = 1e-8
    pct_drops = drops / np.maximum(eps, base_means)[None, :]
    pct_drop_firstK = pct_drops[:, :K].mean(axis=1)

    # Build DataFrame
    drop_cols = {f"drop_comp{k+1}": drops[:, k] for k in range(n_components)}
    pct_cols = {
        f"pct_drop_comp{k+1}": pct_drops[:, k] for k in range(n_components)}
    out = (pd.DataFrame({**drop_cols, **pct_cols})
             .assign(feature=feature_names,
                     mean_drop_firstK=mean_drop_firstK,
                     pct_drop_firstK=pct_drop_firstK)
             .sort_values("mean_drop_firstK", ascending=False)
             .reset_index(drop=True))
    return out


# def cv_cca_leave1out_delta(
#     X1_all, X2, feature_names, *,
#     n_components=5, reg=1e-2, n_splits=5, random_state=0, groups=None,
#     max_components_for_score=None
# ):
#     import numpy as np, pandas as pd
#     from sklearn.model_selection import GroupKFold, KFold
#     import rcca

#     n, p = X1_all.shape
#     if groups is not None and len(np.unique(groups)) >= n_splits:
#         split = GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups)
#     else:
#         split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n))

#     def _fit(Xtr1, Xtr2):
#         cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
#         cca.train([Xtr1, Xtr2]); return cca

#     def _corr_cols(A, B):
#         A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
#         num = np.sum(A * B, axis=0)
#         den = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
#         c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
#         return np.clip(c, -1.0, 1.0)

#     drops = np.zeros((p, n_components)); counts = np.zeros(p, int)

#     for tr, te in split:
#         X1_tr, X2_tr = X1_all[tr], X2[tr]
#         X1_te, X2_te = X1_all[te], X2[te]

#         cca_full = _fit(X1_tr, X2_tr)
#         base = _corr_cols(X1_te @ cca_full.ws[0], X2_te @ cca_full.ws[1])

#         for j in range(p):
#             keep = [k for k in range(p) if k != j]
#             cca_lo = _fit(X1_tr[:, keep], X2_tr)
#             corr_lo = _corr_cols(X1_te[:, keep] @ cca_lo.ws[0], X2_te @ cca_lo.ws[1])
#             drops[j] += (base - corr_lo); counts[j] += 1

#     drops = drops / np.maximum(1, counts)[:, None]
#     K = n_components if max_components_for_score is None else min(max_components_for_score, n_components)
#     score = drops[:, :K].mean(axis=1)
#     import pandas as pd
#     return (pd.DataFrame(drops, columns=[f"drop_comp{k+1}" for k in range(n_components)])
#               .assign(feature=feature_names, mean_drop_firstK=score)
#               .sort_values("mean_drop_firstK", ascending=False)
#               .reset_index(drop=True))


def cv_structure_coefficients(X1_all, X2, feature_names, *,
                              n_components=5, reg=1e-2, n_splits=5,
                              random_state=0, groups=None):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold
    import rcca

    n, p = X1_all.shape
    if groups is not None and len(np.unique(groups)) >= n_splits:
        split = GroupKFold(n_splits=n_splits).split(
            np.arange(n), groups=groups)
    else:
        split = KFold(n_splits=n_splits, shuffle=True,
                      random_state=random_state).split(np.arange(n))

    sc_sums = np.zeros((p, n_components))
    counts = 0

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
        num = A.T @ B
        den = np.sqrt((A**2).sum(0))[:, None] * np.sqrt((B**2).sum(0))[None, :]
        C = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(C, -1.0, 1.0)

    for tr, te in split:
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_all[tr], X2[tr]])
        U_te = X1_all[te] @ cca.ws[0]  # X1 canonical variates on test
        # corr(feature j, canonical comp k) on test:
        C = _corr_cols(X1_all[te], U_te)  # shape (p, n_components)
        sc_sums += C
        counts += 1

    sc_mean = sc_sums / max(1, counts)
    cols = [f"SC_comp{k+1}" for k in range(n_components)]
    import pandas as pd
    return (pd.DataFrame(sc_mean, columns=cols)
              .assign(feature=feature_names)
              .reindex(columns=["feature"]+cols)
              .sort_values("SC_comp1", key=lambda s: np.abs(s), ascending=False)
              .reset_index(drop=True))


# =============================== To run everything & plot ===============================


def cv_partial_cca_all_given_selected(
    X1_sel, X1_all, X2, *, n_components=10, reg=1e-2,
    n_splits=5, random_state=0, groups=None, ridge_alpha=10.0
):
    """
    Returns a dict with CV curves for:
      - Selected-only vs X2
      - Partial: all | Selected  (both all and X2 residualized on Selected)
      - Combined: [Selected + all] vs X2
    """
    n = X1_sel.shape[0]
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits).split(
            np.arange(n), groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state).split(np.arange(n))

    def _fit(Xtr1, Xtr2):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([Xtr1, Xtr2])
        return cca

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
        num = np.sum(A * B, axis=0)
        den = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(c, -1.0, 1.0)

    from sklearn.linear_model import Ridge

    def _resid(X, Z, alpha=ridge_alpha):
        mdl = Ridge(alpha=alpha, fit_intercept=True)
        mdl.fit(Z, X)
        return X - mdl.predict(Z), mdl

    sel_test, part_test, comb_test = [], [], []

    for tr, te in splitter:
        Xsel_tr, Xsel_te = X1_sel[tr], X1_sel[te]
        Xall_tr, Xall_te = X1_all[tr], X1_all[te]
        X2_tr, X2_te = X2[tr], X2[te]

        # Selected-only
        cca_sel = _fit(Xsel_tr, X2_tr)
        cs = _corr_cols(Xsel_te @ cca_sel.ws[0], X2_te @ cca_sel.ws[1])
        sel_test.append(cs)

        # Partial (all | Selected): residualize both sides on Selected
        Xall_tr_res, mdl_r = _resid(Xall_tr, Xsel_tr)
        Xall_te_res = Xall_te - mdl_r.predict(Xsel_te)
        X2_tr_res, mdl_y = _resid(X2_tr, Xsel_tr)
        X2_te_res = X2_te - mdl_y.predict(Xsel_te)

        cca_part = _fit(Xall_tr_res, X2_tr_res)
        cp = _corr_cols(
            Xall_te_res @ cca_part.ws[0], X2_te_res @ cca_part.ws[1])
        part_test.append(cp)

        # Combined: [Selected + all] vs X2
        Xcomb_tr = np.hstack([Xsel_tr, Xall_tr])
        Xcomb_te = np.hstack([Xsel_te, Xall_te])
        cca_comb = _fit(Xcomb_tr, X2_tr)
        cc = _corr_cols(Xcomb_te @ cca_comb.ws[0], X2_te @ cca_comb.ws[1])
        comb_test.append(cc)

    sel_test = np.vstack(sel_test)
    part_test = np.vstack(part_test)
    comb_test = np.vstack(comb_test)

    out = {
        "selected_test_by_fold": sel_test,
        "partial_all_given_selected_by_fold": part_test,
        "combined_test_by_fold": comb_test,
        "selected_mean": sel_test.mean(0), "selected_sd": sel_test.std(0),
        "partial_mean": part_test.mean(0), "partial_sd": part_test.std(0),
        "combined_mean": comb_test.mean(0), "combined_sd": comb_test.std(0),
        "n_splits": sel_test.shape[0],
    }
    return out


def run_all_feature_diagnostics(
    X1_sel_sc, X1_all_sc, X2_sc, all_names,
    trial_ids=None, n_components=10, reg=1e-2, n_splits=5, ridge_alpha=10.0,
    top_k=20, avg_firstK=3
):
    # 1) Leave-one-out Δρ
    imp_loo = cv_cca_leave1out_delta(
        X1_all_sc, X2_sc, all_names,
        n_components=n_components, reg=reg,
        n_splits=n_splits, random_state=0,
        groups=trial_ids, max_components_for_score=avg_firstK
    )

    # 2) Structure coefficients
    sc_df = cv_structure_coefficients(
        X1_all_sc, X2_sc, all_names,
        n_components=n_components, reg=reg,
        n_splits=n_splits, random_state=0, groups=trial_ids
    )

    # 3) Partial CCA (incremental all | Selected)
    partial = cv_partial_cca_all_given_selected(
        X1_sel_sc, X1_all_sc, X2_sc,
        n_components=n_components, reg=reg,
        n_splits=n_splits, random_state=0,
        groups=trial_ids, ridge_alpha=ridge_alpha
    )

    # ---- Quick plots ----
    K = len(partial["selected_mean"])
    x = np.arange(1, K+1)

    # (a) CV curves: Selected-only vs Partial (all|Selected) vs Combined
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, partial["selected_mean"], yerr=partial["selected_sd"],
                 marker="o", capsize=3, label="Selected only")
    plt.errorbar(x, partial["partial_mean"],  yerr=partial["partial_sd"],
                 marker="o", capsize=3, label="all | Selected (partial)")
    plt.errorbar(x, partial["combined_mean"], yerr=partial["combined_sd"],
                 marker="o", capsize=3, label="Selected + all")
    plt.xlabel("Canonical component")
    plt.ylabel("Correlation (test)")
    plt.title("CV: Selected vs Partial(all|Selected) vs Combined")
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (b) Top-K features by Δρ (mean over first K comps)
    top = imp_loo.nlargest(top_k, "mean_drop_firstK")
    plt.figure(figsize=(8, max(4, 0.3*top_k)))
    plt.barh(top["feature"][::-1], top["mean_drop_firstK"][::-1])
    plt.xlabel(f"Δ correlation (avg of first {avg_firstK} comps)")
    plt.ylabel("Feature")
    plt.title("Leave-1-out importance (all set)")
    plt.tight_layout()
    plt.show()

    # (c) Structure coefficients heatmap (absolute, top features)
    # pick features that are top by |SC_comp1|
    sc_top = sc_df.assign(abs1=np.abs(
        sc_df["SC_comp1"])).nlargest(top_k, "abs1")
    H = sc_top[[c for c in sc_df.columns if c.startswith(
        "SC_comp")]].to_numpy()
    plt.figure(figsize=(8, max(4, 0.3*top_k)))
    plt.imshow(np.abs(H[:, :min(5, H.shape[1])]), aspect="auto")
    plt.yticks(np.arange(len(sc_top)), sc_top["feature"])
    plt.xticks(np.arange(min(5, H.shape[1])), [
               f"SC c{k+1}" for k in range(min(5, H.shape[1]))])
    plt.colorbar(label="|structure coefficient|")
    plt.title("Held-out structure coefficients (top by comp1)")
    plt.tight_layout()
    plt.show()

    return imp_loo, sc_df, partial
