import numpy as np
from sklearn.model_selection import GroupKFold, KFold
import rcca

def cv_cca_perm_importance(
    X1_all, X2, feature_names, *,
    n_components=5, reg=1e-2, n_splits=5, random_state=0, groups=None,
    max_components_for_score=None
):
    """
    Cross-validated permutation importance for CCA.
    Returns a DataFrame with per-feature mean drop in test canonical correlations.

    X1_all, X2: numpy arrays (n_samples x p), preprocessed the same way you used in your CCA.
    feature_names: list[str] length p (for X1_all).
    groups: optional array of trial ids for GroupKFold.
    """
    import pandas as pd

    n, p = X1_all.shape
    rng = np.random.default_rng(random_state)

    # splitter
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n))

    # helpers
    def _fit(Xtr1, Xtr2):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([Xtr1, Xtr2])
        return cca

    def _corr_cols(A, B):
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
        num = np.sum(A * B, axis=0)
        den = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(c, -1.0, 1.0)

    drops = np.zeros((p, n_components), dtype=float)
    counts = np.zeros((p,), dtype=int)

    for tr_idx, te_idx in splitter:
        X1_tr, X2_tr = X1_all[tr_idx], X2[tr_idx]
        X1_te, X2_te = X1_all[te_idx], X2[te_idx]

        cca = _fit(X1_tr, X2_tr)
        Z1_te = X1_te @ cca.ws[0]
        Z2_te = X2_te @ cca.ws[1]
        base = _corr_cols(Z1_te, Z2_te)  # baseline test corr per component

        # permute one X1 feature at a time on the test split
        for j in range(p):
            X1_perm = X1_te.copy()
            rng.shuffle(X1_perm[:, j])  # in-place shuffle of that column
            Z1_perm = X1_perm @ cca.ws[0]
            cperm = _corr_cols(Z1_perm, Z2_te)
            drops[j] += (base - cperm)
            counts[j] += 1

    drops = drops / np.maximum(1, counts)[:, None]
    K = n_components if max_components_for_score is None else min(max_components_for_score, n_components)
    score = drops[:, :K].mean(axis=1)  # average contribution over first K comps

    out = (pd.DataFrame(drops, columns=[f"drop_comp{k+1}" for k in range(n_components)])
             .assign(feature=feature_names, mean_drop_firstK=score)
             .sort_values("mean_drop_firstK", ascending=False)
             .reset_index(drop=True))
    return out

def summarize_percent_drop(drops_vec, base_vec, K=3):
    pct = (drops_vec[:K] / np.maximum(base_vec[:K], 1e-8)).mean()
    return float(pct)


def cv_cca_leave1out_delta(
    X1_all, X2, feature_names, *,
    n_components=5, reg=1e-2, n_splits=5, random_state=0, groups=None,
    max_components_for_score=None
):
    import numpy as np, pandas as pd
    from sklearn.model_selection import GroupKFold, KFold
    import rcca

    n, p = X1_all.shape
    if groups is not None and len(np.unique(groups)) >= n_splits:
        split = GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups)
    else:
        split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n))

    def _fit(Xtr1, Xtr2):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([Xtr1, Xtr2]); return cca

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
        num = np.sum(A * B, axis=0)
        den = np.sqrt(np.sum(A**2, axis=0) * np.sum(B**2, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(c, -1.0, 1.0)

    drops = np.zeros((p, n_components)); counts = np.zeros(p, int)

    for tr, te in split:
        X1_tr, X2_tr = X1_all[tr], X2[tr]
        X1_te, X2_te = X1_all[te], X2[te]

        cca_full = _fit(X1_tr, X2_tr)
        base = _corr_cols(X1_te @ cca_full.ws[0], X2_te @ cca_full.ws[1])

        for j in range(p):
            keep = [k for k in range(p) if k != j]
            cca_lo = _fit(X1_tr[:, keep], X2_tr)
            corr_lo = _corr_cols(X1_te[:, keep] @ cca_lo.ws[0], X2_te @ cca_lo.ws[1])
            drops[j] += (base - corr_lo); counts[j] += 1

    drops = drops / np.maximum(1, counts)[:, None]
    K = n_components if max_components_for_score is None else min(max_components_for_score, n_components)
    score = drops[:, :K].mean(axis=1)
    import pandas as pd
    return (pd.DataFrame(drops, columns=[f"drop_comp{k+1}" for k in range(n_components)])
              .assign(feature=feature_names, mean_drop_firstK=score)
              .sort_values("mean_drop_firstK", ascending=False)
              .reset_index(drop=True))


def cv_structure_coefficients(X1_all, X2, feature_names, *,
                              n_components=5, reg=1e-2, n_splits=5,
                              random_state=0, groups=None):
    import numpy as np, pandas as pd
    from sklearn.model_selection import GroupKFold, KFold
    import rcca

    n, p = X1_all.shape
    if groups is not None and len(np.unique(groups)) >= n_splits:
        split = GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups)
    else:
        split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n))

    sc_sums = np.zeros((p, n_components)); counts = 0

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
        num = A.T @ B
        den = np.sqrt((A**2).sum(0))[:,None] * np.sqrt((B**2).sum(0))[None,:]
        C = np.divide(num, den, out=np.zeros_like(num), where=den>0)
        return np.clip(C, -1.0, 1.0)

    for tr, te in split:
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([X1_all[tr], X2[tr]])
        U_te = X1_all[te] @ cca.ws[0]  # X1 canonical variates on test
        # corr(feature j, canonical comp k) on test:
        C = _corr_cols(X1_all[te], U_te)  # shape (p, n_components)
        sc_sums += C; counts += 1

    sc_mean = sc_sums / max(1, counts)
    cols = [f"SC_comp{k+1}" for k in range(n_components)]
    import pandas as pd
    return (pd.DataFrame(sc_mean, columns=cols)
              .assign(feature=feature_names)
              .reindex(columns=["feature"]+cols)
              .sort_values("SC_comp1", key=lambda s: np.abs(s), ascending=False)
              .reset_index(drop=True))






## =============================== To run everything & plot ===============================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import rcca

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
        splitter = GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(n))

    def _fit(Xtr1, Xtr2):
        cca = rcca.CCA(kernelcca=False, reg=reg, numCC=n_components)
        cca.train([Xtr1, Xtr2]); return cca

    def _corr_cols(A, B):
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
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
        cp = _corr_cols(Xall_te_res @ cca_part.ws[0], X2_te_res @ cca_part.ws[1])
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
    plt.figure(figsize=(8,5))
    plt.errorbar(x, partial["selected_mean"], yerr=partial["selected_sd"], marker="o", capsize=3, label="Selected only")
    plt.errorbar(x, partial["partial_mean"],  yerr=partial["partial_sd"],  marker="o", capsize=3, label="all | Selected (partial)")
    plt.errorbar(x, partial["combined_mean"], yerr=partial["combined_sd"], marker="o", capsize=3, label="Selected + all")
    plt.xlabel("Canonical component"); plt.ylabel("Correlation (test)")
    plt.title("CV: Selected vs Partial(all|Selected) vs Combined")
    plt.xticks(x); plt.ylim(0, 1.05); plt.legend(); plt.tight_layout(); plt.show()

    # (b) Top-K features by Δρ (mean over first K comps)
    top = imp_loo.nlargest(top_k, "mean_drop_firstK")
    plt.figure(figsize=(8, max(4, 0.3*top_k)))
    plt.barh(top["feature"][::-1], top["mean_drop_firstK"][::-1])
    plt.xlabel(f"Δ correlation (avg of first {avg_firstK} comps)"); plt.ylabel("Feature")
    plt.title("Leave-1-out importance (all set)"); plt.tight_layout(); plt.show()

    # (c) Structure coefficients heatmap (absolute, top features)
    # pick features that are top by |SC_comp1|
    sc_top = sc_df.assign(abs1=np.abs(sc_df["SC_comp1"])).nlargest(top_k, "abs1")
    H = sc_top[[c for c in sc_df.columns if c.startswith("SC_comp")]].to_numpy()
    plt.figure(figsize=(8, max(4, 0.3*top_k)))
    plt.imshow(np.abs(H[:, :min(5, H.shape[1])]), aspect="auto")
    plt.yticks(np.arange(len(sc_top)), sc_top["feature"])
    plt.xticks(np.arange(min(5, H.shape[1])), [f"SC c{k+1}" for k in range(min(5, H.shape[1]))])
    plt.colorbar(label="|structure coefficient|")
    plt.title("Held-out structure coefficients (top by comp1)"); plt.tight_layout(); plt.show()

    return imp_loo, sc_df, partial
