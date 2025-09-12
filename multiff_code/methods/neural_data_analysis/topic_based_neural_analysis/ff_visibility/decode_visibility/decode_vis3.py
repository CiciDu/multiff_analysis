import numpy as np
from sklearn.metrics import roc_auc_score

def auc_permutation_test(y, scores, groups=None, n_perm=1000, rng=0, mask=None,
                         progress: bool = False, progress_every: int = 200, desc: str = "Permutations"):
    """
    Time-aware permutation test:
      - Circularly shifts scores within each group.
    New:
      - progress: show tqdm progress bar if available; otherwise print every progress_every iters.
    """
    y = np.asarray(y, int).reshape(-1)
    s = np.asarray(scores, float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y and scores must have same length")
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
    if m.shape[0] != y.shape[0]:
        raise ValueError("mask length != data length")
    y, s = y[m], s[m]

    if groups is None:
        g = np.zeros_like(y)
    else:
        g_full = np.asarray(groups).reshape(-1)
        if g_full.shape[0] != m.shape[0]:
            raise ValueError("groups length != data length")
        g = g_full[m]

    if y.min() == y.max():
        raise ValueError("Only one class after masking; AUC undefined.")

    auc_obs = roc_auc_score(y, s)
    rng = np.random.default_rng(rng)
    uniq = np.unique(g)
    # precompute indices per group for speed
    group_idx = [np.where(g == gg)[0] for gg in uniq]
    null = np.empty(int(n_perm), float)

    # progress iterator
    it = range(int(n_perm))
    use_tqdm = False
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=desc, leave=False)
            use_tqdm = True
        except Exception:
            use_tqdm = False

    for k in it:
        s_perm = np.empty_like(s)
        for idx in group_idx:
            if idx.size <= 1:
                s_perm[idx] = s[idx]
            else:
                shift = rng.integers(0, idx.size)
                s_perm[idx] = np.roll(s[idx], shift)
        null[k] = roc_auc_score(y, s_perm)

        if progress and not use_tqdm and ((k + 1) % max(1, progress_every) == 0 or (k + 1) == n_perm):
            print(f"{desc}: {k+1}/{n_perm}", end="\r", flush=True)

    if progress and not use_tqdm:
        print()  # newline after last \r

    # one-sided p-value
    p = (1 + np.sum(null >= auc_obs)) / (1 + n_perm)
    return float(auc_obs), float(p), null


def auc_block_bootstrap_ci(y, scores, groups=None, n_boot=2000, conf=0.95, rng=0, mask=None,
                           progress: bool = False, progress_every: int = 200, desc: str = "Bootstraps"):
    """
    Block/bootstrap AUC by resampling groups with replacement.
    New:
      - progress: show tqdm progress bar if available; otherwise print every progress_every iters.
    """
    y = np.asarray(y, int).reshape(-1)
    s = np.asarray(scores, float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y and scores must have same length")
    m = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
    if m.shape[0] != y.shape[0]:
        raise ValueError("mask length != data length")
    y, s = y[m], s[m]

    if groups is None:
        g = np.arange(y.shape[0])
    else:
        g_full = np.asarray(groups).reshape(-1)
        if g_full.shape[0] != m.shape[0]:
            raise ValueError("groups length != data length")
        g = g_full[m]

    uniq = np.unique(g)
    # precompute indices per group for speed
    group_idx = [np.where(g == gg)[0] for gg in uniq]

    rng = np.random.default_rng(rng)
    aucs = np.empty(int(n_boot), float)

    # progress iterator
    it = range(int(n_boot))
    use_tqdm = False
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=desc, leave=False)
            use_tqdm = True
        except Exception:
            use_tqdm = False

    for b in it:
        sel = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([group_idx[i] for i in sel]) if sel.size else np.array([], int)
        yb, sb = y[idx], s[idx]
        if yb.size == 0 or yb.min() == yb.max():
            aucs[b] = np.nan
        else:
            aucs[b] = roc_auc_score(yb, sb)

        if progress and not use_tqdm and ((b + 1) % max(1, progress_every) == 0 or (b + 1) == n_boot):
            print(f"{desc}: {b+1}/{n_boot}", end="\r", flush=True)

    if progress and not use_tqdm:
        print()

    aucs = aucs[np.isfinite(aucs)]
    if aucs.size == 0:
        raise ValueError("All bootstrap samples degenerate (single class).")
    mean_auc = float(np.mean(aucs))
    lo = float(np.quantile(aucs, (1 - conf) / 2))
    hi = float(np.quantile(aucs, 1 - (1 - conf) / 2))
    return mean_auc, lo, hi, aucs
