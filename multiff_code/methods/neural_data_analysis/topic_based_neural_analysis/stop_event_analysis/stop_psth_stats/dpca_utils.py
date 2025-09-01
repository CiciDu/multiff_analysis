import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------- minimal helpers (standalone) ----------
import numpy as np
import pandas as pd

def dpca_time_outcome(an):
    """
    dPCA-lite for two conditions (event_a/b) and time.
    Uses trial-averaged rates (Hz). Returns loadings & explained variance.
    """
    if not getattr(an, 'psth_data', None):
        an.run_full_analysis()

    seg = an.psth_data['segments']
    bw = an.config.bin_width
    time = an.psth_data['psth']['time_axis']
    U = an.n_clusters
    T = len(time)

    # condition-averaged rates per time (T x U)
    A = seg['event_a'].mean(axis=0) / bw
    B = seg['event_b'].mean(axis=0) / bw

    M = np.stack([A, B], axis=0)          # C x T x U, C=2
    grand = M.mean(axis=(0, 1))           # U

    # time marginalization: avg over condition, center by grand
    Mt = M.mean(axis=0) - grand           # T x U
    # outcome marginalization: avg over time, center by grand
    Mc = M.mean(axis=1) - grand           # C x U (here C=2)

    # SVDs give principal axes in neuron space (U)
    Ut, St, Vt = np.linalg.svd(Mt, full_matrices=False)   # Mt = Ut @ diag(St) @ Vt
    Uc, Sc, Vc = np.linalg.svd(Mc, full_matrices=False)

    # variance explained per marginalization
    var_t = (St**2).sum()
    var_c = (Sc**2).sum()
    var_total = (Mt**2).sum() + (Mc**2).sum()  # ignore interaction for this lite version

    evr_time = (St**2) / max(var_total, 1e-9)
    evr_cond = (Sc**2) / max(var_total, 1e-9)

    out = {
        'time': {
            'components': Vt,             # shape: n_comp x U (neuron loadings)
            'scores': Mt @ Vt.T,          # T x n_comp (trajectory over time)
            'explained_var': evr_time,    # fraction of total signal var
        },
        'outcome': {
            'components': Vc,
            'scores': Mc @ Vc.T,          # C x n_comp
            'explained_var': evr_cond,
        },
        'time_axis': time,
        'clusters': an.clusters,
    }
    return out


def _prestop_Xy(an, t0, t1, standardize=True):
    seg = an.psth_data['segments']; time = an.psth_data['psth']['time_axis']; bw = an.config.bin_width
    i0 = int(np.searchsorted(time, t0, side='left')); i1 = int(np.searchsorted(time, t1, side='right'))
    Xa = seg['event_a'][:, i0:i1, :].mean(axis=1) / bw
    Xb = seg['event_b'][:, i0:i1, :].mean(axis=1) / bw
    X = np.vstack([Xa, Xb])
    y = np.r_[np.ones(len(Xa), int), np.zeros(len(Xb), int)]
    if standardize:
        sc = StandardScaler().fit(X)
        X = sc.transform(X)
    return X, y

def _decode_auc_cv(X, y, k=5, seed=0, C=1.0, class_weight=None):
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=C, class_weight=class_weight, solver='lbfgs')
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

def _per_unit_auc_df(an, window=(-0.3, 0.0), k=5, seed=0, standardize=True) -> pd.DataFrame:
    seg = an.psth_data['segments']; time = an.psth_data['psth']['time_axis']; bw = an.config.bin_width
    i0 = int(np.searchsorted(time, window[0], side='left')); i1 = int(np.searchsorted(time, window[1], side='right'))
    Xa = seg['event_a'][:, i0:i1, :].mean(axis=1) / bw
    Xb = seg['event_b'][:, i0:i1, :].mean(axis=1) / bw
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa), int), np.zeros(len(Xb), int)]
    if standardize:
        sc = StandardScaler().fit(X)
        X = sc.transform(X)
    rows = []
    for j, cid in enumerate(an.clusters):
        m, s = _decode_auc_cv(X[:, [j]], y, k=k, seed=seed)
        rows.append({'cluster': int(cid) if str(cid).isdigit() else cid, 'auc': float(m), 'sd_cv': float(s)})
    df = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)
    df['delta'] = df['auc'] - 0.5
    df['rank'] = np.arange(1, len(df) + 1)
    return df[['rank', 'cluster', 'auc', 'delta', 'sd_cv']]

def per_unit_auc_df(an, window=(-0.3, 0.0), k=5, seed=0, standardize=True) -> pd.DataFrame:
    seg = an.psth_data['segments']; time = an.psth_data['psth']['time_axis']; bw = an.config.bin_width
    i0 = int(np.searchsorted(time, window[0], side='left')); i1 = int(np.searchsorted(time, window[1], side='right'))
    Xa = seg['event_a'][:, i0:i1, :].mean(axis=1) / bw
    Xb = seg['event_b'][:, i0:i1, :].mean(axis=1) / bw
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa), int), np.zeros(len(Xb), int)]
    unit_ids = an.clusters

    if standardize:
        sc = StandardScaler().fit(X)
        X = sc.transform(X)

    rows = []
    for j, cid in enumerate(unit_ids):
        m, s = decode_auc_cv(X[:, [j]], y, k=k, seed=seed)
        rows.append({'cluster': int(cid) if str(cid).isdigit() else cid, 'auc': float(m), 'sd_cv': float(s)})
    df = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)
    df['delta'] = df['auc'] - 0.5
    df['rank'] = np.arange(1, len(df) + 1)
    return df[['rank', 'cluster', 'auc', 'delta', 'sd_cv']]

def _plot_dpca_summary(out, n_bars=5):
    time_scores = out['time']['scores']; time_axis = np.asarray(out.get('time_axis'), float)
    evr_out = np.asarray(out['outcome']['explained_var'], float)
    k = int(min(n_bars, evr_out.size)) if evr_out.size else 0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    ax1.plot(time_axis, time_scores[:, 0], linewidth=2)
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('dPCA time comp 1'); ax1.set_title('Time component 1 trajectory')
    if k > 0:
        xs = np.arange(1, k + 1); ax2.bar(xs, evr_out[:k])
        ax2.set_xticks(xs); ax2.set_xlabel('Outcome component'); ax2.set_ylabel('Explained variance (fraction)')
        ax2.set_title('Outcome variance explained (top components)')
    else:
        ax2.text(0.5, 0.5, 'no outcome components', ha='center', va='center'); ax2.axis('off')
    return fig, (ax1, ax2)

def _plot_dpca_outcome_traces(an, out, comp_idx=0, title='Outcome component (time course)'):
    bw = an.config.bin_width; time = out['time_axis']; w = out['outcome']['components'][comp_idx]
    seg = an.psth_data['segments']; A = seg['event_a'].mean(axis=0) / bw; B = seg['event_b'].mean(axis=0) / bw
    proj_a = A @ w; proj_b = B @ w
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time, proj_a, label=getattr(an, 'event_a_label', 'event_a'), linewidth=2)
    ax.plot(time, proj_b, label=getattr(an, 'event_b_label', 'event_b'), linewidth=2)
    ax.axvline(0, linestyle='--', linewidth=1); ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Projection on outcome comp {comp_idx+1}'); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); return fig, ax

def _top_outcome_units(out, k=20):
    w = out['outcome']['components'][0]; clusters = np.asarray(out['clusters'])
    df = pd.DataFrame({'cluster': clusters, 'abs_loading': np.abs(w), 'loading': w})
    return df.sort_values('abs_loading', ascending=False).head(k).reset_index(drop=True)

def _correlate_loading_vs_auc(out, unit_auc_df):
    w = out['outcome']['components'][0]; clusters = np.asarray(out['clusters'])
    wtab = pd.DataFrame({'cluster': clusters, 'abs_loading': np.abs(w)})
    merged = pd.merge(unit_auc_df[['cluster','auc']], wtab, on='cluster', how='inner')
    r = np.corrcoef(merged['abs_loading'], merged['auc'])[0,1]
    return float(r), merged

def _dpca_outcome_evr_p(an, dpca_fn, n_perm=1000, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    seg = an.psth_data['segments']; A = seg['event_a']; B = seg['event_b']; nA = A.shape[0]
    out_obs = dpca_fn(an)
    evr_obs = float(out_obs['outcome']['explained_var'][0]) if len(out_obs['outcome']['explained_var']) else 0.0
    pool = np.concatenate([A, B], axis=0); null = np.empty(n_perm, float)
    for i in range(n_perm):
        idx = rng.permutation(pool.shape[0]); Aperm = pool[idx[:nA]]; Bperm = pool[idx[nA:]]
        class _Tmp: pass
        tmp = _Tmp()
        tmp.psth_data = {'segments': {'event_a': Aperm, 'event_b': Bperm}, 'psth': an.psth_data['psth']}
        tmp.config = an.config; tmp.n_clusters = an.n_clusters; tmp.clusters = an.clusters
        outp = dpca_fn(tmp)
        null[i] = float(outp['outcome']['explained_var'][0]) if len(outp['outcome']['explained_var']) else 0.0
    p = (1 + np.sum(null >= evr_obs)) / (1 + n_perm)
    return {'evr1_obs': evr_obs, 'p_value': float(p), 'null': null}

def _dpca_reconstruction_R2(an, out, n_time=1, n_outcome=1):
    bw = an.config.bin_width; time = out['time_axis']; seg = an.psth_data['segments']
    A = seg['event_a'].mean(axis=0) / bw; B = seg['event_b'].mean(axis=0) / bw
    M = np.stack([A, B], axis=0)                                # C x T x U
    Vt = out['time']['components'][:n_time, :]                  # (n_time, U)
    St_t = out['time']['scores'][:, :n_time]                    # (T, n_time)
    Tpart = np.tensordot(St_t, Vt, axes=(1, 0))                 # T x U
    Tpart = np.stack([Tpart, Tpart], axis=0)                    # C x T x U
    Vc = out['outcome']['components'][:n_outcome, :]            # (n_outcome, U)
    # recompute condition scores from means:
    Abar = A.mean(axis=0); Bbar = B.mean(axis=0); Mc = np.stack([Abar, Bbar], axis=0)  # C x U
    Sc_c = Mc @ Vc.T                                            # C x n_outcome
    Cpart = np.tensordot(Sc_c, Vc, axes=(1, 0))                 # C x U
    Cpart = np.repeat(Cpart[:, None, :], repeats=len(time), axis=1)   # C x T x U
    Mhat = Tpart + Cpart
    ss_res = np.nansum((M - Mhat)**2); ss_tot = np.nansum((M - M.mean(axis=(0,1), keepdims=True))**2)
    R2 = 1 - ss_res / (ss_tot + 1e-12)
    U = M.shape[2]; R2_unit = np.zeros(U)
    for u in range(U):
        y = M[:, :, u]; yhat = Mhat[:, :, u]
        ssr = np.nansum((y - yhat)**2); sst = np.nansum((y - y.mean())**2)
        R2_unit[u] = 1 - ssr / (sst + 1e-12)
    return {'R2': float(R2), 'R2_unit': R2_unit}


# --- add these small adapters somewhere above run_dpca_report ---

def _standardize_unit_auc_df(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Accepts any of:
      - ['cluster','auc', ...]
      - ['cluster', f'auc_{suffix}', ...]
    and returns a DataFrame containing at least:
      ['cluster', f'auc_{suffix}']
    """
    df = df.copy()
    # normalize cluster dtype a bit
    if 'cluster' not in df.columns:
        raise KeyError("unit AUC table must contain a 'cluster' column")
    # unify auc column name
    if f'auc_{suffix}' in df.columns:
        out = df[['cluster', f'auc_{suffix}']]
    elif 'auc' in df.columns:
        out = df[['cluster', 'auc']].rename(columns={'auc': f'auc_{suffix}'})
    else:
        raise KeyError(f"Expected 'auc' or 'auc_{suffix}' in columns, got {list(df.columns)}")
    # keep optional CV SD if present
    sd_col = f'sd_{suffix}'
    if sd_col in df.columns:
        out[sd_col] = df[sd_col].values
    elif 'sd_cv' in df.columns:
        out[sd_col] = df['sd_cv'].values
    return out

def _has_event_keys(an) -> tuple[str, str]:
    """
    Make sure we can access segments whether they are named event_a/b or legacy capture/miss.
    Returns the keys to use: ('event_a', 'event_b') or ('capture','miss').
    """
    seg = an.psth_data['segments']
    if isinstance(seg, dict):
        keys = set(seg.keys())
        if {'event_a','event_b'} <= keys:
            return 'event_a', 'event_b'
        if {'capture','miss'} <= keys:
            return 'capture', 'miss'
    # if we land here, raise a helpful error
    raise KeyError(f"segments has keys {list(seg.keys())} but needs 'event_a'/'event_b' (or 'capture'/'miss').")

# ---------- the report ----------
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def per_unit_auc_df(an,
                    window=(-0.3, 0.0),
                    k=5,
                    seed=0,
                    standardize=True,
                    C=1.0,
                    class_weight=None,
                    max_iter=1000) -> pd.DataFrame:
    """
    Single-unit decoding AUCs (event_a vs event_b) in a time window.

    Returns a DataFrame with columns:
      ['rank','cluster','auc','delta','sd_cv']

    - 'auc' is mean CV AUC for that unit alone
    - 'sd_cv' is across-fold SD
    - 'delta' = auc - 0.5
    - 'cluster' uses original labels from an.clusters
    """
    if not getattr(an, 'psth_data', None):
        an.run_full_analysis()

    seg   = an.psth_data['segments']
    time  = an.psth_data['psth']['time_axis']
    bw    = an.config.bin_width
    i0    = int(np.searchsorted(time, window[0], side='left'))
    i1    = int(np.searchsorted(time, window[1], side='right'))

    Xa = seg['event_a'][:, i0:i1, :].mean(axis=1) / bw   # nA x U
    Xb = seg['event_b'][:, i0:i1, :].mean(axis=1) / bw   # nB x U
    X  = np.vstack([Xa, Xb])                             # N x U
    y  = np.r_[np.ones(len(Xa), int), np.zeros(len(Xb), int)]

    if standardize:
        sc = StandardScaler().fit(X)
        X = sc.transform(X)

    cv = StratifiedKFold(k, shuffle=True, random_state=seed)

    rows = []
    for j, cid in enumerate(an.clusters):
        aucs = []
        for tr, te in cv.split(X, y):
            clf = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight, solver='lbfgs')
            clf.fit(X[tr, [j]], y[tr])  # single unit j
            p = clf.predict_proba(X[te, [j]])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        m = float(np.mean(aucs))
        s = float(np.std(aucs))
        rows.append({'cluster': int(cid) if str(cid).isdigit() else cid, 'auc': m, 'sd_cv': s})

    df = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)
    df['delta'] = df['auc'] - 0.5
    df['rank']  = np.arange(1, len(df) + 1)
    return df[['rank', 'cluster', 'auc', 'delta', 'sd_cv']]


def run_dpca_report(an,
                    dpca_fn,
                    outcome_comp=0,
                    pre_window=(-0.3, 0.0),
                    post_window=(0.05, 0.35),
                    top_k=20,
                    n_perm=1000,
                    seed=0,
                    show=True):
    """
    Build a compact dPCA report from an analyzer 'an' and a callable 'dpca_fn(an) -> out'.
    Returns a dict with figures and summary tables.
    """
    if not getattr(an, 'psth_data', None):
        an.run_full_analysis()

    # 1) run dPCA
    out = dpca_fn(an)

    # 2) figures: summary + outcome traces
    fig_sum, _ = _plot_dpca_summary(out)
    fig_tr, _ = _plot_dpca_outcome_traces(an, out, comp_idx=outcome_comp)

    # 3) stats: outcome EVR permutation p
    evr_perm = _dpca_outcome_evr_p(an, dpca_fn, n_perm=n_perm, rng=np.random.default_rng(seed))

    # 4) top outcome units
    top_units = _top_outcome_units(out, k=top_k)


    # 5) per-unit AUCs pre/post + correlation with loadings
    try:
        df_pre  = _per_unit_auc_df(an, window=pre_window,  k=5, seed=seed, standardize=True)
        df_post = _per_unit_auc_df(an, window=post_window, k=5, seed=seed, standardize=True)
    except Exception:
        # if you're using an external per_unit_auc_df, just use whatever it returns
        df_pre  = per_unit_auc_df(an, window=pre_window,  k=5, seed=seed, standardize=True)   # noqa: F821
        df_post = per_unit_auc_df(an, window=post_window, k=5, seed=seed, standardize=True)   # noqa: F821

    pre_std  = _standardize_unit_auc_df(df_pre,  'pre')   # ensures ['cluster','auc_pre',('sd_pre')]
    post_std = _standardize_unit_auc_df(df_post, 'post')  # ensures ['cluster','auc_post',('sd_post')]

    wide = pd.merge(pre_std, post_std, on='cluster', how='inner')
    wide['delta'] = wide['auc_post'] - wide['auc_pre']
    wide = wide.sort_values(['auc_post'], ascending=False).reset_index(drop=True)

    # for loading↔AUC correlation, use pre window AUCs (rename to 'auc')
    pre_for_corr = pre_std.rename(columns={'auc_pre': 'auc'})[['cluster','auc']]
    r_loading_auc, loading_vs_auc = _correlate_loading_vs_auc(out, pre_for_corr)

    # 6) reconstruction R^2
    recon = _dpca_reconstruction_R2(an, out, n_time=1, n_outcome=1)

    # 7) quick pre/post decoder baselines (all units)
    Xpre, y = _prestop_Xy(an, *pre_window, standardize=True)
    Xpost, _ = _prestop_Xy(an, *post_window, standardize=True)
    auc_pre_mean, auc_pre_sd   = _decode_auc_cv(Xpre,  y, k=5, seed=seed)
    auc_post_mean, auc_post_sd = _decode_auc_cv(Xpost, y, k=5, seed=seed)

    if show:
        plt.show()

    return {
        'out': out,
        'fig_summary': fig_sum,
        'fig_outcome_traces': fig_tr,
        'evr_perm': evr_perm,                 # {'evr1_obs', 'p_value', 'null'}
        'top_outcome_units': top_units,       # DataFrame
        'unit_auc_pre': df_pre,               # DataFrame
        'unit_auc_post': df_post,             # DataFrame
        'unit_auc_wide': wide,                # DataFrame with pre/post AUC
        'loading_auc_corr_r': r_loading_auc,  # float
        'loading_auc_corr_table': loading_vs_auc,  # DataFrame
        'reconstruction': recon,              # {'R2', 'R2_unit'}
        'auc_all_units': {'pre_mean': auc_pre_mean, 'pre_sd': auc_pre_sd,
                          'post_mean': auc_post_mean, 'post_sd': auc_post_sd},
        'notes': 'Outcome EVR p from label permutations; reconstruction uses 1 time + 1 outcome comp.'
    }


