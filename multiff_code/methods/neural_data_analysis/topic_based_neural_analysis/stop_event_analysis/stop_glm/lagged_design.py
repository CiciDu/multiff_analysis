import numpy as np
import pandas as pd

# ----- helpers you can reuse -----
def make_integer_lags(lag_min_s, lag_max_s, bin_dt):
    Lmin = int(np.floor(lag_min_s / bin_dt))
    Lmax = int(np.ceil(lag_max_s / bin_dt))
    return list(range(Lmin, Lmax + 1))

def _nlin(x, c=1e-5):
    return np.log(x + c)

def _inv_nlin(y, c=1e-5):
    return np.exp(y) - c

def make_raised_cosine_basis(n_basis, lag_min_s, lag_max_s, bin_dt, eps=1e-5):
    lags_bins = make_integer_lags(lag_min_s, lag_max_s, bin_dt)
    t = np.array(lags_bins, float) * bin_dt
    tmin, tmax = t[0], t[-1]
    yrng = _nlin(np.array([max(eps, tmin - tmin), tmax - tmin]), c=eps)
    centers = np.linspace(yrng[0], yrng[1], n_basis)
    centers = _inv_nlin(centers, c=eps) + tmin
    db = (centers[-1] - centers[0]) / (n_basis - 1) if n_basis > 1 else (tmax - tmin if (tmax - tmin) > 0 else bin_dt)
    width = 2 * db
    B = np.zeros((t.size, n_basis))
    for i, c0 in enumerate(centers):
        arg = (t - c0) * (np.pi / width)
        bi = (np.cos(np.clip(arg, -np.pi, np.pi)) + 1.0) / 2.0
        bi[(t < c0 - width) | (t > c0 + width)] = 0.0
        B[:, i] = bi
    basis_df = pd.DataFrame(B, index=lags_bins, columns=[f'b{i}' for i in range(n_basis)])
    return basis_df

# ----- main builder (group-aware, no leakage across stops/segments) -----
def build_lagged_design_by_group(
    df_X,
    df_Y,
    group_col,                      # e.g., 'stop_id' or 'segment_id'
    predictors,                     # list[str], columns in df_X
    order_col=None,                 # optional: sort within group by this time column
    lags_bins=None,                 # list[int] like [-15..+20], OR
    basis_df=None,                  # DataFrame from make_raised_cosine_basis (index = lags)
    keep_cols=None                  # optional: extra columns to carry through (e.g., 'stop_id', 'trial_id')
):
    """
    Returns
    -------
    df_X_design : DataFrame
        Design matrix with lagged (or basis-projected) predictors, offset, group_col (and keep_cols).
        Rows with NaNs introduced by shifting are dropped (per-group leak-safe).
    df_Y_aligned : DataFrame
        df_Y subset aligned to df_X_design.index.
    """
    if (lags_bins is None) == (basis_df is None):
        raise ValueError('Provide exactly one of lags_bins or basis_df.')

    for p in predictors:
        if p not in df_X.columns:
            raise KeyError(f'predictor {p!r} not found in df_X')
    if group_col not in df_X.columns:
        raise KeyError(f'group_col {group_col!r} not found in df_X')

    # ensure alignment of indices
    if not df_X.index.equals(df_Y.index):
        # align by index intersection (common case: already aligned)
        common_idx = df_X.index.intersection(df_Y.index)
        df_X = df_X.loc[common_idx].copy()
        df_Y = df_Y.loc[common_idx].copy()

    # sort for stable, within-group temporal order
    if order_col is not None:
        df_X = df_X.sort_values([group_col, order_col], kind='mergesort')
        df_Y = df_Y.loc[df_X.index]
    else:
        # keep current order; assume already time-ordered within each group
        pass

    # carry-through columns
    carry_cols = [group_col]
    if keep_cols:
        for c in keep_cols:
            if c not in df_X.columns:
                raise KeyError(f'keep_col {c!r} not found in df_X')
        carry_cols += list(keep_cols)

    # per-group construction to avoid leakage across boundaries
    blocks = []
    for gid, gX in df_X.groupby(group_col, sort=False):
        gY = df_Y.loc[gX.index]

        # build discrete lags per predictor (group-local shifting)
        if basis_df is None:
            # discrete lag design
            cols = {}
            for p in predictors:
                s = gX[p].astype(float)
                for k in lags_bins:
                    cols[f'{p}[lag={k}]'] = s.shift(k)  # shift within group only
            Xg = pd.DataFrame(cols, index=gX.index)
        else:
            # basis-projected design (via discrete-lag staging)
            lag_list = list(basis_df.index)
            stage = {}
            for p in predictors:
                s = gX[p].astype(float)
                for k in lag_list:
                    stage[f'{p}[lag={k}]'] = s.shift(k)
            stage = pd.DataFrame(stage, index=gX.index)

            # multiply by basis per predictor
            proj_blocks = []
            for p in predictors:
                # select and sort lags for predictor p
                sub = stage[[c for c in stage.columns if c.startswith(f'{p}[lag=')]].copy()
                lag_idx = [int(c.split('=')[1].strip(']')) for c in sub.columns]
                order = np.argsort(lag_idx)
                sub = sub.iloc[:, order]
                sorted_lags = list(np.array(lag_idx)[order])
                # align basis rows to these lags
                B = basis_df.loc[sorted_lags].to_numpy()  # (T x B)
                XB = sub.to_numpy() @ B                    # (N x B)
                XB = pd.DataFrame(XB, index=sub.index, columns=[f'{p}*{b}' for b in basis_df.columns])
                proj_blocks.append(XB)
            Xg = pd.concat(proj_blocks, axis=1)


        Xg[group_col] = gid
        if keep_cols:
            for c in keep_cols:
                Xg[c] = gX[c].to_numpy()

        # drop rows with any NaNs produced by shifts (edges of each group)
        Xg = Xg.dropna(axis=0, how='any')

        # align Y to Xg after drop
        Yg = gY.loc[Xg.index]
        if Yg.empty or Xg.empty:
            continue   

        blocks.append((Xg, Yg))

    if not blocks:
        raise ValueError('No rows left after lagging and NaN dropping. Check lags / grouping / windows.')

    # concatenate groups
    X_list, Y_list = zip(*blocks)
    df_X_design = pd.concat(X_list, axis=0)
    df_Y_aligned = pd.concat(Y_list, axis=0)

    # final alignment sanity
    df_Y_aligned = df_Y_aligned.loc[df_X_design.index]
    return df_X_design, df_Y_aligned
