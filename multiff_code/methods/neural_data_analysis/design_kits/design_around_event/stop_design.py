import numpy as np
import pandas as pd

# ------------------------- small utilities ----------------------------------


def _zscore_nan(a):
    """
    (Still available if you need it elsewhere.)
    Z-score an array while safely handling NaNs/Infs.
    Not used in history features anymore since you standardize later.
    """
    x = np.asarray(a, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(m):
        return np.zeros_like(x, float)
    s = s if s > 1e-12 else 1.0
    out = (x - m) / s
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _make_rcos_basis(t, centers, width):
    """
    EXPLAIN LIKE I'M A HIGH SCHOOLER 🧠✨ (FUTURE ME: READ THIS)

    What this function does (in plain English):

    Think of time as a straight line:
        -0.2   -0.1    0     0.05    0.1    0.2   (seconds)

    These are moments around some event (like a stop or a spike).

    Instead of asking sharp, annoying questions like:
        "Is this EXACTLY 50 ms after the event?"

    this function asks nicer questions like:
        "Is this time NEAR 50 ms after the event?"

    It does this by placing smooth "hills" on the timeline
    (also called raised-cosine basis functions).

    Each hill:
      - is tallest at its center time
      - slopes down smoothly
      - becomes exactly zero once you're far away

    Example hills:
      - one centered at 0.0 s
      - one centered at 0.05 s
      - one centered at 0.1 s
    (they overlap a bit)

    ------------------------------------------------------------
    What the output actually is
    ------------------------------------------------------------

    For EVERY time point, the function asks:
        "How tall is each hill here?"

    The output is a table (matrix) like:

        time    hill@0.0   hill@0.05   hill@0.1
        --------------------------------------
        -0.2      0           0           0
         0.0      1          0.5          0
         0.05     0           1          0.5
         0.1      0           0           1

    Each row = one time point
    Each column = one hill (basis function)

    At a given time point, the model computes:

        final_value =
            (how close am I to hill #1 × weight #1)
          + (how close am I to hill #2 × weight #2)
          + (how close am I to hill #3 × weight #3)


    - "How close am I?" comes from this function (the basis values)
    - "How much do I care?" comes from the learned weights (betas)

    Adding them up gives a smooth time-dependent effect.

    ------------------------------------------------------------
    Why do this instead of 0/1 time bins?
    ------------------------------------------------------------

    Asking:
        "Was it EXACTLY at 50 ms?"
    is brittle and noisy.

    Asking:
        "Was it NEAR 50 ms?"
    is smooth and forgiving.

    Brains like smooth.
    GLMs like smooth.

    Instead of learning:
        200 separate numbers for 200 time bins 😵

    the GLM learns:
        a few weights for these hills 🙂

    Then it combines the hills to make any smooth shape it wants.

    ------------------------------------------------------------
    One-sentence summary
    ------------------------------------------------------------

    This function turns time into
        "how close am I to these important moments?"
    signals, and the GLM just multiplies by weights and adds them up.
    """


    t = np.asarray(t, float)[:, None]            # (N, 1)
    c = np.asarray(centers, float)[None, :]      # (1, K)
    arg = (t - c) * (np.pi / (2.0 * width))
    B = 0.5 * (1.0 + np.cos(np.clip(arg, -np.pi, np.pi)))
    B[(t < c - width) | (t > c + width)] = 0.0
    return B


def _align_meta_to_pos(meta, pos):
    """
    Align metadata to the set of modeled rows `pos`.
    Ensures we have absolute bin center time `t_center` for each modeled bin.
    """
    if 't_center' not in meta.columns:
        if {'event_time', 'rel_center'}.issubset(meta.columns):
            meta = meta.copy()
            meta['t_center'] = np.asarray(
                meta['event_time'] + meta['rel_center'], float)
        else:
            raise ValueError(
                'meta needs t_center, or both event_time and rel_center to reconstruct it.')
    meta_by_bin = meta.set_index('bin').sort_index()
    m = meta_by_bin.loc[np.asarray(pos, int)].copy()
    return m, meta_by_bin


def build_per_event_table(new_seg_info, extras=('cond', 'duration', 'captured')):
    """
    Build a per-event table:
      - event_id, event_time
      - (optional) cond, duration, captured
      - prev_event_time, next_event_time for history context
    """
    required = {'event_id', 'event_time'}
    if not required.issubset(new_seg_info.columns):
        raise ValueError(
            'new_seg_info must contain columns: event_id, event_time')
    event_tbl = (new_seg_info[['event_id', 'event_time']]
                 .drop_duplicates('event_id')
                 .sort_values('event_time')
                 .reset_index(drop=True))
    extra_cols = [c for c in extras if c in new_seg_info.columns]
    if extra_cols:
        event_tbl = event_tbl.merge(new_seg_info[['event_id'] + extra_cols]
                                    .drop_duplicates('event_id'), on='event_id', how='left')
    event_tbl['prev_event_time'] = event_tbl['event_time'].shift(1)
    event_tbl['next_event_time'] = event_tbl['event_time'].shift(-1)
    return event_tbl


def join_event_tbl_avoid_collisions(m, event_tbl):
    """
    Left-join per-event features into per-bin metadata without overwriting existing columns.
    """
    event_tbl = event_tbl.set_index('event_id')
    cols_to_add = [c for c in event_tbl.columns if c not in m.columns]
    return m.join(event_tbl[cols_to_add], on='event_id')


def _expand_cond_dummies(m, drop_first_cond=True):
    """
    Convert condition labels into one-hot dummies: cond_x, cond_y, ...
    If drop_first_cond=True, drop one column to avoid perfect collinearity.
    """
    if 'cond' not in m.columns:
        return np.empty((len(m), 0), float), []
    cond_dum = pd.get_dummies(m['cond'].fillna(
        '_none_'), prefix='cond', dtype=int)
    if drop_first_cond and cond_dum.shape[1] > 0:
        cond_dum = cond_dum.iloc[:, 1:]
    return cond_dum.to_numpy(dtype=float), list(cond_dum.columns)


def _compute_core_event_features(m, meta_by_bin):
    """
    Core per-bin event features:
      rel_t    : time relative to event center (seconds)
      prepost  : 0=pre-event, 1=post-event
      straddle : 1 if bin spans across event time (rel_left<0<rel_right)
      k_norm   : normalized within-event position in [0,1]
    """
    rel_t = m['rel_center'].to_numpy(dtype=float)
    prepost = (~m['is_pre'].to_numpy(dtype=bool)).astype(
        np.float64)  # 0=pre, 1=post
    straddle = ((m['rel_left'] < 0) & (m['rel_right'] > 0)
                ).astype(np.float64).to_numpy()

    # normalized within-event position
    kmax_event_tbl = meta_by_bin.groupby('event_id')['k_within_seg'].max()
    k_norm = (m['k_within_seg'] /
              m['event_id'].map(kmax_event_tbl).replace(0, np.nan)).astype(float).fillna(0.0).to_numpy()
    return rel_t, prepost, straddle, k_norm


def _to_seconds(arr):
    """
    Convert a pandas Series/array that might be datetime/Timedelta/float
    into float seconds. Works for:
      - Timedelta/Datetime differences → uses .dt.total_seconds()
      - plain floats/ints → returned as float array
    """
    if isinstance(arr, pd.Series):
        if pd.api.types.is_timedelta64_dtype(arr):
            return arr.dt.total_seconds().to_numpy()
        # If datetime-like, caller should pass differences, not raw datetimes.
        return arr.astype(float).to_numpy()
    a = np.asarray(arr)
    # If it's a timedelta64 numpy array
    if np.issubdtype(a.dtype, np.timedelta64):
        return a.astype('timedelta64[ns]').astype(np.int64) / 1e9
    return a.astype(float)


def _compute_history_base_vars(m):
    """
    Primitive timing vars for history (all in seconds):
      ts_prev    : seconds since previous event (NaN if none)
      ts_next    : seconds until next event (NaN if none)
      ts_prev_f  : ts_prev with NaN→0
      ts_next_f  : ts_next with NaN→0
      duration_f : event duration (seconds, NaN→0 if missing)
    Expects columns:
      - 't_center' (time of current event alignment center)
      - 'prev_event_time' (time of previous event or NaT/NaN)
      - 'next_event_time' (time of next event or NaT/NaN)
      - optional 'duration' (seconds or Timedelta)
    """
    # Differences → may be Timedelta; convert to seconds robustly
    ts_prev_sec = _to_seconds(
        m['t_center'] - m['prev_event_time'])   # NaN for first event
    ts_next_sec = _to_seconds(
        m['next_event_time'] - m['t_center'])   # NaN for last event

    ts_prev_f = np.nan_to_num(ts_prev_sec, nan=0.0)
    ts_next_f = np.nan_to_num(ts_next_sec, nan=0.0)

    if 'duration' in m.columns:
        duration_sec = _to_seconds(m['duration'])
        duration_f = np.nan_to_num(duration_sec, nan=0.0)
    else:
        duration_f = np.zeros(len(m), dtype=float)

    return ts_prev_sec, ts_next_sec, ts_prev_f, ts_next_f, duration_f


def _make_history_block(
    include_columns,
    prepost, ts_prev, ts_next, ts_prev_f, ts_next_f,
):
    '''
    Build history-related predictors WITHOUT z-scoring (raw seconds/scalars).

    Parameters
    ----------
    include_columns : iterable of str
        Any of:
          'time_since_prev_event', 'time_to_next_event',
          'time_since_prev_event_pre', 'time_to_next_event_post',
          'isi_len', 'mid_offset',
    prepost : array-like (n_bins,)
        0 for pre-event, 1 for post-event (relative to current event_id).
    ts_prev, ts_next : arrays (n_bins,)
        Raw seconds since previous / until next event (can contain NaN).
    ts_prev_f, ts_next_f : arrays (n_bins,)
        Same as above but with NaN→0 already applied.

    Returns
    -------
    blocks : list of arrays shaped (n_bins, 1)
    names  : list of str
    '''

    # ensure 1D float arrays
    prepost = np.asarray(prepost,  float).reshape(-1)
    ts_prev = np.asarray(ts_prev,  float).reshape(-1)
    ts_next = np.asarray(ts_next,  float).reshape(-1)
    ts_prev_f = np.asarray(ts_prev_f, float).reshape(-1)
    ts_next_f = np.asarray(ts_next_f, float).reshape(-1)

    # derived features
    ts_prev_pre = (1.0 - prepost) * ts_prev_f   # active in PRE window
    ts_next_post = prepost * ts_next_f           # active in POST window

    isi_len = ts_prev + ts_next
    # negative -> closer to prev event
    mid_offset = 0.5 * (ts_next - ts_prev)
    isi_len_f = np.nan_to_num(isi_len, nan=0.0)
    mid_offset_f = np.nan_to_num(mid_offset, nan=0.0)

    # include set + bundles
    inc = set(include_columns or [])

    # build outputs (dedupe-safe)
    blocks, names = [], []

    def _add(name, col):
        if name not in names:
            blocks.append(col[:, None])
            names.append(name)

    if 'time_since_prev_event' in inc:
        _add('time_since_prev_event', ts_prev_f)
    if 'time_to_next_event' in inc:
        _add('time_to_next_event', ts_next_f)

    if 'time_since_prev_event_pre' in inc:
        _add('time_since_prev_event_pre', ts_prev_pre)
    if 'time_to_next_event_post' in inc:
        _add('time_to_next_event_post', ts_next_post)

    if 'isi_len' in inc:
        _add('isi_len', isi_len_f)
    if 'mid_offset' in inc:
        _add('mid_offset', mid_offset_f)

    return blocks, names


def _make_capture_block(m):
    """
    Capture covariate: 0/1 indicator across all bins of a event (if available).
    """
    captured = m['captured'].fillna(0).to_numpy(
        dtype=float) if 'captured' in m.columns else None
    return captured

# ------------------------- main builder -------------------------------------


def build_event_design_from_meta(
    meta: pd.DataFrame,
    pos: np.ndarray,
    new_seg_info: pd.DataFrame,
    speed_used: np.ndarray,
    *,
    rc_centers=None,
    rc_width: float = 0.10,
    add_interactions: bool = True,
    drop_first_cond: bool = True,
    include_columns=(
        'prepost', 'duration', 'time_since_prev_event', 'cond_dummies',
        # optional extras: 'straddle','k_norm','basis','prepost*speed',
        # capture: 'captured','basis*captured','prepost*captured',
    )
):
    '''
    Build event-aware predictors aligned to fitted rows (pos).

    Available columns (no z-scoring here):
      - prepost                  : 0=pre, 1=post
      - straddle                 : bin spans across the event time
      - k_norm                   : normalized within-event position [0,1]
      - duration                 : event duration in seconds
      - cond_dummies             : one-hot condition indicators
      - basis                    : raised-cosine over rel time (peri-event shape)
      - prepost*speed            : interaction (pre vs post) × (speed_used)
      - captured                 : 0/1 capture indicator (per event)
      - basis*captured           : capture-modulated peri-event basis
      - prepost*captured         : interaction of prepost and captured
      - time_since_prev_event     : seconds since previous event (NaN→0)
      - time_to_next_event        : seconds until next event (NaN→0)
      - time_since_prev_event_pre: post-only gated since-prev (pre bins=0)
      - time_to_next_event_post    : pre-only gated to-next (post bins=0)
      - isi_len                  : total inter-event interval length (seconds)
      - mid_offset               : signed offset from midpoint (seconds)
    '''
    # 1) align meta → m
    m, meta_by_bin = _align_meta_to_pos(meta, pos)

    # 2) per-event table
    event_tbl = build_per_event_table(new_seg_info)
    m = join_event_tbl_avoid_collisions(m, event_tbl)

    # 3) core event features
    rel_t, prepost, straddle, k_norm = _compute_core_event_features(
        m, meta_by_bin)

    # 4) history vars
    ts_prev, ts_next, ts_prev_f, ts_next_f, duration_f = _compute_history_base_vars(
        m)

    # 5) condition dummies
    cond_mat, cond_cols = _expand_cond_dummies(
        m, drop_first_cond=drop_first_cond)

    # 6) basis over rel_t
    if rc_centers is None:
        rc_centers = np.array(
            [-0.24, -0.16, -0.08, 0.00, 0.08, 0.16, 0.24], float)
    B = _make_rcos_basis(rel_t, centers=rc_centers, width=rc_width)
    B_names = [f'rcos_{c:+.2f}s' for c in rc_centers]

    # 7) capture primitives
    captured = _make_capture_block(m)

    # 8) interactions prepared
    BxCaptured = None
    BxCaptured_names = []
    if add_interactions and 'basis*captured' in include_columns and captured is not None:
        BxCaptured = B * captured[:, None]
        BxCaptured_names = [f'{bn}*captured' for bn in B_names]

    # 9) assemble blocks
    blocks, names = [], []

    if 'prepost' in include_columns:
        blocks.append(prepost[:, None])
        names.append('prepost')
    if 'straddle' in include_columns:
        blocks.append(straddle[:, None])
        names.append('straddle')
    if 'k_norm' in include_columns:
        blocks.append(k_norm[:, None])
        names.append('k_norm')
    if 'duration' in include_columns:
        blocks.append(duration_f[:, None])
        names.append('duration')
    if 'cond_dummies' in include_columns and cond_mat.size:
        blocks.append(cond_mat)
        names += cond_cols
    if 'basis' in include_columns:
        blocks.append(B)
        names += B_names
    if add_interactions and 'prepost*speed' in include_columns:
        x_prepost_speed = (prepost * np.asarray(speed_used, float))[:, None]
        blocks.append(x_prepost_speed)
        names.append('prepost*speed')
    if add_interactions and 'prepost*captured' in include_columns and captured is not None:
        x_prepost_captured = (prepost * np.asarray(captured, float))[:, None]
        blocks.append(x_prepost_captured)
        names.append('prepost*captured')

    if 'captured' in include_columns and captured is not None:
        blocks.append(captured[:, None])
        names.append('captured')
    if 'basis*captured' in include_columns and BxCaptured is not None:
        blocks.append(BxCaptured)
        names += BxCaptured_names

    H_blocks, H_names = _make_history_block(
        include_columns,
        prepost, ts_prev, ts_next, ts_prev_f, ts_next_f,
    )

    if H_blocks:
        blocks += H_blocks
        names += H_names

    # pack & sanitize
    if not blocks:
        X_event = np.zeros((len(m), 1), float)
        names = ['_zeros_']
    else:
        X_event = np.column_stack(blocks).astype(float)
    X_event = np.nan_to_num(X_event, nan=0.0, posinf=0.0, neginf=0.0)

    X_event_df = pd.DataFrame(X_event, columns=names,
                              index=np.arange(len(X_event)))
    return X_event_df

# ------------------------- programmatic feature glossary ---------------------

def build_event_design_for_decoding(
    meta: pd.DataFrame,
    pos: np.ndarray,
    new_seg_info: pd.DataFrame,
    drop_first_cond: bool = True,
    include_columns=(
        'prepost', 'time_since_prev_event', 'cond_dummies', 'captured'
        # optional extras: 'straddle','k_norm','basis','prepost*speed', 'duration', 'k_norm',
        # capture: 'basis*captured','prepost*captured',
    )
):
    '''
    Build event-aware predictors aligned to fitted rows (pos).

    Available columns (no z-scoring here):
      - prepost                  : 0=pre, 1=post
      - straddle                 : bin spans across the event time
      - k_norm                   : normalized within-event position [0,1]
      - duration                 : event duration in seconds
      - cond_dummies             : one-hot condition indicators
      - basis                    : raised-cosine over rel time (peri-event shape)
      - prepost*speed            : interaction (pre vs post) × (speed_used)
      - captured                 : 0/1 capture indicator (per event)
      - basis*captured           : capture-modulated peri-event basis
      - prepost*captured         : interaction of prepost and captured
      - time_since_prev_event     : seconds since previous event (NaN→0)
      - time_to_next_event        : seconds until next event (NaN→0)
      - time_since_prev_event_pre: post-only gated since-prev (pre bins=0)
      - time_to_next_event_post    : pre-only gated to-next (post bins=0)
      - isi_len                  : total inter-event interval length (seconds)
      - mid_offset               : signed offset from midpoint (seconds)
    '''
    # 1) align meta → m
    m, meta_by_bin = _align_meta_to_pos(meta, pos)

    # 2) per-event table
    event_tbl = build_per_event_table(new_seg_info)
    m = join_event_tbl_avoid_collisions(m, event_tbl)

    # 3) core event features
    rel_t, prepost, straddle, k_norm = _compute_core_event_features(
        m, meta_by_bin)

    # 4) history vars
    ts_prev, ts_next, ts_prev_f, ts_next_f, duration_f = _compute_history_base_vars(
        m)

    # 5) condition dummies
    cond_mat, cond_cols = _expand_cond_dummies(
        m, drop_first_cond=drop_first_cond)
    
    # 6) capture primitives
    captured = _make_capture_block(m)


    # 9) assemble blocks
    blocks, names = [], []

    if 'prepost' in include_columns:
        blocks.append(prepost[:, None])
        names.append('prepost')
    if 'straddle' in include_columns:
        blocks.append(straddle[:, None])
        names.append('straddle')
    if 'k_norm' in include_columns:
        blocks.append(k_norm[:, None])
        names.append('k_norm')
    if 'duration' in include_columns:
        blocks.append(duration_f[:, None])
        names.append('duration')
    if 'cond_dummies' in include_columns and cond_mat.size:
        blocks.append(cond_mat)
        names += cond_cols
    if 'captured' in include_columns and captured is not None:
        blocks.append(captured[:, None])
        names.append('captured')
        
        
    H_blocks, H_names = _make_history_block(
        include_columns,
        prepost, ts_prev, ts_next, ts_prev_f, ts_next_f,
    )

    if H_blocks:
        blocks += H_blocks
        names += H_names

    # pack & sanitize
    if not blocks:
        X_event = np.zeros((len(m), 1), float)
        names = ['_zeros_']
    else:
        X_event = np.column_stack(blocks).astype(float)
    X_event = np.nan_to_num(X_event, nan=0.0, posinf=0.0, neginf=0.0)

    X_event_df = pd.DataFrame(X_event, columns=names,
                              index=np.arange(len(X_event)))
    return X_event_df
