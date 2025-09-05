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
    Raised cosine basis functions over relative time t (seconds).
    Each basis is centered at a value in `centers` with half-width `width`.
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
        if {'stop_time', 'rel_center'}.issubset(meta.columns):
            meta = meta.copy()
            meta['t_center'] = np.asarray(meta['stop_time'] + meta['rel_center'], float)
        else:
            raise ValueError('meta needs t_center, or both stop_time and rel_center to reconstruct it.')
    meta_by_bin = meta.set_index('bin').sort_index()
    m = meta_by_bin.loc[np.asarray(pos, int)].copy()
    return m, meta_by_bin

def _build_per_stop_table(new_seg_info, extras=('cond', 'duration', 'captured')):
    """
    Build a per-stop table:
      - stop_id, stop_time
      - (optional) cond, duration, captured
      - prev_stop_time, next_stop_time for history context
    """
    required = {'stop_id', 'stop_time'}
    if not required.issubset(new_seg_info.columns):
        raise ValueError('new_seg_info must contain columns: stop_id, stop_time')
    stop_tbl = (new_seg_info[['stop_id', 'stop_time']]
                .drop_duplicates('stop_id')
                .sort_values('stop_time')
                .reset_index(drop=True))
    extra_cols = [c for c in extras if c in new_seg_info.columns]
    if extra_cols:
        stop_tbl = stop_tbl.merge(new_seg_info[['stop_id'] + extra_cols]
                                  .drop_duplicates('stop_id'), on='stop_id', how='left')
    stop_tbl['prev_stop_time'] = stop_tbl['stop_time'].shift(1)
    stop_tbl['next_stop_time'] = stop_tbl['stop_time'].shift(-1)
    return stop_tbl

def _join_per_stop_avoid_collisions(m, stop_tbl):
    """
    Left-join per-stop features into per-bin metadata without overwriting existing columns.
    """
    per_stop = stop_tbl.set_index('stop_id')
    cols_to_add = [c for c in per_stop.columns if c not in m.columns]
    return m.join(per_stop[cols_to_add], on='stop_id')

def _expand_cond_dummies(m, drop_first_cond=True):
    """
    Convert condition labels into one-hot dummies: cond_x, cond_y, ...
    If drop_first_cond=True, drop one column to avoid perfect collinearity.
    """
    if 'cond' not in m.columns:
        return np.empty((len(m), 0), float), []
    cond_dum = pd.get_dummies(m['cond'].fillna('_none_'), prefix='cond', dtype=int)
    if drop_first_cond and cond_dum.shape[1] > 0:
        cond_dum = cond_dum.iloc[:, 1:]
    return cond_dum.to_numpy(dtype=float), list(cond_dum.columns)

def _compute_core_stop_features(m, meta_by_bin):
    """
    Core per-bin stop features:
      rel_t    : time relative to stop center (seconds)
      prepost  : 0=pre-stop, 1=post-stop
      straddle : 1 if bin spans across stop time (rel_left<0<rel_right)
      k_norm   : normalized within-stop position in [0,1]
    """
    rel_t = m['rel_center'].to_numpy(dtype=float)
    prepost = (~m['is_pre'].to_numpy(dtype=bool)).astype(np.float64)  # 0=pre, 1=post
    straddle = ((m['rel_left'] < 0) & (m['rel_right'] > 0)).astype(np.float64).to_numpy()

    # normalized within-stop position
    kmax_per_stop = meta_by_bin.groupby('stop_id')['k_within_stop'].max()
    k_norm = (m['k_within_stop'] /
              m['stop_id'].map(kmax_per_stop).replace(0, np.nan)).astype(float).fillna(0.0).to_numpy()
    return rel_t, prepost, straddle, k_norm

def _compute_history_base_vars(m):
    """
    Primitive timing vars for history:
      ts_prev   : seconds since previous stop (NaN if none)
      ts_next   : seconds until next stop (NaN if none)
      ts_prev_f : ts_prev with NaN→0
      ts_next_f : ts_next with NaN→0
      duration_f: stop duration (seconds, NaN→0)
    """
    ts_prev = (m['t_center'] - m['prev_stop_time']).to_numpy(float)     # NaN for first stop
    ts_next = (m['next_stop_time'] - m['t_center']).to_numpy(float)     # NaN for last stop
    ts_prev_f = np.nan_to_num(ts_prev, nan=0.0)
    ts_next_f = np.nan_to_num(ts_next, nan=0.0)
    duration_f = np.nan_to_num(m['duration'].to_numpy(float), nan=0.0) if 'duration' in m.columns else np.zeros(len(m))
    return ts_prev, ts_next, ts_prev_f, ts_next_f, duration_f

def _make_history_block(prepost, ts_prev, ts_next, ts_prev_f, ts_next_f,
                        history_mode, include_columns, history_choice='prev'):
    """
    Build history-related predictors WITHOUT z-scoring (raw seconds/scalars).

    Modes:
      'single' :
         - time_since_prev_stop (raw seconds)
         - time_to_next_stop   (raw seconds)
         (choose one depending on include_columns / history_choice)

      'gated'  :
         - time_since_prev_stop_post = prepost * ts_prev_f  (post-only)
         - time_to_next_stop_pre     = (1-prepost) * ts_next_f (pre-only)

      'sumdiff':
         - isi_len    = ts_prev + ts_next  (total inter-stop interval length, seconds)
         - mid_offset = 0.5 * (ts_next - ts_prev)  (signed offset from interval midpoint, seconds)
    """
    blocks, names = [], []
    wants_any = any(k in include_columns for k in (
        'time_since_prev_stop', 'time_to_next_stop',
        'time_since_prev_stop_post', 'time_to_next_stop_pre',
        'isi_len', 'mid_offset', 'history_gated', 'history_sumdiff'
    ))

    if history_mode == 'single':
        chosen = None
        for key in include_columns:
            if key in ('time_since_prev_stop', 'time_to_next_stop'):
                chosen = key
                break
        if chosen is None and wants_any:
            chosen = 'time_since_prev_stop' if history_choice == 'prev' else 'time_to_next_stop'

        if chosen == 'time_since_prev_stop':
            blocks.append(ts_prev_f[:, None]); names.append('time_since_prev_stop')
        elif chosen == 'time_to_next_stop':
            blocks.append(ts_next_f[:, None]); names.append('time_to_next_stop')

    elif history_mode == 'gated':
        if wants_any or 'history_gated' in include_columns \
           or 'time_since_prev_stop_post' in include_columns \
           or 'time_to_next_stop_pre' in include_columns:
            ts_prev_post = prepost * ts_prev_f           # only active post-stop
            ts_next_pre  = (1.0 - prepost) * ts_next_f    # only active pre-stop
            blocks += [ts_prev_post[:, None], ts_next_pre[:, None]]
            names  += ['time_since_prev_stop_post', 'time_to_next_stop_pre']

    elif history_mode == 'sumdiff':
        if wants_any or 'history_sumdiff' in include_columns \
           or 'isi_len' in include_columns or 'mid_offset' in include_columns:
            isi_len    = ts_prev + ts_next          # seconds
            mid_offset = 0.5 * (ts_next - ts_prev)  # seconds; negative→closer to prev
            isi_len_f    = np.nan_to_num(isi_len, nan=0.0)
            mid_offset_f = np.nan_to_num(mid_offset, nan=0.0)
            blocks += [isi_len_f[:, None], mid_offset_f[:, None]]
            names  += ['isi_len', 'mid_offset']
    else:
        raise ValueError('history_mode must be one of: single, gated, sumdiff')

    return blocks, names

def _make_capture_block(m):
    """
    Capture covariate: 0/1 indicator across all bins of a stop (if available).
    """
    captured = m['captured'].fillna(0).to_numpy(dtype=float) if 'captured' in m.columns else None
    return captured

# ------------------------- main builder -------------------------------------

def build_stop_design_from_meta(
    meta: pd.DataFrame,
    pos: np.ndarray,
    new_seg_info: pd.DataFrame,
    speed_used: np.ndarray,
    *,
    rc_centers=None,
    rc_width: float = 0.10,
    add_interactions: bool = True,
    drop_first_cond: bool = True,
    history_mode: str = 'gated',            # 'single' | 'gated' | 'sumdiff'
    history_choice: str = 'prev',            # used for 'single': 'prev'|'next'
    include_columns=(
        'prepost', 'duration', 'time_since_prev_stop', 'cond_dummies',
        # optional extras: 'straddle','k_norm','basis','prepost*speed',
        # capture: 'captured','basis*captured','prepost*captured',
    )
):
    '''
    Build stop-aware predictors aligned to fitted rows (pos).

    Available columns (no z-scoring here):
      - prepost                  : 0=pre, 1=post
      - straddle                 : bin spans across the stop time
      - k_norm                   : normalized within-stop position [0,1]
      - duration                 : stop duration in seconds
      - cond_dummies             : one-hot condition indicators
      - basis                    : raised-cosine over rel time (peri-stop shape)
      - prepost*speed            : interaction (pre vs post) × (speed_used)
      - captured                 : 0/1 capture indicator (per stop)
      - basis*captured           : capture-modulated peri-stop basis
      - prepost*captured         : interaction of prepost and captured
      - time_since_prev_stop     : seconds since previous stop (NaN→0)
      - time_to_next_stop        : seconds until next stop (NaN→0)
      - time_since_prev_stop_post: post-only gated since-prev (pre bins=0)
      - time_to_next_stop_pre    : pre-only gated to-next (post bins=0)
      - isi_len                  : total inter-stop interval length (seconds)
      - mid_offset               : signed offset from midpoint (seconds)
    '''
    # 1) align meta → m
    m, meta_by_bin = _align_meta_to_pos(meta, pos)

    # 2) per-stop table
    stop_tbl = _build_per_stop_table(new_seg_info)
    m = _join_per_stop_avoid_collisions(m, stop_tbl)

    # 3) core stop features
    rel_t, prepost, straddle, k_norm = _compute_core_stop_features(m, meta_by_bin)

    # 4) history vars
    ts_prev, ts_next, ts_prev_f, ts_next_f, duration_f = _compute_history_base_vars(m)

    # 5) condition dummies
    cond_mat, cond_cols = _expand_cond_dummies(m, drop_first_cond=drop_first_cond)

    # 6) basis over rel_t
    if rc_centers is None:
        rc_centers = np.array([-0.24, -0.16, -0.08, 0.00, 0.08, 0.16, 0.24], float)
    B = _make_rcos_basis(rel_t, centers=rc_centers, width=rc_width)
    B_names = [f'rcos_{c:+.2f}s' for c in rc_centers]

    # 7) capture primitives
    captured = _make_capture_block(m)

    # 8) interactions prepared
    BxCaptured = None; BxCaptured_names = []
    if add_interactions and 'basis*captured' in include_columns and captured is not None:
        BxCaptured = B * captured[:, None]
        BxCaptured_names = [f'{bn}*captured' for bn in B_names]

    # 9) assemble blocks
    blocks, names = [], []

    if 'prepost' in include_columns:
        blocks.append(prepost[:, None]); names.append('prepost')
    if 'straddle' in include_columns:
        blocks.append(straddle[:, None]); names.append('straddle')
    if 'k_norm' in include_columns:
        blocks.append(k_norm[:, None]); names.append('k_norm')
    if 'duration' in include_columns:
        blocks.append(duration_f[:, None]); names.append('duration')
    if 'cond_dummies' in include_columns and cond_mat.size:
        blocks.append(cond_mat); names += cond_cols
    if 'basis' in include_columns:
        blocks.append(B); names += B_names
    if add_interactions and 'prepost*speed' in include_columns:
        x_prepost_speed = (prepost * np.asarray(speed_used, float))[:, None]
        blocks.append(x_prepost_speed); names.append('prepost*speed')
    if add_interactions and 'prepost*captured' in include_columns and captured is not None:
        x_prepost_captured = (prepost * np.asarray(captured, float))[:, None]
        blocks.append(x_prepost_captured); names.append('prepost*captured')

    if 'captured' in include_columns and captured is not None:
        blocks.append(captured[:, None]); names.append('captured')
    if 'basis*captured' in include_columns and BxCaptured is not None:
        blocks.append(BxCaptured); names += BxCaptured_names

    # history block (no z-scoring)
    H_blocks, H_names = _make_history_block(
        prepost, ts_prev, ts_next, ts_prev_f, ts_next_f,
        history_mode, include_columns, history_choice=history_choice
    )
    if H_blocks:
        blocks += H_blocks; names += H_names

    # pack & sanitize
    if not blocks:
        X_stop = np.zeros((len(m), 1), float); names = ['_zeros_']
    else:
        X_stop = np.column_stack(blocks).astype(float)
    X_stop = np.nan_to_num(X_stop, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_stop_df = pd.DataFrame(X_stop, columns=names, index=np.arange(len(X_stop)))
    return X_stop_df

# ------------------------- programmatic feature glossary ---------------------

FEATURE_DESCRIPTIONS = {
    # core stop context
    'prepost': '0 for bins before the stop center; 1 for bins after the stop center.',
    'straddle': '1 if a bin’s left/right edges cross the stop center (rel_left<0<rel_right); else 0.',
    'k_norm': 'Normalized within-stop position in [0,1], based on k_within_stop / max k within that stop.',
    'duration': 'Stop duration in seconds for the current stop (NaN→0).',

    # condition dummies
    # (actual names are generated like 'cond_X', 'cond_Y', ... depending on your data)
    'cond_*': 'One-hot dummy columns for condition labels (first level may be dropped to avoid collinearity).',

    # peri-stop temporal basis
    # (actual names are generated like 'rcos_-0.08s', 'rcos_+0.08s', ...)
    'rcos_*': 'Raised cosine basis functions over time relative to stop center (rel_t).',

    # interactions
    'prepost*speed': 'Interaction between prepost (0/1) and the instantaneous speed used for this model row.',
    'prepost*captured': 'Interaction between prepost (0/1) and the captured indicator (0/1).',
    'basis*captured': 'Elementwise product of peri-stop basis columns and the captured indicator.',

    # capture
    'captured': 'Per-stop 0/1 flag: 1 if this stop culminated in capture; else 0.',

    # history: single
    'time_since_prev_stop': 'Seconds since the previous stop (NaN→0 for first stop).',
    'time_to_next_stop': 'Seconds until the next stop (NaN→0 for last stop).',

    # history: gated
    'time_since_prev_stop_post': 'Post-only version of time_since_prev_stop; pre bins are 0.',
    'time_to_next_stop_pre': 'Pre-only version of time_to_next_stop; post bins are 0.',

    # history: sumdiff
    'isi_len': 'Total inter-stop interval length in seconds: (next_stop_time - prev_stop_time).',
    'mid_offset': 'Signed offset from the midpoint of the inter-stop interval in seconds; negative=closer to prev stop, positive=closer to next stop.',
}
