# stop_design.py

import numpy as np
import pandas as pd

# ------------------------- small utilities ----------------------------------

def _zscore_nan(a):
    x = np.asarray(a, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(m):
        return np.zeros_like(x, float)
    s = s if s > 1e-12 else 1.0
    out = (x - m) / s
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def _make_rcos_basis(t, centers, width):
    t = np.asarray(t, float)[:, None]            # (N, 1)
    c = np.asarray(centers, float)[None, :]      # (1, K)
    arg = (t - c) * (np.pi / (2.0 * width))
    B = 0.5 * (1.0 + np.cos(np.clip(arg, -np.pi, np.pi)))
    B[(t < c - width) | (t > c + width)] = 0.0
    return B

def _align_meta_to_pos(meta, pos):
    if 't_center' not in meta.columns:
        if {'stop_time', 'rel_center'}.issubset(meta.columns):
            meta = meta.copy()
            meta['t_center'] = np.asarray(meta['stop_time'] + meta['rel_center'], float)
        else:
            raise ValueError('meta needs t_center, or both stop_time and rel_center to reconstruct it.')
    meta_by_bin = meta.set_index('bin_idx_array').sort_index()
    m = meta_by_bin.loc[np.asarray(pos, int)].copy()
    return m, meta_by_bin

def _build_per_stop_table(new_seg_info, extras=('cond', 'duration', 'captured', 'rewarded', 'reward_size')):
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
    per_stop = stop_tbl.set_index('stop_id')
    cols_to_add = [c for c in per_stop.columns if c not in m.columns]  # avoid overwriting e.g. stop_time
    return m.join(per_stop[cols_to_add], on='stop_id')

def _expand_cond_dummies(m, drop_first_cond=True):
    if 'cond' not in m.columns:
        return np.empty((len(m), 0), float), []
    cond_dum = pd.get_dummies(m['cond'].fillna('_none_'), prefix='cond', dtype=int)
    if drop_first_cond and cond_dum.shape[1] > 0:
        cond_dum = cond_dum.iloc[:, 1:]
    return cond_dum.to_numpy(dtype=float), list(cond_dum.columns)

def _compute_core_stop_features(m, meta_by_bin):
    rel_t = m['rel_center'].to_numpy(dtype=float)
    prepost = (~m['is_pre'].to_numpy(dtype=bool)).astype(np.float64)  # 0=pre, 1=post
    straddle = ((m['rel_left'] < 0) & (m['rel_right'] > 0)).astype(np.float64).to_numpy()
    # normalized within-stop position
    kmax_per_stop = meta_by_bin.groupby('stop_id')['k_within_stop'].max()
    k_norm = (m['k_within_stop'] /
              m['stop_id'].map(kmax_per_stop).replace(0, np.nan)).astype(float).fillna(0.0).to_numpy()
    return rel_t, prepost, straddle, k_norm

def _compute_history_base_vars(m, standardize_numeric=True):
    ts_prev = (m['t_center'] - m['prev_stop_time']).to_numpy(float)     # NaN for first stop
    ts_next = (m['next_stop_time'] - m['t_center']).to_numpy(float)     # NaN for last stop
    if standardize_numeric:
        ts_prev_f = _zscore_nan(ts_prev)
        ts_next_f = _zscore_nan(ts_next)
        duration_f = _zscore_nan(m['duration'].to_numpy(float)) if 'duration' in m.columns else np.zeros(len(m))
    else:
        ts_prev_f = np.nan_to_num(ts_prev, nan=0.0)
        ts_next_f = np.nan_to_num(ts_next, nan=0.0)
        duration_f = np.nan_to_num(m['duration'].to_numpy(float), nan=0.0) if 'duration' in m.columns else np.zeros(len(m))
    return ts_prev, ts_next, ts_prev_f, ts_next_f, duration_f

def _make_history_block(prepost, ts_prev, ts_next, ts_prev_f, ts_next_f,
                        history_mode, include_columns, history_choice='prev'):
    blocks, names = [], []
    wants_any = any(k in include_columns for k in (
        'time_since_prev_stop_z', 'time_to_next_stop_z',
        'time_since_prev_stop_z_post', 'time_to_next_stop_z_pre',
        'isi_len_z', 'mid_offset_z', 'history_gated', 'history_sumdiff'
    ))
    if history_mode == 'single':
        chosen = None
        for key in include_columns:
            if key in ('time_since_prev_stop_z', 'time_to_next_stop_z'):
                chosen = key; break
        if chosen is None and wants_any:
            chosen = 'time_since_prev_stop_z' if history_choice == 'prev' else 'time_to_next_stop_z'
        if chosen == 'time_since_prev_stop_z':
            blocks.append(ts_prev_f[:, None]); names.append('time_since_prev_stop_z')
        elif chosen == 'time_to_next_stop_z':
            blocks.append(ts_next_f[:, None]); names.append('time_to_next_stop_z')
    elif history_mode == 'gated':
        if wants_any or 'history_gated' in include_columns \
           or 'time_since_prev_stop_z_post' in include_columns \
           or 'time_to_next_stop_z_pre' in include_columns:
            ts_prev_post = prepost * ts_prev_f           # post-only
            ts_next_pre  = (1.0 - prepost) * ts_next_f   # pre-only
            blocks += [ts_prev_post[:, None], ts_next_pre[:, None]]
            names  += ['time_since_prev_stop_z_post', 'time_to_next_stop_z_pre']
    elif history_mode == 'sumdiff':
        if wants_any or 'history_sumdiff' in include_columns \
           or 'isi_len_z' in include_columns or 'mid_offset_z' in include_columns:
            isi_len      = ts_prev + ts_next
            mid_offset   = 0.5 * (ts_next - ts_prev)
            isi_len_f    = _zscore_nan(isi_len)
            mid_offset_f = _zscore_nan(mid_offset)
            blocks += [isi_len_f[:, None], mid_offset_f[:, None]]
            names  += ['isi_len_z', 'mid_offset_z']
    else:
        raise ValueError('history_mode must be one of: single, gated, sumdiff')
    return blocks, names

def _make_capture_reward_blocks(m, prepost):
    captured = m['captured'].fillna(0).to_numpy(dtype=float) if 'captured' in m.columns else None
    rewarded = m['rewarded'].fillna(0).to_numpy(dtype=float) if 'rewarded' in m.columns else None
    reward_size_z = _zscore_nan(m['reward_size'].to_numpy(float)) if 'reward_size' in m.columns else None
    rewarded_post = prepost * rewarded if rewarded is not None else None
    reward_size_z_post = prepost * reward_size_z if reward_size_z is not None else None
    return captured, rewarded_post, reward_size_z_post

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
    standardize_numeric: bool = True,
    history_mode: str = 'single',            # 'single' | 'gated' | 'sumdiff'
    history_choice: str = 'prev',            # used for 'single': 'prev'|'next'
    include_columns=(
        'prepost', 'duration_z', 'time_since_prev_stop_z', 'cond_dummies',
        # optional extras: 'straddle','k_norm','basis','prepost*speed',
        # capture/reward: 'captured','basis*captured','rewarded_post','reward_size_z_post',
    )
):
    '''
    Build stop-aware predictors aligned to your fitted rows (pos).

    Capture/Reward support (if present in new_seg_info):
      - 'captured'              : 0/1 main effect (all bins of that stop)
      - 'basis*captured'        : peri-stop shape differs if captured
      - 'rewarded_post'         : 0/1 post-only effect of reward delivery
      - 'reward_size_z_post'    : post-only graded reward size (z-scored)
    '''
    # 1) align meta → m (rows == pos) and keep full meta_by_bin for per-stop ops
    m, meta_by_bin = _align_meta_to_pos(meta, pos)

    # 2) per-stop table and safe join (avoid column collisions)
    stop_tbl = _build_per_stop_table(new_seg_info)
    m = _join_per_stop_avoid_collisions(m, stop_tbl)

    # 3) core stop features
    rel_t, prepost, straddle, k_norm = _compute_core_stop_features(m, meta_by_bin)

    # 4) history base variables + optional duration
    ts_prev, ts_next, ts_prev_f, ts_next_f, duration_f = _compute_history_base_vars(
        m, standardize_numeric=standardize_numeric
    )

    # 5) condition dummies (generic; if you have no cond column it’s a no-op)
    cond_mat, cond_cols = _expand_cond_dummies(m, drop_first_cond=drop_first_cond)

    # 6) basis over rel_t
    if rc_centers is None:
        rc_centers = np.array([-0.24, -0.16, -0.08, 0.00, 0.08, 0.16, 0.24], float)
    B = _make_rcos_basis(rel_t, centers=rc_centers, width=rc_width)
    B_names = [f'rcos_{c:+.2f}s' for c in rc_centers]

    # 7) capture / reward primitives (and post-gated variants)
    captured, rewarded_post, reward_size_z_post = _make_capture_reward_blocks(m, prepost)

    # 8) interactions prepared
    BxCaptured = None; BxCaptured_names = []
    if add_interactions and 'basis*captured' in include_columns and captured is not None:
        BxCaptured = B * captured[:, None]
        BxCaptured_names = [f'{bn}*captured' for bn in B_names]

    # 9) assemble blocks
    blocks, names = [], []

    # basic stop features
    if 'prepost' in include_columns:
        blocks.append(prepost[:, None]); names.append('prepost')
    if 'straddle' in include_columns:
        blocks.append(straddle[:, None]); names.append('straddle')
    if 'k_norm' in include_columns:
        blocks.append(k_norm[:, None]); names.append('k_norm')
    if 'duration_z' in include_columns:
        blocks.append(duration_f[:, None]); names.append('duration_z')
    if 'cond_dummies' in include_columns and cond_mat.size:
        blocks.append(cond_mat); names += cond_cols
    if 'basis' in include_columns:
        blocks.append(B); names += B_names
    if add_interactions and 'prepost*speed' in include_columns:
        x_prepost_speed = (prepost * np.asarray(speed_used, float))[:, None]
        blocks.append(x_prepost_speed); names.append('prepost*speed')

    # capture / reward
    if 'captured' in include_columns and captured is not None:
        blocks.append(captured[:, None]); names.append('captured')
    if 'basis*captured' in include_columns and BxCaptured is not None:
        blocks.append(BxCaptured); names += BxCaptured_names
    if 'rewarded_post' in include_columns and rewarded_post is not None:
        blocks.append(rewarded_post[:, None]); names.append('rewarded_post')
    if 'reward_size_z_post' in include_columns and reward_size_z_post is not None:
        blocks.append(reward_size_z_post[:, None]); names.append('reward_size_z_post')

    # history block
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


