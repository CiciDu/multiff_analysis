import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import core_stops_psth, get_stops_utils, psth_postprocessing, psth_stats

# ---------- schema & key utilities ----------
def _ensure_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'stop_time' not in out.columns:
        if 'time' in out.columns:
            out = out.rename(columns={'time': 'stop_time'})
        else:
            raise ValueError('event df must have stop_time or time')
    if 'stop_point_index' not in out.columns:
        if 'point_index' in out.columns:
            out = out.rename(columns={'point_index': 'stop_point_index'})
        else:
            out['stop_point_index'] = np.nan
    return out.sort_values('stop_time', kind='stable').reset_index(drop=True)

def _infer_key_cols(df: pd.DataFrame) -> list[str]:
    if 'stop_id' in df.columns:
        return ['stop_id']
    if {'stop_point_index', 'stop_time'}.issubset(df.columns):
        return ['stop_point_index', 'stop_time']
    return ['stop_time']

def _key_series(df: pd.DataFrame, keys: list[str] | None = None, time_round: int = 3) -> pd.Series:
    if keys is None:
        keys = _infer_key_cols(df)
    vals = []
    for k in keys:
        if k == 'stop_time':
            vals.append(df[k].round(time_round))
        else:
            vals.append(df[k])
    return pd.Series(list(zip(*vals)), index=df.index)

def _dedupe_within(df: pd.DataFrame, keys: list[str] | None = None, time_round: int = 3):
    ks = _key_series(df, keys, time_round)
    keep = ~ks.duplicated(keep='first')
    removed = int((~keep).sum())
    return df.loc[keep].copy(), {'removed_within': removed}

def _report_overlap(A: pd.DataFrame, B: pd.DataFrame, keys: list[str] | None = None, time_round: int = 3):
    kA = _key_series(A, keys, time_round)
    kB = _key_series(B, keys, time_round)
    overlap = list(set(kA) & set(kB))
    info = {
        'overlap_pairs': len(overlap),
        'example_keys': overlap[:5]  # small peek
    }
    return info

# ---------- optional per-stop matching ----------
def _standardize_concat(A: pd.DataFrame, B: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    combo = pd.concat([A[features], B[features]], axis=0)
    mu = combo.mean(axis=0)
    sd = combo.std(axis=0).replace(0, 1.0)
    ZA = (A[features] - mu) / sd
    ZB = (B[features] - mu) / sd
    return ZA, ZB

def match_events(
    A: pd.DataFrame,
    B: pd.DataFrame,
    features: list[str],
    strategy: str = 'hungarian',       # 'hungarian' | 'greedy'
    caliper: float | None = None,      # max distance in standardized space
    random_seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return matched (A', B') using standardized feature space.
    - 'hungarian': optimal 1:1 assignment up to min(len(A), len(B)).
    - 'greedy': randomized greedy nearest-neighbor (lighter memory).
    'caliper' drops pairs with distance > caliper.
    """
    if not features:
        raise ValueError('features must be non-empty')
    for f in features:
        if f not in A.columns or f not in B.columns:
            raise ValueError(f'missing feature {f} in A or B')

    ZA, ZB = _standardize_concat(A, B, features)
    A_mat = ZA.to_numpy()
    B_mat = ZB.to_numpy()

    # pairwise squared distances (may be large; use greedy if memory is tight)
    if strategy == 'hungarian':
        D = ((A_mat[:, None, :] - B_mat[None, :, :]) ** 2).sum(axis=2)
        r_ind, c_ind = linear_sum_assignment(D)  # returns min(lenA, lenB) pairs
        dsel = D[r_ind, c_ind]
        mask = np.ones_like(dsel, dtype=bool)
        if caliper is not None:
            mask = dsel <= caliper**2
        r_keep = r_ind[mask]
        c_keep = c_ind[mask]
    elif strategy == 'greedy':
        rng = np.random.default_rng(random_seed)
        order = rng.permutation(A_mat.shape[0])
        used_b = np.zeros(B_mat.shape[0], dtype=bool)
        r_keep, c_keep, dists = [], [], []
        for ai in order:
            db = ((A_mat[ai][None, :] - B_mat) ** 2).sum(axis=1)
            db[used_b] = np.inf
            bj = int(np.argmin(db))
            if not np.isfinite(db[bj]):
                continue
            if caliper is None or db[bj] <= caliper**2:
                used_b[bj] = True
                r_keep.append(ai)
                c_keep.append(bj)
                dists.append(db[bj])
        r_keep, c_keep = np.array(r_keep, int), np.array(c_keep, int)
    else:
        raise ValueError("strategy must be 'hungarian' or 'greedy'")

    A_out = A.iloc[r_keep].reset_index(drop=True)
    B_out = B.iloc[c_keep].reset_index(drop=True)
    return A_out, B_out


def match_features_func(results, key, A1, B1, match_features, match_strategy, match_caliper):
    try:
        A2, B2 = match_events(A1, B1, match_features, strategy=match_strategy, caliper=match_caliper)
        results['matching_logs'][key] = {
            'strategy': match_strategy,
            'caliper': match_caliper,
            'n_matched': len(A2),
            'len_A_in': len(A1),
            'len_B_in': len(B1),
        }
    except Exception as e:
        print(f'warning [{key}]: matching failed with error: {e}. continuing without matching.')
        results['matching_logs'][key] = {'error': str(e)}
        A2, B2 = A1, B1
    return A2, B2, results



def ensure_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize minimal schema: stop_time, stop_point_index; sort by stop_time."""
    out = df.copy()
    if 'stop_time' not in out.columns:
        if 'time' in out.columns:
            out = out.rename(columns={'time': 'stop_time'})
        else:
            raise ValueError('event df must have stop_time or time')
    if 'stop_point_index' not in out.columns:
        if 'point_index' in out.columns:
            out = out.rename(columns={'point_index': 'stop_point_index'})
        else:
            out['stop_point_index'] = np.nan
    return out.sort_values('stop_time', kind='stable').reset_index(drop=True)

def dedupe_within(df: pd.DataFrame,
                  keys: list[str] | None = None,
                  time_round: int = 3) -> pd.DataFrame:
    """Drop exact duplicate stops within a set."""
    if keys is None:
        keys = ['stop_id'] if 'stop_id' in df.columns else ['stop_point_index', 'stop_time'] if {'stop_point_index','stop_time'}.issubset(df.columns) else ['stop_time']
    key_vals = []
    for k in keys:
        v = df[k].round(time_round) if k == 'stop_time' else df[k]
        key_vals.append(v)
    key = pd.Series(list(zip(*key_vals)), index=df.index)
    keep = ~key.duplicated(keep='first')
    return df.loc[keep].copy()

def diff_by(a: pd.DataFrame, b: pd.DataFrame, key: str = 'stop_id') -> pd.DataFrame:
    """Set difference a minus b on a key column."""
    if key not in a.columns or key not in b.columns:
        raise ValueError(f"diff_by requires key '{key}' in both DataFrames")
    return a.loc[~a[key].isin(b[key])].copy()

def titleize(name: str) -> str:
    """Pretty label from a snake key."""
    repl = {
        'guat': 'GUAT', 'taft': 'TAFT',
        'no_capture': 'No-capture',
        'nonfinal': 'Non-final', 'middle': 'Middle',
        'giveup': 'Give-up', 'captures': 'Captures',
        'single_miss': 'Single-miss', 'all_misses': 'All misses',
        'first_misses': 'First-misses', 'both_first': 'Both first'
    }
    parts = name.split('_')
    pretty = []
    for p in parts:
        pretty.append(repl.get(p, p.capitalize()))
    return ' '.join(pretty)


    
ACRONYMS = {'GUAT', 'TAFT'}

def _cap_first_segment(name: str) -> str:
    parts = name.split('_')
    if not parts: 
        return name
    parts[0] = parts[0][:1].upper() + parts[0][1:]
    return '_'.join(parts)

def _pascalize(name: str, acronyms: set[str] = ACRONYMS) -> str:
    out = []
    for w in name.split('_'):
        if w.upper() in acronyms:
            out.append(w.upper())
        elif w:
            out.append(w[:1].upper() + w[1:])
        else:
            out.append(w)
    return ''.join(out)

def _pretty_word(w: str) -> str:
    # lightweight prettifier for common tokens
    mapping = {
        'no': 'No', 'capture': 'capture', 'no_capture': 'No-capture',
        'giveup': 'Give-up', 'nonfinal': 'nonfinal', 'middle': 'middle',
        'single': 'single', 'miss': 'miss', 'first': 'first', 'misses': 'misses',
        'captures': 'Captures', 'all': 'All'
    }
    if w.upper() in ACRONYMS:
        return w.upper()
    return mapping.get(w, w.replace('_', ' '))

def _titleize_side(name: str) -> str:
    # turn 'giveup_GUAT_last' -> 'Give-up GUAT last'
    parts = name.split('_')
    pretty = [_pretty_word(parts[0])]
    for p in parts[1:]:
        pretty.append(_pretty_word(p))
    return ' '.join(pretty)

def build_comparisons(specs: list[dict]) -> list[dict]:
    """Each spec needs {'a': <dataset_key>, 'b': <dataset_key>} and optional 'key'."""
    out = []
    for s in specs:
        a, b = s['a'], s['b']
        base_key = s.get('key', f'{a}_vs_{b}')
        item = {
            'key': base_key,                # machine-friendly
            'a': a, 'b': b,
            # auto labels (override if you provided a_label/b_label)
            'a_label': s.get('a_label', _titleize_side(a)),
            'b_label': s.get('b_label', _titleize_side(b)),
            # human-/file-friendly alternates:
            'slug_capfirst': f'{_cap_first_segment(a)}_vs_{_cap_first_segment(b)}',
            'slug_pascal': f'{_pascalize(a)}_vs_{_pascalize(b)}',
            'title': f'{_titleize_side(a)} vs {_titleize_side(b)}',
        }
        out.append(item)
    return out


def shared_cols(*dfs: pd.DataFrame,
                require: tuple[str, ...] = (),
                prefer_order: tuple[str, ...] = ('stop_id', 'stop_time', 'stop_point_index')) -> list[str]:
    """Intersection of columns across dfs, with key columns first."""
    if not dfs:
        return []
    inter = set(dfs[0].columns)
    for d in dfs[1:]:
        inter &= set(d.columns)
    inter |= set(require)  # ensure required are kept if present
    cols = list(inter)

    # stable order: prefer_order first (if present), then the rest sorted
    preferred = [c for c in prefer_order if c in cols]
    rest = sorted(c for c in cols if c not in preferred)
    return preferred + rest

# ---------- comparisons ----------
# You can omit labels and let them auto-fill from keys. Keep explicit if you want custom phrasing.
def with_labels(specs: list[dict]) -> list[dict]:
    out = []
    for s in specs:
        a_lab = s.get('a_label') or titleize(s['a'])
        b_lab = s.get('b_label') or titleize(s['b'])
        key = s.get('key') or f"{s['a']}_vs_{s['b']}"
        out.append({'key': key, 'a': s['a'], 'b': s['b'], 'a_label': a_lab, 'b_label': b_lab})
    return out


# ---------- (optional) quick validation ----------
def validate(datasets: dict[str, pd.DataFrame], comparisons: list[dict]) -> None:
    missing = []
    empties = []
    for c in comparisons:
        for side in ('a', 'b'):
            name = c[side]
            if name not in datasets:
                missing.append(name)
            elif datasets[name].empty:
                empties.append((c['key'], name))
    if missing:
        raise KeyError(f'missing dataset keys: {sorted(set(missing))}')
    if empties:
        print('warning: empty datasets in some comparisons:', empties)


# ---------- main runner (dedupe-within + overlap warning + optional matching) ----------
def run_all_comparisons(
    comparisons: list[dict],
    datasets: dict[str, pd.DataFrame],
    spikes_df: pd.DataFrame,
    monkey_information: pd.DataFrame,
    config,
    windows_summary: bool = True,
    # dedupe knobs
    dedupe_within_sets: bool = False,
    dedupe_keys: list[str] | None = None,
    time_round: int = 3,
    warn_on_overlap: bool = True,
    # matching knobs
    match_features: list[str] | None = None,     # e.g., ['stop_duration','dist_to_target','cluster_order']
    match_strategy: str = 'hungarian',
    match_caliper: float | None = None,
    align_by_stop_end=False,
) -> dict:
    results = {
        'analyzers': {},
        'psth_long': [],
        'epoch_summaries': [],
        'dedupe_logs': {},
        'overlap_logs': {},
        'matching_logs': {}
    }

    for comp in comparisons:
        key = comp['key']
        a_name, b_name = comp['a'], comp['b']
        a_label, b_label = comp['a_label'], comp['b_label']
        title = comp['title']

        A1, B1 = datasets[a_name], datasets[b_name]

        # across-set overlap: report only, do NOT remove
        overlap_info = _report_overlap(A1, B1, dedupe_keys, time_round)
        results['overlap_logs'][key] = overlap_info
        if warn_on_overlap and overlap_info['overlap_pairs'] > 0:
            print(f'warning [{key}]: {overlap_info["overlap_pairs"]} overlapping stops between {a_name} and {b_name}. '
                  f'leaving them in place. examples: {overlap_info["example_keys"]}')

        # optional per-stop matching (applied to already-deduped A1/B1)
        if match_features:
            A2, B2, results = match_features_func(results, key, A1, B1, match_features, match_strategy, match_caliper)
        else:
            A2, B2 = A1, B1
            results['matching_logs'][key] = {'strategy': 'none', 'n_matched': None}


        if align_by_stop_end:
            alignment = 'Align by stop End'
            print(f'aligning {a_name} and {b_name} by stop end')
            A2['stop_time'] = A2['stop_id_end_time']
            B2['stop_time'] = B2['stop_id_end_time']
        else:
            alignment = 'Align by stop Start'
            print(f'aligning {a_name} and {b_name} by stop start')
            A2['stop_time'] = A2['stop_id_start_time']
            B2['stop_time'] = B2['stop_id_start_time']

        # build analyzer
        an = core_stops_psth.PSTHAnalyzer(
            spikes_df=spikes_df,
            monkey_information=monkey_information,
            config=config,
            event_a_df=A2,
            event_b_df=B2,
            event_a_label=a_label,
            event_b_label=b_label,
        )
        
        plot_title = f'{title} - {alignment}'
        show_results(an, title=plot_title)

        # export
        psth_df = psth_postprocessing.export_psth_to_df(an)
        psth_df['comparison'] = key
        results['psth_long'].append(psth_df)

        if windows_summary:
            epoch_df = psth_postprocessing.summarize_epochs(an)
            epoch_df['comparison'] = key
            results['epoch_summaries'].append(epoch_df)

        results['analyzers'][key] = an

    # concat
    if results['psth_long']:
        results['psth_long'] = pd.concat(results['psth_long'], ignore_index=True)
    else:
        results['psth_long'] = pd.DataFrame(columns=['time', 'cluster', 'condition', 'mean', 'sem', 'lower', 'upper', 'comparison'])

    if results['epoch_summaries']:
        results['epoch_summaries'] = pd.concat(results['epoch_summaries'], ignore_index=True)
    else:
        results['epoch_summaries'] = pd.DataFrame()
        

    return results


def show_results(an, title=None):

    # fig2 = an.plot_comparison(cluster_idx=0)
    # plt.show()

    # windows = {
    #     "pre_bump(-0.3–0.0)": (-0.3, 0.0),
    #     "early_dip(0.0–0.3)": (0.0, 0.3),
    #     "late_rebound(0.3–0.8)": (0.3, 0.8),
    # }
    
    windows = {
        "pre_baseline(-0.5–-0.2)": (-0.5, -0.2),
        "pre_bump(-0.2–-0.1)": (-0.2, -0.1),
        "pre_bump2(-0.1–0.0)": (-0.1, 0.0),
        # "pre_bump3(-0.2--0.0)": (-0.2, -0.0),
        "early_dip(0.0–0.1)": (0.0, 0.1),
        "early_rebound(0.1–0.2)": (0.1, 0.2),
        "early_rebound2(0.2–0.3)": (0.2, 0.3),
        "early_rebound3(0.3–0.4)": (0.3, 0.4),
        "early_rebound4(0.4–0.5)": (0.4, 0.5),
        "late_rebound(0.5–0.8)": (0.5, 0.8),
        "post_tail(0.8–1.2)": (0.8, 1.2),
    }

    window_order = list(windows.keys())
        
    summary = psth_postprocessing.compare_windows(an, windows, alpha=0.05)
    psth_postprocessing.plot_sig_heatmap(summary, title=title, window_order=window_order)
    
    
    # summary_sub = summary.loc[summary['sig_FDR'], ['cluster', 'window', 'p', 'cohens_d', 'sig_FDR']].copy()
    # summary_sub = summary_sub.sort_values(["sig_FDR", "window", "p"], ascending=[False, True, True])
    # print(summary_sub)
    
    
    