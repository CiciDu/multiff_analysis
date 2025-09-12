from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, Tuple, List, Mapping

import warnings
from neural_data_analysis.design_kits.design_by_segment import other_feats
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import BSpline

import numpy as np
import pandas as pd
from scipy.linalg import qr


# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.design_kits.design_by_segment import temporal_feats


def check_design_vs_bases(
    design_df: pd.DataFrame,
    meta: dict,
    *,
    strict: bool = True,
) -> dict:
    """
    Verify that design_df columns match the bases in meta['bases_by_predictor'].
    Treat groups with *no bases* (pure passthrough columns) as valid.

    Returns:
      - ok: bool
      - problems: list[str]
      - table: DataFrame(prefix, L, expected_K, found_K)  # basis-backed prefixes only
    """
    if 'bases_by_predictor' not in meta:
        raise KeyError("meta['bases_by_predictor'] not found.")

    names = list(design_df.columns)
    bmap: dict[str, list[np.ndarray]] = dict(
        meta.get('bases_by_predictor', {}))
    groups: dict[str, list[str]] = dict(meta.get('groups', {}))

    problems: list[str] = []
    rows = []

    # ---------- basis-backed prefixes ----------
    for prefix, Bs in bmap.items():
        if not isinstance(Bs, (list, tuple)) or len(Bs) == 0:
            # If a key exists but is empty, we tolerate it only when there are no basis-style columns.
            bjk_cols = _cols_for_prefix(prefix, names)
            if bjk_cols:
                problems.append(
                    f'{prefix}: has basis-like columns but no bases registered')
            continue

        Ls = {B.shape[0] for B in Bs}
        if len(Ls) != 1:
            problems.append(
                f'{prefix}: bases have differing lag lengths L={sorted(Ls)}')
        L = next(iter(Ls)) if Ls else None

        expected_K = int(sum(B.shape[1] for B in Bs))
        cols = _cols_for_prefix(prefix, names)  # only bjk (basis) columns
        found_K = len(cols)

        if expected_K != found_K:
            problems.append(
                f'{prefix}: expected {expected_K} columns from bases; found {found_K} in design')

        # optional: check groups listing for this prefix (only for basis-style columns)
        if prefix in groups:
            grp_cols = [c for c in groups[prefix] if ':b' in c]
            if set(grp_cols) != set(cols):
                problems.append(
                    f"{prefix}: meta['groups'] mismatch (groups={len(grp_cols)} vs actual={found_K})")

        rows.append((prefix, L, expected_K, found_K))

    # ---------- passthrough groups (no bases, single raw column) ----------
    # A passthrough group is defined as: group exists AND all its columns have no ':'.
    passthrough_prefixes = [
        g for g, cols in groups.items()
        if len(cols) > 0 and all(':' not in c for c in cols) and g not in bmap
    ]
    for prefix in passthrough_prefixes:
        raw_cols = [c for c in names if c == prefix]
        # If the passthrough column was dropped (e.g., all-zero), raw_cols may be empty; that's fine.
        # If present, ensure groups agrees with design.
        grp_cols = [c for c in groups.get(prefix, []) if ':' not in c]
        if set(raw_cols) != set(grp_cols):
            problems.append(
                f"{prefix}: passthrough group mismatch (groups={grp_cols} vs actual={raw_cols})")

    # ---------- sin/cos consistency check for basis-backed circular pairs ----------
    for stem in {p[:-4] for p in bmap.keys() if p.endswith('_sin')}:
        ps, pc = f'{stem}_sin', f'{stem}_cos'
        if ps in bmap and pc in bmap:
            Ls = {B.shape[0] for B in bmap[ps]}
            Lc = {B.shape[0] for B in bmap[pc]}
            if Ls and Lc and (max(Ls) != max(Lc)):
                problems.append(
                    f'{stem}: sin and cos lag lengths differ (sin L={sorted(Ls)}, cos L={sorted(Lc)})')

    table = pd.DataFrame(
        rows, columns=['prefix', 'L', 'expected_K', 'found_K']).sort_values('prefix')
    ok = len(problems) == 0
    if strict and not ok:
        msg = 'check_design_vs_bases failed:\n- ' + '\n- '.join(problems)
        raise AssertionError(msg)
    return {'ok': ok, 'problems': problems, 'table': table}


# =============================
# Tail trimming of unsupported lags (prevents aliasing)
# =============================

def trim_trailing_allzero_by_group(
    design_df: pd.DataFrame,
    meta: dict,
    *,
    pairs: Sequence[Tuple[str, str]] = (('cur_angle_sin', 'cur_angle_cos'),
                                        ('nxt_angle_sin', 'nxt_angle_cos')),
) -> Tuple[pd.DataFrame, dict, Dict[str, List[str]]]:
    """
    For each predictor and each basis block j, drop *trailing* columns that are all zero.
    Updates meta['groups'] and meta['bases_by_predictor'] accordingly.

    If a pair (A,B) is provided, enforce the SAME keep-mask for A and B (to preserve amplitude math).

    Returns (design_df_trimmed, meta_trimmed, dropped_cols_by_group).
    """
    names = list(design_df.columns)
    groups = {k: list(v) for k, v in meta.get('groups', {}).items()}
    bmap = {k: list(v) for k, v in meta.get('bases_by_predictor', {}).items()}
    dropped_by_group: Dict[str, List[str]] = {}

    # 1) compute per-group per-basis keep masks
    keep_masks: Dict[Tuple[str, int], np.ndarray] = {}
    for gname, cols in groups.items():
        # bucket by basis index j
        buckets: Dict[int, List[str]] = {}
        for c in cols:
            if ':b' not in c:  # skip non-bjk columns
                continue
            _, j, k = _parse_bjk(c)
            buckets.setdefault(j, []).append(c)
        # within each basis j, order by k and trim trailing all-zeros
        for j, c_list in buckets.items():
            c_list_sorted = sorted(c_list, key=lambda s: _parse_bjk(s)[2])
            Xblock = design_df[c_list_sorted].to_numpy()
            K = Xblock.shape[1]
            keep = np.ones(K, dtype=bool)
            # drop from the end while all-zero
            for kk in range(K - 1, -1, -1):
                if np.allclose(Xblock[:, kk], 0.0):
                    keep[kk] = False
                else:
                    break
            keep_masks[(gname, j)] = keep

    # 2) enforce paired masks by intersection (so A and B drop the same tail indices)
    for A, B in pairs or []:
        jsA = {j for (g, j) in keep_masks if g == A}
        jsB = {j for (g, j) in keep_masks if g == B}
        for j in sorted(jsA & jsB):
            kA = keep_masks[(A, j)]
            kB = keep_masks[(B, j)]
            kAB = kA & kB
            keep_masks[(A, j)] = kAB
            keep_masks[(B, j)] = kAB

    # 3) actually drop columns and update meta + bases
    to_drop: List[str] = []
    for gname, cols in groups.items():
        new_cols: List[str] = []
        buckets: Dict[int, List[str]] = {}
        for c in cols:
            if ':b' not in c:
                new_cols.append(c)  # passthrough columns untouched
                continue
            _, j, _ = _parse_bjk(c)
            buckets.setdefault(j, []).append(c)

        for j, c_list in buckets.items():
            c_list_sorted = sorted(c_list, key=lambda s: _parse_bjk(s)[2])
            keep = keep_masks.get((gname, j))
            if keep is None:
                new_cols.extend(c_list_sorted)
                continue
            for kk, cname in enumerate(c_list_sorted):
                if keep[kk]:
                    new_cols.append(cname)
                else:
                    to_drop.append(cname)

            # also trim the stored basis columns in meta
            if gname in bmap and j < len(bmap[gname]):
                Bj = bmap[gname][j]
                if Bj.shape[1] == keep.size:
                    bmap[gname][j] = Bj[:, keep]

        groups[gname] = new_cols

    if to_drop:
        design_df = design_df.drop(columns=to_drop)

    # record which were dropped per group
    for gname, cols in meta.get('groups', {}).items():
        before = set(cols)
        after = set(groups.get(gname, []))
        dropped_by_group[gname] = sorted(before - after)

    meta_out = dict(meta)
    meta_out['groups'] = groups
    meta_out['bases_by_predictor'] = bmap
    return design_df, meta_out, dropped_by_group


def _cols_for_prefix(prefix: str, names: Sequence[str]) -> List[str]:
    # our bjk naming: '<prefix>:b<j>:<k>'
    return [n for n in names if n.startswith(prefix + ':')]


def _parse_bjk(col: str) -> Tuple[str, int, int]:
    # '<name>:b{j}:{k}' -> (name, j, k)
    name, rest = col.split(':b', 1)
    j_str, k_str = rest.split(':', 1)
    return name, int(j_str), int(k_str)


def drop_aliased_columns(df: pd.DataFrame, tol: float = 1e-10):
    """
    Remove exactly collinear columns using QR with column pivoting.
    Returns (df_reduced, info)
    """
    X = df.to_numpy(dtype=float, copy=False)
    Q, R, piv = qr(X, mode='economic', pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        rank = 0
    else:
        rank = int((diag > tol * diag.max()).sum())
    keep_idx = np.sort(piv[:rank])
    drop_idx = np.sort(piv[rank:])
    kept = df.columns[keep_idx].tolist()
    dropped = df.columns[drop_idx].tolist()
    return df.iloc[:, keep_idx].copy(), {"rank": rank, "kept": kept, "dropped": dropped}


def force_register_passthrough(meta: dict, design_df: pd.DataFrame,
                               base_names=('RDz', 'RDy', 'LDz', 'LDy')) -> dict:
    """
    Ensure passthrough metadata is consistent for the given base_names.
    - Uses meta['groups'][<base>] if present; otherwise guesses <base>_odd/<base>_mag.
    - Keeps only columns that actually exist in design_df.
    - Populates passthrough_cols, passthrough_specs (with group tags), and passthrough_groups.
    """
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    meta = dict(meta)
    G = dict(meta.get('groups', {}))
    raw_specs = dict(meta.get('raw_specs', {}))
    pt_cols = set(meta.get('passthrough_cols', []))
    pt_specs = dict(meta.get('passthrough_specs', {}))
    pt_groups = dict(meta.get('passthrough_groups', {}))

    for b in base_names:
        # what group claims
        claimed = list(G.get(b, []))
        # keep columns that really exist
        actual = [c for c in claimed if c in design_df.columns]
        if not actual:
            # guess conventional names
            guess = [f'{b}_odd', f'{b}_mag']
            actual = [c for c in guess if c in design_df.columns]
            if actual:
                G.setdefault(b, [])
                for c in actual:
                    if c not in G[b]:
                        G[b].append(c)

        if not actual:
            # nothing to register for this base
            continue

        pt_groups.setdefault(b, [])
        for c in actual:
            pt_cols.add(c)
            spec = dict(raw_specs.get(c, {}))
            # default enc type if raw_specs missing
            if 'encoding' not in spec:
                spec['encoding'] = 'odd' if c.endswith('_odd') else (
                    'even' if c.endswith('_mag') else 'single')
            spec['group'] = b
            pt_specs[c] = spec
            if c not in pt_groups[b]:
                pt_groups[b].append(c)

        pt_groups[b] = _dedup(pt_groups[b])

    meta['groups'] = G
    meta['passthrough_cols'] = _dedup(list(pt_cols))
    meta['passthrough_specs'] = pt_specs
    meta['passthrough_groups'] = pt_groups
    return meta


def finalize_passthrough_meta(meta: dict, design_df: pd.DataFrame,
                              groups=('RDz', 'RDy', 'LDz', 'LDy')) -> dict:
    meta = dict(meta)
    G = dict(meta.get('groups', {}))
    raw_specs = dict(meta.get('raw_specs', {}))

    pt_cols = list(meta.get('passthrough_cols', []))
    pt_specs = dict(meta.get('passthrough_specs', {}))
    pt_groups = dict(meta.get('passthrough_groups', {}))

    for g in groups:
        listed = list(G.get(g, []))
        # keep only columns that exist in design_df
        actual = [c for c in listed if c in design_df.columns]
        if not actual:
            # try to infer the conventional names
            guess = [f'{g}_odd', f'{g}_mag']
            actual = [c for c in guess if c in design_df.columns]
            if actual:
                G.setdefault(g, [])
                for c in actual:
                    if c not in G[g]:
                        G[g].append(c)

        if actual:
            pt_groups.setdefault(g, [])
            for c in actual:
                if c not in pt_cols:
                    pt_cols.append(c)
                spec = dict(raw_specs.get(c, {}))
                spec.setdefault('encoding', 'odd' if c.endswith(
                    '_odd') else 'even' if c.endswith('_mag') else 'single')
                spec['group'] = g
                pt_specs[c] = spec
                if c not in pt_groups[g]:
                    pt_groups[g].append(c)

    other_feats._dedup_inplace(pt_cols)
    for k in pt_groups:
        other_feats._dedup_inplace(pt_groups[k])

    meta['groups'] = G
    meta['passthrough_cols'] = pt_cols
    meta['passthrough_specs'] = pt_specs
    meta['passthrough_groups'] = pt_groups
    return meta

# assumes _build_spatial_knots and _bspline_design_from_knots are already defined


def normalize_passthrough_meta(meta: dict, design_df: pd.DataFrame,
                               base_names=('RDz', 'RDy', 'LDz', 'LDy')) -> dict:
    """
    Mirror passthrough metadata into multiple legacy key layouts so that
    predictor_utils.check_design_vs_bases can find it regardless of which
    meta fields it expects.

    We populate:
      - meta['passthrough_cols']
      - meta['passthrough_specs'][col] with {'group': <base>, 'encoding': ...}
      - meta['passthrough_groups'][<base>] -> [cols]
      - meta['passthrough'] = {'cols','specs','groups'}  # nested legacy style
      - meta['feature_modes'][<base>] = 'passthrough'    # group mode hint
      - meta['predictor_kinds'][col] = 'passthrough'     # per-column hint
    """
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    m = dict(meta)

    G = dict(m.get('groups', {}))
    raw_specs = dict(m.get('raw_specs', {}))

    # Start from any existing passthrough info
    pt_cols = set(m.get('passthrough_cols', []))
    pt_specs = dict(m.get('passthrough_specs', {}))
    pt_groups = dict(m.get('passthrough_groups', {}))

    for base in base_names:
        # Prefer what 'groups' says; otherwise guess <base>_odd/<base>_mag
        claimed = list(G.get(base, []))
        if not claimed:
            claimed = [f'{base}_odd', f'{base}_mag']

        # Keep only columns that actually exist in the design
        actual = [c for c in claimed if c in design_df.columns]
        if not actual:
            # nothing to mirror for this base
            continue

        # Record under passthrough_groups
        pt_groups.setdefault(base, [])
        for c in actual:
            pt_cols.add(c)

            # Build a spec for the column
            spec = dict(raw_specs.get(c, {}))
            if 'encoding' not in spec:
                if c.endswith('_odd'):
                    spec['encoding'] = 'odd'
                elif c.endswith('_mag'):
                    spec['encoding'] = 'even'
                else:
                    spec['encoding'] = 'single'
            spec['group'] = base

            pt_specs[c] = spec

            if c not in pt_groups[base]:
                pt_groups[base].append(c)

        pt_groups[base] = _dedup(pt_groups[base])

    # Flattened keys
    m['passthrough_cols'] = _dedup(list(pt_cols))
    m['passthrough_specs'] = pt_specs
    m['passthrough_groups'] = pt_groups

    # Nested legacy-style alias
    m['passthrough'] = {
        'cols': m['passthrough_cols'],
        'specs': m['passthrough_specs'],
        'groups': m['passthrough_groups'],
    }

    # Mode/kind hints (some validators look for these)
    feature_modes = dict(m.get('feature_modes', {}))
    for base in pt_groups:
        feature_modes[base] = 'passthrough'
    m['feature_modes'] = feature_modes

    predictor_kinds = dict(m.get('predictor_kinds', {}))
    for c in m['passthrough_cols']:
        predictor_kinds[c] = 'passthrough'
    m['predictor_kinds'] = predictor_kinds

    return m
