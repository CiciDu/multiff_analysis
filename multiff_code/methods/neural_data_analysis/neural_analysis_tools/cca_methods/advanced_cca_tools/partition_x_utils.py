import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]

# ============================================================
# Helpers for flexible block specification
# ============================================================

def _normalize_index_list(
    indices: Iterable[int],
    n_cols: int,
    *,
    name: str = "indices"
) -> List[int]:
    """
    Ensure user-provided column indices are valid, unique, and sorted.
    """
    idx = sorted(set(int(i) for i in indices))
    if len(idx) == 0:
        raise ValueError(f"{name} is empty.")
    bad = [i for i in idx if i < 0 or i >= n_cols]
    if bad:
        raise IndexError(f"{name} out of bounds (n_cols={n_cols}): {bad}")
    return idx


def _find_cols_by_regex(names: Sequence[str], pattern: str) -> List[int]:
    """
    Return the positional indices of column names matching a regex pattern.
    Useful when block columns share a prefix like 'speed_lag'.
    """
    rx = re.compile(pattern)
    found = [i for i, c in enumerate(names) if rx.search(str(c))]
    return found


def _find_contiguous_block(start: int, length: int, n_cols: int) -> List[int]:
    """
    Build a contiguous slice [start, start+length) and validate.
    Handy when each feature's lags are stored next to each other.
    """
    if start < 0 or start + length > n_cols:
        raise IndexError(f"Contiguous block [{start}:{start+length}) exceeds matrix bounds (n_cols={n_cols}).")
    return list(range(start, start + length))


def _find_interleaved_block(
    base_col_index: int,
    n_lags: int,
    stride: int,
    n_cols: int
) -> List[int]:
    """
    Build an interleaved block indices list for layouts like:
        [feat0_lag0, feat1_lag0, ..., featM_lag0, feat0_lag1, feat1_lag1, ...]
    Here:
      - base_col_index is the index for this feature at lag 0,
      - stride is the number of features per lag (M+1),
      - n_lags is how many lag stacks exist.
    The generated indices are:
        base + k*stride  for k=0..n_lags-1
    """
    idx = [base_col_index + k * stride for k in range(n_lags)]
    bad = [i for i in idx if i < 0 or i >= n_cols]
    if bad:
        raise IndexError(f"Interleaved block indices exceed bounds: {bad}")
    return idx


# ============================================================
# Main partition function
# ============================================================

def partition_X_base_block(
    X: ArrayLike,
    *,
    # Exactly one of the following ways should be provided to define the BLOCK:
    block_indices: Optional[Iterable[int]] = None,        # explicit positional indices
    block_name_regex: Optional[str] = None,               # regex over column names (DF only)
    block_contiguous: Optional[Tuple[int, int]] = None,   # (start, length)
    block_interleaved: Optional[Tuple[int, int, int]] = None,  # (base_col_index, n_lags, stride)
) -> Dict[str, Union[np.ndarray, List[int], np.ndarray, Dict[int, int]]]:
    """
    Partition X into BASE (all columns except the selected block) and BLOCK (the selected columns).

    Exactly ONE block-spec method should be used:
      - block_indices:      e.g., [120, 121, ..., 130]
      - block_name_regex:   e.g., r'^speed(_lag\d+)?$'  (requires DataFrame with column names)
      - block_contiguous:   (start, length), when lags are stored contiguously
      - block_interleaved:  (base_col_index, n_lags, stride), for interleaved-by-lag layouts

    Returns a dict with:
      - 'X_base'          : (n, p_base) NumPy array of base columns
      - 'X_block'         : (n, p_block) NumPy array of block columns
      - 'base_indices'    : list of positional indices used for BASE
      - 'block_indices'   : list of positional indices used for BLOCK
      - 'base_mask'       : boolean array of length p (True for BASE cols)
      - 'block_mask'      : boolean array of length p (True for BLOCK cols)
      - 'block_posmap'    : mapping from original col index -> local index within BLOCK
                            (useful for traceability and plotting)

    Safety:
      - Validates that BASE and BLOCK are a disjoint partition of columns.
      - Preserves original column order within each subset.
      - Works for NumPy arrays and pandas DataFrames (labels ignored in output arrays).
    """
    # ---- Inspect input and extract shapes/names ----
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        X_np = X.to_numpy()
    else:
        names = [str(i) for i in range(X.shape[1])]
        X_np = np.asarray(X)

    n, p = X_np.shape

    # ---- Decide which block specification is used (mutually exclusive) ----
    specs = [block_indices is not None, block_name_regex is not None,
             block_contiguous is not None, block_interleaved is not None]
    if sum(specs) != 1:
        raise ValueError("Provide exactly ONE way to specify the block: "
                         "block_indices OR block_name_regex OR block_contiguous OR block_interleaved.")

    # ---- Resolve block indices ----
    if block_indices is not None:
        blk_idx = _normalize_index_list(block_indices, p, name="block_indices")

    elif block_name_regex is not None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("block_name_regex requires X to be a pandas DataFrame with column names.")
        blk_idx = _find_cols_by_regex(names, block_name_regex)
        if len(blk_idx) == 0:
            raise ValueError(f"No columns matched regex: {block_name_regex!r}")

    elif block_contiguous is not None:
        start, length = block_contiguous
        blk_idx = _find_contiguous_block(int(start), int(length), p)

    else:  # block_interleaved
        base_col_index, n_lags, stride = block_interleaved
        blk_idx = _find_interleaved_block(int(base_col_index), int(n_lags), int(stride), p)

    # ---- Build masks; ensure no leakage/overlap ----
    block_mask = np.zeros(p, dtype=bool)
    block_mask[np.array(blk_idx, dtype=int)] = True

    if not np.any(block_mask):
        raise ValueError("BLOCK mask ended up empty after resolution. Check your specification.")

    base_mask = ~block_mask
    if not np.any(base_mask):
        raise ValueError("BASE would be empty. You must leave some columns outside the block.")

    # Sanity: no overlap, union is all columns.
    if np.any(block_mask & base_mask):
        raise AssertionError("BUG: BASE and BLOCK masks overlap (should be disjoint).")
    if not np.all(block_mask | base_mask):
        raise AssertionError("BUG: BASE ∪ BLOCK does not cover all columns.")

    # ---- Extract arrays (preserve original order) ----
    X_block = X_np[:, block_mask]
    X_base  = X_np[:, base_mask]

    # ---- Build human-friendly outputs ----
    block_indices_sorted = [i for i in range(p) if block_mask[i]]
    base_indices_sorted  = [i for i in range(p) if base_mask[i]]
    block_posmap = {orig_i: local_j for local_j, orig_i in enumerate(block_indices_sorted)}

    return {
        "X_base": X_base,
        "X_block": X_block,
        "base_indices": base_indices_sorted,
        "block_indices": block_indices_sorted,
        "base_mask": base_mask,
        "block_mask": block_mask,
        "block_posmap": block_posmap,
    }
