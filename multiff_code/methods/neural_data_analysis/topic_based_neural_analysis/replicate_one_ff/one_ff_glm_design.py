from typing import Dict, Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd
from neural_data_analysis.design_kits.design_by_segment import temporal_feats, spatial_feats

# your modules
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from scipy.interpolate import BSpline

def build_continuous_tuning_block(
    data: pd.DataFrame,
    *,
    linear_vars: Sequence[str],
    angular_vars: Sequence[str],
    n_bins: int = 10,
    center: bool = False,
    binrange_dict: Optional[Dict[str, np.ndarray]] = None,
    wrap_angular: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Continuous boxcar tuning block (degree-consistent version).

    - Keeps ALL n_bins (no dropped reference column)
    - No intercept
    - NaNs -> first bin
    - Values outside limits -> assigned to nearest bin
    - binrange_dict STRICTLY determines bin limits
    - Angular variables assumed in DEGREES
    - Optional wrapping to [-180, 180)
    """

    if binrange_dict is None:
        raise ValueError('binrange_dict must be provided for all variables')

    X_blocks = []
    colnames = []
    groups = {}
    bin_edges = {}

    # -------------------------------------------------
    # Helper: deterministic boxcar assignment
    # -------------------------------------------------
    def make_boxcar_design(x, limits):
        x = x.copy()

        edges = np.linspace(limits[0], limits[1], n_bins + 1)

        inds = np.digitize(x, edges) - 1
        inds = np.clip(inds, 0, n_bins - 1)

        X = np.zeros((len(x), n_bins), dtype=float)
        valid_mask = ~np.isnan(x)
        X[np.arange(len(x))[valid_mask], inds[valid_mask]] = 1.0

        return X, edges

    # -------------------------------------------------
    # Linear variables
    # -------------------------------------------------
    for var in linear_vars:
        if var not in data.columns:
            raise KeyError(f'missing column {var!r}')
        if var not in binrange_dict:
            raise KeyError(f'binrange_dict missing entry for {var!r}')

        x_raw = data[var].to_numpy()
        nan_mask = np.isnan(x_raw)

        br = binrange_dict[var]
        limits = (float(br[0]), float(br[1]))

        X, edges = make_boxcar_design(x_raw, limits)

        if np.any(nan_mask):
            X[nan_mask, :] = 0.0
            X[nan_mask, 0] = 1.0

        bin_edges[var] = edges

        names = [f'{var}:bin{k}' for k in range(n_bins)]
        X_blocks.append(X)
        colnames.extend(names)
        groups[var] = names

    # -------------------------------------------------
    # Angular variables (DEGREES)
    # -------------------------------------------------
    for var in angular_vars:
        if var not in data.columns:
            raise KeyError(f'missing column {var!r}')
        if var not in binrange_dict:
            raise KeyError(f'binrange_dict missing entry for {var!r}')

        th_raw = data[var].to_numpy()
        nan_mask = np.isnan(th_raw)

        # Wrap to [-180, 180)
        if wrap_angular:
            th = (th_raw + 180.0) % 360.0 - 180.0
        else:
            th = th_raw

        br = binrange_dict[var]
        limits = (float(br[0]), float(br[1]))

        X, edges = make_boxcar_design(th, limits)

        if np.any(nan_mask):
            X[nan_mask, :] = 0.0
            X[nan_mask, 0] = 1.0

        bin_edges[var] = edges

        names = [f'{var}:bin{k}' for k in range(n_bins)]
        X_blocks.append(X)
        colnames.extend(names)
        groups[var] = names

    # -------------------------------------------------
    # Assemble
    # -------------------------------------------------
    X_all = np.column_stack(X_blocks) if X_blocks else np.empty((len(data), 0))

    X_df = pd.DataFrame(
        X_all,
        columns=colnames,
        index=data.index,
    )

    meta = {
        'linear_vars': list(linear_vars),
        'angular_vars': list(angular_vars),
        'n_bins': int(n_bins),
        'centered': False,
        'groups': groups,
        'bin_edges': bin_edges,
        'wrap_angular': wrap_angular,
        'angular_units': 'degrees',
    }

    return X_df, meta