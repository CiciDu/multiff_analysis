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
    center: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Paper-faithful tuning functions for continuous covariates.

    ALL continuous variables:
        - boxcar bins
        - smoothness enforced later via GAM penalty
    Angular variables:
        - treated as circular boxcar (smoothness via 1Dcirc penalty)

    Returns
    -------
    X_df : DataFrame
        Design-matrix block for continuous tunings.
    meta : dict
        Metadata describing columns per variable.
    """
    X_blocks = []
    colnames = []
    groups = {}

    # ---------- linear variables ----------
    for var in linear_vars:
        if var not in data.columns:
            raise KeyError(f'missing column {var!r}')

        x = data[var].to_numpy()
        X, edges = spatial_feats.boxcar_design(x, n_bins=n_bins)

        if center and X.size:
            X = X - X.mean(axis=0, keepdims=True)

        names = [f'{var}:bin{k}' for k in range(X.shape[1])]
        X_blocks.append(X)
        colnames.extend(names)
        groups[var] = names

    # ---------- angular variables (CIRCULAR BOXCARS) ----------
    for var in angular_vars:
        if var not in data.columns:
            raise KeyError(f'missing column {var!r}')

        th = data[var].to_numpy()

        # Wrap to [-pi, pi) for safety (or [0, 2pi), just be consistent)
        th = np.mod(th + np.pi, 2 * np.pi) - np.pi

        X, edges = spatial_feats.boxcar_design(
            th,
            n_bins=n_bins,
            limits=(-np.pi, np.pi),
        )

        if center and X.size:
            X = X - X.mean(axis=0, keepdims=True)

        names = [f'{var}:bin{k}' for k in range(X.shape[1])]
        X_blocks.append(X)
        colnames.extend(names)
        groups[var] = names

    # ---------- assemble ----------
    X_all = np.column_stack(X_blocks) if X_blocks else np.empty((len(data), 0))
    X_df = pd.DataFrame(X_all, columns=colnames, index=data.index)

    meta = {
        'linear_vars': list(linear_vars),
        'angular_vars': list(angular_vars),
        'n_bins': int(n_bins),
        'centered': bool(center),
        'groups': groups,
    }
    return X_df, meta