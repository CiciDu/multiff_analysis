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
    binrange_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Paper-faithful tuning functions for continuous covariates.

    ALL continuous variables:
        - boxcar bins
        - smoothness enforced later via GAM penalty
    Angular variables:
        - treated as circular boxcar (smoothness via 1Dcirc penalty)

    Parameters
    ----------
    data : DataFrame
        Input data with covariates
    linear_vars : Sequence[str]
        Names of linear variables
    angular_vars : Sequence[str]
        Names of angular variables (circular)
    n_bins : int
        Number of bins (default: 10)
    center : bool
        Whether to center the design matrix (default: True)
    binrange_dict : Optional[Dict[str, np.ndarray]]
        Optional dictionary mapping variable names to [min, max] ranges.
        If provided, these ranges are used for binning instead of data percentiles.

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

    # Store bin edges for each variable
    bin_edges = {}
    
    # ---------- linear variables ----------
    for var in linear_vars:
        if var not in data.columns:
            raise KeyError(f'missing column {var!r}')

        x = data[var].to_numpy()
        
        # Get binrange limits if available
        limits = None
        if binrange_dict is not None and var in binrange_dict:
            binrange = binrange_dict[var]
            limits = (float(binrange[0]), float(binrange[1]))
            
        print(f'{var} limits: {limits}')
        
        X, edges = spatial_feats.boxcar_design(x, n_bins=n_bins, limits=limits)
        bin_edges[var] = edges

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

        # Get binrange limits if available (assuming they're in degrees, convert to radians)
        limits = (-np.pi, np.pi)  # default
        if binrange_dict is not None and var in binrange_dict:
            binrange_deg = binrange_dict[var]
            limits = (np.deg2rad(float(binrange_deg[0])), np.deg2rad(float(binrange_deg[1])))
        
        X, edges = spatial_feats.boxcar_design(
            th,
            n_bins=n_bins,
            limits=limits,
        )
        bin_edges[var] = edges
        
        # print limits
        print(f'{var} limits: {limits}')

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
        'bin_edges': bin_edges,
    }
    return X_df, meta