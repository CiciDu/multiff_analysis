# GAM Plotting Updates Summary

## Overview
Extended the GAM analysis plotting functionality to include proper x-axis labels based on actual covariate/time values, and added plotting functions for spike history and temporal event kernels.

---

## Changes Made

### 1. **`one_ff_glm_design.py`** - Store bin edges for tuning curves

**Changes:**
- Added `bin_edges = {}` dictionary to store bin edges
- Store edges for each linear variable: `bin_edges[var] = edges` (line 75)
- Store edges for each angular variable: `bin_edges[var] = edges` (line 106)
- Added `'bin_edges': bin_edges` to returned metadata (line 129)

**Result:** Tuning curve bin edges are now saved in `tuning_meta['bin_edges']`

---

### 2. **`assemble_one_ff_gam_design.py`** - Store temporal basis lags

**Changes:**

#### `build_temporal_design_base()`:
- Added `basis_info = {}` dictionary to store both lags and basis functions
- Store info for all event kernels:
  - `basis_info['t_move'] = {'lags': lags_move, 'basis': B_move}`
  - `basis_info['t_targ'] = {'lags': lags_targ, 'basis': B_targ}`
  - `basis_info['t_rew'] = {'lags': lags_rew, 'basis': B_rew}`
  - `basis_info['t_stop'] = {'lags': lags_stop, 'basis': B_stop}`
- Added `temporal_meta['basis_info'] = basis_info`

#### `build_design_df()`:
- Added `basis_info = {}` dictionary
- Store info for spike history: `basis_info['spike_hist'] = {'lags': lags_hist, 'basis': B_hist}`
- Store info for coupling filters: `basis_info[f'cpl_{j}'] = {'lags': lags_coup, 'basis': B_coup}`
- Added `hist_meta['basis_info'] = basis_info`

#### `process_unit_design_and_groups()`:
- Changed return value from `hist_meta` to `all_meta`
- `all_meta` combines all metadata: `{'tuning': tuning_meta, 'temporal': temporal_meta, 'hist': hist_meta}`

#### `finalize_one_ff_pgam_design()`:
- Returns `all_meta` instead of just `tuning_meta`

**Result:** All temporal basis lags are now saved and accessible through `all_meta`

---

### 3. **`plot_gam_fit.py`** - New and updated plotting functions

**New Functions:**

#### `plot_spike_history(beta, hist_meta)`
- Plots spike history filter with proper time lags (in milliseconds)
- **Reconstructs the full kernel** by multiplying basis functions with weights: `kernel = basis @ weights`
- Shows reconstructed kernel vs time lag after spike
- Includes zero reference line and grid

#### `plot_event_kernel(var, beta, temporal_meta)`
- Plots temporal event kernels (t_targ, t_move, t_rew, t_stop)
- **Reconstructs the full kernel** by multiplying basis functions with weights: `kernel = basis @ weights`
- Shows reconstructed kernel vs time relative to event (in milliseconds)
- Includes zero reference line, event time marker (red line), and grid

#### `plot_variable(var, beta, all_meta)`
- Automatically determines variable type and calls appropriate plotting function
- Handles:
  - Linear tuning variables
  - Angular tuning variables
  - Event kernels
  - Spike history
  - Coupling filters

#### `plot_all_tuning_curves(beta, all_meta)`
- Plots all linear and angular tuning curves in sequence

#### `plot_all_temporal_filters(beta, all_meta)`
- Plots all event kernels and spike history in sequence

**Updated Functions:**

#### `plot_linear_tuning(var, beta, tuning_meta)`
- Now uses bin centers from `bin_edges` for x-axis
- X-axis shows actual covariate values (e.g., velocity in cm/s)
- Backward compatible with old results (falls back to bin indices with warning)

#### `plot_angular_tuning(var, beta, tuning_meta)`
- Now uses bin centers from `bin_edges` for x-axis
- X-axis shows actual angles in radians
- Backward compatible with old results (falls back to evenly-spaced angles with warning)

---

### 4. **Job Scripts** - Updated to use new metadata structure

**Files updated:**
- `one_ff_back_elim_script.py`
- `one_ff_my_gam_script.py`
- `one_ff_pen_tune_script.py`

**Changes:**
- Changed from `tuning_meta` to `all_meta`
- Updated `save_metadata` parameter: `{'all_meta': all_meta}` instead of `{'tuning_meta': tuning_meta}`
- Updated penalty tuning to access: `all_meta['tuning']['groups']`

---

## Usage Examples

### Example 1: Plot a single variable (automatic type detection)
```python
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    assemble_one_ff_gam_design,
    plot_gam_fit
)
import pickle

# Build design
design_df, y, groups, all_meta = assemble_one_ff_gam_design.finalize_one_ff_pgam_design(
    unit_idx=0, session_num=0
)

# Load fitted coefficients
with open('path/to/fit_result.pkl', 'rb') as f:
    result = pickle.load(f)
    beta = result['beta']

# Plot any variable automatically
plot_gam_fit.plot_variable('v', beta, all_meta)           # Linear tuning
plot_gam_fit.plot_variable('phi', beta, all_meta)         # Angular tuning
plot_gam_fit.plot_variable('t_move', beta, all_meta)      # Event kernel
plot_gam_fit.plot_variable('spike_hist', beta, all_meta)  # Spike history
```

### Example 2: Plot all tuning curves
```python
plot_gam_fit.plot_all_tuning_curves(beta, all_meta)
```

### Example 3: Plot all temporal filters
```python
plot_gam_fit.plot_all_temporal_filters(beta, all_meta)
```

### Example 4: Plot specific types using individual functions
```python
# Tuning curves
plot_gam_fit.plot_linear_tuning('v', beta, all_meta['tuning'])
plot_gam_fit.plot_angular_tuning('phi', beta, all_meta['tuning'])

# Temporal filters
plot_gam_fit.plot_event_kernel('t_move', beta, all_meta['temporal'])
plot_gam_fit.plot_spike_history(beta, all_meta['hist'])
```

---

## Metadata Structure

### `all_meta` dictionary structure:
```python
{
    'tuning': {
        'linear_vars': ['v', 'w', 'd', 'r_targ', 'eye_ver', 'eye_hor'],
        'angular_vars': ['phi', 'theta_targ'],
        'n_bins': 10,
        'centered': True,
        'groups': {'v': ['v:bin0', 'v:bin1', ...], ...},
        'bin_edges': {'v': array([...]), 'phi': array([...]), ...}
    },
    'temporal': {
        'groups': {'t_targ': [...], 't_move': [...], 't_rew': [...], 't_stop': [...]},
        'basis_info': {
            't_targ': {'lags': array([...]), 'basis': array([[...]])},
            't_move': {'lags': array([...]), 'basis': array([[...]])},
            ...
        }
    },
    'hist': {
        'groups': {'spike_hist': [...], 'cpl_0': [...], ...},
        'basis_info': {
            'spike_hist': {'lags': array([...]), 'basis': array([[...]])},
            'cpl_0': {'lags': array([...]), 'basis': array([[...]])},
            ...
        }
    }
}
```

---

## Backward Compatibility

All plotting functions include backward compatibility:
- If `bin_edges` not found: falls back to bin indices with warning
- If `basis_info` not found: plots weights directly instead of reconstructed kernel, with warning
- Old saved results will still plot, but with generic x-axis labels or raw weights
- Re-run analysis with updated code to get proper labels and reconstructed kernels

---

## Key Benefits

1. **Interpretable plots**: X-axes now show actual values (velocity, angles, time) instead of bin/basis indices
2. **Comprehensive plotting**: Can now plot all variable types (tuning, events, spike history)
3. **Automatic type detection**: `plot_variable()` handles all types automatically
4. **Convenience functions**: Plot all variables of a type with one call
5. **Proper units**: 
   - Tuning curves: covariate natural units (e.g., cm/s for velocity, radians for angles)
   - Temporal filters: milliseconds for time
   - Kernels reconstructed from basis functions for smooth, interpretable curves
   - Weights shown as log-rate changes (temporal) or gain multipliers (tuning)
6. **Backward compatible**: Works with old results (with warnings)
