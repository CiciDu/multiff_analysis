# Fix: Temporal Kernel Plotting

## Issue
The temporal kernel plotting was failing with:
```
ValueError: x and y must have same first dimension, but have shapes (120,) and (10,)
```

## Root Cause
The plotting functions were trying to plot:
- **x-axis**: Full time grid (e.g., 120 time points)
- **y-axis**: Basis function weights (e.g., 10 weights)

This is incorrect because temporal kernels use basis function expansions. The 10 weights correspond to 10 basis functions, and the actual kernel needs to be **reconstructed** by multiplying the basis functions by the weights.

## Solution

### Changed metadata structure
**Before:** Only stored time lags
```python
basis_lags = {'t_move': array([...]), 'spike_hist': array([...])}
```

**After:** Store both lags AND basis functions
```python
basis_info = {
    't_move': {
        'lags': array([...]),    # Time points (e.g., 120 values)
        'basis': array([[...]])   # Basis functions matrix (120 x 10)
    },
    'spike_hist': {
        'lags': array([...]),
        'basis': array([[...]])
    }
}
```

### Updated plotting functions
All temporal plotting functions now reconstruct the kernel:

```python
# Get basis info
info = temporal_meta['basis_info'][var]
lags = info['lags']          # (120,) array
basis = info['basis']         # (120, 10) array
w = beta[cols].to_numpy()    # (10,) weights

# Reconstruct kernel
kernel = basis @ w            # (120,) reconstructed kernel

# Plot
plt.plot(lags * 1000, kernel)
```

### Mathematical explanation
- Temporal kernels are represented as: **κ(t) = Σ wᵢ φᵢ(t)**
  - κ(t) = kernel at time t
  - wᵢ = weight for basis function i
  - φᵢ(t) = basis function i at time t

- In matrix form: **κ = B @ w**
  - κ: reconstructed kernel (120 x 1)
  - B: basis functions (120 x 10)
  - w: weights (10 x 1)

## Files Modified

1. **`assemble_one_ff_gam_design.py`**
   - `build_temporal_design_base()`: Store `basis_info` instead of `basis_lags`
   - `build_design_df()`: Store `basis_info` for spike history

2. **`plot_gam_fit.py`**
   - `plot_spike_history()`: Reconstruct kernel using `basis @ weights`
   - `plot_event_kernel()`: Reconstruct kernel using `basis @ weights`
   - `plot_variable()`: Updated coupling filter plotting

## Result
- ✅ Plots now show **smooth reconstructed kernels** instead of just basis weights
- ✅ X-axis properly shows time in milliseconds
- ✅ Kernels are interpretable as actual firing rate modulations over time
- ✅ Backward compatible (falls back to plotting weights if `basis_info` not available)

## Testing
The plotting should now work correctly:

```python
# This should now work without errors
plot_gam_fit.plot_variable('t_move', beta, all_meta)      # Event kernel
plot_gam_fit.plot_variable('spike_hist', beta, all_meta)  # Spike history
```

## Note
If you have **old saved results** (before this fix), you'll need to re-run the analysis to generate new results with `basis_info` included. Old results will still plot (using the fallback), but will show basis weights instead of reconstructed kernels.
