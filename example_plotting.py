#!/usr/bin/env python3
"""
Example script demonstrating the new plotting functionality.

This shows how to use the updated plotting functions to visualize
GAM fit results with proper x-axis labels.
"""

import pickle
from pathlib import Path
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    assemble_one_ff_gam_design,
    plot_gam_fit
)

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
unit_idx = 0
session_num = 0

# Build design to get metadata
print(f"Building design for unit {unit_idx}...")
design_df, y, groups, all_meta = assemble_one_ff_gam_design.finalize_one_ff_pgam_design(
    unit_idx=unit_idx,
    session_num=session_num,
)

print("\nMetadata structure:")
print(f"  - Tuning vars: {all_meta['tuning']['linear_vars'] + all_meta['tuning']['angular_vars']}")
print(f"  - Temporal vars: {list(all_meta['temporal']['groups'].keys())}")
print(f"  - History vars: {list(all_meta['hist']['groups'].keys())}")

# ---------------------------------------------------------------------
# Load fitted coefficients
# ---------------------------------------------------------------------
# Option 1: From a saved fit result
fit_result_path = Path(f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}/fit_results')
pkl_files = list(fit_result_path.glob('*.pkl'))

if pkl_files:
    print(f"\nLoading fit results from {pkl_files[0]}...")
    with open(pkl_files[0], 'rb') as f:
        result = pickle.load(f)
        beta = result['beta']
    
    # ---------------------------------------------------------------------
    # Example 1: Plot individual variables (automatic type detection)
    # ---------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 1: Plot individual variables")
    print("="*60)
    
    # Linear tuning variables
    plot_gam_fit.plot_variable('v', beta, all_meta)  # velocity
    plot_gam_fit.plot_variable('d', beta, all_meta)  # distance
    
    # Angular tuning variables
    plot_gam_fit.plot_variable('phi', beta, all_meta)  # heading angle
    
    # Event kernels
    plot_gam_fit.plot_variable('t_move', beta, all_meta)  # movement onset
    plot_gam_fit.plot_variable('t_targ', beta, all_meta)  # target appearance
    
    # Spike history
    plot_gam_fit.plot_variable('spike_hist', beta, all_meta)
    
    # ---------------------------------------------------------------------
    # Example 2: Plot all tuning curves at once
    # ---------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 2: Plot all tuning curves")
    print("="*60)
    plot_gam_fit.plot_all_tuning_curves(beta, all_meta)
    
    # ---------------------------------------------------------------------
    # Example 3: Plot all temporal filters at once
    # ---------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 3: Plot all temporal filters")
    print("="*60)
    plot_gam_fit.plot_all_temporal_filters(beta, all_meta)
    
    # ---------------------------------------------------------------------
    # Example 4: Use specific plotting functions
    # ---------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 4: Using specific plotting functions")
    print("="*60)
    
    # For tuning curves
    tuning_meta = all_meta['tuning']
    plot_gam_fit.plot_linear_tuning('v', beta, tuning_meta)
    plot_gam_fit.plot_angular_tuning('phi', beta, tuning_meta)
    
    # For temporal filters
    temporal_meta = all_meta['temporal']
    plot_gam_fit.plot_event_kernel('t_move', beta, temporal_meta)
    
    # For spike history
    hist_meta = all_meta['hist']
    plot_gam_fit.plot_spike_history(beta, hist_meta)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
else:
    print(f"\nNo fit results found at {fit_result_path}")
    print("Run the GAM fitting script first:")
    print(f"  python jobs/one_ff/scripts/one_ff_my_gam_script.py --unit_idx {unit_idx}")
