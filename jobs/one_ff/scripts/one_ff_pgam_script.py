#!/usr/bin/env python3
"""
Run PGAM for a single unit on one-firefly session data.
"""

import argparse
import os
import sys
from pathlib import Path

print("[PYTHON][DEBUG] Script started, imports beginning...", flush=True)



for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break


from neural_data_analysis.neural_analysis_tools.pgam_tools import pgam_class
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_pipeline
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_pgam_design
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.parameters import (
    default_prs
)

# -------------------------------------------------------
# Core imports
# -------------------------------------------------------

# PGAM external library
PGAM_PATH = Path(
    'multiff_analysis/external/pgam/src'
).expanduser().resolve()

if str(PGAM_PATH) not in sys.path:
    sys.path.append(str(PGAM_PATH))

import PGAM.gam_data_handlers as gdh  # noqa: F401
from PGAM.GAM_library import *  # noqa: F401,F403
from post_processing import postprocess_results  # noqa: F401

print("[PYTHON][DEBUG] All imports completed successfully", flush=True)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main(args):
    print(f"[PYTHON][DEBUG] main() called for unit_idx={args.unit_idx}", flush=True)
    # ------------------
    # Parameters
    # ------------------
    prs = default_prs()
    pgam_save_dir = 'all_monkey_data/one_ff_data/pgam_results'

    print(f"[PYTHON][DEBUG] Loading session data from .mat file...", flush=True)
    data_obj = one_ff_pipeline.OneFFSessionData(
        mat_path='all_monkey_data/one_ff_data/sessions_python.mat',
        prs=prs,
        session_num=0,
    )
    print(f"[PYTHON][DEBUG] Session data loaded successfully", flush=True)

    covariate_names = [
        'v', 'w', 'd', 'phi',
        'r_targ', 'theta_targ',
        'eye_ver', 'eye_hor',
    ]

    print(f"[PYTHON][DEBUG] Computing covariates...", flush=True)
    data_obj.compute_covariates(covariate_names)
    print(f"[PYTHON][DEBUG] Computing spike counts...", flush=True)
    data_obj.compute_spike_counts()
    print(f"[PYTHON][DEBUG] Smoothing spikes...", flush=True)
    data_obj.smooth_spikes()
    print(f"[PYTHON][DEBUG] Computing events...", flush=True)
    data_obj.compute_events()
    print(f"[PYTHON][DEBUG] Getting binned spikes dataframe...", flush=True)
    binned_spikes_df = data_obj.get_binned_spikes_df()
    print(f"[PYTHON][DEBUG] Data preparation complete", flush=True)

    # first just try to load the data
    print(f"[PYTHON][DEBUG] Initializing PGAM runner...", flush=True)
    pgam_runner = pgam_class.PGAMclass(
        x_var=binned_spikes_df,
        bin_width=prs.dt,
        save_dir=pgam_save_dir,
    )
    print(f"[PYTHON][DEBUG] PGAM runner initialized", flush=True)
    
    print(f"[PYTHON][DEBUG] Checking for existing results for unit {args.unit_idx}...", flush=True)
    try:
        pgam_runner.load_pgam_results(args.unit_idx)
        print(f"[PYTHON][INFO] Loaded existing PGAM results for unit {args.unit_idx}")
        return
    except FileNotFoundError:
        print(f"[PYTHON][INFO] No existing results found, running PGAM for unit {args.unit_idx}")

    # ------------------
    # Build PGAM design
    # ------------------
    print(f"[PYTHON][DEBUG] Building smooth handler for unit {args.unit_idx}...", flush=True)
    sm_handler = one_ff_pgam_design.build_smooth_handler(
        data_obj=data_obj,
        unit_idx=args.unit_idx,
        covariate_names=covariate_names,
        tuning_covariates=covariate_names,
        use_cyclic=set(),
        order=4,
    )
    print(f"[PYTHON][DEBUG] Smooth handler built successfully", flush=True)

    # ------------------
    # PGAM data_obj
    # ------------------
    print(f"[PYTHON][DEBUG] Creating PGAM runner instance...", flush=True)
    pgam_runner = pgam_class.PGAMclass(
        x_var=binned_spikes_df,
        bin_width=data_obj.prs.dt,
        save_dir=pgam_save_dir,
    )

    pgam_runner.sm_handler = sm_handler
    pgam_runner.trial_ids = data_obj.covariate_trial_ids
    pgam_runner.train_trials = pgam_runner.trial_ids % 3 != 1
    print(f"[PYTHON][DEBUG] PGAM runner configured", flush=True)

    # ------------------
    # Run PGAM
    # ------------------
    print(f"[PYTHON][DEBUG] Running PGAM optimization for unit {args.unit_idx}...", flush=True)
    pgam_runner.run_pgam(neural_cluster_number=args.unit_idx)
    print(f"[PYTHON][DEBUG] PGAM optimization complete", flush=True)
    
    print(f"[PYTHON][DEBUG] Post-processing results...", flush=True)
    pgam_runner.kernel_h_length = 100
    pgam_runner.post_processing_results(neural_cluster_number=args.unit_idx)
    print(f"[PYTHON][DEBUG] Saving results...", flush=True)
    pgam_runner.save_results()
    print(f"[PYTHON][INFO] Unit {args.unit_idx} completed successfully!", flush=True)

    all_mean_r2 = []
    num_neurons = pgam_runner.x_var.shape[1]
    for n in range(num_neurons):
        out = pgam_runner.run_pgam_cv(n, n_splits=5, filtwidth=2)
        all_mean_r2.append(out['mean_r2_eval'])
        
    
# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == '__main__':
    print("[PYTHON][DEBUG] Entering __main__ block", flush=True)
    parser = argparse.ArgumentParser(
        description='Run PGAM for a single unit (one-firefly task)'
    )
    parser.add_argument(
        '--unit_idx',
        type=int,
        default=0,
        help='Unit index to run PGAM on',
    )

    print("[PYTHON][DEBUG] Parsing arguments...", flush=True)
    args = parser.parse_args()
    print(f"[PYTHON][DEBUG] Arguments parsed: unit_idx={args.unit_idx}", flush=True)
    main(args)
