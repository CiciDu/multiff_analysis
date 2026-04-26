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


# -------------------------------------------------------
# Core imports
# -------------------------------------------------------

print("[PYTHON][DEBUG] All imports completed successfully", flush=True)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main(args):
    from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
        one_ff_pgam_pipeline,
    )


    print(f"[PYTHON][DEBUG] main() called for unit_idx={args.unit_idx}", flush=True)
    print("[PYTHON][DEBUG] Initializing OneFFPGAMRunner...", flush=True)
    runner = one_ff_pgam_pipeline.OneFFPGAMRunner(session_num=0)
    print("[PYTHON][DEBUG] OneFFPGAMRunner initialized", flush=True)

    print(
        f"[PYTHON][DEBUG] Running PGAM pipeline for unit {args.unit_idx}...",
        flush=True,
    )
    result = runner.run_unit_pgam(
        unit_idx=args.unit_idx,
        n_splits=5,
        filtwidth=2,
        kernel_h_length=100,
        load_if_exists=True,
        retrieve_only=False,
    )

    if result["loaded_existing"]:
        print(
            f"[PYTHON][INFO] Loaded existing PGAM results for unit {args.unit_idx}",
            flush=True,
        )
    else:
        print(
            f"[PYTHON][INFO] No existing results found; fit and saved PGAM for unit {args.unit_idx}",
            flush=True,
        )
    print(f"[PYTHON][INFO] PGAM CV complete for unit {args.unit_idx}", flush=True)

    print(
        f"[PYTHON][DEBUG] Running PGAM category contributions for unit {args.unit_idx}...",
        flush=True,
    )
    cat_result = runner.run_category_variance_contributions(
        unit_idx=args.unit_idx,
        n_splits=5,
        filtwidth=2,
        category_names=None,
        retrieve_only=False,
        load_if_exists=True,
    )
    full_cv = cat_result["full_cv_result"]
    category_contrib = cat_result["category_contributions"]
    contrib_csv = cat_result["category_contributions_csv"]

    print("\nCategory contributions (leave-one-category-out):", flush=True)
    print(
        f"  full model mean r2 eval = {full_cv['mean_r2_eval']:.6f}",
        flush=True,
    )
    for cat_name, res in category_contrib.items():
        print(
            f"  {cat_name}: delta r2 eval = {res['delta_r2_eval']:.6f}",
            flush=True,
        )
    print(f"\nSaved category contributions to: {contrib_csv}", flush=True)


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
