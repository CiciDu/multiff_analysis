#!/usr/bin/env python3

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    assemble_one_ff_gam_design,
    backward_elimination,
    one_ff_gam_fit
)
import pandas as pd
import numpy as np
import matplotlib
from pathlib import Path
import sys
import os
print("[PYTHON][DEBUG] Script started, beginning imports...", flush=True)


print("[PYTHON][DEBUG] Basic imports done, importing matplotlib/numpy/pandas...", flush=True)

print("[PYTHON][DEBUG] Core scientific packages imported", flush=True)


# ---------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------
print("[PYTHON][DEBUG] Locating project root...", flush=True)
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        print(f"[PYTHON][DEBUG] Project root found: {p}", flush=True)
        break
else:
    raise RuntimeError('Could not find Multifirefly-Project root')

# ---------------------------------------------------------------------
# Project-specific imports
# ---------------------------------------------------------------------
print("[PYTHON][DEBUG] Importing project-specific modules...", flush=True)

print("[PYTHON][DEBUG] Project-specific imports complete", flush=True)

# ---------------------------------------------------------------------
# Global config (kept minimal)
# ---------------------------------------------------------------------
print("[PYTHON][DEBUG] Setting up global configuration...", flush=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 200)

np.set_printoptions(suppress=True)
print("[PYTHON][DEBUG] Global configuration complete, all imports finished!", flush=True)

# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def main(unit_idx: int):
    print(f"[PYTHON][DEBUG] main() called for unit_idx={unit_idx}", flush=True)

    print(
        f"[PYTHON][DEBUG] Finalizing PGAM design for unit {unit_idx}...", flush=True)
    design_df, y, groups, all_meta = assemble_one_ff_gam_design.finalize_one_ff_gam_design(
        unit_idx=unit_idx,
        session_num=0,
    )
    print(
        f"[PYTHON][DEBUG] Design finalized, shape: {design_df.shape}", flush=True)

    # Setup output directory and paths
    outdir = Path(
        f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}')
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[PYTHON][DEBUG] Output directory: {outdir}", flush=True)

    # Generate descriptive filename with lambda configuration
    lam_suffix = one_ff_gam_fit.generate_lambda_suffix(groups)
    save_path = outdir / 'backward_elimination' / f'{lam_suffix}.pkl'
    print(f"[PYTHON][DEBUG] Save path: {save_path}", flush=True)

    print(
        f"[PYTHON][INFO] Starting backward elimination for unit {unit_idx}...", flush=True)
    kept, history = backward_elimination.backward_elimination_gam(
        design_df=design_df,
        y=y,
        groups=groups,
        alpha=0.05,
        n_folds=10,
        verbose=True,
        save_path=str(save_path),
        save_metadata={'all_meta': all_meta},
    )

    print('\nFinal retained variables:')
    for g in kept:
        print(' ', g.name)

    # Export history to CSV for easy viewing
    if history:
        pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
        print(f'\n✓ History exported to {outdir / "history.csv"}')

    print(
        f"[PYTHON][INFO] Unit {unit_idx} completed successfully!", flush=True)


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    print("[PYTHON][DEBUG] Entering __main__ block", flush=True)
    import argparse

    parser = argparse.ArgumentParser(description='Run one-FF GAM analysis')
    parser.add_argument('--unit_idx', type=int, required=True)

    print("[PYTHON][DEBUG] Parsing arguments...", flush=True)
    args = parser.parse_args()
    print(
        f"[PYTHON][DEBUG] Arguments parsed: unit_idx={args.unit_idx}", flush=True)

    main(args.unit_idx)
