#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    assemble_one_ff_gam_design,
    one_ff_gam_fit,
    backward_elimination
)

# ---------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
else:
    raise RuntimeError('Could not find Multifirefly-Project root')

# ---------------------------------------------------------------------
# Project-specific imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Global config (kept minimal)
# ---------------------------------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 200)

np.set_printoptions(suppress=True)

# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def main(unit_idx: int):

    design_df, y, groups, all_meta = assemble_one_ff_gam_design.finalize_one_ff_pgam_design(
        unit_idx=unit_idx,
        session_num=0,
    )

    # Setup output directory and paths
    outdir = Path(
        f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}')
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate descriptive filename with lambda configuration
    lam_suffix = one_ff_gam_fit.generate_lambda_suffix(groups)
    save_path = outdir / 'backward_elimination' / f'{lam_suffix}.pkl'

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


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run one-FF GAM analysis')
    parser.add_argument('--unit_idx', type=int, required=True)

    args = parser.parse_args()
    main(args.unit_idx)
