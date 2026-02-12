#!/usr/bin/env python3

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    assemble_one_ff_gam_design,
    penalty_tuning
)
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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
# Global config
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
    print(f'Running GAM penalty tuning for unit {unit_idx}')
    # -------------------------------
    # Per-unit design
    # -------------------------------
    design_df, y, groups, all_meta = assemble_one_ff_gam_design.finalize_one_ff_gam_design(
        unit_idx=unit_idx,
        session_num=0,
    )

    # -------------------------------
    # Penalty tuning
    # -------------------------------
    l1_groups = []  # coupling Laplace prior can go here later

    lam_grid = {
        'lam_f': [10, 50, 100, 300],
        'lam_g': [1, 5, 10, 30],
        'lam_h': [1, 5, 10],
    }

    group_name_map = {
        'lam_f': list(all_meta['tuning']['groups'].keys()),
        'lam_g': ['t_targ', 't_move', 't_rew'],
        'lam_h': ['spike_hist'],
    }

    outdir = Path(
        f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}')
    outdir.mkdir(parents=True, exist_ok=True)

    best_lams, cv_results = penalty_tuning.tune_penalties(
        design_df=design_df,
        y=y,
        base_groups=groups,
        l1_groups=l1_groups,
        lam_grid=lam_grid,
        group_name_map=group_name_map,
        n_folds=5,
        save_path=outdir / 'penalty_tuning.pkl',
        save_metadata={'all_meta': all_meta},
    )

    if best_lams is not None:
        print('Best lambdas:')
        for k, v in best_lams.items():
            print(f'  {k}: {v}')
        print(
            f'Saved penalty tuning results to {outdir / "penalty_tuning.pkl"}')
    else:
        print('ERROR: Penalty tuning failed - no valid model fits found.')
        print('Check the output above for details.')
        import sys
        sys.exit(1)


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run one-FF GAM penalty tuning')
    parser.add_argument('--unit_idx', type=int, required=True)

    args = parser.parse_args()
    main(args.unit_idx)
