#!/usr/bin/env python3

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



from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_pipeline,
)

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
    runner = one_ff_gam_pipeline.OneFFGAMRunner(session_num=0)
    result = runner.run_penalty_tuning(unit_idx=unit_idx, n_folds=5)
    best_lams = result['best_lams']

    if best_lams is not None:
        print('Best lambdas:')
        for k, v in best_lams.items():
            print(f'  {k}: {v}')
        print(f'Saved penalty tuning results to {result["save_path"]}')
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
