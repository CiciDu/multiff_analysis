#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    assemble_one_ff_gam_design,
    one_ff_gam_fit
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
    print(f'Running GAM MAP fit for unit {unit_idx}')
    # -------------------------------
    # Per-unit design
    # -------------------------------
    design_df, y, groups, all_meta = assemble_one_ff_gam_design.finalize_one_ff_pgam_design(
        unit_idx=unit_idx,
        session_num=0,
    )

    outdir = Path(
        f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}')
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'fit_results').mkdir(parents=True, exist_ok=True)

    lam_suffix = one_ff_gam_fit.generate_lambda_suffix(groups)
    save_path = outdir / 'fit_results' / f'{lam_suffix}.pkl'

    fit_res = one_ff_gam_fit.fit_poisson_gam_map(
        design_df=design_df,
        y=y,
        groups=groups,
        l1_groups=[],
        max_iter=200,
        tol=1e-6,
        verbose=True,
        save_path=str(save_path),
        save_metadata={'all_meta': all_meta},
    )

    print('success:', fit_res.success)
    print('message:', fit_res.message)
    print('n_iter:', fit_res.n_iter)
    print('final objective:', fit_res.fun)
    print('grad_norm:', fit_res.grad_norm)


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run one-FF GAM fit')
    parser.add_argument('--unit_idx', type=int, required=True)

    args = parser.parse_args()
    main(args.unit_idx)
