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

# ---------------------------------------------------------------------
# Project-specific imports
# ---------------------------------------------------------------------
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_pipeline,
)
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

VAR_CATEGORIES = {
    'sensory_vars': ['v', 'w'],
    'latent_vars': ['r_targ', 'theta_targ'],
    'position_vars': ['d', 'phi'],
    'eye_position_vars': ['eye_ver', 'eye_hor'],
    'event_vars': ['t_move', 't_targ', 't_stop', 't_rew'],
    'spike_hist_vars': ['spike_hist'],
}


def main(unit_idx: int):
    print(f'Running GAM MAP fit for unit {unit_idx}')
    runner = one_ff_gam_pipeline.OneFFGAMRunner(
        session_num=0,
        var_categories=VAR_CATEGORIES,
        selected_categories=list(VAR_CATEGORIES.keys()),
    )
    result = runner.run_my_gam(unit_idx=unit_idx, n_folds=5, buffer_samples=20)
    fit_res = result['fit_result']

    print('success:', fit_res.success)
    print('message:', fit_res.message)
    print('n_iter:', fit_res.n_iter)
    print('final objective:', fit_res.fun)
    print('grad_norm:', fit_res.grad_norm)

    cv_res = result['cv_result']
    print(cv_res["mean_classical_r2"])
    print(cv_res["mean_pseudo_r2"])

    full_cv = result['full_cv_result']
    category_contrib = result['category_contributions']

    print('\nCategory contributions (leave-one-category-out):')
    print(
        f"  full model mean classical r2 = {full_cv['mean_classical_r2']:.6f}, "
        f"mean pseudo r2 = {full_cv['mean_pseudo_r2']:.6f}"
    )
    for cat_name, res in category_contrib.items():
        print(
            f"  {cat_name}: "
            f"delta classical r2 = {res['delta_classical_r2']:.6f}, "
            f"delta pseudo r2 = {res['delta_pseudo_r2']:.6f}"
        )

    contrib_csv = result['category_contributions_csv']
    print(f'\nSaved category contributions to: {contrib_csv}')


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run one-FF GAM fit')
    parser.add_argument('--unit_idx', type=int, required=True)

    args = parser.parse_args()
    main(args.unit_idx)
