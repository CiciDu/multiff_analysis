#!/usr/bin/env python3

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    parameters,
    one_ff_pipeline,
)

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit,
    assemble_one_ff_gam_design,
)

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
    print(f'Running GAM analysis for unit {unit_idx}')

    covariate_names = [
        'v', 'w', 'd', 'phi',
        'r_targ', 'theta_targ',
        'eye_ver', 'eye_hor', 'move',
    ]

    prs = parameters.default_prs()

    data_obj = one_ff_pipeline.OneFFSessionData(
        mat_path='all_monkey_data/one_ff_data/sessions_python.mat',
        prs=prs,
        session_num=0,
    )

    # -------------------------------
    # Preprocessing
    # -------------------------------
    data_obj.compute_covariates(covariate_names)
    data_obj.compute_spike_counts()
    data_obj.smooth_spikes()
    data_obj.compute_events()

    linear_vars = [
        'v', 'w', 'd', 'r_targ',
        'eye_ver', 'eye_hor',
    ]

    angular_vars = [
        'phi', 'theta_targ',
    ]

    # -------------------------------
    # Build shared design
    # -------------------------------
    temporal_df, temporal_meta, specs_meta = (
        assemble_one_ff_gam_design.build_temporal_design_base(data_obj)
    )

    X_tuning, tuning_meta = (
        assemble_one_ff_gam_design.build_tuning_design(
            data_obj,
            linear_vars,
            angular_vars,
        )
    )

    # -------------------------------
    # Per-unit GAM
    # -------------------------------
    design_df, y, groups = (
        assemble_one_ff_gam_design.assemble_unit_design_and_groups(
            unit_idx=unit_idx,
            data_obj=data_obj,
            temporal_df=temporal_df,
            temporal_meta=temporal_meta,
            X_tuning=X_tuning,
            tuning_meta=tuning_meta,
            specs_meta=specs_meta,
        )
    )

    kept, history = one_ff_gam_fit.backward_elimination_gam(
        design_df=design_df,
        y=y,
        groups=groups,
        alpha=0.05,
        n_folds=10,
        verbose=True,
    )

    print('Final retained variables:')
    for g in kept:
        print(' ', g.name)

    # -------------------------------
    # Save results
    # -------------------------------
    outdir = Path(f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}')
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)

    with open(outdir / 'kept_groups.pkl', 'wb') as f:
        pickle.dump(kept, f)

    print(f'Saved results to {outdir}')


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run one-FF GAM analysis')
    parser.add_argument('--unit_idx', type=int, required=True)

    args = parser.parse_args()
    main(args.unit_idx)