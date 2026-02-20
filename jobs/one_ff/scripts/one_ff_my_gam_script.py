#!/usr/bin/env python3
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_design,
    one_ff_gam_fit,
    gam_variance_explained,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_parameters
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

VAR_CATEGORIES = {
    'sensory_vars': ['v', 'w'],
    'latent_vars': ['r_targ', 'theta_targ'],
    'position_vars': ['d', 'phi'],
    'eye_position_vars': ['eye_ver', 'eye_hor'],
    'event_vars': ['t_move', 't_targ', 't_stop', 't_rew'],
    'spike_hist_vars': ['spike_hist'],
}


def _subset_design_and_groups(design_df, groups, keep_group_names):
    keep_set = set(keep_group_names)
    groups_kept = [g for g in groups if g.name in keep_set]

    keep_cols = []
    for g in groups_kept:
        keep_cols.extend(g.cols)
    keep_cols = set(keep_cols)

    cols_in_order = [
        c for c in design_df.columns
        if (c in keep_cols) or (c == 'const')
    ]
    return design_df.loc[:, cols_in_order], groups_kept


def _compute_category_variance_contributions(
    design_df,
    y,
    groups,
    dt,
    var_categories,
    outdir,
):
    all_group_names = [g.name for g in groups]

    full_path = outdir / 'cv_var_explained' / 'full_model.pkl'
    full_cv = gam_variance_explained.crossval_variance_explained(
        fit_function=one_ff_gam_fit.fit_poisson_gam,
        design_df=design_df,
        y=y,
        groups=groups,
        dt=dt,
        n_folds=5,
        fit_kwargs=dict(
            l1_groups=[],
            max_iter=1000,
            tol=1e-6,
            verbose=False,
            save_path=None,
        ),
        save_path=full_path,
        cv_mode='blocked_time_buffered',
        buffer_samples=20,
    )

    contributions = {}
    for category_name, category_vars in var_categories.items():
        drop_set = set(category_vars)
        keep_group_names = [
            gname for gname in all_group_names
            if gname not in drop_set
        ]
        reduced_df, reduced_groups = _subset_design_and_groups(
            design_df=design_df,
            groups=groups,
            keep_group_names=keep_group_names,
        )

        loo_path = outdir / 'cv_var_explained' / \
            f'leave_out_{category_name}.pkl'
        reduced_cv = gam_variance_explained.crossval_variance_explained(
            fit_function=one_ff_gam_fit.fit_poisson_gam,
            design_df=reduced_df,
            y=y,
            groups=reduced_groups,
            dt=dt,
            n_folds=5,
            fit_kwargs=dict(
                l1_groups=[],
                max_iter=1000,
                tol=1e-6,
                verbose=False,
                save_path=None,
            ),
            save_path=loo_path,
            cv_mode='blocked_time_buffered',
            buffer_samples=20,
        )

        contributions[category_name] = {
            'vars': category_vars,
            'full_mean_classical_r2': full_cv['mean_classical_r2'],
            'full_mean_pseudo_r2': full_cv['mean_pseudo_r2'],
            'leave_out_mean_classical_r2': reduced_cv['mean_classical_r2'],
            'leave_out_mean_pseudo_r2': reduced_cv['mean_pseudo_r2'],
            'delta_classical_r2': (
                full_cv['mean_classical_r2'] - reduced_cv['mean_classical_r2']
            ),
            'delta_pseudo_r2': (
                full_cv['mean_pseudo_r2'] - reduced_cv['mean_pseudo_r2']
            ),
        }

    return full_cv, contributions


def main(unit_idx: int):
    print(f'Running GAM MAP fit for unit {unit_idx}')
    # -------------------------------
    # Per-unit design
    # -------------------------------
    design_df, y, groups, structured_meta_groups, data_obj = one_ff_gam_design.finalize_one_ff_gam_design(
        unit_idx=unit_idx,
        session_num=0,
        selected_categories=list(VAR_CATEGORIES.keys()),
        var_categories=VAR_CATEGORIES,
    )

    outdir = Path(
        f'all_monkey_data/one_ff_data/my_gam_results/neuron_{unit_idx}')
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'fit_results').mkdir(parents=True, exist_ok=True)

    lam_suffix = one_ff_gam_fit.generate_lambda_suffix(
        lambda_config=structured_meta_groups['lambda_config'])
    save_path = outdir / 'fit_results' / f'{lam_suffix}.pkl'

    fit_res = one_ff_gam_fit.fit_poisson_gam(
        design_df=design_df,
        y=y,
        groups=groups,
        l1_groups=[],
        tol=1e-6,
        verbose=True,
        save_path=str(save_path),
        save_metadata={'structured_meta_groups': structured_meta_groups},
    )

    print('success:', fit_res.success)
    print('message:', fit_res.message)
    print('n_iter:', fit_res.n_iter)
    print('final objective:', fit_res.fun)
    print('grad_norm:', fit_res.grad_norm)

    # also get variance explained
    save_path = outdir / 'cv_var_explained' / f'{lam_suffix}.pkl'
    prs = one_ff_parameters.default_prs()
    cv_res = gam_variance_explained.crossval_variance_explained(
        fit_function=one_ff_gam_fit.fit_poisson_gam,
        design_df=design_df,
        y=y,
        groups=groups,
        dt=prs.dt,
        n_folds=5,
        fit_kwargs=dict(
            l1_groups=[],
            max_iter=1000,
            tol=1e-6,
            verbose=False,
            save_path=None,
        ),
        save_path=save_path,
        cv_mode='blocked_time_buffered',
        buffer_samples=20,
    )

    print(cv_res["mean_classical_r2"])
    print(cv_res["mean_pseudo_r2"])

    full_cv, category_contrib = _compute_category_variance_contributions(
        design_df=design_df,
        y=y,
        groups=groups,
        dt=prs.dt,
        var_categories=VAR_CATEGORIES,
        outdir=outdir,
    )

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

    contrib_df = pd.DataFrame.from_dict(category_contrib, orient='index')
    contrib_df.index.name = 'category'
    contrib_csv = outdir / 'cv_var_explained' / 'category_contributions.csv'
    contrib_df.to_csv(contrib_csv)
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
