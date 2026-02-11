#!/usr/bin/env python3
"""
Run PGAM for a single unit on one-firefly session data.
"""

import argparse
import os
import sys
from pathlib import Path



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


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main(args):
    # ------------------
    # Parameters
    # ------------------
    prs = default_prs()
    pgam_save_dir = 'all_monkey_data/one_ff_data/pgam_results'

    data_obj = one_ff_pipeline.OneFFSessionData(
        mat_path='all_monkey_data/one_ff_data/sessions_python.mat',
        prs=prs,
        session_num=0,
    )

    covariate_names = [
        'v', 'w', 'd', 'phi',
        'r_targ', 'theta_targ',
        'eye_ver', 'eye_hor',
    ]

    data_obj.compute_covariates(covariate_names)
    data_obj.compute_spike_counts()
    data_obj.smooth_spikes()
    data_obj.compute_events()
    binned_spikes_df = data_obj.get_binned_spikes_df()

    # first just try to load the data
    pgam_runner = pgam_class.PGAMclass(
        x_var=binned_spikes_df,
        bin_width=prs.dt,
        save_dir=pgam_save_dir,
    )
    try:
        pgam_runner.load_pgam_results(args.unit_idx)
        print(f"loaded PGAM results for unit {args.unit_idx}")
        return
    except FileNotFoundError:
        print(f"running PGAM for unit {args.unit_idx}")

    # ------------------
    # Build PGAM design
    # ------------------
    sm_handler = one_ff_pgam_design.build_smooth_handler(
        data_obj=data_obj,
        unit_idx=args.unit_idx,
        covariate_names=covariate_names,
        tuning_covariates=covariate_names,
        use_cyclic=set(),
        order=4,
    )

    # ------------------
    # PGAM data_obj
    # ------------------
    pgam_runner = pgam_class.PGAMclass(
        x_var=binned_spikes_df,
        bin_width=data_obj.prs.dt,
        save_dir=pgam_save_dir,
    )

    pgam_runner.sm_handler = sm_handler
    pgam_runner.trial_ids = data_obj.covariate_trial_ids
    pgam_runner.train_trials = pgam_runner.trial_ids % 3 != 1

    # ------------------
    # Run PGAM
    # ------------------
    pgam_runner.run_pgam(neural_cluster_number=args.unit_idx)
    pgam_runner.kernel_h_length = 100
    pgam_runner.post_processing_results()
    pgam_runner.save_results()


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run PGAM for a single unit (one-firefly task)'
    )
    parser.add_argument(
        '--unit_idx',
        type=int,
        default=0,
        help='Unit index to run PGAM on',
    )

    args = parser.parse_args()
    main(args)
