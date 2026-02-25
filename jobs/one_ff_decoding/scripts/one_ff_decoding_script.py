#!/usr/bin/env python3
"""Run One-FF population decoding (canoncorr + linear readout) for a session."""
import os
import sys
from pathlib import Path

import argparse
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
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import (
    one_ff_decoding_pipeline,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_parameters,
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


def main(session_num: int = 0, output_root=None, save_dir=None):
    prs = one_ff_parameters.default_prs()
    runner = one_ff_decoding_pipeline.OneFFDecodingRunner(
        session_num=session_num,
        prs=prs,
        output_root=output_root or "all_monkey_data/one_ff_data/decoding",
    )
    stats = runner.run(save_dir=save_dir, cv_mode='group_kfold')
    stats = runner.run(save_dir=save_dir, cv_mode='blocked_time_buffered')
    return stats


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run One-FF population decoding')
    parser.add_argument(
        '--session_num',
        type=int,
        default=0,
        help='Session index (default: 0)',
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default="all_monkey_data/one_ff_data/decoding",
        help='Output root for decoding results (default: all_monkey_data/one_ff_data/decoding)',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Override save directory (default: output_root/session_N)',
    )
    args = parser.parse_args()
    main(session_num=args.session_num, output_root=args.output_root, save_dir=args.save_dir)
