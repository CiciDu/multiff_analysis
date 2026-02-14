
import argparse
import os
import sys
from pathlib import Path

# -------------------------------------------------------
# Repo path bootstrap
# -------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
    decode_pn_pipeline
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_folder_path',
        default=None,
        help='Run decoding only on this raw data folder',
    )
    parser.add_argument('--bin_width', type=float, default=0.04)
    parser.add_argument('--t_max', type=float, default=0.20)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    runner = decode_pn_pipeline.PNDecodingRunner(
        raw_data_folder_path=args.raw_data_folder_path,
        bin_width=args.bin_width,
        t_max=args.t_max,
    )

    results_df = runner.run(
        n_splits=args.n_splits,
        save_dir=args.save_dir,
        shuffle_mode='timeshift_fold',
    )

    results_df = runner.run(
        n_splits=args.n_splits,
        save_dir=args.save_dir,
        shuffle_mode='none',
    )

    print(results_df.head())
    return results_df


if __name__ == '__main__':
    main()
