import gc
from pathlib import Path
import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import hashlib

for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break

from data_wrangling import specific_utils, process_monkey_information, general_utils, combine_info_utils
from neural_data_analysis.neural_analysis_tools.glm_tools.glm_fit import full_session_glm_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_folder_path',
        default=None,
        help='If provided, run GLM only on this raw data folder'
    )
    args = parser.parse_args()

    # Case 1: user explicitly provides a raw data folder
    if args.raw_data_folder_path is not None:
        print('=' * 100)
        print(f'Processing single session: {args.raw_data_folder_path}')

        runner = full_session_glm_runner.FullSessionGLMRunner(
            raw_data_folder_path=args.raw_data_folder_path,
        )
        runner.run()
        return

    # Case 2: no folder provided -> iterate through all sessions
    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'

    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name, 'monkey_Bruno'
    )

    for _, row in sessions_df_for_one_monkey.iterrows():
        print('=' * 100)
        print('=' * 100)
        print(row['data_name'])

        raw_data_folder_path = os.path.join(
            raw_data_dir_name, row['monkey_name'], row['data_name']
        )

        try:
            runner = full_session_glm_runner.FullSessionGLMRunner(
                raw_data_folder_path=raw_data_folder_path,
            )
            runner.run()

        except Exception as e:
            print(f'Error processing {row["data_name"]}: {e}')
            continue


if __name__ == '__main__':
    main()
