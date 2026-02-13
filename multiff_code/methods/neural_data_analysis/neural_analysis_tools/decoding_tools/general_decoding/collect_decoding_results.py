import os
import pandas as pd

from data_wrangling import combine_info_utils
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import cv_decoding


def collect_all_session_decoding_results(
    raw_data_dir_name,
    monkey_name,
    decode_runner_class,
):
    """
    Collect existing CV decoding results across all sessions for one monkey.

    Parameters
    ----------
    raw_data_dir_name : str
        Root directory containing raw data.
    monkey_name : str
        Name of the monkey.
    decode_runner_class : class
        Runner class (e.g., decode_stops_pipeline.StopDecodingRunner).

    Returns
    -------
    all_session_results_df : pd.DataFrame
        Concatenated decoding results across all sessions.
    """

    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name,
        monkey_name,
    )

    all_session_results = []

    for _, row in sessions_df_for_one_monkey.iterrows():

        print('=' * 100)
        print(row['data_name'])

        raw_data_folder_path = os.path.join(
            raw_data_dir_name,
            row['monkey_name'],
            row['data_name'],
        )

        runner = decode_runner_class(
            raw_data_folder_path=raw_data_folder_path,
        )

        save_dir = runner._get_save_dir()

        session_results_df = cv_decoding.run_cv_decoding(
            save_dir=save_dir,
            load_existing_only=True,
        )

        session_results_df['data_name'] = row['data_name']

        all_session_results.append(session_results_df)

    all_session_results_df = pd.concat(
        all_session_results,
        axis=0,
        ignore_index=True,
    )

    return all_session_results_df