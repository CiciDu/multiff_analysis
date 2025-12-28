from data_wrangling import combine_info_utils, specific_utils
import os
import pandas as pd
from pathlib import Path
from neural_data_analysis.neural_analysis_tools.decoding_tools.event_decoding import cmp_decode
from data_wrangling import further_processing_class


def load_all_retry_decoder_results(
        raw_data_dir_name='all_monkey_data/raw_monkey_data',
        monkey_name='monkey_Bruno',
        cumulative=True):
    """
    Wraps the loop that loads retry decoder results across all sessions
    for one monkey and returns df_all_combined and the updated sessions_df.
    """

    # Prepare session list
    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name, monkey_name)

    df_all_combined = pd.DataFrame()

    # Loop through sessions
    for index, row in sessions_df_for_one_monkey.iterrows():
        print(row['data_name'])
        raw_data_folder_path = os.path.join(
            raw_data_dir_name, row['monkey_name'], row['data_name'])
        print(raw_data_folder_path)

        try:
            pn = further_processing_class.FurtherProcessing(
                raw_data_folder_path=raw_data_folder_path
            )
            retry_decoder_dir = (
                pn.retry_decoder_cumulative_folder_path
                if cumulative else pn.retry_decoder_folder_path
            )
            job_result_dir = Path(retry_decoder_dir) / 'runs'

            df_all = cmp_decode._load_all_results(job_result_dir, None)
            df_all['session'] = row['data_name']

            df_all_combined = pd.concat(
                [df_all_combined, df_all],
                ignore_index=True
            )

        except Exception as e:
            print(f"Error in {raw_data_folder_path}: {e}")
            continue

    df_all_combined = df_all_combined.drop_duplicates()

    return df_all_combined
