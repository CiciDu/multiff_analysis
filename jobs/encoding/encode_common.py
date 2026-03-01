"""
Shared utilities for encoding job scripts (encode_stops, encode_pn, encode_vis).
"""

import os
import sys
from pathlib import Path


def bootstrap_repo_path():
    """Add multiff methods to path and chdir to project root."""
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if p.name == "Multifirefly-Project":
            os.chdir(p)
            sys.path.insert(0, str(p / "multiff_analysis/multiff_code/methods"))
            return
    raise RuntimeError("Could not find Multifirefly-Project root")


def get_session_paths(raw_data_folder_path, raw_data_dir_name, monkey_names):
    """
    Return session paths to process.

    If raw_data_folder_path is not None, return [raw_data_folder_path].
    Otherwise, collect all sessions from monkey_names via combine_info_utils.
    """
    if raw_data_folder_path is not None:
        return [raw_data_folder_path]

    from data_wrangling import combine_info_utils

    session_paths = []
    for monkey_name in monkey_names:
        sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(
            raw_data_dir_name, monkey_name
        )
        for _, row in sessions_df.iterrows():
            session_paths.append(
                os.path.join(raw_data_dir_name, row["monkey_name"], row["data_name"])
            )
    return session_paths


DEFAULT_LAMBDA_CONFIG = {
    "lam_f": 100.0,
    "lam_g": 10.0,
    "lam_h": 10.0,
    "lam_p": 10.0,
}
