import sys
import os
import numpy as np
import sys
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from contextlib import contextmanager
from os.path import exists
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def find_intersection(intervals, query):
    """
    Find intersections between intervals. Intervals are open and are 
    represented as pairs (lower bound, upper bound). 
    The source of the code is:
    source: https://codereview.stackexchange.com/questions/203468/
    find-the-intervals-which-have-a-non-empty-intersection-with-a-given-interval

    Parameters
    ----------
    intervals: array_like, shape=(N, 2) 
        Array of intervals.
    query: array_like, shape=(2,) 
        Interval to query

    Returns
    -------
    indices_of_overlapped_intervals: array
        Array of indexes of intervals that overlap with query

    """
    intervals = np.asarray(intervals)
    lower, upper = query
    indices_of_overlapped_intervals = np.where(
        (lower < intervals[:, 1]) & (intervals[:, 0] < upper))[0]
    return indices_of_overlapped_intervals


def save_df_to_csv(df, df_name, data_folder_name, exists_ok=False):
    if data_folder_name:
        csv_name = df_name + '.csv'
        filepath = os.path.join(data_folder_name, csv_name)
        if exists(filepath) & exists_ok:
            print(filepath, 'already exists.')
        else:
            os.makedirs(data_folder_name, exist_ok=True)
            df.to_csv(filepath)
            print("new", df_name, "is stored in ", filepath)


def take_out_a_sample_from_arrays(sampled_indices, *args):
    sampled_args = []
    for arg in args:
        sampled_arg = arg[sampled_indices]
        sampled_args.append(sampled_arg)
    return sampled_args


def take_out_a_sample_from_df(sampled_indices, *args):
    sampled_args = []
    for arg in args:
        sampled_arg = arg.iloc[sampled_indices]
        sampled_args.append(sampled_arg)
    return sampled_args


def find_time_bins_for_an_array(array_of_interest):
    # find mid-points of each interval in monkey_information['time']
    interval_lengths = np.diff(array_of_interest)
    half_interval_lengths = interval_lengths/2
    half_interval_lengths = np.append(
        half_interval_lengths, half_interval_lengths[-1])
    # find the boundaries of boxes that surround each element of monkey_information['time']
    time_bins = array_of_interest + half_interval_lengths
    # add the position of the leftmost boundary
    first_box_boundary_position = array_of_interest[0]-half_interval_lengths[0]
    time_bins = np.append(first_box_boundary_position, time_bins)
    return time_bins


def find_outlier_position_index(data, outlier_z_score_threshold=2):
    data = np.array(data)
    # calculate standard deviation in rel_curv_to_cur_ff_center
    std = np.std(data)
    # find z-score of each point
    z_score = (data - np.mean(data)) / std
    # find outliers
    outlier_positions = np.where(
        np.abs(z_score) > outlier_z_score_threshold)[0]
    return outlier_positions


def make_rotation_matrix(x0, y0, x1, y1):
    # find a rotation matrix so that (x1, y1) is to the north of (x0, y0)

    # Find the angle from the starting point to the target
    theta = pi/2-np.arctan2(y1 - y0, x1 - x0)
    c, s = np.cos(theta), np.sin(theta)
    # Find the rotation matrix
    rotation_matrix = np.array(((c, -s), (s, c)))
    return rotation_matrix


def take_out_data_points_in_valid_intervals(t_array, valid_intervals_df):
    # # take out unique combd_valid_interval_group and flatten all the intervals
    flattened_intervals = valid_intervals_df.values.reshape(-1)
    # see which data_points are within a valid interval (rather than between them)
    match_to_interval = np.searchsorted(flattened_intervals, t_array)
    # if the index is odd, it means the data point is within a valid interval
    within_valid_interval = match_to_interval % 2 == 1
    t_array_valid = t_array[within_valid_interval]
    t_array_valid_index = np.where(within_valid_interval)[0]
    return t_array_valid, t_array_valid_index


@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    """
    Set some parameters for plotting

    """
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = "sans serif"
    global fig
    fig = plt.figure(dpi=dpi)
    yield
    plt.show()


def find_rows_with_na(df, df_name="DataFrame"):
    """
    Find and analyze rows with NA values in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    df_name : str, optional
        Name of the DataFrame for display purposes, defaults to "DataFrame"

    Returns:
    --------
    tuple
        (na_rows, na_cols) where:
        - na_rows: DataFrame containing rows with any NA values
        - na_cols: Index of columns containing any NA values
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        print(f"\n{df_name} is empty")
        return pd.DataFrame(), pd.Index([])

    # Find rows and columns with NA values
    na_rows = df[df.isna().any(axis=1)]
    na_cols = df.columns[df.isna().any(axis=0)]

    # Calculate NA statistics
    na_sum = df.isna().sum()
    na_df = na_sum[na_sum > 0]
    total_rows = len(df)

    # Print analysis if NA values exist
    if len(na_rows) > 0:
        # Print header
        print("\n" + "="*80)
        print(f"NA Values Analysis for {df_name} ({total_rows:,} rows)")
        print("="*80)

        # Print column-wise NA summary
        print("\nColumns with NA values:")
        print("-"*60)
        for col, count in na_df.items():
            percentage = (count / total_rows) * 100
            print(f"{col:<40} {count:>8,} ({percentage:>6.1f}%)")
        print("-"*60)

        # # Print sample of rows with NA values
        # if len(na_rows) > 0:
        #     print("\nSample of rows with NA values (first 5):")
        #     print("-"*60)
        #     print(na_rows.head().to_string())
        #     print("-"*60)
    else:
        print(f"\nNo NA values found in {df_name}")

    return na_rows, na_cols


def find_duplicate_rows(df, column_subset=None):
    print("\n" + "="*80)
    if column_subset is None:
        column_subset = df.columns
        print("🔍 Duplicate Rows Analysis:")
    else:
        print("🔍 Duplicate Rows Analysis for columns: ", column_subset)
    print("="*80)
    # Find duplicate rows
    duplicates = df[column_subset].duplicated(keep=False)
    duplicate_rows = df[duplicates]

    if len(duplicate_rows) > 0:
        # Show how many duplicates we found
        num_duplicates = duplicates.sum()
        print(f"\nFound {num_duplicates:,} duplicate rows")

        # Show the duplicate rows
        if num_duplicates > 0:
            print("\nDuplicate rows:")
            print("-"*60)
            print(duplicate_rows.head())

            # Show which combinations are duplicated
            print("\nDuplicate combinations:")
            print("-"*60)
            duplicate_combinations = duplicate_rows.value_counts()
            print(duplicate_combinations.head())
        print("="*80)

    else:
        print("No duplicate rows found in the dataframe")
    return duplicate_rows
