import sys
import os
import sys
import numpy as np
import sys
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import contextmanager
from os.path import exists
import logging
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


def check_array_integrity(X, name="Array", top_n=10, verbose=True):
    """
    Checks for NaN and infinite values in a NumPy array, and prints summary stats.

    Parameters:
    - X: np.ndarray
    - name: str, name of the array (for display)
    - top_n: int, how many top rows/columns with most NaNs to show
    - verbose: bool, if False, suppress detailed row/column breakdown
    """
    print(f"\n=== Checking: {name} ===")
    print(f"Shape: {X.shape}")

    has_nan = np.isnan(X)
    total_nan = np.sum(has_nan)

    if total_nan > 0:
        percent_nan = 100 * total_nan / X.size
        print(f"❗ Total NaN values: {total_nan} ({percent_nan:.4f}%)")

        rows_with_nan = np.any(has_nan, axis=1)
        cols_with_nan = np.any(has_nan, axis=0)

        print(f"Rows with ≥1 NaN: {np.sum(rows_with_nan)}")
        print(f"Cols with ≥1 NaN: {np.sum(cols_with_nan)}")

        if verbose:
            nan_indices = np.where(has_nan)
            print(
                f"First {top_n} NaN positions (row, col): {list(zip(nan_indices[0][:top_n], nan_indices[1][:top_n]))}")

            # Columns with most NaNs
            nan_per_col = np.sum(has_nan, axis=0)
            top_cols = np.argsort(nan_per_col)[::-1][:top_n]
            print("Columns with most NaNs:")
            for idx in top_cols:
                if nan_per_col[idx] > 0:
                    print(f"  Column {idx}: {nan_per_col[idx]} NaNs")

            # Rows with most NaNs
            nan_per_row = np.sum(has_nan, axis=1)
            top_rows = np.argsort(nan_per_row)[::-1][:top_n]
            print("Rows with most NaNs:")
            for idx in top_rows:
                if nan_per_row[idx] > 0:
                    print(f"  Row {idx}: {nan_per_row[idx]} NaNs")
    else:
        print("✅ No NaN values found.")

    # Infinite values
    has_inf = np.isinf(X)
    total_inf = np.sum(has_inf)

    if total_inf > 0:
        percent_inf = 100 * total_inf / X.size
        print(f"❗ Total infinite values: {total_inf} ({percent_inf:.4f}%)")
    else:
        print("✅ No infinite values found.")

    # Summary statistics
    print("\n--- Summary statistics (excluding NaNs) ---")
    print(f"Min: {np.nanmin(X):.6f}")
    print(f"Max: {np.nanmax(X):.6f}")
    print(f"Mean: {np.nanmean(X):.6f}")
    print(f"Std: {np.nanstd(X):.6f}")


def ensure_boolean_dtype(df):
    """
    Ensure that all object-dtype columns with boolean values are cast to boolean dtype.
    """
    bool_columns = df.select_dtypes(include='object').applymap(
        lambda x: x in [True, False]).all()
    for col in bool_columns.index[bool_columns]:
        df[col] = df[col].astype(bool)
    return df


def check_perfect_correlations(data, threshold=1.0, tol=1e-10):
    """
    data: 2D NumPy array or pandas DataFrame (samples x features)
    threshold: correlation magnitude to look for (default = 1.0)
    tol: numerical tolerance to allow for floating point error
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    corr_matrix = data.corr()
    n = corr_matrix.shape[0]

    perfect_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            corr_ij = corr_matrix.iloc[i, j]
            if np.abs(np.abs(corr_ij) - threshold) < tol:
                perfect_pairs.append(
                    (data.columns[i], data.columns[j], corr_ij))

    if len(perfect_pairs) > 0:
        print(f"Found {len(perfect_pairs)} perfect correlations:")
        for pair in perfect_pairs:
            print(f"  {pair[0]} and {pair[1]}: {pair[2]:.6f}")
    else:
        print("No perfect correlations found.")

    return perfect_pairs


def drop_rows_with_any_na(df):
    """
    Drops rows from the DataFrame that contain any NaN values.
    """
    all_na_rows = df[df.isna().any(axis=1)].index.tolist()
    if len(all_na_rows) > 0:
        print(f"Dropped {len(all_na_rows)} rows with any NaN values out of {len(df)} rows ")
    return df.drop(index=all_na_rows)


def convert_bool_to_int(df):
    """
    Convert all boolean columns to integer type.
    """
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def drop_columns_with_many_nans(df, threshold=0.5):
    """
    Drops columns from the DataFrame that have more than `threshold` proportion of missing values.
    """
    # print(
    #     f"Dropping columns with more than {threshold*100}% missing values ...")
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    if len(cols_to_drop) > 0:
        print(
            f"Dropped {len(cols_to_drop)} columns with more than {threshold*100}% missing values: {cols_to_drop}")
    return df_cleaned, cols_to_drop


def drop_columns_with_only_na(df):
    """
    Drops columns from the DataFrame that contain only NaN values.
    """
    all_na_cols = df.columns[df.isna().all()].tolist()

    print("Dropped columns with all NaN values:", all_na_cols)

    return df.drop(columns=all_na_cols)


def drop_na_cols(df, df_name=None):
    # Identify columns with missing values
    na_counts = df.isna().sum()
    cols_with_na = na_counts[na_counts > 0].index.tolist()

    # Print number and names of dropped columns
    if df_name:
        print(
            f'Dropped {len(cols_with_na)} columns due to containing NA in {df_name} via calling drop_na_cols function: {cols_with_na}')
    else:
        print(
            f'Dropped {len(cols_with_na)} columns due to containing NA via calling drop_na_cols function: {cols_with_na}')

    # Drop columns with missing values
    df.drop(columns=cols_with_na, inplace=True)
    return df


def clean_float(val):
    if isinstance(val, float):
        val_str = f"{val:.10f}".rstrip('0').rstrip(
            '.')  # Remove trailing zeros and dot
        return val_str.replace('.', 'p')
    return str(val)


def check_for_high_correlations(df, threshold=0.9):
    corr = df.corr()
    # Mask upper triangle and diagonal
    mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)
    corr_masked = corr.where(mask)

    # Extract high correlations
    high_corr_pairs = corr_masked.stack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs.abs() > threshold]

    if len(high_corr_pairs) > 0:
        print(high_corr_pairs)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,  # or use `stream=sys.stdout` for console
    )
    print('Set up logging configuration.')


def outlier_cutoff(durations, method='logmad', k=3.5, iqr_k=3.0, q=0.995):
    """
    One-sided high cutoff for durations. Returns (cutoff, mask).
    - Zeros are never 'too large' and are included in the mask.
    - Negatives raise (shouldn't exist for durations).
    """
    x = np.asarray(durations, float)
    if np.any(x < 0):
        raise ValueError("Durations must be >= 0.")

    pos = x[x > 0]  # ignore zeros for cutoff calc
    if pos.size == 0:
        return np.inf, np.ones_like(x, dtype=bool)

    if method == 'logmad':
        g = np.log(pos)
        med = np.median(g)
        mad = np.median(np.abs(g - med))
        mad_normal = 1.4826 * mad
        cutoff = np.exp(med + k * mad_normal)
    elif method == 'iqr':
        q1, q3 = np.percentile(pos, [25, 75])
        iqr = q3 - q1
        cutoff = q3 + iqr_k * iqr
    elif method == 'quantile':
        cutoff = np.quantile(pos, q)
    else:
        raise ValueError("method must be 'logmad', 'iqr', or 'quantile'.")

    mask = x <= cutoff
    return float(cutoff), mask


def check_na_in_df(df, df_name="DataFrame", return_rows_and_columns=True):
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
    na_row_count = len(na_rows)

    # Print analysis if NA values exist
    if na_row_count > 0:
        print("\n" + "="*80)
        print(f"NA Values Analysis for {df_name} ({total_rows:,} rows)")
        print("="*80)
        print(f"\nNumber of rows with at least one NA value: {na_row_count:,}")

        # Print column-wise NA summary
        print("\nColumns with NA values:")
        print("-"*60)
        for col, count in na_df.items():
            percentage = (count / total_rows) * 100
            print(f"{col:<40} {count:>8,} ({percentage:>6.1f}%)")
        print("-"*60)

    else:
        print(f"\nNo NA values found in {df_name}")

    if return_rows_and_columns:
        return na_rows, na_cols
    else:
        return


setup_logging()
