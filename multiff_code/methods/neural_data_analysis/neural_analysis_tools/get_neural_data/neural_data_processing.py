from data_wrangling import time_calib_utils
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib import rc
import scipy.interpolate as interpolate
import subprocess


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def make_spikes_df(raw_data_folder_path, ff_caught_T_sorted,
                   sampling_rate=20000):

    neural_data_path = raw_data_folder_path.replace(
        'raw_monkey_data', 'neural_data')
    sorted_neural_data_name = os.path.join(neural_data_path, "Sorted")
    # time_calibration_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')

    smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(
        raw_data_folder_path)

    spike_times = _load_spike_times(sorted_neural_data_name)

    spike_times_in_s = spike_times/sampling_rate

    # spike_times_in_s = time_calib_utils.calibrate_neural_data_time(spike_times_in_s, raw_data_folder_path, ff_caught_T_sorted)

    offset_neural_txt = time_calib_utils.find_offset_neural_txt_const(
        raw_data_folder_path, ff_caught_T_sorted)
    spike_times_in_s = spike_times_in_s - offset_neural_txt

    spike_clusters = _load_spike_clusters(sorted_neural_data_name)

    spike_times_in_s, spike_clusters = _filter_spike_data(
        spike_times_in_s, spike_clusters, smr_markers_start_time)

    spikes_df = pd.DataFrame(
        {'time': spike_times_in_s, 'cluster': spike_clusters})

    return spikes_df


def _load_spike_times(sorted_neural_data_path):
    """Load and process spike times."""
    spike_times = np.load(os.path.join(
        sorted_neural_data_path, "spike_times.npy"))
    spike_times = spike_times.reshape(-1)
    return spike_times


def _load_spike_clusters(sorted_neural_data_path):
    """Load and process spike clusters."""
    filepath = os.path.join(sorted_neural_data_path, "spike_clusters.npy")
    spike_clusters = np.load(filepath)
    spike_clusters = spike_clusters.reshape(-1)
    return spike_clusters


def _filter_spike_data(spike_times_in_s, spike_clusters, smr_markers_start_time):
    """Filter spike times and clusters based on start time."""
    valid_idx = np.where(spike_times_in_s >= smr_markers_start_time)[0]
    spike_times_in_s = spike_times_in_s[valid_idx]
    spike_clusters = spike_clusters[valid_idx]
    return spike_times_in_s, spike_clusters


# def rebin_raw_spike_data(spike_segs_df, new_seg_info, bin_width=0.2):
#     # This function rebins the data by segment and time bin, and takes the median of the data within each bin
#     # It makes sure that the bins are perfectly aligned within segments (whereas in the previous method, bins are continuously assigned to all time points)
#     # df must contain columns: segment, seg_start_time, seg_end_time, time, bin,

#     new_seg_info.sort_values(by='segment', inplace=True)
#     concat_seg_data = []

#     df = spike_segs_df

#     for _, segment_row in new_seg_info.iterrows():

#         # Not using fully vectorized approach in case there's overlap of time between segments
#         # Subset relevant rows belonging to the segment
#         seg_data_df = df[
#             (df['segment'] == segment_row['segment']) &
#             (df['time'] >= segment_row['new_seg_start_time']) &
#             (df['time'] <= segment_row['new_seg_end_time'])
#         ].copy()  # copy to avoid SettingWithCopyWarning

#         # Create time bins and assign bin index
#         time_bins = np.arange(segment_row['new_seg_start_time'], segment_row['new_seg_end_time'], bin_width)
#         seg_data_df['new_bin'] = np.digitize(
#             seg_data_df['time'], time_bins) - 1

#         cols = ['new_segment', 'new_seg_start_time', 'new_seg_end_time', 'new_seg_duration']
#         seg_data_df[cols] = segment_row[cols].values

#         concat_seg_data.append(seg_data_df)

#     # Concatenate all processed segments
#     concat_seg_data = pd.concat(concat_seg_data, ignore_index=True)
#     concat_seg_data.sort_values(by=['new_segment', 'new_bin'], inplace=True)

#     # Take the median of the data within each bin
#     rebinned_data = concat_seg_data.groupby(
#         ['new_segment', 'new_bin']).median().reset_index(drop=False)

#     return rebinned_data


def _make_all_binned_spikes(spikes_df, min_time=None, max_time=None, bin_width=0.02):
    """Efficiently bin spikes and stack bins for each spike cluster."""
    if max_time is None:
        max_time = math.ceil(spikes_df.time.max())
    if min_time is None:
        min_time = 0

    time_bins = np.arange(min_time, max_time + bin_width, bin_width)
    spikes_df = spikes_df.copy()

    # Assign each spike to a time bin
    # subtract 1 for 0-based indexing
    spikes_df['bin'] = np.digitize(spikes_df['time'], time_bins) - 1

    # Group by bin and cluster, then count spikes
    grouped = spikes_df.groupby(
        ['bin', 'cluster']).size().unstack(fill_value=0)

    # Ensure all bins and all clusters are present
    grouped = grouped.reindex(index=np.arange(
        len(time_bins)-1), columns=np.sort(spikes_df['cluster'].unique()), fill_value=0)

    return time_bins, grouped.values


import numpy as np

def prepare_binned_spikes_df(spikes_df, spike_time_col='time', bin_width=0.02, min_time=None, max_time=None):
    """
    Bin spike times and return both the bin edges and a DataFrame of spike counts per cluster.
    
    Args:
        spikes_df (pd.DataFrame): Must have 'time' and 'cluster' columns.
        bin_width (float): Width of each bin in seconds.
        min_time (float, optional): Minimum time for binning. Defaults to 0.
        max_time (float, optional): Maximum time for binning. Defaults to ceil(max spike time).
    
    Returns:
        time_bins (np.ndarray): Array of bin edges.
        binned_spikes_df (pd.DataFrame): Rows = bins, columns = clusters + 'bin'.
    """
    if max_time is None:
        max_time = np.ceil(spikes_df[spike_time_col].max())
    if min_time is None:
        min_time = 0

    time_bins = np.arange(min_time, max_time + bin_width, bin_width)
    spikes_df = spikes_df.copy()

    # Assign each spike to a bin (0-based)
    spikes_df['bin'] = np.digitize(spikes_df[spike_time_col], time_bins) - 1

    # Count spikes per bin per cluster
    spike_counts = spikes_df.groupby(['bin', 'cluster']).size().unstack(fill_value=0)

    # Reindex to ensure all bins & clusters are present
    binned_spikes_df = spike_counts.reindex(
        index=np.arange(len(time_bins) - 1),
        columns=np.sort(spikes_df['cluster'].unique()),
        fill_value=0
    )

    # Rename columns
    binned_spikes_df.columns = [f'cluster_{c}' for c in binned_spikes_df.columns]
    binned_spikes_df['bin'] = binned_spikes_df.index

    return time_bins, binned_spikes_df


def convolve_neural_data(x_var, kernel_len=7):
    """
    Convolve neural data in Yizhou's way.
    """
    # Define a b-spline
    knots = np.hstack(([-1.001]*3, np.linspace(-1.001, 1.001, 5), [1.001]*3))
    tp = np.linspace(-1., 1., kernel_len)
    bX = splineDesign(knots, tp, ord=4, der=0, outer_ok=False)

    modelX = np.zeros((x_var.shape[0], x_var.shape[1]*bX.shape[1]))
    for neu in range(x_var.shape[1]):
        modelX2 = np.zeros((x_var.shape[0], bX.shape[1]))
        for k in range(bX.shape[1]):
            modelX2[:, k] = np.convolve(x_var[:, neu], bX[:, k], 'same')

        modelX[:, neu*bX.shape[1]:(neu+1)*bX.shape[1]] = modelX2

    return modelX


def add_lags_to_each_feature(var, lag_numbers, trial_vector=None, rearrange_lag_based_on_abs_value=True):
    """
    Add lags to each feature in var, separately within each trial (if provided),
    minimizing explicit Python loops for better performance.
    """
    if rearrange_lag_based_on_abs_value:
        lag_numbers = sorted(lag_numbers, key=lambda x: abs(x))

    if isinstance(var, pd.DataFrame):
        column_names = var.columns.astype(str)
    else:
        column_names = [f"feat{i}" for i in range(var.shape[1])]
        var = pd.DataFrame(var, columns=column_names)

    n_units = var.shape[1]

    # Helper function: create lagged matrix for one trial group
    def lag_group(df):
        n = df.shape[0]
        # Create empty array for all lags: rows=n, cols=n_units*len(lag_numbers)
        lagged_data = np.full((n, n_units * len(lag_numbers)), np.nan)

        # Convert to numpy for fast slicing
        arr = df.drop(columns=['lag_segment_id'], errors='ignore').values

        # For each lag, fill corresponding columns with shifted data
        for idx, lag in enumerate(lag_numbers):
            col_start = idx * n_units
            col_end = col_start + n_units
            if lag < 0:
                lagged_data[:lag, col_start:col_end] = arr[-lag:, :]
            elif lag > 0:
                lagged_data[lag:, col_start:col_end] = arr[:-lag, :]
            else:
                lagged_data[:, col_start:col_end] = arr

        # Return as DataFrame with proper columns
        new_cols = []
        for lag in lag_numbers:
            new_cols.extend([f"{c}_{lag}" for c in column_names])
        return pd.DataFrame(lagged_data, columns=new_cols, index=df.index)

    if trial_vector is None:
        # Single "trial": just apply lag_group on all data at once
        var_lags = lag_group(var)
        # Fill NaNs forward/backward globally
        var_lags = var_lags.ffill().bfill()
    else:
        # Use groupby-apply: pandas efficiently handles groups
        trial_vector = pd.Series(
            trial_vector, index=var.index if isinstance(var, pd.DataFrame) else None)
        var['lag_segment_id'] = trial_vector
        var_lags = var.groupby(
            'lag_segment_id', group_keys=False).apply(lag_group)
        # Fill NaNs forward/backward within each trial group
        var_lags = var_lags.groupby(trial_vector).apply(
            lambda df: df.ffill().bfill())
        var_lags.reset_index(drop=True, inplace=True)

        # Clean up temp column
        var.drop(columns=['lag_segment_id'], inplace=True)

    # ------------------------------------------------------------------
    # fill any residual NaNs from the next‑nearest lag
    # ------------------------------------------------------------------
    pos_lags = sorted(l for l in lag_numbers if l >= 0)
    neg_lags = sorted((l for l in lag_numbers if l <= 0),
                      key=abs)  # e.g. −1, −2, −3…

    def fill_from_neighbour(curr, prev):
        """Copy values row-wise from *prev* lag into *curr* lag where NA."""
        curr_cols = [f"{c}_{curr}" for c in column_names]

        # check if there are any NaNs in var_lags[curr_cols]
        if var_lags[curr_cols].isna().any(axis=1).sum() > 0:
            # print(f"NaNs found in {curr_cols}")
            for feat in column_names:
                curr_col = f"{feat}_{curr}"
                prev_col = f"{feat}_{prev}"
                na_mask = var_lags[curr_col].isna()
                var_lags.loc[na_mask,
                             curr_col] = var_lags.loc[na_mask, prev_col]

            # This is another approach, but it works on all columns with the same lag at once.
            # The potential problem is that if there are NaNs in the same row for some lag_N columns, it will fill the NaNs in the same row for all lag_N columns.
            # prev_cols = [f"{c}_{prev}" for c in column_names]
            # var_lags.loc[var_lags[curr_cols].isna().any(axis=1), curr_cols] = var_lags.loc[var_lags[curr_cols].isna().any(axis=1), prev_cols]

    # positive direction: 1 ← 0 already done, so start at index 1
    for i in range(1, len(pos_lags)):
        fill_from_neighbour(pos_lags[i], pos_lags[i - 1])

    # negative direction: −1 already handled, so start at index 1
    for i in range(1, len(neg_lags)):
        fill_from_neighbour(neg_lags[i], neg_lags[i - 1])

    return var_lags


def splineDesign(knots, x, ord=4, der=0, outer_ok=False):
    """
    Reproduces behavior of R function splineDesign() for use by ns(). See R documentation for more information.
    Python code uses scipy.interpolate.splev to get B-spline basis functions, while R code calls C.
    Note that der is the same across x.
    """
    # Convert knots and x to numpy arrays and sort knots
    knots = np.sort(np.array(knots, dtype=np.float64))
    x = np.array(x, dtype=np.float64)

    # Copy of original x values
    xorig = x.copy()

    # Boolean array indicating non-NaN values in x
    not_nan = ~np.isnan(xorig)

    # Check if any x values are outside the range of knots
    need_outer = any(x[not_nan] < knots[ord - 1]
                     ) or any(x[not_nan] > knots[-ord])

    # Boolean array indicating x values within the range of knots
    in_x = (x >= knots[0]) & (x <= knots[-1]) & not_nan

    # If x values are outside the range of knots and outer_ok is False, raise an error
    if need_outer and not outer_ok:
        raise ValueError("the 'x' data must be in the range %f to %f unless you set outer_ok==True'" % (
            knots[ord - 1], knots[-ord]))

    # If x values are outside the range of knots and outer_ok is True, adjust knots and x
    if need_outer and outer_ok:
        x = x[in_x]
        dkn = np.diff(knots)[::-1]
        reps_start = ord - 1
        reps_end = max(0, ord - np.where(dkn > 0)
                       [0][0] - 1) if any(dkn > 0) else np.nan
        idx = [0] * (ord - 1) + list(range(len(knots))) + \
            [len(knots) - 1] * reps_end
        knots = knots[idx]

    # If x values are within the range of knots and there are NaN values in x, adjust x
    if (not need_outer) and any(~not_nan):
        x = x[in_x]

    # Calculate B-spline basis functions
    m = len(knots) - ord
    v = np.zeros((m, len(x)), dtype=np.float64)
    d = np.eye(m, len(knots))
    for i in range(m):
        v[i] = interpolate.splev(x, (knots, d[i], ord - 1), der=der)

    # Construct design matrix
    design = np.zeros((v.shape[0], xorig.shape[0]), dtype=np.float64)
    for i in range(v.shape[0]):
        design[i, in_x] = v[i]

    # Return transposed design matrix
    return design.transpose()


def get_mapping_table_between_hard_drive_and_local_folders(monkey_name, hdrive_dir, neural_data_folder_name, filter_neural_file_func, save_table=False):
    all_local_data_paths = []
    all_time_calib_paths = []
    all_hdrive_folders = []
    all_hdrive_data_paths = []

    raw_data_dir = os.path.join(
        '/Users/dusiyi/Documents/Multifirefly-Project/all_monkey_data/raw_monkey_data/', monkey_name)
    neural_data_dir = raw_data_dir.replace('raw_monkey_data', 'neural_data')
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    sessions = os.listdir(raw_data_dir)
    sessions = [f for f in sessions if 'data_' in f]

    for data_dir in sessions:
        local_data_path = os.path.join(neural_data_dir, data_dir)
        neural_event_time_path = os.path.join(local_data_path.replace(
            'neural_data', 'time_calibration'), 'neural_event_time.txt')

        data_number = data_dir.split('_')[-1]
        data_name = month_dict[int(data_number[1])] + \
            ' ' + data_number[-2:] + ' 2018'

        all_local_data_paths.append(local_data_path)
        all_time_calib_paths.append(neural_event_time_path)
        hdrive_folder = os.path.join(
            hdrive_dir, data_name, neural_data_folder_name)
        all_hdrive_folders.append(hdrive_folder)

        # get the neural file path
        result = subprocess.run(
            ['ls', '--color=never', hdrive_folder], capture_output=True, text=True, check=True)
        file_list = result.stdout.splitlines()  # Split into clean file names
        neural_files = filter_neural_file_func(file_list)
        if len(neural_files) != 1:
            raise ValueError(
                f"There should be exactly one neural file in the directory. Found {len(neural_files)} files. The files in the directory are {file_list}")
        else:
            hdrive_data_path = os.path.join(hdrive_folder, neural_files[0])

        all_hdrive_data_paths.append(hdrive_data_path)

    mapping_table = pd.DataFrame({'local_path': all_local_data_paths,
                                  'neural_event_time_path': all_time_calib_paths,
                                  'hdrive_path': all_hdrive_data_paths,
                                  'hdrive_folder': all_hdrive_folders})

    if save_table:
        monkey = 'bruno' if 'bruno' in monkey_name.lower() else 'schro'
        mapping_table.to_csv(f'/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/multiff_code/methods/eye_position_analysis/neural_data_analysis/get_neural_data/MATLAB_processing/{monkey}_mapping_table.csv',
                             index=False)
    return mapping_table
