import sys
from data_wrangling import process_monkey_information, specific_utils, time_calib_utils
from pattern_discovery import pattern_by_trials, pattern_by_points, make_ff_dataframe, ff_dataframe_utils, pattern_by_trials, pattern_by_points, cluster_analysis, organize_patterns_and_features, category_class
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
import scipy.interpolate as interpolate
import subprocess


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)



def make_spike_df(raw_data_folder_path, ff_caught_T_sorted,
                      sampling_rate=20000):
    
    neural_data_path = raw_data_folder_path.replace('raw_monkey_data', 'neural_data')
    sorted_neural_data_name = os.path.join(neural_data_path, "Sorted")
    # time_calibration_path = raw_data_folder_path.replace('raw_monkey_data', 'time_calibration')

    smr_markers_start_time, smr_markers_end_time = time_calib_utils.find_smr_markers_start_and_end_time(raw_data_folder_path)

    spike_times = _load_spike_times(sorted_neural_data_name)
    
    spike_times_in_s = spike_times/sampling_rate
    
    #spike_times_in_s = time_calib_utils.calibrate_neural_data_time(spike_times_in_s, raw_data_folder_path, ff_caught_T_sorted)
    
    offset_neural_txt = time_calib_utils.find_offset_neural_txt_const(raw_data_folder_path, ff_caught_T_sorted)
    spike_times_in_s = spike_times_in_s - offset_neural_txt


    spike_clusters = _load_spike_clusters(sorted_neural_data_name)

    spike_times_in_s, spike_clusters = _filter_spike_data(spike_times_in_s, spike_clusters, smr_markers_start_time)

    spike_df = pd.DataFrame({'time': spike_times_in_s, 'cluster': spike_clusters})

    return spike_df

def _load_spike_times(sorted_neural_data_path):
    """Load and process spike times."""
    spike_times = np.load(os.path.join(sorted_neural_data_path, "spike_times.npy"))
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


def bin_spikes(spike_df, bin_width=0.25):
    """Bin spikes and stack bins for each spike cluster."""
    min_time = math.floor(spike_df.time.min())
    max_time = math.ceil(spike_df.time.max())
    time_bins = np.arange(min_time, max_time, bin_width) 
    unique_clusters = np.sort(spike_df.cluster.unique())
    all_binned_spikes = np.zeros([len(time_bins)-1, len(unique_clusters)])

    for i in range(len(unique_clusters)):
        cluster = unique_clusters[i]
        spike_subset = spike_df[spike_df['cluster']==cluster]
        binned_spikes, bin_numbers = np.histogram(spike_subset.time, time_bins)
        all_binned_spikes[:, i] = binned_spikes

    # make sure that the number of time bins in all_binned_spikes does not exceed len(time_bins) - 1
    if all_binned_spikes.shape[0] > len(time_bins) - 1:
        all_binned_spikes = all_binned_spikes[:len(time_bins) - 1, :]

    return time_bins, all_binned_spikes


def calculate_window_parameters(window_width, bin_width):
    """Calculate window parameters and ensure num_bins_in_window is odd."""
    num_bins_in_window = int(window_width/bin_width)
    # make sure num_bins_in_window is odd
    if num_bins_in_window % 2 == 0:
        num_bins_in_window += 1
        window_width = num_bins_in_window * bin_width
    convolve_pattern = np.ones(num_bins_in_window)
    return window_width, num_bins_in_window, convolve_pattern


def prepare_binned_spikes_matrix_and_df(all_binned_spikes, max_bin):
    """
    Prepare the binned_spikes_df dataframe by extracting the maximum bin from final_behavioral_data,
    slicing all_binned_spikes, and creating column names.
    """
    if max_bin is None:
        max_bin = all_binned_spikes.shape[0] - 1
    binned_spikes_matrix = all_binned_spikes[:max_bin + 1, :]
    column_names = 'unit_' + pd.Series(range(binned_spikes_matrix.shape[1])).astype(str)
    binned_spikes_df = pd.DataFrame(binned_spikes_matrix, columns=column_names)
    binned_spikes_df['bin'] = np.arange(binned_spikes_matrix.shape[0])
    return binned_spikes_matrix, binned_spikes_df


def convolve_neural_data(x_var, kernel_len=7):
    """
    Convolve neural data in Yizhou's way.
    """
    # Define a b-spline
    knots = np.hstack(([-1.001]*3, np.linspace(-1.001,1.001,5), [1.001]*3))
    tp = np.linspace(-1.,1.,kernel_len)
    bX = splineDesign(knots, tp, ord=4, der=0, outer_ok=False)

    modelX = np.zeros((x_var.shape[0], x_var.shape[1]*bX.shape[1]))
    for neu in range(x_var.shape[1]):
        modelX2 = np.zeros((x_var.shape[0],bX.shape[1]))
        for k in range(bX.shape[1]):
            modelX2[:,k] = np.convolve(x_var[:,neu], bX[:,k],'same')

        modelX[:,neu*bX.shape[1]:(neu+1)*bX.shape[1]] = modelX2

    return modelX


def add_lags_to_each_feature(var, lag_numbers, rearrange_lag_based_on_abs_value=True):
    """
    Add lags to each feature in the given variable.
    """
    n_units = var.shape[1]
    var_lags = np.zeros((var.shape[0], n_units * len(lag_numbers)))
    column_names = None

    if isinstance(var, pd.DataFrame):
        column_names = var.columns.astype(str)
        var = var.values
        new_column_names = np.tile(column_names, len(lag_numbers))

    if rearrange_lag_based_on_abs_value:
        lag_numbers = sorted(lag_numbers, key=lambda x: abs(x))

    for idx, lag in enumerate(lag_numbers):
        columns_numbers = range(idx*n_units, (idx+1)*n_units)
        if column_names is not None:
            new_column_names[columns_numbers] = column_names + "_" + str(lag)

        if lag < 0:
            var_lags[:lag, columns_numbers] = var[-lag:, :]
        elif lag > 0:
            var_lags[lag:, columns_numbers] = var[:-lag, :]
        else:
            var_lags[:, columns_numbers] = var[:, :]

    if column_names is not None:
        var_lags = pd.DataFrame(var_lags, columns=new_column_names)
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
    need_outer = any(x[not_nan] < knots[ord - 1]) or any(x[not_nan] > knots[-ord])

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
        reps_end = max(0, ord - np.where(dkn > 0)[0][0] - 1) if any(dkn > 0) else np.nan
        idx = [0] * (ord - 1) + list(range(len(knots))) + [len(knots) - 1] * reps_end
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
        design[i,in_x] = v[i]

    # Return transposed design matrix
    return design.transpose()

    
def get_mapping_table_between_hard_drive_and_local_folders(monkey_name, hdrive_dir, neural_data_folder_name, filter_neural_file_func, save_table=False):
    all_local_data_paths = []
    all_time_calib_paths = []
    all_hdrive_folders = []
    all_hdrive_data_paths = []

    raw_data_dir = os.path.join('/Users/dusiyi/Documents/Multifirefly-Project/all_monkey_data/raw_monkey_data/', monkey_name)
    neural_data_dir = raw_data_dir.replace('raw_monkey_data', 'neural_data')
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    sessions = os.listdir(raw_data_dir)
    sessions = [f for f in sessions if 'data_' in f]

    for data_dir in sessions:
        local_data_path = os.path.join(neural_data_dir, data_dir)
        neural_event_time_path = os.path.join(local_data_path.replace('neural_data', 'time_calibration'), 'neural_event_time.txt')
    
        data_number = data_dir.split('_')[-1]
        data_name = month_dict[int(data_number[1])] + ' ' + data_number[-2:] + ' 2018'

        all_local_data_paths.append(local_data_path)
        all_time_calib_paths.append(neural_event_time_path)
        hdrive_folder = os.path.join(hdrive_dir, data_name, neural_data_folder_name)
        all_hdrive_folders.append(hdrive_folder)

        #get the neural file path
        result = subprocess.run(['ls', '--color=never', hdrive_folder], capture_output=True, text=True, check=True)
        file_list = result.stdout.splitlines()  # Split into clean file names
        neural_files = filter_neural_file_func(file_list)
        if len(neural_files) != 1:
            raise ValueError(f"There should be exactly one neural file in the directory. Found {len(neural_files)} files. The files in the directory are {file_list}")
        else:
            hdrive_data_path = os.path.join(hdrive_folder, neural_files[0])
            
        all_hdrive_data_paths.append(hdrive_data_path)

    mapping_table = pd.DataFrame({'local_path': all_local_data_paths, 
                            'neural_event_time_path': all_time_calib_paths,
                            'hdrive_path': all_hdrive_data_paths,
                            'hdrive_folder': all_hdrive_folders})

    if save_table:
        monkey = 'bruno' if 'bruno' in monkey_name.lower() else 'schro'
        mapping_table.to_csv(f'/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods/non_behavioral_analysis/neural_data_analysis/get_neural_data/MATLAB_processing/{monkey}_mapping_table.csv',
                            index=False)
    return mapping_table