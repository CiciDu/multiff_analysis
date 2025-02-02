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
