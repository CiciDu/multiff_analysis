import sys
from null_behaviors import show_null_trajectory

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from matplotlib import rc
from numpy import random

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def sample_null_distributions_func(print_progress = True, num_samples = 1000, num_trials_per_sample = 1000):
    print_progress = True
    all_median_time = []
    all_median_distance = []
    all_median_abs_angle = []
    all_median_abs_angle_boundary = []

    all_mean_time = []
    all_mean_distance = []
    all_mean_abs_angle = []
    all_mean_abs_angle_boundary = []
    all_total_time = []

    
    # For each of 10000 samples 
    for sample in range(num_samples):
        if print_progress:
            if sample % 100 == 0:
                print(sample, "out of", num_samples)
        # Assume there are 1000 trials; maybe this can change based on the number of trial in the data
        min_time_of_trials = []
        ff_distance_of_min_time_trial = []
        ff_angle_of_min_time_trial = []
        ff_angle_boundary_of_min_time_trial = []
        for trial in range(num_trials_per_sample):
            # Suppose the monkey is at the orgin, sample 200 fireflies following poisson distribution
            num_alive_ff = 200
            arena_radius = 1000
            fftheta = random.rand(num_alive_ff) * 2 * pi
            # Select those whose theta is between 0 and pi, so that their y coordinate is positive
            fftheta = fftheta[fftheta < pi]
            # And then sample a radius for each ff and get the x, y coordinates
            ffr = np.sqrt(random.rand(len(fftheta))) * arena_radius
            ffx = np.cos(fftheta) * ffr
            ffy = np.sin(fftheta) * ffr
            ffxy = np.stack((ffx, ffy), axis=1)
            # Select fireflies whose sum of x and y coordinates is less than a threshold 
            manhattan_distance = np.abs(ffx) + np.abs(ffy)
            threshold = 200
            ff_to_be_considered = np.where(manhattan_distance < threshold)[0]
            # If there are too few, then increase the threshold
            while len(ff_to_be_considered) < 3:
                threshold = threshold+50
                ff_to_be_considered = np.where(manhattan_distance < threshold)[0]
            # For each ff selected, calculate the length of the arc
            min_arc_length, min_arc_radius, min_arc_ff_xy, min_arc_ff_distance, min_arc_ff_angle, \
                    min_arc_ff_angle_boundary = show_null_trajectory.find_shortest_arc_among_all_available_ff(ff_x=ffxy[ff_to_be_considered, 0], ff_y=ffxy[ff_to_be_considered, 1], monkey_x=0, monkey_y=0, monkey_angle=pi/2)

            
            # Divide it by the maximum linear velocity (200 cm/s)
            min_time = min_arc_length/200
            min_time_of_trials.append(min_time)
            ff_distance_of_min_time_trial.append(min_arc_ff_distance)
            ff_angle_of_min_time_trial.append(min_arc_ff_angle)
            ff_angle_boundary_of_min_time_trial.append(min_arc_ff_angle_boundary)
        min_time_of_trials = np.array(min_time_of_trials)
        ff_distance_of_min_time_trial = np.array(ff_distance_of_min_time_trial)
        ff_angle_of_min_time_trial = np.array(ff_angle_of_min_time_trial)
        ff_angle_boundary_of_min_time_trial = np.array(ff_angle_boundary_of_min_time_trial)

       
        all_median_time.append(np.median(min_time_of_trials))
        all_median_distance.append(np.median(ff_distance_of_min_time_trial))
        all_median_abs_angle.append(np.median(ff_angle_of_min_time_trial))
        all_median_abs_angle_boundary.append(np.median(ff_angle_boundary_of_min_time_trial))

        all_mean_time.append(np.mean(min_time_of_trials))
        all_mean_distance.append(np.mean(ff_distance_of_min_time_trial))
        all_mean_abs_angle.append(np.mean(ff_angle_of_min_time_trial))
        all_mean_abs_angle_boundary.append(np.mean(ff_angle_boundary_of_min_time_trial))
        all_total_time.append(np.sum(min_time_of_trials))

    
    all_median_time = np.array(all_median_time)
    all_median_distance = np.array(all_median_distance)
    all_median_abs_angle = np.array(all_median_abs_angle)
    all_median_abs_angle_boundary = np.array(all_median_abs_angle_boundary)

    all_mean_time = np.array(all_mean_time)
    all_mean_distance = np.array(all_mean_distance)
    all_mean_abs_angle = np.array(all_mean_abs_angle)
    all_mean_abs_angle_boundary = np.array(all_mean_abs_angle_boundary)
    all_total_time = np.array(all_total_time)

    return all_median_time, all_median_distance, all_median_abs_angle, all_median_abs_angle_boundary, \
           all_mean_time, all_mean_distance, all_mean_abs_angle, all_mean_abs_angle_boundary, all_total_time

