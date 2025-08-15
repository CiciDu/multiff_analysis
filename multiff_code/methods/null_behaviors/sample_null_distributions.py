import numpy as np
from math import pi
from data_wrangling import specific_utils
from null_behaviors import show_null_trajectory

def _run_single_trial(num_alive_ff=200, arena_radius=1000.0):
    # --- generate fireflies (vectorized) ---
    rand = np.random.rand
    theta = rand(num_alive_ff) * 2 * pi
    theta = theta[theta < pi]  # keep y>0
    r = np.sqrt(rand(theta.size)) * arena_radius
    ffx = np.cos(theta) * r
    ffy = np.sin(theta) * r
    ffxy = np.stack((ffx, ffy), axis=1)

    # --- angle filter (±π/4 about heading π/2) ---
    ff_angle = specific_utils.calculate_angles_to_ff_centers(
        ff_x=ffxy[:, 0], ff_y=ffxy[:, 1], mx=0.0, my=0.0, m_angle=pi/2
    )
    angle_mask = np.abs(ff_angle) <= (pi / 4)
    if not np.any(angle_mask):
        return None

    # --- FAST candidate selection: take k nearest by Manhattan among angle-eligible ---
    # replaces threshold-growing while-loop
    manhattan = np.abs(ffx) + np.abs(ffy)
    eligible_idx = np.flatnonzero(angle_mask)
    if eligible_idx.size < 3:
        return None  # keep behavior predictable: need at least 3

    # find indices of the 3 smallest Manhattan distances among eligible
    k = 3
    d_elig = manhattan[eligible_idx]
    # np.partition is O(N) and avoids full sort
    kth = np.partition(d_elig, k - 1)[k - 1]
    # pick everything up to that kth distance (could be >k due to ties; fine)
    take_mask = d_elig <= kth
    ff_indices = eligible_idx[take_mask]

    # --- compute shortest arc on the candidate subset ---
    (min_arc_length,
     _min_arc_radius,
     _min_arc_ff_xy,
     min_arc_ff_distance,
     min_arc_ff_angle,
     min_arc_ff_angle_boundary) = show_null_trajectory.find_shortest_arc_among_all_available_ff(
        ff_x=ffxy[ff_indices, 0],
        ff_y=ffxy[ff_indices, 1],
        monkey_x=0.0, monkey_y=0.0, monkey_angle=pi/2,
        ignore_error=False
    )

    min_time = min_arc_length / 200.0  # cm / (cm/s) → s
    return min_time, min_arc_ff_distance, min_arc_ff_angle, min_arc_ff_angle_boundary


def _aggregate_trial_data(min_time_of_trials, ff_distance_of_min_time_trial,
                           ff_angle_of_min_time_trial, ff_angle_boundary_of_min_time_trial):
    return (
        np.median(min_time_of_trials),
        np.median(ff_distance_of_min_time_trial),
        np.median(ff_angle_of_min_time_trial),
        np.median(ff_angle_boundary_of_min_time_trial),
        np.mean(min_time_of_trials),
        np.mean(ff_distance_of_min_time_trial),
        np.mean(ff_angle_of_min_time_trial),
        np.mean(ff_angle_boundary_of_min_time_trial),
        np.sum(min_time_of_trials)
    )


def sample_null_distributions_func(print_progress=True, num_samples=1000, num_trials_per_sample=1000):
    all_median_time = []
    all_median_distance = []
    all_median_abs_angle = []
    all_median_abs_angle_boundary = []

    all_mean_time = []
    all_mean_distance = []
    all_mean_abs_angle = []
    all_mean_abs_angle_boundary = []
    all_total_time = []

    for sample in range(num_samples):
        if print_progress and sample % 100 == 0:
            print(sample, "out of", num_samples)

        min_time_of_trials = []
        ff_distance_of_min_time_trial = []
        ff_angle_of_min_time_trial = []
        ff_angle_boundary_of_min_time_trial = []

        # fill exactly num_trials_per_sample; resample trial if it returns None
        i = 0
        attempts_left = num_trials_per_sample * 5  # safety cap
        while i < num_trials_per_sample and attempts_left > 0:
            attempts_left -= 1
            result = _run_single_trial()
            if result is None:
                continue
            min_time, min_arc_ff_distance, min_arc_ff_angle, min_arc_ff_angle_boundary = result
            min_time_of_trials.append(min_time)
            ff_distance_of_min_time_trial.append(min_arc_ff_distance)
            ff_angle_of_min_time_trial.append(min_arc_ff_angle)
            ff_angle_boundary_of_min_time_trial.append(min_arc_ff_angle_boundary)
            i += 1

        min_time_of_trials = np.array(min_time_of_trials)
        ff_distance_of_min_time_trial = np.array(ff_distance_of_min_time_trial)
        ff_angle_of_min_time_trial = np.array(ff_angle_of_min_time_trial)
        ff_angle_boundary_of_min_time_trial = np.array(ff_angle_boundary_of_min_time_trial)

        (med_t, med_d, med_a, med_ab,
         mean_t, mean_d, mean_a, mean_ab,
         total_t) = _aggregate_trial_data(min_time_of_trials,
                                          ff_distance_of_min_time_trial,
                                          ff_angle_of_min_time_trial,
                                          ff_angle_boundary_of_min_time_trial)

        all_median_time.append(med_t)
        all_median_distance.append(med_d)
        all_median_abs_angle.append(med_a)
        all_median_abs_angle_boundary.append(med_ab)
        all_mean_time.append(mean_t)
        all_mean_distance.append(mean_d)
        all_mean_abs_angle.append(mean_a)
        all_mean_abs_angle_boundary.append(mean_ab)
        all_total_time.append(total_t)

    return (
        np.array(all_median_time),
        np.array(all_median_distance),
        np.array(all_median_abs_angle),
        np.array(all_median_abs_angle_boundary),
        np.array(all_mean_time),
        np.array(all_mean_distance),
        np.array(all_mean_abs_angle),
        np.array(all_mean_abs_angle_boundary),
        np.array(all_total_time)
    )
