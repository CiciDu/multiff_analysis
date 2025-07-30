import sys
import os
import sys
import torch
import numpy as np
import pandas as pd
import math
from math import pi
from torch.linalg import vector_norm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_ff_flash_from_random_sampling(num_alive_ff, duration, non_flashing_interval_mean=3, flash_on_interval=0.3):
    """
    Randomly sample flashing-on durations for each firefly

    Parameters
    ----------
    num_alive_ff: num
        number of fireflies for which flashing-on durations will be sampled
    duration: num
        total length of time
    non_flashing_interval_mean: num
        the mean length of the gap between every two flashing-on intervals
    flash_on_interval: num
        the length of each flashing-on interval

    Returns
    -------
    ff_flash: list
      containing the time that each firefly flashes on and off

    """

    ff_flash = []
    # for each firefly in the 200
    for i in range(num_alive_ff):
        num_intervals = int(duration)
        # eandomly generate a series of intervals that will be the durations between the flashing intervals
        non_flashing_intervals = torch.poisson(
            torch.ones(num_intervals) * non_flashing_interval_mean)
        # also generate a series of intervals, 0.3s each, for the durations of the flashing intervals
        flashing_intervals = torch.ones(num_intervals) * flash_on_interval
        # make a tensor of all the starting-flashing time for the firefly; pretend that time starts from -10s so that
        # the condition at time 0 is more natural
        t0 = torch.cumsum(non_flashing_intervals, dim=0) + \
            torch.cumsum(flashing_intervals, dim=0) - 10
        # Also make a tensor of all the stopping-flashing time for the firefly
        t1 = t0 + flashing_intervals
        # and we should start at the interval where t1 is greater than zero
        meaningful = (t1 > 0)
        t0 = t0[meaningful]
        t1 = t1[meaningful]
        # then, to prevent future errors, make all negative values 0
        t0[t0 < 0] = 0
        ff_flash.append(torch.stack((t0, t1), dim=1))

        if max(t1) < duration:
            raise ValueError(
                'The flashing-on duration is too short for the given duration')
    return ff_flash


def calculate_angles_to_ff_in_pytorch(ffxy, agentx, agenty, agentheading, ff_radius, ffdistance=None):
    """
    Calculate the angle of a firefly from the monkey's or the agent's perspective

    Parameters
    ----------
    ffxy: torch.tensor
        containing the x-coordinates and the y-coordinates of all fireflies
    agentx: torch.tensor, shape (1,)
        the x-coordinate of the agent
    agenty: torch.tensor, shape (1,)
        the y-coordinate of the agent
    agentheading: torch.tensor, shape (1,)
        the angle that the agent heads toward
    ff_radius: num
        the radius of the reward boundary of each firefly  
    ffdistance: torch.tensor, optional
        containing the distances of the fireflies to the agent

    Returns
    -------
    angle_to_center: torch.tensor
        containing the angles of the centers of the fireflies to the agent
    angle_to_boundary: torch.tensor
        containing the smallest angles of the reward boundaries of the fireflies to the agent

    """

    if ffdistance is None:
        agentxy = torch.cat((agentx, agenty))
        ffdistance = vector_norm(ffxy - agentxy, dim=1)
    # find the angles of the given fireflies to the agent
    angle_to_center = torch.atan2(
        ffxy[:, 1] - agenty, ffxy[:, 0] - agentx) - agentheading
    # make sure that the angles are between (-pi, pi]
    angle_to_center = torch.remainder(angle_to_center, 2*pi)
    angle_to_center[angle_to_center >
                    pi] = angle_to_center[angle_to_center > pi] - 2 * pi
    # Adjust the angle based on reward boundary (i.e. find the smallest angle from the agent to the reward boundary)
    # using trignometry
    side_opposite = ff_radius
    # hypotenuse cannot be smaller than side_opposite
    hypotenuse = torch.clamp(ffdistance, min=side_opposite)
    theta = torch.arcsin(torch.div(side_opposite, hypotenuse))
    # we use absolute values of angles here so that the adjustment will only make the angles smaller
    angle_adjusted_abs = torch.abs(angle_to_center) - torch.abs(theta)
    # thus we can find the smallest absolute angle to the firefly, which is the absolute angle to the boundary of the firefly
    angle_to_boundary_abs = torch.clamp(angle_adjusted_abs, min=0)
    # restore the signs of the angles
    angle_to_boundary = torch.sign(angle_to_center) * angle_to_boundary_abs
    return angle_to_center, angle_to_boundary


def update_noisy_ffxy(ffx_noisy, ffy_noisy, ffx, ffy, ff_uncertainty_all, visible_ff_indices):
    """
    Adding noise to the positions of the fireflies based on how long ago they were seen and 
    meanwhile restoring the accurate positions of the currently visible fireflies

    Parameters
    ----------
    ffx_noisy: torch.tensor
        containing the x-coordinates of all fireflies with noise
    ffy_noisy: torch.tensor
        containing the y-coordinates of all fireflies with noise
    ffx: torch.tensor
        containing the accurate x-coordinates of all fireflies
    ffy: torch.tensor
        containing the accurate y-coordinates of all fireflies
    ff_uncertainty_all: torch.tensor
        containing the values of uncertainty of all fireflies; scaling is based on a parameter for the environment
    visible_ff_indices: torch.tensor
        containing the indices of the visible fireflies


    Returns
    -------
    ffx_noisy: torch.tensor
        containing the x-coordinates of all fireflies with noise
    ffy_noisy: torch.tensor
        containing the y-coordinates of all fireflies with noise
    ffxy_noisy: torch.tensor
        containing the x-coordinates and the y-coordinates of all fireflies with noise

    """

    # update the positions of fireflies with uncertainties added; note that uncertainties are cummulative across steps
    num_alive_ff = len(ff_uncertainty_all)
    ffx_noisy = ffx_noisy + \
        torch.normal(torch.zeros([num_alive_ff, ]), ff_uncertainty_all)
    ffy_noisy = ffy_noisy + \
        torch.normal(torch.zeros([num_alive_ff, ]), ff_uncertainty_all)
    # for the visible fireflies, their positions are updated to be the real positions
    ffx_noisy[visible_ff_indices] = ffx[visible_ff_indices].clone()
    ffy_noisy[visible_ff_indices] = ffy[visible_ff_indices].clone()
    ffxy_noisy = torch.stack((ffx_noisy, ffy_noisy), dim=1)
    return ffx_noisy, ffy_noisy, ffxy_noisy


def find_visible_ff(time, ff_distance_all, ff_angle_all, invisible_distance, invisible_angle, ff_flash):
    """
    Find the indices of the fireflies that are visible at a given time

    Parameters
    ----------
    time: num
        the current moment 
    ff_distance_all: torch.tensor
        containing the distances of all the fireflies to the agent
    ff_angle_all: torch.tensor
        containing the angles (to the reward boundaries) of all the fireflies to the agent   
    invisible_distance: num
        the distance beyond which a firefly will be considered invisible
    invisible_angle: num    
        the angle beyond which a firefly will be considered invisible 
    ff_flash: list
      containing the time that each firefly flashes on and off

    Returns
    -------
    visible_ff_indices: torch.tensor
      containing the indices of the fireflies that are visible at the given time

    """

    # find fireflies that are within the visible distance and angle at this point
    visible_ff = torch.logical_and(
        ff_distance_all < invisible_distance, torch.abs(ff_angle_all) < invisible_angle)
    if ff_flash is not None:
        # among these fireflies, eliminate those that are not flashing on at this point
        for index in visible_ff.nonzero().reshape(-1):
            ff_flashing_durations = ff_flash[index].clone().detach()
            # if no interval contains the current time point
            if not torch.any(torch.logical_and(ff_flashing_durations[:, 0] <= time, ff_flashing_durations[:, 1] >= time)):
                visible_ff[index] = False
    visible_ff_indices = visible_ff.nonzero().reshape(-1)
    return visible_ff_indices


# for making ff_array in the belief

def _normalize_ff_array(ff_array, invisible_distance, full_memory, add_memory, add_ff_time_since_start_visible, visible_time_range=None):
    ff_array[0:2, :] = ff_array[0:2, :] / math.pi
    ff_array[2, :] = (ff_array[2, :] / invisible_distance - 0.5) * 2
    if add_ff_time_since_start_visible:
        ff_array[3, :] = (ff_array[3, :] / visible_time_range - 0.5) * 2
    elif add_memory:
        ff_array[3, :] = (ff_array[3, :] / full_memory - 0.5) * 2
    return ff_array


def _normalize_ff_array_for_env2(ff_array, invisible_distance, visible_time_range):
    # If normalizing the observation
    ff_array[0, :] = ff_array[0, :]/math.pi
    ff_array[1, :] = (ff_array[1, :]/invisible_distance-0.5)*2
    ff_array[2, :] = (ff_array[2, :]/visible_time_range-0.5)*2
    return ff_array


def _get_placeholder_ff(add_memory, add_ff_time_since_start_visible, invisible_distance):
    if add_memory | add_ff_time_since_start_visible:
        return torch.tensor([[0.], [0.], [invisible_distance], [0.]])
    else:
        return torch.tensor([[0.], [0.], [invisible_distance]])


def _get_topk_indices(ff_indices, ff_distance_all, num_obs_ff):
    topk_ff = torch.topk(-ff_distance_all[ff_indices], num_obs_ff).indices
    return ff_indices[topk_ff]


def _get_sorted_indices(ff_indices, ff_distance_all):
    _, sorted_indices = torch.sort(-ff_distance_all[ff_indices])
    return ff_indices[sorted_indices]
