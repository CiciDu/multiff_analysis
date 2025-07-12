# Solution to fix GPFA trial length mismatch issue
# The problem is that different trials have different durations, leading to different numbers of time bins
# GPFA requires all trials to have the same number of time bins

import numpy as np
import pandas as pd
import quantities as pq
import neo


def fix_spiketrains_for_gpfa(spiketrains, target_length=None, method='truncate'):
    """
    Fix spiketrains to have consistent trial lengths for GPFA.

    Parameters:
    -----------
    spiketrains : list of lists of neo.SpikeTrain
        List of trials, where each trial is a list of spiketrains
    target_length : float, optional
        Target duration in seconds. If None, uses the minimum trial duration
    method : str
        'truncate': truncate all trials to the shortest trial length
        'pad': pad shorter trials with zeros
        'min_duration': use minimum trial duration as target

    Returns:
    --------
    fixed_spiketrains : list of lists of neo.SpikeTrain
        Spiketrains with consistent trial lengths
    """

    if method == 'min_duration':
        # Find the minimum trial duration
        trial_durations = []
        for trial in spiketrains:
            if len(trial) > 0:
                trial_durations.append(trial[0].t_stop.magnitude)

        if not trial_durations:
            raise ValueError("No valid trials found")

        target_length = min(trial_durations)
        print(f"Using minimum trial duration: {target_length:.3f} seconds")

    elif target_length is None:
        raise ValueError(
            "target_length must be specified unless method='min_duration'")

    fixed_spiketrains = []

    for trial_idx, trial in enumerate(spiketrains):
        if len(trial) == 0:
            print(f"Warning: Trial {trial_idx} has no spiketrains, skipping")
            continue

        trial_duration = trial[0].t_stop.magnitude

        if method == 'truncate':
            # Truncate trial to target length
            if trial_duration > target_length:
                fixed_trial = []
                for spiketrain in trial:
                    # Keep only spikes within the target duration
                    valid_spikes = spiketrain.times[spiketrain.times <
                                                    target_length * pq.s]
                    fixed_spiketrain = neo.SpikeTrain(
                        times=valid_spikes,
                        t_start=0,
                        t_stop=target_length * pq.s
                    )
                    fixed_trial.append(fixed_spiketrain)
                fixed_spiketrains.append(fixed_trial)
            else:
                # Trial is already shorter than target, keep as is
                fixed_spiketrains.append(trial)

        elif method == 'pad':
            # Pad trial to target length (add zeros)
            if trial_duration < target_length:
                fixed_trial = []
                for spiketrain in trial:
                    # Keep original spikes, extend t_stop
                    fixed_spiketrain = neo.SpikeTrain(
                        times=spiketrain.times,
                        t_start=0,
                        t_stop=target_length * pq.s
                    )
                    fixed_trial.append(fixed_spiketrain)
                fixed_spiketrains.append(fixed_trial)
            else:
                # Trial is already longer than target, truncate
                fixed_trial = []
                for spiketrain in trial:
                    valid_spikes = spiketrain.times[spiketrain.times <
                                                    target_length * pq.s]
                    fixed_spiketrain = neo.SpikeTrain(
                        times=valid_spikes,
                        t_start=0,
                        t_stop=target_length * pq.s
                    )
                    fixed_trial.append(fixed_spiketrain)
                fixed_spiketrains.append(fixed_trial)

    return fixed_spiketrains


def analyze_trial_lengths(spiketrains):
    """
    Analyze trial lengths to help diagnose the issue.

    Parameters:
    -----------
    spiketrains : list of lists of neo.SpikeTrain

    Returns:
    --------
    analysis : dict
        Dictionary with trial length statistics
    """
    trial_durations = []
    trial_lengths = []

    for trial_idx, trial in enumerate(spiketrains):
        if len(trial) > 0:
            duration = trial[0].t_stop.magnitude
            trial_durations.append(duration)

            # Calculate number of bins (assuming 0.02s bin width)
            bin_width = 0.02
            num_bins = int(duration / bin_width)
            trial_lengths.append(num_bins)

    analysis = {
        'num_trials': len(trial_durations),
        'min_duration': min(trial_durations) if trial_durations else None,
        'max_duration': max(trial_durations) if trial_durations else None,
        'mean_duration': np.mean(trial_durations) if trial_durations else None,
        'std_duration': np.std(trial_durations) if trial_durations else None,
        'min_bins': min(trial_lengths) if trial_lengths else None,
        'max_bins': max(trial_lengths) if trial_lengths else None,
        'mean_bins': np.mean(trial_lengths) if trial_lengths else None,
        'std_bins': np.std(trial_lengths) if trial_lengths else None,
        'unique_bin_counts': list(set(trial_lengths)) if trial_lengths else []
    }

    return analysis


# Example usage in your notebook:
"""
# First, analyze the trial lengths
analysis = analyze_trial_lengths(pn.spiketrains)
print("Trial length analysis:")
for key, value in analysis.items():
    print(f"  {key}: {value}")

# Fix the spiketrains using minimum duration
pn.spiketrains_fixed = fix_spiketrains_for_gpfa(pn.spiketrains, method='min_duration')

# Now try GPFA again
pn.spiketrains = pn.spiketrains_fixed
pn.get_gpfa_traj(latent_dimensionality=2, exists_ok=False)
"""
