# Solution to fix GPFA trial length mismatch issue
# The problem is that different trials have different durations, leading to different numbers of time bins
# GPFA requires all trials to have the same number of time bins

import numpy as np


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
        for cluster in range(len(trial)):
            duration = trial[cluster].t_stop.magnitude
            trial_durations.append(duration)

            # Calculate number of bins (assuming 0.02s bin width)
            bin_width = 0.02
            num_bins = int(duration / bin_width)
            trial_lengths.append(num_bins)

    analysis = {
        'num_spiketrains': len(trial_durations),
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
