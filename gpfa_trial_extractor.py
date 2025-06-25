"""
GPFA Trial Extractor: Extract GPFA trials from raw spike data

This module provides utilities to:
1. Load raw spike data
2. Prepare spike trains for GPFA
3. Run GPFA to get latent trajectories
4. Extract GPFA trials for specific segments/trials

Usage:
    # Using the class-based approach (recommended)
    extractor = GPFATrialExtractor(raw_data_path, behavioral_data)
    gpfa_neural_trials = extractor.get_gpfa_neural_trials(latent_dims=10)
    
    # Using standalone functions
    spike_df = load_raw_spike_data(raw_data_path)
    spiketrains = prepare_spiketrains_for_gpfa(spike_df, behavioral_data)
    trajectories = run_gpfa(spiketrains, latent_dims=10)
    gpfa_neural_trials = extract_gpfa_neural_trials(trajectories, segments, trial_lengths)
"""

from data_wrangling import time_calib_utils
from non_behavioral_analysis.neural_data_analysis.gpfa_methods import fit_gpfa_utils, gpfa_regression_utils
from non_behavioral_analysis.neural_data_analysis.get_neural_data import neural_data_processing
import numpy as np
import pandas as pd
import os
import pickle
from elephant.gpfa import GPFA
import quantities as pq
import neo
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import your existing utilities
import sys
sys.path.append(
    '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')


class GPFATrialExtractor:
    """
    Class to extract GPFA trials from raw spike data.

    This class handles the complete pipeline from raw spike data to GPFA trials.
    """

    def __init__(self, raw_data_folder_path, behavioral_data, sampling_rate=20000,
                 bin_width=0.02, align_at_beginning=True):
        """
        Initialize the GPFA trial extractor.

        Parameters:
        -----------
        raw_data_folder_path : str
            Path to the raw data folder containing spike data
        behavioral_data : pd.DataFrame
            Behavioral data with segment information (must have 'segment', 'seg_start_time', 'seg_end_time' columns)
        sampling_rate : int
            Sampling rate of the neural data (default: 20000 Hz)
        bin_width : float
            Bin width for spike binning in seconds (default: 0.02 s)
        align_at_beginning : bool
            Whether to align trials at the beginning (True) or end (False)
        """
        self.raw_data_folder_path = raw_data_folder_path
        self.behavioral_data = behavioral_data
        self.sampling_rate = sampling_rate
        self.bin_width = bin_width
        self.bin_width_w_unit = bin_width * pq.s
        self.align_at_beginning = align_at_beginning

        # Initialize storage
        self.spike_df = None
        self.spike_segs_df = None
        self.spiketrains = None
        self.spiketrain_corr_segs = None
        self.trajectories = None
        self.gpfa_neural_trials = None
        self.behav_trials = None

    def load_raw_spike_data(self):
        """
        Load raw spike data from the neural data folder.

        Returns:
        --------
        pd.DataFrame : Spike data with 'time' and 'cluster' columns
        """
        print("Loading raw spike data...")

        # Get the neural data path
        neural_data_path = self.raw_data_folder_path.replace(
            'raw_monkey_data', 'neural_data')
        sorted_neural_data_name = os.path.join(neural_data_path, "Sorted")

        # Load spike times and clusters
        spike_times = np.load(os.path.join(
            sorted_neural_data_name, "spike_times.npy")).reshape(-1)
        spike_clusters = np.load(os.path.join(
            sorted_neural_data_name, "spike_clusters.npy")).reshape(-1)

        # Convert to seconds
        spike_times_in_s = spike_times / self.sampling_rate

        # Apply time calibration if needed
        # spike_times_in_s = time_calib_utils.calibrate_neural_data_time(
        #     spike_times_in_s, self.raw_data_folder_path, ff_caught_T_sorted)

        # Create spike dataframe
        self.spike_df = pd.DataFrame({
            'time': spike_times_in_s,
            'cluster': spike_clusters
        })

        print(
            f"Loaded {len(self.spike_df)} spikes from {self.spike_df['cluster'].nunique()} clusters")
        return self.spike_df

    def prepare_spike_segments(self):
        """
        Prepare spike segments based on behavioral data time windows.

        Returns:
        --------
        pd.DataFrame : Spike segments with trial information
        """
        print("Preparing spike segments...")

        if self.spike_df is None:
            self.load_raw_spike_data()

        # Create spike segments
        spike_segments = []

        # Filter behavioral data to only include segments with duration > 0
        behavioral_data_sub = self.behavioral_data[self.behavioral_data['seg_duration'] > 0]

        for index, row in behavioral_data_sub.iterrows():
            # Get spikes within the time window
            mask = self.spike_df['time'].between(
                row['seg_start_time'], row['seg_end_time'])
            spikes_sub = self.spike_df[mask].copy()

            # Add segment information
            spikes_sub['segment'] = row['segment']
            spikes_sub['seg_start_time'] = row['seg_start_time']
            spikes_sub['seg_end_time'] = row['seg_end_time']
            spikes_sub['seg_duration'] = row['seg_duration']
            spike_segments.append(spikes_sub)

        self.spike_segs_df = pd.concat(spike_segments, ignore_index=True)
        self.spike_segs_df['t_duration'] = self.spike_segs_df['seg_end_time'] - \
            self.spike_segs_df['seg_start_time']

        print(f"Created {len(spike_segments)} spike segments")
        return self.spike_segs_df

    def create_spiketrains(self):
        """
        Convert spike segments to Neo spiketrain objects for GPFA.

        Returns:
        --------
        tuple : (spiketrains, spiketrain_corr_segs)
        """
        print("Creating spiketrains for GPFA...")

        if self.spike_segs_df is None:
            self.prepare_spike_segments()

        # Get unique clusters and segments
        clusters = self.spike_segs_df['cluster'].unique()
        segments = self.spike_segs_df['segment'].unique()

        # Calculate common stop time
        self.common_t_stop = max(
            self.spike_segs_df['t_duration']) + self.bin_width

        # Create spiketrain objects
        spiketrains = []
        spiketrain_corr_segs = []

        for seg in segments:
            # Get data for this segment
            spike_df_trial = self.spike_segs_df[self.spike_segs_df['segment'] == seg]
            seg_start_time = spike_df_trial['seg_start_time'].iloc[0]

            seg_spiketrain = []

            for cluster in clusters:
                # Get spikes for this cluster in this segment
                sub = spike_df_trial[spike_df_trial['cluster'] == cluster]

                # Calculate relative spike times
                spike_time = sub['time'] - seg_start_time

                if not self.align_at_beginning:
                    padding_at_beginning = self.common_t_stop - \
                        spike_df_trial['seg_duration'].iloc[0]
                    spike_time = spike_time + padding_at_beginning

                # Create SpikeTrain object
                spiketrain = neo.SpikeTrain(
                    times=spike_time.values * pq.s,
                    t_start=0,
                    t_stop=self.common_t_stop * pq.s
                )
                seg_spiketrain.append(spiketrain)

            spiketrains.append(seg_spiketrain)
            spiketrain_corr_segs.append(seg)

        self.spiketrains = spiketrains
        self.spiketrain_corr_segs = np.array(spiketrain_corr_segs)

        print(
            f"Created {len(spiketrains)} spiketrains for {len(segments)} segments")
        return self.spiketrains, self.spiketrain_corr_segs

    def run_gpfa(self, latent_dimensionality=10, save_trajectories=True, trajectories_path=None):
        """
        Run GPFA to extract latent trajectories.

        Parameters:
        -----------
        latent_dimensionality : int
            Number of latent dimensions
        save_trajectories : bool
            Whether to save trajectories to file
        trajectories_path : str
            Path to save trajectories (if None, auto-generated)

        Returns:
        --------
        list : GPFA trajectories
        """
        print(f"Running GPFA with {latent_dimensionality} dimensions...")

        if self.spiketrains is None:
            self.create_spiketrains()

        # Generate trajectory path if not provided
        if trajectories_path is None:
            alignment = 'segStart' if self.align_at_beginning else 'segEnd'
            file_name = f'gpfa_neural_aligned_{alignment}_d{latent_dimensionality}.pkl'
            trajectories_path = os.path.join(
                self.raw_data_folder_path, file_name)

        # Try to load existing trajectories
        if save_trajectories and os.path.exists(trajectories_path):
            try:
                with open(trajectories_path, 'rb') as f:
                    self.trajectories = pickle.load(f)
                print(f'Loaded GPFA trajectories from {trajectories_path}')
                return self.trajectories
            except Exception as e:
                print(f'Failed to load trajectories: {str(e)}. Recomputing...')

        # Run GPFA
        gpfa_model = GPFA(
            bin_size=self.bin_width_w_unit,
            x_dim=latent_dimensionality
        )
        self.trajectories = gpfa_model.fit_transform(self.spiketrains)

        # Save trajectories
        if save_trajectories:
            try:
                with open(trajectories_path, 'wb') as f:
                    pickle.dump(self.trajectories, f)
                print(f'Saved GPFA trajectories to {trajectories_path}')
            except Exception as e:
                print(f'Warning: Failed to save trajectories: {str(e)}')

        return self.trajectories

    def extract_gpfa_neural_trials(self, behavioral_variables, use_lags=False):
        """
        Extract GPFA trials for specific behavioral variables.

        Parameters:
        -----------
        behavioral_variables : pd.DataFrame
            Behavioral variables with 'segment' column
        use_lags : bool
            Whether to use lagged behavioral variables

        Returns:
        --------
        tuple : (gpfa_neural_trials, behav_trials)
        """
        print("Extracting GPFA trials...")

        if self.trajectories is None:
            raise ValueError(
                "Must run GPFA first. Call run_gpfa() before extract_gpfa_neural_trials()")

        self.behav_trials = []
        self.gpfa_neural_trials = []

        # Prepare behavioral variables
        if use_lags and hasattr(behavioral_variables, 'y_var_lags_reduced'):
            gpfa_behav_data = behavioral_variables.y_var_lags_reduced.copy()
            if 'segment_0' in gpfa_behav_data.columns:
                gpfa_behav_data.drop(columns=['segment_0'], inplace=True)
        else:
            gpfa_behav_data = behavioral_variables.copy()

        # Add segment information
        if 'segment' not in gpfa_behav_data.columns:
            gpfa_behav_data['segment'] = behavioral_variables['segment'].values

        # Find shared segments
        segments_behav = gpfa_behav_data['segment'].unique()
        segments_behav = segments_behav[segments_behav != '']
        segments_neural = self.spiketrain_corr_segs
        shared_segments = [
            seg for seg in segments_behav.tolist() if seg in segments_neural.tolist()]

        print(
            f"Found {len(shared_segments)} shared segments between neural and behavioral data")

        # Extract trials
        for seg in shared_segments:
            # Get behavioral data for this segment
            gpfa_behav_data_sub = gpfa_behav_data[gpfa_behav_data['segment'] == seg]
            self.behav_trials.append(gpfa_behav_data_sub.values)

            # Get GPFA trial
            trial_length = gpfa_behav_data_sub.shape[0]
            gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
                self.trajectories, seg, trial_length, self.spiketrain_corr_segs,
                align_at_beginning=self.align_at_beginning
            )
            self.gpfa_neural_trials.append(gpfa_trial)

        print(f"Extracted {len(self.gpfa_neural_trials)} GPFA trials")
        return self.gpfa_neural_trials, self.behav_trials

    def get_gpfa_neural_trials(self, latent_dims=10, behavioral_variables=None, use_lags=False):
        """
        Complete pipeline to get GPFA trials from raw spike data.

        Parameters:
        -----------
        latent_dims : int
            Number of latent dimensions for GPFA
        behavioral_variables : pd.DataFrame
            Behavioral variables (if None, uses self.behavioral_data)
        use_lags : bool
            Whether to use lagged behavioral variables

        Returns:
        --------
        tuple : (gpfa_neural_trials, behav_trials)
        """
        # Run the complete pipeline
        self.load_raw_spike_data()
        self.prepare_spike_segments()
        self.create_spiketrains()
        self.run_gpfa(latent_dimensionality=latent_dims)

        if behavioral_variables is None:
            behavioral_variables = self.behavioral_data

        return self.extract_gpfa_neural_trials(behavioral_variables, use_lags=use_lags)

    def plot_trial_summary(self, n_trials=5):
        """
        Plot summary of GPFA trials.

        Parameters:
        -----------
        n_trials : int
            Number of trials to plot
        """
        if self.gpfa_neural_trials is None or self.behav_trials is None:
            print("No trials available. Run get_gpfa_neural_trials() first.")
            return

        n_trials = min(n_trials, len(self.gpfa_neural_trials))

        fig, axes = plt.subplots(2, n_trials, figsize=(4*n_trials, 8))
        if n_trials == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_trials):
            # Plot GPFA trial
            axes[0, i].plot(self.gpfa_neural_trials[i])
            axes[0, i].set_title(f'GPFA Trial {i+1}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Latent Dim')

            # Plot behavioral trial
            axes[1, i].plot(self.behav_trials[i])
            axes[1, i].set_title(f'Behavior Trial {i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Behavior')

        plt.tight_layout()
        plt.show()


# Standalone functions for direct use
def load_raw_spike_data(raw_data_folder_path, sampling_rate=20000):
    """
    Load raw spike data from neural data folder.

    Parameters:
    -----------
    raw_data_folder_path : str
        Path to raw data folder
    sampling_rate : int
        Sampling rate of neural data

    Returns:
    --------
    pd.DataFrame : Spike data with 'time' and 'cluster' columns
    """
    extractor = GPFATrialExtractor(raw_data_folder_path, None, sampling_rate)
    return extractor.load_raw_spike_data()


def prepare_spiketrains_for_gpfa(spike_df, behavioral_data, bin_width=0.02, align_at_beginning=True):
    """
    Prepare spiketrains for GPFA from spike data and behavioral data.

    Parameters:
    -----------
    spike_df : pd.DataFrame
        Spike data with 'time' and 'cluster' columns
    behavioral_data : pd.DataFrame
        Behavioral data with segment information
    bin_width : float
        Bin width in seconds
    align_at_beginning : bool
        Whether to align at beginning

    Returns:
    --------
    tuple : (spiketrains, spiketrain_corr_segs)
    """
    extractor = GPFATrialExtractor(
        None, behavioral_data, bin_width=bin_width, align_at_beginning=align_at_beginning)
    extractor.spike_df = spike_df
    extractor.prepare_spike_segments()
    return extractor.create_spiketrains()


def run_gpfa(spiketrains, latent_dims=10, bin_width=0.02):
    """
    Run GPFA on spiketrains.

    Parameters:
    -----------
    spiketrains : list
        List of Neo spiketrain objects
    latent_dims : int
        Number of latent dimensions
    bin_width : float
        Bin width in seconds

    Returns:
    --------
    list : GPFA trajectories
    """
    bin_width_w_unit = bin_width * pq.s
    gpfa_model = GPFA(bin_size=bin_width_w_unit, x_dim=latent_dims)
    trajectories = gpfa_model.fit_transform(spiketrains)
    return trajectories


def extract_gpfa_neural_trials(trajectories, segments, trial_lengths, spiketrain_corr_segs, align_at_beginning=True):
    """
    Extract GPFA trials for specific segments.

    Parameters:
    -----------
    trajectories : list
        GPFA trajectories
    segments : list
        Segment identifiers
    trial_lengths : list
        Length of each trial
    spiketrain_corr_segs : array
        Segment identifiers for spiketrains
    align_at_beginning : bool
        Whether to align at beginning

    Returns:
    --------
    list : GPFA trials
    """
    gpfa_neural_trials = []

    for seg, trial_length in zip(segments, trial_lengths):
        gpfa_trial = gpfa_regression_utils.get_latent_neural_data_for_trial(
            trajectories, seg, trial_length, spiketrain_corr_segs, align_at_beginning
        )
        gpfa_neural_trials.append(gpfa_trial)

    return gpfa_neural_trials


# Example usage
def example_usage():
    """
    Example of how to use the GPFA trial extractor.
    """
    # Example data paths (replace with your actual paths)
    raw_data_path = "/path/to/your/raw/data"
    behavioral_data = pd.DataFrame({
        'segment': ['trial_1', 'trial_2', 'trial_3'],
        'seg_start_time': [0.0, 10.0, 20.0],
        'seg_end_time': [5.0, 15.0, 25.0],
        'seg_duration': [5.0, 5.0, 5.0]
    })

    # Method 1: Using the class (recommended)
    extractor = GPFATrialExtractor(raw_data_path, behavioral_data)
    gpfa_neural_trials, behav_trials = extractor.get_gpfa_neural_trials(
        latent_dims=10)

    # Method 2: Using standalone functions
    spike_df = load_raw_spike_data(raw_data_path)
    spiketrains, spiketrain_corr_segs = prepare_spiketrains_for_gpfa(
        spike_df, behavioral_data)
    trajectories = run_gpfa(spiketrains, latent_dims=10)

    # Extract trials for specific segments
    segments = ['trial_1', 'trial_2']
    trial_lengths = [50, 50]  # Number of timepoints per trial
    gpfa_neural_trials = extract_gpfa_neural_trials(
        trajectories, segments, trial_lengths, spiketrain_corr_segs)

    return gpfa_neural_trials, behav_trials


if __name__ == "__main__":
    # Run example
    gpfa_neural_trials, behav_trials = example_usage()
    print(f"Extracted {len(gpfa_neural_trials)} GPFA trials")
    print(
        f"Each trial has shape: {gpfa_neural_trials[0].shape if gpfa_neural_trials else 'No trials'}")
