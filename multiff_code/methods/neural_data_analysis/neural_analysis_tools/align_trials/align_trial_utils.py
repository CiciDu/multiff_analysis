from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
import numpy as np
import pandas as pd


def get_rebinned_var_lags(rebinned_var, trial_vector, lag_numbers=None, rebinned_max_lag_number=2):
    if lag_numbers is None:
        lag_numbers = np.arange(-rebinned_max_lag_number,
                                rebinned_max_lag_number+1)
    else:
        print(f"Using provided lag numbers for rebinned_var_lags: {lag_numbers}")
        
        
    rebinned_var_lags = neural_data_processing.add_lags_to_each_feature(
        rebinned_var, lag_numbers, trial_vector=trial_vector)

    if 'new_bin_0' in rebinned_var_lags.columns:
        rebinned_var_lags['new_bin'] = rebinned_var_lags['new_bin_0'].astype(
            int)
        rebinned_var_lags = rebinned_var_lags.drop(
            columns=[col for col in rebinned_var_lags.columns if 'new_bin_' in col])
    if 'new_segment_0' in rebinned_var_lags.columns:
        rebinned_var_lags['new_segment'] = rebinned_var_lags['new_segment_0'].astype(
            int)
        rebinned_var_lags = rebinned_var_lags.drop(
            columns=[col for col in rebinned_var_lags.columns if 'new_segment_' in col])

    assert rebinned_var_lags['new_bin'].equals(
        rebinned_var['new_bin'])

    return rebinned_var_lags


def check_rank_of_spike_counts_matrix(spike_counts_matrix):
    rank = np.linalg.matrix_rank(spike_counts_matrix)
    n_neurons = spike_counts_matrix.shape[1]
    if rank < n_neurons:
        insufficient_rank = n_neurons - rank
        print(
            f"Rank deficiency detected: {insufficient_rank} redundant neurons")
        print(f"Matrix shape: {spike_counts_matrix.shape}, Rank: {rank}")
        return insufficient_rank
    else:
        # print("Matrix is full rank.")
        return None


def drop_redundant_neurons(spike_counts_per_segment, svd_threshold=1e-10):
    u, s, vh = np.linalg.svd(spike_counts_per_segment.values.T)
    singular_values_sorted_idx = np.argsort(s)

    print("Smallest singular values:")
    print(s[singular_values_sorted_idx[:5]])

    # If smallest singular value is above threshold, stop dropping
    if s[singular_values_sorted_idx[0]] > svd_threshold:
        print("No singular values below threshold; stopping.")
        return spike_counts_per_segment, False

    redundant_direction = u[:, singular_values_sorted_idx[0]]
    most_aligned_neuron = np.argmax(np.abs(redundant_direction))
    redundant_neuron_name = spike_counts_per_segment.columns[most_aligned_neuron]

    print(f"Dropping redundant neuron: {redundant_neuron_name}")

    spike_counts_per_segment_dropped = spike_counts_per_segment.drop(
        columns=redundant_neuron_name)
    return spike_counts_per_segment_dropped, redundant_neuron_name, True


def drop_redundant_neurons_from_concat_raw_spike_data(concat_raw_spike_data):
    spike_counts_per_segment = concat_raw_spike_data.groupby(
        'new_segment').sum()
    spike_counts_per_segment = spike_counts_per_segment[[
        col for col in spike_counts_per_segment.columns if 'cluster_' in col]]

    spike_counts_matrix = spike_counts_per_segment.values
    num_redundant_neurons = check_rank_of_spike_counts_matrix(
        spike_counts_matrix)

    dropped_neurons = []
    while num_redundant_neurons is not None:
        spike_counts_per_segment, redundant_neuron_name, dropped = drop_redundant_neurons(
            spike_counts_per_segment)
        dropped_neurons.append(redundant_neuron_name)
        if not dropped:
            break
        spike_counts_matrix = spike_counts_per_segment.values
        num_redundant_neurons = check_rank_of_spike_counts_matrix(
            spike_counts_matrix)

    concat_raw_spike_data = concat_raw_spike_data.drop(columns=dropped_neurons)
    return concat_raw_spike_data, dropped_neurons


def turn_spiketrains_into_df(spiketrains, cluster_ids):
    rows = []
    for segment_idx, segment in enumerate(spiketrains):  # over segments
        for neuron_idx, st in enumerate(segment):  # over neurons in segment
            cluster = cluster_ids[neuron_idx]
            for spike_time in st.times:
                rows.append({
                    'new_segment': segment_idx,
                    'cluster': cluster,
                    'time': spike_time.rescale('s').magnitude
                })
    df_spikes = pd.DataFrame(rows)
    return df_spikes
