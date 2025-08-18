

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


def compare_baseline_vs_post_event_from_spikes(
    spike_df,
    pre_window=(-0.3, 0.0),
    post_window=(0.0, 0.3),
    method='wilcoxon',  # or 'ttest'
    time_col='rel_spike_time',
    cluster_col='cluster',
    trial_col='new_segment',
    alpha=0.05,
    use_multipletests=True,
    correction_method='fdr_bh'  # multiple testing correction method
):
    """
    Compare baseline vs. post-event firing rates from raw spike times,
    with multiple testing correction on p-values.

    Parameters:
    - spike_df: DataFrame with 'spike_time_rel', 'cluster_id', and 'trial' columns.
    - pre_window / post_window: Time ranges in seconds.
    - method: 'wilcoxon' or 'ttest'
    - alpha: significance level
    - correction_method: method for multiple testing correction (e.g., 'fdr_bh', 'bonferroni')

    Returns:
    - result_df: DataFrame with per-cluster stats including corrected p-values and significance flags.
    """
    results = []

    pre_duration = pre_window[1] - pre_window[0]
    post_duration = post_window[1] - post_window[0]

    for cluster_id in spike_df[cluster_col].unique():
        cluster_spikes = spike_df[spike_df[cluster_col] == cluster_id]

        pre_counts = (
            cluster_spikes[cluster_spikes[time_col].between(*pre_window)]
            .groupby(trial_col)[time_col].count() / pre_duration
        )
        post_counts = (
            cluster_spikes[cluster_spikes[time_col].between(*post_window)]
            .groupby(trial_col)[time_col].count() / post_duration
        )

        common_trials = pre_counts.index.intersection(post_counts.index)
        pre = pre_counts.loc[common_trials]
        post = post_counts.loc[common_trials]

        if len(common_trials) < 3:
            stat, p = np.nan, np.nan
        else:
            if method == 'ttest':
                stat, p = ttest_rel(post, pre)
            elif method == 'wilcoxon':
                try:
                    stat, p = wilcoxon(post, pre)
                except ValueError:
                    stat, p = np.nan, np.nan
            else:
                raise ValueError("method must be 'ttest' or 'wilcoxon'")

        results.append({
            'cluster_id': cluster_id,
            'mean_pre': pre.mean(),
            'mean_post': post.mean(),
            'mean_diff': post.mean() - pre.mean(),
            'p_value': p,
            'n_trials': len(common_trials)
        })

    result_df = pd.DataFrame(results).sort_values(
        'p_value').reset_index(drop=True)
    if use_multipletests:
        result_df = _correct_p_values(
            result_df, alpha=alpha, method=correction_method)

    return result_df


def _correct_p_values(result_df, alpha=0.05, method='fdr_bh'):
    # Only correct non-NaN p-values
    valid_mask = ~result_df['p_value'].isna()
    corrected = np.full(len(result_df), np.nan)  # fill with NaNs
    significant = np.full(len(result_df), np.nan)
    if valid_mask.sum() > 0:
        reject, p_corr, _, _ = multipletests(
            result_df.loc[valid_mask, 'p_value'], alpha=alpha, method=method)
        corrected[valid_mask] = p_corr
        significant[valid_mask] = reject
    result_df['p_corrected'] = corrected
    result_df['significant'] = significant
    result_df['significant'] = result_df['significant'].astype(bool)
    return result_df
