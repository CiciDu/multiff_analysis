from itertools import combinations

import numpy as np
import pandas as pd

from reinforcement_learning.analyze_agents import analyze_policy_smoothness


DEFAULT_DELTA_VALUES = np.linspace(-0.15, 0.15, 21, dtype=np.float32)


def _get_default_delta_values(delta_values):
    if delta_values is None:
        return DEFAULT_DELTA_VALUES.copy()
    return np.asarray(delta_values, dtype=np.float32)


def _get_value_col_from_value_type(value_type, action_dim=0):
    if value_type == 'raw_action':
        return f'action_{action_dim}'
    if value_type == 'delta_action':
        return f'delta_action_{action_dim}'
    if value_type == 'norm':
        return 'action_change_norm'
    raise ValueError("value_type must be 'raw_action', 'delta_action', or 'norm'.")


def _choose_anchor_indices(
    rollout_df,
    obs_array,
    n_anchors,
    anchor_method,
    random_seed,
    standardize_for_clustering=True,
):
    return analyze_policy_smoothness.choose_anchor_indices(
        rollout_df=rollout_df,
        obs_array=obs_array,
        n_anchors=n_anchors,
        method=anchor_method,
        random_seed=random_seed,
        standardize_for_clustering=standardize_for_clustering,
    )


def _predict_action_from_flat_obs(env, rl_agent, flat_obs, deterministic=True):
    obs_input = analyze_policy_smoothness._rebuild_policy_obs_input_from_flat(
        env,
        rl_agent,
        flat_obs,
    )
    action, _ = rl_agent.predict(obs_input, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _make_2d_sweep_row(
    env,
    action,
    anchor_id,
    anchor_index,
    obs_dim_x,
    obs_dim_y,
    base_obs,
    perturbed_obs,
    delta_x,
    delta_y,
):
    row = {
        'anchor_id': int(anchor_id),
        'anchor_index': int(anchor_index),
        'obs_dim_x': int(obs_dim_x),
        'obs_dim_y': int(obs_dim_y),
        'obs_dim_x_label': analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_x),
        'obs_dim_y_label': analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_y),
        'base_value_x': float(base_obs[obs_dim_x]),
        'base_value_y': float(base_obs[obs_dim_y]),
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
        'perturbed_value_x': float(perturbed_obs[obs_dim_x]),
        'perturbed_value_y': float(perturbed_obs[obs_dim_y]),
    }
    row.update({f'action_{j}': float(action[j]) for j in range(len(action))})
    return row


def _add_baseline_and_action_change_columns(sweep_2d_df):
    action_cols = [col for col in sweep_2d_df.columns if col.startswith('action_')]

    baseline_df = (
        sweep_2d_df[
            np.isclose(sweep_2d_df['delta_x'].values, 0.0)
            & np.isclose(sweep_2d_df['delta_y'].values, 0.0)
        ][['anchor_id'] + action_cols]
        .rename(columns={col: f'baseline_{col}' for col in action_cols})
    )

    sweep_2d_df = sweep_2d_df.merge(baseline_df, on='anchor_id', how='left')

    squared_norm = 0.0
    for action_col in action_cols:
        baseline_col = f'baseline_{action_col}'
        delta_action_col = f'delta_{action_col}'
        sweep_2d_df[delta_action_col] = sweep_2d_df[action_col] - sweep_2d_df[baseline_col]
        squared_norm = squared_norm + sweep_2d_df[delta_action_col] ** 2

    sweep_2d_df['action_change_norm'] = np.sqrt(squared_norm)
    return sweep_2d_df


def evaluate_policy_local_2d_sweeps(
    env,
    rl_agent,
    obs_array,
    anchor_indices,
    obs_dim_x,
    obs_dim_y,
    delta_x_values,
    delta_y_values,
    deterministic=True,
    clip_low=-1.0,
    clip_high=1.0,
):
    '''
    Around each real anchor observation, perturb two observation dimensions
    jointly and re-run the policy.
    '''
    rows = []

    delta_x_values = np.asarray(delta_x_values, dtype=np.float32)
    delta_y_values = np.asarray(delta_y_values, dtype=np.float32)

    for anchor_id, anchor_index in enumerate(anchor_indices):
        base_obs = obs_array[anchor_index].copy()
        base_x = float(base_obs[obs_dim_x])
        base_y = float(base_obs[obs_dim_y])

        for delta_x in delta_x_values:
            for delta_y in delta_y_values:
                perturbed_obs = base_obs.copy()
                perturbed_obs[obs_dim_x] = np.clip(base_x + float(delta_x), clip_low, clip_high)
                perturbed_obs[obs_dim_y] = np.clip(base_y + float(delta_y), clip_low, clip_high)

                action = _predict_action_from_flat_obs(
                    env=env,
                    rl_agent=rl_agent,
                    flat_obs=perturbed_obs,
                    deterministic=deterministic,
                )

                rows.append(
                    _make_2d_sweep_row(
                        env=env,
                        action=action,
                        anchor_id=anchor_id,
                        anchor_index=anchor_index,
                        obs_dim_x=obs_dim_x,
                        obs_dim_y=obs_dim_y,
                        base_obs=base_obs,
                        perturbed_obs=perturbed_obs,
                        delta_x=delta_x,
                        delta_y=delta_y,
                    )
                )

    sweep_2d_df = pd.DataFrame(rows)
    return _add_baseline_and_action_change_columns(sweep_2d_df)


def run_local_policy_2d_sweep_for_one_pair(
    env,
    rl_agent,
    obs_array,
    anchor_indices,
    obs_dim_x,
    obs_dim_y,
    delta_x_values=None,
    delta_y_values=None,
    deterministic=True,
):
    '''
    Run one 2-D local sweep for a specific pair of observation dimensions.
    '''
    delta_x_values = _get_default_delta_values(delta_x_values)
    delta_y_values = _get_default_delta_values(delta_y_values)

    return evaluate_policy_local_2d_sweeps(
        env=env,
        rl_agent=rl_agent,
        obs_array=obs_array,
        anchor_indices=anchor_indices,
        obs_dim_x=obs_dim_x,
        obs_dim_y=obs_dim_y,
        delta_x_values=delta_x_values,
        delta_y_values=delta_y_values,
        deterministic=deterministic,
    )


def run_local_policy_2d_sweeps_for_all_pairs(
    env,
    rl_agent,
    rollout_df,
    obs_array,
    obs_dims,
    n_anchors=12,
    anchor_method='cluster',
    random_seed=42,
    delta_x_values=None,
    delta_y_values=None,
    deterministic=True,
    standardize_for_clustering=True,
):
    '''
    Run 2-D local sweeps for all unique pairs of obs_dims.

    Returns
    -------
    sweep_2d_results : dict
        Keys are (obs_dim_x, obs_dim_y), values are sweep_2d_df.
    anchor_indices : np.ndarray
        Anchor indices used for all pairwise sweeps.
    '''
    if len(obs_dims) < 2:
        raise ValueError('Need at least two obs_dims to form pairs.')

    delta_x_values = _get_default_delta_values(delta_x_values)
    delta_y_values = _get_default_delta_values(delta_y_values)

    anchor_indices = _choose_anchor_indices(
        rollout_df=rollout_df,
        obs_array=obs_array,
        n_anchors=n_anchors,
        anchor_method=anchor_method,
        random_seed=random_seed,
        standardize_for_clustering=standardize_for_clustering,
    )

    sweep_2d_results = {}
    for obs_dim_x, obs_dim_y in combinations(obs_dims, 2):
        sweep_2d_results[(obs_dim_x, obs_dim_y)] = run_local_policy_2d_sweep_for_one_pair(
            env=env,
            rl_agent=rl_agent,
            obs_array=obs_array,
            anchor_indices=anchor_indices,
            obs_dim_x=obs_dim_x,
            obs_dim_y=obs_dim_y,
            delta_x_values=delta_x_values,
            delta_y_values=delta_y_values,
            deterministic=deterministic,
        )

    return sweep_2d_results, anchor_indices


def summarize_local_policy_2d_norm(sweep_2d_results):
    '''
    Summarize 2-D smoothness using action_change_norm across all pairs.
    '''
    rows = []

    for (obs_dim_x, obs_dim_y), sweep_2d_df in sweep_2d_results.items():
        if len(sweep_2d_df) == 0:
            continue

        obs_dim_x_label = sweep_2d_df['obs_dim_x_label'].iloc[0]
        obs_dim_y_label = sweep_2d_df['obs_dim_y_label'].iloc[0]

        rows.append({
            'obs_dim_x': int(obs_dim_x),
            'obs_dim_y': int(obs_dim_y),
            'obs_dim_x_label': obs_dim_x_label,
            'obs_dim_y_label': obs_dim_y_label,
            'pair_label': f'{obs_dim_x_label}__vs__{obs_dim_y_label}',
            'mean_action_change_norm': float(sweep_2d_df['action_change_norm'].mean()),
            'median_action_change_norm': float(sweep_2d_df['action_change_norm'].median()),
            'max_action_change_norm': float(sweep_2d_df['action_change_norm'].max()),
            'n_eval': int(len(sweep_2d_df)),
        })

    summary_2d_df = pd.DataFrame(rows)

    if len(summary_2d_df) > 0:
        summary_2d_df = (
            summary_2d_df
            .sort_values('mean_action_change_norm', ascending=False)
            .reset_index(drop=True)
        )

    return summary_2d_df


def run_local_policy_2d_sweeps_for_same_field_pair_across_slots(
    env,
    rl_agent,
    obs_array,
    anchor_indices,
    field_x,
    field_y,
    slot_indices=None,
    delta_x_values=None,
    delta_y_values=None,
    deterministic=True,
):
    if slot_indices is None:
        slot_indices = range(env.num_obs_ff)

    results = {}

    for slot_index in slot_indices:
        obs_dim_x = analyze_policy_smoothness.get_obs_dim_index(
            env,
            slot_index=slot_index,
            field_name=field_x,
        )
        obs_dim_y = analyze_policy_smoothness.get_obs_dim_index(
            env,
            slot_index=slot_index,
            field_name=field_y,
        )

        results[slot_index] = run_local_policy_2d_sweep_for_one_pair(
            env=env,
            rl_agent=rl_agent,
            obs_array=obs_array,
            anchor_indices=anchor_indices,
            obs_dim_x=obs_dim_x,
            obs_dim_y=obs_dim_y,
            delta_x_values=delta_x_values,
            delta_y_values=delta_y_values,
            deterministic=deterministic,
        )

    return results