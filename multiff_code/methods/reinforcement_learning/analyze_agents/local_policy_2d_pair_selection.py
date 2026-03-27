from itertools import combinations

import pandas as pd

from reinforcement_learning.analyze_agents import analyze_policy_smoothness

from reinforcement_learning.analyze_agents.local_policy_2d_core import (
    _choose_anchor_indices,
    run_local_policy_2d_sweep_for_one_pair,
)


def _parse_slot_feature_label(obs_dim_label):
    '''
    Parse labels like:
        slot_0_d_log
        slot_1_sin
        prev_action_0
        prev_action_1
        ego_0  (env uses this for add_action_to_obs)
        ego_1
    '''
    parts = obs_dim_label.split('_')

    if len(parts) >= 3 and parts[0] == 'slot' and parts[1].isdigit():
        return {
            'kind': 'slot_feature',
            'slot_id': int(parts[1]),
            'feature_name': '_'.join(parts[2:]),
        }

    if len(parts) >= 3 and parts[0] == 'prev' and parts[1] == 'action' and parts[2].isdigit():
        return {
            'kind': 'prev_action',
            'action_id': int(parts[2]),
        }

    if len(parts) >= 2 and parts[0] == 'ego' and parts[1].isdigit():
        return {
            'kind': 'prev_action',
            'action_id': int(parts[1]),
        }

    return {'kind': 'other'}


def build_structured_obs_dim_pairs(
    env,
    obs_dims,
    include_same_slot_different_feature=True,
    include_same_feature_different_slots=True,
    include_last_two_actions=True,
):
    '''
    Build selected 2-D pairs using observation labels.
    '''
    obs_info_rows = []

    for obs_dim in obs_dims:
        obs_dim_label = analyze_policy_smoothness.get_obs_dim_label(env, obs_dim)
        parsed_info = _parse_slot_feature_label(obs_dim_label)

        obs_info_rows.append({
            'obs_dim': int(obs_dim),
            'obs_dim_label': obs_dim_label,
            **parsed_info,
        })

    obs_info_df = pd.DataFrame(obs_info_rows)

    selected_pairs = []
    pair_rows = []
    seen_pairs = set()

    def add_pair(obs_dim_x, obs_dim_y, pair_category):
        pair_key = tuple(sorted((int(obs_dim_x), int(obs_dim_y))))
        if pair_key in seen_pairs:
            return

        seen_pairs.add(pair_key)

        label_x = analyze_policy_smoothness.get_obs_dim_label(env, pair_key[0])
        label_y = analyze_policy_smoothness.get_obs_dim_label(env, pair_key[1])

        selected_pairs.append(pair_key)
        pair_rows.append({
            'obs_dim_x': int(pair_key[0]),
            'obs_dim_y': int(pair_key[1]),
            'obs_dim_x_label': label_x,
            'obs_dim_y_label': label_y,
            'pair_category': pair_category,
        })

    slot_feature_df = obs_info_df[obs_info_df['kind'] == 'slot_feature'].copy()

    if include_same_slot_different_feature:
        for _, slot_df in slot_feature_df.groupby('slot_id'):
            slot_obs_dims = sorted(slot_df['obs_dim'].tolist())
            for obs_dim_x, obs_dim_y in combinations(slot_obs_dims, 2):
                add_pair(
                    obs_dim_x=obs_dim_x,
                    obs_dim_y=obs_dim_y,
                    pair_category='same_slot_different_feature',
                )

    if include_same_feature_different_slots:
        for _, feature_df in slot_feature_df.groupby('feature_name'):
            feature_obs_dims = sorted(feature_df['obs_dim'].tolist())
            for obs_dim_x, obs_dim_y in combinations(feature_obs_dims, 2):
                add_pair(
                    obs_dim_x=obs_dim_x,
                    obs_dim_y=obs_dim_y,
                    pair_category='same_feature_different_slots',
                )

    if include_last_two_actions:
        prev_action_df = obs_info_df[obs_info_df['kind'] == 'prev_action'].copy()
        if len(prev_action_df) >= 2 and 'action_id' in prev_action_df.columns:
            prev_action_df = prev_action_df.sort_values('action_id').reset_index(drop=True)
        else:
            prev_action_df = prev_action_df.reset_index(drop=True)

        if len(prev_action_df) >= 2:
            if 'action_id' in prev_action_df.columns:
                obs_dim_by_action = dict(zip(prev_action_df['action_id'], prev_action_df['obs_dim']))
                if 0 in obs_dim_by_action and 1 in obs_dim_by_action:
                    add_pair(
                        obs_dim_x=obs_dim_by_action[0],
                        obs_dim_y=obs_dim_by_action[1],
                        pair_category='last_two_actions',
                    )
                else:
                    add_pair(
                        obs_dim_x=prev_action_df['obs_dim'].iloc[0],
                        obs_dim_y=prev_action_df['obs_dim'].iloc[1],
                        pair_category='last_two_actions',
                    )
            else:
                add_pair(
                    obs_dim_x=prev_action_df['obs_dim'].iloc[0],
                    obs_dim_y=prev_action_df['obs_dim'].iloc[1],
                    pair_category='last_two_actions',
                )

    pair_summary_df = pd.DataFrame(pair_rows)

    if len(pair_summary_df) > 0:
        pair_summary_df = (
            pair_summary_df
            .sort_values(['pair_category', 'obs_dim_x', 'obs_dim_y'])
            .reset_index(drop=True)
        )

    return selected_pairs, pair_summary_df


def run_local_policy_2d_sweeps_for_selected_pairs(
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
    include_same_slot_different_feature=True,
    include_same_feature_different_slots=True,
    include_last_two_actions=True,
):
    '''
    Run 2-D local sweeps only for structured pairs:
    - different firefly features of the same slot
    - same feature across different slots
    - last two actions
    '''
    selected_pairs, pair_summary_df = build_structured_obs_dim_pairs(
        env=env,
        obs_dims=obs_dims,
        include_same_slot_different_feature=include_same_slot_different_feature,
        include_same_feature_different_slots=include_same_feature_different_slots,
        include_last_two_actions=include_last_two_actions,
    )

    if len(selected_pairs) == 0:
        raise ValueError('No matching structured pairs were found in obs_dims.')

    anchor_indices = _choose_anchor_indices(
        rollout_df=rollout_df,
        obs_array=obs_array,
        n_anchors=n_anchors,
        anchor_method=anchor_method,
        random_seed=random_seed,
        standardize_for_clustering=standardize_for_clustering,
    )

    sweep_2d_results = {}
    for obs_dim_x, obs_dim_y in selected_pairs:
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

    return sweep_2d_results, anchor_indices, pair_summary_df