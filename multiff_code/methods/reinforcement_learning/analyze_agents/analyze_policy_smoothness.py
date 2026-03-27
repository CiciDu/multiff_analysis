
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
import pandas as pd
import math
from matplotlib import rc
from math import pi
import logging
import random
from gymnasium import spaces
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def _format_obs_for_policy(env, rl_agent, state_or_obs):
    '''
    Format observation exactly the way the policy sees it.
    Returns both:
    - obs_input: what gets passed into rl_agent.predict(...)
    - obs_flat: a flat numeric vector used for local perturbation analysis
    '''
    obs_input = state_or_obs
    obs_flat = np.asarray(state_or_obs, dtype=np.float32).copy()

    try:
        if hasattr(rl_agent, 'observation_space') and isinstance(rl_agent.observation_space, spaces.Dict):
            n_slots = int(env.num_obs_ff) * int(getattr(env, 'num_elem_per_ff', 0))
            slots = obs_flat[:n_slots].reshape(
                int(env.num_obs_ff),
                int(getattr(env, 'num_elem_per_ff', 0))
            )
            obs_input = {'slots': slots.astype(np.float32)}

            if getattr(env, 'add_action_to_obs', False):
                ego = obs_flat[n_slots:n_slots + 2]
                obs_input['ego'] = ego.astype(np.float32)
    except Exception:
        obs_input = state_or_obs

    return obs_input, obs_flat


def _flatten_policy_obs_input(env, rl_agent, obs_input):
    '''
    Convert the policy input back into one flat vector so perturbation logic
    can work uniformly across plain Box obs and Dict obs.
    '''
    if isinstance(obs_input, dict):
        flat_parts = []
        if 'slots' in obs_input:
            flat_parts.append(np.asarray(obs_input['slots'], dtype=np.float32).reshape(-1))
        if 'ego' in obs_input:
            flat_parts.append(np.asarray(obs_input['ego'], dtype=np.float32).reshape(-1))
        if len(flat_parts) == 0:
            raise ValueError('obs_input is a dict but contains no recognized keys.')
        return np.concatenate(flat_parts, axis=0).astype(np.float32)

    return np.asarray(obs_input, dtype=np.float32).reshape(-1).copy()


def _rebuild_policy_obs_input_from_flat(env, rl_agent, obs_flat):
    '''
    Rebuild the policy input from a flat observation vector after perturbation.
    '''
    obs_flat = np.asarray(obs_flat, dtype=np.float32).reshape(-1)

    try:
        if hasattr(rl_agent, 'observation_space') and isinstance(rl_agent.observation_space, spaces.Dict):
            n_slots = int(env.num_obs_ff) * int(getattr(env, 'num_elem_per_ff', 0))
            slots = obs_flat[:n_slots].reshape(
                int(env.num_obs_ff),
                int(getattr(env, 'num_elem_per_ff', 0))
            )
            obs_input = {'slots': slots.astype(np.float32)}

            if getattr(env, 'add_action_to_obs', False):
                ego = obs_flat[n_slots:n_slots + 2]
                obs_input['ego'] = ego.astype(np.float32)

            return obs_input
    except Exception:
        pass

    return obs_flat.astype(np.float32)


def _step_default_agent_with_logging(env, rl_agent, state_or_obs, deterministic):
    '''
    Same as _step_default_agent, but also returns:
    - obs_input actually passed to the policy
    - flat obs vector for perturbation analysis
    - action
    '''
    obs_input, obs_flat = _format_obs_for_policy(env, rl_agent, state_or_obs)
    action, _ = rl_agent.predict(obs_input, deterministic=deterministic)
    action = np.asarray(action, dtype=np.float32)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    return next_obs, terminated, truncated, obs_input, obs_flat, action

def get_obs_dim_index(env, slot_index=None, field_name=None, action_dim=None):
    '''
    Map a semantic observation component to the flattened observation index.

    Assumes slot features are flattened slot-major:
        [slot0_all_fields, slot1_all_fields, ..., slotN_all_fields, ego?]

    If your env uses a different flattening order, adjust this function only.
    '''
    n_slots = int(env.num_obs_ff)
    n_fields = int(getattr(env, 'num_elem_per_ff', 0))

    if action_dim is not None:
        if not getattr(env, 'add_action_to_obs', False):
            raise ValueError('env.add_action_to_obs is False, so no appended action dims exist.')
        if action_dim not in [0, 1]:
            raise ValueError('action_dim must be 0 or 1.')
        return n_slots * n_fields + action_dim

    if slot_index is None or field_name is None:
        raise ValueError('Provide either (slot_index, field_name) or action_dim.')

    if not hasattr(env, 'slot_fields'):
        raise ValueError('env must define env.slot_fields to map field_name to an index.')

    if field_name not in env.slot_fields:
        raise ValueError(f'field_name must be one of {env.slot_fields}.')

    field_index = env.slot_fields.index(field_name)
    return slot_index * n_fields + field_index


def get_obs_dim_label(env, obs_dim):
    '''
    Human-readable label for a flattened observation dimension.
    '''
    n_slots = int(env.num_obs_ff)
    n_fields = int(getattr(env, 'num_elem_per_ff', 0))
    n_slot_dims = n_slots * n_fields

    if obs_dim < n_slot_dims:
        slot_index = obs_dim // n_fields
        field_index = obs_dim % n_fields
        if hasattr(env, 'slot_fields') and field_index < len(env.slot_fields):
            return f'slot_{slot_index}_{env.slot_fields[field_index]}'
        return f'slot_{slot_index}_field_{field_index}'

    if getattr(env, 'add_action_to_obs', False):
        action_index = obs_dim - n_slot_dims
        return f'ego_{action_index}'

    return f'obs_dim_{obs_dim}'


def collect_policy_inputs_and_actions(
    env,
    rl_agent,
    n_steps=3000,
    deterministic=True,
    first_obs=None,
    seed=42,
):
    '''
    Collect the real observations that are actually fed to rl_agent.predict(...)
    plus the corresponding actions, for default/SB3-style agents.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if first_obs is None:
        state_or_obs, _ = env.reset(seed=seed)
    else:
        state_or_obs = first_obs

    obs_flat_list = []
    action_list = []
    rows = []

    episode_id = 0
    episode_step = 0

    for step in range(n_steps):
        next_obs, terminated, truncated, obs_input, obs_flat, action = _step_default_agent_with_logging(
            env, rl_agent, state_or_obs, deterministic
        )

        obs_flat_list.append(_flatten_policy_obs_input(env, rl_agent, obs_input))
        action_list.append(np.asarray(action, dtype=np.float32).reshape(-1))

        rows.append({
            'global_step': step,
            'episode_id': episode_id,
            'episode_step': episode_step,
            'time': float(getattr(env, 'time', np.nan)),
            'monkey_x': float(env.agentxy[0] + env.arena_center_global[0]),
            'monkey_y': float(env.agentxy[1] + env.arena_center_global[1]),
            'v': float(getattr(env, 'v', np.nan)),
            'w': float(getattr(env, 'w', np.nan)),
            'is_stop': float(getattr(env, 'is_stop', np.nan)),
        })

        state_or_obs = next_obs
        episode_step += 1

        if terminated or truncated:
            episode_id += 1
            episode_step = 0
            state_or_obs, _ = env.reset(seed=seed + episode_id)

    rollout_df = pd.DataFrame(rows)
    obs_array = np.stack(obs_flat_list, axis=0).astype(np.float32)
    action_array = np.stack(action_list, axis=0).astype(np.float32)

    for j in range(action_array.shape[1]):
        rollout_df[f'action_{j}'] = action_array[:, j]

    return rollout_df, obs_array, action_array


def evaluate_policy_local_sweeps(
    env,
    rl_agent,
    obs_array,
    anchor_indices,
    obs_dims,
    delta_values,
    deterministic=True,
    clip_low=-1.0,
    clip_high=1.0,
):
    '''
    Around each real anchor observation, perturb selected dims and re-run the policy.
    '''
    rows = []
    delta_values = np.asarray(delta_values, dtype=np.float32)

    for anchor_id, anchor_index in enumerate(anchor_indices):
        base_obs = obs_array[anchor_index].copy()

        for obs_dim in obs_dims:
            base_value = float(base_obs[obs_dim])

            baseline_action = None

            for delta in delta_values:
                perturbed_obs = base_obs.copy()
                perturbed_obs[obs_dim] = np.clip(base_value + float(delta), clip_low, clip_high)

                obs_input = _rebuild_policy_obs_input_from_flat(env, rl_agent, perturbed_obs)
                action, _ = rl_agent.predict(obs_input, deterministic=deterministic)
                action = np.asarray(action, dtype=np.float32).reshape(-1)

                if np.isclose(delta, 0.0):
                    baseline_action = action.copy()

                rows.append({
                    'anchor_id': int(anchor_id),
                    'anchor_index': int(anchor_index),
                    'obs_dim': int(obs_dim),
                    'obs_dim_label': get_obs_dim_label(env, obs_dim),
                    'base_value': base_value,
                    'delta': float(delta),
                    'perturbed_value': float(perturbed_obs[obs_dim]),
                    **{f'action_{j}': float(action[j]) for j in range(len(action))}
                })

    sweep_df = pd.DataFrame(rows)

    action_cols = [c for c in sweep_df.columns if c.startswith('action_')]
    baseline_df = (
        sweep_df[np.isclose(sweep_df['delta'].values, 0.0)]
        [['anchor_id', 'obs_dim'] + action_cols]
        .rename(columns={c: f'baseline_{c}' for c in action_cols})
    )

    sweep_df = sweep_df.merge(baseline_df, on=['anchor_id', 'obs_dim'], how='left')

    sq_norm = 0.0
    for c in action_cols:
        dc = f'delta_{c}'
        bc = f'baseline_{c}'
        sweep_df[dc] = sweep_df[c] - sweep_df[bc]
        sq_norm = sq_norm + sweep_df[dc] ** 2

    sweep_df['action_change_norm'] = np.sqrt(sq_norm)

    return sweep_df


def summarize_local_smoothness(sweep_df):
    '''
    Summarize sensitivity per perturbed observation dimension.
    '''
    summary_df = (
        sweep_df
        .groupby(['obs_dim', 'obs_dim_label'], as_index=False)
        .agg(
            mean_action_change_norm=('action_change_norm', 'mean'),
            median_action_change_norm=('action_change_norm', 'median'),
            max_action_change_norm=('action_change_norm', 'max'),
            n_eval=('action_change_norm', 'size'),
        )
        .sort_values('mean_action_change_norm', ascending=False)
        .reset_index(drop=True)
    )
    return summary_df


def choose_anchor_indices(
    rollout_df,
    obs_array=None,
    n_anchors=25,
    method='uniform',
    random_seed=0,
    standardize_for_clustering=True,
):
    '''
    Choose anchor states from the real rollout.

    Parameters
    ----------
    rollout_df : pandas.DataFrame
        One row per collected timestep.
    obs_array : np.ndarray or None
        Flat observation array of shape (n_steps, obs_dim).
        Required when method='cluster'.
    n_anchors : int
        Number of anchors to select.
    method : str
        'uniform', 'random', or 'cluster'
    random_seed : int
        Random seed used for random sampling / clustering.
    standardize_for_clustering : bool
        Whether to z-score observation dimensions before clustering.

    Returns
    -------
    anchor_indices : np.ndarray
        Indices into rollout_df / obs_array for selected anchors.
    '''
    n = len(rollout_df)
    if n == 0:
        raise ValueError('rollout_df is empty.')

    n_anchors = min(n_anchors, n)

    if method == 'uniform':
        return np.linspace(0, n - 1, n_anchors, dtype=int)

    if method == 'random':
        rng = np.random.default_rng(random_seed)
        return np.sort(rng.choice(n, size=n_anchors, replace=False))

    if method == 'cluster':
        if obs_array is None:
            raise ValueError("obs_array must be provided when method='cluster'.")

        obs_array = np.asarray(obs_array, dtype=np.float32)
        if obs_array.ndim != 2:
            raise ValueError('obs_array must have shape (n_samples, obs_dim).')
        if obs_array.shape[0] != n:
            raise ValueError('obs_array and rollout_df must have the same number of rows.')

        # Standardize so large-scale dimensions do not dominate clustering
        if standardize_for_clustering:
            obs_mean = obs_array.mean(axis=0, keepdims=True)
            obs_std = obs_array.std(axis=0, keepdims=True)
            obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)
            obs_for_clustering = (obs_array - obs_mean) / obs_std
        else:
            obs_for_clustering = obs_array.copy()

        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "method='cluster' requires scikit-learn. "
                "Install it or use method='uniform'/'random'."
            ) from exc

        kmeans = KMeans(
            n_clusters=n_anchors,
            random_state=random_seed,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(obs_for_clustering)
        centroids = kmeans.cluster_centers_

        anchor_indices = []

        for cluster_id in range(n_anchors):
            cluster_member_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_points = obs_for_clustering[cluster_member_indices]
            centroid = centroids[cluster_id]

            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            nearest_local_index = np.argmin(distances)
            anchor_index = cluster_member_indices[nearest_local_index]
            anchor_indices.append(anchor_index)

        anchor_indices = np.array(anchor_indices, dtype=int)

        # Sort by rollout time for easier inspection / plotting
        return np.sort(anchor_indices)

    raise ValueError("method must be 'uniform', 'random', or 'cluster'.")


def run_local_policy_sweep(
    env,
    rl_agent,
    rollout_df,
    obs_array,
    action_array=None,
    n_anchors=25,
    obs_dims=None,
    delta_values=None,
    deterministic=True,
    seed=42,
    anchor_method='cluster',
    standardize_for_clustering=True,
):
    '''
    End-to-end helper:
    - choose anchor states
    - do local perturbation sweeps
    - summarize smoothness
    '''
    if obs_dims is None:
        raise ValueError('Please provide obs_dims explicitly.')

    if delta_values is None:
        delta_values = np.linspace(-0.15, 0.15, 11, dtype=np.float32)

    obs_array = np.asarray(obs_array, dtype=np.float32)
    if len(rollout_df) != obs_array.shape[0]:
        raise ValueError('rollout_df and obs_array must have the same number of rows.')

    anchor_indices = choose_anchor_indices(
        rollout_df=rollout_df,
        obs_array=obs_array,
        n_anchors=n_anchors,
        method=anchor_method,
        random_seed=seed,
        standardize_for_clustering=standardize_for_clustering,
    )

    sweep_df = evaluate_policy_local_sweeps(
        env=env,
        rl_agent=rl_agent,
        obs_array=obs_array,
        anchor_indices=anchor_indices,
        obs_dims=obs_dims,
        delta_values=delta_values,
        deterministic=deterministic,
    )

    summary_df = summarize_local_smoothness(sweep_df)

    return sweep_df, summary_df