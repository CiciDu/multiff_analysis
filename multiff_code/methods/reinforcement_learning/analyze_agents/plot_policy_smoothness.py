from reinforcement_learning.analyze_agents import analyze_policy_smoothness

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


def plot_local_policy_sweep(
    sweep_df,
    obs_dim,
    action_dim=0,
    max_anchors_to_plot=8,
):
    '''
    Plot local response curves for one observation dimension.
    '''
    value_col = f'action_{action_dim}'
    dim_df = sweep_df[sweep_df['obs_dim'] == obs_dim].copy()

    if len(dim_df) == 0:
        raise ValueError(f'No rows found for obs_dim={obs_dim}.')

    plt.figure(figsize=(6, 4))

    anchor_ids = np.sort(dim_df['anchor_id'].unique())[:max_anchors_to_plot]
    for anchor_id in anchor_ids:
        anchor_df = dim_df[dim_df['anchor_id'] == anchor_id].sort_values('delta')
        plt.plot(anchor_df['delta'].values, anchor_df[value_col].values, alpha=0.8)

    plt.axvline(0.0)
    plt.xlabel(f'delta on {dim_df["obs_dim_label"].iloc[0]}')
    plt.ylabel(value_col)
    plt.title(f'Local policy sweep: {dim_df["obs_dim_label"].iloc[0]}')
    plt.tight_layout()
    plt.show()


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

        baseline_action = None

        for delta_x in delta_x_values:
            for delta_y in delta_y_values:
                perturbed_obs = base_obs.copy()
                perturbed_obs[obs_dim_x] = np.clip(base_x + float(delta_x), clip_low, clip_high)
                perturbed_obs[obs_dim_y] = np.clip(base_y + float(delta_y), clip_low, clip_high)

                obs_input = analyze_policy_smoothness._rebuild_policy_obs_input_from_flat(env, rl_agent, perturbed_obs)
                action, _ = rl_agent.predict(obs_input, deterministic=deterministic)
                action = np.asarray(action, dtype=np.float32).reshape(-1)

                if np.isclose(delta_x, 0.0) and np.isclose(delta_y, 0.0):
                    baseline_action = action.copy()

                rows.append({
                    'anchor_id': int(anchor_id),
                    'anchor_index': int(anchor_index),
                    'obs_dim_x': int(obs_dim_x),
                    'obs_dim_y': int(obs_dim_y),
                    'obs_dim_x_label': analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_x),
                    'obs_dim_y_label': analyze_policy_smoothness.get_obs_dim_label(env, obs_dim_y),
                    'base_value_x': base_x,
                    'base_value_y': base_y,
                    'delta_x': float(delta_x),
                    'delta_y': float(delta_y),
                    'perturbed_value_x': float(perturbed_obs[obs_dim_x]),
                    'perturbed_value_y': float(perturbed_obs[obs_dim_y]),
                    **{f'action_{j}': float(action[j]) for j in range(len(action))}
                })

    sweep_2d_df = pd.DataFrame(rows)

    action_cols = [c for c in sweep_2d_df.columns if c.startswith('action_')]

    baseline_df = (
        sweep_2d_df[
            np.isclose(sweep_2d_df['delta_x'].values, 0.0)
            & np.isclose(sweep_2d_df['delta_y'].values, 0.0)
        ][['anchor_id'] + action_cols]
        .rename(columns={c: f'baseline_{c}' for c in action_cols})
    )

    sweep_2d_df = sweep_2d_df.merge(baseline_df, on='anchor_id', how='left')

    sq_norm = 0.0
    for c in action_cols:
        dc = f'delta_{c}'
        bc = f'baseline_{c}'
        sweep_2d_df[dc] = sweep_2d_df[c] - sweep_2d_df[bc]
        sq_norm = sq_norm + sweep_2d_df[dc] ** 2

    sweep_2d_df['action_change_norm'] = np.sqrt(sq_norm)

    return sweep_2d_df


def plot_local_policy_2d_heatmap(
    sweep_2d_df,
    action_dim=0,
    anchor_id=None,
    value_type='delta_action',
    average_across_anchors=False,
):
    '''
    Plot a 2-D heatmap of local policy response.

    value_type:
        'raw_action' -> action_k
        'delta_action' -> delta_action_k
        'norm' -> action_change_norm
    '''
    if value_type == 'raw_action':
        value_col = f'action_{action_dim}'
    elif value_type == 'delta_action':
        value_col = f'delta_action_{action_dim}'
    elif value_type == 'norm':
        value_col = 'action_change_norm'
    else:
        raise ValueError("value_type must be 'raw_action', 'delta_action', or 'norm'.")

    plot_df = sweep_2d_df.copy()

    if not average_across_anchors:
        if anchor_id is None:
            anchor_id = int(np.sort(plot_df['anchor_id'].unique())[0])
        plot_df = plot_df[plot_df['anchor_id'] == anchor_id].copy()

    grouped = (
        plot_df
        .groupby(['delta_y', 'delta_x'], as_index=False)[value_col]
        .mean()
    )

    pivot_df = grouped.pivot(index='delta_y', columns='delta_x', values=value_col)
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        pivot_df.values,
        aspect='auto',
        origin='lower',
        extent=[
            pivot_df.columns.min(), pivot_df.columns.max(),
            pivot_df.index.min(), pivot_df.index.max()
        ],
    )
    plt.colorbar(label=value_col)

    x_label = plot_df['obs_dim_x_label'].iloc[0]
    y_label = plot_df['obs_dim_y_label'].iloc[0]

    plt.xlabel(f'delta on {x_label}')
    plt.ylabel(f'delta on {y_label}')

    if average_across_anchors:
        plt.title(f'2-D local response (mean across anchors): {value_col}')
    else:
        plt.title(f'2-D local response (anchor {anchor_id}): {value_col}')

    plt.tight_layout()
    plt.show()


def plot_local_policy_2d_contour(
    sweep_2d_df,
    action_dim=0,
    anchor_id=None,
    value_type='delta_action',
    average_across_anchors=False,
    n_levels=12,
):
    '''
    Same data as heatmap, but contour lines are often better for judging smoothness.
    '''
    if value_type == 'raw_action':
        value_col = f'action_{action_dim}'
    elif value_type == 'delta_action':
        value_col = f'delta_action_{action_dim}'
    elif value_type == 'norm':
        value_col = 'action_change_norm'
    else:
        raise ValueError("value_type must be 'raw_action', 'delta_action', or 'norm'.")

    plot_df = sweep_2d_df.copy()

    if not average_across_anchors:
        if anchor_id is None:
            anchor_id = int(np.sort(plot_df['anchor_id'].unique())[0])
        plot_df = plot_df[plot_df['anchor_id'] == anchor_id].copy()

    grouped = (
        plot_df
        .groupby(['delta_y', 'delta_x'], as_index=False)[value_col]
        .mean()
    )

    pivot_df = grouped.pivot(index='delta_y', columns='delta_x', values=value_col)
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)

    x = pivot_df.columns.values.astype(np.float32)
    y = pivot_df.index.values.astype(np.float32)
    z = pivot_df.values.astype(np.float32)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(x, y, z, levels=n_levels)
    plt.colorbar(contour, label=value_col)

    x_label = plot_df['obs_dim_x_label'].iloc[0]
    y_label = plot_df['obs_dim_y_label'].iloc[0]

    plt.xlabel(f'delta on {x_label}')
    plt.ylabel(f'delta on {y_label}')

    if average_across_anchors:
        plt.title(f'2-D contour (mean across anchors): {value_col}')
    else:
        plt.title(f'2-D contour (anchor {anchor_id}): {value_col}')

    plt.tight_layout()
    plt.show()