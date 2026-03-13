import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import math

from pattern_discovery.learning.proportion_trend import analyze_proportion_trend, plot_stacked_bars_utils

import pandas as pd


def plot_retry_outcome_sweep_ci(df, config, sweep_key):

    category_order = config['category_order']
    agent_col = config.get('agent_col', 'id')
    title = config.get('title', '')
    # strip off 'Across Early and Late Sessions' from title
    title = title.replace('Across Early and Late Sessions', '').strip()

    label_mapping = config.get('new_label_mapping')

    df = plot_stacked_bars_utils.apply_label_mapping(df, label_mapping)

    fig, axes = plt.subplots(
        1, len(category_order),
        figsize=(4*len(category_order),4),
        sharey=True,
        dpi=300
    )

    sweep_vals = sorted(df[sweep_key].unique())

    # keep only sweep values that actually have usable data
    valid_sweep_vals = []
    for v in sweep_vals:
        sub = df[df[sweep_key] == v]
        if sub.empty:
            continue
        if (sub['new_label'].isin(category_order)).any():
            valid_sweep_vals.append(v)

    sweep_vals = valid_sweep_vals


    for ax, cat in zip(axes, category_order):

        data = []

        for v in sweep_vals:

            vals = []

            for agent, df_agent in df.groupby(agent_col):

                sub = df_agent[df_agent[sweep_key] == v]

                if len(sub) == 0:
                    continue

                M = plot_stacked_bars_utils.prepare_for_stacked_bar_no_phase(
                    sub,
                    category_order
                )

                vals.append(M.iloc[0][cat])

            vals = np.array(vals)
            data.append(vals)

            ax.scatter(
                np.full_like(vals, v, dtype=float),
                vals,
                color='black',
                alpha=0.5,
                s=25
            )

        means = np.array([np.mean(x) for x in data])
        ses = np.array([np.std(x)/np.sqrt(len(x)) for x in data])

        ax.plot(
            sweep_vals,
            means,
            color='black',
            linewidth=2
        )

        ax.fill_between(
            sweep_vals,
            means - 1.96*ses,
            means + 1.96*ses,
            alpha=0.2,
            color='black'
        )

        ax.set_title(cat)
        ax.set_xlabel(sweep_key)


    axes[0].set_ylabel("Proportion")

    for ax in axes:
        ax.set_xticks(sweep_vals)
        
    fig.suptitle(title, fontsize=14, weight='bold')

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_retry_strategy_lines(df, config, sweep_key):

    category_order = config['category_order']
    agent_col = config.get('agent_col', 'id')
    title = config.get('title', '')
    # strip off 'Across Early and Late Sessions' from title
    title = title.replace('Across Early and Late Sessions', '').strip()

    label_mapping = config.get('new_label_mapping')
    

    df = plot_stacked_bars_utils.apply_label_mapping(df, label_mapping)

    category_colors = config.get('category_colors')
    if category_colors is None:
        category_colors = ['#0072B2', '#CC79A7', '#E69F00', '#009E73',
                        '#F0E442', '#56B4E9', '#000000']
    colors = {label: color for label, color in zip(category_order, category_colors)}

    sweep_vals = sorted(df[sweep_key].unique())

    valid_sweep_vals = []
    for v in sweep_vals:
        sub = df[df[sweep_key] == v]
        if sub.empty:
            continue
        if (sub['new_label'].isin(category_order)).any():
            valid_sweep_vals.append(v)

    sweep_vals = valid_sweep_vals
    
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)

    for cat in category_order:

        data = []

        for v in sweep_vals:

            vals = []

            for agent, df_agent in df.groupby(agent_col):

                sub = df_agent[df_agent[sweep_key] == v]

                if len(sub) == 0:
                    continue

                M = plot_stacked_bars_utils.prepare_for_stacked_bar_no_phase(
                    sub,
                    category_order
                )

                val = M.iloc[0][cat]
                vals.append(val)

                ax.scatter(
                    v,
                    val,
                    color=colors[cat],
                    alpha=0.35,
                    s=20
                )

            data.append(np.array(vals))

        means = np.array([np.mean(x) for x in data])
        ses = np.array([np.std(x)/np.sqrt(len(x)) for x in data])

        ax.plot(
            sweep_vals,
            means,
            color=colors[cat],
            linewidth=3,
            label=cat
        )

        ax.fill_between(
            sweep_vals,
            means - 1.96*ses,
            means + 1.96*ses,
            color=colors[cat],
            alpha=0.15
        )

    ax.set_xlabel(sweep_key)
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    
    ax.set_xticks(sweep_vals)

    ax.set_ylim(0,1)

    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig, ax