import numpy as np
import matplotlib.pyplot as plt

from pattern_discovery.learning.proportion_trend import plot_stacked_bars_utils



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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _prepare_retry_strategy_plot_df(df, config, sweep_key):
    '''
    Shared preprocessing for retry-strategy line plots.
    Applies label mapping, enforces sweep ordering, and extracts seed if needed.
    '''
    category_order = config['category_order']
    agent_col = config.get('agent_col', 'id')
    label_mapping = config.get('new_label_mapping')

    df_plot = df.copy()

    if label_mapping is not None:
        df_plot = plot_stacked_bars_utils.apply_label_mapping(df_plot, label_mapping)

    if 'seed' not in df_plot.columns:
        df_plot['seed'] = df_plot[agent_col].astype(str).str.extract(r'seed(\d+)')
        if df_plot['seed'].notna().any():
            df_plot['seed'] = df_plot['seed'].astype('Int64')

    sweep_vals = sorted(df_plot[sweep_key].dropna().unique())

    valid_sweep_vals = []
    for sweep_val in sweep_vals:
        sub = df_plot[df_plot[sweep_key] == sweep_val]
        if sub.empty:
            continue
        if sub['new_label'].isin(category_order).any():
            valid_sweep_vals.append(sweep_val)

    return df_plot, valid_sweep_vals, category_order, agent_col


def _build_retry_strategy_long_df(df_plot, sweep_vals, category_order, sweep_key, agent_col):
    '''
    Convert raw per-agent rows into a long dataframe with one row per:
    agent x sweep value x category
    '''
    rows = []

    for agent, df_agent in df_plot.groupby(agent_col):
        seed = df_agent['seed'].iloc[0] if 'seed' in df_agent.columns else pd.NA

        for sweep_val in sweep_vals:
            sub = df_agent[df_agent[sweep_key] == sweep_val]
            if sub.empty:
                continue

            M = plot_stacked_bars_utils.prepare_for_stacked_bar_no_phase(
                sub,
                category_order
            )

            if M.empty:
                continue

            for cat in category_order:
                rows.append({
                    'agent': agent,
                    'seed': seed,
                    sweep_key: sweep_val,
                    'category': cat,
                    'proportion': M.iloc[0][cat]
                })

    return pd.DataFrame(rows)


def _get_retry_strategy_title(config):
    title = config.get('title', '')
    title = title.replace('Across Early and Late Sessions', '').strip()
    return title


def plot_retry_strategy_lines(df, config, sweep_key):
    '''
    Original combined plot:
    one line per category, with scatter over agents.
    '''
    df_plot, sweep_vals, category_order, agent_col = _prepare_retry_strategy_plot_df(
        df,
        config,
        sweep_key
    )
    long_df = _build_retry_strategy_long_df(
        df_plot,
        sweep_vals,
        category_order,
        sweep_key,
        agent_col
    )

    category_colors = config.get('category_colors')
    if category_colors is None:
        category_colors = ['#0072B2', '#CC79A7', '#E69F00', '#009E73',
                           '#F0E442', '#56B4E9', '#000000']
    colors = {label: color for label, color in zip(category_order, category_colors)}

    title = _get_retry_strategy_title(config)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    for cat in category_order:
        df_cat = long_df[long_df['category'] == cat].copy()
        if df_cat.empty:
            continue

        for _, row in df_cat.iterrows():
            ax.scatter(
                row[sweep_key],
                row['proportion'],
                color=colors[cat],
                alpha=0.35,
                s=20
            )

        stats_df = (
            df_cat.groupby(sweep_key)['proportion']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        stats_df['ci'] = 1.96 * stats_df['se']

        x = stats_df[sweep_key].to_numpy()
        y = stats_df['mean'].to_numpy()
        ci = stats_df['ci'].fillna(0).to_numpy()

        ax.plot(
            x,
            y,
            color=colors[cat],
            linewidth=3,
            label=cat
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color=colors[cat],
            alpha=0.15
        )

    ax.set_xlabel(sweep_key)
    ax.set_ylabel('Proportion')
    ax.set_title(title)
    ax.set_xticks(sweep_vals)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_retry_strategy_lines_separate_by_seed(df, config, sweep_key, jitter = 0.05):
    '''
    New plot:
    - one subplot per category
    - each seed gets its own color
    - within each seed, plot the mean across agents belonging to that seed
    - faint scatter shows individual agents
    '''
    df_plot, sweep_vals, category_order, agent_col = _prepare_retry_strategy_plot_df(
        df,
        config,
        sweep_key
    )
    long_df = _build_retry_strategy_long_df(
        df_plot,
        sweep_vals,
        category_order,
        sweep_key,
        agent_col
    )

    if long_df.empty:
        raise ValueError('No data available to plot.')

    if 'seed' not in long_df.columns or long_df['seed'].isna().all():
        raise ValueError('Seed column is missing or could not be extracted from id.')

    title = _get_retry_strategy_title(config)

    seed_vals = sorted(long_df['seed'].dropna().unique())
    
    if len(seed_vals) <= 20:
        cmap = plt.cm.get_cmap('tab20', len(seed_vals))
        seed_to_color = {seed: cmap(i) for i, seed in enumerate(seed_vals)}
    else:
        cmap = plt.cm.viridis
        seed_to_color = {
            seed: cmap(i / (len(seed_vals) - 1))
            for i, seed in enumerate(seed_vals)
        }

    n_categories = len(category_order)
    n_cols = min(2, n_categories)
    n_rows = math.ceil(n_categories / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 5.5 * n_rows),
        dpi=100,
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, cat in zip(axes, category_order):
        df_cat = long_df[long_df['category'] == cat].copy()

        if df_cat.empty:
            ax.set_visible(False)
            continue

        for seed in seed_vals:
            df_seed = df_cat[df_cat['seed'] == seed].copy()
            if df_seed.empty:
                continue

            for _, row in df_seed.iterrows():
                x_jittered = row[sweep_key] + np.random.uniform(-jitter, jitter)

                ax.scatter(
                    x_jittered,
                    row['proportion'],
                    color=seed_to_color[seed],
                    alpha=0.6,
                    s=20
                )

        # overall mean across all seeds
        stats_df = (
            df_cat.groupby(sweep_key)['proportion']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )

        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        stats_df['ci'] = 1.96 * stats_df['se']

        x = stats_df[sweep_key].to_numpy()
        y = stats_df['mean'].to_numpy()
        ci = stats_df['ci'].fillna(0).to_numpy()

        ax.plot(
            x,
            y,
            color='black',
            linewidth=3,
            label='mean'
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color='black',
            alpha=0.15
        )


        ax.set_title(cat)
        ax.set_xticks(sweep_vals)
        ax.set_ylim(0, 1)
        ax.set_xlabel(sweep_key)
        ax.set_ylabel('Proportion')
        #ax.legend(fontsize=9)

    for ax in axes[n_categories:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, y=1.02)

    plt.tight_layout()
    plt.show()

    return fig, axes