import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Columns that store angles in radians and need ×180/π before display/comparison
_ANGLE_COLS = {
    'diff_in_abs_angle_to_nxt_ff_median',
    'diff_in_angle_to_nxt_ff_median',
    'diff_in_abs_angle_to_nxt_ff_ci_low_95',
    'diff_in_abs_angle_to_nxt_ff_ci_high_95',
}

METRIC_CONFIGS = {
    'heading': dict(
        metric_col='diff_in_abs_angle_to_nxt_ff_median',
        ci_low_col='diff_in_abs_angle_to_nxt_ff_ci_low_95',
        ci_high_col='diff_in_abs_angle_to_nxt_ff_ci_high_95',
        is_angle=True,
        scale=1,
        ylabel='heading diff (°)',
        df_key='median',
    ),
    'curv': dict(
        metric_col='diff_in_abs_d_curv_median',
        ci_low_col='diff_in_abs_d_curv_ci_low_95',
        ci_high_col='diff_in_abs_d_curv_ci_high_95',
        is_angle=False,
        scale=1,
        ylabel='curvature diff (°/m)',
        df_key='median',
    ),
    'perc': dict(
        metric_col='perc',
        ci_low_col='perc_ci_low_95',
        ci_high_col='perc_ci_high_95',
        is_angle=False,
        scale=100,
        ylabel='same-side (%)',
        df_key='perc',
    ),
}


def _convert_angles(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col] * 180 / np.pi
    return df


def _apply_conditions(df, conditions):
    for col, val in conditions.items():
        if col in df.columns:
            df = df[df[col] == val]
    return df


def _match_top_combos(df, top_combos, sweep_keys):
    mask = pd.Series(False, index=df.index)
    for _, combo in top_combos.iterrows():
        m = pd.Series(True, index=df.index)
        for k, v in combo.items():
            m &= df[k].isna() if pd.isna(v) else df[k] == v
        mask |= m
    return df[mask]


def _make_agent_label(row, sweep_keys):
    parts = []
    for k in sweep_keys:
        v = row.get(k)
        if k == 'num_obs_ff':
            parts.append(f"ff{int(v)}" if not pd.isna(v) else 'ff?')
        elif k == 'max_in_memory_time':
            parts.append(f"mem{v}" if not pd.isna(v) else 'mem?')
        elif k == 'identity_slot_strategy':
            parts.append(str(v)[:8] if isinstance(v, str) else '?')
        else:
            parts.append(str(v) if not pd.isna(v) else '?')
    return '_'.join(parts)


def rank_agents_vs_monkey(df, metric_col, sweep_keys, fixed_conditions,
                          ref_key='ref_point_value', is_angle=False):
    """
    Rank agent param combos by MSE vs monkey across all ref_point_values.
    Averages over seeds/reps before computing MSE.

    Returns a DataFrame sorted by MSE (ascending), with columns
    sweep_keys + ['mse', 'rmse'].
    """
    df = df.copy()
    if is_angle:
        df = _convert_angles(df, _ANGLE_COLS)

    df = _apply_conditions(df, fixed_conditions)

    monkey = df[df['id'] == 'monkey'].copy()
    agents = df[df['id'] != 'monkey'].copy()

    both_have_ref = (
        ref_key in monkey.columns and monkey[ref_key].notna().any() and
        ref_key in agents.columns and agents[ref_key].notna().any()
    )
    ref_group = [ref_key] if both_have_ref else []

    agent_avg = (agents.groupby(sweep_keys + ref_group, dropna=False)[metric_col]
                 .mean().reset_index()
                 .rename(columns={metric_col: 'agent_val'}))

    if ref_group:
        monkey_vals = (monkey.groupby(ref_group)[metric_col]
                       .mean().reset_index()
                       .rename(columns={metric_col: 'monkey_val'}))
        merged = agent_avg.merge(monkey_vals, on=ref_group, how='inner')
    else:
        merged = agent_avg.copy()
        merged['monkey_val'] = monkey[metric_col].mean()

    merged['sq_err'] = (merged['agent_val'] - merged['monkey_val']) ** 2

    result = (merged.groupby(sweep_keys, dropna=False)['sq_err']
              .mean().reset_index()
              .rename(columns={'sq_err': 'mse'}))
    result['rmse'] = np.sqrt(result['mse'])
    return result.sort_values('mse').reset_index(drop=True)


def get_overlay_data(df, metric_col, ci_low_col, ci_high_col,
                     sweep_keys, rank_conditions, plot_conditions, top_n,
                     ref_key='ref_point_value', is_angle=False,
                     precomputed_ranking=None):
    """
    Return (monkey_df, agents_avg_df, agents_have_ref) for overlay plotting.

    monkey_df         : filtered monkey rows (angle-converted if is_angle)
    agents_avg_df     : top-N agent combos averaged over seeds, with a 'label' column
    agents_have_ref   : bool — whether agents have ref_point_value variation
    precomputed_ranking: optional pre-ranked DataFrame (output of rank_agents_vs_monkey);
                        skips re-ranking if provided
    """
    df = df.copy()
    if is_angle:
        df = _convert_angles(df, _ANGLE_COLS)

    # Use pre-computed ranking if supplied, otherwise rank now
    if precomputed_ranking is not None:
        result = precomputed_ranking
    else:
        result = rank_agents_vs_monkey(df, metric_col, sweep_keys, rank_conditions,
                                       ref_key=ref_key, is_angle=False)  # already converted
    top_combos = result.head(top_n)[sweep_keys].reset_index(drop=True)

    monkey = _apply_conditions(df[df['id'] == 'monkey'].copy(), plot_conditions)
    if ref_key in monkey.columns:
        monkey = monkey.sort_values(ref_key)

    agents = _apply_conditions(df[df['id'] != 'monkey'].copy(), plot_conditions)
    agents = _match_top_combos(agents, top_combos, sweep_keys)

    agents_have_ref = ref_key in agents.columns and agents[ref_key].notna().any()
    group_keys = sweep_keys + ([ref_key] if agents_have_ref else [])

    agg = {c: 'mean' for c in [metric_col, ci_low_col, ci_high_col]
           if c in agents.columns}
    agents_avg = agents.groupby(group_keys, dropna=False).agg(agg).reset_index()
    agents_avg['label'] = agents_avg.apply(
        lambda r: _make_agent_label(r, sweep_keys), axis=1
    )

    # Attach rank (1 = best) from the MSE ranking so the caller can color by it
    rank_lookup = result.head(top_n)[sweep_keys].copy()
    rank_lookup['rank'] = range(1, len(rank_lookup) + 1)
    agents_avg = agents_avg.merge(rank_lookup, on=sweep_keys, how='left')

    return monkey, agents_avg, agents_have_ref


def plot_monkey_vs_top_agents(
    median_df,
    perc_df,
    sweep_keys,
    rank_conditions,
    plot_conditions,
    top_n=5,
    ref_key='ref_point_value',
    metrics=('heading', 'curv', 'perc'),
    show_error_bands=True,
    ranking=None,
):
    """
    For each metric in `metrics`, plot monkey (thick black) overlaid with the
    top-N best-matching agents (colored lines), across ref_point_value on x-axis.

    If agents lack ref_point_value variation (e.g. perc), they are drawn as
    horizontal dashed lines at their mean value.

    Parameters
    ----------
    median_df : DataFrame — combined monkey + agent median df (both_median_df)
    perc_df   : DataFrame — combined monkey + agent perc df (new_both_perc_df)
    sweep_keys: list of agent param columns to rank/group on
    rank_conditions : dict — conditions used for ranking (e.g. test_or_control='difference')
    plot_conditions : dict — conditions used for display (can differ from rank)
    top_n     : int — how many top agents to overlay
    ref_key   : str — x-axis column name
    metrics   : which metrics to plot; subset of ('heading', 'curv', 'perc')
    ranking   : optional dict keyed by metric name with pre-computed ranking DataFrames
                (output of the ranking cell); skips re-ranking when provided
    """
    df_map = {'median': median_df, 'perc': perc_df}

    for metric_name in metrics:
        cfg = METRIC_CONFIGS[metric_name]
        df = df_map[cfg['df_key']]
        if df is None or cfg['metric_col'] not in df.columns:
            print(f"Skipping {metric_name}: data not available.")
            continue

        precomputed = ranking.get(metric_name) if ranking is not None else None
        monkey, agents_avg, agents_have_ref = get_overlay_data(
            df,
            cfg['metric_col'], cfg['ci_low_col'], cfg['ci_high_col'],
            sweep_keys, rank_conditions, plot_conditions, top_n,
            ref_key=ref_key, is_angle=cfg['is_angle'],
            precomputed_ranking=precomputed,
        )

        scale = cfg.get('scale', 1)
        mc, cl, ch = cfg['metric_col'], cfg['ci_low_col'], cfg['ci_high_col']

        fig, ax = plt.subplots(figsize=(10, 6))
        n_agents = agents_avg['label'].nunique()
        # rank 1 (best) → darkest, rank N → lightest
        cmap = plt.cm.plasma_r
        def rank_color(rank):
            return cmap(0.15 + 0.7 * (rank - 1) / max(n_agents - 1, 1))

        # Monkey — line if ref_key present, horizontal band otherwise
        x_monk = monkey[ref_key] if ref_key in monkey.columns else None
        monk_has_ref = x_monk is not None and x_monk.notna().any()
        if monk_has_ref:
            ax.plot(x_monk, monkey[mc] * scale,
                    color='black', lw=2.5, marker='o', ms=5,
                    label='monkey', zorder=10)
            if show_error_bands and cl in monkey.columns:
                ax.fill_between(x_monk,
                                monkey[cl] * scale, monkey[ch] * scale,
                                alpha=0.2, color='black')
        else:
            val = monkey[mc].mean() * scale
            ax.axhline(val, color='black', lw=2.5, linestyle='-',
                       label='monkey', zorder=10)
            if show_error_bands and cl in monkey.columns:
                ax.axhspan(monkey[cl].mean() * scale, monkey[ch].mean() * scale,
                           color='black', alpha=0.15)

        # Top agents — color by rank (rank 1 = best = darkest), legend in rank order
        for label, grp in agents_avg.sort_values('rank').groupby('label', sort=False):
            rank = grp['rank'].iloc[0]
            color = rank_color(rank)
            if agents_have_ref:
                grp = grp.sort_values(ref_key)
                ax.plot(grp[ref_key], grp[mc] * scale,
                        color=color, lw=1.5, marker='o', ms=3,
                        label=label, alpha=0.85)
                if show_error_bands and cl in grp.columns:
                    ax.fill_between(grp[ref_key],
                                    grp[cl] * scale, grp[ch] * scale,
                                    alpha=0.12, color=color)
            else:
                val = grp[mc].mean() * scale
                ax.axhline(val, color=color, lw=1.5, linestyle='--',
                           label=label, alpha=0.85)
                if show_error_bands and cl in grp.columns and ch in grp.columns:
                    ax.axhspan(grp[cl].mean() * scale, grp[ch].mean() * scale,
                               color=color, alpha=0.10)

        cond_str = ', '.join(f'{k}={v}' for k, v in plot_conditions.items())
        ax.set_xlabel(ref_key)
        ax.set_ylabel(cfg['ylabel'])
        ax.set_title(f'Monkey vs top-{top_n} agents — {metric_name}\n({cond_str})')
        ax.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.show()
