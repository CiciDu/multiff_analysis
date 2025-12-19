import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import math

from pattern_discovery.learning.proportion_trend import analyze_proportion_trend

# ---------- helpers (unchanged API where possible) ----------


def get_p_values(phase_df, category_order):
    p_values = {}
    for new_label in category_order:
        sub = phase_df[phase_df['new_label'] == new_label]
        pval_el, test_name = analyze_proportion_trend.test_early_late(
            sub, new_label)
        p_values[new_label] = float(pval_el)
    return p_values


def p_to_stars(p: float) -> str:
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return ''


def text_color_for_bg(hex_color: str) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    lum = 0.2126*(r/255)**2.2 + 0.7152*(g/255)**2.2 + 0.0722*(b/255)**2.2
    return 'black' if lum > 0.5 else 'white'


def prepare_for_stacked_bar(df_monkey, category_order):
    df_monkey_sub = df_monkey[df_monkey['new_label'].isin(
        category_order)].copy()
    ses = analyze_proportion_trend.tertile_phase(df_monkey_sub)
    phase_df = (ses.groupby(['phase', 'new_label', 'item'], observed=True)[['frequency', 'denom_count']]
                .sum().reset_index(drop=False))
    phase_df['denom_count'] = phase_df['denom_count'].astype(int)
    phase_df['ratio'] = phase_df['frequency'] / phase_df['denom_count']
    phase_df_sub = phase_df[phase_df['phase'].isin(['early', 'late'])].copy()
    phase_df_sub['phase'] = pd.Categorical(
        phase_df_sub['phase'], ['early', 'late'], ordered=True)
    phase_df_sub['new_label'] = pd.Categorical(
        phase_df_sub['new_label'], category_order, ordered=True)
    return phase_df_sub

# ---------- plotting ----------


def plot_stacked_bar(M, p_values, category_order, category_colors=None, ax=None, *,
                     x_positions=None, bar_width=0.6,
                     min_pct_to_label=0.02, center_factor=0.55):

    if category_colors is None:
        category_colors = ['#0072B2', '#CC79A7', '#E69F00', '#009E73',
                           '#F0E442', '#56B4E9', '#000000']

    color_map = {label: color for label, color in zip(
        category_order, category_colors)}
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=300)

    # centers for the 2 bars (early, late)
    if x_positions is None:
        x = np.arange(len(M.index))
    else:
        x = np.asarray(x_positions, dtype=float)

    bottoms = np.zeros(len(M.index), float)
    for lbl in category_order:
        vals = M[lbl].to_numpy()
        color = color_map[lbl]
        bars = ax.bar(x, vals, width=bar_width, bottom=bottoms,
                      color=color, edgecolor='white', linewidth=0.7)
        stars = p_to_stars(p_values.get(lbl, np.nan))
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v <= min_pct_to_label:
                continue
            y = bottoms[i] + v * center_factor

            # >>> CHANGE: no bold MathText; plain thin-space + stars <<<
            # \u2009 = thin space
            star_part = f'\u2009{stars}' if stars else ''
            label_txt = f'{lbl}{star_part}'

            ax.text(
                bar.get_x() + bar.get_width()/2.0, y, label_txt,
                ha='center', va='center', fontsize=7.5,
                fontweight='normal',                   # ensure not bold
                color=text_color_for_bg(color)
            )
        bottoms += vals
    # cosmetics
    ax.grid(False)
    ax.set_xticks(x, ['early', 'late'])
    ax.set_xlabel('Session Phase', fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.tick_params(axis='both', direction='out',
                   length=3, width=0.8, labelsize=10)
    ax.set_ylim(0, 1.0)

    # make sure the tight pair is fully visible
    pad = 0.25 if len(x) == 2 else 0.1
    ax.set_xlim(x.min() - pad, x.max() + pad)

    return ax


def plot_outcomes_by_phase_side_by_side(combd_pattern_frequencies, 
                                        category_order, 
                                        title, 
                                        y_label,
                                        id_col='monkey',
                                        id_values=None,
                                        category_colors=None):
    if category_colors is None:
        category_colors = ['#0072B2', '#CC79A7', '#E69F00', '#009E73',
                           '#F0E442', '#56B4E9', '#000000']

    rc_local = {
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
    }
    
    if id_values is None:
        id_values = combd_pattern_frequencies[id_col].unique()
    
    with mpl.rc_context(rc=rc_local):
        # Choose number of columns
        cols = min(len(id_values), 3)            # up to 3 per row
        rows = math.ceil(len(id_values) / cols)  # enough rows to fit everything
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(3.6*cols, 3.6*rows),   # scale figure size with grid
            dpi=300,
            sharey=True,
            gridspec_kw={'wspace': 0.25, 'hspace': 0.25}
        )
        # If only one row/column, ensure axes is iterable
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # tighter spacing within each monkey: close early/late centers and narrow bars
        x_positions = np.array([0.0, 0.42])   # close pair
        bar_width = 0.34

        for ax, monkey in zip(axes, id_values):
            df_monkey = combd_pattern_frequencies[combd_pattern_frequencies[id_col] == monkey].copy(
            )
            phase_df_sub = prepare_for_stacked_bar(df_monkey, category_order)
            p_values = get_p_values(phase_df_sub, category_order)
            M = (phase_df_sub.pivot_table(index='phase', columns='new_label', values='ratio', aggfunc='sum')
                 .reindex(index=['early', 'late'], columns=category_order)
                 .fillna(0.0))
            plot_stacked_bar(
                M, p_values, category_order, category_colors, ax=ax,
                x_positions=x_positions, bar_width=bar_width
            )
            ax.set_title(monkey, weight='bold', pad=6)
            ax.set_xlabel('Session Phase', fontsize=10)

        axes[0].set_ylabel(y_label, fontsize=11)
        fig.suptitle(title, fontsize=13, weight='bold', y=0.998)
        fig.align_ylabels(axes)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig, axes


def prepare_for_stacked_bar_no_phase(df_monkey, category_order):
    df_monkey_sub = df_monkey[df_monkey['new_label'].isin(category_order)].copy()

    # aggregate over items (or any existing grouping that yields frequency + denom_count)
    agg = (
        df_monkey_sub.groupby(['new_label', 'item'], observed=True)[['frequency', 'denom_count']]
        .sum()
        .reset_index(drop=False)
    )

    agg['denom_count'] = agg['denom_count'].astype(int)
    agg['ratio'] = agg['frequency'] / agg['denom_count']

    # now get one ratio per label = sum(freq) / sum(denom)
    M = (
        agg.groupby('new_label')[['frequency', 'denom_count']]
        .sum()
        .assign(ratio=lambda x: x['frequency'] / x['denom_count'])
    )

    # reorder into row vector
    M = M['ratio'].reindex(category_order).fillna(0.0).to_frame().T
    M.index = ['single']   # only one bar
    return M

def plot_stacked_bar_all_ids(df, category_order, title='', y_label='Proportion', id_col='monkey',
                             category_colors=None):
    if category_colors is None:
        category_colors = ['#0072B2', '#CC79A7', '#E69F00', '#009E73',
                           '#F0E442', '#56B4E9', '#000000']
    color_map = {label: color for label, color in zip(category_order, category_colors)}

    ids = df[id_col].unique()
    n = len(ids)

    fig, ax = plt.subplots(figsize=(1.4*n + 2, 4.2), dpi=300)

    # compute each id’s stacked proportions
    matrices = {i: prepare_for_stacked_bar_no_phase(
        df[df[id_col] == i], category_order
    ) for i in ids}

    x_positions = np.arange(n)
    bar_width = 0.75

    bottoms = np.zeros(n, float)
    for lbl in category_order:
        vals = np.array([matrices[i].iloc[0][lbl] for i in ids])
        color = color_map[lbl]
        ax.bar(x_positions, vals, width=bar_width, bottom=bottoms,
               color=color, edgecolor='white', linewidth=0.7)
        bottoms += vals

    # cosmetics
    ax.set_xticks(x_positions, ids)
    ax.set_ylabel(y_label)
    ax.set_title(title, weight='bold')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend (all categories)
    handles = [plt.Rectangle((0,0),1,1,color=color_map[c]) for c in category_order]
    ax.legend(handles, category_order, bbox_to_anchor=(1.02, 1), loc='upper left', title='Category')

    fig.tight_layout()
    plt.show()
    return fig, ax
