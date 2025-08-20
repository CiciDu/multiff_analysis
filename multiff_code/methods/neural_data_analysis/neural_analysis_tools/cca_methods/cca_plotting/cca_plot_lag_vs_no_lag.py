from neural_data_analysis.neural_analysis_tools.cca_methods.cca_plotting import cca_plot_cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def plot_cca_lag_vs_nolag_and_train_vs_test(
    df, dataset_name, chunk_size=30, alpha=0.8,
    base_width=0.35, narrower_width_ratio=0.4,
    width_per_var=0.25, fig_width=8,
    mode='train_offset',
    title_prefix=''
):

    use_cross_view_corr = False if 'canonical_component' in df.columns else True

    vars_unique = df['variable'].unique()
    num_chunks = int(np.ceil(len(vars_unique) / chunk_size))
    train_test_statuses = ['train', 'test']
    lag_statuses = sorted(df['whether_lag'].unique())

    offset_groups, width_groups, colors = get_plot_config(
        mode, train_test_statuses, lag_statuses)

    for i in range(num_chunks):
        chunk_vars, chunk_df = get_chunk_data(df, vars_unique, i, chunk_size)

        # reverse chunk_vars so that the horizontal bars are drawn from top to bottom
        chunk_vars = chunk_vars[::-1]

        fig, ax = create_plot_axes(
            chunk_vars, chunk_size, width_per_var, fig_width)
        y = np.arange(len(chunk_vars))

        draw_bars(
            ax, chunk_df, chunk_vars, y, offset_groups, width_groups,
            mode, base_width, narrower_width_ratio, alpha, colors
        )

        finalize_plot(ax, y, chunk_vars, dataset_name,
                      i, chunk_size, use_cross_view_corr, title_prefix)
        plt.tight_layout()
        plt.show()
        # print("=" * 150)



def get_plot_config(mode, train_test_statuses, lag_statuses):
    if mode == 'train_offset':
        offset_groups = train_test_statuses
        width_groups = lag_statuses
        colors = {
            ('train', 'no_lag'): '#1f77b4',
            ('train', 'lag'): '#aec7e8',
            ('test', 'no_lag'): '#ff7f0e',
            ('test', 'lag'): '#ffbb78',
        }
    elif mode == 'lag_offset':
        offset_groups = lag_statuses
        width_groups = train_test_statuses
        colors = {
            # ('lag', 'train'): '#1f77b4',
            # ('lag', 'test'): '#d62728',
            # ('no_lag', 'train'): '#2ca02c',
            # ('no_lag', 'test'): '#ff7f0e',
            ('lag', 'test'): '#1f77b4',
            ('lag', 'train'): '#aec7e8',
            ('no_lag', 'test'): '#ff7f0e',
            ('no_lag', 'train'): '#ffbb78',
        }
    else:
        raise ValueError("mode must be 'train_offset' or 'lag_offset'")
    return offset_groups, width_groups, colors


def get_chunk_data(df, vars_unique, chunk_idx, chunk_size):
    chunk_vars = vars_unique[chunk_idx *
                             chunk_size:(chunk_idx + 1) * chunk_size]
    chunk_df = df[df['variable'].isin(chunk_vars)].copy()
    chunk_df['variable'] = pd.Categorical(
        chunk_df['variable'], categories=chunk_vars, ordered=True)
    return chunk_vars, chunk_df


def create_plot_axes(chunk_vars, chunk_size, width_per_var, fig_width):
    fig_height = max(chunk_size * width_per_var, 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    return fig, ax


def draw_bars(
    ax, chunk_df, chunk_vars, y, offset_groups, width_groups,
    mode, base_width, narrower_width_ratio, alpha, colors
):

    corr_var = 'corr' if 'corr' in chunk_df.columns else 'mean_corr'

    n_offsets = len(offset_groups)
    offset_width = base_width * 1.5

    for idx, offset_group in enumerate(offset_groups):
        offset = (idx - (n_offsets - 1) / 2) * offset_width

        for width_group in width_groups:
            subset_df, width, color, alpha_adj = get_bar_attributes(
                chunk_df, chunk_vars, offset_group, width_group,
                mode, base_width, narrower_width_ratio, alpha, colors
            )

            ax.barh(
                y + offset, subset_df[corr_var],
                height=width, alpha=alpha_adj,
                label=f'{offset_group.capitalize()} - {width_group}',
                color=color
            )


def get_bar_attributes(
    chunk_df, chunk_vars, offset_group, width_group,
    mode, base_width, narrower_width_ratio, alpha, colors
):
    if mode == 'train_offset':
        subset_df = chunk_df[
            (chunk_df['train_or_test'] == offset_group) &
            (chunk_df['whether_lag'] == width_group)
        ]
        width = base_width if width_group == 'lag' else base_width * narrower_width_ratio
        alpha_adj = alpha
        key = (offset_group, width_group)
    else:  # lag_offset
        subset_df = chunk_df[
            (chunk_df['whether_lag'] == offset_group) &
            (chunk_df['train_or_test'] == width_group)
        ]
        width = base_width if width_group == 'train' else base_width * narrower_width_ratio
        alpha_adj = alpha
        # alpha_adj = alpha if width_group == 'train' else alpha * 0.7
        key = (offset_group, width_group)

    subset_df = subset_df.set_index('variable').reindex(chunk_vars)
    color = colors.get(key, 'gray')

    return subset_df, width, color, alpha_adj


def finalize_plot(ax, y, chunk_vars, dataset_name, chunk_idx, chunk_size, use_cross_view_corr, title_prefix=''):
    ax.set_title(
        f"{title_prefix} {dataset_name} - Variables {chunk_idx * chunk_size + 1} to {(chunk_idx + 1) * chunk_size}")
    ax.set_yticks(y)
    ax.set_yticklabels(chunk_vars, ha='right')
    ax.set_ylabel("Variable")
    ax.set_xlabel("Correlation")

    ax = cca_plot_cv._add_lines(ax, y, use_cross_view_corr)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Groups',
              loc='center left', bbox_to_anchor=(1.02, 0.5))


def _build_corr_df(corrs, x_cols, train_or_test, whether_lag, dataset):
    return pd.DataFrame({
        'dataset': dataset,
        'train_or_test': train_or_test,
        'corr': corrs,
        'whether_lag': whether_lag,
        'variable': x_cols,
    })
