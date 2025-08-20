import numpy as np
import matplotlib.pyplot as plt
import math



import numpy as np
import matplotlib.pyplot as plt


def plot_cca_cv_results(
    stats_df, data_type='X1', component=1,
    filter_significant=False, sort_by_significance=False,
    significance_threshold=2, max_vars_per_plot=20,
):
    """
    Plot results from cross-validated CCA:
    - If use_cross_view_corr is True: plot correlation between variables and canonical projections
    - Else: plot canonical correlations

    Parameters:
        stats_df: pandas DataFrame from crossvalidated_cca_analysis
        data_type: 'X1' or 'X2'
        component: canonical component index
        use_cross_view_corr: toggle between loading-style and canonical correlation plots
        filter_significant: show only significant variables (|mean| > threshold * std)
        significance_threshold: std multiplier for significance
        title_prefix: optional string to prepend to title
        max_vars_per_plot: maximum number of variables per subplot
    """

    use_cross_view_corr = False if 'canonical_component' in stats_df.columns else True
    if component < 1:
        raise ValueError("Component must be greater than 0")

    values_train, values_test, errors_train, errors_test, final_labels = _extract_plot_data(
        stats_df, data_type, component, use_cross_view_corr,
        filter_significant, significance_threshold, sort_by_significance
    )

    title = f"Canonical Loading - {data_type} - Component {component}" if not use_cross_view_corr else f"Cross-View Correlation - {data_type}"

    if filter_significant:
        flip_y_axis = True
    else:
        flip_y_axis = False

    _plot_bars(
        values_train, values_test, errors_train, errors_test, final_labels,
        title=title,
        use_cross_view_corr=use_cross_view_corr,
        max_vars_per_plot=max_vars_per_plot,
        flip_y_axis=flip_y_axis,
    )


def _extract_plot_data(
    stats_df, data_type, component,
    use_cross_view_corr, filter_significant, significance_threshold, sort_by_significance
):
    """
    Extract data to be plotted, apply filtering if requested.
    """
    # Filter DataFrame for the specific dataset and component
    if use_cross_view_corr:
        # For cross-view correlations, filter by dataset only (no component)
        df_filtered = stats_df[stats_df['dataset'] == data_type]
    else:
        # For canonical correlations, filter by dataset and component
        df_filtered = stats_df[
            (stats_df['dataset'] == data_type) &
            (stats_df['canonical_component'] == component)
        ]

    # Extract train and test data
    train_data = df_filtered[df_filtered['train_or_test'] == 'train']
    test_data = df_filtered[df_filtered['train_or_test'] == 'test']

    # Get values and errors
    values_train = train_data['mean_corr'].values
    values_test = test_data['mean_corr'].values
    if 'std_corr' in train_data.columns:
        errors_train = train_data['std_corr'].values
        errors_test = test_data['std_corr'].values
    else:
        errors_train = np.zeros_like(values_train)
        errors_test = np.zeros_like(values_test)

    # Get labels (variables)
    labels = train_data['variable'].values

    if filter_significant:
        mask = np.abs(values_test) > significance_threshold * errors_test
        if not np.any(mask):
            print(
                f"No significant {data_type} variables for component {component}.")
            return [], [], [], [], []
        labels = labels[mask]
        values_train = values_train[mask]
        values_test = values_test[mask]
        errors_train = errors_train[mask]
        errors_test = errors_test[mask]

    if sort_by_significance:
        print('sorting by significance')
        # sig_score = np.abs(values_test) / errors_test
        sig_score = np.abs(values_test)
        sorted_idx = np.argsort(-sig_score)
        labels = np.array(labels)[sorted_idx]
        values_train = values_train[sorted_idx]
        values_test = values_test[sorted_idx]
        errors_train = errors_train[sorted_idx]
        errors_test = errors_test[sorted_idx]

    return values_train, values_test, errors_train, errors_test, labels


def _plot_bars(values_train, values_test, errors_train, errors_test, labels, title, use_cross_view_corr, max_vars_per_plot=20,
               flip_y_axis=False):
    """
    Bar plot with train/test mean ± std. Split into multiple subplots if too many variables.
    """
    if len(labels) == 0:
        return

    # Calculate number of subplots needed
    n_vars = len(labels)
    n_plots = math.ceil(n_vars / max_vars_per_plot)
    ylabel = 'Correlation'

    if n_plots == 1:
        # Single plot case
        _plot_single_bar_subplot(
            values_train, values_test, errors_train, errors_test, labels,
            ylabel, title, use_cross_view_corr=use_cross_view_corr
        )
    else:
                # Multiple subplots case
        for i in range(n_plots):
            start_idx = i * max_vars_per_plot
            end_idx = min((i + 1) * max_vars_per_plot, n_vars)

            subplot_title = f"{title} (Part {i+1}/{n_plots})"

            # Create a fresh figure/axis for each part
            fig, ax = plt.subplots(
                1, 1,
                figsize=(8, max(5, 3 + len(values_train[start_idx:end_idx]) * 0.3))
            )

            # Slice values for this chunk
            vtr = values_train[start_idx:end_idx]
            vte = values_test[start_idx:end_idx]
            etr = errors_train[start_idx:end_idx]
            ete = errors_test[start_idx:end_idx]
            lbl = labels[start_idx:end_idx]

            if flip_y_axis:
                vtr, vte, etr, ete, lbl = vtr[::-1], vte[::-1], etr[::-1], ete[::-1], lbl[::-1]

            _plot_single_bar_subplot(
                vtr, vte, etr, ete, lbl,
                ylabel, subplot_title, ax=ax,
                use_cross_view_corr=use_cross_view_corr
            )

            plt.tight_layout()
            plt.show()

        
        # # Multiple subplots case
        # fig, axes = plt.subplots(n_plots, 1, figsize=(
        #     9, (max(5, 3 + len(labels) * 0.1)) * n_plots))
        # if n_plots == 1:
        #     axes = [axes]

        # for i in range(n_plots):
        #     start_idx = i * max_vars_per_plot
        #     end_idx = min((i + 1) * max_vars_per_plot, n_vars)

        #     subplot_title = f"{title} (Part {i+1}/{n_plots})"

        #     if flip_y_axis:
        #         _plot_single_bar_subplot(
        #             values_train[start_idx:end_idx][::-1],
        #             values_test[start_idx:end_idx][::-1],
        #             errors_train[start_idx:end_idx][::-1],
        #             errors_test[start_idx:end_idx][::-1],
        #             labels[start_idx:end_idx][::-1],
        #             ylabel, subplot_title, ax=axes[i],
        #             use_cross_view_corr=use_cross_view_corr
        #         )
        #     else:
        #         _plot_single_bar_subplot(
        #             values_train[start_idx:end_idx],
        #             values_test[start_idx:end_idx],
        #             errors_train[start_idx:end_idx],
        #             errors_test[start_idx:end_idx],
        #             labels[start_idx:end_idx],
        #             ylabel, subplot_title, ax=axes[i],
        #             use_cross_view_corr=use_cross_view_corr
        #         )

        # plt.tight_layout()
        # plt.show()



def _plot_single_bar_subplot(values_train, values_test, errors_train, errors_test, labels, ylabel, title, ax=None, use_cross_view_corr=True):
    """
    Create a single bar subplot with train/test mean ± std.
    """
    if len(labels) == 0:
        return

    if ax is None:
        # Single plot case
        fig, ax = plt.subplots(figsize=(7, max(5, len(labels) * 0.25)))
        single_plot = True
    else:
        single_plot = False

    y = np.arange(len(labels))
    bar_width = 0.35

    if np.all(errors_train == 0) and np.all(errors_test == 0):
        errors_train = None
        errors_test = None
    ax.barh(y - bar_width/2, values_train, bar_width, xerr=errors_train,
            label='Train', alpha=0.7, capsize=2)
    ax.barh(y + bar_width/2, values_test, bar_width, xerr=errors_test,
            label='Test', alpha=0.7, capsize=2)

    ax = _add_lines(ax, y, use_cross_view_corr)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend()

    if single_plot:
        plt.tight_layout()
        plt.show()


def _add_lines(ax, y, use_cross_view_corr=True):
    # add grid lines
    for grid_pos in y[:-1] + 0.5:
        ax.axhline(grid_pos, color='gray', linestyle='--',
                   linewidth=0.7, alpha=0.5)

    ax.grid(False)
    ax.xaxis.grid(True, linestyle=':', alpha=0.7)

    # add vertical line at 0.1
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    ax.axvline(0.1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    if not use_cross_view_corr:
        # also add a line at x = -0.1
        ax.axvline(x=-0.1, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7)

    return ax
