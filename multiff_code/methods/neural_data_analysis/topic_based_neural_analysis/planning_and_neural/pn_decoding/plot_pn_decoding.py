import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_decoding_heatmaps_with_n(
    results_df,
    ff_visibility_col,
    metric='r2_cv',
    model_col='model_name',
    behav_feature_col='behav_feature',
    n_col='n_samples',
    vmin=0,
    min_vmax=0.1,
    cmap='viridis',
    behav_features_to_plot=None,
):
    """
    Plot heatmaps of decoding performance (e.g. R^2) stratified by FF visibility,
    with sample size annotated in y-axis labels.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of decoding functions.
    ff_visibility_col : str
        Column indicating FF visibility (e.g. 'num_ff_visible' or 'num_ff_in_memory').
    metric : str
        Metric to plot (default: 'r2_cv').
    model_col : str
        Column name for model identifier.
    behav_feature_col : str
        Column name for behavioral feature.
    n_col : str
        Column name for sample size.
    vmin : float
        Lower bound for color scale.
    min_vmax : float
        Minimum upper bound for color scale.
    cmap : str
        Colormap name.
    """

    if behav_features_to_plot is None:
        behav_features_to_plot = results_df[behav_feature_col].unique()

    for behav_feature in behav_features_to_plot:
        sub = results_df.query(f"{behav_feature_col} == @behav_feature")

        # --- pivot metric ---
        df = (
            sub
            .pivot_table(
                index=ff_visibility_col,
                columns=model_col,
                values=metric,
                aggfunc='mean'
            )
            .sort_index()
        )

        # --- compute sample size per row ---
        n_per_row = (
            sub
            .groupby(ff_visibility_col)[n_col]
            .median()   # median is safer than mean
            .loc[df.index]
            .astype(int)
        )

        # --- y-axis labels with sample size ---
        def _fmt_idx(idx):
            return 'any' if idx == -1 else str(idx)

        yticklabels = [
            f'{_fmt_idx(idx)} - {n}'
            for idx, n in zip(df.index, n_per_row)
        ]

        # --- plot ---
        plt.figure(figsize=(1.2 * df.shape[1], 1.2 * df.shape[0]))
        vmax = max(min_vmax, df.max().max())

        ax = sns.heatmap(
            df,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': metric}
        )

        ax.set_yticklabels(yticklabels, rotation=0)

        plt.title(behav_feature)
        plt.ylabel(f'{ff_visibility_col} (with sample size)')
        plt.xlabel('Model')
        plt.tight_layout()
        plt.show()
