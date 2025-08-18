import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

def reorder_based_on_clustering(data):
    clust = sns.clustermap(data, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
    plt.close()  # don't show this figure
    row_order = clust.dendrogram_row.reordered_ind
    col_order = clust.dendrogram_col.reordered_ind
    # if data is df
    if isinstance(data, pd.DataFrame):
        sorted_data = data.iloc[row_order, col_order]
    else:
        sorted_data = data[row_order, :][:, col_order]
    
    return sorted_data, row_order, col_order

def plot_correlation_heatmap(corr_df, annotation_threshold=0.3):
    
    sorted_corr, row_order, col_order = reorder_based_on_clustering(corr_df)
    # sorted_corr, row_order, col_order = _reorder_based_on_clustering(corr_df)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 15))

    vmin, vmax = -1, 1
    im = ax.imshow(sorted_corr, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # Annotate
    annotate_heatmap(sorted_corr, annotation_threshold=annotation_threshold, vmin=vmin, vmax=vmax, ax=ax, im=im)

    # Set ticks
    ax.set_xticks(np.arange(sorted_corr.shape[1]))
    ax.set_yticks(np.arange(sorted_corr.shape[0]))
    ax.set_xticklabels(sorted_corr.columns, rotation=90)
    ax.set_yticklabels(sorted_corr.index)

    # Gridlines and layout
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()
    

def annotate_heatmap(corr_df, annotation_threshold, vmin, vmax, ax, im, fmt="{:.2f}", fontsize=8):
    num_rows, max_cols = corr_df.shape
    for row in range(num_rows):
        for col in range(max_cols):
            val = corr_df.iloc[row, col]
            if pd.isna(val) or abs(val) < annotation_threshold:
                continue
            # Normalize value for colormap
            normed_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            bg_color = im.cmap(normed_val)
            brightness = 0.299*bg_color[0] + 0.587*bg_color[1] + 0.114*bg_color[2]
            text_color = 'white' if brightness < 0.4 else 'black'
            ax.text(col, row, fmt.format(val), ha='center', va='center', fontsize=fontsize, color=text_color)
    return ax


def _reorder_based_on_clustering(data):
    """Reorder data based on hierarchical clustering using sklearn's AgglomerativeClustering.

    This is much faster than seaborn.clustermap as it doesn't create visualizations.
    """
    # Use fewer clusters for more meaningful grouping
    # Group into ~20 clusters or half the features
    n_row_clusters = min(20, data.shape[0] // 2)
    # Group into ~10 clusters or half the components
    n_col_clusters = min(10, data.shape[1] // 2)

    # Cluster rows (features)
    row_cluster = AgglomerativeClustering(
        n_clusters=n_row_clusters, compute_full_tree=True, linkage='ward')
    row_labels = row_cluster.fit_predict(data)

    # Cluster columns (components)
    col_cluster = AgglomerativeClustering(
        n_clusters=n_col_clusters, compute_full_tree=True, linkage='ward')
    col_labels = col_cluster.fit_predict(data.T)

    # Get ordering within each cluster, then order clusters by their mean values
    row_order = _get_hierarchical_order(data, row_labels, axis=0)
    col_order = _get_hierarchical_order(data.T, col_labels, axis=0)

    data = data.iloc[row_order, col_order]
    return data, row_order, col_order


def _get_hierarchical_order(data, cluster_labels, axis=0):
    """Get hierarchical ordering within clusters and between clusters."""
    unique_clusters = np.unique(cluster_labels)
    final_order = []

    # Sort clusters by their mean values
    cluster_means = []
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_mean = np.mean(
            data[cluster_mask] if axis == 0 else data[:, cluster_mask])
        cluster_means.append((cluster, cluster_mean))

    # Sort clusters by mean value
    cluster_means.sort(key=lambda x: x[1], reverse=True)

    # Within each cluster, sort by individual values
    for cluster, _ in cluster_means:
        cluster_mask = cluster_labels == cluster
        cluster_indices = np.where(cluster_mask)[0]

        if axis == 0:
            cluster_values = np.mean(data[cluster_mask], axis=1)
        else:
            cluster_values = np.mean(data[:, cluster_mask], axis=0)

        # Sort within cluster
        within_cluster_order = np.argsort(cluster_values)[
            ::-1]  # Descending order
        final_order.extend(cluster_indices[within_cluster_order])

    return final_order