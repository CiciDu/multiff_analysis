import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from .axis_utils import cross_validate_axis

import numpy as np
import matplotlib.pyplot as plt
from .axis_visualization import (
    plot_window_heatmap,
    plot_event_aligned_projection,
    plot_event_projection_hist,
    plot_event_projection_scatter,
)
from .axis_utils import extract_event_windows
from .axis_visualization import plot_window_heatmap


def window_search_dashboard(
    analyzer,
    start_ms_values,
    end_ms_values,
    model="logreg",
    n_splits=5,
    metric="mean_auc",
    heatmap_metric="mean_auc",
    window_ms_for_plot=(-200, 400),
    ci="bootstrap",
    n_boot=500,
    verbose=True,
    plot_heatmap=False,
):
    """
    High-level window search dashboard:
      1) Scan many windows
      2) Plot heatmap of window performance
      3) Select best window by chosen metric
      4) Fit axis for best window
      5) Plot event-aligned projection (bootstrap CI)
      6) Plot distribution & scatter
      7) Print summary

    Parameters
    ----------
    analyzer : ContinuousBehaviorAxisAnalyzer
    start_ms_values, end_ms_values : arrays of window start/end values
    model : str
        'logreg', 'lda', 'ridge'
    metric : str
        Selection metric: 'mean_auc', 'mean_accuracy', 'axis_cosine_similarity'
    heatmap_metric : str
        Metric used in heatmap visualization.
    window_ms_for_plot : tuple(ms_start, ms_end)
        Time window for event-aligned projection.
    ci : 'sem', 'ci95', 'bootstrap'
        Error band type for event-aligned plot.
    n_boot : bootstrap iterations
    """

    # ---------------------------------------------------
    # 1. Full window scan
    # ---------------------------------------------------
    from .window_search import scan_windows  # local import to avoid circular
    df = scan_windows(
        analyzer,
        start_ms_values=start_ms_values,
        end_ms_values=end_ms_values,
        model=model,
        n_splits=n_splits,
        verbose=verbose,
    )

    if df.empty:
        raise ValueError("No windows found. Check grid ranges.")

    # ---------------------------------------------------
    # 2. Heatmap of window performance
    # ---------------------------------------------------
    if plot_heatmap:
        plot_window_heatmap(df, value=heatmap_metric)

    # ---------------------------------------------------
    # 3. Identify best window
    # ---------------------------------------------------
    if metric not in df.columns:
        raise ValueError(f"metric must be one of {list(df.columns)}")

    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]
    best_window = (int(best_row["start_ms"]), int(best_row["end_ms"]))

    if verbose:
        print("\n=== BEST WINDOW ===")
        print(f"Selected by metric: {metric}")
        print(f"Window: {best_window[0]}–{best_window[1]} ms")
        print("Row:", best_row.to_dict())

    # ---------------------------------------------------
    # 4. Fit axis for best window
    # ---------------------------------------------------
    axis_info = analyzer.compute_event_axis(
        window_a_ms=best_window,
        window_b_ms=best_window,
        model=model,
    )

    # ---------------------------------------------------
    # 5. Event-aligned projection
    # ---------------------------------------------------
    # Build aligned matrices
    proj = axis_info["projection"]
    start_ms, end_ms = window_ms_for_plot
    so = int(start_ms / analyzer.bin_width_ms)
    eo = int(end_ms / analyzer.bin_width_ms)
    offsets = np.arange(so, eo)

    a_bins = analyzer._events_to_bins(analyzer.event_a_times)
    b_bins = analyzer._events_to_bins(analyzer.event_b_times)

    # Safe indexing: clip to valid range to avoid negative-wrap
    idx_A = a_bins[:, None] + offsets[None, :]
    idx_B = b_bins[:, None] + offsets[None, :]
    T = len(proj)
    idx_A = np.clip(idx_A, 0, T - 1)
    idx_B = np.clip(idx_B, 0, T - 1)
    aligned_A = proj[idx_A]
    aligned_B = proj[idx_B]

    time_axis = offsets * analyzer.bin_width_s

    plot_event_aligned_projection(
        aligned_A,
        aligned_B,
        time_axis,
        label_a=analyzer.event_a_label,
        label_b=analyzer.event_b_label,
        ci=ci,
        n_boot=n_boot,
        train_window=best_window,
        title=f"{analyzer.event_a_label} vs {analyzer.event_b_label}",
    )

    cv = analyzer.cross_validate_axis(best_window, best_window)
    analyzer.diagnose_axis(axis_info, cv_results=cv, window_ms=best_window)

    return {
        "df": df,
        "best_window": best_window,
        "best_row": best_row,
        "axis_info": axis_info,
    }


# ========================================================================
#  SCAN WINDOW GRID
# ========================================================================
def scan_windows(
    analyzer,
    start_ms_values: List[int],
    end_ms_values: List[int],
    model: str = "logreg",
    n_splits: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Scan a grid of time windows and evaluate axis quality via cross-validation.

    Parameters
    ----------
    analyzer : ContinuousBehaviorAxisAnalyzer
        The analyzer object that provides build_event_vectors().
    start_ms_values : list of int
        Start times (ms) relative to event.
    end_ms_values : list of int
        End times (ms) relative to event.
    model : str
        'logreg', 'lda', or 'ridge'.
    n_splits : int
        Number of cross-validation folds.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    DataFrame with columns:
        - start_ms
        - end_ms
        - duration_ms
        - mean_auc
        - mean_accuracy
        - axis_cosine_similarity
    """

    rows = []

    for s in start_ms_values:
        for e in end_ms_values:

            # Window must be valid
            if e <= s:
                continue

            window = (s, e)

            # Build X, y for this window
            X, y = analyzer.build_event_vectors(window, window)

            # Cross-validate
            cv = cross_validate_axis(X, y, model=model, n_splits=n_splits)

            row = {
                "start_ms": s,
                "end_ms": e,
                "duration_ms": e - s,
                "mean_auc": cv["mean_auc"],
                "mean_accuracy": cv["mean_accuracy"],
                "axis_cosine_similarity": cv["axis_cosine_similarity"],
            }
            rows.append(row)

            if verbose:
                print(
                    f"Window {s:>4d}–{e:<4d} ms → "
                    f"AUC={cv['mean_auc']:.3f}, "
                    f"Acc={cv['mean_accuracy']:.3f}, "
                    f"Cos={cv['axis_cosine_similarity']:.3f}"
                )

    if not rows:
        return pd.DataFrame(columns=[
            "start_ms", "end_ms", "duration_ms",
            "mean_auc", "mean_accuracy", "axis_cosine_similarity"
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_auc", ascending=False).reset_index(drop=True)
    return df


# ========================================================================
#  FIND BEST WINDOW
# ========================================================================
def find_best_window(
    analyzer,
    start_ms_values: List[int],
    end_ms_values: List[int],
    model: str = "logreg",
    n_splits: int = 5,
    metric: str = "mean_auc",
    verbose: bool = True
) -> Dict:
    """
    Run window scan, return best window + axis_info + full table.

    Parameters
    ----------
    analyzer : ContinuousBehaviorAxisAnalyzer
    start_ms_values, end_ms_values : array-like
        Window grid.
    model : str
        'logreg', 'lda', 'ridge'.
    n_splits : int
        Cross-validation folds.
    metric : str
        One of: ['mean_auc', 'mean_accuracy', 'axis_cosine_similarity'].

    Returns
    -------
    dict {
        "best_window": (start_ms, end_ms),
        "best_row": DataFrame row,
        "axis_info": dict from compute_event_axis,
        "df": full scan DataFrame
    }
    """

    df = scan_windows(
        analyzer,
        start_ms_values=start_ms_values,
        end_ms_values=end_ms_values,
        model=model,
        n_splits=n_splits,
        verbose=verbose
    )

    if df.empty:
        raise ValueError("No valid windows found — check start/end ranges.")

    if metric not in df.columns:
        raise ValueError(f"metric must be one of {list(df.columns)}")

    # Identify best-performing window
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]

    best_window = (int(best_row["start_ms"]), int(best_row["end_ms"]))

    if verbose:
        print(
            f"\nBest window by {metric}: "
            f"{best_window[0]}–{best_window[1]} ms  "
            f"({metric}={best_row[metric]:.3f})"
        )

    # Train final axis using this window
    axis_info = analyzer.compute_event_axis(
        window_a_ms=best_window,
        window_b_ms=best_window,
        model=model
    )

    return {
        "best_window": best_window,
        "best_row": best_row,
        "axis_info": axis_info,
        "df": df,
    }
