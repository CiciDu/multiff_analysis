#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare decoding outputs across models.

Loads CSVs written by the decode job under:
  .../retry_decoder/.../decoding/runs/<model>/*.csv

Then concatenates them, prints a compact summary (peak AUC by model
and comparison), and renders timecourse plots with lines per model.

Example:
  python jobs/scripts/py \
      --base path/to/.../retry_decoder/.../decoding/runs \
      --models svm logreg rf mlp
"""

from __future__ import annotations
import seaborn as sns
import ast

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_data_analysis.neural_analysis_tools.decoding_tools import plot_decoding
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event


def _ensure_methods_on_path() -> None:
    """Make sure `multiff_code/methods` is importable.

    Walk up from CWD until `Multifirefly-Project` is found, then insert
    its `multiff_analysis/multiff_code/methods` to sys.path.
    """
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if p.name == "Multifirefly-Project":
            methods = p / "multiff_analysis" / "multiff_code" / "methods"
            if methods.exists():
                if str(methods) not in sys.path:
                    sys.path.insert(0, str(methods))
            break


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare decoding results across models")
    ap.add_argument(
        "--base",
        type=str,
        default=str(Path("runs")),
        help="Base directory containing per-model result subfolders (e.g., .../retry_decoder/.../decoding/runs)",
    )
    ap.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of model folder names to include (default: all)",
    )
    ap.add_argument(
        "--key",
        type=str,
        default=None,
        help="Filter to a single comparison key (matches df['key'])",
    )
    ap.add_argument(
        "--align",
        type=str,
        choices=("start", "end", "both"),
        default="both",
        help="Which alignment(s) to include",
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional path to write the summary CSV",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, skip plotting and only print summary",
    )
    return ap.parse_args()


def _load_all_results(base_dir: Path, models: Optional[List[str]]) -> pd.DataFrame:
    """Load all per-model CSVs and concatenate with a model_name column.

    Parameters
    ----------
    base_dir : Path
        `runs` directory that contains subdirs per model.
    models : list[str] or None
        If provided, only include these model subdirectories.
    """
    rows: list[pd.DataFrame] = []
    if not base_dir.exists():
        raise FileNotFoundError(f"Base results dir not found: {base_dir}")

    for model_dir in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
        model_name = model_dir.name
        if models and model_name not in models:
            continue

        csv_files = sorted(model_dir.glob("*.csv"))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[warn] Skipping unreadable CSV: {csv_path} ({e})")
                continue

            if "model_name" not in df.columns:
                df["model_name"] = model_name
            rows.append(df)

    if not rows:
        raise RuntimeError(
            f"No CSVs found under {base_dir} for models={models or 'ALL'}"
        )

    df_all = pd.concat(rows, ignore_index=True)
    # Ensure expected numeric columns exist with proper dtype
    for col in ("window_start", "window_end", "mean_auc", "sd_auc"):
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
            
    df_all = df_all.sort_values(by=['key', 'model_name', 'window_start'], ascending=True)
    return df_all


def _apply_filters(df: pd.DataFrame, key: Optional[str], align: str) -> pd.DataFrame:
    dfa = df.copy()
    if key is not None and "key" in dfa.columns:
        dfa = dfa.loc[dfa["key"] == key]

    if align != "both":
        want_end = True if align == "end" else False
        if "align_by_stop_end" in dfa.columns:
            dfa = dfa.loc[dfa["align_by_stop_end"] == want_end]
        else:
            print("[info] align_by_stop_end not present; keeping all alignments")
    return dfa


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a compact summary: peak AUC per model/comparison/alignment."""
    if {"window_start", "window_end", "mean_auc"}.issubset(df.columns):
        df = df.assign(window_center=(
            df["window_start"] + df["window_end"]) / 2.0)

    group_cols = [
        c for c in ("a_label", "b_label", "align_by_stop_end", "model_name") if c in df.columns
    ]
    if not group_cols:
        raise ValueError(
            "Input dataframe lacks necessary columns to summarize.")

    # Peak AUC and where it occurs
    def _agg_peak(g: pd.DataFrame) -> pd.Series:
        idx = g["mean_auc"].idxmax()
        row = g.loc[idx]
        return pd.Series(
            {
                "peak_auc": float(row["mean_auc"]),
                "peak_t": float(
                    row["window_start"]
                    if "window_end" not in row or pd.isna(row["window_end"]) else (row["window_start"] + row["window_end"]) / 2.0
                ),
                "n_windows": int(g.shape[0]),
            }
        )

    summary = df.groupby(group_cols, as_index=False).apply(
        _agg_peak, include_groups=False)
    # Sort: highest AUC first within each comparison/alignment
    sort_cols = [
        c for c in ("a_label", "b_label", "align_by_stop_end", "peak_auc") if c in summary.columns
    ]
    ascending = [True, True, True, False][: len(sort_cols)]
    summary = summary.sort_values(sort_cols, ascending=ascending)
    return summary


def _plot_summary_bars(summary: pd.DataFrame, title_prefix: str = "Peak AUC by model") -> None:
    """Visualize peak AUC summary as grouped bar charts.

    - One subplot per comparison (a_label vs b_label)
    - Bars grouped by model; colors split by alignment (start/end) when present
    """
    if summary.empty:
        return

    # Build a human-readable comparison label
    if {"a_label", "b_label"}.issubset(summary.columns):
        dfp = summary.copy()
        dfp["comparison"] = dfp["a_label"].astype(
            str) + " vs " + dfp["b_label"].astype(str)
    else:
        dfp = summary.copy()
        dfp["comparison"] = "All"

    comparisons = list(dfp["comparison"].unique())
    n = len(comparisons)
    ncols = min(3, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        5 * ncols, 3.8 * nrows), squeeze=False)

    for idx, comp in enumerate(comparisons):
        ax = axes[idx // ncols][idx % ncols]
        sub = dfp.loc[dfp["comparison"] == comp]

        models = sorted(sub["model_name"].unique()
                        ) if "model_name" in sub.columns else ["model"]
        x = np.arange(len(models))

        # Alignment handling
        if "align_by_stop_end" in sub.columns:
            align_values = sorted(
                sub["align_by_stop_end"].dropna().unique().tolist())
            if not align_values:
                align_values = [False]
        else:
            align_values = [False]

        align_labels = {False: "start", True: "end"}
        n_align = len(align_values)
        width = 0.8 / max(1, n_align)

        for i, align_val in enumerate(align_values):
            vals = []
            for m in models:
                if "align_by_stop_end" in sub.columns:
                    rows = sub[(sub["model_name"] == m) & (
                        sub["align_by_stop_end"] == align_val)]
                else:
                    rows = sub[sub["model_name"] == m]
                v = rows["peak_auc"].max() if not rows.empty else np.nan
                vals.append(v)

            offs = x + (i - (n_align - 1) / 2.0) * width
            ax.bar(offs, vals, width=width, label=align_labels.get(
                align_val, str(align_val)))

        ax.set_title(comp)
        ax.set_ylabel("Peak AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        if n_align > 1:
            ax.legend(title="Align")

    # Hide any unused subplots
    total_axes = nrows * ncols
    for j in range(n, total_axes):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title_prefix)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def summarize_and_plot_decoding(raw_data_folder_path, cumulative=False):
    """Load decoding results, summarize AUC, and save timecourse plots for each alignment."""
    # Initialize analysis object
    pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
        raw_data_folder_path=raw_data_folder_path
    )

    # Define result directories
    retry_decoder_dir = pn.retry_decoder_folder_path if not cumulative else pn.retry_decoder_cumulative_folder_path
    job_result_dir = Path(retry_decoder_dir) / 'runs'
    summary_csv = Path(retry_decoder_dir) / 'cross_model' / 'sum.csv'

    # Ensure decoding methods are accessible
    _ensure_methods_on_path()

    # Load all decoding results
    df_all = _load_all_results(job_result_dir, None)
    if df_all.empty:
        raise RuntimeError('No rows matched the provided filters.')

    df_all = df_all[df_all['align_by_stop_end'] == True].copy()

    # Summarize decoding results
    summary = _summarize(df_all)
    # print('\n=== Peak AUC by model and comparison ===')
    # print(summary.to_string(index=False))

    # Save summary CSV
    out_path = Path(summary_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    # print(f'[write] Summary CSV → {out_path}')

    # --- Extract monkey name and session date from path ---
    # Example path: all_monkey_data/raw_monkey_data/monkey_Bruno/data_0301
    raw_path = Path(raw_data_folder_path)
    monkey_part = raw_path.parts[-2]  # 'monkey_Bruno'
    monkey = monkey_part.replace('monkey_', '')  # 'Bruno'
    date_part = raw_path.parts[-1]  # 'data_0301'
    date_str = date_part.replace('data_', '')  # '0301'

    # --- Loop over alignments and save a plot for each ---
    plot_dir = Path('all_monkey_data/retry_decoder/plots') if not cumulative else Path(
        'all_monkey_data/retry_decoder_cumulative/plots')
    plot_dir.mkdir(parents=True, exist_ok=True)

    align_col = 'align_by_stop_end'
    if align_col not in df_all.columns:
        print(
            f"[warn] '{align_col}' not in dataframe; assuming all align_by_stop_end=False")
        df_all[align_col] = False

    for align_val in df_all[align_col].unique():
        suffix = 'end' if align_val else 'start'
        png_path = plot_dir / f'{monkey}_{date_str}_{suffix}.png'

        fig = plot_decoding.plot_decoding_timecourse(
            df_all[df_all[align_col] == align_val],
            groupby_cols=('model_name',),
            split_by=('a_label', 'b_label'),
            align_col=align_col,
            value_col='mean_auc',
            sig_col='sig_ttest' if 'sig_ttest' in df_all.columns else 'sig_FDR',
            err_col='sd_auc' if 'sd_auc' in df_all.columns else None,
            title_prefix=f'{monkey} {date_str}',
            save_path=png_path,  # uses the new save_path parameter
        )
        plt.show()

    print(f'\n[done] Summary and plots generated for {monkey} {date_str}\n')
    return df_all


import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def safe_sorted_categories(series):
    vals = series.dropna().unique()
    str_vals = sorted([v for v in vals if isinstance(v, str)])
    num_vals = sorted([v for v in vals if not isinstance(v, str)])
    return list(num_vals) + str_vals


def plot_best_params_3d(df_all, model_name, param_x=None, param_y=None, param_z=None):
    """Visualize 1–3 hyperparameters for a given model (handles clf__ prefixes, clean ticks)."""
    df = df_all[df_all['model_name'] == model_name].copy()
    if df.empty:
        print(f'No entries for model {model_name}')
        return None


    df['best_params'] = df['best_params'].apply(safe_parse)

    # ---- Normalize keys (strip 'clf__' etc.) ----
    def strip_prefixes(d):
        return {k.split('__')[-1]: v for k, v in d.items()} if isinstance(d, dict) else {}
    df['best_params'] = df['best_params'].apply(strip_prefixes)

    # ---- Extract requested params ----
    params = [p for p in [param_x, param_y, param_z] if p is not None]
    for p in params:
        df[p] = df['best_params'].apply(lambda d: d.get(p, None))

    for p in params:
        df[p] = df[p].apply(lambda v: 'None' if pd.isna(v) else v)
    
    if df.empty:
        print(f'No valid entries for {model_name} with params {params}')
        return None

    # ---- 1️⃣ One param → bar plot ----
    if len(params) == 1:
        p = params[0]
        count_df = df[p].value_counts().reset_index()
        count_df.columns = [p, 'count']
        sns.barplot(x=p, y='count', data=count_df, palette='viridis')
        plt.title(f'{model_name}: Frequency of Best {p}')
        plt.tight_layout()
        plt.show()
        return count_df

    # ---- 2️⃣ Two params → single heatmap ----
    elif len(params) == 2:
        x, y = params
        count_df = df.groupby([x, y]).size().reset_index(name='count')


        count_df[x] = pd.Categorical(count_df[x], categories=safe_sorted_categories(count_df[x]), ordered=True)
        count_df[y] = pd.Categorical(count_df[y], categories=safe_sorted_categories(count_df[y]), ordered=True)

        pivot = count_df.pivot(index=y, columns=x, values='count').fillna(0)
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
        plt.title(f'{model_name}: Frequency of Best Params ({x} vs {y})')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.show()
        return count_df.sort_values('count', ascending=False)

    # ---- 3️⃣ Three params → multiple heatmaps ----
    elif len(params) == 3:
        x, y, z = params
        count_df = df.groupby([x, y, z]).size().reset_index(name='count')

        # Ensure discrete numeric order across all slices
        x_order = safe_sorted_categories(count_df[x])
        y_order = safe_sorted_categories(count_df[y])
        z_order = safe_sorted_categories(count_df[z])

        n = len(z_order)
        ncols = min(n, 4)
        nrows = (n - 1) // ncols + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), sharey=True)
        axes = axes.flatten() if n > 1 else [axes]

        for ax, z_val in zip(axes, z_order):
            subset = count_df[count_df[z] == z_val].copy()

            # Force discrete ordering for consistent ticks
            subset[x] = pd.Categorical(subset[x], categories=x_order, ordered=True)
            subset[y] = pd.Categorical(subset[y], categories=y_order, ordered=True)

            pivot = subset.pivot(index=y, columns=x, values='count').fillna(0)
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax)

            ax.set_title(f'{z} = {z_val}')
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        # Hide unused subplots if any
        for ax in axes[len(z_order):]:
            ax.axis('off')

        plt.suptitle(f'{model_name}: Frequency of Best Params ({x}, {y}, {z})')
        plt.tight_layout()
        plt.show()

        return count_df.sort_values('count', ascending=False)


import json
import ast

def safe_parse(x):
    """Parse either Python-literal dicts or JSON dicts safely."""
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}

    # Try Python literal syntax first
    try:
        return ast.literal_eval(x)
    except Exception:
        pass

    # Try JSON syntax second
    try:
        return json.loads(x)
    except Exception:
        pass

    # Fallback
    return {}
