#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare decoding outputs across models.

Loads CSVs written by the decode job under:
  multiff_analysis/results/decoding/<model>/*.csv

Then concatenates them, prints a compact summary (peak AUC by model
and comparison), and renders timecourse plots with lines per model.

Example:
  python scripts/cmp_decode.py \
      --base multiff_analysis/results/decoding \
      --models svm logreg rf mlp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        default=str(Path("multiff_analysis") / "results" / "decoding"),
        help="Base directory containing per-model result subfolders",
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
        `results/decoding` directory that contains subdirs per model.
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


def main() -> None:
    args = _parse_args()

    base_dir = Path(args.base)

    _ensure_methods_on_path()
    try:
        from neural_data_analysis.neural_analysis_tools.decoding_tools import plot_decoding
    except Exception as e:
        raise RuntimeError(
            f"Could not import plotting utilities. Ensure methods path is set. ({e})"
        )

    df_all = _load_all_results(base_dir, args.models)
    df_all = _apply_filters(df_all, key=args.key, align=args.align)

    if df_all.empty:
        raise RuntimeError("No rows matched the provided filters.")

    summary = _summarize(df_all)
    print("\n=== Peak AUC by model and comparison ===")
    print(summary.to_string(index=False))

    if args.summary_csv:
        out_path = Path(args.summary_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"[write] Summary CSV → {out_path}")

    if not args.no_plot:
        # Bar charts of peak AUC by model, split per comparison/alignment
        _plot_summary_bars(summary, title_prefix="Peak AUC by model")
        # Overlay timecourses with a line per model, subplots per comparison
        plot_decoding.plot_decoding_timecourse(
            df_all,
            groupby_cols=("model_name",),
            split_by=("a_label", "b_label"),
            align_col="align_by_stop_end",
            value_col="mean_auc",
            sig_col="sig_ttest" if "sig_ttest" in df_all.columns else "sig_FDR",
            err_col="sd_auc" if "sd_auc" in df_all.columns else None,
            title_prefix="Decoding timecourse by model",
        )


if __name__ == "__main__":
    main()
