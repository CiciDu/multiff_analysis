#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_script.py
====================

Cluster-ready decoding analysis script for the Multi-Firefly Project.

Example:
    python decode_script.py --idx 0 --n_jobs 8 \
        --model svm --model_kwargs '{"C":2.0,"gamma":0.05}'
"""

# ----------------------------------------------------------------------
# 1. Environment setup
# ----------------------------------------------------------------------
import os
import sys
from pathlib import Path

# Ensure project root is on path
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == "Multifirefly-Project":
        os.chdir(p)
        sys.path.insert(0, str(p / "multiff_analysis/multiff_code/methods"))
        break

# ----------------------------------------------------------------------
# 2. Core imports
# ----------------------------------------------------------------------
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Decoding utilities
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import compare_events
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    get_stops_utils, collect_stop_data,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools import (
    decoding_utils, decoding_analysis,
)

print("[setup] Environment and imports ready.")


# ----------------------------------------------------------------------
# 3. Data loader
# ----------------------------------------------------------------------
def load_monkey_data(raw_data_path):
    """Load and prepare monkey data for decoding."""
    pn, datasets, comparisons = collect_stop_data.collect_stop_data_func(raw_data_path)
    _ = get_stops_utils.prepare_no_capture_and_captures(
        monkey_information=pn.monkey_information,
        closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
        ff_caught_T_new=pn.ff_caught_T_new,
        distance_col="distance_from_ff_to_stop",
    )
    return pn, datasets, comparisons


# ----------------------------------------------------------------------
# 4. Main decoding runner
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run decoding analysis for the Multi-Firefly project.")
    parser.add_argument("--comparisons", type=str, default="comparisons.json",
                        help="Path to JSON list of comparison dicts.")
    parser.add_argument("--idx", type=int, default=None,
                        help="If provided, only run this comparison index.")
    parser.add_argument("--keys", type=str, nargs="*", default=None,
                        help="Specific comparison keys to run (optional).")
    parser.add_argument("--model", type=str, default="svm",
                        help="Decoder model name (svm, logreg, etc.).")
    parser.add_argument("--model_kwargs", type=str, default="{}",
                        help="Model parameters as JSON string, e.g. '{\"C\":2.0,\"gamma\":0.05}'.")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Parallel threads per job.")
    parser.add_argument("--n_perm", type=int, default=0,
                        help="Number of permutations.")
    parser.add_argument("--do_testing", action="store_true",
                        help="Enable statistical testing.")
    # Tuning toggle (default True to preserve previous behavior)
    parser.add_argument("--tune", dest="tune", action="store_true",
                        help="Enable hyperparameter tuning.")
    parser.add_argument("--no-tune", dest="tune", action="store_false",
                        help="Disable hyperparameter tuning.")
    parser.set_defaults(tune=True)
    parser.add_argument("--raw_data", type=str,
                        default="all_monkey_data/raw_monkey_data/monkey_Bruno/data_0327",
                        help="Path to raw monkey dataset.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"[data] Loading monkey data from: {args.raw_data}")
    pn, datasets, comparisons = load_monkey_data(args.raw_data)

    # Load comparison definitions
    if os.path.exists(args.comparisons):
        with open(args.comparisons, "r") as f:
            all_comps = json.load(f)
        comps = [all_comps[args.idx]] if args.idx is not None else all_comps
    else:
        print(f"[warning] {args.comparisons} not found; using defaults from dataset.")
        comps = comparisons

    keys = args.keys if args.keys is not None else [c["key"] for c in comps]

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError:
        print(f"[warning] Could not parse model_kwargs JSON: {args.model_kwargs}")
        model_kwargs = {}

    # ------------------------------------------------------------------
    # Run decoding
    # ------------------------------------------------------------------
    print(f"[run] Starting decoding with model={args.model}, "
          f"params={model_kwargs}, n_jobs={args.n_jobs}, tune={args.tune}")

    df_all = decoding_analysis.run_all_decoding_comparisons(
        comparisons=comps,
        keys=keys,
        datasets=datasets,
        pn=pn,
        cfg=compare_events.core_stops_psth.PSTHConfig(
            pre_window=0.5, post_window=0.5,
            bin_width=0.05, smoothing_sigma=0.1,
            min_trials=5, normalize="zscore",
        ),
        model_name=args.model,
        model_kwargs=model_kwargs,
        k=3,
        n_perm=args.n_perm,
        alpha=0.05,
        windows=[(round(t, 2), round(t + 0.1, 2)) for t in np.arange(-0.2, 0.3, 0.05)],
        do_testing=args.do_testing,
        plot=False,
        save_dir = 'multiff_analysis/results/decoding',
        overwrite=False,
        tune=args.tune,
    )
    
    print("\n[done] Decoding complete for model:", args.model)


# ----------------------------------------------------------------------
# 5. Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
