from __future__ import annotations

from typing import Optional

from data_wrangling import combine_info_utils
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_pipelines import (
    encoding_models,
    encoding_runner,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import plot_gam_fit
import os

_DEFAULT_CV_MODE = "blocked_time_buffered"


def collect_category_ecdf_from_sessions(
    monkey_names,
    *,
    raw_data_dir_name='all_monkey_data/raw_monkey_data',
    task_class=None,
    bin_width=0.04,
    exists_ok=True,
    save_path=None,
    use_neural_coupling=False,
    cv_mode: Optional[str] = None,
):
    """
    Gather category variance ECDF data from all sessions for given monkeys.

    Parameters
    ----------
    monkey_names : list of str
        e.g. ['monkey_Schro'] or ['monkey_Bruno', 'monkey_Schro']
    raw_data_dir_name : str
        Base directory for raw data.
    task_class : class
        e.g. encoding_tasks.StopEncodingTask (uses global if None)
    bin_width : float
    exists_ok : bool
        If True, load from save_path when available; save results after computing.
    save_path : str or Path or None
        Where to save/load. If None and exists_ok, use default cache under multiff_analysis.
    use_neural_coupling : bool
        If True, use coupling subdir in path and pass to runner (matches base_encoding_runner).
    cv_mode : str or None
        Cross-validation mode for the runner and default cache layout. If None, uses
        ``'blocked_time_buffered'`` (same default as ``BaseEncodingRunner``). Default
        cache path includes a ``{cv_mode}/`` segment under the coupling folder.

    Returns
    -------
    session_results_list : list of (session_label, all_results)
    """
    import pickle
    from pathlib import Path

    tc = task_class
    if tc is None:
        tc = globals().get('task_class')
    if tc is None:
        raise ValueError('task_class must be passed or set globally')
    if tc.__name__.endswith("EncodingRunner"):
        raise ValueError(
            "collect_category_ecdf_from_sessions now expects a task class "
            "(e.g., encoding_tasks.StopEncodingTask), not a legacy runner class."
        )

    effective_cv_mode = cv_mode if cv_mode is not None else _DEFAULT_CV_MODE

    # Default save path when exists_ok
    # Uses: .../encoding_outputs/{stop|vis|pn}_encoder_outputs/{coupling|no_coupling}/{cv_mode}/category_ecdf_{monkeys}.pkl
    if save_path is None:
        # Map runner class to encoder type
        tc_name = tc.__name__
        if 'Stop' in tc_name:
            encoder_dir = 'stop_encoder_outputs'
        elif 'Vis' in tc_name:
            encoder_dir = 'vis_encoder_outputs'
        elif 'PN' in tc_name:
            encoder_dir = 'pn_encoder_outputs'
        else:
            encoder_dir = f'{tc_name}_encoder_outputs'

        base = raw_data_dir_name.replace('raw_monkey_data', 'planning_and_neural')
        coupling_subdir = "coupling" if use_neural_coupling else "no_coupling"
        # Use first monkey for path; filename includes all monkeys for multi-monkey runs
        monkey_for_path = sorted(monkey_names)[0]
        monkeys_str = '_'.join(sorted(monkey_names)).replace(' ', '_')
        cache_dir = (
            Path(base)
            / monkey_for_path
            / "combined_data"
            / "encoding_outputs"
            / encoder_dir
            / coupling_subdir
            / str(effective_cv_mode)
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_path = cache_dir / f"category_ecdf_{monkeys_str}.pkl"
    elif save_path is not None:
        save_path = Path(save_path)

    def _augment_unassigned_vars_if_present(session_results_list):
        """
        Backfill `unassigned_vars` from saved category_contributions CSV files.

        This keeps older cached `category_ecdf_*.pkl` files compatible after
        adding `unassigned_vars` support, without forcing a full recompute.
        """
        import csv

        for _, all_results in session_results_list:
            for res in all_results:
                cc = res.setdefault('category_contributions', {})
                if 'unassigned_vars' in cc:
                    continue

                contrib_csv = res.get('category_contributions_csv', None)
                if contrib_csv is None or not Path(contrib_csv).exists():
                    continue

                try:
                    with open(contrib_csv, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            category_name = row.get('category', None)
                            if category_name != 'unassigned_vars':
                                continue
                            parsed_row = {}
                            for key, value in row.items():
                                if key == 'category' or value in (None, '', 'nan', 'NaN'):
                                    continue
                                if key == 'vars':
                                    parsed_row[key] = value
                                    continue
                                try:
                                    parsed_row[key] = float(value)
                                except (TypeError, ValueError):
                                    parsed_row[key] = value
                            if parsed_row:
                                cc['unassigned_vars'] = parsed_row
                            break
                except Exception:
                    continue

        return session_results_list

    def _filter_empty_categories(session_results_list):
        """Remove category_contributions with no valid data so empty subplots are not created."""
        default_metric = 'delta_pseudo_r2_clip_leave'

        def _has_valid_value(v):
            if v is None:
                return False
            if isinstance(v, float) and v != v:  # NaN
                return False
            return True

        for _, all_results in session_results_list:
            for res in all_results:
                cc = res.get('category_contributions', {})
                if not cc:
                    continue
                filtered = {
                    cat: metrics
                    for cat, metrics in cc.items()
                    if isinstance(metrics, dict)
                    and default_metric in metrics
                    and _has_valid_value(metrics[default_metric])
                }
                res['category_contributions'] = filtered
        return session_results_list

    if exists_ok and save_path is not None and save_path.exists():
        with open(save_path, 'rb') as f:
            session_results_list = pickle.load(f)
        print(f'[collect_category_ecdf] Loaded from {save_path}')
        session_results_list = _augment_unassigned_vars_if_present(session_results_list)
        return _filter_empty_categories(session_results_list)

    session_results_list = []
    for monkey_name in monkey_names:
        sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(
            raw_data_dir_name, monkey_name
        )
        for _, row in sessions_df.iterrows():
            raw_data_folder_path = os.path.join(
                raw_data_dir_name, row['monkey_name'], row['data_name']
            )
            session_label = f"{row['monkey_name']}/{row['data_name']}"
            print('=' * 80)
            print(f'Processing: {raw_data_folder_path}')
            print('=' * 80)
            try:
                prs = multiff_encoding_params.default_prs()
                task = tc(
                    raw_data_folder_path=raw_data_folder_path,
                    bin_width=bin_width,
                    encoder_prs=prs,
                )
                model = encoding_models.PGAMModel(cv_mode=effective_cv_mode)
                runner = encoding_runner.EncodingRunner(task, model)
                all_results = plot_gam_fit.run_unit_ecdf_collect(runner, 
                                                                 use_neural_coupling=use_neural_coupling)
                session_results_list.append((session_label, all_results))
            except Exception as e:
                print(f'[ERROR] Failed for {raw_data_folder_path}: {e}')
                continue

    session_results_list = _augment_unassigned_vars_if_present(session_results_list)
    session_results_list = _filter_empty_categories(session_results_list)

    if session_results_list:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(session_results_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[collect_category_ecdf] Saved to {save_path}')

    return session_results_list