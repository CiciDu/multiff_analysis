from pathlib import Path
import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import hashlib
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
# isort: off
# fmt: off
from neural_data_analysis.neural_analysis_tools.decoding_tools.event_decoding import (
    decoding_utils, decoding_analysis,
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    get_stops_utils, collect_stop_data,
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import compare_events
# fmt: on
# isort: on

# ----------------------------------------------------------------------
# Project path setup
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Helper: load monkey data
# ----------------------------------------------------------------------
def load_monkey_data(raw_data_folder_path):
    """Load and prepare monkey data for decoding."""
    pn, datasets, comparisons = collect_stop_data.collect_stop_data_func(
        raw_data_folder_path)
    get_stops_utils.prepare_no_capture_and_captures(
        monkey_information=pn.monkey_information,
        closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
        ff_caught_T_new=pn.ff_caught_T_new,
        distance_col='distance_from_ff_to_stop',
    )
    return pn, datasets, comparisons


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run decoding analysis for the Multi-Firefly project.'
    )
    parser.add_argument('--comparisons', type=str, default='comparisons.json',
                        help='Path to JSON list of comparison dicts.')
    parser.add_argument('--idx', type=int, default=None,
                        help='If provided, only run this comparison index.')
    parser.add_argument('--keys', type=str, nargs='*', default=None,
                        help='Specific comparison keys to run (optional).')
    parser.add_argument('--model', type=str, default='svm',
                        help='Decoder model name (svm, logreg, etc.).')
    parser.add_argument('--model_kwargs', type=str, default='{}',
                        help='Model parameters as JSON string, e.g. {"C":2.0,"gamma":0.05}.')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Parallel threads per job.')
    parser.add_argument('--n_perm', type=int, default=0,
                        help='Number of permutations.')
    parser.add_argument('--do_testing', action='store_true',
                        help='Enable statistical testing.')
    parser.add_argument('--exists_ok', action='store_true',
                        help='Skip comparisons whose outputs already exist.')
    parser.add_argument('--tune', dest='tune', action='store_true',
                        help='Enable hyperparameter tuning.')
    parser.add_argument('--no-tune', dest='tune', action='store_false',
                        help='Disable hyperparameter tuning.')
    parser.set_defaults(tune=True)
    parser.add_argument('--perm_search', type=str, default=None,
                        help='Permutation search method (grid, random, None).')
    parser.add_argument('--cumulative', action='store_true',
                        help='Use cumulative decoding (features grow across windows).')
    parser.add_argument('--raw_data', type=str,
                        default='all_monkey_data/raw_monkey_data/monkey_Bruno/data_0328',
                        help='Path to raw monkey dataset.')
    parser.add_argument('--iterate-all', dest='iterate_all', action='store_true',
                        help='Iterate all raw_data folders under a monkey directory.')
    parser.add_argument('--monkey_dir', type=str, default=None,
                        help="Path to 'all_monkey_data/raw_monkey_data/monkey_*' directory.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Inner helper: single session run
    # ------------------------------------------------------------------
    def run_once(raw_data_folder_path: str):
        print(f'[data] Loading monkey data from: {raw_data_folder_path}')

        # Derive output directory
        replace_target = '/retry_decoder_cumulative/' if args.cumulative else '/retry_decoder/'
        session_retry_dir = Path(str(raw_data_folder_path).replace(
            '/raw_monkey_data/', replace_target
        ))
        model_dir_pre = session_retry_dir / 'runs' / args.model

        # Skip if all results already exist
        if args.exists_ok and model_dir_pre.exists():
            keys_for_check = args.keys
            if keys_for_check is None and os.path.exists(args.comparisons):
                try:
                    with open(args.comparisons, 'r') as f:
                        all_comps = json.load(f)
                    comps_tmp = [all_comps[args.idx]
                                 ] if args.idx is not None else all_comps
                    keys_for_check = [c['key']
                                      for c in comps_tmp if 'key' in c]
                except Exception:
                    keys_for_check = None

            if keys_for_check and all(
                decoding_analysis._should_skip_existing_results(model_dir_pre, k, True)[1] and
                decoding_analysis._should_skip_existing_results(
                    model_dir_pre, k, False)[1]
                for k in keys_for_check
            ):
                print('[skip] All requested keys already exist. Skipping.')
                return

        pn, datasets, comparisons = load_monkey_data(raw_data_folder_path)

        # Load or fall back to default comparisons
        comps = None
        if os.path.exists(args.comparisons):
            try:
                with open(args.comparisons, 'r') as f:
                    all_comps = json.load(f)
                comps = [all_comps[args.idx]
                         ] if args.idx is not None else all_comps
            except json.JSONDecodeError as e:
                print(f'[warning] Could not parse {args.comparisons}: {e}')
        if comps is None:
            comps = comparisons

        keys = args.keys if args.keys else [c['key'] for c in comps]

        try:
            model_kwargs_local = json.loads(args.model_kwargs)
        except json.JSONDecodeError:
            print(
                f'[warning] Could not parse model_kwargs: {args.model_kwargs}')
            model_kwargs_local = {}

        # Avoid nested parallelism: keep model kwargs single-threaded
        model_kwargs_local.pop('n_jobs', None)

        print(f'[run] Starting decoding with model={args.model}, '
              f'params={model_kwargs_local}, n_jobs={args.n_jobs}, tune={args.tune}, '
              f'mode={"cumulative" if args.cumulative else "windowed"}')

        # Prepare output directory
        base_retry = Path(pn.retry_decoder_folder_path)
        if args.cumulative:
            base_retry = Path(str(base_retry).replace(
                '/retry_decoder/', '/retry_decoder_cumulative/'))
        out_base = base_retry / 'runs'
        out_base.mkdir(parents=True, exist_ok=True)

        # Run decoding
        decode_fn = (
            decoding_analysis.run_all_decoding_comparisons_cumulative
            if args.cumulative else
            decoding_analysis.run_all_decoding_comparisons
        )
        decode_fn(
            comparisons=comps,
            keys=keys,
            datasets=datasets,
            pn=pn,
            cfg=compare_events.core_stops_psth.PSTHConfig(
                pre_window=0.5, post_window=0.5,
                bin_width=0.05, smoothing_sigma=0.1,
                min_trials=5, normalize='zscore',
            ),
            model_name=args.model,
            model_kwargs=model_kwargs_local,
            k=5,
            n_perm=args.n_perm,
            alpha=0.05,
            windows=[(round(t, 2), round(t + 0.1, 2))
                     for t in np.arange(-0.2, 0.3, 0.05)],
            do_testing=args.do_testing,
            plot=False,
            save_dir=out_base,
            overwrite=False,
            exists_ok=args.exists_ok,
            session_info={
                'monkey': Path(raw_data_folder_path).parent.name,
                'session': Path(raw_data_folder_path).name,
            },
            tune=args.tune,
            perm_search=args.perm_search,
            n_jobs=args.n_jobs,
        )

        print(f'\n[done] Decoding complete for: {raw_data_folder_path}')

    # ------------------------------------------------------------------
    # Iterate across all sessions (optional)
    # ------------------------------------------------------------------
    if args.iterate_all:
        print(f'[iterate] Iterating across all sessions: {args.raw_data}')
        monkey_dir = Path(args.monkey_dir or Path(args.raw_data).parent)
        if not monkey_dir.exists():
            raise FileNotFoundError(
                f'Monkey directory not found: {monkey_dir}')

        all_raw_dirs = sorted([p.name for p in monkey_dir.iterdir()
                               if p.is_dir() and p.name.startswith('data_')])
        if not all_raw_dirs:
            print(f'[warning] No data_* folders found under: {monkey_dir}')
            return

        replace_target = '/retry_decoder_cumulative/' if args.cumulative else '/retry_decoder/'
        retry_monkey_dir = Path(str(monkey_dir).replace(
            '/raw_monkey_data/', replace_target
        ))
        retry_monkey_dir.mkdir(parents=True, exist_ok=True)
        # Build model-/key-specific progress filename to avoid cross-model/key collisions
        model_tag = str(args.model).replace('/', '_')
        if args.keys:
            keys_sorted = sorted(args.keys)
            keys_str = ','.join(keys_sorted)
            keys_hash = hashlib.md5(keys_str.encode('utf-8')).hexdigest()[:8]
            progress_filename = f"_decoding_progress_{model_tag}_k{keys_hash}.json"
        else:
            progress_filename = f"_decoding_progress_{model_tag}.json"
        progress_path = retry_monkey_dir / progress_filename

        progress = {
            'monkey': monkey_dir.name,
            'all': all_raw_dirs,
            'done': [],
            'pending': all_raw_dirs,
            'last_updated': None,
            'model': args.model,
            'keys': args.keys,
        }

        # Load progress if exists
        if progress_path.exists():
            try:
                with open(progress_path, 'r') as f:
                    saved = json.load(f)
                done_saved = saved.get('done', [])
                progress['done'] = [d for d in done_saved if d in all_raw_dirs]
                progress['pending'] = [
                    d for d in all_raw_dirs if d not in progress['done']]
                print(f'[iterate] Progress loaded: {progress}')
            except Exception as e:
                print(f'[warning] Could not read progress file: {e}')
                traceback.print_exc()

        print(f'[iterate] Monkey: {monkey_dir.name} | total={len(all_raw_dirs)} | '
              f'done={len(progress["done"])} | pending={len(progress["pending"])}')

        for data_name in progress['pending'][:]:
            raw_path = str(monkey_dir / data_name)
            print(f'\n[iterate] Running: {data_name}')
            try:
                run_once(raw_path)
                if data_name not in progress['done']:
                    progress['done'].append(data_name)
                progress['pending'] = [
                    d for d in all_raw_dirs if d not in progress['done']]
                progress['last_updated'] = datetime.now(
                ).isoformat(timespec='seconds')

                with open(progress_path, 'w') as f:
                    json.dump(progress, f, indent=2)

                print(f'[iterate] Marked done: {data_name} | '
                      f'remaining={len(progress["pending"])}')

            except Exception as e:
                print(
                    f'[error] Failed on {data_name}: {e}. Moving to next session.')
                traceback.print_exc()
                continue

        print(f'\n[iterate] Completed. done={len(progress["done"])} | '
              f'pending={len(progress["pending"])}')
    else:
        print(f'[run] Only running for one session: {args.raw_data}')
        run_once(args.raw_data)


if __name__ == '__main__':
    main()
