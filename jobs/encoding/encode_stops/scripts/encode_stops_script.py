import argparse
import os
import sys
from pathlib import Path

# -------------------------------------------------------
# Repo path bootstrap (keep this here, not in the library)
# -------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics import encode_stops_pipeline
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis import stop_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_folder_path',
        default=None,
        help='Run encoding only on this raw data folder',
    )
    parser.add_argument('--bin_width', type=float, default=0.04)
    parser.add_argument('--t_max', type=float, default=0.20)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--save_dir', default=None)

    args = parser.parse_args()

    from data_wrangling import combine_info_utils

    raw_data_dir_name = 'all_monkey_data/raw_monkey_data'
    monkey_names = ['monkey_Bruno', 'monkey_Schro']
    tuning_feature_mode = 'boxcar_only'
    
    load_if_exists = True

    all_session_results = {}

    # -------------------------------------------------------
    # If a single session is specified, override loop
    # -------------------------------------------------------
    if args.raw_data_folder_path is not None:
        session_paths = [args.raw_data_folder_path]
    else:
        session_paths = []
        for monkey_name in monkey_names:
            sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(
                raw_data_dir_name, monkey_name
            )
            for _, row in sessions_df.iterrows():
                session_paths.append(
                    os.path.join(
                        raw_data_dir_name,
                        row['monkey_name'],
                        row['data_name'],
                    )
                )

    # -------------------------------------------------------
    # Main loop
    # -------------------------------------------------------
    for raw_data_folder_path in session_paths:

        print('=' * 80)
        print(f'Processing: {raw_data_folder_path}')
        print('=' * 80)

        try:
            prs = stop_parameters.default_prs()

            runner = encode_stops_pipeline.StopEncodingRunner(
                raw_data_folder_path=raw_data_folder_path,
                bin_width=args.bin_width,
                t_max=args.t_max,
                stop_prs=prs,
            )

            runner._collect_data(
                exists_ok=load_if_exists,
                tuning_feature_mode=tuning_feature_mode,
            )

            all_neuron_r2 = runner.crossval_stop_variance_explained_all_neurons(
                lam_f=100.0,
                lam_g=10.0,
                lam_h=10.0,
                lam_p=10.0,
                n_folds=args.n_splits,
                load_if_exists=load_if_exists,
                cv_mode='blocked_time_buffered',
                buffer_samples=20,
                verbose=True,
                plot_cdf=False,
                log_x=False,
            )

            all_session_results[raw_data_folder_path] = all_neuron_r2

        except Exception as e:
            print(f'[ERROR] Failed for {raw_data_folder_path}: {e}')
            import traceback
            traceback.print_exc()

    return


if __name__ == '__main__':
    main()