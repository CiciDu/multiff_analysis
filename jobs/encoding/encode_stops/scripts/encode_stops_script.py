import argparse
import sys
from pathlib import Path
import os 

for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break


from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils



def main():
    encoding_design_utils.bootstrap_repo_path()

    from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics import (
        encode_stops_pipeline,
    )
    from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_folder_path",
        default=None,
        help="Run encoding only on this raw data folder",
    )
    parser.add_argument("--bin_width", type=float, default=0.04)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--save_dir", default=None)

    args = parser.parse_args()

    raw_data_dir_name = "all_monkey_data/raw_monkey_data"
    monkey_names = ["monkey_Bruno", "monkey_Schro"]
    load_if_exists = True

    session_paths = encoding_design_utils.get_session_paths(
        args.raw_data_folder_path, raw_data_dir_name, monkey_names
    )

    all_session_results = {}

    for raw_data_folder_path in session_paths:
        print("=" * 80)
        print(f"Processing: {raw_data_folder_path}")
        print("=" * 80)

        try:
            prs = multiff_encoding_params.default_prs()

            runner = encode_stops_pipeline.StopEncodingRunner(
                raw_data_folder_path=raw_data_folder_path,
                bin_width=args.bin_width,
        
                encoder_prs=prs,
            )

            runner.collect_data(
                exists_ok=load_if_exists,
            )


            all_neuron_r2 = runner.crossval_variance_explained_all_neurons(
                n_folds=args.n_splits,
                load_if_exists=load_if_exists,
                cv_mode="blocked_time_buffered",
                buffer_samples=20,
                verbose=True,
                plot_cdf=False,
                log_x=False,
            )

            all_session_results[raw_data_folder_path] = all_neuron_r2

            runner.run_category_contributions_etc_for_all_neurons(
                n_folds=args.n_splits,
                buffer_samples=20,
                backward_n_folds=10,
                alpha=0.05,
                load_if_exists=load_if_exists,
                verbose=True,
            )

        except Exception as e:
            print(f"[ERROR] Failed for {raw_data_folder_path}: {e}")
            import traceback
            traceback.print_exc()

    return


if __name__ == "__main__":
    main()
