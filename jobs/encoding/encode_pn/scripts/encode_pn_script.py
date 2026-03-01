import argparse
import sys
from pathlib import Path

# Allow importing encode_common from jobs/encoding/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encode_common import (
    bootstrap_repo_path,
    get_session_paths,
    DEFAULT_LAMBDA_CONFIG,
)


def main():
    bootstrap_repo_path()

    from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics import (
        encode_pn_pipeline,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_folder_path",
        default=None,
        help="Run encoding only on this raw data folder",
    )
    parser.add_argument("--bin_width", type=float, default=0.04)
    parser.add_argument("--t_max", type=float, default=0.20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--save_dir", default=None)

    args = parser.parse_args()

    raw_data_dir_name = "all_monkey_data/raw_monkey_data"
    monkey_names = ["monkey_Bruno", "monkey_Schro"]
    load_if_exists = True

    session_paths = get_session_paths(
        args.raw_data_folder_path, raw_data_dir_name, monkey_names
    )

    all_session_results = {}

    for raw_data_folder_path in session_paths:
        print("=" * 80)
        print(f"Processing: {raw_data_folder_path}")
        print("=" * 80)

        try:
            runner = encode_pn_pipeline.PNEncodingRunner(
                raw_data_folder_path=raw_data_folder_path,
                bin_width=args.bin_width,
                t_max=args.t_max,
            )

            runner._collect_data(exists_ok=load_if_exists)

            all_neuron_r2 = runner.crossval_variance_explained_all_neurons(
                lam_f=DEFAULT_LAMBDA_CONFIG["lam_f"],
                lam_g=DEFAULT_LAMBDA_CONFIG["lam_g"],
                lam_h=DEFAULT_LAMBDA_CONFIG["lam_h"],
                lam_p=DEFAULT_LAMBDA_CONFIG["lam_p"],
                n_folds=args.n_splits,
                load_if_exists=load_if_exists,
                cv_mode="blocked_time_buffered",
                buffer_samples=20,
                verbose=True,
                plot_cdf=False,
                log_x=False,
            )

            all_session_results[raw_data_folder_path] = all_neuron_r2

            runner.run_category_contributions_and_penalty_tuning_all_neurons(
                lambda_config=DEFAULT_LAMBDA_CONFIG,
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
