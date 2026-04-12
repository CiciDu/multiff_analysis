"""Shared logic for decoding scripts (decode_stops, decode_pn, decode_vis)."""

import argparse
from typing import Any, Optional, Type, TypeVar

RunnerT = TypeVar("RunnerT")


def run_decoding_main(
    runner_class: Type[RunnerT],
    *,
    cv_mode: str = "group_kfold",
    run_kwargs: Optional[dict[str, Any]] = None,
) -> tuple:
    """
    Common main logic for decoding scripts.

    Args:
        runner_class: The decoding runner class (e.g. StopDecodingRunner,
            PNDecodingRunner, FFVisDecodingRunner).
        run_kwargs: Optional extra kwargs passed to runner.run() (e.g.
            cv_decoding_verbosity=2).

    Returns:
        Tuple of (results_df, results_df_shuffled, results_one_ff_style).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_folder_path',
        default=None,
        help='Run decoding only on this raw data folder',
    )
    parser.add_argument('--bin_width', type=float, default=0.04)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--save_dir', default=None)

    args = parser.parse_args()

    for smoothing_width in [None, 10, 20, 40]:
        print(f"Running with smoothing_width={smoothing_width}")
    
        runner = runner_class(
            raw_data_folder_path=args.raw_data_folder_path,
            bin_width=args.bin_width,
            smoothing_width=smoothing_width,
            cv_mode=cv_mode,
            use_spike_history=True,
        )

        run_kwargs = run_kwargs or {}
        common_run = dict(
            n_splits=args.n_splits,
            save_dir=args.save_dir,
            fit_kernelwidth=True,
            **run_kwargs,
        )


        results_df = runner.run(
            **common_run,
            shuffle_mode='none',
        )
        
        results_df_shuffled = runner.run(
            **common_run,
            shuffle_mode='timeshift_fold',
        )

        # also get ANOVA results
        anova = runner.run_anova_all_neurons(alpha=0.05)
        lm = runner.run_lm_all_neurons(include_all_feats=True, alpha=0.05)

        # results_one_ff_style = runner.run_one_ff_style()

        print(runner.results_df.head())
    return 
