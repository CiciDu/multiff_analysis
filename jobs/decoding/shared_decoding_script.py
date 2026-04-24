"""Shared logic for decoding scripts using decoding_pipelines."""

import argparse
from typing import Any, Optional, Type, TypeVar

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_pipelines import (
    decoding_models,
    decoding_runner,
)

TaskT = TypeVar("TaskT")


def run_decoding_main(
    task_class: Type[TaskT],
    *,
    model_class = decoding_models.CVDecodingModel,
    cv_mode: str = "blocked_time_buffered",
    use_spike_history: bool = True,
    run_kwargs: Optional[dict[str, Any]] = None,
    task_kwargs: Optional[dict[str, Any]] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> tuple:
    """
    Common main logic for decoding scripts.

    Args:
        task_class: The decoding task class (e.g. StopTask,
            PNTask, VisTask).
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

    for smoothing_width in [10, 20]:
        print(f"Running with smoothing_width={smoothing_width}")
    
        task_kwargs = task_kwargs or {}
        model_kwargs = model_kwargs or {}
        runner = decoding_runner.DecodingRunner(
            task_class(
            raw_data_folder_path=args.raw_data_folder_path,
            bin_width=args.bin_width,
            smoothing_width=smoothing_width,
            use_spike_history=use_spike_history,
            **task_kwargs,
            ),
            model_class(
                cv_mode=cv_mode,
                **model_kwargs,
            ),
        )

        runner.collect_data(exists_ok=True)

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
        
        # results_df_shuffled = runner.run(
        #     **common_run,
        #     shuffle_mode='timeshift_fold',
        # )

        # results_one_ff_style = runner.run_one_ff_style()

    return 
