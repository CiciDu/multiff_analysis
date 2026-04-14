from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import show_decoding_results


def collect_results_over_param(
    raw_data_dir_name,
    monkey_name,
    task_class,
    param_name,
    param_values,
    shuffle_mode='none',
    fixed_kwargs=None,
):
    if fixed_kwargs is None:
        fixed_kwargs = {}

    results_dict = {}
    for val in param_values:
        kwargs = dict(fixed_kwargs)  # copy
        kwargs[param_name] = val

        results_df = show_decoding_results.collect_all_session_decoding_results(
            raw_data_dir_name,
            monkey_name,
            task_class,
            shuffle_mode=shuffle_mode,
            detrend_spikes=False,
            **kwargs,
        )

        if len(results_df) > 0:
            results_dict[val] = results_df
    return results_dict