import io
import os
import sys
import contextlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from reinforcement_learning.agents.feedforward import sb3_env
from reinforcement_learning.base_classes import rl_base_utils
from reinforcement_learning.agents.feedforward import sb3_class

from planning_analysis.agent_analysis import agent_plan_factors_x_agents_class
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils

def _get_sweep_results_save_path(overall_folder_name, save_filename='sweep_results.pkl'):
    '''
    Build the path used to save/retrieve the aggregated dataframe.
    '''
    return os.path.join(overall_folder_name, save_filename)


def _load_saved_sweep_results(save_path):
    '''
    Load a previously saved cost results dataframe if it exists.
    '''
    if os.path.exists(save_path):
        return pd.read_pickle(save_path)
    return None


def _build_single_agent_result(rl, agent_name):
    '''
    Build one row of the summary dataframe for a single agent.
    '''
    num_caught_ff = len(rl.ff_caught_T_sorted)

    if not hasattr(rl, 'env_for_data_collection'):
        rl.load_latest_agent(load_replay_buffer=False)
        rl.env_for_data_collection = sb3_env.CollectInformation(
            **rl.current_env_kwargs
        )

    result = {
        'dv': rl.env_for_data_collection.dv_cost_factor,
        'jerk': rl.env_for_data_collection.jerk_cost_factor,
        'stop': rl.env_for_data_collection.cost_per_stop,
        'num_obs_ff': rl.env_for_data_collection.num_obs_ff,
        'max_in_memory_time': rl.env_for_data_collection.max_in_memory_time,
        'identity_slot_strategy': rl.env_for_data_collection.identity_slot_strategy,
        'identity_slot_base': rl.env_for_data_collection.identity_slot_base,
        'new_ff_scope': rl.env_for_data_collection.new_ff_scope,

        # New sweep params
        'v_noise_std': rl.env_for_data_collection.v_noise_std,
        'w_noise_std': rl.env_for_data_collection.w_noise_std,
        'obs_perc_r': rl.env_for_data_collection.obs_noise.perc_r,
        'obs_perc_th': rl.env_for_data_collection.obs_noise.perc_th,
        'obs_mem_r': rl.env_for_data_collection.obs_noise.mem_r,
        'obs_mem_th': rl.env_for_data_collection.obs_noise.mem_th,

        'num_caught_ff': num_caught_ff,
        'num_stops': (rl.monkey_information['monkey_speeddummy'] == 0).sum(),
        'agent_name': agent_name,
        'seed': rl.env_for_data_collection._default_seed,
    }

    return pd.DataFrame([result])


def collect_sweep_results_df(
    overall_folder_name='RL_models/sb3_stored_models/all_agents/sb3_conditional/ff5_mem2p5_drop_fill_visible_only',
    n_steps=8000,
    retrieve_data_only=False,
    save_df=True,
    load_saved_overall_df=False,
    save_filename='sweep_results.pkl',
    max_n_agents=None,
    verbose=False,
):
    '''
    Collect the agent summary dataframe across all agent folders.

    Parameters
    ----------
    overall_folder_name : str
        Folder containing all agent subfolders.
    n_steps : int
        Number of steps passed to rl.collect_data().
    retrieve_data_only : bool
        Passed to rl.collect_data().
    save_df : bool
        Whether to save the aggregated dataframe to disk.
    load_saved_overall_df : bool
        If True, return the saved dataframe directly when available.
    save_filename : str
        Filename used for saving/loading the dataframe.
    max_n_agents : int or None
        Maximum number of agents to iterate through. If None, use all agents.
    verbose : bool
        If False (default), suppress all output except a single progress line
        ("Processing agent X/N"). If True, print all output as usual.

    Returns
    -------
    sweep_results : pd.DataFrame
        Aggregated dataframe across agents.
    '''
    if max_n_agents is not None:
        base_name, ext = os.path.splitext(save_filename)
        save_filename = f'{base_name}_max{max_n_agents}{ext}'

    save_path = _get_sweep_results_save_path(
        overall_folder_name=overall_folder_name,
        save_filename=save_filename,
    )

    if load_saved_overall_df:
        saved_df = _load_saved_sweep_results(save_path)
        if saved_df is not None:
            print(f'Loaded saved dataframe from: {save_path}')
            return saved_df

    agent_folders = rl_base_utils.get_agent_folders(path=overall_folder_name)

    if max_n_agents is not None:
        agent_folders = agent_folders[:max_n_agents]

    sweep_results = pd.DataFrame()
    n_total = len(agent_folders)

    for i, folder in enumerate(agent_folders):
        print(f'\rProcessing agent {i + 1}/{n_total}', end='', flush=True)
        agent_name = os.path.basename(folder)

        _ctx = (
            contextlib.nullcontext()
            if verbose
            else contextlib.ExitStack()
        )
        if not verbose:
            _ctx.enter_context(contextlib.redirect_stdout(io.StringIO()))
            _ctx.enter_context(contextlib.redirect_stderr(io.StringIO()))

        with _ctx:
            rl = sb3_class.SB3forMultifirefly(overall_folder=folder)

            success = rl.collect_data(
                n_steps=n_steps,
                exists_ok=True,
                save_data=True,
                retrieve_data_only=retrieve_data_only,
                retrieve_ff_dataframe=False,
            )
            if not success:
                continue

            agent_result = _build_single_agent_result(
                rl=rl,
                agent_name=agent_name,
            )

        sweep_results = pd.concat(
            [sweep_results, agent_result],
            ignore_index=True,
        )

    print()  # newline after progress line

    if save_df:
        os.makedirs(overall_folder_name, exist_ok=True)
        sweep_results.to_pickle(save_path)
        print(f'Saved dataframe to: {save_path}')

    return sweep_results


def collect_plan_results_df(
    overall_folder_name='RL_models/sb3_stored_models/all_agents',
    num_datasets_to_collect=10,
    save_df=True,
    load_saved_overall_df=False,
    save_filename='planfactor_results.pkl',
    max_n_agents=None,
    use_stored_data_only=False,
    high_level_only=False,
    verbose=False,
):
    '''
    Collect pooled plan factor results across agents.

    Parameters
    ----------
    overall_folder_name : str
        Folder containing all agent subfolders.
    num_datasets_to_collect : int
        Number of datasets to collect per agent.
    save_df : bool
        Whether to save the aggregated results.
    load_saved_overall_df : bool
        If True, load saved results if available.
    save_filename : str
        Filename for saving/loading.
    max_n_agents : int or None
        Limit number of agents.
    overall_df_exists_ok : bool
        Passed to pfaa method.
    use_stored_data_only : bool
        Passed to pfaa method.
    high_level_only : bool
        Passed to pfaa method.
    verbose : bool
        If False (default), suppress all output from the aggregation and print
        only a start/done status line. If True, print all output as usual.

    Returns
    -------
    agent_median_df : pd.DataFrame
    agent_perc_df : pd.DataFrame
    '''
    import os
    import pandas as pd

    if max_n_agents is not None:
        base_name, ext = os.path.splitext(save_filename)
        save_filename = f'{base_name}_max{max_n_agents}{ext}'

    save_path = os.path.join(overall_folder_name, save_filename)

    # ---------- Load ----------
    if load_saved_overall_df and os.path.exists(save_path):
        print(f'Loaded saved dataframe from: {save_path}')
        saved = pd.read_pickle(save_path)
        return saved['median'], saved['perc']

    # ---------- Build pfaa ----------
    _ctx = contextlib.nullcontext() if verbose else contextlib.ExitStack()
    if not verbose:
        _ctx.enter_context(contextlib.redirect_stdout(io.StringIO()))
        _ctx.enter_context(contextlib.redirect_stderr(io.StringIO()))

    with _ctx:
        pfaa = agent_plan_factors_x_agents_class.PlanFactorsAcrossAgents(
            overall_folder_name=overall_folder_name
        )

    # ---------- Filter agents ----------
    filtered_agent_folders = [
        folder
        for folder in pfaa.agent_folders
        if compare_monkey_and_agent_utils.extract_ff_num(folder)[0] > 1
    ]

    sorted_agent_folders = sorted(
        filtered_agent_folders,
        key=compare_monkey_and_agent_utils.extract_ff_num
    )

    if max_n_agents is not None:
        sorted_agent_folders = sorted_agent_folders[:max_n_agents]

    pfaa.agent_folders = sorted_agent_folders

    # ---------- Run aggregation ----------
    n_agents = len(sorted_agent_folders)
    print(f'Processing {n_agents} agents...', end='', flush=True)

    _ctx2 = contextlib.nullcontext() if verbose else contextlib.ExitStack()
    if not verbose:
        _ctx2.enter_context(contextlib.redirect_stdout(io.StringIO()))
        _ctx2.enter_context(contextlib.redirect_stderr(io.StringIO()))

    with _ctx2:
        pfaa.make_all_ref_pooled_median_x_agents_AND_pooled_perc_x_agents(
            exists_ok=False,
            num_datasets_to_collect=num_datasets_to_collect,
            intermediate_products_exist_ok=True,
            agent_data_exists_ok=True,
            use_stored_data_only=use_stored_data_only,
            high_level_only=high_level_only,
        )

    print(' done.')

    agent_median_df = pfaa.all_ref_pooled_median_x_agents.copy()
    agent_perc_df = pfaa.pooled_perc_x_agents.copy()

    if len(agent_median_df) == 0:
        raise ValueError('No agent data found')

    # ---------- Save ----------
    if save_df:
        os.makedirs(overall_folder_name, exist_ok=True)
        pd.to_pickle(
            {'median': agent_median_df, 'perc': agent_perc_df},
            save_path
        )
        print(f'Saved dataframe to: {save_path}')

    return agent_median_df, agent_perc_df


def collect_pattern_frequencies_df(
    overall_folder_name='RL_models/sb3_stored_models/all_agents',
    retrieve_only=True,
    save_df=True,
    load_saved_overall_df=False,
    save_filename='pattern_frequencies.pkl',
    max_n_agents=None,
    verbose=False,
):
    '''
    Collect pattern frequency dataframe across agents.

    Parameters
    ----------
    overall_folder_name : str
        Folder containing all agent subfolders.
    retrieve_only : bool
        Passed to agent.make_df_related_to_patterns_and_features().
    save_df : bool
        Whether to save the aggregated dataframe.
    load_saved_overall_df : bool
        If True, load saved dataframe if available.
    save_filename : str
        Filename for saving/loading.
    max_n_agents : int or None
        Limit number of agents.
    verbose : bool
        If False (default), suppress all output except a single progress line
        ("Processing agent X/N"). If True, print all output as usual.

    Returns
    -------
    all_agents_pattern_frequencies : pd.DataFrame
    '''
    import os
    import pandas as pd

    if max_n_agents is not None:
        base_name, ext = os.path.splitext(save_filename)
        save_filename = f'{base_name}_max{max_n_agents}{ext}'

    save_path = os.path.join(overall_folder_name, save_filename)

    # ---------- Load ----------
    if load_saved_overall_df and os.path.exists(save_path):
        print(f'Loaded saved dataframe from: {save_path}')
        return pd.read_pickle(save_path)

    # ---------- Get agents ----------
    agent_folders = rl_base_utils.get_agent_folders(path=overall_folder_name)

    if max_n_agents is not None:
        agent_folders = agent_folders[:max_n_agents]

    num_agents = len(agent_folders)

    all_agents_pattern_frequencies = pd.DataFrame()

    # ---------- Loop ----------
    for i, model_folder_name in enumerate(agent_folders):
        print(f'\rProcessing agent {i + 1}/{num_agents}', end='', flush=True)

        _ctx = (
            contextlib.nullcontext()
            if verbose
            else contextlib.ExitStack()
        )
        if not verbose:
            _ctx.enter_context(contextlib.redirect_stdout(io.StringIO()))
            _ctx.enter_context(contextlib.redirect_stderr(io.StringIO()))

        with _ctx:
            agent = sb3_class.SB3forMultifirefly(
                model_folder_name=model_folder_name
            )

            try:
                agent.make_df_related_to_patterns_and_features(
                    retrieve_only=retrieve_only
                )

                df, _ = compare_monkey_and_agent_utils \
                    .add_agent_id_and_essential_agent_params_info_to_df(
                        agent.pattern_frequencies,
                        model_folder_name
                    )

                all_agents_pattern_frequencies = pd.concat(
                    [all_agents_pattern_frequencies, df],
                    axis=0,
                    ignore_index=True,
                )

            except Exception as e:
                if verbose:
                    print(f'Error for {model_folder_name}: {e}')
                continue

    print()  # newline after progress line

    if len(all_agents_pattern_frequencies) == 0:
        print('No agent data found')

    # ---------- Save ----------
    if save_df:
        os.makedirs(overall_folder_name, exist_ok=True)
        all_agents_pattern_frequencies.to_pickle(save_path)
        print(f'Saved dataframe to: {save_path}')

    return all_agents_pattern_frequencies