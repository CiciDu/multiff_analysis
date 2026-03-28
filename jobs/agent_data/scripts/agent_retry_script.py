#!/usr/bin/env python3
"""Run retry-related analysis for one SB3 agent folder."""
import os
import sys
from pathlib import Path

import argparse

# ---------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
else:
    raise RuntimeError('Could not find Multifirefly-Project root')

def main():
    from reinforcement_learning.agents.feedforward import sb3_class
    from planning_analysis.agent_analysis import compare_monkey_and_agent_utils

    parser = argparse.ArgumentParser(description='Run agent retry analysis')
    parser.add_argument('--agent-path', type=str, required=True,
                        help='Agent folder path to process')
    args = parser.parse_args()

    print(f'[SCRIPT] Agent path: {args.agent_path}')

    agent = sb3_class.SB3forMultifirefly(model_folder_name=args.agent_path)

    try:
        agent.make_df_related_to_patterns_and_features(retrieve_only=False)
        agent.pattern_frequencies, _ = compare_monkey_and_agent_utils.add_agent_id_and_essential_agent_params_info_to_df(agent.pattern_frequencies, args.agent_path)
    except Exception as e:
        print(f'Error making df related to patterns and features for {args.agent_path}: {e}')
        raise



if __name__ == '__main__':
    main()