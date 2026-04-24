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
    from reinforcement_learning.collect_data import agent_patterns_class

    parser = argparse.ArgumentParser(description='Run agent retry analysis')
    parser.add_argument('--agent-path', type=str, required=True,
                        help='Agent folder path to process')
    parser.add_argument('--num-datasets-to-collect', type=int, default=5,
                        help='Number of datasets to collect')
    args = parser.parse_args()

    print(f'[SCRIPT] Agent path: {args.agent_path}')

    #agent = sb3_class.SB3forMultifirefly(model_folder_name=args.agent_path)
    ap = agent_patterns_class.AgentPatterns(model_folder_name=args.agent_path)

    try:
        ap.combine_or_retrieve_patterns_and_features(num_datasets_to_collect=args.num_datasets_to_collect)

    except Exception as e:
        print(f'Error making df related to patterns and features for {args.agent_path}: {e}')
        raise



if __name__ == '__main__':
    main()