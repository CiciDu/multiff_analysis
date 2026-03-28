#!/usr/bin/env python3
"""Run One-FF population decoding (canoncorr + linear readout) for a session."""
import os
import sys
from pathlib import Path

import argparse
import matplotlib
import numpy as np
import pandas as pd

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

from planning_analysis.agent_analysis import agent_plan_factors_x_sess_class, agent_plan_factors_x_agents_class

def main():
    parser = argparse.ArgumentParser(description='Run agent planning analysis')
    parser.add_argument('--agent-path', type=str, default=None,
                        help='Optional: explicit agent folder path (enables direct processing)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Optional: explicit save directory for outputs')
    # add the argument num_datasets_to_collect and num_steps_per_dataset
    parser.add_argument('--num-datasets-to-collect', type=int, default=10,
                        help='Number of datasets to collect')
    parser.add_argument('--num-steps-per-dataset', type=int, default=9000,
                        help='Number of steps per dataset')
    args = parser.parse_args()

    print(f'[SCRIPT] Agent path: {args.agent_path}')
    
    # If agent path is provided, process directly using PlanFactorsAcrossAgentSessions
    if args.agent_path:
        print(f'[SCRIPT] Agent path provided: {args.agent_path}')
        print(f'[SCRIPT] Processing agent directly...')
        
        pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(
            model_folder_name=args.agent_path)
        
        # Determine save directory
        if args.save_dir:
            save_dir = args.save_dir
        else:
            # Infer from agent path structure
            save_dir = args.agent_path.replace(
                'all_agents',
                'all_collected_data/planning/combined_data_x_agents'
            )
        
        print(f'[SCRIPT] Save directory: {save_dir}')
        
        pfas.process_and_save(
            save_dir=save_dir,
            intermediate_products_exist_ok=True,
            agent_data_exists_ok=True,
            num_steps_per_dataset=args.num_steps_per_dataset,
            num_datasets_to_collect=args.num_datasets_to_collect,
            use_stored_data_only=False
        )


if __name__ == '__main__':
    main()