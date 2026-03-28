#!/usr/bin/env python3
"""Build agent folder manifest for a given RL agent directory."""
import json
import os
import sys
from pathlib import Path

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

import argparse

from reinforcement_learning.base_classes import rl_base_utils

DEFAULT_OVERALL_FOLDER = (
    "multiff_analysis/RL_models/sb3_stored_models/all_agents"
)


def main():
    parser = argparse.ArgumentParser(description="Build agent folder manifest")
    parser.add_argument(
        "--overall-folder",
        default=DEFAULT_OVERALL_FOLDER,
        help="Path to RL agent folder (relative to project root or absolute)",
    )
    args = parser.parse_args()
    overall_folder = args.overall_folder

    agent_folders = rl_base_utils.get_agent_folders(path=overall_folder)

    manifest_path = os.path.join(overall_folder, "agent_folder_manifest.json")
    
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, 'w') as f:
        json.dump(agent_folders, f, indent=2)

    print(f'Saved manifest with {len(agent_folders)} agents')
    print(f'Manifest path: {manifest_path}')


if __name__ == '__main__':
    main()