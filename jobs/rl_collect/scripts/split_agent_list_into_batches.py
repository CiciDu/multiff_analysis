# File: multiff_analysis/jobs/rl_collect/scripts/split_agent_list_into_batches.py

import json
import math
import os
from pathlib import Path

for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        break


def split_agent_list(input_json, output_dir, batch_size=10):
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json.open('r') as f:
        agents = json.load(f)

    num_batches = math.ceil(len(agents) / batch_size)
    out_files = []

    for i in range(num_batches):
        batch = agents[i*batch_size:(i+1)*batch_size]
        out_path = output_dir / f"dir_batch_{i+1}.json"
        with out_path.open('w') as f:
            json.dump(batch, f, indent=4)
        out_files.append(str(out_path))

    print(f"Created {len(out_files)} agent batches:")
    for f in out_files:
        print("  ", f)

    return out_files


if __name__ == "__main__":
    split_agent_list(
        input_json="multiff_analysis/RL_models/meta/directory_of_agents/dir1.json",
        output_dir="multiff_analysis/RL_models/meta/directory_of_agents/batches",
        batch_size=10
    )
