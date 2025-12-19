import argparse
import json
import logging
import os
import sys
import time as time_package
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  # imported but not used directly; kept for completeness

# ---------------------------------------------------------------------
# Find project root "Multifirefly-Project" and set paths correctly
# ---------------------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break

# isort: off
# fmt: off
from planning_analysis.plan_factors import monkey_plan_factors_x_sess_class  # noqa: F401
from planning_analysis.factors_vs_indicators import process_variations_utils  # noqa: F401
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_utils  # noqa: F401
from planning_analysis.agent_analysis import compare_monkey_and_agent_utils, agent_plan_factors_x_sess_class
from reinforcement_learning.base_classes import rl_base_utils
# fmt: on
# isort: on

os.environ.setdefault('PYTORCH_DISABLE_DYNAMO', '1')


# =====================================================================
# Pipeline class
# =====================================================================

class AgentProcessingPipeline:
    def __init__(
        self,
        agent_list_json: Path,
        log_dir: Path,
        num_datasets_to_collect: int,
        num_steps_per_dataset: int,
        intermediate_products_exist_ok: bool,
        agent_data_exists_ok: bool,
        force_reprocess: bool,
    ):
        self.agent_list_json = Path(agent_list_json)
        self.log_dir = Path(log_dir)
        self.num_datasets_to_collect = num_datasets_to_collect
        self.num_steps_per_dataset = num_steps_per_dataset
        self.intermediate_products_exist_ok = intermediate_products_exist_ok
        self.agent_data_exists_ok = agent_data_exists_ok
        self.force_reprocess = force_reprocess

        self.agent_folders = self._load_agent_folders()
        self.logger = self._setup_logging()

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    def _load_agent_folders(self):
        if not self.agent_list_json.is_file():
            raise FileNotFoundError(
                f'Agent list JSON not found: {self.agent_list_json}')

        with self.agent_list_json.open('r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f'Agent list JSON must contain a list of folder paths, got: {type(data)}')

        folders = [str(Path(d)) for d in data]
        return folders

    def _setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp + optional Slurm job ID for unique log filename
        timestamp = time_package.strftime('%Y%m%d_%H%M%S')
        slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
        if slurm_job_id is not None:
            log_name = f'agent_pipeline_job{slurm_job_id}_{timestamp}.log'
        else:
            log_name = f'agent_pipeline_{timestamp}.log'

        log_path = self.log_dir / log_name

        logger = logging.getLogger('agent_pipeline')
        logger.setLevel(logging.INFO)
        logger.propagate = False  # avoid double logging if root logger is configured

        # Clear existing handlers if re-instantiated
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # File handler
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream handler (stdout, Slurm-safe)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        logger.info(f'Logging to {log_path}')
        logger.info(
            f'Loaded {len(self.agent_folders)} agent folders from {self.agent_list_json}')

        return logger

    # -----------------------------------------------------------------
    # Restart / skip logic
    # -----------------------------------------------------------------
    def _done_marker_path(self, agent_folder: str) -> Path:
        # Simple marker file placed in the agent folder
        return Path(agent_folder) / 'agent_plan_factors_done.json'

    def already_processed(self, agent_folder: str) -> bool:
        if self.force_reprocess:
            return False
        done_path = self._done_marker_path(agent_folder)
        return done_path.is_file()

    def _write_done_marker(self, agent_folder: str, params: dict):
        done_path = self._done_marker_path(agent_folder)
        info = {
            'agent_folder': agent_folder,
            'timestamp': time_package.strftime('%Y-%m-%d %H:%M:%S'),
            'env_params_keys': sorted(list(params.keys())),
        }
        with done_path.open('w') as f:
            json.dump(info, f, indent=2)

    # -----------------------------------------------------------------
    # Per-agent processing
    # -----------------------------------------------------------------
    def process_agent(self, agent_folder: str):
        self.logger.info(f'Processing agent folder: {agent_folder}')

        # Read manifest and env params
        manifest = rl_base_utils.read_checkpoint_manifest(agent_folder)
        if isinstance(manifest, dict) and ('env_params' in manifest):
            params = manifest['env_params']
        else:
            msg = f'No env params found in manifest for folder: {agent_folder}'
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Compute agent plan factors
        pfas = agent_plan_factors_x_sess_class.PlanFactorsAcrossAgentSessions(
            model_folder_name=agent_folder
        )

        self.logger.info(
            f'Running streamline_getting_y_values for {agent_folder} '
            f'num_datasets_to_collect={self.num_datasets_to_collect}, '
            f'num_steps_per_dataset={self.num_steps_per_dataset}'
        )

        pfas.streamline_getting_y_values(
            model_folder_name=agent_folder,
            intermediate_products_exist_ok=self.intermediate_products_exist_ok,
            agent_data_exists_ok=self.agent_data_exists_ok,
            num_datasets_to_collect=self.num_datasets_to_collect,
            num_steps_per_dataset=self.num_steps_per_dataset,
            **params,
        )

        # Attach essential meta-info
        agent_all_ref_pooled_median_info = rl_base_utils.add_essential_agent_params_info(
            pfas.all_ref_pooled_median_info,
            params,
        )
        agent_all_perc_df = rl_base_utils.add_essential_agent_params_info(
            pfas.pooled_perc_info,
            params,
        )

        # Save outputs next to the agent folder for convenience
        output_dir = Path(agent_folder) / 'agent_plan_factors_outputs'
        output_dir.mkdir(parents=True, exist_ok=True)

        median_path = output_dir / 'all_ref_pooled_median_info.csv'
        perc_path = output_dir / 'pooled_perc_info.csv'

        agent_all_ref_pooled_median_info.to_csv(median_path, index=False)
        agent_all_perc_df.to_csv(perc_path, index=False)

        self.logger.info(f'Saved median info to {median_path}')
        self.logger.info(f'Saved percentile info to {perc_path}')

        # Mark as done
        self._write_done_marker(agent_folder, params)
        self.logger.info(f'Marked {agent_folder} as done.')

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def run(self):
        num_total = len(self.agent_folders)
        num_skipped = 0
        num_processed = 0

        for folder in self.agent_folders:
            if self.already_processed(folder):
                self.logger.info(f'Skipping already processed agent: {folder}')
                num_skipped += 1
                continue

            try:
                self.process_agent(folder)
                num_processed += 1
            except Exception as e:
                self.logger.exception(f'Error while processing {folder}: {e}')

        self.logger.info(
            f'Pipeline finished. Total={num_total}, processed={num_processed}, skipped={num_skipped}'
        )


# =====================================================================
# Argparse CLI
# =====================================================================

def build_argparser():
    parser = argparse.ArgumentParser(
        description='Run agent plan factors pipeline for a list of RL agents.'
    )

    parser.add_argument(
        '--agent-directory',
        type=str,
        default='multiff_analysis/RL_models/meta/directory_of_agents/dir1.json',
        help='Path to JSON file containing a list of agent model folders.',
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='multiff_analysis/RL_models/meta/pipeline_logs',
        help='Directory to store pipeline log files.',
    )

    parser.add_argument(
        '--num-datasets-to-collect',
        type=int,
        default=1,
        help='Number of datasets to collect per agent.',
    )

    parser.add_argument(
        '--num-steps-per-dataset',
        type=int,
        default=10000,
        help='Number of steps per dataset.',
    )

    parser.add_argument(
        '--intermediate-products-exist-ok',
        action='store_true',
        help='If set, do not recompute intermediate products if they already exist.',
    )

    parser.add_argument(
        '--agent-data-exists-ok',
        action='store_true',
        help='If set, do not re-collect agent data if it already exists.',
    )

    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='If set, ignore done markers and reprocess all agents.',
    )

    return parser


# =====================================================================
# Main entry point
# =====================================================================

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()

    pipeline = AgentProcessingPipeline(
        agent_list_json=Path(args.agent_directory),
        log_dir=Path(args.log_dir),
        num_datasets_to_collect=args.num_datasets_to_collect,
        num_steps_per_dataset=args.num_steps_per_dataset,
        intermediate_products_exist_ok=args.intermediate_products_exist_ok,
        agent_data_exists_ok=args.agent_data_exists_ok,
        force_reprocess=args.force_reprocess,
    )

    pipeline.run()
