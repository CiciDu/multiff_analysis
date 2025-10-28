import os
import time as time_package
import json
import shutil
import pandas as pd


def _slurm_context():
    return {
        'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'slurm_array_job_id': os.getenv('SLURM_ARRAY_JOB_ID'),
        'slurm_array_task_id': os.getenv('SLURM_ARRAY_TASK_ID'),
        'slurm_job_name': os.getenv('SLURM_JOB_NAME'),
        'slurm_submit_dir': os.getenv('SLURM_SUBMIT_DIR'),
    }


def _ensure_csv(path: str, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def _logs_root_from_overall(overall_folder: str) -> str:
    # Walk up until we find multiff_analysis, then use its logs/ subdir
    p = os.path.abspath(os.path.expanduser(overall_folder))
    cur = p
    while True:
        if os.path.basename(cur) == 'multiff_analysis':
            return os.path.join(cur, 'logs')
        parent = os.path.dirname(cur)
        if parent == cur:
            # Fallback: sibling logs next to overall_folder
            return os.path.join(os.path.dirname(p), 'logs')
        cur = parent


def _runs_log_path(overall_folder: str) -> str:
    logs_root = _logs_root_from_overall(overall_folder)
    suffix = _run_suffix()
    return os.path.join(logs_root, f'runs_log_{suffix}.csv')


def _curriculum_log_path(overall_folder: str) -> str:
    logs_root = _logs_root_from_overall(overall_folder)
    suffix = _run_suffix()
    return os.path.join(logs_root, f'curriculum_stages_{suffix}.csv')


def _training_log_path(overall_folder: str) -> str:
    logs_root = _logs_root_from_overall(overall_folder)
    suffix = _run_suffix()
    return os.path.join(logs_root, f'training_metrics_{suffix}.csv')


def _run_suffix() -> str:
    job_id = os.getenv('SLURM_ARRAY_JOB_ID') or os.getenv('SLURM_JOB_ID')
    task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    if job_id and task_id:
        return f'job{job_id}_task{task_id}'
    if job_id:
        return f'job{job_id}'
    # Fallback: timestamp + pid to avoid collisions
    try:
        ts = time_package.strftime('%Y%m%d-%H%M%S', time_package.localtime())
    except Exception:
        ts = 'na'
    pid = os.getpid()
    return f'local_{ts}_{pid}'


def _flatten_dict(prefix, d):
    out = {}
    for k, v in (d or {}).items():
        out[f'{prefix}{k}'] = v
    return out


def log_run_start(overall_folder: str, agent_type: str, sweep_params: dict, extra_info: dict = None):
    now = time_package.strftime('%Y-%m-%d %H:%M:%S', time_package.localtime())
    record = {
        'timestamp': now,
        'event': 'run_start',
        'agent_type': agent_type,
        'overall_folder': os.path.expanduser(overall_folder),
        **_flatten_dict('param_', sweep_params),
        **_flatten_dict('slurm_', _slurm_context()),
    }
    if isinstance(extra_info, dict):
        record.update(_flatten_dict('extra_', extra_info))

    path = _runs_log_path(overall_folder)
    _ensure_csv(path, columns=list(record.keys()))
    # Union columns if schema evolves
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(path, index=False)


def log_run_end(overall_folder: str, agent_type: str, sweep_params: dict, status: str = 'finished', metrics: dict = None):
    now = time_package.strftime('%Y-%m-%d %H:%M:%S', time_package.localtime())
    record = {
        'timestamp': now,
        'event': 'run_end',
        'status': status,
        'agent_type': agent_type,
        'overall_folder': os.path.expanduser(overall_folder),
        **_flatten_dict('param_', sweep_params),
        **_flatten_dict('metric_', metrics or {}),
        **_flatten_dict('slurm_', _slurm_context()),
    }
    path = _runs_log_path(overall_folder)
    _ensure_csv(path, columns=list(record.keys()))
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(path, index=False)
    print('run_end logged to:', path)


def log_curriculum_stage(overall_folder: str, agent_type: str, sweep_params: dict, stage_payload: dict):
    now = time_package.strftime('%Y-%m-%d %H:%M:%S', time_package.localtime())
    record = {
        'timestamp': now,
        'event': 'curriculum_stage',
        'agent_type': agent_type,
        'overall_folder': os.path.expanduser(overall_folder),
        **_flatten_dict('param_', sweep_params),
        **(stage_payload or {}),
        **_flatten_dict('slurm_', _slurm_context()),
    }
    path = _curriculum_log_path(overall_folder)
    _ensure_csv(path, columns=list(record.keys()))
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(path, index=False)
    print('curriculum_stage logged to:', path)


def log_training_metrics(overall_folder: str, agent_type: str, sweep_params: dict, step: int, metrics: dict):
    """
    Append a single training metrics row to logs/training_metrics_<suffix>.csv for this run.
    Includes flattened sweep params and slurm context, and a monotonically increasing step.
    """
    if overall_folder is None:
        return
    now = time_package.strftime('%Y-%m-%d %H:%M:%S', time_package.localtime())
    record = {
        'timestamp': now,
        'event': 'training_metrics',
        'agent_type': agent_type,
        'overall_folder': os.path.expanduser(overall_folder),
        'step': int(step),
        **_flatten_dict('param_', sweep_params or {}),
        **_flatten_dict('metric_', metrics or {}),
        **_flatten_dict('slurm_', _slurm_context()),
    }
    path = _training_log_path(overall_folder)
    _ensure_csv(path, columns=list(record.keys()))
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(path, index=False)


def collect_model_to_job_dir(overall_folder: str, source_dir: str, preferred_name: str = None, force_copy: bool = False):
    """
    Create a symlink (or copy fallback) of the model folder into a per-job collection directory.
    Layout: <stored_models_root>/jobs/<SLURM_ARRAY_JOB_ID or SLURM_JOB_ID>/<name>
    """
    job_id = os.getenv('SLURM_ARRAY_JOB_ID') or os.getenv('SLURM_JOB_ID')
    if not job_id:
        return
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    overall_folder = os.path.abspath(os.path.expanduser(overall_folder))
    stored_models_root = os.path.abspath(os.path.join(overall_folder, os.pardir, os.pardir))
    job_root = os.path.join(stored_models_root, 'jobs', str(job_id))
    os.makedirs(job_root, exist_ok=True)

    if preferred_name is None:
        preferred_name = f"{os.path.basename(overall_folder)}__{os.path.basename(source_dir)}"
    dest_path = os.path.join(job_root, preferred_name)

    # If exists, remove to refresh
    if os.path.islink(dest_path) or os.path.isfile(dest_path):
        try:
            os.unlink(dest_path)
        except OSError:
            pass
    elif os.path.isdir(dest_path):
        try:
            shutil.rmtree(dest_path)
        except OSError:
            pass

    # Try symlink first for space efficiency; fallback to copytree
    if not force_copy:
        try:
            os.symlink(source_dir, dest_path)
            return
        except OSError:
            pass
    try:
        shutil.copytree(source_dir, dest_path)
    except Exception as e:
        # Last resort: do nothing but print a warning
        print('[logger] failed to collect model to job dir:', e)


