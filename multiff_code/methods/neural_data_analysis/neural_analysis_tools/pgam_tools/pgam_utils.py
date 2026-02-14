import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

RESULTS_FORMAT_VERSION = "1.0"


def _atomic_save_npz(path: Path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def _has_object_fields(struct: np.ndarray) -> bool:
    """Correctly detect object dtypes in a structured array."""
    if struct.dtype.names is None:
        return False
    for name in struct.dtype.names:
        # dtype.fields[name] -> (dtype, offset[, title])
        field_dtype = struct.dtype.fields[name][0]
        if field_dtype == object:
            return True
    return False


def save_full_results_npz(
    base_dir: Path,
    cluster_name: str,
    results_struct: np.ndarray,
    reduced_vars: list[str],
    extra_meta: Optional[Dict[str, Any]] = None,
    full_time_bin: Optional[np.ndarray] = None,
    full_var_list: Optional[list[str]] = None,
    full_beta: Optional[np.ndarray] = None,
) -> Path:
    """
    Save PGAM results for a single neuron into a compressed .npz file.

    Parameters
    ----------
    base_dir : Path
        Root directory that contains (or will contain) 'pgam_res'.
    cluster_name : str
        Neuron/cluster name used in the filename.
    results_struct : np.ndarray
        Structured array returned by postprocess_results(...).
    reduced_vars : list[str]
        Features kept in the reduced model.
    extra_meta : Optional[dict]
        Optional JSON-serializable metadata.
    full_time_bin : Optional[np.ndarray]
        Time bins from the full model.
    full_var_list : Optional[list[str]]
        Variable list from the full model.
    full_beta : Optional[np.ndarray]
        Beta coefficients from the full model.

    Returns
    -------
    Path
        Path to the saved .npz file.
    """
    base_dir = Path(base_dir)
    out_path = base_dir / "pgam_res" / f"neuron_{cluster_name}.npz"

    cols = list(results_struct.dtype.names or [])
    meta = dict(extra_meta or {})
    meta.update(
        {
            "format_version": RESULTS_FORMAT_VERSION,
            "cluster_name": str(cluster_name),
            "saved_at_unix": time.time(),
            "columns": cols,
            "has_object_columns": _has_object_fields(results_struct),
        }
    )

    # Prepare arrays to save
    arrays_to_save = {
        "results_struct": results_struct,
        "reduced_vars": np.array(list(reduced_vars), dtype="U100"),
        "meta_json": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    
    # Add full model data if provided
    if full_time_bin is not None:
        arrays_to_save["full_time_bin"] = np.asarray(full_time_bin)
    if full_var_list is not None:
        arrays_to_save["full_var_list"] = np.array(list(full_var_list), dtype="U100")
    if full_beta is not None:
        arrays_to_save["full_beta"] = np.asarray(full_beta)

    _atomic_save_npz(out_path, **arrays_to_save)
    return out_path


def load_full_results_npz(base_dir: Path, cluster_name: str):
    """
    Load structured PGAM results for one neuron. Backward-compatible with old files
    that stored the array under key 'results' instead of 'results_struct'.

    Returns
    -------
    (results_struct, reduced_vars, meta_dict, full_model_data)
    
    where full_model_data is a dict with keys:
        - 'time_bin': np.ndarray or None
        - 'var_list': list[str] or None
        - 'beta': np.ndarray or None
    """
    fn = Path(base_dir) / "pgam_res" / f"neuron_{cluster_name}.npz"
    if not fn.exists() or fn.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty results file: {fn}")

    with np.load(fn, allow_pickle=True) as data:
        keys = set(data.files)

        # Try new schema first
        if "results_struct" in keys:
            results_struct = data["results_struct"].copy()
            reduced_vars = list(
                data["reduced_vars"]) if "reduced_vars" in keys else []
            meta = json.loads(bytes(data["meta_json"].tolist()).decode(
                "utf-8")) if "meta_json" in keys else {}
            
            # Load full model data if present (copy to avoid memory-mapped issues)
            full_model_data = {
                "time_bin": data["full_time_bin"].copy() if "full_time_bin" in keys else None,
                "var_list": list(data["full_var_list"]) if "full_var_list" in keys else None,
                "beta": data["full_beta"].copy() if "full_beta" in keys else None,
            }
        # Fallback to old/simple schema
        elif "results" in keys:
            results_struct = data["results"]
            if results_struct.dtype == object and results_struct.shape == ():
                # unwrap 0-d object array that holds the struct
                results_struct = results_struct.item()
            else:
                results_struct = results_struct.copy()
            reduced_vars = []
            meta = {}
            full_model_data = {"time_bin": None, "var_list": None, "beta": None}
        else:
            raise KeyError(
                f"No structured results found in {fn}. Keys: {sorted(keys)}")

    # Sanity: ensure it's a structured array and has the 'variable' field your code uses
    if results_struct.dtype.names is None:
        raise TypeError("Loaded results are not a structured array.")
    if "variable" not in results_struct.dtype.names:
        raise KeyError(
            "Field 'variable' missing inside the structured results array.")

    return results_struct, reduced_vars, meta, full_model_data
