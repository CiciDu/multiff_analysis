from pathlib import Path
import os, json, time, tempfile
import numpy as np

RESULTS_FORMAT_VERSION = "1.0"


def save_full_results_npz(
    base_dir: Path,
    cluster_name: str,
    results_struct: np.ndarray,
    reduced_var_list: list[str],
    extra_meta: dict | None = None,
) -> Path:
    """
    Save PGAM results for a single neuron into a compressed .npz file.

    Parameters
    ----------
    base_dir : Path
        Path to your main results folder (e.g. Path('/user_data/.../processed_neural_data_folder_path')).
        The function will create a subfolder called 'pgam_res' if it does not exist.

    cluster_name : str
        Name or ID of the neuron (e.g. 'neuron_12' or whatever column name you used).

    results_struct : np.ndarray
        The structured NumPy array returned by `postprocess_results(...)`.
        This array holds all the fields: neuron_id, variable, pseudo R2s, pvals, kernels, etc.

    reduced_var_list : list[str]
        The list of variables kept in the reduced model (usually `self.reduced.var_list`).

    extra_meta : dict, optional
        Any additional metadata you want saved (e.g. bin width, number of trials, AIC scores).
        Keys must be strings, values should be JSON-serializable.

    Returns
    -------
    Path
        The path to the saved `.npz` file.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    fn = base_dir / "pgam_res" / f"neuron_{cluster_name}.npz"
    fn.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(extra_meta or {})
    meta.update({
        "format_version": RESULTS_FORMAT_VERSION,
        "cluster_name": str(cluster_name),
        "saved_at_unix": time.time(),
        "columns": list(results_struct.dtype.names),
        "has_object_columns": any(dt == object for dt in results_struct.dtype.fields.values())
    })

    _atomic_save_npz(
        fn,
        results_struct=results_struct,
        reduced_var_list=np.array(list(reduced_var_list), dtype="U100"),
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    )
    return fn



def _atomic_save_npz(path: Path, **arrays):
    path = Path(path)
    tmp = Path(tempfile.mkstemp(dir=path.parent, suffix=".npz")[1])
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        try: tmp.unlink()
        except: pass


def load_full_results_npz(base_dir: Path, cluster_name: str):
    """
    Loads the structured array. Must use allow_pickle=True because of object fields.
    """
    fn = Path(base_dir) / "pgam_res" / f"neuron_{cluster_name}.npz"
    if not fn.exists() or fn.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty: {fn}")
    with np.load(fn, allow_pickle=True) as data:
        meta = json.loads(bytes(data["meta_json"].tolist()).decode("utf-8"))
        results_struct = data["results_struct"]
        reduced_var_list = list(data["reduced_var_list"])
    return results_struct, reduced_var_list, meta
       
        
        