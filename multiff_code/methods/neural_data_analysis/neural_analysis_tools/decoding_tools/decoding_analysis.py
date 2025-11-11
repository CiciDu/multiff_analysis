import json
import hashlib
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional, Dict, Any

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# classifiers
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from neural_data_analysis.neural_analysis_tools.decoding_tools import decoding_utils
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_psth import compare_events
from neural_data_analysis.neural_analysis_tools.decoding_tools import plot_decoding


def _json_default(o):
    """Make numpy/scientific types JSON-serializable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set,)):
        return list(o)
    # Fallback: string representation
    return str(o)


def run_population_decoding(an, window=(0, 0.3), model_name='svm',
                            k=3, tune=False, model_kwargs=None,
                            alpha=0.05, do_testing=True):
    """Compute per-neuron AUCs and (optionally) perform population-level test."""
    auc_per_neuron, unit_ids = decoding_utils.get_auc_per_neuron(
        an, window=window, model_name=model_name, k=k,
        seed=0, tune=tune, standardize=True, model_kwargs=model_kwargs
    )

    if do_testing:
        pop_res = decoding_utils.population_decoding_test(
            auc_per_neuron, method='wilcoxon', alpha=alpha
        )
        print(f'Population mean AUC={pop_res["mean_auc"]:.3f}, '
              f'p={pop_res["p"]:.4g}, sig={pop_res["sig"]}')
    else:
        pop_res = {'mean_auc': np.mean(auc_per_neuron),
                   'p': np.nan, 'sig': np.nan}

    return pop_res, auc_per_neuron, unit_ids


def run_window_decoding(an, window, model_name='svm', k=3, tune=False,
                        model_kwargs=None, alpha=0.05, n_perm=0,
                        do_testing=True,
                        perm_search: Optional[str] = None,
                        perm_param_grid: Optional[Dict[str, Any]] = None,
                        perm_n_iter: int = 10):
    """
    Decode neural data for a single time window with optional significance testing.

    Returns
    -------
    dict : mean_auc, sd_auc, tstat, p_ttest, sig_ttest, p_perm, n_units
    """
    res = decoding_utils.run_decoding(
        an, window=window, model_name=model_name,
        tune=tune, k=k, model_kwargs=model_kwargs
    )

    # Skip significance if not requested
    if not do_testing:
        # Compute sample size for this window
        X_tmp, y_tmp, _ = decoding_utils.build_Xy(an, window)
        sample_size = int(y_tmp.shape[0])
        return {
            'window_start': window[0], 'window_end': window[1],
            'mean_auc': res['mean_auc'], 'sd_auc': res['sd_auc'],
            'tstat': np.nan, 'p_ttest': np.nan, 'sig_ttest': np.nan,
            'p_perm': np.nan, 'n_units': res['n_units'],
            'sample_size': sample_size
        }

    # Build data
    X, y, _ = decoding_utils.build_Xy(an, window)
    cv = StratifiedKFold(k, shuffle=True, random_state=0)
    aucs = []
    for tr, te in cv.split(X, y):
        model = decoding_utils.get_decoder(
            model_name, model_kwargs=model_kwargs)[0]
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))

    aucs = np.asarray(aucs)
    tstat, p_ttest = stats.ttest_1samp(aucs, 0.5, alternative='greater')
    sig_ttest = p_ttest < alpha

    # Optional permutation test
    p_perm = np.nan
    if n_perm > 0:
        # Strategy: if perm_search provided, do NOT use fixed_params; otherwise reuse tuned params
        fixed_params = None if perm_search in (
            'grid', 'random') else res.get('best_params', None)
        observed_auc = res.get('mean_auc', None)
        # permutation_test_auc returns (real_auc, auc_null, pval)
        _, _, p_perm = decoding_utils.permutation_test_auc(
            X, y, model_name=model_name, k=k, n_perm=n_perm,
            show_progress=False, fixed_params=fixed_params, real_auc=observed_auc,
            param_grid=perm_param_grid, perm_search=perm_search, perm_n_iter=perm_n_iter
        )

    return {
        'window_start': window[0], 'window_end': window[1],
        'mean_auc': res['mean_auc'], 'sd_auc': res['sd_auc'],
        'tstat': tstat, 'p_ttest': p_ttest, 'sig_ttest': sig_ttest,
        'p_perm': p_perm, 'n_units': res['n_units'],
        'best_params': json.dumps(res.get('best_params', {}), default=_json_default),
        'sample_size': int(y.shape[0])
    }


def run_full_decoding_for_comparison(
    comp, datasets, pn, cfg,
    model_name='svm', model_kwargs=None,
    tune=False, k=3, n_perm=0, alpha=0.05,
    windows=None, align_by_stop_end=False,
    do_testing=True, n_jobs=1, verbose=True,
    perm_search: Optional[str] = None,
    perm_param_grid: Optional[Dict[str, Any]] = None,
    perm_n_iter: int = 10
):
    """
    Run decoding across all time windows for a single comparison.
    Uses joblib to parallelize across windows for faster execution.

    Parameters
    ----------
    comp : dict
        Comparison definition (keys: a_label, b_label, key).
    datasets : dict
        Dataset dictionary for building analyzer.
    pn, cfg : objects
        Monkey data and configuration objects.
    model_name : str
        Decoder model (svm, logreg, etc.).
    model_kwargs : dict
        Model parameters for decoder.
    tune : bool
        Whether to tune hyperparameters.
    k : int
        Number of CV folds.
    n_perm : int
        Number of permutations per window (0 = skip).
    alpha : float
        Significance threshold.
    windows : list[tuple]
        List of (start, end) windows for decoding.
    align_by_stop_end : bool
        If True, align by stop end; otherwise align by stop start.
    do_testing : bool
        Whether to perform t-tests or permutations.
    n_jobs : int
        Number of parallel processes (set to SLURM_CPUS_PER_TASK on cluster).
    verbose : bool
        Print progress messages.

    Returns
    -------
    df : pd.DataFrame
        Decoding results across windows for this comparison.
    """
    # -----------------------------------------------------------
    # Build analyzer once per comparison
    # -----------------------------------------------------------
    if verbose:
        print(f"[run_full_decoding_for_comparison] Building analyzer for "
              f"{comp['a_label']} vs {comp['b_label']} "
              f"({'stop end' if align_by_stop_end else 'stop start'})")

    an, _ = compare_events.build_analyzer(
        comp, datasets, pn.spikes_df, pn.monkey_information,
        cfg, verbose=False, align_by_stop_end=align_by_stop_end
    )
    an.run_full_analysis(cluster_idx=None)

    # Log model and parameters for this comparison
    if verbose:
        try:
            params_str = json.dumps(model_kwargs or {}, default=_json_default)
        except Exception:
            params_str = str(model_kwargs)
        print(
            f"[run_full_decoding_for_comparison] model={model_name}, params={params_str}"
        )

    # -----------------------------------------------------------
    # Parallel decoding across windows
    # -----------------------------------------------------------
    if verbose:
        print(f"[run_full_decoding_for_comparison] Running {len(windows)} windows "
              f"using {n_jobs} parallel workers...")

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_window_decoding)(
            an, window,
            model_name=model_name, k=k, tune=tune,
            model_kwargs=model_kwargs, alpha=alpha,
            n_perm=n_perm, do_testing=do_testing,
            perm_search=perm_search, perm_param_grid=perm_param_grid, perm_n_iter=perm_n_iter
        )
        for window in windows
    )

    # -----------------------------------------------------------
    # Combine results into a tidy DataFrame
    # -----------------------------------------------------------
    df = pd.DataFrame(results)
    # Ensure sample_size column exists; compute if missing
    if 'sample_size' not in df.columns:
        sample_sizes = []
        for window in windows:
            X_tmp, y_tmp, _ = decoding_utils.build_Xy(an, window)
            sample_sizes.append(int(y_tmp.shape[0]))
        df['sample_size'] = sample_sizes
    df['a_label'] = comp['a_label']
    df['b_label'] = comp['b_label']
    df['key'] = comp['key']
    df['align_by_stop_end'] = align_by_stop_end
    df['n_perm'] = n_perm
    df['model_name'] = model_name

    if verbose:
        print(f"[run_full_decoding_for_comparison] Done: "
              f"{comp['a_label']} vs {comp['b_label']} ({len(windows)} windows)")

    return df


def save_decoding_results(df, comp_key, model_name, save_dir, metadata=None, overwrite=False):
    """
    Save decoding results and optional metadata with a unique, readable filename.

    Parameters
    ----------
    df : pd.DataFrame
        The decoding results to save.
    comp_key : str
        Comparison key, e.g. 'guat_first_vs_taft_first'.
    model_name : str
        Model name, e.g. 'svm' or 'logreg'.
    save_dir : str or Path
        Base directory to save results (e.g. '.../retry_decoder/.../decoding').
    metadata : dict or None
        Optional metadata dictionary to save alongside as JSON.
    overwrite : bool
        If True, overwrite existing files with same name.

    Returns
    -------
    tuple : (csv_path, json_path)
        Paths to the saved CSV and JSON files.
    """
    # Ensure directories exist
    save_dir = Path(save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build filename base and ensure uniqueness if needed
    safe_key = comp_key.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"{safe_key}_{timestamp}"
    csv_path = save_dir / f"{base_name}.csv"
    json_path = save_dir / f"{base_name}.json"

    # If a file already exists and overwrite is False, append an incrementing suffix
    if csv_path.exists() and not overwrite:
        suffix_idx = 1
        while True:
            candidate_csv = save_dir / f"{base_name}_{suffix_idx:02d}.csv"
            candidate_json = save_dir / f"{base_name}_{suffix_idx:02d}.json"
            if not candidate_csv.exists() and not candidate_json.exists():
                csv_path = candidate_csv
                json_path = candidate_json
                break
            suffix_idx += 1

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"[save_decoding_results] Saved CSV → {csv_path}")

    # Save metadata (if provided)
    if metadata is not None:
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[save_decoding_results] Saved metadata → {json_path}")

    # Append to central decode index CSV for provenance and easy querying
    try:
        # Compute a short, stable hash of model_kwargs
        params_hash = hashlib.sha1(
            json.dumps((metadata or {}).get('model_kwargs', {}),
                       sort_keys=True, default=_json_default).encode('utf-8')
        ).hexdigest()[:10]

        # Locate project logs directory under multiff_analysis/logs/decode
        def _decode_logs_index_path() -> Path:
            p = Path(__file__).resolve()
            for parent in [p] + list(p.parents):
                if parent.name == "multiff_analysis":
                    logs_dir = parent / "logs" / "decode"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    return logs_dir / "index.csv"
            # Fallback
            logs_dir = Path.cwd() / "multiff_analysis" / "logs" / "decode"
            logs_dir.mkdir(parents=True, exist_ok=True)
            return logs_dir / "index.csv"

        index_path = _decode_logs_index_path()

        row = {
            'timestamp': (metadata or {}).get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            'monkey': (metadata or {}).get('monkey', None),
            'session': (metadata or {}).get('session', None),
            'model': model_name,
            'key': comp_key,
            'align_by_stop_end': (metadata or {}).get('align_by_stop_end', None),
            'status': 'done',
            'csv_path': str(csv_path),
            'json_path': str(json_path) if metadata is not None else '',
            'n_perm': (metadata or {}).get('n_perm', None),
            'tune': (metadata or {}).get('tune', None),
            'k_folds': (metadata or {}).get('k_folds', None),
            'params_hash': params_hash,
        }

        # Append (creating file with columns if needed)
        if index_path.exists():
            try:
                df_idx = pd.read_csv(index_path)
            except Exception:
                df_idx = pd.DataFrame()
        else:
            df_idx = pd.DataFrame()
        df_idx = pd.concat([df_idx, pd.DataFrame([row])], ignore_index=True)
        df_idx.to_csv(index_path, index=False)
        print(f"[index] Appended → {index_path}")
    except Exception as e:
        print(f"[index] Failed to append decode index: {e}")

    return csv_path, json_path


def _should_skip_existing_results(model_dir, comp_key, align_by_stop_end):
    """Check if decoding results already exist for this model/key/alignment."""
    safe_key = comp_key.replace(' ', '_')
    try:
        existing_csvs = list(model_dir.glob(f'{safe_key}_*.csv'))
    except Exception:
        existing_csvs = []

    if not existing_csvs:
        return False

    target_align_str = 'true' if align_by_stop_end else 'false'
    for csv_path in existing_csvs:
        # Try reading align info from CSV
        try:
            df_existing = pd.read_csv(csv_path, usecols=['align_by_stop_end'])
            if (df_existing['align_by_stop_end'].astype(str)
                    .str.strip().str.lower() == target_align_str).any():
                return True
        except Exception:
            pass

        # Fallback: read JSON sidecar
        json_sidecar = csv_path.with_suffix('.json')
        if json_sidecar.exists():
            try:
                with open(json_sidecar, 'r') as f:
                    meta = json.load(f)
                meta_align = str(
                    meta.get('align_by_stop_end', '')).strip().lower()
                if meta_align == target_align_str:
                    return True
            except Exception:
                pass

    return False


def run_all_decoding_comparisons(
    comparisons, keys, datasets, pn, cfg,
    model_name='svm', model_kwargs=None,
    tune=False, k=3, n_perm=0, alpha=0.05,
    windows=None, do_testing=True, plot=False,
    save_dir=None, save_format='csv',
    overwrite=False, exists_ok=False,
    session_info=None, verbose=True,
    n_jobs: int = 1,
    perm_search: Optional[str] = None,
    perm_param_grid: Optional[Dict[str, Any]] = None,
    perm_n_iter: int = 10
):
    """
    Run decoding for all comparisons (and alignments),
    using save_decoding_results() to write per-comparison CSV + JSON.

    Parameters
    ----------
    comparisons : list[dict]
        Comparison dictionaries (a_label, b_label, key, etc.).
    keys : list[str]
        Subset of comparison keys to run.
    datasets : dict
        Dataset dictionary.
    pn : object
        Monkey data object.
    cfg : PSTHConfig
        Configuration for decoding windows and binning.
    model_name : str
        Decoder name (svm, logreg, etc.).
    model_kwargs : dict
        Decoder model hyperparameters.
    tune : bool
        Whether to tune hyperparameters.
    k : int
        Number of CV folds.
    n_perm : int
        Number of permutations.
    alpha : float
        Significance level for tests.
    windows : list[tuple]
        Time windows for decoding.
    do_testing : bool
        Whether to run t-tests / permutation tests.
    plot : bool
        Whether to plot timecourse or heatmaps.
    save_dir : str or Path or None
        Base directory for saving results (disabled if None).
    save_format : str
        'csv' or 'parquet'.
    overwrite : bool
        Overwrite existing files if True.
    exists_ok : bool
        If True and outputs already exist for this model and comparison key,
        skip running that comparison.
    session_info : dict or None
        Optional dict with keys like {'monkey': str, 'session': str} to embed
        provenance in saved metadata and the central index.
    verbose : bool
        Print progress messages.

    Returns
    -------
    df_all : pd.DataFrame
        Combined DataFrame of all decoding results.
    """

    def _log(msg):
        if verbose:
            print(msg)

    all_results = []

    # ------------------------------------------------------------------
    # Prepare save directory
    # ------------------------------------------------------------------
    if save_dir is not None:
        save_dir = Path(save_dir)
        model_dir = save_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        _log(f'[run_all_decoding_comparisons] Saving to {model_dir}')
    else:
        model_dir = None
        _log('[run_all_decoding_comparisons] Saving disabled (save_dir=None)')

    # ------------------------------------------------------------------
    # Main decoding loop
    # ------------------------------------------------------------------
    for align_by_stop_end in [True, False]:
        for comp in comparisons:
            if comp['key'] not in keys:
                continue

            _log(f"\n>>> {comp['a_label']} vs {comp['b_label']} "
                 f"(Aligned by {'stop end' if align_by_stop_end else 'stop start'})")

            # Skip existing results if requested
            if exists_ok and model_dir is not None:
                if _should_skip_existing_results(model_dir, comp['key'], align_by_stop_end):
                    _log(f"[run_all_decoding_comparisons] Skip existing → "
                         f"model={model_name}, key={comp['key']}, align_by_stop_end={align_by_stop_end}")
                    continue

            # --- Run decoding for this comparison ---
            df = run_full_decoding_for_comparison(
                comp, datasets, pn, cfg,
                model_name=model_name, model_kwargs=model_kwargs,
                tune=tune, k=k, n_perm=n_perm, alpha=alpha,
                windows=windows, align_by_stop_end=align_by_stop_end,
                do_testing=do_testing, n_jobs=n_jobs,
                perm_search=perm_search, perm_param_grid=perm_param_grid, perm_n_iter=perm_n_iter
            )

            all_results.append(df)

            # --- Save results using helper ---
            if model_dir is not None:
                best_params_by_window = [
                    {
                        'window_start': float(row.get('window_start', np.nan)),
                        'window_end': float(row.get('window_end', np.nan)),
                        'best_params': (
                            json.loads(row['best_params'])
                            if isinstance(row.get('best_params'), str)
                            else row.get('best_params')
                        ),
                    }
                    for _, row in df.iterrows() if 'best_params' in df.columns
                ]

                metadata = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'comparison': comp['key'],
                    'model_name': model_name,
                    'model_kwargs': model_kwargs,
                    'k_folds': k,
                    'n_perm': n_perm,
                    'alpha': alpha,
                    'tune': tune,
                    'do_testing': do_testing,
                    'windows': windows,
                    'align_by_stop_end': align_by_stop_end,
                    'best_params_by_window': best_params_by_window,
                    'save_format': save_format,
                }

                if isinstance(session_info, dict):
                    metadata.update({
                        k_: v for k_, v in session_info.items()
                        if k_ in ('monkey', 'session')
                    })

                save_decoding_results(
                    df=df,
                    comp_key=comp['key'],
                    model_name=model_name,
                    save_dir=save_dir,
                    metadata=metadata,
                    overwrite=overwrite
                )

        # ------------------------------------------------------------------
        # Optional plotting after each alignment
        # ------------------------------------------------------------------
        if plot and all_results:
            try:
                df_align = pd.concat(
                    all_results, ignore_index=True, copy=False)
                title = 'Align by stop end' if align_by_stop_end else 'Align by stop start'
                plot_decoding.plot_decoding_auc_heatmap(
                    df_align, threshold=0.55, cmap='magma', title=title
                )
                plt.show()
            except Exception as e:
                _log(f'[plot] Skipping plots due to missing dependency: {e}')

    # ------------------------------------------------------------------
    # Combine and return all results
    # ------------------------------------------------------------------
    if not all_results:
        _log('\n[run_all_decoding_comparisons] No results to combine '
             '(all comparisons may have been skipped or produced no output). Returning empty DataFrame.')
        return pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True, copy=False)
    _log('\n[run_all_decoding_comparisons] All comparisons complete.')
    return df_all
