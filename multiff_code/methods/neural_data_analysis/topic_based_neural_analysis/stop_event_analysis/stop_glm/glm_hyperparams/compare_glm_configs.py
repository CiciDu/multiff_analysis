from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib


def show_hyperparams_for_config(results, config_label):
    """
    Return (winners_per_cluster, winner_counts) for a given config.
    Works whether the per-config dict has 'metrics_df' or only 'cv_tables_df'.
    """
    # find that config's result dict
    res = None
    for r in results:
        if r.get('config') == config_label or r.get('summary', {}).get('config') == config_label:
            res = r
            break
    if res is None:
        raise ValueError(f"Config '{config_label}' not found in results")

    # prefer CV table (it explicitly marks winners)
    cv = res.get('cv_tables_df')
    if cv is not None and len(cv):
        winners = (cv.loc[cv['selected'], ['cluster', 'alpha', 'l1_wt', 'score']]
                     .sort_values(['cluster'])
                     .reset_index(drop=True))
    else:
        # fallback: try metrics_df (must contain per-cluster alpha/l1_wt)
        mkey = 'metrics_df' if 'metrics_df' in res else (
            'metrics' if 'metrics' in res else None)
        if mkey is None:
            raise KeyError(
                "Neither 'cv_tables_df' nor 'metrics_df' present in this config's result")
        mdf = res[mkey]
        need = [c for c in ['cluster', 'alpha', 'l1_wt'] if c not in mdf.columns]
        if need:
            raise KeyError(f"metrics table missing columns: {need}")
        winners = mdf[['cluster', 'alpha', 'l1_wt']].sort_values(
            'cluster').reset_index(drop=True)

    # distribution of chosen combos across clusters
    dist = (winners.groupby(['alpha', 'l1_wt'])
            .size().rename('n_clusters')
            .reset_index()
            .sort_values('n_clusters', ascending=False)
            .reset_index(drop=True))
    return winners, dist


def _slugify(s: str) -> str:
    """Filesystem-safe label."""
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(s)).strip('_')


def save_sweep_results(results, summary_df, cv_tables_df, out_dir):
    """
    Persist outputs from sweep_and_compare.
      results      : list[dict] (per-config bundles from run_model_config)
      summary_df   : leaderboard (one row per config)
      cv_tables_df : concatenated CV grids (all configs)
    Writes CSVs + PNGs into a directory tree:

    out_dir/
      summary.csv
      cv_tables.csv
      env.json
      configs/<config_slug>/
        config.json
        summary_row.csv
        coefs.csv
        metrics.csv
        population_tests.csv
        cv_table.csv
        figures/
          coef_dists.png
          model_quality.png
          forest.png
          rr_hist.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-level artifacts
    _save_df(summary_df, out_dir / 'summary.csv')
    _save_df(cv_tables_df, out_dir / 'cv_tables.csv')

    # Minimal environment manifest (handy for reproducibility)
    try:
        import sys, statsmodels, sklearn
        env = {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "statsmodels": statsmodels.__version__,
            "sklearn": sklearn.__version__,
        }
    except Exception:
        env = {}
    (out_dir / 'env.json').write_text(json.dumps(env, indent=2))

    # Per-config artifacts
    for r in results:
        label = r.get('config', r.get('label', 'config'))
        slug  = _slugify(label)
        cfg_dir = out_dir / 'configs' / slug
        fig_dir = cfg_dir / 'figures'
        cfg_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Save the config dict (if present)
        cfg = r.get('cfg', {})
        (cfg_dir / 'config.json').write_text(json.dumps(cfg, indent=2))

        # Save the single-row summary for this config
        summ = r.get('summary', {})
        if summ:
            pd.DataFrame([summ]).to_csv(cfg_dir / 'summary_row.csv', index=False)

        # Pull tables from the run output
        out = r.get('out', {})
        _save_df(out.get('coefs_df'),            cfg_dir / 'coefs.csv')
        _save_df(out.get('metrics_df'),          cfg_dir / 'metrics.csv')
        _save_df(out.get('population_tests_df'), cfg_dir / 'population_tests.csv')

        # Config-specific CV grid (if present)
        _save_df(r.get('cv_tables_df'),          cfg_dir / 'cv_table.csv')


        # a quick winners table (α, L1_wt per cluster) saved per config
        cv = r.get('cv_tables_df')
        if cv is not None and len(cv):
            winners = (cv.loc[cv.get('selected', pd.Series(False, index=cv.index))]
                        .sort_values(['cluster'])
                        [['cluster','alpha','l1_wt','score']])
            winners.to_csv(cfg_dir / 'winners.csv', index=False)


        # Save figures (if present)
        figs = out.get('figures', {}) or {}
        for name, fig in figs.items():
            if fig is None:
                continue
            try:
                fig.savefig(fig_dir / f'{name}.png', dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f'[warn] could not save figure {label}/{name}: {e}')
