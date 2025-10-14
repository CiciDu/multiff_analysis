from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from collections import Counter
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.stop_glm.glm_fit import stop_glm_fit, cv_stop_glm


# ----------------------------- types -----------------------------------------

@dataclass
class ConfigSpec:
    name: str
    regularization: str = 'none'          # 'none' | 'elasticnet'
    alpha_grid: tuple[float, ...] = (0.0,)
    l1_wt_grid: tuple[float, ...] = (0.0,)
    cv_metric: str = 'loglik'
    n_splits: int = 5
    cov_type: str = 'HC1'
    refit_on_support: bool = False

    # You can include any extra, config-specific kwargs in a dict:
    extra: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['alpha_grid'] = list(self.alpha_grid)
        d['l1_wt_grid'] = list(self.l1_wt_grid)
        return d


def _config_to_label(cfg: ConfigSpec | dict) -> str:
    if isinstance(cfg, dict):
        name = cfg.get('name', 'unnamed')
        reg = cfg.get('regularization', 'none')
        cvm = cfg.get('cv_metric', 'loglik')
        ns = cfg.get('n_splits', 5)
    else:
        name, reg, cvm, ns = cfg.name, cfg.regularization, cfg.cv_metric, cfg.n_splits
    return f'{name} | {reg} | {cvm} | {ns}fold'


# ------------------------- main runner class ---------------------------------

class SweepAndCompare:
    """
    Stateful, resumable runner for sweeping GLM configs and comparing results.

    You provide:
      - configs: list[ConfigSpec | dict]
      - data objects (df_X, df_Y, offset_log, ...)
      - a 'fit_fn' callable that performs the actual per-config fit and returns:
            result_dict, summary_row, cv_table_df
        where:
          * result_dict: arbitrary python dict with detailed outputs for this config
          * summary_row: dict-like summarizing the config (will become one row)
          * cv_table_df: pd.DataFrame with CV diagnostics for this config
    The class:
      - stores intermediate results on self.* attributes
      - checkpoints to disk after each config
      - can resume from checkpoint
    """

    # ---------------------- construction & (re)loading -----------------------

    def __init__(
        self,
        *,
        configs: Iterable[ConfigSpec | dict],
        df_X: pd.DataFrame,
        df_Y: pd.DataFrame,
        offset_log: np.ndarray | pd.Series,
        fit_fn: Callable[..., tuple[dict, dict, pd.DataFrame]],
        feature_names: Optional[list[str]] = None,
        cluster_ids: Optional[list[str]] = None,
        groups: Optional[np.ndarray] = None,
        cov_type: str = 'HC1',
        show_plots: bool = False,
        auto_pick_forest_term: bool = False,
        alpha_for_forest: float = 0.05,
        out_dir: str | Path = 'glm_sweep_compare_out',
        run_id: Optional[str] = None,
        autosave: bool = True,
        autosave_every: int = 1,       # checkpoint every N configs
        extra_fit_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.configs: list[ConfigSpec] = [
            c if isinstance(c, ConfigSpec) else ConfigSpec(**c) for c in configs
        ]
        self.df_X = df_X
        self.df_Y = df_Y
        self.offset_log = offset_log
        self.feature_names = feature_names
        self.cluster_ids = cluster_ids
        self.groups = groups
        self.cov_type = cov_type
        self.show_plots = show_plots
        self.auto_pick_forest_term = auto_pick_forest_term
        self.alpha_for_forest = alpha_for_forest
        self.fit_fn = fit_fn
        self.extra_fit_kwargs = extra_fit_kwargs or {}

        # stateful attributes you can inspect at any time
        self.per_config_results: list[dict] = []      # detailed dicts per config
        self.summary_rows: list[dict] = []            # one dict per config
        self.cv_tables: list[pd.DataFrame] = []       # list of per-config CV tables
        self.completed_indices: set[int] = set()      # which config indices are done

        # bookkeeping
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
        self.autosave = autosave
        self.autosave_every = max(1, int(autosave_every))
        self._since_last_save = 0

        # derived tables (built on demand)
        self.summary_df: Optional[pd.DataFrame] = None
        self.cv_tables_df: Optional[pd.DataFrame] = None

    # ----------------------------- running -----------------------------------

    def run(self, start_at: int = 0, end_at: Optional[int] = None) -> None:
        """
        Execute sweep over configs[start_at:end_at].
        Checkpoints after each config (or according to autosave_every).
        """
        n = len(self.configs)
        end = n if end_at is None else min(end_at, n)

        try:
            for i in range(start_at, end):
                if i in self.completed_indices:
                    # already done (e.g., resuming) — skip
                    continue

                cfg = self.configs[i]
                label = _config_to_label(cfg)

                # ---- run one config
                result_dict, summary_row, cv_table_df = self._run_one_config(i, cfg)

                # ---- record state
                self.per_config_results.append(
                    {'index': i, 'label': label, 'config': cfg.to_dict(), **result_dict}
                )
                # ensure the summary row carries an index and label
                self.summary_rows.append({'index': i, 'label': label, **summary_row})
                self.cv_tables.append(self._attach_cfg_label(cv_table_df, i, label))
                self.completed_indices.add(i)

                # invalidate cached concatenations
                self.summary_df = None
                self.cv_tables_df = None

                # ---- autosave
                self._since_last_save += 1
                if self.autosave and (self._since_last_save % self.autosave_every == 0):
                    self.save_checkpoint()
                    self._since_last_save = 0

            # final save at the end of the span
            if self.autosave:
                self.save_checkpoint()

        except KeyboardInterrupt:
            # save what we have and re-raise so you see the interrupt
            self.save_checkpoint()
            raise

    # --------------------------- one-config core ------------------------------

    def _run_one_config(self, idx: int, cfg: ConfigSpec) -> tuple[dict, dict, pd.DataFrame]:
        """
        Calls the provided fit_fn. You can tailor what you pass here.
        Your fit_fn must *not* mutate class state; it should return pure outputs.
        """
        # prepare kwargs sent to your model-fitting function
        kwargs = dict(
            df_X=self.df_X,
            df_Y=self.df_Y,
            offset_log=self.offset_log,
            cluster_ids=self.cluster_ids,
            groups=self.groups,
            cov_type=cfg.cov_type if hasattr(cfg, 'cov_type') else self.cov_type,
            regularization=cfg.regularization,
            alpha_grid=tuple(cfg.alpha_grid),
            l1_wt_grid=tuple(cfg.l1_wt_grid),
            n_splits=cfg.n_splits,
            cv_metric=cfg.cv_metric,
            refit_on_support=cfg.refit_on_support,
            feature_names=self.feature_names,
            show_plots=self.show_plots,
            auto_pick_forest_term=self.auto_pick_forest_term,
            alpha_for_forest=self.alpha_for_forest,
        )
        if cfg.extra:
            kwargs.update(cfg.extra)
        kwargs.update(self.extra_fit_kwargs)

        # delegate to your project-specific fitting code
        result_dict, summary_row, cv_table_df = self.fit_fn(**kwargs)

        # attach config name to outputs (useful in downstream analyses)
        result_dict = {**result_dict, 'config_name': cfg.name}
        summary_row = {**summary_row, 'config_name': cfg.name}

        return result_dict, summary_row, cv_table_df

    # ------------------------- convenience getters ---------------------------

    def get_summary(self) -> pd.DataFrame:
        if self.summary_df is None:
            self.summary_df = pd.DataFrame(self.summary_rows).sort_values('index').reset_index(drop=True)
        return self.summary_df

    def get_cv_tables(self) -> pd.DataFrame:
        if self.cv_tables_df is None:
            if len(self.cv_tables) == 0:
                self.cv_tables_df = pd.DataFrame()
            else:
                self.cv_tables_df = pd.concat(self.cv_tables, axis=0, ignore_index=True)
        return self.cv_tables_df

    # --------------------------- saving / loading -----------------------------

    @property
    def _ckpt_dir(self) -> Path:
        d = self.out_dir / f'run_{self.run_id}'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def checkpoint_paths(self) -> dict[str, Path]:
        return {
            'state_json': self._ckpt_dir / 'state.json',
            'summary_parquet': self._ckpt_dir / 'summary.parquet',
            'cv_parquet': self._ckpt_dir / 'cv_tables.parquet',
            'per_config_ndjson': self._ckpt_dir / 'per_config_results.ndjson',
        }

    def save_checkpoint(self) -> None:
        paths = self.checkpoint_paths

        # 1) lightweight JSON with run meta + which indices are done
        state = {
            'run_id': self.run_id,
            'completed_indices': sorted(self.completed_indices),
            'n_configs': len(self.configs),
            'configs': [c.to_dict() for c in self.configs],
            'autosave': self.autosave,
            'autosave_every': self.autosave_every,
            'cov_type': self.cov_type,
            'auto_pick_forest_term': self.auto_pick_forest_term,
            'alpha_for_forest': self.alpha_for_forest,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(paths['state_json'], 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # 2) summaries & cv tables as parquet (fast & robust)
        if len(self.summary_rows):
            self.get_summary().to_parquet(paths['summary_parquet'], index=False)
        if len(self.cv_tables):
            self.get_cv_tables().to_parquet(paths['cv_parquet'], index=False)

        # 3) detailed per-config dicts as NDJSON (each line is one config)
        if len(self.per_config_results):
            with open(paths['per_config_ndjson'], 'w', encoding='utf-8') as f:
                for row in self.per_config_results:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

    @classmethod
    def load_from_checkpoint(
        cls,
        *,
        checkpoint_dir: str | Path,
        df_X: pd.DataFrame,
        df_Y: pd.DataFrame,
        offset_log: np.ndarray | pd.Series,
        fit_fn: Callable[..., tuple[dict, dict, pd.DataFrame]],
        feature_names: Optional[list[str]] = None,
        cluster_ids: Optional[list[str]] = None,
        groups: Optional[np.ndarray] = None,
        show_plots: bool = False,
        extra_fit_kwargs: Optional[dict[str, Any]] = None,
    ) -> 'SweepAndCompare':
        """
        Re-hydrate a runner from an existing checkpoint directory.
        You must still pass the data and the same fit_fn to continue the run.
        """
        ckpt_dir = Path(checkpoint_dir)
        state_path = ckpt_dir / 'state.json'
        if not state_path.exists():
            raise FileNotFoundError(f'No state.json at {state_path}')

        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # reconstruct configs
        configs = [ConfigSpec(**c) for c in state['configs']]

        runner = cls(
            configs=configs,
            df_X=df_X,
            df_Y=df_Y,
            offset_log=offset_log,
            fit_fn=fit_fn,
            feature_names=feature_names,
            cluster_ids=cluster_ids,
            groups=groups,
            cov_type=state.get('cov_type', 'HC1'),
            show_plots=show_plots,
            auto_pick_forest_term=state.get('auto_pick_forest_term', False),
            alpha_for_forest=state.get('alpha_for_forest', 0.05),
            out_dir=ckpt_dir.parent,
            run_id=Path(ckpt_dir).name.replace('run_', ''),
            autosave=state.get('autosave', True),
            autosave_every=state.get('autosave_every', 1),
            extra_fit_kwargs=extra_fit_kwargs,
        )

        # restore completed indices
        runner.completed_indices = set(state.get('completed_indices', []))

        # load previously saved tables if present
        summary_path = ckpt_dir / 'summary.parquet'
        if summary_path.exists():
            runner.summary_df = pd.read_parquet(summary_path)
            runner.summary_rows = runner.summary_df.to_dict(orient='records')

        cv_path = ckpt_dir / 'cv_tables.parquet'
        if cv_path.exists():
            runner.cv_tables_df = pd.read_parquet(cv_path)
            # also keep a split list to maintain the same shape as during run()
            runner.cv_tables = [runner.cv_tables_df]

        # load detailed per-config dicts if present
        ndjson_path = ckpt_dir / 'per_config_results.ndjson'
        if ndjson_path.exists():
            with open(ndjson_path, 'r', encoding='utf-8') as f:
                runner.per_config_results = [json.loads(line) for line in f]

        return runner

    # ------------------------------- utils -----------------------------------

    @staticmethod
    def _attach_cfg_label(df: pd.DataFrame, index: int, label: str) -> pd.DataFrame:
        df = df.copy()
        df.insert(0, 'config_index', index)
        df.insert(1, 'config_label', label)
        return df


# assumes: _config_to_label, stop_glm_fit.glm_mini_report, cv_score_per_cluster are imported

def _compute_sparsity_and_sig(coefs: pd.DataFrame) -> tuple[float, float]:
    terms = pd.Series(coefs['term']).dropna().unique()
    intercept_names = [t for t in terms
                       if isinstance(t, str) and t.lower() in ('const', 'intercept', '_cons')]
    is_intercept = coefs['term'].isin(intercept_names) if len(intercept_names) else pd.Series(False, index=coefs.index)
    nz_mask = np.isfinite(coefs['coef']) & (coefs['coef'] != 0.0)
    sparsity = 1.0 - float(nz_mask[~is_intercept].mean()) if (~is_intercept).any() else np.nan
    sig_rate = float(coefs['sig_FDR'].mean()) if 'sig_FDR' in coefs.columns else 0.0
    return float(sparsity), float(sig_rate)


def fit_fn(
    *,
    df_X: pd.DataFrame,
    df_Y: pd.DataFrame,
    offset_log,
    cluster_ids: Optional[list[str]] = None,
    cov_type: str = 'HC1',
    regularization: str = 'none',
    alpha_grid: Tuple[float, ...] = (0.1, 0.3, 1.0),
    l1_wt_grid: Tuple[float, ...] = (1.0, 0.5, 0.0),
    n_splits: int = 5,
    cv_metric: str = 'loglik',
    groups=None,
    refit_on_support: bool = True,
    feature_names: Optional[list[str]] = None,
    show_plots: bool = False,
    auto_pick_forest_term: bool = False,   # kept for interface symmetry
    alpha_for_forest: float = 0.05,        # kept for interface symmetry
    cv_splitter: Optional[str] = None,
    use_overdispersion_scale: bool = False,
    add_outer_cv_summary: bool = False,    # << restored
) -> tuple[dict, dict, pd.DataFrame]:

    cfg = dict(
        cov_type=cov_type,
        regularization=regularization,
        alpha_grid=tuple(alpha_grid),
        l1_wt_grid=tuple(l1_wt_grid),
        n_splits=int(n_splits),
        cv_metric=cv_metric,
        refit_on_support=bool(refit_on_support),
        cv_splitter=cv_splitter,
        use_overdispersion_scale=bool(use_overdispersion_scale),
    )
    label = _config_to_label(cfg)

    out = stop_glm_fit.glm_mini_report(
        df_X=df_X, df_Y=df_Y, offset_log=offset_log,
        feature_names=feature_names, cluster_ids=cluster_ids,
        cov_type=cov_type,
        regularization=regularization,
        alpha_grid=alpha_grid,
        l1_wt_grid=l1_wt_grid,
        n_splits=n_splits,
        cv_metric=cv_metric,
        groups=groups,
        refit_on_support=refit_on_support,
        cv_splitter=cv_splitter,
        use_overdispersion_scale=use_overdispersion_scale,
        show_plots=show_plots,
        save_dir=None,
    )

    coefs = out['coefs_df'].copy()
    mets  = out['metrics_df'].copy()
    cvt   = out['cv_tables_df'].copy()

    # compute sparsity + sig rate exactly like before
    sparsity, sig_rate = _compute_sparsity_and_sig(coefs)

    # optional OUTER CV (independent generalization score)
    outer_cv_df = None
    if add_outer_cv_summary:
        outer_cv_df = cv_stop_glm.cv_score_per_cluster(
            df_X=df_X,
            df_Y=df_Y,
            offset_log=offset_log,
            groups=groups,
            n_splits=n_splits,
        )
        # attach to per-cluster metrics for convenience
        mets = mets.merge(
            outer_cv_df[['cluster', 'll_test', 'll_test_null', 'mcfadden_R2_cv']],
            on='cluster', how='left'
        )

    # rebuild your original agg
    mean_devexp   = float(mets['deviance_explained'].mean(skipna=True)) if 'deviance_explained' in mets else np.nan
    median_devexp = float(mets['deviance_explained'].median(skipna=True)) if 'deviance_explained' in mets else np.nan
    mean_r2       = float(mets['mcfadden_R2'].mean(skipna=True)) if 'mcfadden_R2' in mets else np.nan
    median_r2     = float(mets['mcfadden_R2'].median(skipna=True)) if 'mcfadden_R2' in mets else np.nan
    mean_alpha    = float(mets['alpha'].mean(skipna=True)) if 'alpha' in mets else np.nan
    median_alpha  = float(mets['alpha'].median(skipna=True)) if 'alpha' in mets else np.nan
    mode_l1_wt    = float(mets['l1_wt'].mode(dropna=True).iloc[0]) if ('l1_wt' in mets and mets['l1_wt'].dropna().size) else np.nan

    summary_row = {
        'config': label,
        'n_clusters': int(mets.shape[0]),
        'mean_deviance_explained': mean_devexp,
        'median_deviance_explained': median_devexp,
        'mean_mcfadden_R2': mean_r2,
        'median_mcfadden_R2': median_r2,
        'mean_alpha': mean_alpha,
        'median_alpha': median_alpha,
        'mode_l1_wt': mode_l1_wt,
        'sparsity': sparsity,
        'sig_rate': sig_rate,
        # outer-CV rollup (NaN if not computed)
        'mean_mcfadden_R2_cv': float(mets['mcfadden_R2_cv'].mean(skipna=True)) if ('mcfadden_R2_cv' in mets) else np.nan,
    }

    result_dict = {
        'label': label,
        'cfg': cfg,
        'coefs_df': coefs.to_dict(orient='records'),
        'metrics_df': mets.to_dict(orient='records'),
        'population_tests_df': out.get('population_tests_df', pd.DataFrame()).to_dict(orient='records')
            if isinstance(out.get('population_tests_df', None), pd.DataFrame) else None,
        'outer_cv_df': None if outer_cv_df is None else outer_cv_df.to_dict(orient='records'),
    }

    cv_table_df = cvt
    return result_dict, summary_row, cv_table_df
