"""Shared utility functions for encoding pipelines."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def anova_spike_counts_by_categorical(
    y: np.ndarray,
    cat_series: pd.Series,
    alpha: float = 0.05,
) -> Dict:
    """
    One-way ANOVA of spike counts grouped by a single categorical variable.

    Parameters
    ----------
    y : np.ndarray, shape (n_bins,)
        Spike counts (or rates) for one neuron, aligned with ``cat_series``.
    cat_series : pd.Series
        Category label for each bin.  NaN values are excluded when grouping.
    alpha : float
        Significance threshold for the ``significant`` flag.

    Returns
    -------
    dict with keys:
        n_categories : int
        F            : float  (NaN when fewer than 2 valid groups)
        p_value      : float  (NaN when fewer than 2 valid groups)
        significant  : bool
    """
    categories = sorted(cat_series.dropna().unique())
    cat_arr = cat_series.to_numpy()
    groups_data = [y[cat_arr == cat] for cat in categories]
    valid = [g for g in groups_data if len(g) > 1]

    if len(valid) < 2:
        return dict(
            n_categories=len(categories),
            F=np.nan,
            p_value=np.nan,
            significant=False,
        )

    f_stat, p_val = scipy_stats.f_oneway(*valid)
    return dict(
        n_categories=len(categories),
        F=float(f_stat),
        p_value=float(p_val),
        significant=bool(p_val < alpha),
    )


def anova_spike_counts_for_columns(
    y: np.ndarray,
    binned_feats: pd.DataFrame,
    categorical_cols: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run one-way ANOVA for every column in ``categorical_cols``.

    Parameters
    ----------
    y : np.ndarray, shape (n_bins,)
        Spike counts for one neuron.
    binned_feats : pd.DataFrame
        Feature matrix whose rows are aligned with ``y``.
    categorical_cols : list of str
        Columns of ``binned_feats`` to test.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        One row per variable; columns:
        ``variable``, ``n_categories``, ``F``, ``p_value``, ``significant``
    """
    rows: List[Dict] = []
    for col in categorical_cols:
        result = anova_spike_counts_by_categorical(
            y=y,
            cat_series=binned_feats[col],
            alpha=alpha,
        )
        rows.append({"variable": col, **result})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["variable", "n_categories", "F", "p_value", "significant"]
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers  (private shared logic + public wrappers)
# ---------------------------------------------------------------------------

def _print_test_single_unit(
    result_df: pd.DataFrame,
    unit_idx: int,
    alpha: float,
    label: str,
) -> None:
    """Shared formatter for single-unit ANOVA/LM result tables."""
    n_sig  = int(result_df["significant"].sum())
    header = (
        f"  {label}  unit {unit_idx}  "
        f"({n_sig}/{len(result_df)} significant at α={alpha})  "
    )
    bar = "─" * max(len(header), 50)
    print(f"\n{bar}\n{header}\n{bar}")
    if not result_df.empty:
        disp            = result_df.copy().sort_values("p_value")
        disp["F"]       = disp["F"].map("{:8.2f}".format)
        disp["p_value"] = disp["p_value"].map("{:.4f}".format)
        disp["sig"]     = result_df.loc[disp.index, "significant"].map(
            lambda v: "  *" if v else "   "
        )
        disp = disp.drop(columns=["significant"])
        print(disp.to_string(index=False))
    print(bar)


def _print_test_all_neurons(
    results: Dict[int, pd.DataFrame],
    alpha: float,
    label: str,
) -> None:
    """Shared formatter for all-neuron variable × neuron significance grids."""
    unit_ids  = sorted(results.keys())
    n_neurons = len(unit_ids)

    all_vars: List[str] = list(
        dict.fromkeys(
            var
            for uid in unit_ids
            for var in results[uid]["variable"]
        )
    )
    sig_matrix = {
        var: {
            uid: bool(
                results[uid]
                .loc[results[uid]["variable"] == var, "significant"]
                .values[0]
            )
            if var in results[uid]["variable"].values
            else False
            for uid in unit_ids
        }
        for var in all_vars
    }
    all_vars_sorted = sorted(
        all_vars,
        key=lambda v: sum(sig_matrix[v].values()),
        reverse=True,
    )
    col_w            = 6
    var_w            = max(len(v) for v in all_vars_sorted) + 2
    n_sig_per_neuron = [
        sum(sig_matrix[v][uid] for v in all_vars_sorted)
        for uid in unit_ids
    ]
    header_line = f"{'variable':<{var_w}}" + "".join(
        f"{'n' + str(uid):^{col_w}}" for uid in unit_ids
    ) + f"  {'n_sig':>5}"
    bar   = "─" * len(header_line)
    title = (
        f"  {label} summary  {n_neurons} neurons  "
        f"{len(all_vars_sorted)} variables  α={alpha}  "
    )
    rule = "═" * max(len(title), len(bar))
    print(f"\n{rule}\n{title}\n{rule}")
    print(header_line)
    print(bar)
    for var in all_vars_sorted:
        n_sig_var = sum(sig_matrix[var].values())
        cells = "".join(
            f"{'  *  ' if sig_matrix[var][uid] else '  .  ':^{col_w}}"
            for uid in unit_ids
        )
        print(f"{var:<{var_w}}{cells}  {n_sig_var:>5}")
    print(bar)
    totals = "".join(f"{n_sig_per_neuron[i]:^{col_w}}" for i in range(n_neurons))
    print(f"{'n_sig (neuron)':<{var_w}}{totals}")
    print(f"{rule}\n")


def print_anova_single_unit(
    result_df: pd.DataFrame,
    unit_idx: int,
    alpha: float = 0.05,
) -> None:
    """
    Print a formatted ANOVA result table for a single neuron.

    Rows are sorted by p-value (most significant first).  Significant rows
    are flagged with ``*`` in a trailing ``sig`` column.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of ``anova_spike_counts_for_columns`` for one neuron.
    unit_idx : int
        Neuron index used in the header line.
    alpha : float
        Significance threshold displayed in the header.
    """
    _print_test_single_unit(result_df, unit_idx, alpha, label="ANOVA")


def print_anova_all_neurons(
    results: Dict[int, pd.DataFrame],
    alpha: float = 0.05,
) -> None:
    """
    Print a variable × neuron significance grid for all-neuron ANOVA results.

    Rows correspond to categorical variables (sorted by number of significant
    neurons, descending).  Columns correspond to neuron indices.  Each cell
    shows ``*`` when significant and ``.`` otherwise.  Column and row totals
    are appended.

    Parameters
    ----------
    results : dict
        Mapping ``{unit_idx: result_df}`` as returned by
        ``run_anova_all_neurons``.
    alpha : float
        Significance threshold displayed in the title.
    """
    _print_test_all_neurons(results, alpha, label="ANOVA")


def print_lm_single_unit(
    result_df: pd.DataFrame,
    unit_idx: int,
    alpha: float = 0.05,
) -> None:
    """
    Print a formatted LM partial-F result table for a single neuron.

    Same layout as ``print_anova_single_unit``; rows sorted by p-value with
    ``*`` flags for significant effects.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of ``lm_spike_counts_for_columns`` for one neuron.
    unit_idx : int
        Neuron index used in the header line.
    alpha : float
        Significance threshold displayed in the header.
    """
    _print_test_single_unit(result_df, unit_idx, alpha, label="LM")


def print_lm_all_neurons(
    results: Dict[int, pd.DataFrame],
    alpha: float = 0.05,
) -> None:
    """
    Print a variable × neuron significance grid for all-neuron LM results.

    Mirrors ``print_anova_all_neurons`` but labelled "LM".

    Parameters
    ----------
    results : dict
        Mapping ``{unit_idx: result_df}`` as returned by
        ``run_lm_all_neurons``.
    alpha : float
        Significance threshold displayed in the title.
    """
    _print_test_all_neurons(results, alpha, label="LM")


def plot_anova_results(
    anova_results: Dict[int, pd.DataFrame],
    alpha: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise the output of ``run_anova_all_neurons`` as two panels.

    * **Left** — heatmap of −log₁₀(p-value) for every (variable, neuron)
      pair.  Significant cells are marked with a white star.  A dashed line
      on the colorbar marks the −log₁₀(α) threshold.
    * **Right** — horizontal bar chart: fraction of neurons that are
      significant for each variable (same row order as the heatmap).

    Variables are sorted by number of significant neurons (descending), then
    by minimum p-value, so the most informative variables appear at the top.

    Parameters
    ----------
    anova_results : dict
        ``{unit_idx: DataFrame}`` as returned by ``run_anova_all_neurons``.
        Each DataFrame must have columns:
        ``variable``, ``n_categories``, ``F``, ``p_value``, ``significant``.
    alpha : float
        Significance threshold; controls the colorbar reference line and
        the bar colours.
    figsize : tuple, optional
        ``(width, height)`` in inches.  Auto-scaled from neuron/variable
        counts when omitted.
    title : str, optional
        Figure suptitle.  Pass an empty string to suppress.

    Returns
    -------
    plt.Figure
    """
    if not anova_results:
        return plt.figure()

    # ── assemble (n_vars × n_neurons) matrices ────────────────────────────
    unit_ids = sorted(anova_results.keys())
    all_vars = list(
        dict.fromkeys(
            var
            for uid in unit_ids
            for var in anova_results[uid]["variable"]
        )
    )

    n_vars    = len(all_vars)
    n_neurons = len(unit_ids)
    pval_mat  = np.ones((n_vars, n_neurons))
    sig_mat   = np.zeros((n_vars, n_neurons), dtype=bool)

    for col_i, uid in enumerate(unit_ids):
        df = anova_results[uid].set_index("variable")
        for row_i, var in enumerate(all_vars):
            if var in df.index:
                pval_mat[row_i, col_i] = df.loc[var, "p_value"]
                sig_mat[row_i, col_i]  = bool(df.loc[var, "significant"])

    # Sort rows: most-significant-across-neurons first, then by min p-value
    sort_key = np.lexsort((pval_mat.min(axis=1), -sig_mat.sum(axis=1)))
    all_vars = [all_vars[i] for i in sort_key]
    pval_mat = pval_mat[sort_key]
    sig_mat  = sig_mat[sort_key]

    log_p     = -np.log10(np.clip(pval_mat, 1e-300, 1.0))
    threshold = -np.log10(alpha)
    frac_sig  = sig_mat.sum(axis=1) / n_neurons

    # ── figure layout ─────────────────────────────────────────────────────
    if figsize is None:
        figsize = (
            max(8, n_neurons * 0.65 + 4),
            max(4, n_vars    * 0.55 + 1.8),
        )
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [max(3, n_neurons), 1.6]},
    )
    fig.subplots_adjust(wspace=0.05)

    # ── heatmap ───────────────────────────────────────────────────────────
    vmax = max(threshold * 2.5, log_p.max() * 1.05, threshold + 0.5)
    im   = ax_heat.imshow(
        log_p,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    for row_i in range(n_vars):
        for col_i in range(n_neurons):
            if sig_mat[row_i, col_i]:
                ax_heat.text(
                    col_i, row_i, "★",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
    ax_heat.set_xticks(range(n_neurons))
    ax_heat.set_xticklabels([f"n{uid}" for uid in unit_ids], fontsize=8)
    ax_heat.set_yticks(range(n_vars))
    ax_heat.set_yticklabels(all_vars, fontsize=9)
    ax_heat.set_xlabel("neuron", fontsize=9)
    ax_heat.set_title("−log₁₀(p-value)  [★ = significant]", fontsize=9)

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    cb.ax.axhline(threshold, color="steelblue", linewidth=1.5, linestyle="--")
    cb.ax.text(
        1.1, threshold / vmax, f"α={alpha}",
        transform=cb.ax.transAxes,
        va="center", ha="left", fontsize=7.5, color="steelblue",
    )
    cb.set_label("−log₁₀(p)", fontsize=8)

    # ── fraction-significant bars ─────────────────────────────────────────
    bar_colors = [
        "#d73027" if f >= 0.5 else "#fc8d59" if f > 0 else "#e0e0e0"
        for f in frac_sig
    ]
    ax_bar.barh(
        range(n_vars), frac_sig,
        color=bar_colors, edgecolor="white", linewidth=0.5,
    )
    ax_bar.axvline(0.5, color="grey", linewidth=0.8, linestyle=":")
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_ylim(-0.5, n_vars - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xticks([0, 0.5, 1])
    ax_bar.set_xticklabels(["0", ".5", "1"], fontsize=8)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("frac. sig.", fontsize=9)
    ax_bar.set_title("neurons\nsignificant", fontsize=9)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.01)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LM (OLS with partial F-tests)
# ---------------------------------------------------------------------------

def lm_spike_counts_for_columns(
    y: np.ndarray,
    binned_feats: pd.DataFrame,
    categorical_cols: List[str],
    continuous_cols: Optional[List[str]] = None,
    covariate_cols: Optional[List[str]] = None,
    reduce_design: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Fit a single OLS model with all predictors and test each categorical
    variable via a partial F-test (Type-II sum of squares: drop-one approach).

    Each categorical variable is dummy-coded (``drop_first=True``).  Optional
    continuous columns are included as covariates to control for their
    variance, and their partial effects are also reported.

    The partial F-test controls for every other predictor simultaneously,
    making results directly comparable with one-way ANOVA:

    * If both ANOVA and LM are significant → effect is robust to other vars.
    * ANOVA sig but LM not → the univariate effect was confounded.
    * LM sig but ANOVA not → the effect was suppressed in isolation.

    Parameters
    ----------
    y : np.ndarray, shape (n_bins,)
        Spike counts (or rates) for one neuron.
    binned_feats : pd.DataFrame
        Feature matrix aligned with ``y``.
    categorical_cols : list of str
        Columns to dummy-encode and test (reported in output).
    continuous_cols : list of str, optional
        Continuous predictors that are both included in the model and
        reported in the output (their partial F-tests appear as rows).
    covariate_cols : list of str, optional
        Columns included in the design matrix purely to absorb variance
        (e.g. the full feature set from ``feats_to_decode``).  They are
        **not** tested or reported — only ``categorical_cols`` and
        ``continuous_cols`` appear in the returned DataFrame.  When
        ``reduce_design=True`` these are pre-filtered by the runner via
        caching; when calling this function directly pass unreduced covariates
        and set ``reduce_design=True`` to let the function handle it.
    reduce_design : bool
        When ``True``, apply ``process_encode_design.reduce_encoding_design``
        (corr threshold 0.95, VIF threshold 20) to the **full assembled
        design matrix** (test-variable dummies + continuous + covariates)
        so that near-collinear columns are removed in their joint context.
        Any test-variable columns that the reduction would drop are
        **always added back** before fitting; only covariate columns can
        be discarded.  Defaults to ``False`` because the runner's
        ``_resolve_lm_covariate_cols`` already performs this reduction in
        the correct context when caching is used.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        One row per tested variable; columns:
        ``variable``, ``n_params``, ``F``, ``p_value``, ``significant``

        ``n_params`` is the number of dummy columns for the variable
        (= n_categories − 1 for categoricals, 1 for continuous).
    """
    import statsmodels.api as sm

    y_arr      = np.asarray(y, dtype=float).ravel()
    cont_cols  = list(continuous_cols or [])
    cov_cols   = list(covariate_cols  or [])
    test_vars  = list(categorical_cols) + cont_cols

    if not test_vars:
        return pd.DataFrame(
            columns=["variable", "n_params", "F", "p_value", "significant"]
        )

    # ── design matrix ─────────────────────────────────────────────────────
    parts: List[pd.DataFrame] = []
    col_groups: Dict[str, List[str]] = {}

    for col in categorical_cols:
        dummies = pd.get_dummies(
            binned_feats[col], prefix=col, drop_first=True, dtype=float
        )
        col_groups[col] = list(dummies.columns)
        parts.append(dummies)

    for col in cont_cols:
        col_groups[col] = [col]
        parts.append(binned_feats[[col]].astype(float))

    # Covariates: included in full model but not tested — de-duplicate against
    # columns already added so the design matrix stays full-rank.
    already_added = {c for cols in col_groups.values() for c in cols}
    for col in cov_cols:
        if col not in already_added:
            parts.append(binned_feats[[col]].astype(float))
            already_added.add(col)

    X_full     = pd.concat(parts, axis=1)
    valid_mask = np.isfinite(y_arr) & X_full.notna().all(axis=1).to_numpy()
    y_v        = y_arr[valid_mask]
    X_v        = X_full.loc[valid_mask]

    # ── optional design-matrix reduction ──────────────────────────────────
    if reduce_design:
        from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
            process_encode_design,
        )
        # Protected columns are the test-variable dummies + continuous_cols:
        # these must survive regardless of what the reduction decides.
        protected: set = {c for cols in col_groups.values() for c in cols}

        X_v_reduced = process_encode_design.reduce_encoding_design(
            X_v,
            corr_threshold_for_lags=0.9,
            vif_threshold=5,
            verbose=False,
        )
        # Add back any protected columns the reduction dropped
        dropped_protected = [c for c in protected if c not in X_v_reduced.columns]
        if dropped_protected:
            X_v_reduced = pd.concat(
                [X_v_reduced, X_v[dropped_protected]], axis=1
            )
        X_v = X_v_reduced

    X_v = sm.add_constant(X_v, has_constant="add")

    def _nan_row(var: str) -> Dict:
        return {"variable": var, "n_params": len(col_groups[var]),
                "F": np.nan, "p_value": np.nan, "significant": False}

    if len(y_v) < X_v.shape[1] + 2:
        return pd.DataFrame([_nan_row(v) for v in test_vars])

    full_model  = sm.OLS(y_v, X_v).fit()
    rss_full    = full_model.ssr
    df_residual = full_model.df_resid

    rows: List[Dict] = []
    for var in test_vars:
        var_cols  = col_groups[var]
        n_params  = len(var_cols)
        X_reduced = X_v[[c for c in X_v.columns if c not in var_cols]]

        rss_reduced = sm.OLS(y_v, X_reduced).fit().ssr

        if df_residual > 0 and rss_full > 1e-14:
            F_stat = ((rss_reduced - rss_full) / n_params) / (rss_full / df_residual)
            p_val  = float(scipy_stats.f.sf(max(float(F_stat), 0.0), n_params, df_residual))
        else:
            F_stat, p_val = np.nan, np.nan

        rows.append({
            "variable":    var,
            "n_params":    n_params,
            "F":           float(F_stat) if np.isfinite(F_stat) else np.nan,
            "p_value":     p_val,
            "significant": bool(p_val < alpha) if p_val is not None and np.isfinite(p_val) else False,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LM vs ANOVA comparison plot
# ---------------------------------------------------------------------------

def plot_lm_vs_anova(
    anova_results: Dict[int, pd.DataFrame],
    lm_results: Dict[int, pd.DataFrame],
    alpha: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Compare ANOVA and LM partial-F results across all neurons.

    Two panels:

    * **Left** — scatter of −log₁₀(p) values.  Each point is one
      (variable, neuron) pair; x-axis = ANOVA p, y-axis = LM p.
      Points are coloured by variable.  Dashed lines mark the α threshold on
      each axis; the diagonal y = x represents perfect agreement.
      Quadrant labels indicate the interpretation of each region.

    * **Right** — agreement grid.  Rows = variables (same sort order as the
      left panel), columns = neurons.  Each cell is one of four states:

      ■ green  — both significant
      ■ orange — ANOVA only
      ■ blue   — LM only
      ■ grey   — neither

    Parameters
    ----------
    anova_results : dict
        ``{unit_idx: DataFrame}`` from ``run_anova_all_neurons``.
    lm_results : dict
        ``{unit_idx: DataFrame}`` from ``run_lm_all_neurons``.
    alpha : float
        Significance threshold.
    figsize : tuple, optional
        Auto-scaled when omitted.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    plt.Figure
    """
    if not anova_results or not lm_results:
        return plt.figure()

    unit_ids = sorted(set(anova_results) & set(lm_results))
    all_vars = list(
        dict.fromkeys(
            var
            for uid in unit_ids
            for var in anova_results[uid]["variable"]
            if var in lm_results[uid]["variable"].values
        )
    )

    n_vars    = len(all_vars)
    n_neurons = len(unit_ids)

    # ── assemble aligned arrays ───────────────────────────────────────────
    anova_p = np.ones((n_vars, n_neurons))
    lm_p   = np.ones((n_vars, n_neurons))

    for col_i, uid in enumerate(unit_ids):
        a_df = anova_results[uid].set_index("variable")
        l_df = lm_results[uid].set_index("variable")
        for row_i, var in enumerate(all_vars):
            if var in a_df.index:
                anova_p[row_i, col_i] = a_df.loc[var, "p_value"]
            if var in l_df.index:
                lm_p[row_i, col_i]   = l_df.loc[var, "p_value"]

    # Sort rows: most frequently sig in either test first
    a_sig = anova_p < alpha
    l_sig = lm_p   < alpha
    sort_key = np.lexsort(
        (anova_p.min(axis=1), -(a_sig | l_sig).sum(axis=1))
    )
    all_vars = [all_vars[i] for i in sort_key]
    anova_p  = anova_p[sort_key]
    lm_p    = lm_p[sort_key]
    a_sig    = a_sig[sort_key]
    l_sig    = l_sig[sort_key]

    log_a     = -np.log10(np.clip(anova_p, 1e-300, 1.0))
    log_l     = -np.log10(np.clip(lm_p,   1e-300, 1.0))
    threshold = -np.log10(alpha)

    # ── figure layout ─────────────────────────────────────────────────────
    if figsize is None:
        figsize = (max(10, n_neurons * 0.4 + 6), max(5, n_vars * 0.45 + 2))
    fig, (ax_sc, ax_grid) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.4, max(2, n_neurons * 0.4)]},
    )
    fig.subplots_adjust(wspace=0.35)

    # ── scatter ───────────────────────────────────────────────────────────
    cmap   = plt.cm.get_cmap("tab10", n_vars)
    lim    = max(log_a.max(), log_l.max()) * 1.1 + 0.3

    # quadrant shading
    ax_sc.axhspan(threshold, lim,  xmin=0, xmax=threshold / lim,
                  color="#fff3e0", alpha=0.6, zorder=0)   # LM only
    ax_sc.axhspan(0, threshold,    xmin=threshold / lim, xmax=1.0,
                  color="#e3f2fd", alpha=0.6, zorder=0)   # ANOVA only
    ax_sc.axhspan(threshold, lim,  xmin=threshold / lim, xmax=1.0,
                  color="#e8f5e9", alpha=0.6, zorder=0)   # both sig

    ax_sc.plot([0, lim], [0, lim],  color="grey",      lw=0.8, ls="--", zorder=1)
    ax_sc.axvline(threshold, color="#e65100", lw=0.9, ls=":", zorder=1)
    ax_sc.axhline(threshold, color="#1565c0", lw=0.9, ls=":", zorder=1)

    for row_i, var in enumerate(all_vars):
        color = cmap(row_i)
        ax_sc.scatter(
            log_a[row_i], log_l[row_i],
            color=color, s=28, alpha=0.75, linewidths=0,
            label=var, zorder=2,
        )

    ax_sc.set_xlim(0, lim)
    ax_sc.set_ylim(0, lim)
    ax_sc.set_xlabel("ANOVA  −log₁₀(p)", fontsize=9)
    ax_sc.set_ylabel("LM  −log₁₀(p)",   fontsize=9)
    ax_sc.set_title("Effect consistency\n(each point = variable × neuron)", fontsize=9)
    ax_sc.text(lim * 0.02, lim * 0.88, "LM only",  fontsize=7, color="#e65100")
    ax_sc.text(lim * 0.60, lim * 0.05, "ANOVA only", fontsize=7, color="#1565c0")
    ax_sc.text(lim * 0.60, lim * 0.88, "both sig",   fontsize=7, color="#2e7d32")
    ax_sc.legend(
        fontsize=7, markerscale=1.2, framealpha=0.7,
        loc="lower right", title="variable", title_fontsize=7,
    )

    # ── agreement grid ────────────────────────────────────────────────────
    #   state: 0=neither, 1=ANOVA only, 2=LM only, 3=both
    state = (a_sig.astype(int) + 2 * l_sig.astype(int))   # 0,1,2,3
    palette = np.array([
        [0.88, 0.88, 0.88, 1.0],   # 0  neither     grey
        [1.00, 0.60, 0.10, 1.0],   # 1  ANOVA only  orange
        [0.15, 0.47, 0.74, 1.0],   # 2  LM only    blue
        [0.17, 0.63, 0.17, 1.0],   # 3  both        green
    ])
    rgb = palette[state]           # (n_vars, n_neurons, 4)
    ax_grid.imshow(rgb, aspect="auto", interpolation="nearest")

    ax_grid.set_xticks(range(n_neurons))
    ax_grid.set_xticklabels([f"n{uid}" for uid in unit_ids], fontsize=8)
    ax_grid.set_yticks(range(n_vars))
    ax_grid.set_yticklabels(all_vars, fontsize=9)
    ax_grid.set_xlabel("neuron", fontsize=9)
    ax_grid.set_title("Agreement\n(ANOVA vs LM)", fontsize=9)

    # legend patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=palette[3, :3], label="both sig"),
        Patch(color=palette[1, :3], label="ANOVA only"),
        Patch(color=palette[2, :3], label="LM only"),
        Patch(color=palette[0, :3], label="neither"),
    ]
    ax_grid.legend(
        handles=legend_handles, fontsize=7,
        loc="lower right", framealpha=0.85,
        bbox_to_anchor=(1.0, -0.22), ncol=2,
    )

    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.02)

    plt.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Fraction tuned — neuroGAM-compatible (based on backward elimination)
# ---------------------------------------------------------------------------

def fraction_tuned_from_elimination(
    elimination_results: Dict[int, Dict],
    variable_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute the fraction of neurons tuned to each variable using backward
    elimination results.  Matches neuroGAM's definition: a neuron is tuned
    to variable X if X survives backward elimination in the joint model AND
    the best model is significantly better than the null (neurons where
    ``kept_groups`` is empty are counted as tuned to nothing).

    Parameters
    ----------
    elimination_results : dict
        Mapping ``{unit_idx: result_dict}`` where each ``result_dict`` is the
        output of ``PGAMModel.run_backward_elimination`` or
        ``BaseEncodingGAMAnalysisHelper.run_backward_elimination``.
        Each dict must have a ``kept_groups`` key containing a list of
        ``GroupSpec``-like objects (or dicts) with a ``name`` attribute/key.
    variable_names : list of str, optional
        Ordered list of variable names to include.  When ``None``, all names
        found across any neuron's kept groups are used (sorted).

    Returns
    -------
    pd.DataFrame
        One row per variable; columns:
        ``variable``, ``n_tuned``, ``n_neurons``, ``fraction_tuned``

        Neurons where ``kept_groups`` is empty (null model not rejected) are
        counted in ``n_neurons`` but contribute 0 to ``n_tuned`` for all
        variables, exactly as in neuroGAM.
    """
    n_neurons = len(elimination_results)
    if n_neurons == 0:
        cols = ["variable", "n_tuned", "n_neurons", "fraction_tuned"]
        return pd.DataFrame(columns=cols)

    # Collect kept group names per neuron
    def _kept_names(result: Dict) -> set:
        kept = result.get("kept_groups", [])
        names = set()
        for g in kept:
            # GroupSpec object (has .name) or plain dict (has 'name' key)
            name = g.name if hasattr(g, "name") else g.get("name")
            if name is not None:
                names.add(name)
        return names

    kept_per_neuron = {uid: _kept_names(res) for uid, res in elimination_results.items()}

    # Determine variable universe
    if variable_names is None:
        variable_names = sorted({v for names in kept_per_neuron.values() for v in names})

    rows = []
    for var in variable_names:
        n_tuned = sum(1 for names in kept_per_neuron.values() if var in names)
        rows.append({
            "variable": var,
            "n_tuned": n_tuned,
            "n_neurons": n_neurons,
            "fraction_tuned": n_tuned / n_neurons if n_neurons > 0 else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("fraction_tuned", ascending=False).reset_index(drop=True)


def print_fraction_tuned(
    df: pd.DataFrame,
    title: str = "Fraction tuned (backward elimination)",
) -> None:
    """
    Pretty-print the output of ``fraction_tuned_from_elimination``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``fraction_tuned_from_elimination``.
    title : str
        Header line.
    """
    bar = "─" * max(len(title) + 4, 50)
    print(f"\n{bar}\n  {title}\n{bar}")
    if df.empty:
        print("  (no results)")
        print(bar)
        return
    for _, row in df.iterrows():
        n_tuned   = int(row["n_tuned"])
        n_neurons = int(row["n_neurons"])
        frac      = row["fraction_tuned"]
        stars     = "█" * round(frac * 20)
        print(f"  {row['variable']:<30}  {n_tuned:>3}/{n_neurons}  "
              f"({100 * frac:5.1f}%)  {stars}")
    print(bar)