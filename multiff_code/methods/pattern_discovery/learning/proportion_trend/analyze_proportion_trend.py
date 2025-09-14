
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter
import seaborn as sns


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact, norm
from math import sqrt
import statsmodels.api as sm


# --- NEW: helpers for Early vs Late p-value ---------------------------------
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact
import numpy as np


def test_early_late(phase_df_sub, item):
    k1 = int(phase_df_sub.loc[phase_df_sub['phase']
             == 'early', 'frequency'].iloc[0])
    n1 = int(phase_df_sub.loc[phase_df_sub['phase']
             == 'early', 'denom_count'].iloc[0])
    k2 = int(phase_df_sub.loc[phase_df_sub['phase']
             == 'late', 'frequency'].iloc[0])
    n2 = int(phase_df_sub.loc[phase_df_sub['phase']
             == 'late', 'denom_count'].iloc[0])
    pval_el, test_name = _pval_early_vs_late(k1, n1, k2, n2)
    return pval_el, test_name


def _pval_early_vs_late(k1, n1, k2, n2):
    """Two-proportion z-test by default; use Fisher if any cell < 5."""
    # Edge cases
    if n1 == 0 or n2 == 0:
        return np.nan, "NA"  # not testable

    table = np.array([[k1, n1 - k1],
                      [k2, n2 - k2]], dtype=int)

    if (table < 5).any():  # small counts → exact test
        _, p = fisher_exact(table, alternative="two-sided")
        return float(p), "Fisher"
    else:
        _, p = proportions_ztest(count=[k1, k2], nobs=[
                                 n1, n2], alternative="two-sided")
        return float(p), "Z-test"


def _early_late_agg(df, session_col, event_count_col, denom_count_col='denom_count'):
    """Aggregate early/late totals (using your tertile split)."""
    ses = tertile_phase(df, session_col=session_col)
    el = (
        ses[ses["phase"].isin(["early", "late"])]
        .groupby("phase", as_index=False, observed=True)
        .agg(**{
            event_count_col: (event_count_col, "sum"),
            denom_count_col: (denom_count_col, "sum")
        })
        .set_index("phase")
        .reindex(["early", "late"])
    )
    # ensure integers
    k1 = int(el.loc["early", event_count_col])
    n1 = int(el.loc["early", denom_count_col])
    k2 = int(el.loc["late", event_count_col])
    n2 = int(el.loc["late", denom_count_col])
    return k1, n1, k2, n2, el.reset_index()


def evaluate_proportion_trend(df_sess_counts, event_count_col="success", denom_count_col="stops",
                              title=None, ylabel="P(Events | Baseline)"):

    glm_pois = smf.glm(
        f"{event_count_col} ~ session",
        data=df_sess_counts,
        family=sm.families.Poisson(),
        offset=np.log(df_sess_counts[denom_count_col].clip(
            lower=1))  # keep your offset guard
    ).fit(cov_type="HC0")

    # Collect summaries (trend over sessions)
    results = [summarize_glm(glm_pois, "Poisson")]
    results_df = pd.DataFrame(results)
    print(results_df)

    # p-value for TREND (keep on the session fit plot)
    pval_trend = results_df.iloc[0]['pval']

    # p-value for EARLY vs LATE (use on the bar+CI plot)
    k1, n1, k2, n2, _ = _early_late_agg(
        df_sess_counts, "session", event_count_col, denom_count_col)
    pval_el, test_name = _pval_early_vs_late(k1, n1, k2, n2)

    if title is None:
        title = f"Ratio of {event_count_col}"

    # --- Plot session trend (annotate GLM p-val) ---
    plot_poisson_proportion_fit(
        df_sess_counts, glm_pois,
        session_col="session", event_count_col=event_count_col, denom_count_col=denom_count_col,
        title=title, ylabel=ylabel,
        pval=pval_trend
    )

    # --- Plot Early vs Late (annotate group-diff p-val) ---
    plot_early_late_proportion(
        df_sess_counts,
        session_col="session", event_count_col=event_count_col, denom_count_col=denom_count_col,
        ylabel=ylabel, title=f"Early vs Late ({test_name})",
        pval=pval_el
    )

    plt.show()


# ------------------
# Utilities
# ------------------


def wilson_ci(k, n, alpha=0.05):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - alpha/2)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - half), min(1, center + half)


def add_wilson_to_session_counts(df_sessions, event_count_col="success", denom_col="stops"):
    """Add p_hat and Wilson CI columns to session-level counts."""
    out = df_sessions.copy()
    p = out[event_count_col] / out[denom_col].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n))
                   for k, n in zip(out[event_count_col], out[denom_col])])
    out["p_hat"] = p
    out["p_lo"] = ci[:, 0]
    out["p_hi"] = ci[:, 1]
    return out


def tertile_phase(df, session_col='session'):
    """
    Add 'phase' ∈ {early, mid, late} based on tertiles of the *unique session numbers*.
    Falls back to a median split if fewer than 3 unique sessions.
    """
    g = df.copy()
    g[session_col] = pd.to_numeric(g[session_col])

    # Get sorted unique sessions
    uniq = np.sort(df[session_col].unique())
    n = len(uniq)

    if n >= 3:
        # Assign phase per unique session index
        cut_points = np.linspace(0, n, 4).astype(int)  # [0, n/3, 2n/3, n]
        bins = [uniq[cut_points[0]:cut_points[1]],
                uniq[cut_points[1]:cut_points[2]],
                uniq[cut_points[2]:cut_points[3]]]

        mapping = {}
        for phase, vals in zip(['early', 'mid', 'late'], bins):
            mapping.update({s: phase for s in vals})

        g['phase'] = g[session_col].map(mapping)
    else:
        # Fallback to median split
        med = np.median(uniq)
        g['phase'] = np.where(g[session_col] <= med, 'early', 'late')

    return g


def add_pval_to_plot(ax, pval, fontsize=12):
    """Annotate an Axes with p-value text and stars if significant."""
    # p-value text
    if pval < 0.001:
        pval_text = 'p < 0.001'
    elif pval < 0.01:
        pval_text = f'p = {pval:.3f}'
    elif pval < 0.05:
        pval_text = f'p = {pval:.3f}'
    else:
        pval_text = f'p = {pval:.3f}'

    # significance stars
    if pval < 0.001:
        stars = '***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = ''

    if stars:
        pval_text = f'{pval_text} {stars}'

    ax.text(
        0.02, 0.98, pval_text,
        transform=ax.transAxes,        # anchor to Axes coordinates
        va='top', ha='left',           # vertical/horizontal alignment
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=fontsize
    )



def summarize_glm(model, label):
    """Return dict of effect per 10 sessions."""
    coef = model.params["session"]
    se = model.bse["session"]
    z = coef / se
    pval = model.pvalues["session"]

    # 95% CI
    ci_low, ci_high = model.conf_int().loc["session"].tolist()

    # Rate ratio interpretation
    effect = np.exp(coef * 10)  # rate ratio per 10 sessions
    ci_low_eff, ci_high_eff = np.exp(ci_low * 10), np.exp(ci_high * 10)
    return {
        "model": label,
        "coef": coef,
        "se": se,
        "z": z,
        "pval": pval,
        "rate_ratio_per_10_sessions": effect,
        "95% CI": f"[{ci_low_eff:.3f}, {ci_high_eff:.3f}]"
    }


def plot_early_late_proportion(
    df_sess_counts,
    session_col='session',
    event_count_col='success',
    denom_count_col='stops',
    ylabel='P(captures | stops)',
    title='Early vs Late',
    pval=None
):
    """
    Hybrid chart: translucent bars + point estimate + 95% CI (Wilson).
    Annotates the *group difference* p-value (two-proportion z-test or Fisher).
    Assumes a helper `wilson_ci(k, n)` -> (lo, hi) on the proportion scale.
    """
    sns.set_style(style='white')

    # --- Aggregate early/late ---
    k1, n1, k2, n2, el = _early_late_agg(
        df_sess_counts, session_col, event_count_col, denom_count_col
    )

    # >>> INSERT FIXES HERE (consistent denominator handling & zero-n guard) <<<
    el = el.copy()
    # Initialize columns
    el['p_hat'] = np.nan
    el['p_lo'] = np.nan
    el['p_hi'] = np.nan

    zero_n = el[denom_count_col].to_numpy() == 0
    mask = ~zero_n

    # Compute p_hat and Wilson CI using the same n (no clip divergence)
    if mask.any():
        # Proportions
        el.loc[mask, 'p_hat'] = (
            el.loc[mask, event_count_col].astype(float).to_numpy() /
            el.loc[mask, denom_count_col].astype(float).to_numpy()
        )
        # Wilson CI (vectorized loop for clarity)
        ci = np.array([
            wilson_ci(int(k), int(n))
            for k, n in zip(el.loc[mask, event_count_col], el.loc[mask, denom_count_col])
        ])
        el.loc[mask, 'p_lo'] = ci[:, 0]
        el.loc[mask, 'p_hi'] = ci[:, 1]

    # Order for plotting
    dfp = (
        el.set_index('phase')
        # ensures order; will raise if missing; catch below if needed
        .loc[['early', 'late']]
        .reset_index()
    )

    # Convert to percentages for display
    dfp['pct'] = 100.0 * dfp['p_hat']
    dfp['pct_lo'] = 100.0 * dfp['p_lo']
    dfp['pct_hi'] = 100.0 * dfp['p_hi']

    # --- Plot ---
    x = np.array([-0.3, 0.3], dtype=float)
    pct = dfp['pct'].to_numpy(float)
    pctlo = dfp['pct_lo'].to_numpy(float)
    pcthi = dfp['pct_hi'].to_numpy(float)

    # >>> INSERT FIXES HERE (non-negative yerr & NaN-safe) <<<
    with np.errstate(invalid='ignore'):
        yerr = np.vstack([
            np.clip(pct - pctlo, 0.0, None),
            np.clip(pcthi - pct, 0.0, None)
        ])

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.bar(x, pct, width=0.35, alpha=0.35, zorder=1, color='tab:blue')
    ax.errorbar(x, pct, yerr=yerr, fmt='o', lw=2, capsize=5,
                zorder=3, color='tab:blue', ecolor='tab:blue')
    ax.scatter(x, pct, s=40, zorder=4, color='tab:blue')

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(['Early', 'Late'], fontsize=12)

    # k/n labels above bars (skip if NaN)
    for xi, yi, k, n in zip(x, pct, dfp[event_count_col], dfp[denom_count_col]):
        if np.isfinite(yi):
            ax.text(xi, yi + 2.0, f'{int(k)}/{int(n)}',
                    ha='center', va='bottom', fontsize=9)

    # Y-axis formatting
    ymax = float(np.nanmax([np.nanmax(pcthi), np.nanmax(pct) + 6.0]))
    ymax = min(100.0, ymax * 1.10 if np.isfinite(ymax) else 100.0)
    ax.set_ylim(0, ymax)
    # Percent formatter: ticks are 0–100
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))

    ax.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    # >>> INSERT FIXES HERE (robust p-value computation) <<<
    # Compute p-value here if not provided and both groups are valid
    if pval is None:
        # Valid if neither group has zero denominator and both phases present
        phases_ok = set(dfp['phase'].tolist()) == {'early', 'late'}
        if phases_ok and not zero_n.any():
            pval, _ = _pval_early_vs_late(k1, n1, k2, n2)

    # Annotate p-value (if testable)
    if pval is not None and np.isfinite(pval):
        ptxt = 'p < 0.001' if pval < 1e-3 else f'p = {pval:.3f}'
        ax.text(0.02, 0.98, ptxt, transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Results table
    display_cols = [event_count_col, denom_count_col, 'p_hat', 'p_lo', 'p_hi']
    print('\n--- Early vs Late Results ---')
    print(dfp[['phase'] + display_cols].to_string(index=False))

    return dfp[['phase'] + display_cols]


def show_event_proportion(df_monkey, event, title=None, ylabel=None):
    df_event = df_monkey[df_monkey['item'] == event].sort_values(
        by='session').reset_index(drop=True)

    event_count_col = event
    denom_count_col = "denom_count"

    df_event = df_event.rename(columns={'frequency': event_count_col,
                                        }, errors='ignore')

    evaluate_proportion_trend(df_event, event_count_col=event_count_col,
                              denom_count_col=denom_count_col, title=title, ylabel=ylabel)


from typing import Literal, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_poisson_proportion_fit(
    df_sessions_counts: pd.DataFrame,
    glm_pois,  # single fitted model OR dict {monkey_name: fitted_model}
    session_col: str = 'session',
    event_count_col: str = 'success',
    denom_count_col: str = 'stops',
    title: str = 'Ratio across sessions',
    ylabel: str = 'Ratio',
    pval: Optional[float] = None,
    *,
    monkey_col: str = 'monkey',
    monkeys: Tuple[str, str] = ('Bruno', 'Schro'),
):
    """
    If df_sessions_counts has a `monkey` column and `glm_pois` is a dict keyed by monkey,
    draws side-by-side panels for the requested `monkeys`. Otherwise behaves like single-plot.
    Assumes glm_pois is a Poisson GLM: success ~ session + offset(log(stops)).
    """

    sns.set_style(style='white')

    def _observed_df(d: pd.DataFrame) -> pd.DataFrame:
        # Add Wilson CI per session (expects columns named 'success' and 'stops')
        plot_df = add_wilson_to_session_counts(
            d.rename(columns={event_count_col: 'success', denom_count_col: 'stops'}),
            event_count_col='success', denom_col='stops'
        ).sort_values(session_col)
        return plot_df

    def _pred_df(model, d: pd.DataFrame) -> pd.DataFrame:
        # Predict expected successes per session using its own offset; divide by stops to get rate
        grid = d[[session_col, denom_count_col]].drop_duplicates().sort_values(session_col)
        preds = []
        for s, n in zip(grid[session_col], grid[denom_count_col]):
            n_safe = max(1.0, float(n))
            sf = model.get_prediction(
                pd.DataFrame({session_col: [s]}),
                offset=np.log([n_safe])
            ).summary_frame()
            rate = sf['mean'].iloc[0] / n_safe
            lo = sf['mean_ci_lower'].iloc[0] / n_safe
            hi = sf['mean_ci_upper'].iloc[0] / n_safe
            preds.append((s, rate, lo, hi))
        pred_df = pd.DataFrame(preds, columns=[session_col, 'fit_rate', 'fit_lo', 'fit_hi'])
        return pred_df

    def _get_pval(model, fallback: float = 1.0) -> float:
        if pval is not None:
            return float(pval)
        try:
            return float(getattr(model, 'pvalues', {}).get('session', fallback))
        except Exception:
            return fallback

    def _plot_on_ax(ax, obs: pd.DataFrame, pred: pd.DataFrame, panel_title: str,
                    p_value: float, *, markersize: int, font_axis: int,
                    font_tick: int, font_title: int, font_legend: int):
        yerr = np.vstack([obs['p_hat'] - obs['p_lo'], obs['p_hi'] - obs['p_hat']])
        ax.errorbar(
            obs[session_col], obs['p_hat'],
            yerr=yerr, fmt='o', capsize=3, alpha=0.85,
            markersize=markersize,
            label='Observed (Wilson 95% CI)'
        )
        ax.plot(pred[session_col], pred['fit_rate'], lw=2, label='Poisson offset fit')
        ax.fill_between(pred[session_col], pred['fit_lo'], pred['fit_hi'], alpha=0.2, label='95% CI')

        ax.set_xlabel('Session', fontsize=font_axis)
        ax.set_title(panel_title, fontsize=font_title, y=1.02)
        ax.tick_params(axis='both', labelsize=font_tick)

        # Percent formatter
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # p-value box (uses your existing helper)
        add_pval_to_plot(ax, p_value)

        # y-lims per panel will be set globally outside

    # --- Single-plot mode -------------------------------------------------------
    if monkey_col not in df_sessions_counts.columns or not isinstance(glm_pois, dict):
        obs = _observed_df(df_sessions_counts)
        pred = _pred_df(glm_pois, df_sessions_counts)
        p_value = _get_pval(glm_pois)

        plt.figure(figsize=(5.6, 4.2), dpi=300)
        ax = plt.gca()
        _plot_on_ax(
            ax, obs, pred,
            title, p_value,
            markersize=6, font_axis=13, font_tick=12, font_title=15, font_legend=12
        )
        ax.set_ylabel(ylabel, fontsize=13)
        # Global y-lims
        y_min = min(obs['p_lo'].min(), pred['fit_lo'].min())
        y_max = max(obs['p_hi'].max(), pred['fit_hi'].max())
        ax.set_ylim(y_min * 0.9, y_max * 1.2)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        return

    # --- Side-by-side mode ------------------------------------------------------
    df2 = df_sessions_counts[df_sessions_counts[monkey_col].isin(monkeys)].copy()
    panels = []
    for m in monkeys:
        d_m = df2[df2[monkey_col] == m]
        if d_m.empty or m not in glm_pois:
            continue
        obs_m = _observed_df(d_m)
        pred_m = _pred_df(glm_pois[m], d_m)
        p_m = _get_pval(glm_pois[m])
        panels.append((m, obs_m, pred_m, p_m))

    if not panels:
        raise ValueError('No matching monkeys/models found to plot.')

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.4), dpi=300, sharey=True)
    if n == 1:
        axes = [axes]

    # Font & marker sizes for side-by-side mode (matching your rate plot)
    marker_size = 6
    font_tick = 12
    font_axis = 13
    font_title = 15
    font_legend = 12

    # Global y-limits across panels
    global_min = np.inf
    global_max = -np.inf
    for _, obs_m, pred_m, _ in panels:
        global_min = min(global_min, obs_m['p_lo'].min(), pred_m['fit_lo'].min())
        global_max = max(global_max, obs_m['p_hi'].max(), pred_m['fit_hi'].max())

    for i, (ax, (m, obs_m, pred_m, p_m)) in enumerate(zip(axes, panels)):
        panel_title = f'{m}: {title}'
        _plot_on_ax(
            ax, obs_m, pred_m, panel_title, p_m,
            markersize=marker_size, font_axis=font_axis, font_tick=font_tick,
            font_title=font_title, font_legend=font_legend
        )

        # Left panel keeps y-label; right panel drops it but keeps tick labels
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=font_axis)
        else:
            ax.set_ylabel('')
            ax.legend(fontsize=font_legend)
        ax.yaxis.set_tick_params(labelleft=True)

        # Apply shared y-lims
        ax.set_ylim(global_min * 0.9, global_max * 1.2)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.show()
