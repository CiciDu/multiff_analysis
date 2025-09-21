from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pattern_discovery.learning.proportion_trend import analyze_proportion_trend


def fit_poisson_by_session(df_sessions: pd.DataFrame,
                           *, session_col: str = "session",
                           count_col: str = "captures",
                           exposure_col: str = "total_duration"):
    """Replicate your manual Poisson: captures ~ session with log(exposure) offset."""
    offset = np.log(df_sessions[exposure_col].astype(float))
    po = smf.glm(formula=f"{count_col} ~ {session_col}", data=df_sessions,
                 family=sm.families.Poisson(), offset=offset).fit(cov_type="HC0")
    return po


def predict_poisson_rate_curve(po, session_min: int, session_max: int, *, per: Literal["second", "minute"] = "minute"):
    sess_grid = pd.DataFrame(
        {"session": np.arange(session_min, session_max + 1)})
    base_exposure = 60.0 if per == "minute" else 1.0
    pred = po.get_prediction(exog=sess_grid, offset=np.log(
        np.full(len(sess_grid), base_exposure))).summary_frame()
    out = sess_grid.copy()
    out["fit"] = pred["mean"].to_numpy()
    out["p_lo"] = pred["mean_ci_lower"].to_numpy()
    out["p_hi"] = pred["mean_ci_upper"].to_numpy()
    return out



def fit_ols_logT_by_session(df_trials: pd.DataFrame,
                            *, session_col: str = "session",
                            value_col: str = "duration_sec"):
    trials = df_trials[[session_col, value_col]].copy()
    trials["logT"] = np.log(trials[value_col].astype(float))
    ols = smf.ols("logT ~ session", data=trials).fit(
        cov_type="cluster", cov_kwds={"groups": trials[session_col]})
    return ols


def predict_duration_curve_seconds(ols, session_min: int, session_max: int):
    sess_grid = pd.DataFrame(
        {"session": np.arange(session_min, session_max + 1)})
    pred = ols.get_prediction(sess_grid).summary_frame()
    out = sess_grid.copy()
    out["fit_sec"] = np.exp(pred["mean"])            # back-transform
    out["lo_sec"] = np.exp(pred["mean_ci_lower"])   # 95% CI
    out["hi_sec"] = np.exp(pred["mean_ci_upper"])   # 95% CI
    return out



def fit_and_plot_capture_rate_and_duration(df_trials: pd.DataFrame,
                                           df_sessions: pd.DataFrame,
                                           *,
                                           session_col: str = "session",
                                           value_col: str = "duration_sec",
                                           count_col: str = "captures",
                                           exposure_col: str = "total_duration",
                                           rate_per: Literal["second",
                                                             "minute"] = "minute",
                                           title_prefix: str = "",
                                           make_plots: bool = True):
    """Fit BOTH models (Poisson w/ offset; OLS on logT) using your cleaned
    inputs, optionally render both plots, and return the results.

    Returns
    -------
    dict with keys:
      - "po": fitted GLM Poisson results (captures ~ session, offset=log(exposure))
      - "ols": fitted OLS results (logT ~ session, cluster-robust by session)
      - "po_curve": DataFrame of predicted rate curve (session, fit, lo, hi)
      - "ols_curve": DataFrame of predicted duration curve (session, fit_sec, lo_sec, hi_sec)
    """
    # Fit models
    po = fit_poisson_by_session(df_sessions, session_col=session_col,
                                count_col=count_col, exposure_col=exposure_col)
    ols = fit_ols_logT_by_session(df_trials, session_col=session_col,
                                  value_col=value_col)

    # Predictions for convenience
    smin, smax = int(df_sessions[session_col].min()), int(
        df_sessions[session_col].max())
    po_curve = predict_poisson_rate_curve(po, smin, smax, per=rate_per)
    ols_curve = predict_duration_curve_seconds(ols, smin, smax)

    # Plots (optional)
    if make_plots:
        plot_poisson_rate_fit(df_sessions, po,
                              session_col=session_col,
                              count_col=count_col,
                              exposure_col=exposure_col,
                              per=rate_per,
                              title_prefix=title_prefix,
                              title=f"Reward Throughput (Captures per {'Minute' if rate_per=='minute' else 'Second'})")
        plot_duration_fit(df_trials, ols,
                          session_col=session_col,
                          value_col=value_col,
                          title_prefix=title_prefix,
                          title="Pursuit Duration (Geometric Mean)")

    return {"po": po, "ols": ols, "po_curve": po_curve, "ols_curve": ols_curve}




def fit_both_models(df_trials: pd.DataFrame,
                    df_sessions: pd.DataFrame,
                    *,
                    session_col: str = "session",
                    value_col: str = "duration_sec",
                    count_col: str = "captures",
                    exposure_col: str = "total_duration"):
    """Fit BOTH models and return them (no plotting)."""
    po = fit_poisson_by_session(df_sessions, session_col=session_col,
                                count_col=count_col, exposure_col=exposure_col)
    ols = fit_ols_logT_by_session(df_trials, session_col=session_col,
                                  value_col=value_col)
    return {"po": po, "ols": ols}


def extract_estimates_from_poisson_fit(po):
    """
    Inputs:
        po: statsmodels GLM (Poisson) on log(rate) with a 'session' regressor
    Outputs:
        DataFrame with percent change per session and over 10 sessions,
        plus a 95% CI for the 10-session percent change.
    """
    beta = float(po.params["session"])
    ci_lo, ci_hi = po.conf_int().loc["session"]

    pct_per_session = (np.exp(beta) - 1) * 100
    pct_per_10 = (np.exp(10 * beta) - 1) * 100
    ci10 = (
        (np.exp(10 * ci_lo) - 1) * 100,
        (np.exp(10 * ci_hi) - 1) * 100
    )

    po_dict = {
        "model": "Poisson (captures/min)",
        "%change_per_session":  f"{pct_per_session:.2f}",
        "%change_per_10_sessions": f"{pct_per_10:.1f}",
        "95%_CI_per_10_sessions": (round(ci10[0], 2), round(ci10[1], 2)),
        "p_value": round(float(po.pvalues["session"]), 4),
        "scale": "percent change in rate"
    }
    return pd.DataFrame([po_dict])


def extract_estimates_from_ols_fit(ols):
    """
    Inputs:
        ols: statsmodels OLS on log(duration) with a 'session' regressor
    Outputs:
        DataFrame with percent change per session and over 10 sessions,
        plus a 95% CI for the 10-session percent change.
    """
    beta = float(ols.params["session"])
    ci_lo, ci_hi = ols.conf_int().loc["session"]

    pct_per_session = (np.exp(beta) - 1) * 100
    pct_per_10 = (np.exp(10 * beta) - 1) * 100
    ci10 = (
        (np.exp(10 * ci_lo) - 1) * 100,
        (np.exp(10 * ci_hi) - 1) * 100
    )

    ols_dict = {
        "model": "OLS (log-duration)",
        "%change_per_session": f"{pct_per_session:.2f}",
        "%change_per_10_sessions": f"{pct_per_10:.1f}",
        "95%_CI_per_10_sessions": (round(ci10[0], 2), round(ci10[1], 2)),
        "p_value": round(float(ols.pvalues["session"]), 4),
        "scale": "percent change in duration"
    }
    return pd.DataFrame([ols_dict])



from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_duration_fit(
    df_trials: pd.DataFrame,
    ols,  # single model OR dict {monkey_name: ols_model}
    *,
    session_col: str = 'session',
    value_col: str = 'duration_sec',
    title: str = 'Pursuit Duration (Geometric Mean)',
    title_prefix: str = '',
    monkey_col: str = 'monkey',
    monkeys: Tuple[str, str] = ('Bruno', 'Schro'),
):
    """
    If df_trials has a `monkey` column and `ols` is a dict keyed by monkey,
    this will draw side-by-side panels for the requested `monkeys`.
    Otherwise, behaves like the original single-plot version.
    """

    def _prepare_per_session(_df: pd.DataFrame):
        tmp = _df[[session_col, value_col]].copy()
        tmp['logT'] = np.log(tmp[value_col].astype(float))
        per_sess = tmp.groupby(session_col, as_index=False).agg(mean_log=('logT', 'mean'))
        per_sess['geom_sec'] = np.exp(per_sess['mean_log'])
        return per_sess

    def _fit_curve(_ols, _df: pd.DataFrame):
        sess_min, sess_max = int(_df[session_col].min()), int(_df[session_col].max())
        fit_df = predict_duration_curve_seconds(_ols, sess_min, sess_max)
        p_value = getattr(_ols, 'pvalues', {}).get('session', 1.0) if hasattr(_ols, 'pvalues') else 1.0
        return fit_df, p_value

    def _plot_on_ax(ax, per_sess: pd.DataFrame, fit_df: pd.DataFrame,
                    panel_title: str, p_value: float,
                    *, markersize: int, font_axis: int, font_tick: int,
                    font_title: int, font_legend: int):
        ax.scatter(per_sess[session_col], per_sess['geom_sec'],
                   alpha=0.8, s=markersize**2,
                   label='Geometric mean (per session)')
        ax.plot(fit_df['session'], fit_df['fit_sec'], lw=2, label='OLS fit on log-duration')
        ax.fill_between(fit_df['session'], fit_df['lo_sec'], fit_df['hi_sec'],
                        alpha=0.2, label='95% CI')

        ax.set_xlabel('Session', fontsize=font_axis)
        ax.set_title(panel_title, fontsize=font_title, y=1.02)
        ax.tick_params(axis='both', labelsize=font_tick)

        analyze_proportion_trend.add_pval_to_plot(ax, p_value)

        y_min = min(fit_df['lo_sec'].min(), per_sess['geom_sec'].min())
        y_max = max(fit_df['hi_sec'].max(), per_sess['geom_sec'].max())
        ax.set_ylim(y_min * 0.9, y_max * 1.2)

    # --- Single-monkey mode -----------------------------------------------------
    if monkey_col not in df_trials.columns or not isinstance(ols, dict):
        per_sess = _prepare_per_session(df_trials)
        fit_df, p_value = _fit_curve(ols, df_trials)

        plt.figure(figsize=(5.6, 4.2), dpi=300)
        ax = plt.gca()
        _plot_on_ax(ax, per_sess, fit_df,
                    f'{title_prefix}{title}' if title_prefix else title,
                    p_value,
                    markersize=7, font_axis=13, font_tick=12,
                    font_title=15, font_legend=12)
        ax.set_ylabel('Typical pursuit duration (s)', fontsize=13)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        return

    # --- Side-by-side mode ------------------------------------------------------
    df2 = df_trials[df_trials[monkey_col].isin(monkeys)].copy()
    panels = []
    for m in monkeys:
        d_m = df2[df2[monkey_col] == m]
        if d_m.empty or m not in ols:
            continue
        per_sess_m = _prepare_per_session(d_m)
        fit_m, p_m = _fit_curve(ols[m], d_m)
        panels.append((m, per_sess_m, fit_m, p_m))

    if not panels:
        raise ValueError('No matching monkeys/models found to plot.')

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.4), dpi=300, sharey=True)
    if n == 1:
        axes = [axes]

    # Font & marker sizes for side-by-side mode
    marker_size = 6
    font_tick = 12
    font_axis = 13
    font_title = 15
    font_legend = 12

    # Global y-limits
    global_min = np.inf
    global_max = -np.inf
    for _, per_sess, fit_df, _ in panels:
        global_min = min(global_min, fit_df['lo_sec'].min(), per_sess['geom_sec'].min())
        global_max = max(global_max, fit_df['hi_sec'].max(), per_sess['geom_sec'].max())

    for i, (ax, (m, per_sess, fit_df, p_value)) in enumerate(zip(axes, panels)):
        panel_title = f'{m}: {title_prefix}{title}' if title_prefix else f'{m}: {title}'
        _plot_on_ax(ax, per_sess, fit_df, panel_title, p_value,
                    markersize=marker_size,
                    font_axis=font_axis, font_tick=font_tick,
                    font_title=font_title, font_legend=font_legend)

        if i == 0:
            ax.set_ylabel('Typical pursuit duration (s)', fontsize=font_axis)
        else:
            ax.set_ylabel('')  # remove duplicate label

        ax.set_ylim(global_min * 0.9, global_max * 1.2)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.legend(fontsize=font_legend)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)  # extra breathing room
    plt.show()




from typing import Optional, Iterable, Literal, Union, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Iterable, Literal, Union, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Iterable, Literal, Union, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Iterable, Literal, Union, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_poisson_rate_fit(
    df_sessions: pd.DataFrame,
    po,  # single model OR dict {monkey} OR nested {monkey: {condition}}
    *,
    session_col: str = 'session',
    count_col: str = 'captures',
    exposure_col: str = 'total_duration',
    per: Literal['second', 'minute'] = 'minute',
    title: str = 'Reward Rate (Captures per Minute)',
    title_prefix: str = '',
    pval: Optional[float] = None,
    monkey_col: str = 'monkey',
    monkeys: tuple = ('Bruno', 'Schro'),
    # overlay options (safe defaults; old calls unaffected)
    condition_col: Optional[str] = 'condition',
    conditions: Optional[tuple] = None,
    jitter: float = 0.08,
):
    def _prepare_obs(_df: pd.DataFrame):
        E = _df[exposure_col].astype(float).to_numpy()
        k = _df[count_col].astype(float).to_numpy()
        scale = 60.0 if per == 'minute' else 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = (k / E) * scale
            se = np.sqrt(np.clip(k, a_min=0, a_max=None)) / np.clip(E, a_min=1e-12, a_max=None) * scale
            lo = np.clip(rate - 1.96 * se, a_min=0, a_max=None)
            hi = np.clip(rate + 1.96 * se, a_min=0, a_max=None)
        obs = _df[[session_col]].copy()
        obs['rate'], obs['p_lo'], obs['p_hi'] = rate, lo, hi
        return obs

    def _fit_curve(_po, _df: pd.DataFrame):
        sess_min, sess_max = int(_df[session_col].min()), int(_df[session_col].max())
        fit_df = predict_poisson_rate_curve(_po, sess_min, sess_max, per=per)
        p_value = getattr(_po, 'pvalues', {}).get('session', 1.0) if hasattr(_po, 'pvalues') else 1.0
        return fit_df, p_value

    def _stacked_pvals(ax, texts, loc='upper left', line_h=0.085, fontsize=12):
        # texts: list of strings (already formatted, includes condition name)
        # place as stacked lines inside axes coords
        x0, y0, ha, va = {
            'upper left':  (0.02, 0.98, 'left',  'top'),
            'upper right': (0.98, 0.98, 'right', 'top'),
            'lower left':  (0.02, 0.02, 'left',  'bottom'),
            'lower right': (0.98, 0.02, 'right', 'bottom'),
        }[loc]
        for i, t in enumerate(texts):
            dy = i * line_h if 'lower' in loc else -i * line_h
            ax.text(x0, y0 + dy, t, transform=ax.transAxes, ha=ha, va=va,
                    fontsize=fontsize, 
                    #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )

    # --------- ORIGINAL BEHAVIOR PATH (no multi-monkey dict) ----------
    if (monkey_col not in df_sessions.columns) or (not isinstance(po, dict)):
        obs = _prepare_obs(df_sessions)
        fit_df, p_value = _fit_curve(po, df_sessions)
        plt.figure(figsize=(5.6, 4.2), dpi=300)
        ax = plt.gca()

        ax.errorbar(
            obs[session_col], obs['rate'],
            yerr=[obs['rate'] - obs['p_lo'], obs['p_hi'] - obs['rate']],
            fmt='o', alpha=0.7, label=f'Observed ({per}) ±95% CI'
        )
        ax.plot(fit_df['session'], fit_df['fit'], lw=2, label=f'Poisson fit ({per})')
        ax.fill_between(fit_df['session'], fit_df['p_lo'], fit_df['p_hi'], alpha=0.2, label='95% CI')
        ax.set_xlabel('Session', fontsize=12)
        ax.set_ylabel(f'Captures per {per}', fontsize=12)
        analyze_proportion_trend.add_pval_to_plot(ax, p_value)
        ax.set_title(f'{title_prefix}{title}' if title_prefix else title, fontsize=14, y=1.02)

        y_min = min(obs['p_lo'].min(), fit_df['p_lo'].min())
        y_max = max(obs['p_hi'].max(), fit_df['p_hi'].max())
        ax.set_ylim(y_min * 0.9, y_max * 1.2)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    # --------- SIDE-BY-SIDE MONKEY PANELS ----------
    has_condition = bool(condition_col) and (condition_col in df_sessions.columns)
    df2 = df_sessions[df_sessions[monkey_col].isin(monkeys)].copy()

    panels = []
    for m in monkeys:
        d_m = df2[df2[monkey_col] == m]
        if d_m.empty or m not in po:
            continue
        if not has_condition:
            obs_m = _prepare_obs(d_m)
            fit_m, p_m = _fit_curve(po[m], d_m)
            panels.append((m, [(None, obs_m, fit_m, p_m)]))
        else:
            if conditions is None:
                cond_vals = tuple(sorted(d_m[condition_col].dropna().unique()))
            else:
                cond_vals = tuple(conditions)
            nested = isinstance(po[m], dict)
            rows = []
            for c in cond_vals:
                d_mc = d_m[d_m[condition_col] == c]
                if d_mc.empty:
                    continue
                model_mc = po[m][c] if nested and (c in po[m]) else po[m]
                obs_mc = _prepare_obs(d_mc)
                fit_mc, p_mc = _fit_curve(model_mc, d_mc)
                rows.append((c, obs_mc, fit_mc, p_mc))
            if rows:
                panels.append((m, rows))

    if not panels:
        raise ValueError('No matching monkeys/models found to plot.')

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.4), dpi=300, sharey=True)
    if n == 1:
        axes = [axes]

    # Global y-limits
    gmin, gmax = np.inf, -np.inf
    for _, rows in panels:
        for _, obs_m, fit_m, _ in rows:
            gmin = min(gmin, obs_m['p_lo'].min(), fit_m['p_lo'].min())
            gmax = max(gmax, obs_m['p_hi'].max(), fit_m['p_hi'].max())

    # Fonts
    marker_size = 6
    font_tick = 12
    font_axis = 13
    font_title = 15
    font_legend = 12

    # Build a consistent color map per condition (overlay mode)
    cond_colors: Dict[str, str] = {}
    if has_condition:
        # use the global prop cycle for stable, clean colors
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3'])
        cond_order = []
        for _, rows in panels:
            for c, *_ in rows:
                if c is not None and c not in cond_order:
                    cond_order.append(c)
        for i, c in enumerate(cond_order):
            cond_colors[c] = prop_cycle[i % len(prop_cycle)]

    # Plot panels
    for i, (ax, (m, rows)) in enumerate(zip(axes, panels)):
        panel_title = f'{m}: {title_prefix}{title}' if title_prefix else f'{m}: {title}'

        if not has_condition:
            (c_none, obs_m, fit_m, p_m) = rows[0]
            ax.errorbar(
                obs_m[session_col], obs_m['rate'],
                yerr=[obs_m['rate'] - obs_m['p_lo'], obs_m['p_hi'] - obs_m['rate']],
                fmt='o', alpha=0.7, markersize=marker_size,
                label=f'Observed ({per}) ±95% CI'
            )
            ax.plot(fit_m['session'], fit_m['fit'], lw=2, label=f'Poisson fit ({per})')
            ax.fill_between(fit_m['session'], fit_m['p_lo'], fit_m['p_hi'], alpha=0.2, label='95% CI')

            ax.set_xlabel('Session', fontsize=font_axis)
            if i == 0:
                ax.set_ylabel(f'Captures per {per}', fontsize=font_axis)
            else:
                ax.set_ylabel('')
            ax.legend(fontsize=font_legend)
            analyze_proportion_trend.add_pval_to_plot(ax, p_m)
            ax.set_title(panel_title, fontsize=font_title, y=1.02)
            ax.tick_params(axis='both', labelsize=font_tick)
            ax.set_ylim(gmin * 0.9, gmax * 1.32)
            ax.yaxis.set_tick_params(labelleft=True)

        else:
            # overlay with matched colors & jitter; build a unified legend WITH p-values
            k = len(rows)
            offsets = np.linspace(-jitter, jitter, num=k) if k > 1 else np.zeros(k)

            panel_handles = []
            panel_labels = []

            for off, (c, obs_m, fit_m, p_m) in zip(offsets, rows):
                color = cond_colors.get(c, None)

                # observed with error bars
                ax.errorbar(
                    obs_m[session_col].to_numpy().astype(float) + off,
                    obs_m['rate'],
                    yerr=[obs_m['rate'] - obs_m['p_lo'], obs_m['p_hi'] - obs_m['rate']],
                    fmt='o', alpha=0.7, markersize=marker_size,
                    color=color
                )
                # fit line
                ax.plot(fit_m['session'], fit_m['fit'], lw=2, color=color)
                # CI band with same color
                ax.fill_between(fit_m['session'], fit_m['p_lo'], fit_m['p_hi'], alpha=0.2, color=color)

                # legend handle for this condition
                handle = Line2D([0], [0], color=color, marker='o', lw=2)
                # format inline p-value with stars
                if p_m is None or np.isnan(p_m):
                    p_txt = 'p = n/a'
                elif p_m < 0.001:
                    p_txt = 'p < 0.001 ***'
                else:
                    p_txt = f'p = {float(p_m):.3f}'.rstrip('0').rstrip('.')
                    if p_m < 0.01:
                        p_txt += ' **'
                    elif p_m < 0.05:
                        p_txt += ' *'
                label = f"{str(c).replace('_', ' ')}: {p_txt}"
                panel_handles.append(handle)
                panel_labels.append(label)

            # cosmetics (match original fonts/sizes)
            ax.set_xlabel('Session', fontsize=font_axis)
            if i == 0:
                ax.set_ylabel(f'Captures per {per}', fontsize=font_axis)
            else:
                ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize=font_tick)
            ax.set_ylim(gmin * 0.9, gmax * 1.32)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_title(panel_title, fontsize=font_title, y=1.02)

            # panel-local legend with combined condition + p-value lines
            ax.legend(panel_handles, panel_labels, loc='upper right', fontsize=font_legend, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)

    # Legend handling
    # (no global legend when conditions are present, since each panel shows its own
    # combined condition+p legend; keep nothing else here)
    plt.show()
