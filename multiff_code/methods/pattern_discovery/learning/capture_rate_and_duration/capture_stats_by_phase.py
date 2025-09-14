
from scipy import stats
from planning_analysis.factors_vs_indicators import make_variations_utils, process_variations_utils
from planning_analysis.factors_vs_indicators.plot_plan_indicators import plot_variations_class, plot_variations_utils
from data_wrangling import specific_utils, process_monkey_information, base_processing_class, combine_info_utils, further_processing_class

from pattern_discovery.learning.proportion_trend import analyze_proportion_trend


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os


# ---------- Helpers ----------
def geom_mean_per_session(df_trials):
    out = (
        df_trials
        .groupby("session", as_index=False)
        .agg(logT_mean=("logT", "mean"), n=("logT", "size"))
    )
    out["geom_mean_T"] = np.exp(out["logT_mean"])
    return out


def poisson_rate_per_min(df_sessions):
    rate = df_sessions["captures"] / (df_sessions["total_duration"]/60.0)
    # 95% CI using Poisson approx for counts with exposure E = total_duration/60
    E = (df_sessions["total_duration"]/60.0).to_numpy()
    lam_hat = rate.to_numpy()       # captures per minute
    se = np.sqrt(df_sessions["captures"].to_numpy()) / E
    lo = lam_hat - 1.96*se
    hi = lam_hat + 1.96*se
    tmp = df_sessions[["session"]].copy()
    tmp["rate_per_min"] = lam_hat
    tmp["rate_lo"] = np.clip(lo, a_min=0, a_max=None)
    tmp["rate_hi"] = np.clip(hi, a_min=0, a_max=None)
    return tmp

# ---------- Plot 1: Captures per minute with Poisson fit ----------

# ---------- shared helpers ----------


def _early_late_cuts_from_sessions(df_sessions, session_col="session"):
    sessions = np.sort(df_sessions[session_col].unique())
    n = len(sessions)
    if n < 3:
        raise ValueError(f"Need ≥3 sessions to define tertiles, got n={n}.")
    early_idx = max(0, int(np.floor(n/3)) - 1)
    late_idx = min(n-1, int(np.ceil(2*n/3)) - 1)
    return sessions[early_idx], sessions[late_idx]


def _phase_from_cuts(series_session, early_cut, late_cut):
    return np.where(series_session <= early_cut, "early",
                    np.where(series_session >= late_cut, "late", "mid"))


def _agg_with_ci(df, value_col, phase_col="phase", zero_floor=True):
    sub = df[df[phase_col].isin(["early", "late"])].copy()
    out = (sub.groupby(phase_col, as_index=False)
           .agg(n=(value_col, "size"),
                mean=(value_col, "mean"),
                se=(value_col, lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
    lo = out["mean"] - 1.96*out["se"]
    hi = out["mean"] + 1.96*out["se"]
    out["lo"] = np.clip(lo, 0, None) if zero_floor else lo
    out["hi"] = hi
    out = out.set_index(phase_col).loc[["early", "late"]].reset_index()
    return out


def _welch_t_and_effect(a, b):
    t, p = stats.ttest_ind(b, a, equal_var=False,
                           nan_policy="omit")  # late vs early
    mean_a, mean_b = np.nanmean(a), np.nanmean(b)
    diff = mean_b - mean_a
    ratio = np.nan if mean_a == 0 else (mean_b / mean_a)
    pct = np.nan if not np.isfinite(ratio) else (ratio - 1) * 100.0
    return {"diff": diff, "ratio": ratio, "percent_change": pct, "t": t, "pval": p}

# ---------- RATE: descriptive + GLM Poisson with offset(time) ----------


def summarize_early_late_event_rate_with_glm(df_sessions, session_col="session",
                                             time_col="total_duration", capture_col="captures",
                                             plot=True, title="Early vs Late: Reward rate"):
    """
    Outputs:
      phase_tbl            : early/late means ± 95% CI for captures/min
      ttest_contrast_tbl   : Welch t-test late vs early on session rates
      glm_contrast_tbl     : Poisson GLM late vs early (rate ratio, CI, p)
      effect_summary_tbl   : one-row summary combining descriptive ratio + GLM RR
    """
    early_cut, late_cut = _early_late_cuts_from_sessions(
        df_sessions, session_col=session_col)

    # Session rates (assumes your helper returns ['session','rate_per_min', ...])
    rates = poisson_rate_per_min(df_sessions)
    rates["phase"] = _phase_from_cuts(rates[session_col], early_cut, late_cut)

    # Descriptive phase table
    phase_tbl = _agg_with_ci(
        rates, value_col="rate_per_min", phase_col="phase")
    phase_tbl = phase_tbl.rename(columns={
        "mean": "rate_per_min_mean",
        "lo":   "rate_per_min_lo",
        "hi":   "rate_per_min_hi"
    })

    # Welch t-test on session rates
    early_vals = rates.loc[rates["phase"] ==
                           "early", "rate_per_min"].to_numpy()
    late_vals = rates.loc[rates["phase"] == "late",  "rate_per_min"].to_numpy()
    tstats = _welch_t_and_effect(early_vals, late_vals)
    ttest_contrast_tbl = pd.DataFrame([{
        "contrast": "late_vs_early",
        "diff_rate_per_min": tstats["diff"],
        "rate_ratio_late_over_early": tstats["ratio"],
        "percent_change_late_vs_early": tstats["percent_change"],
        "t_stat": tstats["t"],
        "pval": tstats["pval"]
    }])

    # GLM Poisson with offset(time) on early vs late
    df_phase = df_sessions.copy()
    df_phase["phase"] = _phase_from_cuts(
        df_phase[session_col], early_cut, late_cut)
    sub = df_phase[df_phase["phase"].isin(["early", "late"])].copy()

    glm_phase = smf.glm(
        f"{capture_col} ~ C(phase)",
        data=sub,
        family=sm.families.Poisson(),
        offset=np.log(sub[time_col])
    ).fit(cov_type="HC0")

    coef = glm_phase.params["C(phase)[T.late]"]
    RR = float(np.exp(coef))
    ci_low, ci_high = glm_phase.conf_int().loc["C(phase)[T.late]"].tolist()
    RR_lo, RR_hi = float(np.exp(ci_low)), float(np.exp(ci_high))
    pval = float(glm_phase.pvalues["C(phase)[T.late]"])

    glm_contrast_tbl = pd.DataFrame([{
        "contrast": "late_vs_early",
        "rate_ratio_GLMPoisson": RR,
        "RR_95CI_low": RR_lo,
        "RR_95CI_high": RR_hi,
        "pval": pval
    }])

    # Compact summary row combining both viewpoints
    effect_summary_tbl = pd.DataFrame([{
        "metric": "captures_per_min",
        "descriptive_ratio_late_over_early": tstats["ratio"],
        "descriptive_percent_change": tstats["percent_change"],
        "GLM_rate_ratio": RR,
        "GLM_95CI": f"[{RR_lo:.3f}, {RR_hi:.3f}]",
        "GLM_pval": pval
    }])

    # Plot
    if plot:
        plot_early_late_with_ci(
            phase_tbl,
            mean_col='rate_per_min_mean',
            lo_col='rate_per_min_lo',
            hi_col='rate_per_min_hi',
            ylabel='Captures per minute',
            title=title
        )

    return phase_tbl, ttest_contrast_tbl, glm_contrast_tbl, effect_summary_tbl

# ---------- DURATION: descriptive + OLS on logT (clustered by session) ----------


def summarize_early_late_duration_with_glm(df_trials, df_sessions,
                                           session_col="session",
                                           logT_col="logT",
                                           plot=True, title="Early vs Late: Duration"):
    """
    Outputs:
      phase_tbl            : early/late means ± 95% CI for geometric-mean duration (seconds)
      ttest_contrast_tbl   : Welch t-test late vs early on per-session geometric means
      glm_contrast_tbl     : OLS(logT ~ C(phase)) cluster-robust by session (percent change, CI, p)
      effect_summary_tbl   : one-row summary combining descriptive ratio + OLS % change
    NOTE: Requires df_trials to contain per-trial logT and session.
    """
    early_cut, late_cut = _early_late_cuts_from_sessions(
        df_sessions, session_col=session_col)

    # Per-session geometric mean durations (assumes helper returns ['session','geom_mean_T', ...])
    per_sess = geom_mean_per_session(df_trials)
    per_sess["phase"] = _phase_from_cuts(
        per_sess[session_col], early_cut, late_cut)

    # Descriptive phase table (no zero-floor because durations are >0 and we work on seconds)
    phase_tbl = _agg_with_ci(
        per_sess, value_col="geom_mean_T", phase_col="phase", zero_floor=False)
    phase_tbl = phase_tbl.rename(columns={
        "mean": "geomT_mean",
        "lo":   "geomT_lo",
        "hi":   "geomT_hi"
    })

    # Welch t-test on per-session geometric means
    early_vals = per_sess.loc[per_sess["phase"]
                              == "early", "geom_mean_T"].to_numpy()
    late_vals = per_sess.loc[per_sess["phase"]
                             == "late",  "geom_mean_T"].to_numpy()
    tstats = _welch_t_and_effect(early_vals, late_vals)
    ttest_contrast_tbl = pd.DataFrame([{
        "contrast": "late_vs_early",
        "diff_seconds": tstats["diff"],
        "ratio_late_over_early": tstats["ratio"],
        "percent_change_late_vs_early": tstats["percent_change"],
        "t_stat": tstats["t"],
        "pval": tstats["pval"]
    }])

    # OLS on per-trial logT with cluster-robust SE by session
    trials = df_trials.copy()
    trials["phase"] = _phase_from_cuts(
        trials[session_col], early_cut, late_cut)
    sub2 = trials[trials["phase"].isin(["early", "late"])].copy()

    ols_phase = smf.ols(f"{logT_col} ~ C(phase)", data=sub2).fit(
        cov_type="cluster", cov_kwds={"groups": sub2[session_col]}
    )
    coef = float(ols_phase.params["C(phase)[T.late]"])
    ci_low, ci_high = ols_phase.conf_int().loc["C(phase)[T.late]"].tolist()
    pct = (np.exp(coef) - 1) * 100.0
    ci_pct = (np.exp(ci_low) - 1) * 100.0, (np.exp(ci_high) - 1) * 100.0
    pval = float(ols_phase.pvalues["C(phase)[T.late]"])

    glm_contrast_tbl = pd.DataFrame([{
        "contrast": "late_vs_early",
        "percent_change_duration_OLS": pct,
        "pct_95CI_low": ci_pct[0],
        "pct_95CI_high": ci_pct[1],
        "pval": pval
    }])

    effect_summary_tbl = pd.DataFrame([{
        "metric": "duration_seconds",
        "descriptive_ratio_late_over_early": tstats["ratio"],
        "descriptive_percent_change": tstats["percent_change"],
        "OLS_percent_change": pct,
        "OLS_95CI": f"[{ci_pct[0]:+.1f}%, {ci_pct[1]:+.1f}%]",
        "OLS_pval": pval
    }])

    # Plot
    if plot:
        plot_early_late_with_ci(
            phase_tbl,
            mean_col='geomT_mean',
            lo_col='geomT_lo',
            hi_col='geomT_hi',
            ylabel='Typical pursuit duration (s)',
            title=title
        )

    return phase_tbl, ttest_contrast_tbl, glm_contrast_tbl, effect_summary_tbl

# ---------- Wrapper ----------


def plot_early_late_with_ci(
    phase_tbl: pd.DataFrame,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    title: str,
    order=('early', 'late'),
    fmt='{:.2f}'
):
    """
    Bar plot with asymmetric CI error bars for Early vs Late phases.

    Parameters
    ----------
    phase_tbl : pd.DataFrame
        Must contain columns [phase, mean_col, lo_col, hi_col].
    mean_col, lo_col, hi_col : str
        Column names for the mean and confidence interval bounds.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    order : tuple of str
        Order of phases on the x-axis (default: ('early', 'late')).
    fmt : str
        Format string for value labels on top of bars.
    """
    sub = phase_tbl.set_index('phase').reindex(order)

    x = np.arange(len(order))
    y = sub[mean_col].to_numpy()
    lo = sub[lo_col].to_numpy()
    hi = sub[hi_col].to_numpy()

    yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])

    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    bars = ax.bar(
        x, y, width=0.6,
        yerr=yerr,
        error_kw={'capsize': 5, 'elinewidth': 1.2, 'alpha': 0.9},
        edgecolor='black', linewidth=0.8, alpha=0.9
    )

    ax.set_xticks(x, [ph.capitalize() for ph in order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            fmt.format(h),
            ha='center', va='bottom',
            fontsize=9, clip_on=False
        )

    fig.tight_layout()
    plt.show()


def analyze_early_late_capture_and_duration(df_sessions, df_trials):
    """
    Runs both metrics with plots, and returns:
      rate_phase_tbl, rate_ttest_tbl, rate_glm_tbl, rate_effect_summary_tbl,
      dur_phase_tbl,  dur_ttest_tbl,  dur_glm_tbl,  dur_effect_summary_tbl
    """
    (rate_phase_tbl, rate_ttest_tbl, rate_glm_tbl, rate_effect_summary_tbl
     ) = summarize_early_late_event_rate_with_glm(df_sessions)

    (dur_phase_tbl, dur_ttest_tbl, dur_glm_tbl, dur_effect_summary_tbl
     ) = summarize_early_late_duration_with_glm(df_trials, df_sessions)

    return (rate_phase_tbl, rate_ttest_tbl, rate_glm_tbl, rate_effect_summary_tbl,
            dur_phase_tbl,  dur_ttest_tbl,  dur_glm_tbl,  dur_effect_summary_tbl)
