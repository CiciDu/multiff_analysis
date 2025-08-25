
from planning_analysis.factors_vs_indicators import make_variations_utils, plot_variations_utils, process_variations_utils
from data_wrangling import specific_utils, process_monkey_information, base_processing_class, combine_info_utils, further_processing_class



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
        .agg(logT_mean=("logT","mean"), n=("logT","size"))
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
def plot_poisson_rate_fit(df_sessions, po):
    sess_grid = pd.DataFrame({"session": np.arange(df_sessions["session"].min(),
                                                   df_sessions["session"].max()+1)})
    # Predict a RATE by setting offset = log(60 sec) ⇒ rate per minute
    pred = po.get_prediction(
        exog=sess_grid.assign(**{}),
        offset=np.log(np.full(len(sess_grid), 60.0))
    ).summary_frame()  # columns: mean, mean_ci_lower, mean_ci_upper on link=log scale already exp'ed
    # NOTE: statsmodels GLM returns predictions on response scale by default for Poisson

    sess_grid["fit_rate_per_min"] = pred["mean"].values
    sess_grid["fit_lo"] = pred["mean_ci_lower"].values
    sess_grid["fit_hi"] = pred["mean_ci_upper"].values

    obs = poisson_rate_per_min(df_sessions)

    plt.figure(figsize=(7,4))
    # observed points + error bars
    plt.errorbar(obs["session"], obs["rate_per_min"], 
                 yerr=[obs["rate_per_min"]-obs["rate_lo"], obs["rate_hi"]-obs["rate_per_min"]],
                 fmt="o", alpha=0.7, label="Observed (rate ±95% CI)")
    # fitted curve + band
    plt.plot(sess_grid["session"], sess_grid["fit_rate_per_min"], lw=2, label="Poisson fit (per min)")
    plt.fill_between(sess_grid["session"], sess_grid["fit_lo"], sess_grid["fit_hi"], alpha=0.2, label="95% CI")
    plt.xlabel("Session")
    plt.ylabel("Captures per minute")
    plt.title("Reward throughput (captures/min) with Poisson GLM fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Plot 2: Duration with OLS(logT) fit ----------
def plot_duration_fit(df_trials, ols):
    per_sess = geom_mean_per_session(df_trials)
    sess_grid = pd.DataFrame({"session": np.arange(df_trials["session"].min(),
                                                   df_trials["session"].max()+1)})

    # Predict on log scale then exponentiate to plot on seconds
    pred = ols.get_prediction(sess_grid).summary_frame()  # mean, mean_ci_lower, ...
    sess_grid["fit_T"] = np.exp(pred["mean"].values)
    sess_grid["fit_lo"] = np.exp(pred["mean_ci_lower"].values)
    sess_grid["fit_hi"] = np.exp(pred["mean_ci_upper"].values)

    plt.figure(figsize=(7,4))
    plt.scatter(per_sess["session"], per_sess["geom_mean_T"], alpha=0.8, label="Geometric mean (per session)")
    plt.plot(sess_grid["session"], sess_grid["fit_T"], lw=2, label="OLS fit on log-duration")
    plt.fill_between(sess_grid["session"], sess_grid["fit_lo"], sess_grid["fit_hi"], alpha=0.2, label="95% CI")
    plt.xlabel("Session")
    plt.ylabel("Typical pursuit duration (s)")
    plt.title("Pursuit duration with log-linear fit")
    plt.legend()
    plt.tight_layout()
    plt.show()








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# ---------- shared helpers ----------
def _early_late_cuts_from_sessions(df_sessions, session_col="session"):
    sessions = np.sort(df_sessions[session_col].unique())
    n = len(sessions)
    if n < 3:
        raise ValueError(f"Need ≥3 sessions to define tertiles, got n={n}.")
    early_idx = max(0, int(np.floor(n/3)) - 1)
    late_idx  = min(n-1, int(np.ceil(2*n/3)) - 1)
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
    t, p = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")  # late vs early
    mean_a, mean_b = np.nanmean(a), np.nanmean(b)
    diff = mean_b - mean_a
    ratio = np.nan if mean_a == 0 else (mean_b / mean_a)
    pct = np.nan if not np.isfinite(ratio) else (ratio - 1) * 100.0
    return {"diff": diff, "ratio": ratio, "percent_change": pct, "t": t, "pval": p}

# ---------- RATE: descriptive + GLM Poisson with offset(time) ----------
def summarize_early_late_rate_with_glm(df_sessions, session_col="session",
                                       time_col="total_duration", capture_col="captures",
                                       plot=True, title="Early vs Late: Reward rate"):
    """
    Outputs:
      phase_tbl            : early/late means ± 95% CI for captures/min
      ttest_contrast_tbl   : Welch t-test late vs early on session rates
      glm_contrast_tbl     : Poisson GLM late vs early (rate ratio, CI, p)
      effect_summary_tbl   : one-row summary combining descriptive ratio + GLM RR
    """
    early_cut, late_cut = _early_late_cuts_from_sessions(df_sessions, session_col=session_col)

    # Session rates (assumes your helper returns ['session','rate_per_min', ...])
    rates = poisson_rate_per_min(df_sessions)
    rates["phase"] = _phase_from_cuts(rates[session_col], early_cut, late_cut)

    # Descriptive phase table
    phase_tbl = _agg_with_ci(rates, value_col="rate_per_min", phase_col="phase")
    phase_tbl = phase_tbl.rename(columns={
        "mean": "rate_per_min_mean",
        "lo":   "rate_per_min_lo",
        "hi":   "rate_per_min_hi"
    })

    # Welch t-test on session rates
    early_vals = rates.loc[rates["phase"] == "early", "rate_per_min"].to_numpy()
    late_vals  = rates.loc[rates["phase"] == "late",  "rate_per_min"].to_numpy()
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
    df_phase["phase"] = _phase_from_cuts(df_phase[session_col], early_cut, late_cut)
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
        plt.figure(figsize=(6,4))
        x = np.arange(2)
        y   = phase_tbl.set_index("phase").loc[["early","late"], "rate_per_min_mean"].values
        ylo = phase_tbl.set_index("phase").loc[["early","late"], "rate_per_min_lo"].values
        yhi = phase_tbl.set_index("phase").loc[["early","late"], "rate_per_min_hi"].values
        yerr = np.vstack([y - ylo, yhi - y])
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
        plt.xticks(x, ["Early", "Late"])
        plt.ylabel("Captures per minute")
        plt.title(title)
        plt.tight_layout()
        plt.show()

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
    early_cut, late_cut = _early_late_cuts_from_sessions(df_sessions, session_col=session_col)

    # Per-session geometric mean durations (assumes helper returns ['session','geom_mean_T', ...])
    per_sess = geom_mean_per_session(df_trials)
    per_sess["phase"] = _phase_from_cuts(per_sess[session_col], early_cut, late_cut)

    # Descriptive phase table (no zero-floor because durations are >0 and we work on seconds)
    phase_tbl = _agg_with_ci(per_sess, value_col="geom_mean_T", phase_col="phase", zero_floor=False)
    phase_tbl = phase_tbl.rename(columns={
        "mean": "geomT_mean",
        "lo":   "geomT_lo",
        "hi":   "geomT_hi"
    })

    # Welch t-test on per-session geometric means
    early_vals = per_sess.loc[per_sess["phase"] == "early", "geom_mean_T"].to_numpy()
    late_vals  = per_sess.loc[per_sess["phase"] == "late",  "geom_mean_T"].to_numpy()
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
    trials["phase"] = _phase_from_cuts(trials[session_col], early_cut, late_cut)
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
        plt.figure(figsize=(6,4))
        x = np.arange(2)
        y   = phase_tbl.set_index("phase").loc[["early","late"], "geomT_mean"].values
        ylo = phase_tbl.set_index("phase").loc[["early","late"], "geomT_lo"].values
        yhi = phase_tbl.set_index("phase").loc[["early","late"], "geomT_hi"].values
        yerr = np.vstack([y - ylo, yhi - y])
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
        plt.xticks(x, ["Early", "Late"])
        plt.ylabel("Typical pursuit duration (s)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return phase_tbl, ttest_contrast_tbl, glm_contrast_tbl, effect_summary_tbl

# ---------- Wrapper ----------
def plot_early_late_contrasts(df_sessions, df_trials):
    """
    Runs both metrics with plots, and returns:
      rate_phase_tbl, rate_ttest_tbl, rate_glm_tbl, rate_effect_summary_tbl,
      dur_phase_tbl,  dur_ttest_tbl,  dur_glm_tbl,  dur_effect_summary_tbl
    """
    (rate_phase_tbl, rate_ttest_tbl, rate_glm_tbl, rate_effect_summary_tbl
     ) = summarize_early_late_rate_with_glm(df_sessions)

    (dur_phase_tbl, dur_ttest_tbl, dur_glm_tbl, dur_effect_summary_tbl
     ) = summarize_early_late_duration_with_glm(df_trials, df_sessions)

    return (rate_phase_tbl, rate_ttest_tbl, rate_glm_tbl, rate_effect_summary_tbl,
            dur_phase_tbl,  dur_ttest_tbl,  dur_glm_tbl,  dur_effect_summary_tbl)




def get_key_data(raw_data_dir_name='all_monkey_data/raw_monkey_data', monkey_name = 'monkey_Bruno'):
    
    sessions_df_for_one_monkey = combine_info_utils.make_sessions_df_for_one_monkey(
        raw_data_dir_name, monkey_name)

    all_trial_durations_df = pd.DataFrame()
    all_stop_df = pd.DataFrame()
    all_VBLO_df = pd.DataFrame()

    for index, row in sessions_df_for_one_monkey.iterrows():
        if row['finished'] is True:
            continue

        data_name = row['data_name']
        raw_data_folder_path = os.path.join(
            raw_data_dir_name, row['monkey_name'], data_name)
        print(raw_data_folder_path)
        data_item = further_processing_class.FurtherProcessing(
            raw_data_folder_path=raw_data_folder_path)
        
        # disable printing
        data_item.retrieve_or_make_monkey_data()
        data_item.make_or_retrieve_ff_dataframe()
        
        trial_durations = np.diff(data_item.ff_caught_T_new)
        trial_durations_df = pd.DataFrame(
            {'duration_sec': trial_durations, 'trial_index': np.arange(len(trial_durations))})
        trial_durations_df['data_name'] = data_name
        all_trial_durations_df = pd.concat(
            [all_trial_durations_df, trial_durations_df])
        
        num_stops = data_item.monkey_information.loc[data_item.monkey_information['whether_new_distinct_stop'] == True, ['time']].shape[0]     
        num_captures = len(data_item.ff_caught_T_new)
        stop_df = pd.DataFrame(
            {
                'stops': [num_stops],
                'captures': [num_captures],
                'data_name': [data_name],
            }
        )
        all_stop_df = pd.concat([all_stop_df, stop_df])
        
        data_item.get_visible_before_last_one_trials_info()
        num_VBLO_trials = len(data_item.vblo_target_cluster_df)
        all_selected_base_trials = len(data_item.selected_base_trials)
        VBLO_df = pd.DataFrame(
            {
                'VBLO_trials': [num_VBLO_trials],
                'base_trials': [all_selected_base_trials],
                'data_name': [data_name],
            }
        )
        all_VBLO_df = pd.concat([all_VBLO_df, VBLO_df])

        
    all_trial_durations_df = make_variations_utils.assign_session_id(all_trial_durations_df, 'session')
    all_stop_df = make_variations_utils.assign_session_id(all_stop_df, 'session')
    all_VBLO_df = make_variations_utils.assign_session_id(all_VBLO_df, 'session')

    return all_trial_durations_df, all_stop_df, all_VBLO_df
    
