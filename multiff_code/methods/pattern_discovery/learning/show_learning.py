
import numpy as np, pandas as pd

def get_stop_df(monkey_information, ff_caught_T_new):
    # 0) Build stop table (ensure sorted!)
    stop_df = (monkey_information.loc[monkey_information['whether_new_distinct_stop'] == True, ['time']]
        .rename(columns={'time': 'stop_time'})
        .sort_values('stop_time', ignore_index=True)
        )
    stop_df['captures'] = 0

    # 1) Build capture table (sorted)
    cap_df = pd.DataFrame({'capture_time': np.asarray(ff_caught_T_new, dtype=float)})
    cap_df = cap_df.sort_values('capture_time', ignore_index=True)
        
    # 2) Map each capture to the most recent stop at or before it
    #    (captures occurring before the first stop yield NaN and are dropped)
    mapped = pd.merge_asof(cap_df, stop_df[['stop_time']], 
                    left_on='capture_time', right_on='stop_time',
                    direction='backward')

    # 3) Mark those stops as captures
    hit_idx = mapped['stop_time'].dropna().map(
    {t: i for i, t in enumerate(stop_df['stop_time'])}
    ).dropna().astype(int)

    stop_df.loc[hit_idx.unique(), 'captures'] = 1
    
    return stop_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm

# ------------------
# Utilities
# ------------------
def wilson_ci(k, n, alpha=0.05):
    """Wilson score CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - alpha/2)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - half), min(1, center + half)

def session_prop_with_ci(df_sessions_counts):
    """df with columns session, captures, stops -> add p_hat and Wilson CI."""
    out = df_sessions_counts.copy()
    p_hat = out["captures"] / out["stops"].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n)) for k, n in zip(out["captures"], out["stops"])])
    out["p_hat"] = p_hat
    out["p_lo"] = ci[:,0]
    out["p_hi"] = ci[:,1]
    return out

# ------------------
# 1) Logistic GLM (stop-level) fit plot
# ------------------
def plot_logistic_stop_success_fit(all_stop_df_sessions, glm_logit):
    # Observed per-session success with Wilson CIs
    df_sessions = (
        all_stop_df_sessions.groupby("session", as_index=False)
        .agg(captures=("captures","sum"), stops=("captures","size"))
    )
    plot_df = session_prop_with_ci(df_sessions)

    # Model predictions across integer sessions
    sess_grid = pd.DataFrame({"session": np.arange(plot_df["session"].min(),
                                                   plot_df["session"].max()+1)})
    pred = glm_logit.get_prediction(sess_grid).summary_frame()  # response scale for Binomial
    sess_grid["fit_p"]  = pred["mean"].values
    sess_grid["fit_lo"] = pred["mean_ci_lower"].values
    sess_grid["fit_hi"] = pred["mean_ci_upper"].values

    plt.figure(figsize=(7,4))
    # observed points + CI
    yerr = np.vstack([
        plot_df["p_hat"] - plot_df["p_lo"],
        plot_df["p_hi"] - plot_df["p_hat"]
    ])
    plt.errorbar(plot_df["session"], plot_df["p_hat"], yerr=yerr, fmt="o", capsize=3,
                 alpha=0.8, label="Observed (per session, Wilson 95% CI)")
    # fitted curve + band
    plt.plot(sess_grid["session"], sess_grid["fit_p"], lw=2, label="Logistic GLM fit")
    plt.fill_between(sess_grid["session"], sess_grid["fit_lo"], sess_grid["fit_hi"],
                     alpha=0.2, label="95% CI")
    plt.ylim(0, 1)
    plt.xlabel("Session")
    plt.ylabel("P(capture | stop)")
    plt.title("Stop→Success probability with logistic GLM fit")
    plt.legend()
    plt.tight_layout()

# ------------------
# 2) Poisson (captures per stop) fit plot
# ------------------
def plot_poisson_captures_per_stop_fit(df_sessions, glm_pois):
    # Observed per-session success (same as above)
    plot_df = session_prop_with_ci(df_sessions)

    # Predict captures per stop by setting offset=log(stops)
    sess_grid = pd.DataFrame({"session": np.arange(df_sessions["session"].min(),
                                                   df_sessions["session"].max()+1)})
    # Need a representative offset; we'll predict on the logit scale via linear predictor trick:
    # Easier: build a grid matched to actual sessions, then linearly interpolate for continuity.
    grid = df_sessions[["session","stops"]].copy()
    grid = grid.drop_duplicates("session").sort_values("session")
    pred_list = []
    for s, n in zip(grid["session"], grid["stops"]):
        sf = glm_pois.get_prediction(
            exog=pd.DataFrame({"session":[s]}),
            offset=np.log([max(1.0, n)])
        ).summary_frame()
        # response scale => expected captures; divide by stops to get per-stop rate ~ success prob
        rate = sf["mean"].iloc[0] / max(1.0, n)
        lo   = sf["mean_ci_lower"].iloc[0] / max(1.0, n)
        hi   = sf["mean_ci_upper"].iloc[0] / max(1.0, n)
        pred_list.append((s, rate, lo, hi))
    pred_df = pd.DataFrame(pred_list, columns=["session","fit_rate","fit_lo","fit_hi"])

    plt.figure(figsize=(7,4))
    # observed points + CI
    yerr = np.vstack([
        plot_df["p_hat"] - plot_df["p_lo"],
        plot_df["p_hi"] - plot_df["p_hat"]
    ])
    plt.errorbar(plot_df["session"], plot_df["p_hat"], yerr=yerr, fmt="o", capsize=3,
                 alpha=0.8, label="Observed (per session, Wilson 95% CI)")

    # fitted curve + band (discrete sessions)
    plt.plot(pred_df["session"], pred_df["fit_rate"], lw=2, label="Poisson offset fit (captures per stop)")
    plt.fill_between(pred_df["session"], pred_df["fit_lo"], pred_df["fit_hi"], alpha=0.2, label="95% CI")

    plt.ylim(0, 1)
    plt.xlabel("Session")
    plt.ylabel("Capture rate per stop")
    plt.title("Captures per stop with Poisson-offset fit")
    plt.legend()
    plt.tight_layout()

# ------------------
# 3) Early vs Late bar charts with 95% CI
# ------------------
def plot_early_late_success_bars(all_stop_df_sessions):
    # aggregate to session level first
    df_sessions = (
        all_stop_df_sessions.groupby("session", as_index=False)
        .agg(captures=("captures","sum"),
             stops=("captures","size"))
    )
    sessions = np.sort(df_sessions["session"].unique())
    n = len(sessions)
    early_cut = sessions[int(np.floor(n/3)) - 1]
    late_cut  = sessions[int(np.ceil(2*n/3)) - 1]

    df_sessions["phase"] = np.where(df_sessions["session"] <= early_cut, "early",
                             np.where(df_sessions["session"] >= late_cut, "late", "mid"))

    agg = (df_sessions[df_sessions["phase"].isin(["early","late"])]
           .groupby("phase", as_index=False)
           .agg(captures=("captures","sum"),
                stops=("stops","sum")))

    agg["p_hat"] = agg["captures"] / agg["stops"].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n)) for k, n in zip(agg["captures"], agg["stops"])])
    agg["p_lo"], agg["p_hi"] = ci[:,0], ci[:,1]

    # plot
    plt.figure(figsize=(6,4))
    x = np.arange(2)
    y   = agg.set_index("phase").loc[["early","late"], "p_hat"].values
    ylo = agg.set_index("phase").loc[["early","late"], "p_lo"].values
    yhi = agg.set_index("phase").loc[["early","late"], "p_hi"].values
    yerr = np.vstack([y - ylo, yhi - y])

    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.xticks(x, ["Early", "Late"])
    plt.ylim(0, 1)
    plt.ylabel("P(capture | stop)")
    plt.title("Early vs Late: stop→success probability")
    plt.tight_layout()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# ---------- Plot 3: Early vs Late contrasts (bars with 95% CI) ----------
def plot_early_late_contrasts(df_sessions, df_trials):
    # define early/late tertiles from df_sessions (ensures same splits for both plots)
    sessions = np.sort(df_sessions["session"].unique())
    n = len(sessions)
    early_cut = sessions[int(np.floor(n/3)) - 1]
    late_cut  = sessions[int(np.ceil(2*n/3)) - 1]

    # Rates per minute
    rates = poisson_rate_per_min(df_sessions)
    rates = rates.merge(df_sessions[["session","total_duration","captures"]], on="session", how="left")
    rates["phase"] = np.where(rates["session"] <= early_cut, "early",
                       np.where(rates["session"] >= late_cut, "late", "mid"))
    agg_r = (rates[rates["phase"].isin(["early","late"])]
             .groupby("phase", as_index=False)
             .agg(rate=("rate_per_min","mean"),
                  # rough CI via normal approx on session-means of rates
                  se=("rate_per_min", lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
    agg_r["lo"] = np.clip(agg_r["rate"] - 1.96*agg_r["se"], 0, None)
    agg_r["hi"] = agg_r["rate"] + 1.96*agg_r["se"]

    # Durations: geometric mean per session then early/late means
    per_sess = geom_mean_per_session(df_trials)
    per_sess["phase"] = np.where(per_sess["session"] <= early_cut, "early",
                          np.where(per_sess["session"] >= late_cut, "late", "mid"))
    agg_d = (per_sess[per_sess["phase"].isin(["early","late"])]
             .groupby("phase", as_index=False)
             .agg(geomT=("geom_mean_T","mean"),
                  se=("geom_mean_T", lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
    agg_d["lo"] = np.clip(agg_d["geomT"] - 1.96*agg_d["se"], 0, None)
    agg_d["hi"] = agg_d["geomT"] + 1.96*agg_d["se"]

    # Plot rates
    plt.figure(figsize=(6,4))
    x = np.arange(2)
    y = agg_r.set_index("phase").loc[["early","late"], "rate"].values
    ylo = agg_r.set_index("phase").loc[["early","late"], "lo"].values
    yhi = agg_r.set_index("phase").loc[["early","late"], "hi"].values
    yerr = np.vstack([y - ylo, yhi - y])
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.xticks(x, ["Early", "Late"])
    plt.ylabel("Captures per minute")
    plt.title("Early vs Late: Reward rate")
    plt.tight_layout()

    # Plot durations
    plt.figure(figsize=(6,4))
    y = agg_d.set_index("phase").loc[["early","late"], "geomT"].values
    ylo = agg_d.set_index("phase").loc[["early","late"], "lo"].values
    yhi = agg_d.set_index("phase").loc[["early","late"], "hi"].values
    yerr = np.vstack([y - ylo, yhi - y])
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.xticks(x, ["Early", "Late"])
    plt.ylabel("Typical pursuit duration (s)")
    plt.title("Early vs Late: Duration")
    plt.tight_layout()

# ---------- Run the plots ----------

