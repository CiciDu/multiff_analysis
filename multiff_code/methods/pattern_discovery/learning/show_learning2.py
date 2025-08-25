
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact, norm
from math import sqrt
import statsmodels.api as sm

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
import statsmodels.formula.api as smf
from scipy.stats import norm
from math import sqrt

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

def session_counts_from_stop_level(df_stop_level, session_col="session",
                                   success_col="captures", stop_col=None):
    """
    Build session-level counts from stop-level data.
    If stop_col is None, assumes 1 stop per row.
    Returns df with columns: [session_col, 'success', 'stops'].
    """
    df = df_stop_level.copy()
    if stop_col is None or stop_col not in df.columns:
        df["_ones"] = 1
        stop_col_eff = "_ones"
    else:
        stop_col_eff = stop_col
    out = (df.groupby(session_col, as_index=False)
             .agg(success=(success_col, "sum"),
                  stops=(stop_col_eff, "sum")))
    if "_ones" in out.columns:
        out = out.drop(columns=[c for c in ["_ones"] if c in out.columns], errors="ignore")
    return out

def add_wilson_to_session_counts(df_sessions, success_col="success", denom_col="stops"):
    """Add p_hat and Wilson CI columns to session-level counts."""
    out = df_sessions.copy()
    p = out[success_col] / out[denom_col].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n))
                   for k, n in zip(out[success_col], out[denom_col])])
    out["p_hat"] = p
    out["p_lo"] = ci[:, 0]
    out["p_hi"] = ci[:, 1]
    return out

def tertile_phase(df, session_col="session"):
    """Add 'phase' ∈ {early, mid, late} via tertiles of session values (fallback to median split)."""
    g = df.copy()
    g[session_col] = pd.to_numeric(g[session_col])
    uniq = np.sort(g[session_col].unique())
    if len(uniq) >= 3:
        g["phase"] = pd.qcut(g[session_col], q=[0, 1/3, 2/3, 1], labels=["early","mid","late"])
    else:
        med = g[session_col].median()
        g["phase"] = np.where(g[session_col] <= med, "early", "late")
    return g

# ------------------
# 1) Logistic GLM (stop-level) fit plot
# ------------------
def plot_logistic_stop_success_fit_generic(
    df_stop_level,
    glm_logit,
    session_col="session",
    success_col="captures",
    stop_col=None,
    title="Stop→success probability across sessions",
    ylabel="P(success | stop)"
):
    """
    df_stop_level: stop-level rows with success_col ∈ {0,1}.
    glm_logit: fitted Binomial GLM on stop-level (captures ~ session).
    """
    # aggregate to session proportions for plotting
    ses = session_counts_from_stop_level(df_stop_level, session_col, success_col, stop_col)
    plot_df = add_wilson_to_session_counts(ses, success_col="success", denom_col="stops")

    # prediction grid (integer sessions in range)
    sess_grid = pd.DataFrame({session_col: np.arange(plot_df[session_col].min(),
                                                     plot_df[session_col].max()+1)})
    pred = glm_logit.get_prediction(sess_grid).summary_frame()  # response scale
    sess_grid["fit_p"]  = pred["mean"].values
    sess_grid["fit_lo"] = pred["mean_ci_lower"].values
    sess_grid["fit_hi"] = pred["mean_ci_upper"].values

    # plot
    plt.figure(figsize=(7,4))
    yerr = np.vstack([plot_df["p_hat"]-plot_df["p_lo"], plot_df["p_hi"]-plot_df["p_hat"]])
    plt.errorbar(plot_df[session_col], plot_df["p_hat"], yerr=yerr, fmt="o", capsize=3,
                 alpha=0.85, label="Observed (Wilson 95% CI)")
    plt.plot(sess_grid[session_col], sess_grid["fit_p"], lw=2, label="Logistic GLM fit")
    plt.fill_between(sess_grid[session_col], sess_grid["fit_lo"], sess_grid["fit_hi"],
                     alpha=0.2, label="95% CI")
    plt.ylim(0,1)
    plt.xlabel("Session"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.tight_layout()

# ------------------
# 2) Poisson (captures per stop) fit plot (session-level)
# ------------------
def plot_poisson_captures_per_stop_fit_generic(
    df_sessions_counts,
    glm_pois,
    session_col="session",
    success_count_col="success",
    stop_count_col="stops",
    title="Captures per stop across sessions",
    ylabel="Capture rate per stop"
):
    """
    df_sessions_counts: session-level counts with columns [session_col, success_count_col, stop_count_col].
    glm_pois: Poisson GLM fitted as success ~ session with offset(log stops).
    """
    # observed session proportions + CI
    plot_df = add_wilson_to_session_counts(
        df_sessions_counts.rename(columns={success_count_col:"success", stop_count_col:"stops"}),
        success_col="success", denom_col="stops"
    )

    # predict expected captures for each session using its own offset, then divide by stops to get per-stop rate
    grid = df_sessions_counts[[session_col, stop_count_col]].drop_duplicates().sort_values(session_col)
    preds = []
    for s, n in zip(grid[session_col], grid[stop_count_col]):
        sf = glm_pois.get_prediction(
            pd.DataFrame({session_col:[s]}),
            offset=np.log([max(1.0, n)])
        ).summary_frame()  # response scale: expected successes
        rate = sf["mean"].iloc[0] / max(1.0, n)
        lo   = sf["mean_ci_lower"].iloc[0] / max(1.0, n)
        hi   = sf["mean_ci_upper"].iloc[0] / max(1.0, n)
        preds.append((s, rate, lo, hi))
    pred_df = pd.DataFrame(preds, columns=[session_col,"fit_rate","fit_lo","fit_hi"])

    # plot
    plt.figure(figsize=(7,4))
    yerr = np.vstack([plot_df["p_hat"]-plot_df["p_lo"], plot_df["p_hi"]-plot_df["p_hat"]])
    plt.errorbar(plot_df[session_col], plot_df["p_hat"], yerr=yerr, fmt="o", capsize=3,
                 alpha=0.85, label="Observed (Wilson 95% CI)")
    plt.plot(pred_df[session_col], pred_df["fit_rate"], lw=2, label="Poisson offset fit")
    plt.fill_between(pred_df[session_col], pred_df["fit_lo"], pred_df["fit_hi"], alpha=0.2, label="95% CI")
    plt.ylim(0,1)
    plt.xlabel("Session"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.tight_layout()

# ------------------
# 3) Early vs Late bar plot (session-level Wilson CIs)
# ------------------
def plot_early_late_success_generic(
    df_stop_level,
    session_col="session",
    success_col="captures",
    stop_col=None,
    ylabel="P(success | stop)",
    title="Early vs Late: stop→success probability"
):
    """
    Works from stop-level data; aggregates to session-level and splits via tertiles.
    """
    ses = session_counts_from_stop_level(df_stop_level, session_col, success_col, stop_col)
    ses = tertile_phase(ses, session_col)

    el = (ses[ses["phase"].isin(["early","late"])]
          .groupby("phase", as_index=False)
          .agg(success=("success","sum"),
               stops=("stops","sum")))

    el["p_hat"] = el["success"] / el["stops"].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n)) for k,n in zip(el["success"], el["stops"])])
    el["p_lo"], el["p_hi"] = ci[:,0], ci[:,1]

    order = ["early","late"]
    dfp = el.set_index("phase").loc[order].reset_index()
    x = np.arange(2)
    y = dfp["p_hat"].values
    ylo = dfp["p_lo"].values
    yhi = dfp["p_hi"].values
    yerr = np.vstack([y - ylo, yhi - y])

    plt.figure(figsize=(6,4))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.xticks(x, ["Early","Late"])
    plt.ylim(0,1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()




# --- 3) Summarize results ---
def summarize_glm(model, label, transform="logit"):
    """Return dict of effect per 10 sessions."""
    coef = model.params["session"]
    se = model.bse["session"]
    z = coef / se
    pval = model.pvalues["session"]

    # 95% CI
    ci_low, ci_high = model.conf_int().loc["session"].tolist()

    if transform == "logit":
        # Odds ratio interpretation
        effect = np.exp(coef * 10)  # per 10 sessions
        ci_low_eff, ci_high_eff = np.exp(ci_low * 10), np.exp(ci_high * 10)
        return {
            "model": label,
            "coef": coef,
            "se": se,
            "z": z,
            "pval": pval,
            "effect_per_10_sessions": effect,
            "95% CI": f"[{ci_low_eff:.3f}, {ci_high_eff:.3f}]"
        }
    elif transform == "poisson":
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
