
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


def evaluate_ratio_trend(df_sess_counts, event_count_col="success", denom_count_col="stops"):

    glm_pois = smf.glm(
        f"{event_count_col} ~ session",
        data=df_sess_counts,
        family=sm.families.Poisson(),
        offset=np.log(df_sess_counts[denom_count_col])
    ).fit(cov_type="HC0")

    # Collect summaries
    results = [
        summarize_glm(glm_pois, "Poisson", transform="poisson")
    ]
    

    results_df = pd.DataFrame(results)
    print(results_df)

    plot_poisson_ratio_fit_generic(
        df_sess_counts, glm_pois,
        session_col="session", event_count_col=event_count_col, denom_count_col=denom_count_col,
        title=f"Ratio of {event_count_col}", ylabel=f"Ratio of {event_count_col}"
    )

    plot_early_late_ratio(
        df_sess_counts,
        session_col="session", event_count_col=event_count_col, denom_count_col=denom_count_col,
        ylabel=f"P({event_count_col} | {denom_count_col})", title="Early vs Late (custom)"
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


def tertile_phase(df, session_col="session"):
    """Add 'phase' ∈ {early, mid, late} via tertiles of session values (fallback to median split)."""
    g = df.copy()
    g[session_col] = pd.to_numeric(g[session_col])
    uniq = np.sort(g[session_col].unique())
    if len(uniq) >= 3:
        g["phase"] = pd.qcut(g[session_col], q=[0, 1/3, 2/3, 1],
                             labels=["early", "mid", "late"])
    else:
        med = g[session_col].median()
        g["phase"] = np.where(g[session_col] <= med, "early", "late")
    return g

# ------------------
# 1) Logistic GLM (stop-level) fit plot
# ------------------


# ------------------
# 2) Poisson (captures per stop) fit plot (session-level)
# ------------------


def plot_poisson_ratio_fit_generic(
    df_sessions_counts,
    glm_pois,
    session_col="session",
    event_count_col="success",
    denom_count_col="stops",
    title="Ratio across sessions",
    ylabel="Ratio"
):
    """
    df_sessions_counts: session-level counts with columns [session_col, event_count_col, denom_count_col].
    glm_pois: Poisson GLM fitted as success ~ session with offset(log stops).
    """
    # observed session proportions + CI
    plot_df = add_wilson_to_session_counts(
        df_sessions_counts.rename(
            columns={event_count_col: "success", denom_count_col: "stops"}),
        event_count_col="success", denom_col="stops"
    )

    # predict expected captures for each session using its own offset, then divide by stops to get per-stop rate
    grid = df_sessions_counts[[session_col, denom_count_col]
                              ].drop_duplicates().sort_values(session_col)
    preds = []
    for s, n in zip(grid[session_col], grid[denom_count_col]):
        sf = glm_pois.get_prediction(
            pd.DataFrame({session_col: [s]}),
            offset=np.log([max(1.0, n)])
        ).summary_frame()  # response scale: expected successes
        rate = sf["mean"].iloc[0] / max(1.0, n)
        lo = sf["mean_ci_lower"].iloc[0] / max(1.0, n)
        hi = sf["mean_ci_upper"].iloc[0] / max(1.0, n)
        preds.append((s, rate, lo, hi))
    pred_df = pd.DataFrame(
        preds, columns=[session_col, "fit_rate", "fit_lo", "fit_hi"])

    # plot
    plt.figure(figsize=(7, 4))
    yerr = np.vstack([plot_df["p_hat"]-plot_df["p_lo"],
                     plot_df["p_hi"]-plot_df["p_hat"]])
    plt.errorbar(plot_df[session_col], plot_df["p_hat"], yerr=yerr, fmt="o", capsize=3,
                 alpha=0.85, label="Observed (Wilson 95% CI)")
    plt.plot(pred_df[session_col], pred_df["fit_rate"],
             lw=2, label="Poisson offset fit")
    plt.fill_between(pred_df[session_col], pred_df["fit_lo"],
                     pred_df["fit_hi"], alpha=0.2, label="95% CI")
    plt.ylim(0, 1)
    plt.xlabel("Session")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

# ------------------
# 3) Early vs Late bar plot (session-level Wilson CIs)
# ------------------

def plot_early_late_ratio(
    df_sess_counts,
    session_col="session",
    event_count_col="success",
    denom_count_col="stops",
    ylabel="P(success | baseline)",
    title="Early vs Late: stop→success probability"
):
    """
    Works from stop-level data; aggregates to session-level and splits via tertiles.
    """
    ses = tertile_phase(df_sess_counts, session_col)

    el = (
        ses[ses["phase"].isin(["early", "late"])]
        .groupby("phase", as_index=False)
        .agg(**{event_count_col: (event_count_col, "sum"),
                denom_count_col: (denom_count_col, "sum")})
    )

    el["p_hat"] = el[event_count_col] / el[denom_count_col].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n))
                  for k, n in zip(el[event_count_col], el[denom_count_col])])
    el["p_lo"], el["p_hi"] = ci[:, 0], ci[:, 1]

    # Reorder for plotting
    order = ["early", "late"]
    dfp = el.set_index("phase").loc[order].reset_index()

    # Plot
    x = np.arange(2)
    y = dfp["p_hat"].values
    ylo = dfp["p_lo"].values
    yhi = dfp["p_hi"].values
    yerr = np.vstack([y - ylo, yhi - y])

    plt.figure(figsize=(6, 4))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.xticks(x, ["Early", "Late"])
    plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    # Print table of results
    display_cols = [event_count_col, denom_count_col, "p_hat", "p_lo", "p_hi"]
    print("\n--- Early vs Late Results ---")
    print(dfp[["phase"] + display_cols].to_string(index=False))

    return dfp[["phase"] + display_cols]


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
