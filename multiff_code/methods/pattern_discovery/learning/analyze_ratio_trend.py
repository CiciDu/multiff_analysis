
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact, norm
from math import sqrt
import statsmodels.api as sm


def evaluate_ratio_trend(df_sess_counts, event_count_col="success", denom_count_col="stops",
                         title=None, ylabel="P(Events | Baseline)"):

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

    # Get p-value from results
    pval = results_df.iloc[0]['pval']
    
    if title is None:
        title = f"Ratio of {event_count_col}"

    plot_poisson_ratio_fit_generic(
        df_sess_counts, glm_pois,
        session_col="session", event_count_col=event_count_col, denom_count_col=denom_count_col,
        title=title, ylabel=ylabel,
        pval=pval
    )

    plot_early_late_ratio(
        df_sess_counts,
        session_col="session", event_count_col=event_count_col, denom_count_col=denom_count_col,
        ylabel=ylabel, title="Early vs Late",
        pval=pval
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
    ylabel="Ratio",
    pval=None
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
    # plt.ylim(0, 1)
    y_min = min(plot_df["p_lo"].min(), pred_df["fit_lo"].min())
    y_max = max(plot_df["p_hi"].max(), pred_df["fit_hi"].max())
    plt.ylim(y_min * 0.9, y_max * 1.2)

    plt.xlabel("Session", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add p-value to plot if provided
    if pval is not None:
        pval_text = f"p = {pval:.4f}"
        if pval < 0.001:
            pval_text = "p < 0.001"
        elif pval < 0.01:
            pval_text = f"p = {pval:.3f}"
        elif pval < 0.05:
            pval_text = f"p = {pval:.3f}"
        else:
            pval_text = f"p = {pval:.3f}"

        plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    y_min = min(plot_df["p_lo"].min(), pred_df["fit_lo"].min())
    y_max = max(plot_df["p_hi"].max(), pred_df["fit_hi"].max())
    plt.ylim(y_min * 0.9, y_max * 1.2)


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


def plot_early_late_ratio(
    df_sess_counts,
    session_col="session",
    event_count_col="success",
    denom_count_col="stops",
    ylabel="P(captures | stops)",
    title="Early vs Late",
    pval=None
):
    """
    Hybrid chart: translucent bars + point estimate + 95% CI.
    Bars are narrower and closer together, figure is thinner.
    """
    # --- Aggregate ---
    ses = tertile_phase(df_sess_counts, session_col)
    el = (
        ses[ses["phase"].isin(["early", "late"])]
        .groupby("phase", as_index=False)
        .agg(**{event_count_col: (event_count_col, "sum"),
                denom_count_col: (denom_count_col, "sum")})
    )

    # Proportions + Wilson CI
    el["p_hat"] = el[event_count_col] / el[denom_count_col].clip(lower=1)
    ci = np.array([wilson_ci(int(k), int(n))
                   for k, n in zip(el[event_count_col], el[denom_count_col])])
    el["p_lo"], el["p_hi"] = ci[:, 0], ci[:, 1]

    # Order for plotting
    dfp = el.set_index("phase").loc[["early", "late"]].reset_index()

    # Convert to percentages
    dfp["pct"] = 100.0 * dfp["p_hat"]
    dfp["pct_lo"] = 100.0 * dfp["p_lo"]
    dfp["pct_hi"] = 100.0 * dfp["p_hi"]

    # --- Plot ---
    x = np.array([-0.3, 0.3])  # Closer positions for smaller gap
    pct = dfp["pct"].to_numpy()
    pctlo = dfp["pct_lo"].to_numpy()
    pcthi = dfp["pct_hi"].to_numpy()
    yerr = np.vstack([pct - pctlo, pcthi - pct])

    # Thinner figure
    fig, ax = plt.subplots(figsize=(4.8, 4.6))

    # Bars (narrow + translucent)
    ax.bar(x, pct, width=0.35, alpha=0.35, zorder=1, color="tab:blue")

    # Point estimate + CI whiskers on top
    ax.errorbar(x, pct, yerr=yerr, fmt="o", lw=2, capsize=5,
                zorder=3, color="tab:blue", ecolor="tab:blue")
    ax.scatter(x, pct, s=40, zorder=4, color="tab:blue")

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(["Early 1/3 sessions", "Late 2/3 sessions"], fontsize=12)

    # k/n labels above bars
    for xi, yi, k, n in zip(x, pct, dfp[event_count_col], dfp[denom_count_col]):
        ax.text(xi, yi + 2.0, f"{int(k)}/{int(n)}",
                ha="center", va="bottom", fontsize=9)

    # Y-axis formatting
    ymax = min(100.0, max(pcthi.max() * 1.10, pct.max() + 6.0))
    ax.set_ylim(0, ymax)
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(round(t))}%" for t in ticks])

    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    # Optional p-value box
    if pval is not None:
        ptxt = "p < 0.001" if pval < 1e-3 else f"p = {pval:.3f}"
        ax.text(0.02, 0.98, ptxt, transform=ax.transAxes,
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    plt.tight_layout()

    # Results table
    display_cols = [event_count_col, denom_count_col, "p_hat", "p_lo", "p_hi"]
    print("\n--- Early vs Late Results ---")
    print(dfp[["phase"] + display_cols].to_string(index=False))

    return dfp[["phase"] + display_cols]


def show_event_ratio(df_monkey, event):
    df_event = df_monkey[df_monkey['Item'] == event].sort_values(by='Session').reset_index(drop=True)

    event_count_col = event
    denom_count_col = "all_trial_count"

    df_event.rename(columns={'Session': 'session',
                            'Frequency':event_count_col,
                            'N_total': denom_count_col,
                            }, inplace=True)

    evaluate_ratio_trend(df_event, event_count_col=event_count_col, denom_count_col=denom_count_col)