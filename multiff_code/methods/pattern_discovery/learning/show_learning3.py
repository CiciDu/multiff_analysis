from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---- Spec with your field names ---------------------------------------------
@dataclass
class MetricSpec:
    kind: Literal["binom", "rate", "continuous"]
    event_count_col: Optional[str] = None   # for "binom" and "rate"
    denom_count_col: Optional[str] = None   # for "binom" and "rate"
    value_col: Optional[str] = None         # for "continuous"
    ylabel: str = "Value"
    title: str = "Early vs Late"

# ---- Utilities ---------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2.0)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / den
    half = (z / den) * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))
    return (max(0.0, center - half), min(1.0, center + half))

def _early_late_cuts_from_sessions(df_sessions: pd.DataFrame,
                                   session_col: str = "session") -> Tuple[np.ndarray, np.ndarray]:
    uniq = (df_sessions[[session_col]]
            .drop_duplicates()
            .sort_values(session_col)
            .reset_index(drop=True))
    n = len(uniq)
    if n < 2:
        return uniq[session_col].to_numpy(), np.array([], dtype=uniq[session_col].dtype)
    cut1 = int(np.floor(n/3))
    cut2 = int(np.floor(2*n/3))
    early = uniq.iloc[:max(1, cut1)][session_col].to_numpy()
    late  = uniq.iloc[max(cut2, n-1):][session_col].to_numpy()
    return early, late

def _agg_phase_tables(df: pd.DataFrame,
                      spec: MetricSpec,
                      phase: pd.Series,
                      alpha: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    df["phase"] = phase.values

    if spec.kind in ("binom", "rate"):
        g = (df.groupby("phase", as_index=False)
               .agg(event=(spec.event_count_col, "sum"),
                    denom=(spec.denom_count_col, "sum")))
        if spec.kind == "binom":
            g["p_hat"] = g["event"] / g["denom"].clip(lower=1)
            cis = np.array([wilson_ci(int(k), int(n)) for k, n in zip(g["event"], g["denom"])])
            g["ci_lo"], g["ci_hi"] = cis[:, 0], cis[:, 1]
            g["display"] = g["p_hat"]
        else:
            g["rate"] = g["event"] / g["denom"].clip(lower=1e-12)
            z = norm.ppf(0.975)
            with np.errstate(divide="ignore", invalid="ignore"):
                se_log = (1 / np.sqrt(g["event"].clip(lower=1))).replace([np.inf], np.nan)
            log_r = np.log(g["rate"].replace(0, np.nan))
            g["ci_lo"] = np.exp(log_r - z*se_log)
            g["ci_hi"] = np.exp(log_r + z*se_log)
            g["display"] = g["rate"]
        return g

    if spec.kind == "continuous":
        g = (df.groupby("phase", as_index=False)
               .agg(n=(spec.value_col, "size"),
                    mean=(spec.value_col, "mean"),
                    std=(spec.value_col, "std")))
        g["se"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
        z = norm.ppf(0.975)
        g["ci_lo"] = g["mean"] - z*g["se"]
        g["ci_hi"] = g["mean"] + z*g["se"]
        g["display"] = g["mean"]
        return g

    raise ValueError("Unknown spec.kind")

# ---- Main -------------------------------------------------------------------
def summarize_early_late(df: pd.DataFrame,
                         df_sessions: pd.DataFrame,
                         spec: MetricSpec,
                         session_col: str = "session",
                         plot: bool = True,
                         return_models: bool = False,
                         alpha: float = 0.05):
    """
    Returns: (phase_tbl, ttest_contrast_tbl, model_contrast_tbl, effect_summary_tbl)
    If return_models=True, also returns a 5th item: dict of fitted models.
    """
    early_cut, late_cut = _early_late_cuts_from_sessions(df_sessions, session_col=session_col)

    sess = df[session_col].to_numpy()
    phase = pd.Series(np.where(np.isin(sess, early_cut), "early",
                        np.where(np.isin(sess, late_cut), "late", "mid")),
                      index=df.index)

    mask = phase.isin(["early", "late"])
    # after df2/phase2 are created
    df2 = df.loc[mask].reset_index(drop=True)
    phase2 = phase.loc[mask].reset_index(drop=True)
    df2["phase"] = phase2.values  # <<< ensure 'phase' exists in df2


    phase_tbl = _agg_phase_tables(df2, spec, phase2, alpha=alpha)
    models_dict: Dict[str, Any] = {}

    if spec.kind == "continuous":
        e = df2.loc[phase2.eq("early"), spec.value_col].dropna()
        l = df2.loc[phase2.eq("late"),  spec.value_col].dropna()
        t_stat, p_t = ttest_ind(e, l, equal_var=False)
        diff = l.mean() - e.mean()
        ttest_contrast_tbl = pd.DataFrame({
            "contrast": ["late - early"],
            "estimate": [diff],
            "test": ["Welch t"],
            "stat": [t_stat],
            "p_value": [p_t]
        })
        mdl = smf.ols(f"{spec.value_col} ~ C(phase)",
            data=df2[[spec.value_col, "phase"]]).fit()

        models_dict["ols"] = mdl
        coef = mdl.params.get("C(phase)[T.late]", np.nan)
        se = mdl.bse.get("C(phase)[T.late]", np.nan)
        z975 = norm.ppf(0.975)
        ci = (coef - z975*se, coef + z975*se)
        model_contrast_tbl = pd.DataFrame({
            "model": ["OLS"],
            "effect": ["late - early"],
            "estimate": [coef],
            "ci_lo": [ci[0]],
            "ci_hi": [ci[1]],
            "p_value": [mdl.pvalues.get("C(phase)[T.late]", np.nan)]
        })
        effect_summary_tbl = pd.DataFrame({
            "metric": [spec.ylabel],
            "early_display": [phase_tbl.loc[phase_tbl["phase"].eq("early"), "display"].values[0]],
            "late_display":  [phase_tbl.loc[phase_tbl["phase"].eq("late"),  "display"].values[0]],
            "difference": [diff]
        })

    elif spec.kind == "binom":
        row_e = phase_tbl.loc[phase_tbl["phase"].eq("early")].iloc[0]
        row_l = phase_tbl.loc[phase_tbl["phase"].eq("late")].iloc[0]
        # Simple two-proportion Wald z (guarded)
        count = np.array([row_l["event"], row_e["event"]], dtype=int)
        nobs  = np.array([row_l["denom"], row_e["denom"]], dtype=int)
        if (nobs > 0).all():
            p_pool = count.sum() / nobs.sum()
            se = np.sqrt(p_pool*(1-p_pool)*((1/nobs[0]) + (1/nobs[1])))
            z = (row_l["p_hat"] - row_e["p_hat"]) / se if se > 0 else np.nan
            p_val = 2*(1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        else:
            z, p_val = np.nan, np.nan

        ttest_contrast_tbl = pd.DataFrame({
            "contrast": ["late - early (pct-pts)"],
            "estimate": [(row_l["p_hat"] - row_e["p_hat"]) * 100],
            "test": ["Two-proportion (Wald)"],
            "stat": [z],
            "p_value": [p_val]
        })

        # Binomial GLM at session level (weights = denom)
        sess_agg = (df2.groupby([session_col, "phase"], as_index=False)
                    .agg(event=(spec.event_count_col, "sum"),
                        denom=(spec.denom_count_col, "sum")))
        mdl = smf.glm("event ~ C(phase)", data=sess_agg,
                    family=sm.families.Binomial(),
                    freq_weights=sess_agg["denom"]).fit(cov_type="HC0")

        models_dict["binom"] = mdl
        coef = mdl.params.get("C(phase)[T.late]", np.nan)
        se = mdl.bse.get("C(phase)[T.late]", np.nan)
        z975 = norm.ppf(0.975)
        ci = (coef - z975*se, coef + z975*se)
        or_est = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        or_ci = tuple(np.exp(ci)) if all(np.isfinite(ci)) else (np.nan, np.nan)
        model_contrast_tbl = pd.DataFrame({
            "model": ["Binomial GLM (logit)"],
            "effect": ["late vs early (odds ratio)"],
            "estimate": [or_est],
            "ci_lo": [or_ci[0]],
            "ci_hi": [or_ci[1]],
            "p_value": [mdl.pvalues.get("C(phase)[T.late]", np.nan)]
        })
        effect_summary_tbl = pd.DataFrame({
            "metric": [spec.ylabel],
            "early_display": [row_e["p_hat"]],
            "late_display":  [row_l["p_hat"]],
            "difference_pct_pts": [(row_l["p_hat"] - row_e["p_hat"]) * 100]
        })

    elif spec.kind == "rate":
        row_e = phase_tbl.loc[phase_tbl["phase"].eq("early")].iloc[0]
        row_l = phase_tbl.loc[phase_tbl["phase"].eq("late")].iloc[0]
        # Session-level Poisson GLM with offset
        sess_agg = (df2.groupby([session_col, "phase"], as_index=False)
                      .agg(event=(spec.event_count_col, "sum"),
                           denom=(spec.denom_count_col, "sum")))
        mdl = smf.glm("event ~ C(phase)", data=sess_agg,
                      family=sm.families.Poisson(),
                      offset=np.log(sess_agg["denom"].clip(lower=1e-12))).fit(cov_type="HC0")
        models_dict["glm_pois"] = mdl
        coef = mdl.params.get("C(phase)[T.late]", np.nan)  # log rate ratio
        se = mdl.bse.get("C(phase)[T.late]", np.nan)
        z975 = norm.ppf(0.975)
        ci = (coef - z975*se, coef + z975*se)
        rr = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        rr_ci = tuple(np.exp(ci)) if all(np.isfinite(ci)) else (np.nan, np.nan)

        ttest_contrast_tbl = pd.DataFrame({
            "contrast": ["late / early (rate ratio)"],
            "estimate": [rr],
            "test": ["Poisson GLM (Wald on log RR)"],
            "stat": [coef / se if (np.isfinite(coef) and np.isfinite(se) and se > 0) else np.nan],
            "p_value": [mdl.pvalues.get("C(phase)[T.late]", np.nan)]
        })
        model_contrast_tbl = pd.DataFrame({
            "model": ["Poisson GLM (log link)"],
            "effect": ["late / early (rate ratio)"],
            "estimate": [rr],
            "ci_lo": [rr_ci[0]],
            "ci_hi": [rr_ci[1]],
            "p_value": [mdl.pvalues.get("C(phase)[T.late]", np.nan)]
        })
        effect_summary_tbl = pd.DataFrame({
            "metric": [spec.ylabel],
            "early_display": [row_e["rate"]],
            "late_display":  [row_l["rate"]],
            "rate_ratio": [rr]
        })

    else:
        raise ValueError("Unknown spec.kind")

    # ---- Optional plot -------------------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(3.6, 3.8))
        order = ["early", "late"]
        dat = phase_tbl.set_index("phase").loc[order].reset_index()
        x = np.arange(len(dat))
        ax.bar(x, dat["display"].values, alpha=0.35, width=0.55, edgecolor="none")
        ax.scatter(x, dat["display"].values, zorder=3)
        lo = dat["ci_lo"].to_numpy()
        hi = dat["ci_hi"].to_numpy()
        ax.vlines(x, lo, hi)
        cap = 0.1
        for xi, l, h in zip(x, lo, hi):
            ax.hlines([l, h], xi - cap, xi + cap)
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel(spec.ylabel)
        ax.set_title(spec.title, pad=8, loc="center")
        ax.margins(x=0.15)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.show()

    results = (phase_tbl, ttest_contrast_tbl, model_contrast_tbl, effect_summary_tbl)
    if return_models:
        return (*results, models_dict)
    return results
