"""
General early-vs-late contrasts (v2) + manual-model replicas
------------------------------------------------------------
This version:
- Keeps the generic early/late pipeline for rate/continuous/proportion
- Adds `return_models` flag to return fitted model objects
- Adds manual-style helpers to exactly replicate your code:
    * fit_poisson_by_session / plot_poisson_rate_fit_manual
    * fit_ols_logT_by_session / plot_duration_fit_manual
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --------------------------- Phase helpers ---------------------------


def _early_late_cuts_from_sessions(df_sessions: pd.DataFrame, session_col: str = "session") -> Tuple[int, int]:
    sessions = np.sort(df_sessions[session_col].unique())
    n = len(sessions)
    if n < 3:
        raise ValueError(f"Need ≥3 sessions to define tertiles, got n={n}.")
    early_idx = max(0, int(np.floor(n/3)) - 1)
    late_idx = min(n-1, int(np.ceil(2*n/3)) - 1)
    return int(sessions[early_idx]), int(sessions[late_idx])


def _phase_from_cuts(series_session: pd.Series, early_cut: int, late_cut: int) -> np.ndarray:
    return np.where(series_session <= early_cut, "early",
                    np.where(series_session >= late_cut, "late", "mid"))


# --------------------------- CI helpers ------------------------------

def _wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        spread = (z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)) / denom
        lo = np.clip(center - spread, 0, 1)
        hi = np.clip(center + spread, 0, 1)
    return lo, hi


# --------------------------- Spec definition -------------------------

@dataclass
class MetricSpec:
    kind: Literal["rate", "continuous", "proportion"]
    name: str
    ylabel: str
    title: str

    # rate
    count_col: Optional[str] = None
    exposure_col: Optional[str] = None
    scale: float = 1.0

    # continuous
    value_col: Optional[str] = None
    transform: Literal["identity", "log"] = "identity"

    # proportion
    success_col: Optional[str] = None
    total_col: Optional[str] = None

    @staticmethod
    def rate(name: str, count_col: str, exposure_col: str, *, scale: float = 1.0, ylabel: str = "", title: str = "") -> "MetricSpec":
        return MetricSpec(kind="rate", name=name, count_col=count_col, exposure_col=exposure_col, scale=scale, ylabel=ylabel, title=title)

    @staticmethod
    def continuous(name: str, value_col: str, *, transform: Literal["identity", "log"] = "identity", ylabel: str = "", title: str = "") -> "MetricSpec":
        return MetricSpec(kind="continuous", name=name, value_col=value_col, transform=transform, ylabel=ylabel, title=title)

    @staticmethod
    def proportion(name: str, success_col: str, total_col: str, *, ylabel: str = "", title: str = "") -> "MetricSpec":
        return MetricSpec(kind="proportion", name=name, success_col=success_col, total_col=total_col, ylabel=ylabel, title=title)


# --------------------------- Core summarizer -------------------------

def summarize_early_late(df: pd.DataFrame,
                         df_sessions: pd.DataFrame,
                         spec: MetricSpec,
                         session_col: str = "session",
                         plot: bool = True,
                         return_models: bool = False):
    """Generic early-vs-late summary across metric types.

    Returns: (phase_tbl, ttest_contrast_tbl, model_contrast_tbl, effect_summary_tbl)

    If return_models=True, also returns a 5th item: a dict of fitted models, e.g.
    {"glm": <GLMResults>}, {"ols": <RegressionResults>}, or {"binom": <GLMResults>}.
    """
    early_cut, late_cut = _early_late_cuts_from_sessions(
        df_sessions, session_col=session_col)
    models_dict = {}

    if spec.kind == "rate":
        # Per-session observed rate + Poisson CI
        tmp = df[[session_col, spec.count_col, spec.exposure_col]].copy()
        E = tmp[spec.exposure_col].astype(float).to_numpy()
        k = tmp[spec.count_col].astype(float).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (k / E) * spec.scale
            se = np.sqrt(k) / E * spec.scale
            lo = np.clip(rate - 1.96*se, a_min=0, a_max=None)
            hi = np.clip(rate + 1.96*se, a_min=0, a_max=None)
        obs = tmp[[session_col]].copy()
        obs["value"], obs["lo"], obs["hi"] = rate, lo, hi
        obs["phase"] = _phase_from_cuts(obs[session_col], early_cut, late_cut)
        sub = obs[obs["phase"].isin(["early", "late"])].copy()

        phase_tbl = (sub.groupby("phase", as_index=False)
                        .agg(n=("value", "size"),
                             mean=("value", "mean"),
                             se=("value", lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
        phase_tbl["lo"] = phase_tbl["mean"] - 1.96*phase_tbl["se"]
        phase_tbl["hi"] = phase_tbl["mean"] + 1.96*phase_tbl["se"]
        phase_tbl = phase_tbl.set_index(
            "phase").loc[["early", "late"]].reset_index()

        a = sub.loc[sub["phase"] == "early", "value"].to_numpy()
        b = sub.loc[sub["phase"] == "late", "value"].to_numpy()
        t, p = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
        diff = float(np.nanmean(b) - np.nanmean(a))
        ratio = float(np.nanmean(b) / np.nanmean(a)
                      ) if np.nanmean(a) != 0 else np.nan
        pct = (ratio - 1)*100 if np.isfinite(ratio) else np.nan
        ttest_contrast_tbl = pd.DataFrame(
            [{"contrast": "late_vs_early", "diff": diff, "ratio": ratio, "percent_change": pct, "t_stat": t, "pval": p}])

        glm_df = df[[session_col, spec.count_col, spec.exposure_col]].copy()
        glm_df["phase"] = _phase_from_cuts(
            glm_df[session_col], early_cut, late_cut)
        glm_sub = glm_df[glm_df["phase"].isin(["early", "late"])].copy()
        model = smf.glm(f"{spec.count_col} ~ C(phase)", data=glm_sub, family=sm.families.Poisson(
        ), offset=np.log(glm_sub[spec.exposure_col].astype(float))).fit(cov_type="HC0")
        models_dict["glm"] = model
        coef = float(model.params.get("C(phase)[T.late]", np.nan))
        if np.isfinite(coef):
            RR = float(np.exp(coef))
            ci_l, ci_h = model.conf_int().loc["C(phase)[T.late]"].tolist()
            RR_lo, RR_hi = float(np.exp(ci_l)), float(np.exp(ci_h))
            pval = float(model.pvalues["C(phase)[T.late]"])
        else:
            RR = RR_lo = RR_hi = pval = np.nan
        model_contrast_tbl = pd.DataFrame(
            [{"contrast": "late_vs_early", "rate_ratio": RR, "RR_95CI_low": RR_lo, "RR_95CI_high": RR_hi, "pval": pval}])
        effect_summary_tbl = pd.DataFrame([{"metric": spec.name, "descriptive_ratio": ratio, "descriptive_percent_change": pct, "GLM_rate_ratio": RR, "GLM_95CI": (
            f"[{RR_lo:.3g}, {RR_hi:.3g}]" if np.all(np.isfinite([RR_lo, RR_hi])) else np.nan), "GLM_pval": pval}])

    elif spec.kind == "continuous":
        val = df[[session_col, spec.value_col]].copy()
        if spec.transform == "log":
            val["tval"] = np.log(val[spec.value_col].astype(float))
            per_sess = (val.groupby(session_col, as_index=False).agg(
                mean_log=("tval", "mean")))
            per_sess["value"] = np.exp(per_sess["mean_log"])  # geometric mean
        else:
            val["tval"] = val[spec.value_col].astype(float)
            per_sess = (val.groupby(session_col, as_index=False).agg(
                value=("tval", "mean")))

        per_sess["phase"] = _phase_from_cuts(
            per_sess[session_col], early_cut, late_cut)
        sub = per_sess[per_sess["phase"].isin(["early", "late"])].copy()
        phase_tbl = (sub.groupby("phase", as_index=False)
                        .agg(n=("value", "size"), mean=("value", "mean"), se=("value", lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
        phase_tbl["lo"] = phase_tbl["mean"] - 1.96*phase_tbl["se"]
        phase_tbl["hi"] = phase_tbl["mean"] + 1.96*phase_tbl["se"]
        phase_tbl = phase_tbl.set_index(
            "phase").loc[["early", "late"]].reset_index()

        a = sub.loc[sub["phase"] == "early", "value"].to_numpy()
        b = sub.loc[sub["phase"] == "late", "value"].to_numpy()
        t, pval_t = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
        diff = float(np.nanmean(b) - np.nanmean(a))
        ratio = float(np.nanmean(b) / np.nanmean(a)
                      ) if np.nanmean(a) != 0 else np.nan
        pct = (ratio - 1)*100 if np.isfinite(ratio) else np.nan
        ttest_contrast_tbl = pd.DataFrame(
            [{"contrast": "late_vs_early", "diff": diff, "ratio": ratio, "percent_change": pct, "t_stat": t, "pval": pval_t}])

        trials = df[[session_col, spec.value_col]].copy()
        trials["phase"] = _phase_from_cuts(
            trials[session_col], early_cut, late_cut)
        trials = trials[trials["phase"].isin(["early", "late"])].copy()
        trials["tval"] = (np.log(trials[spec.value_col].astype(
            float)) if spec.transform == "log" else trials[spec.value_col].astype(float))
        ols = smf.ols("tval ~ C(phase)", data=trials).fit(
            cov_type="cluster", cov_kwds={"groups": trials[session_col]})
        models_dict["ols"] = ols
        coef = float(ols.params.get("C(phase)[T.late]", np.nan))
        if "C(phase)[T.late]" in ols.params.index:
            ci_l, ci_h = ols.conf_int().loc["C(phase)[T.late]"].tolist()
            pval = float(ols.pvalues["C(phase)[T.late]"])
        else:
            ci_l = ci_h = pval = np.nan
        if spec.transform == "log":
            pct_change = (np.exp(coef) - 1) * \
                100.0 if np.isfinite(coef) else np.nan
            ci_pct_l = (np.exp(ci_l) - 1) * \
                100.0 if np.isfinite(ci_l) else np.nan
            ci_pct_h = (np.exp(ci_h) - 1) * \
                100.0 if np.isfinite(ci_h) else np.nan
            model_contrast_tbl = pd.DataFrame(
                [{"contrast": "late_vs_early", "percent_change": pct_change, "pct_95CI_low": ci_pct_l, "pct_95CI_high": ci_pct_h, "pval": pval}])
            effect_summary_tbl = pd.DataFrame([{"metric": spec.name, "descriptive_ratio": ratio, "descriptive_percent_change": pct, "OLS_percent_change": pct_change, "OLS_95CI": (
                f"[{ci_pct_l:.1f}%, {ci_pct_h:.1f}%]" if np.all(np.isfinite([ci_pct_l, ci_pct_h])) else np.nan), "OLS_pval": pval}])
        else:
            model_contrast_tbl = pd.DataFrame(
                [{"contrast": "late_vs_early", "coef_diff_on_identity_scale": coef, "coef_95CI_low": ci_l, "coef_95CI_high": ci_h, "pval": pval}])
            effect_summary_tbl = pd.DataFrame([{"metric": spec.name, "descriptive_ratio": ratio, "descriptive_percent_change": pct, "OLS_diff": coef, "OLS_95CI": (
                f"[{ci_l:.3g}, {ci_h:.3g}]" if np.all(np.isfinite([ci_l, ci_h])) else np.nan), "OLS_pval": pval}])

    elif spec.kind == "proportion":
        agg = df[[session_col, spec.success_col, spec.total_col]].copy().groupby(
            session_col, as_index=False).sum(numeric_only=True)
        k = agg[spec.success_col].astype(float).to_numpy()
        n = agg[spec.total_col].astype(float).to_numpy()
        p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
        lo, hi = _wilson_ci(k, n)
        per_sess = agg[[session_col]].copy()
        per_sess["value"], per_sess["lo"], per_sess["hi"], per_sess["n_trials"] = p, lo, hi, n

        per_sess["phase"] = _phase_from_cuts(
            per_sess[session_col], early_cut, late_cut)
        sub = per_sess[per_sess["phase"].isin(["early", "late"])].copy()
        phase_tbl = (sub.groupby("phase", as_index=False)
                        .agg(n=("value", "size"), mean=("value", "mean"), se=("value", lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
        phase_tbl["lo"] = np.clip(
            phase_tbl["mean"] - 1.96*phase_tbl["se"], 0, 1)
        phase_tbl["hi"] = np.clip(
            phase_tbl["mean"] + 1.96*phase_tbl["se"], 0, 1)
        phase_tbl = phase_tbl.set_index(
            "phase").loc[["early", "late"]].reset_index()

        a = sub.loc[sub["phase"] == "early", "value"].to_numpy()
        b = sub.loc[sub["phase"] == "late", "value"].to_numpy()
        t, pval_t = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
        diff = float(np.nanmean(b) - np.nanmean(a))
        ratio = float(np.nanmean(b) / np.nanmean(a)
                      ) if np.nanmean(a) not in (0, np.nan) else np.nan
        pct = (ratio - 1)*100 if np.isfinite(ratio) else np.nan
        ttest_contrast_tbl = pd.DataFrame(
            [{"contrast": "late_vs_early", "diff": diff, "ratio": ratio, "percent_change": pct, "t_stat": t, "pval": pval_t}])

        glmdf = agg.copy()
        glmdf["phase"] = _phase_from_cuts(
            glmdf[session_col], early_cut, late_cut)
        glmdf = glmdf[glmdf["phase"].isin(["early", "late"])].copy()
        glmdf["prop"] = np.divide(glmdf[spec.success_col], glmdf[spec.total_col], out=np.nan *
                                  np.ones_like(glmdf[spec.success_col], dtype=float), where=glmdf[spec.total_col] > 0)
        model = smf.glm("prop ~ C(phase)", data=glmdf, family=sm.families.Binomial(
        ), freq_weights=glmdf[spec.total_col].astype(float)).fit(cov_type="HC0")
        models_dict["binom"] = model
        coef = float(model.params.get("C(phase)[T.late]", np.nan))
        if np.isfinite(coef):
            OR = float(np.exp(coef))
            ci_l, ci_h = model.conf_int().loc["C(phase)[T.late]"].tolist()
            OR_lo, OR_hi = float(np.exp(ci_l)), float(np.exp(ci_h))
            pval = float(model.pvalues["C(phase)[T.late]"])
        else:
            OR = OR_lo = OR_hi = pval = np.nan
        model_contrast_tbl = pd.DataFrame(
            [{"contrast": "late_vs_early", "odds_ratio": OR, "OR_95CI_low": OR_lo, "OR_95CI_high": OR_hi, "pval": pval}])
        effect_summary_tbl = pd.DataFrame([{"metric": spec.name, "descriptive_ratio": ratio, "descriptive_percent_change": pct, "GLM_odds_ratio": OR, "GLM_95CI": (
            f"[{OR_lo:.3g}, {OR_hi:.3g}]" if np.all(np.isfinite([OR_lo, OR_hi])) else np.nan), "GLM_pval": pval}])

    else:
        raise ValueError(f"Unknown spec.kind: {spec.kind}")

    if plot:
        # Get p-value from the appropriate model for annotation
        pval = None
        if spec.kind == "rate" and "glm" in models_dict:
            pval = float(models_dict["glm"].pvalues.get(
                "C(phase)[T.late]", 1.0))
        elif spec.kind == "continuous" and "ols" in models_dict:
            pval = float(models_dict["ols"].pvalues.get(
                "C(phase)[T.late]", 1.0))
        elif spec.kind == "proportion" and "binom" in models_dict:
            pval = float(models_dict["binom"].pvalues.get(
                "C(phase)[T.late]", 1.0))

        # Use percentage formatting for proportion data
        if spec.kind == "proportion":
            _plot_phase_bar_percentage(
                phase_tbl, ylabel=spec.ylabel, title=spec.title, pval=pval)
        else:
            _plot_phase_bar(phase_tbl, ylabel=spec.ylabel,
                            title=spec.title, pval=pval)

    if return_models:
        return phase_tbl, ttest_contrast_tbl, model_contrast_tbl, effect_summary_tbl, models_dict
    return phase_tbl, ttest_contrast_tbl, model_contrast_tbl, effect_summary_tbl


# --------------------------- Plotting --------------------------------

def _plot_phase_bar(phase_tbl: pd.DataFrame, *, ylabel: str, title: str, pval: Optional[float] = None):
    """
    Create early vs late comparison plot similar to analyze_ratio_trend style.

    Parameters:
    -----------
    phase_tbl : pd.DataFrame
        DataFrame with columns ['phase', 'mean', 'lo', 'hi', 'n']
    ylabel : str
        Y-axis label
    title : str
        Plot title
    pval : Optional[float]
        P-value to display on plot
    """
    # Get data in correct order
    dfp = phase_tbl.set_index("phase").loc[["early", "late"]].reset_index()

    # Position bars closer together
    x = np.array([-0.3, 0.3])
    y = dfp["mean"].to_numpy()
    ylo = dfp["lo"].to_numpy()
    yhi = dfp["hi"].to_numpy()
    yerr = np.vstack([y - ylo, yhi - y])

    # Create figure with similar dimensions to analyze_ratio_trend
    fig, ax = plt.subplots(figsize=(4.8, 4.6))

    # Bars (narrow + translucent)
    ax.bar(x, y, width=0.35, alpha=0.35, zorder=1, color="tab:blue")

    # Point estimate + CI whiskers on top
    ax.errorbar(x, y, yerr=yerr, fmt="o", lw=2, capsize=5,
                zorder=3, color="tab:blue", ecolor="tab:blue")
    ax.scatter(x, y, s=40, zorder=4, color="tab:blue")

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Early 1/3 sessions", "Late 2/3 sessions"], fontsize=12)

    # Add sample size labels above bars if available
    if "n" in dfp.columns:
        for xi, yi, n in zip(x, y, dfp["n"]):
            ax.text(xi, yi + (yhi.max() - y.min()) * 0.05, f"n={int(n)}",
                    ha="center", va="bottom", fontsize=9)

    # Y-axis formatting
    ymax = max(yhi.max() * 1.10, y.max() + (yhi.max() - y.min()) * 0.1)
    ax.set_ylim(0, ymax)

    # Add grid
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
    plt.show()


def _plot_phase_bar_percentage(phase_tbl: pd.DataFrame, *, ylabel: str, title: str, pval: Optional[float] = None):
    """
    Create early vs late comparison plot with percentage formatting, similar to analyze_ratio_trend style.

    Parameters:
    -----------
    phase_tbl : pd.DataFrame
        DataFrame with columns ['phase', 'mean', 'lo', 'hi', 'n']
    ylabel : str
        Y-axis label
    title : str
        Plot title
    pval : Optional[float]
        P-value to display on plot
    """
    # Get data in correct order
    dfp = phase_tbl.set_index("phase").loc[["early", "late"]].reset_index()

    # Convert to percentages
    dfp["pct"] = 100.0 * dfp["mean"]
    dfp["pct_lo"] = 100.0 * dfp["lo"]
    dfp["pct_hi"] = 100.0 * dfp["hi"]

    # Position bars closer together
    x = np.array([-0.3, 0.3])
    pct = dfp["pct"].to_numpy()
    pctlo = dfp["pct_lo"].to_numpy()
    pcthi = dfp["pct_hi"].to_numpy()
    yerr = np.vstack([pct - pctlo, pcthi - pct])

    # Create figure with similar dimensions to analyze_ratio_trend
    fig, ax = plt.subplots(figsize=(4.8, 4.6))

    # Bars (narrow + translucent)
    ax.bar(x, pct, width=0.35, alpha=0.35, zorder=1, color="tab:blue")

    # Point estimate + CI whiskers on top
    ax.errorbar(x, pct, yerr=yerr, fmt="o", lw=2, capsize=5,
                zorder=3, color="tab:blue", ecolor="tab:blue")
    ax.scatter(x, pct, s=40, zorder=4, color="tab:blue")

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Early 1/3 sessions", "Late 2/3 sessions"], fontsize=12)

    # Add sample size labels above bars if available
    if "n" in dfp.columns:
        for xi, yi, n in zip(x, pct, dfp["n"]):
            ax.text(xi, yi + 2.0, f"n={int(n)}",
                    ha="center", va="bottom", fontsize=9)

    # Y-axis formatting as percentages
    ymax = min(100.0, max(pcthi.max() * 1.10, pct.max() + 6.0))
    ax.set_ylim(0, ymax)
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(round(t))}%" for t in ticks])

    # Add grid
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
    plt.show()


# --------------------------- Manual-style fits (replica) -------------

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
    out["lo"] = pred["mean_ci_lower"].to_numpy()
    out["hi"] = pred["mean_ci_upper"].to_numpy()
    return out


def plot_poisson_rate_fit_manual(df_sessions: pd.DataFrame,
                                 po,
                                 *, session_col: str = "session",
                                 count_col: str = "captures",
                                 exposure_col: str = "total_duration",
                                 per: Literal["second", "minute"] = "minute",
                                 title: str = "Reward throughput (captures/min)",
                                 title_prefix: str = ""):
    E = df_sessions[exposure_col].astype(float).to_numpy()
    k = df_sessions[count_col].astype(float).to_numpy()
    scale = 60.0 if per == "minute" else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (k / E) * scale
        se = np.sqrt(k) / E * scale
        lo = np.clip(rate - 1.96*se, a_min=0, a_max=None)
        hi = np.clip(rate + 1.96*se, a_min=0, a_max=None)

    obs = df_sessions[[session_col]].copy()
    obs["rate"], obs["lo"], obs["hi"] = rate, lo, hi

    sess_min, sess_max = int(df_sessions[session_col].min()), int(
        df_sessions[session_col].max())
    fit_df = predict_poisson_rate_curve(po, sess_min, sess_max, per=per)

    # Extract p-value for session coefficient
    # Default to 1.0 if 'session' not found
    p_value = po.pvalues.get('session', 1.0)
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else f"p < 0.001"

    plt.figure(figsize=(7, 5))
    plt.errorbar(obs[session_col], obs["rate"], yerr=[obs["rate"]-obs["lo"],
                 obs["hi"]-obs["rate"]], fmt="o", alpha=0.7, label=f"Observed ({per}) ±95% CI")
    plt.plot(fit_df["session"], fit_df["fit"],
             lw=2, label=f"Poisson fit ({per})")
    plt.fill_between(fit_df["session"], fit_df["lo"],
                     fit_df["hi"], alpha=0.2, label="95% CI")
    plt.xlabel("Session")
    plt.ylabel(f"Captures per {per}")

    # Add title with prefix and p-value annotation
    full_title = f"{title_prefix}{title}" if title_prefix else title
    plt.title(f"{full_title}\n{p_text}")

    plt.legend()
    plt.tight_layout()
    plt.show()


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


def plot_duration_fit_manual(df_trials: pd.DataFrame,
                             ols,
                             *, session_col: str = "session",
                             value_col: str = "duration_sec",
                             title: str = "Pursuit duration with log-linear fit",
                             title_prefix: str = ""):
    tmp = df_trials[[session_col, value_col]].copy()
    tmp["logT"] = np.log(tmp[value_col].astype(float))
    per_sess = (tmp.groupby(session_col, as_index=False).agg(
        mean_log=("logT", "mean")))
    per_sess["geom_sec"] = np.exp(per_sess["mean_log"])

    sess_min, sess_max = int(tmp[session_col].min()), int(
        tmp[session_col].max())
    fit_df = predict_duration_curve_seconds(ols, sess_min, sess_max)

    # Extract p-value for session coefficient
    # Default to 1.0 if 'session' not found
    p_value = ols.pvalues.get('session', 1.0)
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else f"p < 0.001"

    plt.figure(figsize=(7, 5))
    plt.scatter(per_sess[session_col], per_sess["geom_sec"],
                alpha=0.8, label="Geometric mean (per session)")
    plt.plot(fit_df["session"], fit_df["fit_sec"],
             lw=2, label="OLS fit on log-duration")
    plt.fill_between(fit_df["session"], fit_df["lo_sec"],
                     fit_df["hi_sec"], alpha=0.2, label="95% CI")
    plt.xlabel("Session")
    plt.ylabel("Typical pursuit duration (s)")

    # Add title with prefix and p-value annotation
    full_title = f"{title_prefix}{title}" if title_prefix else title
    plt.title(f"{full_title}\n{p_text}")

    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------- One-shot helper (both models + plots) ---

def fit_and_plot_manual(df_trials: pd.DataFrame,
                        df_sessions: pd.DataFrame,
                        *,
                        session_col: str = "session",
                        value_col: str = "duration_sec",
                        count_col: str = "captures",
                        exposure_col: str = "total_duration",
                        rate_per: Literal["second", "minute"] = "minute",
                        title_prefix: str = "",
                        make_plots: bool = True):
    """Fit BOTH manual models (Poisson w/ offset; OLS on logT) using your cleaned
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
        plot_poisson_rate_fit_manual(df_sessions, po,
                                     session_col=session_col,
                                     count_col=count_col,
                                     exposure_col=exposure_col,
                                     per=rate_per,
                                     title_prefix=title_prefix,
                                     title=f"Reward throughput (captures/{'min' if rate_per=='minute' else 'sec'})")
        plot_duration_fit_manual(df_trials, ols,
                                 session_col=session_col,
                                 value_col=value_col,
                                 title_prefix=title_prefix,
                                 title="Pursuit duration (log-linear fit)")

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
