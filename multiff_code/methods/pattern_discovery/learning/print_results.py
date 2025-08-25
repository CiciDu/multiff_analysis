import numpy as np
import pandas as pd

# ---------- formatting helpers ----------
def _fmt_ci(lo, hi, digits=3, pct=False, signed=False):
    if pd.isna(lo) or pd.isna(hi):
        return ""
    if pct:
        lo, hi = lo, hi
        fmt = f"{{:{'+' if signed else ''}.{digits}f}}%"
        return f"[{fmt.format(lo)}, {fmt.format(hi)}]"
    else:
        fmt = f"{{:.{digits}f}}"
        return f"[{fmt.format(lo)}, {fmt.format(hi)}]"

def _fmt_p(p):
    if pd.isna(p): return ""
    stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
    return f"{p:.3g} {stars}"

def _style(
    df: pd.DataFrame,
    caption: str = "",
    round_map: dict | None = None,
    percent_cols: list[str] | None = None,
    ci_cols_as_text: list[str] | None = None,
    highlight_sig_on: str | None = None
):
    # Copy so we don’t mutate originals
    df = df.copy()

    # Round numeric columns
    if round_map:
        for col, nd in round_map.items():
            if col in df.columns:
                df[col] = df[col].astype(float).round(nd)

    # Percent columns formatting
    percent_cols = percent_cols or []
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")

    # Convert CI columns to plain text (avoid scientific)
    if ci_cols_as_text:
        for col in ci_cols_as_text:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # p-value column to stars
    if "pval" in df.columns:
        df["pval"] = df["pval"].map(_fmt_p)

    styler = (df.style
                .hide(axis="index")
                .set_caption(caption)
                .set_table_styles([
                    {"selector": "caption", "props": [("font-size", "14px"), 
                                                      ("font-weight", "600"), 
                                                      ("margin", "0 0 8px 0"),
                                                      ("white-space", "nowrap"), 
                                                      ("overflow", "hidden"),     
                                                      ]}
                ])
             )

    # Optional highlight by significance
    if highlight_sig_on and highlight_sig_on in df.columns:
        # If pval already formatted as text, we can’t threshold. Skip.
        pass

    # Align
    styler = styler.set_properties(**{"text-align": "right"})
    if "phase" in df.columns:
        styler = styler.set_properties(subset=["phase"], **{"text-align": "left"})
    if "contrast" in df.columns:
        styler = styler.set_properties(subset=["contrast"], **{"text-align": "left"})
    if "metric" in df.columns:
        styler = styler.set_properties(subset=["metric"], **{"text-align": "left"})

    return styler

# ---------- prettifiers for our four tables ----------
def pretty_rate_phase(rate_phase_tbl):
    df = rate_phase_tbl.rename(columns={
        "phase": "Phase",
        "n": "Sessions",
        "rate_per_min_mean": "Rate (per min)",
        "rate_per_min_lo": "95% CI low",
        "rate_per_min_hi": "95% CI high"
    })
    # Build CI text column
    df["95% CI"] = [
        _fmt_ci(lo, hi, digits=3, pct=False)
        for lo, hi in zip(df["95% CI low"], df["95% CI high"])
    ]
    df = df.drop(columns=["95% CI low","95% CI high"])
    return _style(
        df[["Phase","Sessions","Rate (per min)","95% CI"]],
        caption="Early vs Late: Reward rate (session means ± 95% CI)",
        round_map={"Rate (per min)": 3}
    )

def pretty_rate_ttest(rate_ttest_tbl):
    df = rate_ttest_tbl.rename(columns={
        "contrast": "Contrast",
        "diff_rate_per_min": "Δ Rate/min (late−early)",
        "rate_ratio_late_over_early": "Rate Ratio (late/early)",
        "percent_change_late_vs_early": "% Change (late vs early)",
        "t_stat": "t",
        "pval": "pval"
    })
    return _style(
        df,
        caption="Welch t-test on session rates",
        round_map={"Δ Rate/min (late−early)": 3, "Rate Ratio (late/early)": 3, "t": 2},
        percent_cols=["% Change (late vs early)"]
    )

def pretty_rate_glm(rate_glm_tbl):
    df = rate_glm_tbl.rename(columns={
        "contrast": "Contrast",
        "rate_ratio_GLMPoisson": "GLM Rate Ratio",
        "RR_95CI_low": "RR low",
        "RR_95CI_high": "RR high",
        "pval": "pval"
    })
    df["95% CI"] = [_fmt_ci(lo, hi, digits=3) for lo, hi in zip(df["RR low"], df["RR high"])]
    df = df.drop(columns=["RR low","RR high"])
    return _style(
        df[["Contrast","GLM Rate Ratio","95% CI","pval"]],
        caption="Poisson GLM with offset(time): Late vs Early",
        round_map={"GLM Rate Ratio": 3}
    )

def pretty_rate_effect(rate_effect_tbl):
    df = rate_effect_tbl.rename(columns={
        "metric": "Metric",
        "descriptive_ratio_late_over_early": "Descriptive Ratio",
        "descriptive_percent_change": "Descriptive % Change",
        "GLM_rate_ratio": "GLM Rate Ratio",
        "GLM_95CI": "GLM 95% CI",
        "GLM_pval": "pval"
    })
    return _style(
        df[["Metric","Descriptive Ratio","Descriptive % Change","GLM Rate Ratio","GLM 95% CI","pval"]],
        caption="Summary: Reward rate (descriptive vs GLM)",
        round_map={"Descriptive Ratio": 3, "GLM Rate Ratio": 3},
        percent_cols=["Descriptive % Change"],
        ci_cols_as_text=["GLM 95% CI"]
    )

def pretty_dur_phase(dur_phase_tbl):
    df = dur_phase_tbl.rename(columns={
        "phase": "Phase",
        "n": "Sessions",
        "geomT_mean": "Typical duration (s)",
        "geomT_lo": "95% CI low",
        "geomT_hi": "95% CI high"
    })
    df["95% CI"] = [_fmt_ci(lo, hi, digits=1, pct=False) for lo, hi in zip(df["95% CI low"], df["95% CI high"])]
    df = df.drop(columns=["95% CI low","95% CI high"])
    return _style(
        df[["Phase","Sessions","Typical duration (s)","95% CI"]],
        caption="Early vs Late: Duration (geometric mean ± 95% CI)",
        round_map={"Typical duration (s)": 1}
    )

def pretty_dur_ttest(dur_ttest_tbl):
    df = dur_ttest_tbl.rename(columns={
        "contrast": "Contrast",
        "diff_seconds": "Δ Seconds (late−early)",
        "ratio_late_over_early": "Ratio (late/early)",
        "percent_change_late_vs_early": "% Change (late vs early)",
        "t_stat": "t",
        "pval": "pval"
    })
    return _style(
        df,
        caption="Welch t-test on per-session geometric mean durations",
        round_map={"Δ Seconds (late−early)": 1, "Ratio (late/early)": 3, "t": 2},
        percent_cols=["% Change (late vs early)"]
    )

def pretty_dur_glm(dur_glm_tbl):
    df = dur_glm_tbl.rename(columns={
        "contrast": "Contrast",
        "percent_change_duration_OLS": "OLS % Change",
        "pct_95CI_low": "CI low %",
        "pct_95CI_high": "CI high %",
        "pval": "pval"
    })
    df["95% CI"] = [_fmt_ci(lo, hi, digits=1, pct=True, signed=True) for lo, hi in zip(df["CI low %"], df["CI high %"])]
    df = df.drop(columns=["CI low %","CI high %"])
    return _style(
        df[["Contrast","OLS % Change","95% CI","pval"]],
        caption="OLS on log(duration) with cluster-robust SE: Late vs Early",
        percent_cols=["OLS % Change"],
        ci_cols_as_text=["95% CI"]
    )

def pretty_dur_effect(dur_effect_tbl):
    df = dur_effect_tbl.rename(columns={
        "metric": "Metric",
        "descriptive_ratio_late_over_early": "Descriptive Ratio",
        "descriptive_percent_change": "Descriptive % Change",
        "OLS_percent_change": "OLS % Change",
        "OLS_95CI": "OLS 95% CI",
        "OLS_pval": "pval"
    })
    return _style(
        df[["Metric","Descriptive Ratio","Descriptive % Change","OLS % Change","OLS 95% CI","pval"]],
        caption="Summary: Duration (descriptive vs OLS)",
        round_map={"Descriptive Ratio": 3},
        percent_cols=["Descriptive % Change","OLS % Change"],
        ci_cols_as_text=["OLS 95% CI"]
    )

# ---------- terminal fallback (optional) ----------
def print_table_terminal(df: pd.DataFrame, floatfmt=".3f"):
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=floatfmt))
    except Exception:
        # Safe fallback
        with pd.option_context("display.float_format", lambda v: f"{v:{floatfmt}}"):
            print(df.to_string(index=False))

from IPython.display import display, HTML

def show_all_pretty_tables(
    rate_phase, rate_ttest, rate_glm, rate_effect,
    dur_phase,  dur_ttest,  dur_glm,  dur_effect
):
    display(pretty_rate_phase(rate_phase))
    display(pretty_rate_ttest(rate_ttest))
    display(pretty_rate_glm(rate_glm))
    display(pretty_rate_effect(rate_effect))

    display(pretty_dur_phase(dur_phase))
    display(pretty_dur_ttest(dur_ttest))
    display(pretty_dur_glm(dur_glm))
    display(pretty_dur_effect(dur_effect))

# after you compute the tables:
# (rate_phase, rate_ttest, rate_glm, rate_effect) = summarize_early_late_rate_with_glm(df_sessions)
# (dur_phase,  dur_ttest,  dur_glm,  dur_effect ) = summarize_early_late_duration_with_glm(df_trials, df_sessions)
# show_all_pretty_tables(rate_phase, rate_ttest, rate_glm, rate_effect,
#                        dur_phase,  dur_ttest,  dur_glm,  dur_effect)
