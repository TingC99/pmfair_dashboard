import os
import math
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Optional: capture plotly click events (needed for badge click)
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

import re
from datetime import datetime, time
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe

import plotly.graph_objects as go

# APP_VERSION = "v14.8 (Interactive: click variant row to highlight in DFG with instant jump)"



# -----------------------------
# Helpers
# -----------------------------
def _pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Cannot find any of {candidates}. Available cols: {list(cols)[:30]} ...")


@st.cache_data(show_spinner=True)
def load_xes_as_tables(xes_path: str):
    """Load XES -> (events_df, cases_df). Ensure cases columns are unique."""
    log = xes_importer.apply(xes_path)
    df = convert_to_dataframe(log)

    case_col = _pick_col(df.columns, ["case:concept:name", "case_id", "case"])
    act_col = _pick_col(df.columns, ["concept:name", "activity"])
    ts_col = _pick_col(df.columns, ["time:timestamp", "timestamp", "time"])

    # events
    events = df[[case_col, act_col, ts_col]].copy()
    events.columns = ["case_id", "activity", "ts"]
    events["ts"] = pd.to_datetime(events["ts"], errors="coerce")
    events = events.dropna(subset=["case_id", "activity", "ts"]).sort_values(["case_id", "ts"])

    # cases (dedupe + unique column names)
    case_attr_cols = [c for c in df.columns if c.startswith("case:") and c != case_col]
    cases = df[[case_col] + case_attr_cols].drop_duplicates(subset=[case_col]).copy()
    cases = cases.rename(columns={case_col: "case_id"})

    new_cols = ["case_id"]
    seen = {"case_id"}
    for c in cases.columns[1:]:
        name = c.replace("case:", "", 1)
        if name in seen:
            base = name
            i = 2
            name = f"{base}_{i}"
            while name in seen:
                i += 1
                name = f"{base}_{i}"
        seen.add(name)
        new_cols.append(name)
    cases.columns = new_cols

    return events, cases


@st.cache_data(show_spinner=True)
def build_edge_case_support(events: pd.DataFrame):
    """
    For each case, compute set of consecutive edges (activity_i -> activity_{i+1}).
    Returns edge_case_df: [case_id, src, dst]
    """
    edge_records = []
    for case_id, grp in events.groupby("case_id", sort=False):
        acts = grp["activity"].tolist()
        if len(acts) < 2:
            continue
        edges = set(zip(acts[:-1], acts[1:]))  # case-level support
        for (a, b) in edges:
            edge_records.append((case_id, a, b))
    return pd.DataFrame(edge_records, columns=["case_id", "src", "dst"])




# -----------------------------
# Variants (Variant Explorer)
# ----------------------------- (Variant Explorer)
# -----------------------------
@st.cache_data(show_spinner=True)
def build_case_variants(events: pd.DataFrame):
    """
    Build trace variants (full activity sequence) at case-level.

    Returns:
      variants_df: [variant_id, k, variant (tuple), variant_str, case_support_all]
      variant_case_df: [case_id, variant_id]

    Notes:
      Uses pandas.factorize to guarantee unique variant_id assignment even if the
      underlying mapping/index would otherwise be non-unique.
    """
    if events is None or events.empty:
        return (
            pd.DataFrame(columns=["variant_id", "k", "variant", "variant_str", "case_support_all"]),
            pd.DataFrame(columns=["case_id", "variant_id"]),
        )

    # Ensure stable ordering within each case
    ev = events.sort_values(["case_id", "ts"], kind="mergesort")[["case_id", "activity"]].copy()

    # Build per-case sequence; coerce to strings to avoid unhashable / NaN edge-cases
    seq = ev.groupby("case_id", sort=False)["activity"].apply(lambda s: tuple(str(x) for x in s.tolist()))
    variant_case = pd.DataFrame({"case_id": seq.index.astype(str), "variant": seq.values})

    if variant_case.empty:
        return (
            pd.DataFrame(columns=["variant_id", "k", "variant", "variant_str", "case_support_all"]),
            pd.DataFrame(columns=["case_id", "variant_id"]),
        )

    # factorize gives codes 0..n-1 and uniques in first-seen order
    codes, uniques = pd.factorize(variant_case["variant"], sort=False)
    variant_case["variant_id"] = codes.astype(int)

    # Case-support per variant_id, aligned to 0..n-1
    support = variant_case.groupby("variant_id", sort=False)["case_id"].nunique()

    variants_df = pd.DataFrame(
        {
            "variant_id": list(range(len(uniques))),
            "k": [int(len(v)) for v in uniques],
            "variant": list(uniques),
            "variant_str": [" → ".join(v) for v in uniques],
            "case_support_all": [int(support.get(i, 0)) for i in range(len(uniques))],
        }
    )

    return variants_df, variant_case[["case_id", "variant_id"]].copy()


@st.cache_data(show_spinner=False)
def compute_variant_disparity(
    variants_df: pd.DataFrame,
    variant_case_df: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    measure: str,
    band_pp: float,
    band_ratio_low: float,
    band_ratio_high: float,
) -> pd.DataFrame:
    """Compute per-variant occurrence disparity (case-level) between target and reference."""
    if variants_df is None or variants_df.empty:
        return pd.DataFrame()
    if variant_case_df is None or variant_case_df.empty:
        return pd.DataFrame()
    if cases is None or cases.empty or sensitive_attr not in cases.columns:
        return pd.DataFrame()

    cc = cases[["case_id", sensitive_attr]].dropna().copy()
    if cc.empty:
        return pd.DataFrame()

    vc = variant_case_df.merge(cc, on="case_id", how="inner").rename(columns={sensitive_attr: "group"})

    n_by_group = cc.groupby(sensitive_attr)["case_id"].nunique().to_dict()
    n_ref = int(n_by_group.get(ref_group, 0))
    n_tgt = int(n_by_group.get(target_group, 0))
    if n_ref == 0 or n_tgt == 0:
        return pd.DataFrame()

    support = (
        vc.groupby(["group", "variant_id"])["case_id"]
        .nunique()
        .reset_index(name="case_support")
    )

    def get_support(grp_val):
        d = support[support["group"] == grp_val][["variant_id", "case_support"]].copy()
        return d.rename(columns={"case_support": "support"})

    ref_s = get_support(ref_group).rename(columns={"support": "support_ref"})
    tgt_s = get_support(target_group).rename(columns={"support": "support_tgt"})

    m = variants_df.merge(ref_s, on="variant_id", how="left").merge(tgt_s, on="variant_id", how="left")
    m = m.fillna(0)

    m["n_ref"] = n_ref
    m["n_tgt"] = n_tgt
    m["p_ref"] = m["support_ref"] / n_ref
    m["p_tgt"] = m["support_tgt"] / n_tgt
    m["delta_pp"] = (m["p_tgt"] - m["p_ref"]) * 100.0
    # Ratio with continuity correction (keeps finite values when a cell is 0 or 100%)
    cc = 0.5
    st_sup = m["support_tgt"].astype(float)
    sr_sup = m["support_ref"].astype(float)
    nt = float(n_tgt)
    nr = float(n_ref)

    need_cc = (st_sup == 0) | (sr_sup == 0) | (st_sup == nt) | (sr_sup == nr)
    # raw ratio (safe when sr_sup>0)
    ratio_raw = (st_sup / nt) / (sr_sup / nr)
    # Katz-style continuity corrected ratio
    ratio_cc = ((st_sup + cc) / (nt + 2 * cc)) / ((sr_sup + cc) / (nr + 2 * cc))
    m["ratio"] = np.where(need_cc, ratio_cc, ratio_raw)

    if measure == "Δpp":
        m["is_unfair"] = m["delta_pp"].abs() > float(band_pp)
        m["metric_value"] = m["delta_pp"]
        m["metric_label"] = m["delta_pp"].map(lambda x: f"{x:+.2f}pp")
        cis = [
            ci_diff_proportion_wald(int(st), int(n_tgt), int(sr), int(n_ref))
            for st, sr in zip(m["support_tgt"], m["support_ref"])
        ]
        m["ci_low"] = [lo * 100.0 for lo, _ in cis]
        m["ci_high"] = [hi * 100.0 for _, hi in cis]
    else:
        m["is_unfair"] = ~m["ratio"].between(float(band_ratio_low), float(band_ratio_high), inclusive="both")
        m["metric_value"] = (m["ratio"] - 1.0).abs()
        m["metric_label"] = m["ratio"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}×")
        cis = [
            ci_ratio_katz(int(st), int(n_tgt), int(sr), int(n_ref))
            for st, sr in zip(m["support_tgt"], m["support_ref"])
        ]
        m["ci_low"] = [lo for lo, _ in cis]
        m["ci_high"] = [hi for _, hi in cis]

    m["_score"] = m["metric_value"].abs() if measure == "Δpp" else m["metric_value"]
    return m


def build_variant_forest_figure(
    var_stats: pd.DataFrame,
    measure: str,
    band_pp: float,
    band_ratio_low: float,
    band_ratio_high: float,
    ref_label: str,
    tgt_label: str,
    top_k: int = 20,
):
    """Forest plot for Top-K variants with CI and fairness band."""
    if var_stats is None or var_stats.empty:
        return None, pd.DataFrame()

    df = var_stats.copy()

    if measure != "Δpp":
        df = df[~df["ratio"].isna()].copy()
        if df.empty:
            return None, pd.DataFrame()

    df = df.sort_values(["_score", "case_support_all"], ascending=[False, False]).head(int(max(1, top_k))).copy()
    if df.empty:
        return None, pd.DataFrame()

    df = df.reset_index(drop=True)
    df["variant_label"] = [f"Variant {i+1}" for i in range(len(df))]

    colors = ["rgba(220,0,0,0.85)" if bool(u) else "rgba(46,125,50,0.85)" for u in df["is_unfair"].tolist()]

    fig = go.Figure()
    # --- pixel-accurate node border & arrow placement ---
    # Plotly marker sizes are in pixels, while node coordinates are in data space (0..1).
    # To place arrow tips on the node border, we approximate each node as an ellipse in data space,
    # whose radii are derived from marker.size and the figure's pixel width/height.
    FIG_H_PX = 650
    FIG_W_PX = int(st.session_state.get("DFG_RENDER_W_PX", 1000))  # keep in sync with st.plotly_chart(use_container_width=False)
    NODE_SIZE_PX = 64  # keep in sync with marker.size
    NODE_R_PX = NODE_SIZE_PX / 2.0

    RX = NODE_R_PX / FIG_W_PX  # node radius in x (data units)
    RY = NODE_R_PX / FIG_H_PX  # node radius in y (data units)

    def _border_dist_along_dir(ux: float, uy: float, rx: float = RX, ry: float = RY) -> float:
        """Distance from center to ellipse border along direction (ux,uy)."""
        denom = math.sqrt((ux / rx) ** 2 + (uy / ry) ** 2) or 1.0
        return 1.0 / denom

    def _px_to_data_along_dir(pad_px: float, ux: float, uy: float, fig_w: float = FIG_W_PX, fig_h: float = FIG_H_PX) -> float:
        """Convert a pixel padding to a data-space distance along direction (ux,uy)."""
        denom = math.sqrt((fig_w * ux) ** 2 + (fig_h * uy) ** 2) or 1.0
        return pad_px / denom


    for i, r in df.iterrows():
        y = r["variant_label"]
        c = colors[i]
        fig.add_trace(
            go.Scatter(
                x=[float(r["ci_low"]), float(r["ci_high"])],
                y=[y, y],
                mode="lines",
                line=dict(width=4, color=c),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if measure == "Δpp":
        xs = df["delta_pp"].astype(float).tolist()
        x_title = "Δpp (percentage points)"
        v0 = 0.0
        band_lines = [(-float(band_pp),), (float(band_pp),)]
        hover_metric = "Δpp"
        hover_fmt = "%{x:+.2f}pp"
    else:
        xs = df["ratio"].astype(float).tolist()
        x_title = "Ratio (target / reference)"
        v0 = 1.0
        band_lines = [(float(band_ratio_low),), (float(band_ratio_high),)]
        hover_metric = "Ratio"
        hover_fmt = "%{x:.2f}×"

    custom = np.stack(
        [
            df["variant_str"].astype(str).to_numpy(),
            df["case_support_all"].astype(int).to_numpy(),
            df["support_tgt"].astype(int).to_numpy(),
            df["n_tgt"].astype(int).to_numpy(),
            df["support_ref"].astype(int).to_numpy(),
            df["n_ref"].astype(int).to_numpy(),
            df["p_tgt"].astype(float).to_numpy(),
            df["p_ref"].astype(float).to_numpy(),
            df["delta_pp"].astype(float).to_numpy(),
            df["ratio"].astype(float).to_numpy(),
            df["ci_low"].astype(float).to_numpy(),
            df["ci_high"].astype(float).to_numpy(),
            df["is_unfair"].astype(bool).to_numpy(),
        ],
        axis=1,
    )

    hovertemplate = (
        "<b>%{y}</b><br>"
        "%{customdata[0]}<br>"
        "Case support (all): %{customdata[1]}<br>"
        f"p(target={tgt_label}): %{{customdata[6]:.3f}} (%{{customdata[2]}}/%{{customdata[3]}})<br>"
        f"p(ref={ref_label}): %{{customdata[7]:.3f}} (%{{customdata[4]}}/%{{customdata[5]}})<br>"
        f"{hover_metric}: <b>{hover_fmt}</b><br>"
        "95% CI: [%{customdata[10]:.2f}, %{customdata[11]:.2f}]"
        + ("pp" if measure == "Δpp" else "×")
        + "<br>"
        "Unfair: %{customdata[12]}<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=df["variant_label"].tolist(),
            mode="markers",
            marker=dict(size=10, color=colors, line=dict(width=1, color="rgba(0,0,0,0.6)")),
            customdata=custom,
            hovertemplate=hovertemplate,
            showlegend=False,
        )
    )

    fig.add_vline(x=v0, line_width=1, line_dash="solid", line_color="rgba(0,0,0,0.55)")
    for (x,) in band_lines:
        fig.add_vline(x=float(x), line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.35)")

    h = 220 + 34 * len(df)
    fig.update_layout(
        height=h,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=x_title, zeroline=False),
        yaxis=dict(title="", autorange="reversed"),
        hovermode="closest",
    )
    return fig, df.copy()

def compute_disparity(
    edge_case_df: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    measure: str,
    band_pp: float,
    band_ratio_low: float,
    band_ratio_high: float,
):
    """
    per-edge:
      p_ref = P(edge present | case in ref_group)
      p_tgt = P(edge present | case in target_group)
      delta_pp = (p_tgt - p_ref) * 100
      ratio = p_tgt / p_ref
    """
    cc = cases[["case_id", sensitive_attr]].dropna().copy()

    ec = edge_case_df.merge(cc, on="case_id", how="inner")
    ec = ec.rename(columns={sensitive_attr: "group"})

    n_by_group = cc.groupby(sensitive_attr)["case_id"].nunique().to_dict()
    n_ref = int(n_by_group.get(ref_group, 0))
    n_tgt = int(n_by_group.get(target_group, 0))
    if n_ref == 0 or n_tgt == 0:
        return pd.DataFrame()

    support = (
        ec.groupby(["group", "src", "dst"])["case_id"]
        .nunique()
        .reset_index(name="case_support")
    )
    overall_support = (
        ec.groupby(["src", "dst"])["case_id"]
        .nunique()
        .reset_index(name="case_support_all")
    )

    def get_support(grp_val):
        d = support[support["group"] == grp_val][["src", "dst", "case_support"]].copy()
        return d.rename(columns={"case_support": "support"})

    ref_s = get_support(ref_group).rename(columns={"support": "support_ref"})
    tgt_s = get_support(target_group).rename(columns={"support": "support_tgt"})

    m = overall_support.merge(ref_s, on=["src", "dst"], how="left").merge(tgt_s, on=["src", "dst"], how="left")
    m = m.fillna(0)

    m["n_ref"] = n_ref
    m["n_tgt"] = n_tgt

    m["p_ref"] = m["support_ref"] / n_ref
    m["p_tgt"] = m["support_tgt"] / n_tgt
    m["delta_pp"] = (m["p_tgt"] - m["p_ref"]) * 100.0
    # Ratio with continuity correction (keeps finite values when a cell is 0 or 100%)
    cc = 0.5
    st_sup = m["support_tgt"].astype(float)
    sr_sup = m["support_ref"].astype(float)
    nt = float(n_tgt)
    nr = float(n_ref)

    need_cc = (st_sup == 0) | (sr_sup == 0) | (st_sup == nt) | (sr_sup == nr)
    # raw ratio (safe when sr_sup>0)
    ratio_raw = (st_sup / nt) / (sr_sup / nr)
    # Katz-style continuity corrected ratio
    ratio_cc = ((st_sup + cc) / (nt + 2 * cc)) / ((sr_sup + cc) / (nr + 2 * cc))
    m["ratio"] = np.where(need_cc, ratio_cc, ratio_raw)

    if measure == "Δpp":
        m["is_unfair"] = (m["delta_pp"].abs() > band_pp)
        m["metric_label"] = m["delta_pp"].map(lambda x: f"{x:+.1f}pp")
        m["metric_value"] = m["delta_pp"]
    else:
        m["is_unfair"] = ~m["ratio"].between(band_ratio_low, band_ratio_high, inclusive="both")
        m["metric_label"] = m["ratio"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}×")
        m["metric_value"] = m["ratio"]

    return m


@st.cache_data(show_spinner=False)
def compute_case_last_activity(events: pd.DataFrame) -> pd.DataFrame:
    """
    Case-level outcome proxy: last activity of each case.
    Returns: [case_id, outcome]
    """
    if events.empty:
        return pd.DataFrame(columns=["case_id", "outcome"])
    last = (
        events.sort_values(["case_id", "ts"])
        .groupby("case_id", sort=False)
        .tail(1)[["case_id", "activity"]]
        .rename(columns={"activity": "outcome"})
    )
    return last


@st.cache_data(show_spinner=False)
def compute_outcome_disparity(
    case_outcome_df: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    measure: str,
    band_pp: float,
    band_ratio_low: float,
    band_ratio_high: float,
) -> pd.DataFrame:
    """
    Compare outcome probabilities between target vs reference group.
    Outcome is defined as each case's last activity.
    Returns per-outcome:
      p_ref, p_tgt, delta_pp, ratio, supports, unfair flag, etc.
    """
    if case_outcome_df.empty:
        return pd.DataFrame()

    cc = cases[["case_id", sensitive_attr]].dropna().copy()

    oc = case_outcome_df.merge(cc, on="case_id", how="inner")
    oc = oc.rename(columns={sensitive_attr: "group"})

    n_by_group = cc.groupby(sensitive_attr)["case_id"].nunique().to_dict()
    n_ref = int(n_by_group.get(ref_group, 0))
    n_tgt = int(n_by_group.get(target_group, 0))
    if n_ref == 0 or n_tgt == 0:
        return pd.DataFrame()

    support = (
        oc.groupby(["group", "outcome"])["case_id"]
        .nunique()
        .reset_index(name="case_support")
    )
    overall_support = (
        oc.groupby(["outcome"])["case_id"]
        .nunique()
        .reset_index(name="case_support_all")
    )

    def get_support(grp_val):
        d = support[support["group"] == grp_val][["outcome", "case_support"]].copy()
        return d.rename(columns={"case_support": "support"})

    ref_s = get_support(ref_group).rename(columns={"support": "support_ref"})
    tgt_s = get_support(target_group).rename(columns={"support": "support_tgt"})

    m = overall_support.merge(ref_s, on=["outcome"], how="left").merge(tgt_s, on=["outcome"], how="left")
    m = m.fillna(0)

    m["n_ref"] = n_ref
    m["n_tgt"] = n_tgt
    m["p_ref"] = m["support_ref"] / n_ref
    m["p_tgt"] = m["support_tgt"] / n_tgt
    m["delta_pp"] = (m["p_tgt"] - m["p_ref"]) * 100.0
    # Ratio with continuity correction (keeps finite values when a cell is 0 or 100%)
    cc = 0.5
    st_sup = m["support_tgt"].astype(float)
    sr_sup = m["support_ref"].astype(float)
    nt = float(n_tgt)
    nr = float(n_ref)

    need_cc = (st_sup == 0) | (sr_sup == 0) | (st_sup == nt) | (sr_sup == nr)
    # raw ratio (safe when sr_sup>0)
    ratio_raw = (st_sup / nt) / (sr_sup / nr)
    # Katz-style continuity corrected ratio
    ratio_cc = ((st_sup + cc) / (nt + 2 * cc)) / ((sr_sup + cc) / (nr + 2 * cc))
    m["ratio"] = np.where(need_cc, ratio_cc, ratio_raw)

    if measure == "Δpp":
        m["is_unfair"] = (m["delta_pp"].abs() > band_pp)
        m["metric_value"] = m["delta_pp"]
        m["metric_label"] = m["delta_pp"].map(lambda x: f"{x:+.2f}pp")
    else:
        m["is_unfair"] = ~m["ratio"].between(band_ratio_low, band_ratio_high, inclusive="both")
        m["metric_value"] = (m["ratio"] - 1.0).abs()
        m["metric_label"] = m["ratio"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}×")

    return m


@st.cache_data(show_spinner=False)
def compute_case_processing_time(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-case processing time as (last timestamp - first timestamp).
    Returns a DataFrame: [case_id, processing_time_days]
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=["case_id", "processing_time_days"])

    df = events[["case_id", "ts"]].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=["case_id", "processing_time_days"])

    g = df.groupby("case_id", as_index=False)["ts"].agg(ts_min="min", ts_max="max")
    g["processing_time_days"] = (g["ts_max"] - g["ts_min"]).dt.total_seconds() / 86400.0
    # Guard against negative durations (shouldn't happen, but keep safe)
    g.loc[g["processing_time_days"] < 0, "processing_time_days"] = np.nan
    return g[["case_id", "processing_time_days"]]


@st.cache_data(show_spinner=False)
def compute_processing_time_disparity(
    case_time_df: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    measure: str,
    band_time_days: float,
    band_ratio_low: float,
    band_ratio_high: float,
    agg: str = "mean",
) -> dict:
    """
    Compare case processing time between target and reference groups.
    - agg: 'mean' or 'median'
    - measure:
        - 'Δpp' -> uses band_time_days on absolute delta in days
        - 'Ratio' -> uses [band_ratio_low, band_ratio_high] on center ratio
    Returns a dict with centers in days, delta_days, ratio, supports, is_unfair.
    """
    if case_time_df is None or case_time_df.empty:
        return {}

    if cases is None or cases.empty or sensitive_attr not in cases.columns:
        return {}

    cc = cases[["case_id", sensitive_attr]].dropna().copy()
    if cc.empty:
        return {}

    df = case_time_df.merge(cc, on="case_id", how="inner")
    df = df.rename(columns={sensitive_attr: "group"})
    df = df.dropna(subset=["processing_time_days"])

    if df.empty:
        return {}

    ref = df[df["group"] == ref_group]["processing_time_days"]
    tgt = df[df["group"] == target_group]["processing_time_days"]

    support_ref = int(ref.shape[0])
    support_tgt = int(tgt.shape[0])
    if support_ref == 0 or support_tgt == 0:
        return {}

    agg = (agg or "mean").lower()
    if agg not in {"mean", "median"}:
        agg = "mean"

    if agg == "median":
        center_ref = float(ref.median())
        center_tgt = float(tgt.median())
    else:
        center_ref = float(ref.mean())
        center_tgt = float(tgt.mean())

    delta_days = center_tgt - center_ref
    ratio = (center_tgt / center_ref) if center_ref > 0 else np.nan

    if measure == "Δpp":
        is_unfair = abs(delta_days) > float(band_time_days)
    else:
        # Ratio measure
        if pd.isna(ratio):
            is_unfair = False
        else:
            is_unfair = not (float(band_ratio_low) <= ratio <= float(band_ratio_high))

    return {
        "center_name": agg,
        "center_ref_days": center_ref,
        "center_tgt_days": center_tgt,
        "delta_days": delta_days,
        "ratio": ratio,
        "support_ref": support_ref,
        "support_tgt": support_tgt,
        "is_unfair": bool(is_unfair),
    }


# -----------------------------
# Confidence intervals + Sparklines (Outcome dashboard)
# -----------------------------
_Z_95 = 1.959963984540054  # approx for 95% CI

def ci_diff_proportion_wald(s_tgt: int, n_tgt: int, s_ref: int, n_ref: int, z: float = _Z_95):
    """Wald CI for difference in proportions (p_tgt - p_ref). Returns (low, high) in proportion units."""
    if n_tgt <= 0 or n_ref <= 0:
        return (float("nan"), float("nan"))
    p_t = s_tgt / n_tgt
    p_r = s_ref / n_ref
    diff = p_t - p_r
    se = math.sqrt((p_t * (1 - p_t)) / max(n_tgt, 1) + (p_r * (1 - p_r)) / max(n_ref, 1))
    low = diff - z * se
    high = diff + z * se
    return (low, high)

def ci_ratio_katz(s_tgt: int, n_tgt: int, s_ref: int, n_ref: int, z: float = _Z_95, cc: float = 0.5):
    """Katz log CI for risk ratio p_tgt/p_ref. Returns (low, high). Adds continuity correction if needed."""
    if n_tgt <= 0 or n_ref <= 0:
        return (float("nan"), float("nan"))
    # continuity correction if any zero cell
    if s_tgt == 0 or s_ref == 0 or s_tgt == n_tgt or s_ref == n_ref:
        s_tgt = s_tgt + cc
        s_ref = s_ref + cc
        n_tgt = n_tgt + 2 * cc
        n_ref = n_ref + 2 * cc

    p_t = s_tgt / n_tgt
    p_r = s_ref / n_ref
    if p_r <= 0 or p_t <= 0:
        return (float("nan"), float("nan"))
    rr = p_t / p_r

    # SE for log(RR): sqrt(1/s_t - 1/n_t + 1/s_r - 1/n_r)
    se_log = math.sqrt(max(0.0, (1.0 / s_tgt) - (1.0 / n_tgt) + (1.0 / s_ref) - (1.0 / n_ref)))
    lo = math.exp(math.log(rr) - z * se_log)
    hi = math.exp(math.log(rr) + z * se_log)
    return (lo, hi)

@st.cache_data(show_spinner=False)
def compute_processing_time_ci(
    case_time_df: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    agg: str = "mean",
    n_boot: int = 400,
    seed: int = 7,
):
    """Bootstrap percentile CI for processing time delta (tgt-ref) and ratio (tgt/ref)."""
    if case_time_df is None or case_time_df.empty:
        return {}
    if cases is None or cases.empty or sensitive_attr not in cases.columns:
        return {}

    cc = cases[["case_id", sensitive_attr]].dropna().copy()
    df = case_time_df.merge(cc, on="case_id", how="inner").rename(columns={sensitive_attr: "group"})
    df = df.dropna(subset=["processing_time_days"])
    ref = df[df["group"] == ref_group]["processing_time_days"].to_numpy(dtype=float)
    tgt = df[df["group"] == target_group]["processing_time_days"].to_numpy(dtype=float)
    if ref.size == 0 or tgt.size == 0:
        return {}

    agg = (agg or "mean").lower()
    def center(x):
        return float(np.nanmedian(x)) if agg == "median" else float(np.nanmean(x))

    rng = np.random.default_rng(seed)

    deltas = []
    ratios = []
    n_ref = ref.size
    n_tgt = tgt.size
    for _ in range(int(max(n_boot, 200))):
        rs = rng.choice(ref, size=n_ref, replace=True)
        ts = rng.choice(tgt, size=n_tgt, replace=True)
        c_r = center(rs)
        c_t = center(ts)
        deltas.append(c_t - c_r)
        ratios.append((c_t / c_r) if c_r > 0 else np.nan)

    deltas = np.array(deltas, dtype=float)
    ratios = np.array(ratios, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    ratios = ratios[~np.isnan(ratios)]
    out = {}
    if deltas.size:
        out["delta_low"] = float(np.percentile(deltas, 2.5))
        out["delta_high"] = float(np.percentile(deltas, 97.5))
    if ratios.size:
        out["ratio_low"] = float(np.percentile(ratios, 2.5))
        out["ratio_high"] = float(np.percentile(ratios, 97.5))
    return out

def _fill_nans_forward(vals):
    vals = list(vals)
    # forward fill
    last = None
    for i in range(len(vals)):
        if vals[i] is None or (isinstance(vals[i], float) and math.isnan(vals[i])):
            vals[i] = last
        else:
            last = vals[i]
    # backward fill
    last = None
    for i in range(len(vals) - 1, -1, -1):
        if vals[i] is None:
            vals[i] = last
        else:
            last = vals[i]
    # final fallback
    return [0.0 if v is None else float(v) for v in vals]

def make_sparkline_svg(values, stroke="#999", fill="rgba(0,0,0,0.06)"):
    """Return an inline SVG sparkline for a list of numeric values."""
    if not values:
        values = [0.0, 0.0]
    values = _fill_nans_forward(values)
    n = len(values)
    w, h = 120, 34
    pad_x, pad_y = 2, 4
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    xs = [pad_x + (w - 2 * pad_x) * i / (n - 1) for i in range(n)]
    ys = [pad_y + (h - 2 * pad_y) * (1 - (v - vmin) / (vmax - vmin)) for v in values]
    pts = " ".join([f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys)])

    # area fill
    area_pts = f"{xs[0]:.2f},{h-pad_y:.2f} " + pts + f" {xs[-1]:.2f},{h-pad_y:.2f}"
    svg = f'''
<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <polyline points="{area_pts}" fill="{fill}" stroke="none" />
  <polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''.strip()
    return svg

@st.cache_data(show_spinner=False)
def compute_offer_reject_series(
    events: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    offer_act: str,
    reject_act: str,
    n_bins: int = 12,
):
    """Compute per-time-bin delta offer/reject rates (pp) based on case end time."""
    if events is None or events.empty:
        return {"offer": [], "reject": []}
    # end activity + end ts per case
    last = (
        events.sort_values(["case_id", "ts"])
        .groupby("case_id", sort=False)
        .tail(1)[["case_id", "activity", "ts"]]
        .rename(columns={"activity": "outcome", "ts": "end_ts"})
    )

    cc = cases[["case_id", sensitive_attr]].dropna().copy()
    df = last.merge(cc, on="case_id", how="inner").rename(columns={sensitive_attr: "group"})
    if df.empty:
        return {"offer": [], "reject": []}

    # bins by time range
    tmin = df["end_ts"].min()
    tmax = df["end_ts"].max()
    if pd.isna(tmin) or pd.isna(tmax) or tmin == tmax:
        return {"offer": [], "reject": []}

    bins = pd.date_range(tmin, tmax, periods=int(max(n_bins, 6)) + 1)
    df["bin"] = pd.cut(df["end_ts"], bins=bins, include_lowest=True)

    out_offer, out_reject = [], []
    for b, g in df.groupby("bin", sort=True):
        n_ref = int((g["group"] == ref_group).sum())
        n_tgt = int((g["group"] == target_group).sum())
        if n_ref < 3 or n_tgt < 3:
            out_offer.append(np.nan)
            out_reject.append(np.nan)
            continue

        sup_offer_ref = int(((g["group"] == ref_group) & (g["outcome"] == offer_act)).sum())
        sup_offer_tgt = int(((g["group"] == target_group) & (g["outcome"] == offer_act)).sum())
        p_offer_ref = sup_offer_ref / n_ref
        p_offer_tgt = sup_offer_tgt / n_tgt
        out_offer.append((p_offer_tgt - p_offer_ref) * 100.0)

        sup_rej_ref = int(((g["group"] == ref_group) & (g["outcome"] == reject_act)).sum())
        sup_rej_tgt = int(((g["group"] == target_group) & (g["outcome"] == reject_act)).sum())
        p_rej_ref = sup_rej_ref / n_ref
        p_rej_tgt = sup_rej_tgt / n_tgt
        out_reject.append((p_rej_tgt - p_rej_ref) * 100.0)

    return {"offer": out_offer, "reject": out_reject}

@st.cache_data(show_spinner=False)
def compute_processing_time_series(
    events: pd.DataFrame,
    cases: pd.DataFrame,
    sensitive_attr: str,
    ref_group,
    target_group,
    agg: str = "mean",
    n_bins: int = 12,
):
    """Compute per-time-bin delta processing time (days) based on case end time."""
    if events is None or events.empty:
        return []
    # end ts per case
    last_ts = (
        events.sort_values(["case_id", "ts"])
        .groupby("case_id", sort=False)
        .tail(1)[["case_id", "ts"]]
        .rename(columns={"ts": "end_ts"})
    )
    case_time = compute_case_processing_time(events)
    df = case_time.merge(last_ts, on="case_id", how="inner")

    cc = cases[["case_id", sensitive_attr]].dropna().copy()
    df = df.merge(cc, on="case_id", how="inner").rename(columns={sensitive_attr: "group"})
    df = df.dropna(subset=["processing_time_days", "end_ts"])
    if df.empty:
        return []

    tmin = df["end_ts"].min()
    tmax = df["end_ts"].max()
    if pd.isna(tmin) or pd.isna(tmax) or tmin == tmax:
        return []

    bins = pd.date_range(tmin, tmax, periods=int(max(n_bins, 6)) + 1)
    df["bin"] = pd.cut(df["end_ts"], bins=bins, include_lowest=True)

    agg = (agg or "mean").lower()
    def center(x):
        return float(np.nanmedian(x)) if agg == "median" else float(np.nanmean(x))

    out = []
    for b, g in df.groupby("bin", sort=True):
        ref = g[g["group"] == ref_group]["processing_time_days"].to_numpy(dtype=float)
        tgt = g[g["group"] == target_group]["processing_time_days"].to_numpy(dtype=float)
        if ref.size < 3 or tgt.size < 3:
            out.append(np.nan)
            continue
        out.append(center(tgt) - center(ref))
    return out


@st.cache_data(show_spinner=False)
def compute_node_layout(events: pd.DataFrame, n_layers: int = 20, margin: float = 0.08):
    """
    计算节点布局 - 使用基于BFS的分层布局
    
    这个新方法：
    1. 构建有向图
    2. 从起始节点开始BFS
    3. 计算每个节点的"层"（距离起始节点的距离）
    4. 按层排列节点，减少线条交叉
    """
    pos = {}
    freq = {}
    edges = []  # 存储有向边 (src, dst)

    # 第1步：计算平均位置和频率，收集边
    for _, grp in events.groupby("case_id", sort=False):
        acts = grp["activity"].tolist()
        L = len(acts)
        if L == 0:
            continue
        denom = max(L - 1, 1)
        for i, a in enumerate(acts):
            pos.setdefault(a, []).append(i / denom)
            freq[a] = freq.get(a, 0) + 1
        
        # 收集有向边
        for i in range(len(acts) - 1):
            edges.append((acts[i], acts[i+1]))

    avg_pos = {a: float(np.mean(v)) for a, v in pos.items()}
    if not avg_pos:
        return [], {}, {}

    # 第2步：构建邻接表
    from collections import defaultdict, deque
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set(avg_pos.keys())
    
    for node in all_nodes:
        in_degree[node] = 0
    
    for src, dst in edges:
        if src in all_nodes and dst in all_nodes:
            graph[src].append(dst)
            in_degree[dst] += 1
    
    # 第3步：使用BFS计算节点的层级（拓扑排序方式）
    node_level = {}
    queue = deque()
    
    # 找到起始节点（入度为0或起始概率最低的）
    start_nodes = [n for n in all_nodes if in_degree[n] == 0]
    if not start_nodes:
        start_nodes = sorted(all_nodes, key=lambda x: avg_pos.get(x, 0))[:1]
    
    for start in start_nodes:
        node_level[start] = 0
        queue.append(start)
    
    # BFS遍历
    visited = set(start_nodes)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                node_level[neighbor] = node_level.get(node, 0) + 1
                queue.append(neighbor)
    
    # 对于没有被访问的节点，设置默认层级（基于起始位置）
    for node in all_nodes:
        if node not in node_level:
            node_level[node] = int(round(avg_pos[node] * (n_layers - 1)))

    # 第4步：按层级分组
    max_level = max(node_level.values()) if node_level else 0
    n_actual_layers = min(max_level + 1, n_layers)
    
    layers = {i: [] for i in range(n_actual_layers)}
    for node, level in node_level.items():
        level_idx = min(level, n_actual_layers - 1)
        layers[level_idx].append(node)

    # 在每层内按频率排序
    for i in range(n_actual_layers):
        layers[i].sort(key=lambda a: (-freq.get(a, 0), str(a)))

    # 第5步：计算x坐标（按层级均匀分布）
    x_norm = {}
    for i in range(n_actual_layers):
        x = 0.05 + 0.90 * (i / (n_actual_layers - 1)) if n_actual_layers > 1 else 0.5
        for a in layers[i]:
            x_norm[a] = x

    # 第6步：计算y坐标（在每层内均匀分布）
    y_norm = {}
    for i in range(n_actual_layers):
        nodes = layers[i]
        k = len(nodes)
        if k == 0:
            continue
        for j, a in enumerate(nodes):
            y_norm[a] = margin + (j + 0.5) * (1 - 2 * margin) / k

    # 第7步：生成排序后的节点列表
    nodes_sorted = []
    for i in range(n_actual_layers):
        nodes_sorted.extend(layers[i])

    return nodes_sorted, x_norm, y_norm

@st.cache_data(show_spinner=False)
def build_code_maps(events: pd.DataFrame):
    """Stable code map based on global activity order (avg position)."""
    nodes_all, _, _ = compute_node_layout(events)  # 你已有这个函数
    codes = make_codes(len(nodes_all))
    act_to_code = {a: c for a, c in zip(nodes_all, codes)}
    code_to_act = {c: a for a, c in act_to_code.items()}
    return act_to_code, code_to_act, nodes_all


def parse_code_input(s: str, code_to_act: dict):
    """
    Accept: 'ACD' or 'A,C,D' or 'A C D' .
    For multi-letter codes like 'AA', use comma/space separated.
    """
    s = (s or "").strip().upper()
    if not s:
        return []

    if re.search(r"[,\s;]", s):
        parts = [p for p in re.split(r"[,\s;]+", s) if p]
    else:
        parts = list(s)  # treat as A C D

    acts = []
    for p in parts:
        if p in code_to_act:
            acts.append(code_to_act[p])
    return acts

def make_codes(n: int):
    """A..Z, AA..AZ, BA..BZ ..."""
    codes = []
    i = 0
    while len(codes) < n:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = x // 26 - 1
            if x < 0:
                break
        codes.append(s)
        i += 1
    return codes


def wrap_label(text: str, max_chars: int = 12, max_lines: int = 3) -> str:
    """
    Wrap label into multiple lines so it fits inside a circular node.
    Uses <br> for Plotly line breaks.

    Parameters
    ----------
    max_chars : int
        Approx. maximum characters per line.
    max_lines : int
        Maximum number of lines to render (extra lines are truncated with an ellipsis).
    """
    s = str(text).strip()
    if not s:
        return ""

    lines = textwrap.wrap(
        s,
        width=max_chars,
        break_long_words=True,
        break_on_hyphens=False,
    )

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        # add ellipsis on last line
        last = lines[-1]
        if len(last) >= max_chars:
            lines[-1] = last[: max(1, max_chars - 1)] + "…"
        else:
            lines[-1] = last + "…"

    return "<br>".join(lines)


def build_dfg_figure(
    edge_stats: pd.DataFrame,
    nodes_sorted,
    x_norm,
    y_norm,
    min_support: int,
    max_edges: int,
    measure: str,
    ref_label: str,
    tgt_label: str,
    highlight_activity=None,
    pinned_popup: dict | None = None,
    badge_t: float = 0.45,
    badge_spread: float = 0.020,
    highlight_variant_edges: set | None = None,
):
    """
    Build a DFG (Directly-Follows Graph) view using Plotly.
    - Edge width = case frequency (case_support_all)
    - Unfair edges = red
    - Badge (!) = yellow marker placed on the edge (click-to-select)
    - Optional pinned_popup renders a Plotly annotation near the clicked badge
    Returns: fig, badge_df (rows corresponding to badges)
    """
    if edge_stats.empty:
        return None, pd.DataFrame()

    df = edge_stats[edge_stats["case_support_all"] >= min_support].copy()
    df = df.sort_values("case_support_all", ascending=False).head(max_edges)
    if df.empty:
        return None, pd.DataFrame()

    # focus (dim non-focus edges)
    if highlight_activity:
        df["is_focus"] = (df["src"] == highlight_activity) | (df["dst"] == highlight_activity)
    else:
        df["is_focus"] = True

    # keep only nodes that exist in layout
    node_set = set(nodes_sorted)
    df = df[df["src"].isin(node_set) & df["dst"].isin(node_set)].copy()
    if df.empty:
        return None, pd.DataFrame()

    fig = go.Figure()
    # --- pixel-accurate node border & arrow placement ---
    # Plotly marker sizes are in pixels, while node coordinates are in data space (0..1).
    # To place arrow tips on the node border, we approximate each node as an ellipse in data space,
    # whose radii are derived from marker.size and the figure's pixel width/height.
    FIG_H_PX = 650
    FIG_W_PX = int(st.session_state.get("DFG_RENDER_W_PX", 1000))  # keep in sync with st.plotly_chart(use_container_width=False)
    NODE_SIZE_PX = 64  # keep in sync with marker.size
    NODE_R_PX = NODE_SIZE_PX / 2.0

    RX = NODE_R_PX / FIG_W_PX  # node radius in x (data units)
    RY = NODE_R_PX / FIG_H_PX  # node radius in y (data units)

    def _border_dist_along_dir(ux: float, uy: float, rx: float = RX, ry: float = RY) -> float:
        """Distance from center to ellipse border along direction (ux,uy)."""
        denom = math.sqrt((ux / rx) ** 2 + (uy / ry) ** 2) or 1.0
        return 1.0 / denom

    def _px_to_data_along_dir(pad_px: float, ux: float, uy: float, fig_w: float = FIG_W_PX, fig_h: float = FIG_H_PX) -> float:
        """Convert a pixel padding to a data-space distance along direction (ux,uy)."""
        denom = math.sqrt((fig_w * ux) ** 2 + (fig_h * uy) ** 2) or 1.0
        return pad_px / denom


    # --- edges (one trace per edge so we can control width & hover precisely)
    # 处理三种情况：1. 自循环（self-loop）2. 双向边（bidirectional）3. 单向边（unidirectional）
    
    edge_set = set(zip(df["src"], df["dst"]))
    arrows_to_draw = []  # 存储需要绘制的箭头
    
    for _, r in df.iterrows():
        s, d = r["src"], r["dst"]
        x0, y0 = float(x_norm.get(s, 0.5)), float(y_norm.get(s, 0.5))
        x1, y1 = float(x_norm.get(d, 0.5)), float(y_norm.get(d, 0.5))

        # 获取频率信息（用于显示，不用于线宽计算）
        w = float(r["case_support_all"])
        width = 2.0  # 统一的线宽

        # Check if this edge is part of the selected variant
        is_variant_edge = (highlight_variant_edges and (s, d) in highlight_variant_edges)

        if is_variant_edge:
            alpha = 0.90 if bool(r["is_focus"]) else 0.30
            color = f"rgba(0,100,255,{alpha})"
        elif bool(r["is_unfair"]):
            alpha = 0.75 if bool(r["is_focus"]) else 0.10
            color = f"rgba(220,0,0,{alpha})"
        else:
            alpha = 0.40 if bool(r["is_focus"]) else 0.08
            color = f"rgba(120,120,120,{alpha})"

        edge_text = f"{s} → {d}"

        # customdata for hovertemplate - 【修复1】：正确的索引
        if measure == "Δpp":
            metric_val = float(r["delta_pp"])
            metric_label = "Δpp"
            metric_fmt = "%{customdata[3]:+.1f}pp"
        else:
            metric_val = float(r["ratio"]) if not pd.isna(r["ratio"]) else float("nan")
            metric_label = "Ratio"
            metric_fmt = "%{customdata[3]:.2f}×"

        row = [
            edge_text,                    # [0]
            float(r["p_tgt"]),           # [1]
            float(r["p_ref"]),           # [2]
            metric_val,                  # [3]
            float(r["support_tgt"]),     # [4]
            float(r["support_ref"]),     # [5]
            float(r["n_tgt"]),           # [6]
            float(r["n_ref"]),           # [7]
            bool(r["is_unfair"]),        # [8]
            w,                           # [9] - case_support_all
        ]

        # 【修复1】：正确的 hovertemplate 索引
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Case frequency: %{customdata[9]:.0f}<br>"
            f"p(target={tgt_label}): %{{customdata[1]:.3f}} (%{{customdata[4]:.0f}}/%{{customdata[6]:.0f}})<br>"
            f"p(ref={ref_label}): %{{customdata[2]:.3f}} (%{{customdata[5]:.0f}}/%{{customdata[7]:.0f}})<br>"
            f"{metric_label}: <b>{metric_fmt}</b><br>"
            "Unfair: %{customdata[8]}<extra></extra>"
        )
        
        # ========== 情况 1：自循环 (Self-loop) ==========
        if s == d:
            # 绘制自循环为一个圆形弧
            node_size = 64  # 节点大小
            radius = 0.06  # 循环半径（相对坐标）
            
            # 绘制圆形（使用多个点）
            angles = np.linspace(0, 2 * np.pi, 100)
            loop_x = [x0 + radius * np.cos(a) for a in angles]
            loop_y = [y0 + radius/2 + radius * np.sin(a) for a in angles]
            
            # 【关键】：确保 customdata 的长度与 x, y 一致
            fig.add_trace(
                go.Scatter(
                    x=loop_x,
                    y=loop_y,
                    mode="lines",
                    line=dict(width=width, color=color),
                    customdata=[row] * len(loop_x),  # ✓ 确保每个点都有数据
                    hovertemplate=hovertemplate,
                    showlegend=False,
                    name="",
                )
            )
            
            # 【修复2】：自循环箭头位置改为向下（270°）
            arrow_x = x0 + radius * 1
            arrow_y = y0 + radius/2
            arrows_to_draw.append({
                "x": arrow_x,
                "y": arrow_y,
                "color": color,
                "angle": 270,  # ✓ 改为 270°（向下）
            })
        else:
            # ========== 情况 2 & 3：正常边（双向或单向） ==========
            reverse_exists = (d, s) in edge_set
            
            # 计算从A到B的方向
            dx = x1 - x0
            dy = y1 - y0
            edge_length = math.sqrt(dx*dx + dy*dy) or 1.0
            edge_angle = math.atan2(dy, dx)
            ux = dx / edge_length
            uy = dy / edge_length

            # small pads (in pixels). End pad can be slightly negative if arrow looks too far from the node.
            START_PAD_PX = 0
            END_PAD_PX = -2  # move arrow tip 2px closer into the node border (tune: -4..+2)

            t0 = _border_dist_along_dir(ux, uy) + _px_to_data_along_dir(START_PAD_PX, ux, uy)
            t1 = _border_dist_along_dir(ux, uy) + _px_to_data_along_dir(END_PAD_PX, ux, uy)

            # start at src border, end at dst border (t1 may pull slightly closer)
            line_start_x = x0 + t0 * ux
            line_start_y = y0 + t0 * uy
            line_end_x   = x1 - t1 * ux
            line_end_y   = y1 - t1 * uy

            if reverse_exists:
                # 双向边：使用曲线
                mid_x = (line_start_x + line_end_x) / 2
                mid_y = (line_start_y + line_end_y) / 2
                
                # 计算垂直方向（偏移控制点）
                perp_x = -dy / edge_length
                perp_y = dx / edge_length
                
                offset = 0.04
                curve_x = mid_x + perp_x * offset
                curve_y = mid_y + perp_y * offset
                
                # 生成平滑的二次贝塞尔曲线（50个点）
                t_values = np.linspace(0, 1, 50)
                bezier_x = []
                bezier_y = []
                for t in t_values:
                    bx = (1-t)**2 * line_start_x + 2*(1-t)*t * curve_x + t**2 * line_end_x
                    by = (1-t)**2 * line_start_y + 2*(1-t)*t * curve_y + t**2 * line_end_y
                    bezier_x.append(bx)
                    bezier_y.append(by)
                
                # 【修复1】：确保 customdata 长度与曲线点数一致
                fig.add_trace(
                    go.Scatter(
                        x=bezier_x,
                        y=bezier_y,
                        mode="lines",
                        line=dict(width=width, color=color, shape="spline"),
                        customdata=[row] * len(bezier_x),  # ✓ 对应曲线上的每一个点
                        hovertemplate=hovertemplate,
                        showlegend=False,
                        name="",
                    )
                )
                
                # 箭头位置：就在B节点边缘处
                angle = math.degrees(edge_angle)
                arrow_x = line_end_x
                arrow_y = line_end_y
            else:
                # 单向边：使用平滑曲线（即使是直线也用平滑渲染）
                # 【修复1】：确保 hover 信息完整
                fig.add_trace(
                    go.Scatter(
                        x=[line_start_x, line_end_x],
                        y=[line_start_y, line_end_y],
                        mode="lines",
                        line=dict(width=width, color=color, shape="spline"),
                        customdata=[row, row],  # ✓ 两个端点都有数据
                        hovertemplate=hovertemplate,
                        showlegend=False,
                        name="",
                    )
                )
                
                # 箭头位置：就在B节点边缘处
                angle = math.degrees(edge_angle)
                arrow_x = line_end_x
                arrow_y = line_end_y
            
            arrows_to_draw.append({
                "x": arrow_x,
                "y": arrow_y,
                "color": color,
                "angle": angle,
            })
    
    # ========== 绘制所有箭头 ==========
    # 使用 Plotly annotation 的 arrowhead 来绘制细致的箭头
    # 这样箭头看起来更专业、更标准
    for arrow_info in arrows_to_draw:
        ax, ay = arrow_info["x"], arrow_info["y"]
        angle = arrow_info["angle"]
        color = arrow_info["color"]
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 箭头长度（向后延伸）：用像素长度换算到 data 单位，保证不同尺寸下视觉一致
        ARROW_LEN_PX = 8
        arrow_length = _px_to_data_along_dir(ARROW_LEN_PX, cos_a, sin_a)
        # 箭头应该在线的末端显示（当前的ax,ay是线的终点）
        # 箭头尾部向后偏移
        start_x = ax - arrow_length * cos_a
        start_y = ay - arrow_length * sin_a
        
        # 使用 Plotly annotation 绘制箭头
        # 注意：不显示文本，只显示箭头
        fig.add_annotation(
            x=ax,  # 箭头尖端（线的末端）
            y=ay,
            ax=start_x,  # 箭头尾部（线上稍微前面的位置）
            ay=start_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",  # 不显示文本
            arrowhead=2,  # 标准箭头样式
            arrowsize=0.8,  # 箭头大小倍数
            arrowwidth=1.5,  # 箭头线条宽度
            arrowcolor=color,  # 箭头颜色与边一致
            showarrow=True,
        )


    
    # --- nodes (activity names)
    NODE_SIZE = 64
    FONT_SIZE = 9
    MAX_CHARS = max(6, int(NODE_SIZE / (FONT_SIZE * 0.62)))
    node_text = [wrap_label(a, max_chars=MAX_CHARS, max_lines=3) for a in nodes_sorted]

    fig.add_trace(
        go.Scatter(
            x=[float(x_norm.get(a, 0.5)) for a in nodes_sorted],
            y=[float(y_norm.get(a, 0.5)) for a in nodes_sorted],
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            textfont=dict(size=FONT_SIZE),
            marker=dict(size=NODE_SIZE, color="rgba(235,235,235,1.0)", line=dict(width=1, color="rgba(80,80,80,0.6)")),
            customdata=nodes_sorted,
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            showlegend=False,
        )
    )

    # --- badges for unfair edges (yellow !), placed on the edge
    if highlight_activity:
        badge_df = df[df["is_unfair"] & df["is_focus"]].copy()
    else:
        badge_df = df[df["is_unfair"]].copy()

    badge_df = badge_df.reset_index(drop=True)

    if not badge_df.empty:
        out_rank = {}
        out_count = {}
        for src, g in badge_df.groupby("src"):
            dsts = g["dst"].tolist()
            dsts_sorted = sorted(dsts, key=lambda dst: (y_norm.get(dst, 0.5), str(dst)))
            out_rank[src] = {dst: i for i, dst in enumerate(dsts_sorted)}
            out_count[src] = len(dsts_sorted)

        bx, by, btext, bid = [], [], [], []
        t = float(min(max(badge_t, 0.05), 0.95))
        spread = float(max(badge_spread, 0.0))

        for i, r in badge_df.iterrows():
            s, d = r["src"], r["dst"]
            x0, y0 = float(x_norm.get(s, 0.5)), float(y_norm.get(s, 0.5))
            x1, y1 = float(x_norm.get(d, 0.5)), float(y_norm.get(d, 0.5))

            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)

            vx, vy = (x1 - x0), (y1 - y0)
            norm = math.sqrt(vx * vx + vy * vy) or 1.0
            nx, ny = (-vy / norm, vx / norm)

            k = out_count.get(s, 1)
            j = out_rank.get(s, {}).get(d, 0)
            offset = (j - (k - 1) / 2) * spread

            x = x + nx * offset
            y = y + ny * offset

            x = float(min(max(x, 0.02), 0.98))
            y = float(min(max(y, 0.02), 0.98))

            bx.append(x)
            by.append(y)
            btext.append("!")
            bid.append(int(i))

        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="markers+text",
                text=btext,
                textposition="middle center",
                textfont=dict(color="black", size=10),
                marker=dict(size=18, color="rgba(255,215,0,0.95)", line=dict(width=1, color="black")),
                customdata=bid,
                meta="badge",
                hovertemplate="Click badge to pin details<extra></extra>",
                showlegend=False,
            )
        )

    # --- pinned popup annotation (click badge -> show next to badge)
    if pinned_popup and isinstance(pinned_popup, dict):
        try:
            px = float(pinned_popup.get("x"))
            py = float(pinned_popup.get("y"))
            txt = str(pinned_popup.get("text", ""))
            fig.add_annotation(
                x=px,
                y=py,
                xref="x",
                yref="y",
                text=txt,
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-40,
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="rgba(220,0,0,0.55)",
                borderwidth=1,
                borderpad=6,
                align="left",
                font=dict(size=12, color="black"),
            )
        except Exception:
            pass

    # NOTE:
    # - Do NOT set a fixed width here; Streamlit containers can be narrower than the assumed pixel width.
    #   A fixed width would get clipped on the right.
    # - Keep an assumed pixel width only for arrow/node geometry; the actual rendering will be responsive.
    EXTRA_PX = 26
    pad_x = RX + (EXTRA_PX / max(FIG_W_PX, 1))
    pad_y = RY + (EXTRA_PX / max(FIG_H_PX, 1))

        # Use the actual node coordinate extents (not fixed [0,1]) to avoid clipping on the right/left.
    x_vals = [float(x_norm.get(a, 0.5)) for a in nodes_sorted]
    y_vals = [float(y_norm.get(a, 0.5)) for a in nodes_sorted]
    x_min, x_max = (min(x_vals), max(x_vals)) if x_vals else (0.0, 1.0)
    y_min, y_max = (min(y_vals), max(y_vals)) if y_vals else (0.0, 1.0)

    fig.update_layout(
        autosize=False,
        width=FIG_W_PX,
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[x_min - pad_x, x_max + pad_x]),
        yaxis=dict(visible=False, range=[y_min - pad_y, y_max + pad_y]),
        clickmode="event+select",
        hovermode="closest",
    )
    return fig, badge_df, df
# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PM Fairness Prototype", layout="wide")
st.markdown(f"## PM Fairness Dashboard (Prototype test)")

# Fixed render width is required for pixel-accurate arrow placement.
st.session_state.setdefault("DFG_RENDER_W_PX", 1000)

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
xes_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".xes")]
if not xes_files:
    st.error(f"No .xes files found in {data_dir}. Put your XES file into /data.")
    st.stop()

# ===== Top Toolbar =====
t1, t2, t3, t4, t5, t6, t7 = st.columns([1.4, 1.6, 1.6, 1.2, 1.4, 1.0, 1.0])

dataset = t1.selectbox("Dataset", xes_files, index=0)
xes_path = os.path.join(data_dir, dataset)

events, cases = load_xes_as_tables(xes_path)
# ---- stable codes A/B/C... for activities (global) ----
act_to_code, code_to_act, nodes_all = build_code_maps(events)

# ---- time range ----
ts_min = events["ts"].min()
ts_max = events["ts"].max()
min_date = ts_min.date()
max_date = ts_max.date()

cand_attrs = []
for col in cases.columns:
    if col == "case_id":
        continue
    nunique = cases[col].dropna().nunique()
    if 2 <= nunique <= 20:
        cand_attrs.append(col)
if not cand_attrs:
    st.error("No suitable sensitive attributes found (2~20 unique values).")
    st.stop()

sensitive_attr = t2.selectbox("Sensitive Attribute", cand_attrs)

groups = sorted(cases[sensitive_attr].dropna().unique().tolist(), key=lambda x: str(x))
if len(groups) < 2:
    st.error("Selected sensitive attribute has <2 groups.")
    st.stop()

ref_group = t3.selectbox("Reference Group", groups, index=0)
tgt_candidates = [g for g in groups if g != ref_group]
target_group = t4.selectbox("Target Group", tgt_candidates, index=0)

measure = t5.selectbox("Disparity Measure", ["Δpp", "Ratio"])
blind_mode = t6.checkbox("Blind Mode", value=False)
t7.button("Export (later)")

st.divider()

# ===== Blueprint layout =====
left_pane, central_pane = st.columns([1, 3], vertical_alignment="top")

with left_pane:
    st.markdown("### Left Filter Pane")

    with st.expander("Activity Filter", expanded=True):
        all_acts = sorted(act_to_code.keys(), key=lambda x: str(x))
        
        st.write('<span title="Select which activities to include in the analysis. Only events with selected activities will be shown in all views below.">Filter the process by selecting activities to analyze</span>', unsafe_allow_html=True)
        
        selected_acts = st.multiselect(
            "Include activities",
            options=all_acts,
            default=all_acts,
            help="Choose which activities to include. Unchecked activities are excluded from all analysis."
        )


    with st.expander("Time Window", expanded=True):
        apply_time = st.checkbox("Apply time window filter", value=False, help="Enable to restrict analysis to a specific date range. All events within the selected dates will be included.")
        
        if apply_time:
            # 显示当前时间范围信息（使用 columns + write 而不是 metric，避免截断）
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write("**Log start date**")
                st.write(f"<h3 style='margin: 0; color: #1f77b4;'>{ts_min.strftime('%Y-%m-%d')}</h3>", unsafe_allow_html=True)
            with col_info2:
                st.write("**Log end date**")
                st.write(f"<h3 style='margin: 0; color: #1f77b4;'>{ts_max.strftime('%Y-%m-%d')}</h3>", unsafe_allow_html=True)
            
            st.divider()
            
            # 日期范围选择
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter cases by their activity timestamps. Includes entire 24-hour period (00:00-23:59)."
            )
            
            # 处理单日期或范围的情况（避免"not enough values to unpack"错误）
            if not isinstance(date_range, (tuple, list)):
                # 用户只选了一个日期
                date_range = (date_range, date_range)
            elif len(date_range) == 1:
                # 以防万一，如果是列表但只有一个元素
                date_range = (date_range[0], date_range[0])
        else:
            # 未启用时，使用默认的全日期范围
            date_range = (min_date, max_date)


with central_pane:
    st.markdown("### ① Fairness Analysis: Choose Level")

    # --- view toggle with help icons ---
    col_radio, col_spacer = st.columns([6, 1])
    
    with col_radio:
        st.session_state.setdefault("view_mode", "Edges (Activity Transitions)")
        # Programmatic navigation (e.g., jump from Variant Explorer -> Global Path Patterns)
        st.session_state.setdefault("jump_to_global", False)
        st.session_state.setdefault("scroll_to_dfg", False)
        if st.session_state.get("jump_to_global"):
            # Must be set BEFORE the widget with key="view_mode" is instantiated
            st.session_state["view_mode"] = "Edges (Activity Transitions)"
            # Keep jump_to_global until we finish scrolling in the Global Path Patterns view
            st.session_state["scroll_to_dfg"] = True
        view_mode_index = 0 if st.session_state.get("view_mode") == "Edges (Activity Transitions)" else 1
        view_mode = st.radio(
            " ",
            ["Edges (Activity Transitions)", "Variant Explorer"],
            horizontal=True,
            index=view_mode_index,
            label_visibility="collapsed",
            key="view_mode",
        )
    
    # Show help info based on selected view
    if view_mode == "Edges (Activity Transitions)":
        st.caption("📊 **Displays all activity transitions** in the process as a graph. Nodes are activities, edges show direct-follows relationships. Unfair edges appear in red with a yellow (!) badge—click to see detailed fairness metrics.")
    else:  # Variant Explorer
        st.caption("📊 **Shows how often each complete process trace (variant)** occurs in the target vs. reference group. Each variant is ranked by fairness disparity magnitude. The forest plot displays point estimates and 95% confidence intervals, with shaded fairness bands indicating acceptable vs. unacceptable disparities. Use this view to identify which full process paths are unfair and how common they are.")

    # --- shared fairness band (top-toolbar item in blueprint) ---
    with st.expander("Fairness band", expanded=False):
        if measure == "Δpp":
            band_pp = st.number_input(
                "Fairness band (pp): |Δ| ≤",
                value=float(st.session_state.get('band_pp', 5.0)),
                step=0.5,
                key="band_pp",
            )
            band_ratio_low, band_ratio_high = 0.8, 1.25
        else:
            band_pp = float(st.session_state.get('band_pp', 5.0))
            band_ratio_low = st.number_input(
                "Fairness band low (ratio)",
                value=float(st.session_state.get('band_ratio_low', 0.80)),
                step=0.05,
                key="band_ratio_low",
            )
            band_ratio_high = st.number_input(
                "Fairness band high (ratio)",
                value=float(st.session_state.get('band_ratio_high', 1.25)),
                step=0.05,
                key="band_ratio_high",
            )

    # --- group labels ---
    if blind_mode:
        mapped = {g: f"Group {chr(ord('A') + i)}" for i, g in enumerate(groups)}
        ref_label = mapped[ref_group]
        tgt_label = mapped[target_group]
    else:
        ref_label = str(ref_group)
        tgt_label = str(target_group)

    st.caption(f"Comparing **{tgt_label}** vs **{ref_label}** on **{sensitive_attr}**")

    # --- apply left-pane filters (activity + time window) ---
    events_f = events.copy()

    if selected_acts:
        events_f = events_f[events_f["activity"].isin(selected_acts)]

    if apply_time:
        if isinstance(date_range, (tuple, list)):
            start_d, end_d = date_range
        else:
            start_d = end_d = date_range

        # 构建时间戳范围（整天）：
        # start_dt = 起始日期 00:00:00
        # end_dt = 结束日期 23:59:59
        from datetime import timezone
        start_dt = datetime.combine(start_d, time(hour=0, minute=0, second=0))
        end_dt = datetime.combine(end_d, time(hour=23, minute=59, second=59))
        
        # 确保与 events_f["ts"] 的 timezone 兼容（通常是 UTC）
        if events_f["ts"].dt.tz is not None:
            # 如果数据有 timezone 信息，则将比较值也转换为 UTC
            start_dt = start_dt.replace(tzinfo=timezone.utc)
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        
        events_f = events_f[(events_f["ts"] >= start_dt) & (events_f["ts"] <= end_dt)]
        
        # 显示实际应用的时间范围
        st.caption(f"⏱️ Time window applied: **{start_dt.strftime('%Y-%m-%d 00:00')}** to **{end_dt.strftime('%Y-%m-%d 23:59')}**")

    keep_cases = set(events_f["case_id"].unique().tolist())
    cases_f = cases[cases["case_id"].isin(keep_cases)].copy()

    # --- ① View A: Global Path Patterns (DFG) ---
    if view_mode == "Edges (Activity Transitions)":
        # 【修复4】：添加跳转锚点和显眼提示，确保跳转到DFG而不是下方的表格
        st.markdown("#### 🌐 Edges (Activity Transitions)")
        
        # 如果来自 Variant Explorer 的跳转，显示大的提示框并添加空白来调整滚动位置
        if st.session_state.get("scroll_to_dfg"):
            # 显示显眼的提示（占用空间，影响滚动位置）
            st.info(
                f"🎯 **Variant Selected**: {' → '.join(st.session_state.get('selected_variant_tuple', []))}\n\n"
                "The edges for this variant are highlighted in **blue** on the DFG below.",
                icon="📍",
            )

            st.write("")  # 额外空白，帮助调整滚动
        
        with st.expander("Activity Transition settings", expanded=True):
            cA, cB, cC = st.columns([1, 1, 1])
            min_support = cA.slider("Min case-support per edge", 1, 200, 10, 1)
            max_edges = cB.slider("Max edges to draw", 20, 250, 120, 10)
            show_table = cC.checkbox("Show edge table", value=True)

        # choose highlight activity
        nodes_sorted, x_norm, y_norm = compute_node_layout(events_f)
        options = ["(none)"] + nodes_sorted
        sel = st.selectbox("Highlight activity (incoming + outgoing edges)", options=options)
        highlight_activity = None if sel == "(none)" else sel

        # compute edge disparity
        edge_case_df = build_edge_case_support(events_f)
        edge_stats = compute_disparity(
            edge_case_df=edge_case_df,
            cases=cases_f,
            sensitive_attr=sensitive_attr,
            ref_group=ref_group,
            target_group=target_group,
            measure=measure,
            band_pp=band_pp,
            band_ratio_low=band_ratio_low,
            band_ratio_high=band_ratio_high,
        )

        # build figure
        nodes_sorted, x_norm, y_norm = compute_node_layout(events_f)
        
        # ---- Compute highlight_variant_edges from session state ----
        highlight_variant_edges = None
        if st.session_state.get('selected_variant_tuple'):
            variant_tuple = st.session_state.get('selected_variant_tuple')
            if variant_tuple and len(variant_tuple) > 1:
                # Build set of edges from variant tuple: (A, B, C) -> {(A,B), (B,C)}
                highlight_variant_edges = set(zip(variant_tuple[:-1], variant_tuple[1:]))
                st.info(f"🔵 Currently highlighting variant: **{' → '.join(variant_tuple)}**", icon="ℹ️")
        
        fig, badge_df, draw_df = build_dfg_figure(
            edge_stats=edge_stats,
            nodes_sorted=nodes_sorted,
            x_norm=x_norm,
            y_norm=y_norm,
            min_support=min_support,
            max_edges=max_edges,
            measure=measure,
            ref_label=ref_label,
            tgt_label=tgt_label,
            highlight_activity=highlight_activity,
            pinned_popup=st.session_state.get('badge_popup'),
            badge_t=st.session_state.get('badge_t', 0.45),
            badge_spread=st.session_state.get('badge_spread', 0.020),
            highlight_variant_edges=highlight_variant_edges,
        )

        if fig is None:
            st.warning("No edges to draw with current settings (try lower Min support).")
        else:
            st.session_state.setdefault('selected_badge_edge', None)
            st.session_state.setdefault('badge_popup', None)
            st.session_state.setdefault('last_badge_evt', None)
            st.session_state.setdefault('dfg_click_nonce', 0)

            # ---- Jump / scroll to the DFG graph ----
            st.markdown('<div id="dfg-scroll-target"></div>', unsafe_allow_html=True)
            if st.session_state.get('scroll_to_dfg'):
                components.html(
                    """
<script>
(function(){
  const doScroll = () => {
    const el = window.parent.document.getElementById('dfg-scroll-target');
    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
  };
  setTimeout(doScroll, 80);
})();
</script>
""",
                    height=0,
                )
                # Clear navigation flags after executing the scroll
                st.session_state['scroll_to_dfg'] = False
                st.session_state['jump_to_global'] = False

            # ---- Render DFG + handle badge click ----
            if HAS_PLOTLY_EVENTS:
                clicked = plotly_events(
                    fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=650,
                    key=f"dfg_click_{st.session_state.dfg_click_nonce}",
                )
                if clicked and not badge_df.empty:
                    ev = clicked[0]
                    curve = ev.get("curveNumber", None)
                    if curve is not None and 0 <= int(curve) < len(fig.data):
                        if getattr(fig.data[int(curve)], "meta", None) == "badge":
                            idx = ev.get("customdata", None)
                            if idx is None:
                                idx = ev.get("pointIndex", ev.get("pointNumber", None))
                            if idx is not None:
                                idx = int(idx)
                                if 0 <= idx < len(badge_df):
                                    st.session_state.selected_badge_edge = badge_df.iloc[idx].to_dict()

                                r = st.session_state.selected_badge_edge
                                edge_name = f"{r['src']} → {r['dst']}"
                                if measure == "Δpp":
                                    metric_line = f"Δpp = {float(r['delta_pp']):+.2f}pp"
                                else:
                                    metric_line = f"Ratio = {float(r['ratio']):.2f}×" if not pd.isna(r.get('ratio', np.nan)) else "Ratio = NA"

                                txt = (
                                    f"<b>{edge_name}</b><br>"
                                    f"Target ({tgt_label}) vs Reference ({ref_label})<br>"
                                    f"{metric_line}<br>"
                                    f"p_tgt = {float(r['p_tgt']):.3f} ({int(r['support_tgt'])}/{int(r['n_tgt'])}), "
                                    f"p_ref = {float(r['p_ref']):.3f} ({int(r['support_ref'])}/{int(r['n_ref'])})<br>"
                                    f"case frequency (all) = {int(r['case_support_all'])}"
                                )

                                evt_sig = (ev.get('curveNumber'), ev.get('pointIndex'), ev.get('x'), ev.get('y'))
                                if st.session_state.last_badge_evt != evt_sig:
                                    st.session_state.last_badge_evt = evt_sig
                                    st.session_state.badge_popup = {
                                        'x': float(ev.get('x', 0.5)),
                                        'y': float(ev.get('y', 0.5)),
                                        'text': txt,
                                    }
                                    st.rerun()
            else:
                # Responsive rendering; avoids right-side clipping on narrower containers
                st.plotly_chart(fig, use_container_width=False)


            # ---- DFG actions (placed directly below the DFG) ----
            bbtn1, bbtn2 = st.columns(2)
            with bbtn1:
                if st.button("Clear pinned", key="clear_pinned"):
                    st.session_state.badge_popup = None
                    st.session_state.selected_badge_edge = None
                    st.session_state.last_badge_evt = None
                    st.session_state.dfg_click_nonce += 1
                    st.rerun()  # refresh to remove popup
            with bbtn2:
                if st.button("Clear variant", key="clear_variant"):
                    st.session_state.selected_variant_id = None
                    st.session_state.selected_variant_tuple = None
                    # also reset Variant Explorer selector state (so it stays cleared when switching tabs)
                    st.session_state.selected_variant_str = "(none)"
                    st.rerun()  # refresh to remove blue highlight

            # ---- Edge table (current view) ----
            if show_table:
                table_df = draw_df.copy()
                if not table_df.empty:
                    table_df['edge'] = table_df['src'].astype(str) + ' → ' + table_df['dst'].astype(str)
                    if measure == 'Δpp':
                        table_df['_score'] = table_df['delta_pp'].abs()
                    else:
                        table_df['_score'] = (table_df['ratio'] - 1.0).abs()

                    table_df = table_df.sort_values(
                        by=['is_unfair', '_score', 'case_support_all'],
                        ascending=[False, False, False],
                    )

                    cols = ['edge', 'case_support_all', 'p_tgt', 'p_ref']
                    cols += (['delta_pp'] if measure == 'Δpp' else ['ratio'])
                    cols += ['support_tgt', 'n_tgt', 'support_ref', 'n_ref', 'is_unfair']

                    st.dataframe(
                        table_df[cols].drop(columns=['_score'], errors='ignore'),
                        use_container_width=True,
                        hide_index=True,
                    )
    # --- ① View B: Variant Explorer (forest plot + table) ---
    elif view_mode == "Variant Explorer":
        st.markdown("#### 📋 Variant Explorer")

        with st.expander("Variant Explorer settings", expanded=True):
            c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
            top_k = c1.slider("Top-K most unfair variants", 5, 60, 20, 1)
            only_unfair = c2.checkbox("Show only unfair variants", value=False, key="var_only_unfair")
            show_table = c3.checkbox("Show variant table", value=True, key="var_show_table")

            q = st.text_input("Search variant (contains)", value="", key="var_search").strip().lower()
            st.caption("Ranking: Top-K by unfairness (|Δpp| or |ratio-1|), ties broken by overall case support.")

        variants_df, variant_case_df = build_case_variants(events_f)
        var_stats = compute_variant_disparity(
            variants_df=variants_df,
            variant_case_df=variant_case_df,
            cases=cases_f,
            sensitive_attr=sensitive_attr,
            ref_group=ref_group,
            target_group=target_group,
            measure=measure,
            band_pp=band_pp,
            band_ratio_low=band_ratio_low,
            band_ratio_high=band_ratio_high,
        )

        if var_stats is None or var_stats.empty:
            st.info("No variants available (check filters).")
        else:
            dfv = var_stats.copy()

            if q:
                dfv = dfv[dfv["variant_str"].str.lower().str.contains(q, na=False)].copy()

            if only_unfair:
                dfv = dfv[dfv["is_unfair"] == True].copy()

            if dfv.empty:
                st.info("No variants match the current settings.")
            else:
                # Always rank by unfairness magnitude
                dfv = dfv.sort_values(["_score", "case_support_all"], ascending=[False, False])

                fig, disp_df = build_variant_forest_figure(
                    var_stats=dfv,
                    measure=measure,
                    band_pp=band_pp,
                    band_ratio_low=band_ratio_low,
                    band_ratio_high=band_ratio_high,
                    ref_label=ref_label,
                    tgt_label=tgt_label,
                    top_k=int(top_k),
                )

                if fig is None or disp_df.empty:
                    st.info("No valid variants to plot (e.g., ratio=NA).")
                else:
                    st.plotly_chart(fig, use_container_width=False)
                    
                    # ---- 快速高亮不公平变体 ----
                    st.markdown("### 💡 Quick Highlight Unfair Variants")
                    
                    # 显示不公平变体列表，每个都有高亮按钮
                    unfair_variants = disp_df[disp_df["is_unfair"] == True].copy()
                    if not unfair_variants.empty:
                        st.caption("Click any variant below to highlight in Edges (Activity Transitions):")
                        
                        # 创建列来放置高亮按钮
                        cols_unfair = st.columns([3, 1, 1])
                        with cols_unfair[0]:
                            st.write("**Unfair Variant**")
                        with cols_unfair[1]:
                            st.write("**Metric**")
                        with cols_unfair[2]:
                            st.write("**Action**")
                        
                        # 为每个不公平变体显示一个高亮按钮
                        for idx, (_, variant_row) in enumerate(unfair_variants.iterrows()):
                            cols_btn = st.columns([3, 1, 1])
                            
                            with cols_btn[0]:
                                # 变体名称和路径
                                variant_label = variant_row["variant_label"]
                                variant_path = variant_row["variant_str"][:60]  # 截断显示
                                st.write(f"**{variant_label}:** {variant_path}...")
                            
                            with cols_btn[1]:
                                # 显示不公平度量值
                                if measure == "Δpp":
                                    metric_val = f"{float(variant_row['delta_pp']):+.3f}pp"
                                else:
                                    metric_val = f"{float(variant_row['ratio']):.2f}×"
                                st.write(metric_val)
                            
                            with cols_btn[2]:
                                # 高亮按钮
                                if st.button(f"🔗 Highlight", key=f"highlight_unfair_{idx}"):
                                    st.session_state.selected_variant_id = int(variant_row["variant_id"])
                                    st.session_state.selected_variant_tuple = tuple(variant_row["variant"])
                                    # 【修复4】：request switching view on next rerun (must happen before radio is created)
                                    st.session_state["jump_to_global"] = True
                                    st.session_state["scroll_to_dfg"] = True
                                    
                                    st.success(f"✅ **{variant_label}** selected! Jumping to Edges (Activity Transitions)...")
                                    import time as time_module
                                    time_module.sleep(0.3)
                                    st.rerun()
                    else:
                        st.info("✓ No unfair variants found!")
                    
                    st.divider()

                    # 【修复3】：将 Dropdown 移到 All Variants table 之前
                    st.markdown("### 🔍 Select Variant to Highlight")
                    var_options = ["(clear)"] + disp_df["variant_str"].tolist()
                    selected_var_str = st.selectbox(
                        "Highlight variant path in DFG",
                        options=var_options,
                        index=0,
                        key="selected_variant_str",
                        label_visibility="collapsed"
                    )

                    # Store selected variant to session state
                    if selected_var_str and selected_var_str != "(clear)":
                        matching = disp_df[disp_df["variant_str"] == selected_var_str]
                        if not matching.empty:
                            st.session_state.selected_variant_id = int(matching.iloc[0]["variant_id"])
                            st.session_state.selected_variant_tuple = tuple(matching.iloc[0]["variant"])
                            st.info(f"✓ Selected variant: **{selected_var_str}** will be highlighted in Edges (Activity Transitions)", icon="ℹ️")
                        else:
                            st.session_state.selected_variant_id = None
                            st.session_state.selected_variant_tuple = None
                    else:
                        st.session_state.selected_variant_id = None
                        st.session_state.selected_variant_tuple = None
                    
                    st.divider()

                    # 【修复3】：All Variants table 在 Dropdown 之后
                    if show_table:
                        st.markdown("### 📋 All Variants")
                        out = disp_df.copy()
                        cols = [
                            "variant_label",
                            "case_support_all",
                            "p_tgt",
                            "p_ref",
                            "delta_pp" if measure == "Δpp" else "ratio",
                            "ci_low",
                            "ci_high",
                            "support_tgt",
                            "n_tgt",
                            "support_ref",
                            "n_ref",
                            "is_unfair",
                            "variant_str",  # 显示完整的 variant_str，不缩短
                        ]
                        cols = [c for c in cols if c in out.columns]
                        
                        # 重命名列标题
                        display_df = out[cols].copy()
                        display_df = display_df.rename(columns={"variant_str": "Variant Path"})
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # =============================
    # ② Outcome Disparity Dashboard
    # =============================
    st.divider()
    st.markdown("### ② Outcome Disparity Dashboard")

    events_view = events_f
    cases_view = cases_f
    st.caption("Dashboard scope: **all cases** after current filters")
    st.caption("Outcomes: Offer / Reject (based on last activity) + Processing time (case duration).")

    acts_view = sorted(events_view['activity'].dropna().unique().tolist(), key=lambda x: str(x))

    def _guess_act(pats):
        for a in acts_view:
            s = str(a)
            for pat in pats:
                if re.search(pat, s, flags=re.IGNORECASE):
                    return a
        return None

    guess_offer = _guess_act([r"offer", r"approved", r"accept"])
    guess_reject = _guess_act([r"reject", r"rejected", r"declin", r"denied"])

    with st.expander("Outcome dashboard settings", expanded=False):
        if not acts_view:
            st.warning("No activities available after current filters.")
            offer_act = None
            reject_act = None
        else:
            offer_act = st.selectbox(
                "Offer terminal activity",
                options=acts_view,
                index=(acts_view.index(guess_offer) if guess_offer in acts_view else 0),
                key='offer_act',
            )
            reject_act = st.selectbox(
                "Reject terminal activity",
                options=acts_view,
                index=(acts_view.index(guess_reject) if guess_reject in acts_view else min(1, len(acts_view) - 1)),
                key='reject_act',
            )

        time_agg = st.selectbox("Processing time aggregation", ["mean", "median"], index=0, key='time_agg')
        band_time_days = st.number_input(
            "Processing time fairness band (days): |Δ| ≤ (used when Disparity Measure=Δpp)",
            value=1.0,
            step=0.25,
            key='band_time_days',
        )
        show_outcome_table = st.checkbox("Show dashboard table", value=True, key='show_outcome_table')

    # ---- Prepare group sizes ----
    cc = cases_view[['case_id', sensitive_attr]].dropna().copy()
    n_by_group = cc.groupby(sensitive_attr)['case_id'].nunique().to_dict()
    n_ref = int(n_by_group.get(ref_group, 0))
    n_tgt = int(n_by_group.get(target_group, 0))

    if n_ref == 0 or n_tgt == 0:
        st.info("No cases available for one of the selected groups (check filters / selected pattern).")
    else:
        # ---- Offer / Reject rates (based on last activity) ----
        last_df = compute_case_last_activity(events_view)
        oc = last_df.merge(cc, on='case_id', how='inner').rename(columns={sensitive_attr: 'group'})

        def _outcome_row(act_name: str, title: str) -> dict:
            if not act_name:
                return {
                    'metric': title,
                    'type': 'prob',
                    'p_tgt': float('nan'),
                    'p_ref': float('nan'),
                    'delta_pp': float('nan'),
                    'ratio': float('nan'),
                    'support_tgt': 0,
                    'support_ref': 0,
                    'case_support_all': 0,
                    'is_unfair': False,
                    'unknown': True,
                }

            sup_ref = int(oc[(oc['group'] == ref_group) & (oc['outcome'] == act_name)]['case_id'].nunique())
            sup_tgt = int(oc[(oc['group'] == target_group) & (oc['outcome'] == act_name)]['case_id'].nunique())
            sup_all = int(oc[oc['outcome'] == act_name]['case_id'].nunique())

            p_ref = sup_ref / n_ref
            p_tgt = sup_tgt / n_tgt
            delta_pp = (p_tgt - p_ref) * 100.0
            ratio = (p_tgt / p_ref) if p_ref > 0 else float('nan')

            if measure == 'Δpp':
                is_unfair = abs(delta_pp) > float(band_pp)
                unknown = False
            else:
                unknown = bool(pd.isna(ratio))
                is_unfair = (not unknown) and (not (float(band_ratio_low) <= float(ratio) <= float(band_ratio_high)))

            return {
                'metric': title,
                'type': 'prob',
                'p_tgt': p_tgt,
                'p_ref': p_ref,
                'delta_pp': delta_pp,
                'ratio': ratio,
                'support_tgt': sup_tgt,
                'support_ref': sup_ref,
                'case_support_all': sup_all,
                'is_unfair': bool(is_unfair),
                'unknown': unknown,
            }

        offer_row = _outcome_row(offer_act, 'Offer rate')
        reject_row = _outcome_row(reject_act, 'Reject rate')

        # ---- Processing time ----
        case_time_df = compute_case_processing_time(events_view)
        time_stats = compute_processing_time_disparity(
            case_time_df=case_time_df,
            cases=cases_view,
            sensitive_attr=sensitive_attr,
            ref_group=ref_group,
            target_group=target_group,
            measure=measure,
            band_time_days=band_time_days,
            band_ratio_low=band_ratio_low,
            band_ratio_high=band_ratio_high,
            agg=time_agg,
        )

        if not time_stats:
            time_row = {
                'metric': 'Processing time',
                'type': 'time',
                'center_tgt_days': float('nan'),
                'center_ref_days': float('nan'),
                'delta_days': float('nan'),
                'ratio': float('nan'),
                'support_tgt': 0,
                'support_ref': 0,
                'is_unfair': False,
                'unknown': True,
                'center_name': time_agg,
            }
        else:
            ratio = float(time_stats.get('ratio', float('nan')))
            unknown = bool(pd.isna(ratio)) if measure == 'Ratio' else False
            time_row = {
                'metric': 'Processing time',
                'type': 'time',
                'center_tgt_days': float(time_stats.get('center_tgt_days', float('nan'))),
                'center_ref_days': float(time_stats.get('center_ref_days', float('nan'))),
                'delta_days': float(time_stats.get('delta_days', float('nan'))),
                'ratio': ratio,
                'support_tgt': int(time_stats.get('support_tgt', 0)),
                'support_ref': int(time_stats.get('support_ref', 0)),
                'is_unfair': bool(time_stats.get('is_unfair', False)),
                'unknown': unknown,
                'center_name': str(time_stats.get('center_name', time_agg)),
            }

        # ---- Inject CSS once (shared with your existing outcome cards) ----
        st.markdown(
            """<style>
.pm-card{background:#ffffff;border:1px solid rgba(0,0,0,0.06);border-radius:18px;padding:16px 16px 12px 16px;box-shadow:0 6px 20px rgba(0,0,0,0.06);} 
.pm-card.pm-unknown{background:#fafafa;border-color:rgba(0,0,0,0.06);} 
.pm-card.pm-good{background:linear-gradient(180deg, rgba(102,187,106,0.18), #ffffff 55%);border-color:rgba(102,187,106,0.35);} 
.pm-card.pm-warn{background:linear-gradient(180deg, rgba(255,167,38,0.22), #ffffff 55%);border-color:rgba(255,167,38,0.40);} 
.pm-card.pm-bad{background:linear-gradient(180deg, rgba(229,115,115,0.22), #ffffff 55%);border-color:rgba(229,115,115,0.45);} 
.pm-top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;} 
.pm-title{font-size:15px;font-weight:750;margin-top:2px;line-height:1.2;} 
.pm-sub{font-size:12px;color:rgba(20,20,20,0.72);line-height:1.2;} 
.pm-plus{font-size:18px;color:rgba(50,50,50,0.45);font-weight:800;line-height:1;} 
.pm-main{font-size:38px;font-weight:900;margin:10px 0 2px 0;letter-spacing:-0.5px;} 
.pm-ci{font-size:12px;color:rgba(20,20,20,0.70);margin-bottom:6px;} 
.pm-spark{margin:6px 0 2px 0;} 
.pm-foot{display:flex;justify-content:space-between;align-items:center;margin-top:6px;} 
.pm-n{font-size:12px;color:rgba(20,20,20,0.70);} 
.pm-hint{font-size:12px;color:rgba(20,20,20,0.55);} 
</style>""",
            unsafe_allow_html=True,
        )

        # ---- Add 95% CI + target-worse flag ----
        for r in [offer_row, reject_row]:
            if bool(r.get('unknown', False)):
                r['ci_low'] = np.nan
                r['ci_high'] = np.nan
                r['target_worse'] = False
                continue

            sup_t = int(r.get('support_tgt', 0))
            sup_r = int(r.get('support_ref', 0))

            if measure == 'Δpp':
                lo, hi = ci_diff_proportion_wald(sup_t, n_tgt, sup_r, n_ref)
                r['ci_low'] = lo * 100.0
                r['ci_high'] = hi * 100.0
            else:
                lo, hi = ci_ratio_katz(sup_t, n_tgt, sup_r, n_ref)
                r['ci_low'] = lo
                r['ci_high'] = hi

            if str(r.get('metric', '')).lower().startswith('offer'):
                r['target_worse'] = (float(r.get('delta_pp', 0.0)) < 0.0) if measure == 'Δpp' else (float(r.get('ratio', np.nan)) < 1.0)
            else:
                r['target_worse'] = (float(r.get('delta_pp', 0.0)) > 0.0) if measure == 'Δpp' else (float(r.get('ratio', np.nan)) > 1.0)

        if not bool(time_row.get('unknown', False)):
            time_ci = compute_processing_time_ci(
                case_time_df=case_time_df,
                cases=cases_view,
                sensitive_attr=sensitive_attr,
                ref_group=ref_group,
                target_group=target_group,
                agg=time_agg,
                n_boot=400,
                seed=7,
            )
            time_row['ci_delta_low'] = time_ci.get('delta_low', np.nan)
            time_row['ci_delta_high'] = time_ci.get('delta_high', np.nan)
            time_row['ci_ratio_low'] = time_ci.get('ratio_low', np.nan)
            time_row['ci_ratio_high'] = time_ci.get('ratio_high', np.nan)

            if measure == 'Δpp':
                time_row['target_worse'] = float(time_row.get('delta_days', 0.0)) > 0.0
            else:
                time_row['target_worse'] = float(time_row.get('ratio', np.nan)) > 1.0
        else:
            time_row['target_worse'] = False

        # ---- Sparklines (within current scope) ----
        try:
            series_or = compute_offer_reject_series(
                events=events_view,
                cases=cases_view,
                sensitive_attr=sensitive_attr,
                ref_group=ref_group,
                target_group=target_group,
                offer_act=str(offer_act),
                reject_act=str(reject_act),
                n_bins=12,
            )
            offer_series = series_or.get('offer', [])
            reject_series = series_or.get('reject', [])
        except Exception:
            offer_series, reject_series = [], []

        try:
            time_series = compute_processing_time_series(
                events=events_view,
                cases=cases_view,
                sensitive_attr=sensitive_attr,
                ref_group=ref_group,
                target_group=target_group,
                agg=time_agg,
                n_bins=12,
            )
        except Exception:
            time_series = []

        rows = [offer_row, reject_row, time_row]
        cols_cards = st.columns(3)

        def _theme(r: dict):
            if bool(r.get('unknown', False)):
                return ('pm-unknown', '#999999', 'rgba(0,0,0,0.06)')
            if bool(r.get('is_unfair', False)):
                if bool(r.get('target_worse', False)):
                    return ('pm-bad', '#e57373', 'rgba(229,115,115,0.18)')
                return ('pm-warn', '#ffa726', 'rgba(255,167,38,0.18)')
            return ('pm-good', '#66bb6a', 'rgba(102,187,106,0.18)')

        def _fmt_ci(lo, hi, unit: str, decimals: int = 1):
            if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
                return '95% CI [NA]'
            fmt = f"{{:.{decimals}f}}"
            return f"95% CI [{fmt.format(lo)}, {fmt.format(hi)}]{(' ' + unit) if unit else ''}"

        def _fmt_main_pp(x):
            if pd.isna(x):
                return 'NA'
            return f"{x:+.0f}pp" if abs(float(x)) >= 10 else f"{x:+.1f}pp"

        def _fmt_main_days(x):
            if pd.isna(x):
                return 'NA'
            return f"{x:+.1f} days"

        def _fmt_main_ratio(x):
            if pd.isna(x):
                return 'NA'
            return f"{float(x):.2f}×"

        for i, r in enumerate(rows):
            klass, stroke, fill = _theme(r)

            if r['type'] == 'prob':
                if measure == 'Δpp':
                    main = _fmt_main_pp(float(r.get('delta_pp', np.nan)))
                    ci_txt = _fmt_ci(float(r.get('ci_low', np.nan)), float(r.get('ci_high', np.nan)), '', 1)
                    spark_vals = offer_series if str(r.get('metric','')).lower().startswith('offer') else reject_series
                else:
                    main = _fmt_main_ratio(float(r.get('ratio', np.nan)))
                    ci_txt = _fmt_ci(float(r.get('ci_low', np.nan)), float(r.get('ci_high', np.nan)), '×', 2)
                    spark_vals = offer_series if str(r.get('metric','')).lower().startswith('offer') else reject_series

                n_used = int(n_ref + n_tgt)
            else:
                if measure == 'Δpp':
                    main = _fmt_main_days(float(r.get('delta_days', np.nan)))
                    ci_txt = _fmt_ci(float(r.get('ci_delta_low', np.nan)), float(r.get('ci_delta_high', np.nan)), 'days', 2)
                else:
                    main = _fmt_main_ratio(float(r.get('ratio', np.nan)))
                    ci_txt = _fmt_ci(float(r.get('ci_ratio_low', np.nan)), float(r.get('ci_ratio_high', np.nan)), '×', 2)

                spark_vals = time_series
                n_used = int(r.get('support_tgt', 0) + r.get('support_ref', 0))

            svg = make_sparkline_svg(spark_vals, stroke=stroke, fill=fill)

            if r['type'] == 'prob':
                hint = f"Fair band: |Δ| ≤ {float(band_pp):.1f}pp" if measure == 'Δpp' else f"Fair band: {float(band_ratio_low):.2f}–{float(band_ratio_high):.2f}×"
            else:
                hint = f"Fair band: |Δ| ≤ {float(band_time_days):.2f} days" if measure == 'Δpp' else f"Fair band: {float(band_ratio_low):.2f}–{float(band_ratio_high):.2f}×"

            title = r['metric'] if r['type'] == 'prob' else f"Processing time ({str(r.get('center_name','mean'))})"

            with cols_cards[i]:
                st.markdown(
                    f"""
<div class="pm-card {klass}">
  <div class="pm-top">
    <div>
      <div class="pm-sub">{tgt_label} vs {ref_label}</div>
      <div class="pm-title">{title}</div>
    </div>
    <div class="pm-plus">＋</div>
  </div>

  <div class="pm-main" style="color:{stroke};">{main}</div>
  <div class="pm-ci">{ci_txt}</div>

  <div class="pm-spark">{svg}</div>

  <div class="pm-foot">
    <div class="pm-n">n = {n_used}</div>
    <div class="pm-hint">{hint}</div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        if show_outcome_table:
            tab_rows = []

            for r in [offer_row, reject_row]:
                tab_rows.append(
                    {
                        'metric': r['metric'],
                        'target_value': r['p_tgt'],
                        'ref_value': r['p_ref'],
                        'delta': r['delta_pp'],
                        'ratio': r['ratio'],
                        'ci_95': (
                            f"[{float(r.get('ci_low', np.nan)):.1f}, {float(r.get('ci_high', np.nan)):.1f}] pp"
                            if measure == 'Δpp'
                            else f"[{float(r.get('ci_low', np.nan)):.2f}, {float(r.get('ci_high', np.nan)):.2f}]×"
                        ),
                        'is_unfair': r['is_unfair'],
                        'support_tgt': r['support_tgt'],
                        'support_ref': r['support_ref'],
                        'n_tgt': n_tgt,
                        'n_ref': n_ref,
                    }
                )

            tab_rows.append(
                {
                    'metric': 'Processing time (' + str(time_row.get('center_name', 'mean')) + ') [days]',
                    'target_value': time_row.get('center_tgt_days', np.nan),
                    'ref_value': time_row.get('center_ref_days', np.nan),
                    'delta': time_row.get('delta_days', np.nan),
                    'ratio': time_row.get('ratio', np.nan),
                    'ci_95': (
                        f"[{float(time_row.get('ci_delta_low', np.nan)):.2f}, {float(time_row.get('ci_delta_high', np.nan)):.2f}] days"
                        if measure == 'Δpp'
                        else f"[{float(time_row.get('ci_ratio_low', np.nan)):.2f}, {float(time_row.get('ci_ratio_high', np.nan)):.2f}]×"
                    ),
                    'is_unfair': time_row.get('is_unfair', False),
                    'support_tgt': time_row.get('support_tgt', 0),
                    'support_ref': time_row.get('support_ref', 0),
                    'n_tgt': n_tgt,
                    'n_ref': n_ref,
                }
            )

            tab_df = pd.DataFrame(tab_rows)
            st.dataframe(tab_df, use_container_width=True, hide_index=True)
