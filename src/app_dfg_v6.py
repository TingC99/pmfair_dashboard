import os
import math
import textwrap
import pandas as pd
import numpy as np
import streamlit as st

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
    m["ratio"] = np.where(m["p_ref"] > 0, m["p_tgt"] / m["p_ref"], np.nan)

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
    m["ratio"] = np.where(m["p_ref"] > 0, m["p_tgt"] / m["p_ref"], np.nan)

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
@st.cache_data(show_spinner=False)
def compute_node_layout(events: pd.DataFrame, n_layers: int = 8, margin: float = 0.08):
    pos = {}
    freq = {}

    for _, grp in events.groupby("case_id", sort=False):
        acts = grp["activity"].tolist()
        L = len(acts)
        if L == 0:
            continue
        denom = max(L - 1, 1)
        for i, a in enumerate(acts):
            pos.setdefault(a, []).append(i / denom)
            freq[a] = freq.get(a, 0) + 1

    avg_pos = {a: float(np.mean(v)) for a, v in pos.items()}
    if not avg_pos:
        return [], {}, {}

    # --- assign layer by average position ---
    def layer_of(a):
        return int(round(avg_pos[a] * (n_layers - 1)))

    layers = {i: [] for i in range(n_layers)}
    for a in avg_pos:
        layers[layer_of(a)].append(a)

    # sort within each layer: frequent first
    for i in range(n_layers):
        layers[i].sort(key=lambda a: (-freq.get(a, 0), str(a)))

    # x coordinate per layer
    x_norm = {}
    for i in range(n_layers):
        x = 0.05 + 0.90 * (i / (n_layers - 1)) if n_layers > 1 else 0.5
        for a in layers[i]:
            x_norm[a] = x

    # y coordinate within each layer (spread evenly with margin)
    y_norm = {}
    for i in range(n_layers):
        nodes = layers[i]
        k = len(nodes)
        if k == 0:
            continue
        # evenly distribute centers between [margin, 1-margin]
        for j, a in enumerate(nodes):
            y_norm[a] = margin + (j + 0.5) * (1 - 2 * margin) / k

    # nodes_sorted used for labels/order
    nodes_sorted = []
    for i in range(n_layers):
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

    max_w = float(df["case_support_all"].max()) if float(df["case_support_all"].max()) > 0 else 1.0

    fig = go.Figure()

    # --- edges (one trace per edge so we can control width & hover precisely)
    for _, r in df.iterrows():
        s, d = r["src"], r["dst"]
        x0, y0 = float(x_norm.get(s, 0.5)), float(y_norm.get(s, 0.5))
        x1, y1 = float(x_norm.get(d, 0.5)), float(y_norm.get(d, 0.5))

        w = float(r["case_support_all"])
        width = 1.0 + 10.0 * math.sqrt(w / max_w)

        if bool(r["is_unfair"]):
            alpha = 0.75 if bool(r["is_focus"]) else 0.10
            color = f"rgba(220,0,0,{alpha})"
        else:
            alpha = 0.40 if bool(r["is_focus"]) else 0.08
            color = f"rgba(120,120,120,{alpha})"

        edge_text = f"{s} → {d}"

        # customdata for hovertemplate (two points, same payload)
        if measure == "Δpp":
            metric_val = float(r["delta_pp"])
            metric_label = "Δpp"
            metric_fmt = "%{customdata[3]:+.1f}pp"
        else:
            metric_val = float(r["ratio"]) if not pd.isna(r["ratio"]) else float("nan")
            metric_label = "Ratio"
            metric_fmt = "%{customdata[3]:.2f}×"

        row = [
            edge_text,
            float(r["p_tgt"]),
            float(r["p_ref"]),
            metric_val,
            float(r["support_tgt"]),
            float(r["support_ref"]),
            float(r["n_tgt"]),
            float(r["n_ref"]),
            bool(r["is_unfair"]),
            w,
        ]
        customdata = [row, row]

        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Case frequency: %{customdata[9]:.0f}<br>"
            f"p(target={tgt_label}): %{{customdata[1]:.3f}} (%{{customdata[4]:.0f}}/%{{customdata[6]:.0f}})<br>"
            f"p(ref={ref_label}): %{{customdata[2]:.3f}} (%{{customdata[5]:.0f}}/%{{customdata[7]:.0f}})<br>"
            f"{metric_label}: <b>{metric_fmt}</b><br>"
            "Unfair: %{customdata[8]}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color),
                customdata=customdata,
                hovertemplate=hovertemplate,
                showlegend=False,
            )
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

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        clickmode="event+select",
        hovermode="closest",
    )
    return fig, badge_df, df
# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PM Fairness Prototype", layout="wide")
st.markdown("## PM Fairness Dashboard (Prototype test)")

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
        selected_acts = st.multiselect(
            "Include activities",
            options=all_acts,
            default=all_acts
        )


    with st.expander("Time Window", expanded=True):
        apply_time = st.checkbox("Apply time window", value=False)
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # 其他 case filter 之后再加
    with st.expander("Case Filter (later)", expanded=False):
        st.write("后续可加：case_id 搜索、多属性过滤（age/education/citizen 等）。")

with central_pane:
    st.markdown("### ① Variant & Local Pattern Fairness Map")

    view_mode = st.radio(
        " ",
        ["Global Path Patterns", "Local Process Models (later)"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

    st.caption("DFG: edge width = case frequency. Unfair edges are red with a yellow badge (!) you can click.")
    colA, colB = st.columns([1, 6])
    with colA:
        if st.button("Clear pinned", key="clear_pinned"):
            st.session_state.badge_popup = None
            st.session_state.selected_badge_edge = None
            st.session_state.last_badge_evt = None


    if view_mode != "Global Path Patterns":
        st.info("Local Process Models 先不做，这里先留占位。")
    else:
        # settings
        with st.expander("Global Path View Settings", expanded=True):
            cA, cB, cC = st.columns([1, 1, 1])
            min_support = cA.slider("Min case-support per edge", 1, 200, 10, 1)
            max_edges = cB.slider("Max edges to draw", 20, 250, 120, 10)
            show_table = cC.checkbox("Show edge table", value=True)

            if measure == "Δpp":
                band_pp = st.number_input("Fairness band (pp): |Δ| ≤", value=5.0, step=0.5)
                band_ratio_low, band_ratio_high = 0.8, 1.25
            else:
                band_pp = 5.0
                band_ratio_low = st.number_input("Fairness band low (ratio)", value=0.8, step=0.05)
                band_ratio_high = st.number_input("Fairness band high (ratio)", value=1.25, step=0.05)

        # group labels
        if blind_mode:
            mapped = {g: f"Group {chr(ord('A') + i)}" for i, g in enumerate(groups)}
            ref_label = mapped[ref_group]
            tgt_label = mapped[target_group]
        else:
            ref_label = str(ref_group)
            tgt_label = str(target_group)

        st.caption(f"Comparing **{tgt_label}** vs **{ref_label}** on **{sensitive_attr}**")
        
        # ---- Step 5: apply filters (data-level) ----
        events_f = events.copy()

        # 1) Activity subset filter (selected_acts 来自左侧 Activity Filter)
        if selected_acts:
            events_f = events[events["activity"].isin(selected_acts)]


        # 2) Time window filter (apply_time / date_range 来自左侧 Time Window)
        if apply_time:
            # date_range 可能是单个日期，也可能是 (start, end)
            if isinstance(date_range, (tuple, list)):
                start_d, end_d = date_range
            else:
                start_d = end_d = date_range

            start_dt = datetime.combine(start_d, time.min)
            end_dt = datetime.combine(end_d, time.max)

            events_f = events_f[(events_f["ts"] >= start_dt) & (events_f["ts"] <= end_dt)]

        # 3) Keep only cases that still have events after filters
        keep_cases = set(events_f["case_id"].unique().tolist())
        cases_f = cases[cases["case_id"].isin(keep_cases)].copy()

        # ---- Step 6: legend search + choose highlight activity ----
        nodes_sorted, x_norm, y_norm = compute_node_layout(events_f)
        # options = ["(none)"] + [f"{c} — {a}" for c, a in legend_rows_filtered]
        # sel = st.selectbox("Highlight activity (incoming + outgoing edges)", options=options, index=0)
        options = ["(none)"] + nodes_sorted
        sel = st.selectbox("Highlight activity (incoming + outgoing edges)", options=options)

        highlight_activity = None
        if sel != "(none)":
            highlight_activity = sel


        # compute stats
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

        # node layout + sankey
        nodes_sorted, x_norm, y_norm = compute_node_layout(events_f)

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
        )



        if fig is None:
            st.warning("No edges to draw with current settings (try lower Min support).")
        else:
            if "selected_badge_edge" not in st.session_state:
                st.session_state.selected_badge_edge = None
            if "badge_popup" not in st.session_state:
                st.session_state.badge_popup = None
            if "last_badge_evt" not in st.session_state:
                st.session_state.last_badge_evt = None

                        # ---- Render DFG + handle badge click ----
            if HAS_PLOTLY_EVENTS:
                clicked = plotly_events(
                    fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=650,
                    key="dfg_click",
                )
                if clicked and not badge_df.empty:
                    ev = clicked[0]
                    curve = ev.get("curveNumber", None)
                    if curve is not None and 0 <= int(curve) < len(fig.data):
                        # badge trace has meta="badge"
                        if getattr(fig.data[int(curve)], "meta", None) == "badge":
                            idx = ev.get("customdata", None)
                            if idx is None:
                                idx = ev.get("pointIndex", ev.get("pointNumber", None))
                            if idx is not None:
                                idx = int(idx)
                                if 0 <= idx < len(badge_df):
                                    st.session_state.selected_badge_edge = badge_df.iloc[idx].to_dict()

                                # Pin a popup next to the badge (Plotly annotation)
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
                # Fallback: use Streamlit selection (requires box/lasso in some versions)
                chart_event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key="dfg_chart",
                )
                sel = None
                if chart_event is not None:
                    if hasattr(chart_event, "selection"):
                        sel = chart_event.selection
                    elif isinstance(chart_event, dict):
                        sel = chart_event.get("selection", None)

                pts = []
                if isinstance(sel, dict):
                    pts = sel.get("points", []) or []

                if pts and not badge_df.empty:
                    pt = pts[0]
                    curve = pt.get("curve_number", pt.get("curveNumber", None))
                    if curve is not None and 0 <= int(curve) < len(fig.data):
                        if getattr(fig.data[int(curve)], "meta", None) == "badge":
                            idx = pt.get("customdata", None)
                            if idx is None:
                                idx = pt.get("point_number", pt.get("pointNumber", pt.get("pointIndex", None)))
                            if idx is not None:
                                idx = int(idx)
                                if 0 <= idx < len(badge_df):
                                    st.session_state.selected_badge_edge = badge_df.iloc[idx].to_dict()

            # ---- Edge table (current view) ----
            if show_table:
                table_df = draw_df.copy()
                if not table_df.empty:
                    table_df["edge"] = table_df["src"].astype(str) + " → " + table_df["dst"].astype(str)

                    # Sort: unfair first, then by magnitude of measure, then by frequency
                    if measure == "Δpp":
                        table_df["_score"] = table_df["delta_pp"].abs()
                    else:
                        # distance from 1.0
                        table_df["_score"] = (table_df["ratio"] - 1.0).abs()

                    table_df = table_df.sort_values(
                        by=["is_unfair", "_score", "case_support_all"],
                        ascending=[False, False, False],
                    )

                    cols = ["edge", "case_support_all", "p_tgt", "p_ref"]
                    if measure == "Δpp":
                        cols += ["delta_pp"]
                    else:
                        cols += ["ratio"]
                    cols += ["support_tgt", "n_tgt", "support_ref", "n_ref", "is_unfair"]

                    st.dataframe(
                        table_df[cols].drop(columns=["_score"], errors="ignore"),
                        use_container_width=True,
                        hide_index=True,
                    )


            # =============================
            # ② Outcome Disparity Dashboard
            # =============================
            st.divider()
            st.markdown("### ② Outcome Disparity Dashboard")
            st.caption("Outcome proxy: last activity of each case (after current filters). Cards are sorted by |disparity|.")

            with st.expander("Outcome dashboard settings", expanded=False):
                top_k_outcomes = st.slider("Show top-K outcomes (by case frequency)", 2, 20, 8, 1)
                show_outcome_table = st.checkbox("Show outcome table", value=True)

            case_outcome_df = compute_case_last_activity(events_f)
            outcome_stats = compute_outcome_disparity(
                case_outcome_df=case_outcome_df,
                cases=cases_f,
                sensitive_attr=sensitive_attr,
                ref_group=ref_group,
                target_group=target_group,
                measure=measure,
                band_pp=band_pp,
                band_ratio_low=band_ratio_low,
                band_ratio_high=band_ratio_high,
            )

            if outcome_stats.empty:
                st.info("No outcome statistics available (check filters or group sizes).")
            else:
                # Keep top outcomes by frequency
                outcome_stats = outcome_stats.sort_values("case_support_all", ascending=False).head(int(top_k_outcomes)).copy()

                # Sort by disparity magnitude (unfair first)
                if measure == "Δpp":
                    outcome_stats["_score"] = outcome_stats["delta_pp"].abs()
                else:
                    outcome_stats["_score"] = (outcome_stats["ratio"] - 1.0).abs()

                outcome_stats = outcome_stats.sort_values(
                    by=["is_unfair", "_score", "case_support_all"],
                    ascending=[False, False, False],
                )

                # Render cards
                ncols = 3
                cols_cards = st.columns(ncols)
                for i, r in enumerate(outcome_stats.to_dict("records")):
                    col = cols_cards[i % ncols]

                    p_tgt = float(r["p_tgt"])
                    p_ref = float(r["p_ref"])
                    sup_tgt = int(r["support_tgt"])
                    sup_ref = int(r["support_ref"])
                    n_tgt = int(r["n_tgt"])
                    n_ref = int(r["n_ref"])

                    if measure == "Δpp":
                        metric_line = f"Δpp = {float(r['delta_pp']):+.2f}pp"
                    else:
                        metric_line = "Ratio = NA" if pd.isna(r.get("ratio", np.nan)) else f"Ratio = {float(r['ratio']):.2f}×"

                    # Simple coloring: fair -> green, unfair -> red, unknown -> gray
                    if (measure != "Δpp") and pd.isna(r.get("ratio", np.nan)):
                        bg = "#f2f2f2"
                        bd = "#cccccc"
                    else:
                        if bool(r["is_unfair"]):
                            bg = "#ffe9e9"
                            bd = "#e57373"
                        else:
                            bg = "#e9f7ee"
                            bd = "#66bb6a"

                    html = f"""
                    <div style="
                        background:{bg};
                        border:1px solid {bd};
                        border-radius:14px;
                        padding:14px 14px 12px 14px;
                        margin-bottom:12px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
                    ">
                      <div style="font-size:14px; font-weight:700; margin-bottom:6px;">
                        {r['outcome']}
                      </div>
                      <div style="font-size:12px; margin-bottom:6px;">
                        Target ({tgt_label}): <b>{p_tgt:.3f}</b> ({sup_tgt}/{n_tgt})<br>
                        Reference ({ref_label}): <b>{p_ref:.3f}</b> ({sup_ref}/{n_ref})
                      </div>
                      <div style="font-size:12px;">
                        <b>{metric_line}</b><br>
                        Case frequency (all) = {int(r['case_support_all'])}
                      </div>
                    </div>
                    """
                    with col:
                        st.markdown(html, unsafe_allow_html=True)

                if show_outcome_table:
                    tab = outcome_stats.copy()
                    cols = ["outcome", "case_support_all", "p_tgt", "p_ref"]
                    if measure == "Δpp":
                        cols += ["delta_pp"]
                    else:
                        cols += ["ratio"]
                    cols += ["support_tgt", "n_tgt", "support_ref", "n_ref", "is_unfair"]

                    st.dataframe(
                        tab[cols].drop(columns=["_score"], errors="ignore"),
                        use_container_width=True,
                        hide_index=True,
                    )
