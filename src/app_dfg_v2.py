import os
import math
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

# HAS_PLOTLY_EVENTS = False  # legacy flag (DFG view uses Streamlit on_select)


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


def wrap_label(text: str, max_chars: int = 16) -> str:
    """Wrap label into 1~2 lines to fit inside a node box."""
    s = str(text)
    if len(s) <= max_chars:
        return s
    # try to wrap by spaces first
    parts = s.split(" ")
    if len(parts) == 1:
        return s[:max_chars] + "<br>" + s[max_chars:]
    line1 = []
    cur = 0
    for p in parts:
        add = len(p) + (1 if line1 else 0)
        if cur + add <= max_chars:
            line1.append(p)
            cur += add
        else:
            break
    if not line1:
        return s[:max_chars] + "<br>" + s[max_chars:]
    rest = " ".join(parts[len(line1):])
    return " ".join(line1) + "<br>" + rest



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
    node_code_map,
    highlight_activity=None,
):
    """
    Build a DFG (Directly-Follows Graph) view using Plotly.
    - Edge width = case frequency (case_support_all)
    - Unfair edges = red
    - Badge (!) = yellow marker placed on the edge (click-to-select)
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

        head = f"<b>{s} → {d}</b><br>"
        base = f"Case frequency: {int(w)}<br>"
        probs = (
            f"p(target={tgt_label}): {float(r['p_tgt']):.3f} ({int(r['support_tgt'])}/{int(r['n_tgt'])})<br>"
            f"p(ref={ref_label}): {float(r['p_ref']):.3f} ({int(r['support_ref'])}/{int(r['n_ref'])})<br>"
        )
        if measure == "Δpp":
            metric = f"Δpp: <b>{float(r['delta_pp']):+.1f}pp</b><br>"
        else:
            rv = r["ratio"]
            metric = f"Ratio: <b>{'NA' if pd.isna(rv) else f'{float(rv):.2f}×'}</b><br>"
        tail = f"Unfair: {bool(r['is_unfair'])}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color),
                hovertemplate=head + base + probs + metric + tail,
                showlegend=False,
            )
        )

    # --- nodes (activity names)
    node_text = [wrap_label(str(a), max_chars=18) for a in nodes_sorted]
    fig.add_trace(
        go.Scatter(
            x=[float(x_norm.get(a, 0.5)) for a in nodes_sorted],
            y=[float(y_norm.get(a, 0.5)) for a in nodes_sorted],
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            marker=dict(size=28, color="rgba(235,235,235,1.0)", line=dict(width=1, color="rgba(80,80,80,0.6)")),
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
        t = 0.55
        spread = 0.020

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
                hovertemplate="Click badge to view unfairness details<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        clickmode="event+select",
        dragmode="select",
    )
    return fig, badge_df


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
        options = ["(none)"] + [f"{act_to_code[a]} — {a}" for a in nodes_sorted]
        sel = st.selectbox("Highlight activity (incoming + outgoing edges)", options=options)

        highlight_activity = None
        if sel != "(none)":
            highlight_activity = sel.split(" — ", 1)[1]


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

        fig, badge_df = build_dfg_figure(
            edge_stats=edge_stats,
            nodes_sorted=nodes_sorted,
            x_norm=x_norm,
            y_norm=y_norm,
            min_support=min_support,
            max_edges=max_edges,
            measure=measure,
            ref_label=ref_label,
            tgt_label=tgt_label,
            node_code_map=act_to_code,
            highlight_activity=highlight_activity
        )



        if fig is None:
            st.warning("No edges to draw with current settings (try lower Min support).")
        else:
            if "selected_badge_edge" not in st.session_state:
                st.session_state.selected_badge_edge = None

            st.write("HAS_PLOTLY_EVENTS =", HAS_PLOTLY_EVENTS)

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
                # st.write("plotly_events clicked =", clicked)
            

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
# Show detail card if badge clicked
            if st.session_state.selected_badge_edge:
                r = st.session_state.selected_badge_edge
                edge_name = f"{r['src']} → {r['dst']}"
                st.markdown("#### Unfairness Details (badge click)")
                if measure == "Δpp":
                    st.error(
                        f"**{edge_name}**  \n"
                        f"Target ({tgt_label}) vs Reference ({ref_label})  \n"
                        f"Δpp = **{float(r['delta_pp']):+.2f}pp**  \n"
                        f"p_tgt = {float(r['p_tgt']):.3f} ({int(r['support_tgt'])}/{int(r['n_tgt'])}), "
                        f"p_ref = {float(r['p_ref']):.3f} ({int(r['support_ref'])}/{int(r['n_ref'])})  \n"
                        f"case frequency (all) = {int(r['case_support_all'])}"
                    )
                else:
                    st.error(
                        f"**{edge_name}**  \n"
                        f"Target ({tgt_label}) vs Reference ({ref_label})  \n"
                        f"Ratio = **{float(r['ratio']):.3f}×**  \n"
                        f"p_tgt = {float(r['p_tgt']):.3f} ({int(r['support_tgt'])}/{int(r['n_tgt'])}), "
                        f"p_ref = {float(r['p_ref']):.3f} ({int(r['support_ref'])}/{int(r['n_ref'])})  \n"
                        f"case frequency (all) = {int(r['case_support_all'])}"
                    )
            #     cols_legend[i % 3].markdown(item)

            # keep the edge table
            if show_table and not edge_stats.empty:
                st.markdown("### Edge Disparity Table")
                tbl = edge_stats.copy()
                tbl["edge"] = tbl["src"] + " → " + tbl["dst"]
                if measure == "Δpp":
                    tbl["sort_key"] = tbl["delta_pp"].abs()
                    cols = ["edge", "case_support_all", "p_tgt", "p_ref", "delta_pp", "is_unfair"]
                else:
                    tbl["sort_key"] = (tbl["ratio"] - 1.0).abs()
                    cols = ["edge", "case_support_all", "p_tgt", "p_ref", "ratio", "is_unfair"]

                tbl = tbl.sort_values(["is_unfair", "sort_key", "case_support_all"], ascending=[False, False, False])
                st.dataframe(tbl[cols].head(50), use_container_width=True)

        

    st.divider()
    st.markdown("### ② Outcome Disparity Dashboard")
    st.info("这里先留占位：之后放 outcome cards / 指标对比 / group comparison 等。")

st.divider()
st.markdown("### Evidence Tray")
st.info("这里先留占位：之后放 selection history / saved comparisons / export evidence 等。")
