"""app.py - Streamlit UI for the Agent X Campus Energy & Traffic Optimizer.
Run:  streamlit run app.py"""
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (LABELS, N_SLOTS, Campus, Log, guess_columns, parse_target, run_llm_agent,
                  run_rule_agent, synth_energy, synth_traffic, to_hourly)

st.set_page_config(page_title="Agent X - Campus Optimizer", page_icon="🌿", layout="wide")
ICON = {"THINK": "🧠", "PLAN": "📝", "ACT": "⚙️", "IMPROVE": "🔁"}


@st.cache_data
def read_csv(b):
    return pd.read_csv(io.BytesIO(b))


def load_series(upload, kind, label):
    """Upload -> hourly series. Falls back to synthetic data so the demo never breaks."""
    if upload is None:
        return (synth_traffic() if kind == "traffic" else synth_energy()), "synthetic demo data"
    df = read_csv(upload.getvalue())
    tg, vg = guess_columns(df, kind)
    cols = list(df.columns)
    tcol = st.sidebar.selectbox(f"{label}: time column", cols, index=cols.index(tg) if tg in cols else 0, key=kind + "t")
    nums = list(df.select_dtypes("number").columns)
    vcol = st.sidebar.selectbox(f"{label}: value column", nums, index=nums.index(vg) if vg in nums else 0, key=kind + "v")
    return to_hourly(df, tcol, vcol), upload.name


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Data")
up_t = st.sidebar.file_uploader("gate_traffic.csv", type="csv")
up_e = st.sidebar.file_uploader("power_logs.csv", type="csv")
traffic_s, t_src = load_series(up_t, "traffic", "Traffic")
energy_s, e_src = load_series(up_e, "energy", "Energy")

st.sidebar.header("Assumptions (editable)")
cfg = dict(
    bg_peak=st.sidebar.number_input("Non-lecture gate traffic at peak hour (people/h)", 100, 3000, 600, 50),
    capacity=st.sidebar.number_input("Gate capacity (people / 30 min)", 100, 3000, 650, 50),
    avg_size=st.sidebar.slider("Avg students per class group", 40, 120, 75),
    buildings=st.sidebar.slider("Academic buildings", 1, 20, 5),
    essential_frac=st.sidebar.slider("Essential night load (x of lowest hour)", 0.0, 1.0, 0.6),
    recover_frac=st.sidebar.slider("Recoverable share of idle night load", 0.0, 1.0, 0.5),
    grid_factor=st.sidebar.number_input("Grid factor (kg CO2/kWh)", 0.1, 1.5, 0.779, 0.001, format="%.3f"),
    tariff=st.sidebar.number_input("Tariff (Rs/kWh)", 1.0, 30.0, 8.0, 0.5),
)
st.sidebar.header("Agent brain")
use_llm = st.sidebar.toggle("Use Claude (tool calling)", value=False)
api_key = st.sidebar.text_input("Anthropic API key", type="password") if use_llm else ""
model = st.sidebar.text_input("Model", "claude-sonnet-5-5") if use_llm else ""

# ------------------------------------------------------------------ header
st.title("🌿 AGENT X - Campus Energy & Traffic Optimizer")
st.warning(f"**SIMULATION**: traffic = {t_src}, energy = {e_src}. Timetable and rooms are simulated. "
           "Connect real gate counters, smart meters and room bookings to get measured results.")

campus = Campus(traffic_s, energy_s, cfg)
if "plan" in st.session_state:
    campus.plan, campus.used = st.session_state.plan, st.session_state.used

goal = st.text_input("🎯 Goal for the agent", "Reduce gate peak by 20% and save energy")
if st.button("▶ Run agent", type="primary"):
    box = st.empty()

    def render(entries):
        with box.container():
            for e in entries:
                st.markdown(f"{ICON[e['phase']]} **{e['phase']}**: {e['text']}")
                if e["data"]:
                    st.json(e["data"], expanded=False)

    log = Log(on_add=render)
    with st.spinner("Agent is working..."):
        try:
            if use_llm and api_key:
                run_llm_agent(campus, goal, log, api_key, model)
            else:
                if use_llm:
                    st.info("No API key entered: using the built-in rule-based agent.")
                run_rule_agent(campus, goal, log)
        except Exception as ex:
            st.error(f"LLM agent failed ({ex}). Falling back to rule-based agent.")
            log = Log(on_add=render)
            run_rule_agent(campus, goal, log)
    st.session_state.update(plan=campus.plan, used=campus.used, log=log.entries, target=parse_target(goal))
    st.rerun()

# ------------------------------------------------------------------ results
m = campus.metrics() if campus.plan else None
if m:
    c = st.columns(5)
    c[0].metric("Gate peak (30 min)", m["peak_after"], f"-{m['reduction_pct']}%  (was {m['peak_before']})", delta_color="inverse")
    c[1].metric("Lectures re-timed", f"{m['classes_moved']}/{m['classes_total']}")
    c[2].metric("Energy saved", f"{m['total_kwh_saved_per_day']} kWh/day")
    c[3].metric("CO₂ avoided", f"{m['co2_kg_avoided_per_day']} kg/day")
    c[4].metric("Cost saved", f"₹{m['cost_saved_inr_per_day']}/day")
    if m["violations"]:
        st.error(f"{m['violations']} constraint violations!")
    else:
        st.success("0 constraint violations: no room / faculty / student-group clashes, only approved slots.")
else:
    st.info("Press **Run agent** to generate and simulate a plan. Baseline view below.")

tabs = st.tabs(["🧠 Agent log", "🚦 Gate traffic", "⚡ Energy", "📄 Report", "🗓 Timetable"])

with tabs[0]:
    for e in st.session_state.get("log", []):
        st.markdown(f"{ICON[e['phase']]} **{e['phase']}**: {e['text']}")
        if e["data"]:
            st.json(e["data"], expanded=False)
    if "log" not in st.session_state:
        st.caption("Agent steps appear here (Think → Plan → Act → Improve).")

with tabs[1]:
    from core import gate_load
    fig = go.Figure()
    fig.add_bar(x=LABELS, y=gate_load(campus.tt0, campus.bg), name="Before (current timetable)", marker_color="#ef4444")
    if m:
        fig.add_bar(x=LABELS, y=m["_after"], name="After (agent plan)", marker_color="#22c55e")
    fig.add_hline(y=cfg["capacity"], line_dash="dash", annotation_text="gate capacity")
    fig.update_layout(barmode="group", title="Simulated gate arrivals per 30 min", xaxis_title="time", yaxis_title="people")
    st.plotly_chart(fig, width="stretch")
    hp = campus.tprof
    f2 = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color="#3b82f6"))
    f2.update_layout(title="Average hourly traffic from the traffic CSV (real shape)", yaxis_title="volume")
    st.plotly_chart(f2, width="stretch")

with tabs[2]:
    ep = campus.eprof
    colors = ["#f59e0b" if (h >= 22 or h <= 5) else "#3b82f6" for h in ep.index]
    f3 = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in ep.index], y=ep.values, marker_color=colors))
    f3.update_layout(title="Average hourly energy from the power CSV (orange = night, possible idle waste)", yaxis_title="kWh")
    st.plotly_chart(f3, width="stretch")
    if m:
        f4 = go.Figure()
        f4.add_scatter(x=LABELS, y=m["_h0"], name="HVAC baseline (07:30-17:30 all rooms)", line=dict(color="#ef4444"))
        f4.add_scatter(x=LABELS, y=m["_h1"], name="HVAC optimized (only when needed)", line=dict(color="#22c55e"))
        f4.update_layout(title="Simulated classroom HVAC energy per 30 min", yaxis_title="kWh")
        st.plotly_chart(f4, width="stretch")
    st.json(campus.detect_energy_waste(), expanded=False)

with tabs[3]:
    if m:
        rep = campus.green_report()["markdown"]
        st.markdown(rep)
        st.download_button("⬇ Download report (.md)", rep, "green_report.md")
    else:
        st.caption("Run the agent to generate the report.")

with tabs[4]:
    rows = []
    for c0, c1 in zip(campus.tt0, campus.plan or campus.tt0):
        rows.append({"class": c0["id"], "group": c0["group"], "room": c0["room"], "faculty": c0["faculty"], "students": c0["size"],
                     "old start": LABELS[c0["start"]], "new start": LABELS[c1["start"]], "moved": c0["start"] != c1["start"]})
    df = pd.DataFrame(rows)
    st.dataframe(df.sort_values("moved", ascending=False), width="stretch", hide_index=True)
    st.caption("Simulated timetable. Changes are recommendations that need human approval.")
