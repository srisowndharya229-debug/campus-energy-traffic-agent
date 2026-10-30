"""core.py - data loading, simulation, optimizer, agent tools and agent loops.
Everything is a SIMULATION built on proxy datasets (see README)."""
import json
import random
import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
SLOT_MIN = 30                      # time resolution
DAY_START_H = 6                    # slot 0 = 06:00
N_SLOTS = 28                       # 06:00 -> 20:00
APPROVED_START = range(4, 22)      # lectures may start 08:00 .. 16:30
ROOMS = [f"R{i:02d}" for i in range(1, 15)]
FACULTY = [f"F{i:02d}" for i in range(1, 25)]
GROUPS = [f"{d}-Y{y}" for d in ("AI", "CSE", "ECE", "MECH") for y in (1, 2, 3)]
ROOM_KW = 4.0                      # assumed AC + lights load per occupied room (kW)
GAP_PEOPLE = 1                     # 30-min movement gap for same group / faculty
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def slot_label(s):
    m = DAY_START_H * 60 + s * SLOT_MIN
    return f"{m // 60:02d}:{m % 60:02d}"


LABELS = [slot_label(s) for s in range(N_SLOTS)]


# ------------------------------------------------------------------ data
def _is_datetime_col(series):
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def guess_columns(df, kind):
    """Return (time_col, value_col) guesses for a raw dataframe."""
    tcol = None
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time", "stamp")) and _is_datetime_col(df[c]):
            tcol = c
            break
    nums = [c for c in df.select_dtypes("number").columns if c != tcol]
    keys = {"traffic": ("traffic", "volume", "count", "vehicle"),
            "energy": ("kwh", "energy", "consum", "power", "meter")}[kind]
    vcol = next((c for c in nums if any(k in c.lower() for k in keys)), nums[0] if nums else None)
    return tcol, vcol


def to_hourly(df, tcol, vcol):
    s = pd.DataFrame({"t": pd.to_datetime(df[tcol], errors="coerce"),
                      "v": pd.to_numeric(df[vcol], errors="coerce")}).dropna()
    s = s.groupby("t")["v"].mean().sort_index()          # merges duplicate timestamps
    return s.resample("1h").mean().dropna()


def hourly_profile(s):
    p = s.groupby(s.index.hour).mean().reindex(range(24))
    return p.interpolate().bfill().ffill()


def synth_traffic(seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=24 * 28, freq="h")
    h = idx.hour.values + 0.0
    v = (300 + 3000 * np.exp(-((h - 8.5) / 1.1) ** 2) + 2400 * np.exp(-((h - 17) / 1.6) ** 2)
         + 1300 * np.exp(-((h - 13) / 2.0) ** 2))
    v = v * np.where(idx.weekday >= 5, 0.45, 1.0) * rng.normal(1, 0.06, len(idx))
    return pd.Series(v, index=idx)


def synth_energy(seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=24 * 28, freq="h")
    h = idx.hour.values
    v = np.where((h >= 8) & (h <= 17), 160.0, 75.0)       # day load vs wasteful night load
    v = v * np.where(idx.weekday >= 5, 0.6, 1.0) * rng.normal(1, 0.05, len(idx))
    return pd.Series(v, index=idx)


# ------------------------------------------------------------------ timetable
def _conflict(tt, cand, skip=None):
    a0, a1 = cand["start"], cand["start"] + cand["dur"]
    for j, o in enumerate(tt):
        if j == skip:
            continue
        b0, b1 = o["start"], o["start"] + o["dur"]
        if o["room"] == cand["room"] and a0 < b1 and b0 < a1:
            return True
        g = GAP_PEOPLE
        if (o["faculty"] == cand["faculty"] or o["group"] == cand["group"]) and a0 < b1 + g and b0 < a1 + g:
            return True
    return False


def make_timetable(avg_size=75, seed=7):
    """SIMULATED timetable: 12 student groups x 4 lectures, heavy 09:00 cluster."""
    rng = random.Random(seed)
    tt = []
    for g in GROUPS:
        size = max(30, int(rng.gauss(avg_size, 10)))
        for base in (6, 10, 15, 18):                       # 09:00, 11:00, 13:30, 15:00
            for _ in range(60):
                st = base + (rng.choice([0, 0, 0, 0, -1]) if base == 6 else rng.choice([0, 0, 1, -1]))
                cand = dict(id=len(tt), group=g, faculty=rng.choice(FACULTY), room=rng.choice(ROOMS),
                            size=size, start=st, orig=st, dur=2)
                if not _conflict(tt, cand):
                    tt.append(cand)
                    break
    for i, c in enumerate(tt):
        c["id"] = i
    return tt


def validate(tt):
    """Count hard-constraint violations in a timetable."""
    bad = 0
    for i, c in enumerate(tt):
        if c["start"] not in APPROVED_START:
            bad += 1
        if _conflict(tt[:i] + tt[i + 1:], c):
            bad += 1
    return bad


# ------------------------------------------------------------------ models
def background_per_slot(tprof, bg_peak):
    """Non-lecture gate traffic (staff, visitors, vehicles) from the traffic CSV, scaled to campus."""
    p = tprof / tprof.max() * bg_peak
    return np.array([p[((DAY_START_H * 60 + s * SLOT_MIN) // 60) % 24] / (60 / SLOT_MIN)
                     for s in range(N_SLOTS)])


def gate_load(tt, bg):
    """People arriving at the gate per 30-min slot: background + first-lecture arrivals of each group."""
    load = bg.copy()
    first = {}
    for c in tt:
        if c["group"] not in first or c["start"] < first[c["group"]]["start"]:
            first[c["group"]] = c
    for c in first.values():
        for off, w in ((1, 0.6), (2, 0.4)):                # 60% arrive 30 min before, 40% 60 min before
            k = c["start"] - off
            if 0 <= k < N_SLOTS:
                load[k] += c["size"] * w
    return load


def hvac_kwh(tt, mode, precool=1, gap_keep=1):
    """HVAC energy per slot. baseline = all rooms on 07:30-17:30; optimized = only when needed."""
    on = {r: np.zeros(N_SLOTS, bool) for r in ROOMS}
    if mode == "baseline":
        for r in ROOMS:
            on[r][3:23] = True
    else:
        for c in tt:
            on[c["room"]][max(0, c["start"] - precool): c["start"] + c["dur"]] = True
        for r in ROOMS:                                    # keep ON across very short gaps
            idx = np.where(on[r])[0]
            for x, y in zip(idx[:-1], idx[1:]):
                if 1 < y - x <= gap_keep + 1:
                    on[r][x:y] = True
    return sum(on[r].astype(float) for r in ROOMS) * ROOM_KW * SLOT_MIN / 60


# ------------------------------------------------------------------ optimizer
def optimize(tt0, bg, max_shift, max_moves, gamma):
    """Local search: minimise  peak + 0.02*load_variance_proxy + gamma*disruption  under hard constraints."""
    tt = [dict(c, start=c["orig"]) for c in tt0]

    def J():
        L = gate_load(tt, bg)
        dis = sum(abs(c["start"] - c["orig"]) for c in tt)
        return L.max() + 0.02 * (L ** 2).sum() / max(L.sum(), 1) + gamma * dis

    best, moved, improved = J(), set(), True
    while improved:
        improved = False
        for i in sorted(range(len(tt)), key=lambda k: -tt[k]["size"]):
            c = tt[i]
            if i not in moved and len(moved) >= max_moves:
                continue
            cur, best_s = c["start"], c["start"]
            for d in range(-max_shift, max_shift + 1):
                ns = c["orig"] + d
                if ns == cur or ns not in APPROVED_START or _conflict(tt, dict(c, start=ns), skip=i):
                    continue
                c["start"] = ns
                j = J()
                c["start"] = cur
                if j < best - 1e-9:
                    best, best_s = j, ns
            if best_s != cur:
                c["start"] = best_s
                improved = True
                moved.add(i) if best_s != c["orig"] else moved.discard(i)
    return tt


# ------------------------------------------------------------------ the Campus (holds state + tools)
class Campus:
    def __init__(self, traffic_s, energy_s, cfg):
        self.traffic_s, self.energy_s, self.cfg = traffic_s, energy_s, cfg
        self.tprof, self.eprof = hourly_profile(traffic_s), hourly_profile(energy_s)
        self.bg = background_per_slot(self.tprof, cfg["bg_peak"])
        self.tt0 = make_timetable(cfg["avg_size"])
        self.plan, self.used = None, {}

    # ---- tool 1
    def find_peak_hours(self):
        top = self.tprof.sort_values(ascending=False).head(3)
        load = gate_load(self.tt0, self.bg)
        wd = self.traffic_s.groupby(self.traffic_s.index.dayofweek).mean()
        over = [LABELS[i] for i in range(N_SLOTS) if load[i] > self.cfg["capacity"]]
        return {"csv_top_hours": {f"{int(h):02d}:00": round(float(v), 1) for h, v in top.items()},
                "csv_busiest_weekday": DAYS[int(wd.idxmax())],
                "simulated_peak_slot": LABELS[int(load.argmax())],
                "simulated_peak_arrivals_per_30min": round(float(load.max())),
                "gate_capacity_per_30min": self.cfg["capacity"],
                "slots_over_capacity": over,
                "note": "traffic shape from CSV; lecture arrivals from SIMULATED timetable"}

    # ---- tool 2
    def detect_energy_waste(self):
        s, h = self.energy_s, self.energy_s.index.hour
        night, day = s[(h >= 22) | (h <= 5)], s[(h >= 9) & (h <= 16)]
        essential = self.cfg["essential_frac"] * float(self.eprof.min())
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        waste = float(sum(max(0.0, self.eprof[x] - essential) for x in night_hours))
        mad = float((night - night.median()).abs().median()) or 1.0
        thr = float(night.median() + 3 * mad)
        by_day = night.groupby(night.index.date).mean().sort_values(ascending=False).head(3)
        return {"night_mean_kwh": round(float(night.mean()), 1), "day_mean_kwh": round(float(day.mean()), 1),
                "night_to_day_ratio": round(float(night.mean() / day.mean()), 2),
                "idle_waste_kwh_per_day_per_building": round(waste, 1),
                "abnormal_night_readings": int((night > thr).sum()),
                "worst_nights": {str(k): round(float(v), 1) for k, v in by_day.items()},
                "note": "flagged for investigation, not proven waste"}

    # ---- tool 3
    def run_optimizer(self, max_shift=1, max_moves=12, disruption_weight=15):
        max_shift = int(np.clip(max_shift, 0, 6))
        max_moves = int(np.clip(max_moves, 0, len(self.tt0)))
        gamma = float(np.clip(disruption_weight, 0, 100))
        self.plan = optimize(self.tt0, self.bg, max_shift, max_moves, gamma)
        self.used = dict(max_shift_slots=max_shift, max_shift_minutes=max_shift * SLOT_MIN,
                         max_moves=max_moves, disruption_weight=gamma)
        m = self.metrics()
        return {"params": self.used, "classes_moved": m["classes_moved"],
                "peak_before": m["peak_before"], "peak_after": m["peak_after"],
                "constraint_violations": m["violations"]}

    # ---- metrics (used by tools 4 and 5)
    def metrics(self):
        before = gate_load(self.tt0, self.bg)
        after = gate_load(self.plan, self.bg)
        cap, c = self.cfg["capacity"], self.cfg
        h0, h1 = hvac_kwh(self.tt0, "baseline"), hvac_kwh(self.plan, "optimized")
        idle = self.detect_energy_waste()["idle_waste_kwh_per_day_per_building"]
        night_saved = idle * c["recover_frac"] * c["buildings"]
        hvac_saved = float(h0.sum() - h1.sum())
        total = hvac_saved + night_saved
        moved = [x for x in self.plan if x["start"] != x["orig"]]
        return {"peak_before": round(float(before.max())), "peak_after": round(float(after.max())),
                "reduction_pct": round(float((before.max() - after.max()) / before.max() * 100), 1),
                "over_cap_slots_before": int((before > cap).sum()), "over_cap_slots_after": int((after > cap).sum()),
                "classes_moved": len(moved), "classes_total": len(self.plan),
                "avg_shift_min": round(float(np.mean([abs(x["start"] - x["orig"]) for x in moved]) * SLOT_MIN), 1) if moved else 0,
                "violations": validate(self.plan),
                "hvac_kwh_baseline": round(float(h0.sum()), 1), "hvac_kwh_optimized": round(float(h1.sum()), 1),
                "hvac_kwh_saved": round(hvac_saved, 1), "night_idle_kwh_saved": round(float(night_saved), 1),
                "total_kwh_saved_per_day": round(float(total), 1),
                "co2_kg_avoided_per_day": round(float(total * c["grid_factor"]), 1),
                "cost_saved_inr_per_day": round(float(total * c["tariff"])),
                "_before": before, "_after": after, "_h0": h0, "_h1": h1}

    # ---- tool 4
    def simulate_plan(self):
        if self.plan is None:
            return {"error": "call run_optimizer first"}
        return {k: v for k, v in self.metrics().items() if not k.startswith("_")}

    # ---- tool 5
    def green_report(self):
        if self.plan is None:
            return {"error": "call run_optimizer first"}
        m, c = self.simulate_plan(), self.cfg
        md = f"""## Daily Green-Footprint Report  (SIMULATION)
| Metric | Value |
|---|---|
| Gate peak (per 30 min) | {m['peak_before']} -> {m['peak_after']}  (**-{m['reduction_pct']}%**) |
| Slots over gate capacity | {m['over_cap_slots_before']} -> {m['over_cap_slots_after']} |
| Lectures re-timed | {m['classes_moved']} of {m['classes_total']} (avg shift {m['avg_shift_min']} min) |
| HVAC energy saved | {m['hvac_kwh_saved']} kWh |
| Night idle load recovered | {m['night_idle_kwh_saved']} kWh |
| **Total energy saved** | **{m['total_kwh_saved_per_day']} kWh/day** |
| CO2 avoided | {m['co2_kg_avoided_per_day']} kg/day (factor {c['grid_factor']} kg/kWh) |
| Cost saved | Rs {m['cost_saved_inr_per_day']}/day (tariff Rs {c['tariff']}/kWh) |

**Assumptions:** simulated timetable and rooms; {ROOM_KW} kW per occupied room; baseline HVAC on 07:30-17:30;
{int(c['recover_frac']*100)}% of night idle load recoverable; {c['buildings']} building(s); proxy CSV datasets.
Recommendations need human approval before any timetable or HVAC change.
"""
        return {"markdown": md, **{k: m[k] for k in ("total_kwh_saved_per_day", "co2_kg_avoided_per_day", "cost_saved_inr_per_day")}}

    # ---- dispatcher
    def call(self, name, **kw):
        fn = {"find_peak_hours": self.find_peak_hours, "detect_energy_waste": self.detect_energy_waste,
              "run_optimizer": self.run_optimizer, "simulate_plan": self.simulate_plan,
              "green_report": self.green_report}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**kw)
        except Exception as e:                              # never crash the agent loop
            return {"error": str(e)}


TOOL_SPECS = [
    {"name": "find_peak_hours", "description": "Analyse gate traffic: busiest hours/weekday from CSV and the simulated peak slot / over-capacity slots.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "detect_energy_waste", "description": "Analyse power logs: night vs day load, idle waste kWh/day, abnormal night readings.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_optimizer", "description": "Re-time lectures to flatten the gate peak under hard constraints (no room/faculty/group clash, approved slots). Larger max_shift / max_moves and smaller disruption_weight = more reduction but more disruption.",
     "input_schema": {"type": "object", "properties": {
         "max_shift": {"type": "integer", "description": "max shift in 30-min slots (1-6)"},
         "max_moves": {"type": "integer", "description": "max number of lectures to move"},
         "disruption_weight": {"type": "number", "description": "penalty per slot moved (0-100)"}}}},
    {"name": "simulate_plan", "description": "Before/after metrics of the current plan: peak reduction %, kWh, CO2, cost, constraint violations.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "green_report", "description": "Final daily green-footprint report (markdown) for the current plan.",
     "input_schema": {"type": "object", "properties": {}}},
]


# ------------------------------------------------------------------ agents
class Log:
    """Collects Think/Plan/Act/Improve steps; `on_add` lets Streamlit render them live."""
    def __init__(self, on_add=None):
        self.entries, self.on_add = [], on_add

    def add(self, phase, text, data=None):
        self.entries.append({"phase": phase, "text": text, "data": data})
        if self.on_add:
            self.on_add(self.entries)


def parse_target(goal, default=20.0):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", goal)
    return float(m.group(1)) if m else default


def _short(d):
    return {k: v for k, v in d.items() if k != "markdown"}


def run_rule_agent(campus, goal, log):
    """Deterministic Think -> Plan -> Act -> Improve loop (works with no API key)."""
    target = parse_target(goal)
    log.add("THINK", f"Goal: '{goal}'. Target gate-peak reduction = {target}%. First I need to understand the data.")
    log.add("ACT", "find_peak_hours()", _short(campus.call("find_peak_hours")))
    w = campus.call("detect_energy_waste")
    log.add("ACT", "detect_energy_waste()", w)
    log.add("THINK", f"Night load is {w['night_to_day_ratio']}x day load; idle waste about {w['idle_waste_kwh_per_day_per_building']} kWh/day/building.")
    ladder = [(1, 12, 20), (2, 20, 12), (3, 30, 8), (4, 48, 4), (6, 48, 0)]
    reached = False
    for n, (ms, mm, g) in enumerate(ladder, 1):
        log.add("PLAN", f"Attempt {n}: shift lectures up to {ms * SLOT_MIN} min, move at most {mm} lectures, disruption penalty {g}.")
        log.add("ACT", f"run_optimizer(max_shift={ms}, max_moves={mm}, disruption_weight={g})", campus.call("run_optimizer", max_shift=ms, max_moves=mm, disruption_weight=g))
        sim = campus.call("simulate_plan")
        log.add("ACT", "simulate_plan()", sim)
        if sim["violations"] == 0 and sim["reduction_pct"] >= target:
            log.add("IMPROVE", f"Reduction {sim['reduction_pct']}% >= target {target}% with 0 constraint violations. Plan accepted.")
            reached = True
            break
        log.add("IMPROVE", f"Reduction {sim['reduction_pct']}% < target {target}%. Relaxing limits and retrying.")
    if not reached:
        log.add("IMPROVE", "Target not reachable within constraints; keeping the best plan found and reporting honestly.")
    rep = campus.call("green_report")
    log.add("ACT", "green_report()", _short(rep))
    log.add("THINK", "Done. All numbers above come from simulation code, not from the language model.")


def run_llm_agent(campus, goal, log, api_key, model):
    """Claude tool-calling agent. It chooses the tools; numbers still come from our code."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    system = ("You are Agent X, an autonomous campus sustainability agent. Work in a Think -> Plan -> Act -> Improve loop. "
              "Start with find_peak_hours and detect_energy_waste, then run_optimizer and simulate_plan. If the reduction target "
              "or constraint_violations==0 is not met, change max_shift / max_moves / disruption_weight and retry (max 5 attempts). "
              "Finish with green_report and a short summary for the campus administrator. NEVER invent numbers: quote only tool "
              "results. State clearly that results are a simulation on proxy data.")
    msgs = [{"role": "user", "content": goal}]
    for _ in range(16):
        r = client.messages.create(model=model, max_tokens=1500, system=system, tools=TOOL_SPECS, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        for b in r.content:
            if b.type == "text" and b.text.strip():
                log.add("THINK", b.text.strip())
        if r.stop_reason != "tool_use":
            break
        results = []
        for b in r.content:
            if b.type == "tool_use":
                out = campus.call(b.name, **(b.input or {}))
                log.add("ACT", f"{b.name}({json.dumps(b.input)})", _short(out))
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o))})
        msgs.append({"role": "user", "content": results})
    if campus.plan is None:
        log.add("IMPROVE", "LLM did not produce a plan; falling back to the rule-based agent.")
        run_rule_agent(campus, goal, log)
