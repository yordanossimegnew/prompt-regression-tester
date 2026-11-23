"""Streamlit UI for the Prompt Regression Tester.

    uvicorn main:app            # terminal 1
    streamlit run ui/app.py     # terminal 2
"""

from __future__ import annotations

import html
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")
# Reference palette: categorical slots 1 and 2 (validated pair); text stays in neutral ink.
BASELINE_COLOR, CANDIDATE_COLOR = "#2a78d6", "#eb6834"
MUTED, GRID = "#52514e", "#e7e6e2"
BANNER = {"pass": ("✅ PASS: no regression detected", "#166534", "#dcfce7"),
          "improvement": ("🚀 IMPROVEMENT", "#166534", "#dcfce7"),
          "inconclusive": ("⚠️ INCONCLUSIVE: some cases dropped, not enough evidence", "#854d0e", "#fef9c3"),
          "regression": ("❌ REGRESSION", "#991b1b", "#fee2e2")}
STATUS = {"regressed": "🔴 regressed", "worse": "🟠 worse (not significant)", "same": "⚪ same",
          "better": "🟢 better", "fixed": "🟢 fixed"}

st.set_page_config(page_title="Prompt Regression Tester", page_icon="🧭", layout="wide")
st.markdown("""<style>
.banner {padding:14px 18px;border-radius:10px;font-size:1.3rem;font-weight:700;margin:6px 0 10px}
.out {font:12.5px/1.5 Consolas,monospace;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px;white-space:pre-wrap}
.ok {color:#15803d} .bad {color:#b91c1c}
.diff {font:12.5px/1.45 Consolas,monospace;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}
.diff div {white-space:pre-wrap;padding:0 10px} .diff .a {background:#ecfdf5} .diff .d {background:#fef2f2} .diff .h {color:#6366f1}
</style>""", unsafe_allow_html=True)


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=60, **kw)
        if "text/plain" in r.headers.get("content-type", ""):
            return r.text
        body = r.json()
        return body if r.ok else {"_error": body.get("detail", r.text)}
    except requests.RequestException as exc:
        return {"_error": str(exc)}


def pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


def case_chart(cases: list[dict]):
    rows = []
    for c in cases:
        for side, label in (("baseline", "Baseline"), ("candidate", "Candidate")):
            k, n = map(int, c[side].split("/"))
            rows.append({"case": c["id"], "version": label, "pass rate": k / n if n else 0})
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("case:N", title=None, sort=[c["id"] for c in cases], axis=alt.Axis(labelColor=MUTED, labelOverlap=False)),
        yOffset=alt.YOffset("version:N", sort=["Baseline", "Candidate"]),
        x=alt.X("pass rate:Q", scale=alt.Scale(domain=[0, 1]), title="Pass rate across repeats",
                axis=alt.Axis(format=".0%", gridColor=GRID, labelColor=MUTED, titleColor=MUTED)),
        color=alt.Color("version:N", scale=alt.Scale(domain=["Baseline", "Candidate"], range=[BASELINE_COLOR, CANDIDATE_COLOR]),
                        legend=alt.Legend(orient="top", title=None, labelColor=MUTED)),
        tooltip=["case:N", "version:N", alt.Tooltip("pass rate:Q", format=".0%")],
    ).properties(height=alt.Step(13)).configure_view(strokeWidth=0)  # one step per grouped bar


health = api("GET", "/health")
ok = isinstance(health, dict) and "_error" not in health
prompts = api("GET", "/prompts") if ok else {}
suites = api("GET", "/suites") if ok else []

with st.sidebar:
    st.title("🧭 Prompt Regression")
    st.caption("Catch prompt regressions before they ship")
    if not ok:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn main:app`.", icon="🚫")
    else:
        st.markdown(f"**Judge:** `{health['judge_model']}`")
        st.markdown("### Prompts (from git)")
        for name, versions in (prompts or {}).items():
            st.markdown(f"- **{name}**: " + " → ".join(f"`{v}`" for v in versions))
        st.markdown("### Suites")
        for s in suites or []:
            st.markdown(f"- **{s['name']}**: {s.get('cases', '?')} cases × {s.get('repeats', '?')} repeats")

tab_run, tab_report, tab_diff = st.tabs(["▶️ Run", "📋 Report", "🔀 Prompt diff"])

with tab_run:
    if suites:
        suite = st.selectbox("Suite", [s["name"] for s in suites])
        meta = next(s for s in suites if s["name"] == suite)
        versions = prompts.get(meta["prompt"], [])
        c1, c2, c3 = st.columns(3)
        baseline = c1.selectbox("Baseline", versions, index=max(0, len(versions) - 2))
        candidate = c2.selectbox("Candidate", versions, index=len(versions) - 1)
        repeats = c3.number_input("Repeats per case", 1, 20, meta.get("repeats", 4))
        st.caption(f"{meta['cases']} cases × {repeats} repeats × 2 versions = {meta['cases'] * repeats * 2} outputs "
                   "(unchanged prompt versions are served from the cache)")
        if st.button("Run regression test", type="primary", disabled=baseline == candidate):
            res = api("POST", "/runs", json={"suite": suite, "baseline": baseline, "candidate": candidate,
                                             "repeats": int(repeats)})
            st.error(res["_error"]) if "_error" in res else st.success("Started. Open the Report tab.")


@st.fragment(run_every=1.5)
def watch_progress(run_id: str):
    """Only this small block refreshes while a run is in progress; when it ends, re-render the page once."""
    run = api("GET", f"/runs/{run_id}")
    if run.get("status") == "running":
        st.info(f"Running: {run['progress'].get('stage', '')}…")
    else:
        st.rerun()


def report_panel():
    runs = api("GET", "/runs")
    if not isinstance(runs, list) or not runs:
        st.info("No runs yet.")
        return
    labels = {r["id"]: f"{r['prompt']}: {r['baseline']} → {r['candidate']} · started {r['created_at'][11:19]}" for r in runs}
    run_id = st.selectbox("Run", list(labels), format_func=labels.get, key="run_pick")
    run = api("GET", f"/runs/{run_id}")
    if run["status"] == "running":
        watch_progress(run_id)
        return
    if run["status"] == "failed":
        st.error(run["error"])
        return
    s = run["summary"]
    text, fg, bg = BANNER[s["verdict"]]
    st.markdown(f'<div class="banner" style="color:{fg};background:{bg}">{text}</div>', unsafe_allow_html=True)
    for reason in s["reasons"]:
        st.markdown(f"- {reason}")

    pr, g = s["pass_rate"], s["graders"]
    m = st.columns(4)
    m[0].metric("Pass rate", f"{pr['candidate']:.0%}", f"{pr['delta']:+.0%} vs baseline")
    if g["judge"]["baseline"] is not None:
        m[1].metric("Judge pass rate", f"{g['judge']['candidate']:.0%}",
                    f"{g['judge']['candidate'] - g['judge']['baseline']:+.0%}")
    pw = g["pairwise"]
    m[2].metric("Pairwise wins (candidate–baseline)", f"{pw['candidate_wins']}–{pw['baseline_wins']}")
    m[2].caption(f"{pw['ties']} ties · sign test p={pw['p_worse']:.3f}")
    cal = s["calibration"]
    if cal.get("items"):
        m[3].metric("Judge calibration", f"{cal['agreement']:.0%}")
        m[3].caption(("✓ trusted" if cal["trusted"] else "✗ not trusted: excluded from the verdict")
                     + f" ({cal['items']} known examples)")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Pass rate by case")
        st.altair_chart(case_chart(s["cases"]), width="stretch")
    with right:
        st.markdown("#### Cases")
        st.dataframe(pd.DataFrame([{"case": c["id"], "baseline": c["baseline"], "candidate": c["candidate"],
                                    "status": STATUS[c["status"]], "p (worse)": c["p_worse"]} for c in s["cases"]]),
                     hide_index=True, width="stretch")

    st.markdown("#### What changed")
    outputs = run["outputs"]
    pairs = {(p["case_id"], p["repeat"]): p for p in run["pairwise"]}
    for c in [c for c in s["cases"] if c["status"] != "same"]:
        with st.expander(f"{STATUS[c['status']]} · {c['id']} · {c['baseline']} → {c['candidate']}", expanded=c["status"] == "regressed"):
            b = next(o for o in outputs if o["case_id"] == c["id"] and o["side"] == "baseline" and o["repeat"] == 0)
            k = next(o for o in outputs if o["case_id"] == c["id"] and o["side"] == "candidate" and o["repeat"] == 0)
            cols = st.columns(2)
            for col, o, title in ((cols[0], b, f"Baseline {run['baseline']}"), (cols[1], k, f"Candidate {run['candidate']}")):
                col.markdown(f"**{title}**")
                col.markdown(f'<div class="out">{html.escape(pretty(o["text"]))}</div>', unsafe_allow_html=True)
                checks = " · ".join(f'<span class="{"ok" if ch["passed"] else "bad"}">{"✓" if ch["passed"] else "✗"} '
                                    f'{html.escape(ch["detail"])}</span>' for ch in o["checks"])
                col.markdown(checks, unsafe_allow_html=True)
                if o["judge"]:
                    col.caption(f"Judge {o['judge']['score']}/5: {o['judge']['reason']}")
            p = pairs.get((c["id"], 0))
            if p:
                st.caption(f"Pairwise: **{p['winner']}** preferred, {p['reason']}")


with tab_report:
    report_panel()

with tab_diff:
    if prompts:
        name = st.selectbox("Prompt", list(prompts))
        vs = prompts[name]
        c1, c2 = st.columns(2)
        a = c1.selectbox("From", vs, index=max(0, len(vs) - 2), key="da")
        b = c2.selectbox("To", vs, index=len(vs) - 1, key="db")
        diff = api("GET", f"/prompts/{name}/diff/{a}/{b}") if a != b else ""
        if isinstance(diff, str) and diff:
            rows = []
            for line in diff.splitlines():
                css = "h" if line.startswith(("@@", "---", "+++")) else "a" if line.startswith("+") else "d" if line.startswith("-") else ""
                rows.append(f'<div class="{css}">{html.escape(line) or "&nbsp;"}</div>')
            st.markdown(f'<div class="diff">{"".join(rows)}</div>', unsafe_allow_html=True)
        else:
            st.info("Pick two different versions.")
