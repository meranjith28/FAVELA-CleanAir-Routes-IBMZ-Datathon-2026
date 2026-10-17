"""CleanAir Routes - Flask API + map UI.

    python app.py                        # replay mode: 'now' = start of held-out test period, steps 1 h per tick
    REPLAY_START=2026-09-01T08:00 python app.py
Open http://localhost:5000  (on LinuxONE: ssh -L 5000:localhost:5000 linux1@<vm-ip>)
"""
import functools
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from src import officials
from src.advisory import aqi_category, genai_status, guidance, llm_explain, warm_up, COLORS
from src.config import AREAS, PROFILES, REPORTS, STATIONS, TEST_DAYS, MODES
from src.data import load_processed
from src.model import Forecaster
from src.routing import Router, area_of, available_areas

app = Flask(__name__, static_folder="web")
STATE = threading.RLock()     # routing/forecast state is shared, so those requests run one at a time


def locked(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with STATE:
            return fn(*a, **k)
    return wrapper
PM, W = load_processed()
FC = Forecaster()
ROUTERS = {}
for _a in available_areas():
    ROUTERS[_a] = Router(_a)
    print(f"routing area '{_a}': {ROUTERS[_a].G.number_of_nodes()} junctions")
if not ROUTERS:
    raise SystemExit("No road graph found - run: python3 -m src.routing fetch")
LIVE = os.getenv("LIVE") == "1"          # LIVE=1: pull latest readings + weather forecast from the internet
REFRESH_S = int(os.getenv("REFRESH_MIN", 15)) * 60
_start = os.getenv("REPLAY_START")


def default_replay_start():
    """12 h before the worst city-wide PM2.5 hour inside the held-out test period, so the demo
    shows the forecast warning about a real spike the model never trained on."""
    try:
        m = json.loads((REPORTS / "metrics.json").read_text())
        t0, t1 = pd.Timestamp(m["test_start"]), pd.Timestamp(m["test_end"])
    except Exception:
        t1 = PM.time.max(); t0 = t1 - pd.Timedelta(days=TEST_DAYS)
    win = PM[(PM.time >= t0 + pd.Timedelta(days=3)) & (PM.time <= t1 - pd.Timedelta(days=2))].dropna(subset=["pm25"])
    g = win.groupby("time")["pm25"]
    n_st = win.station.nunique()
    # only hours when most stations report, so the "worst hour" is a real city-wide spike
    ok = g.count() >= max(3, int(0.75 * n_st))
    city = g.mean()[ok]
    if city.empty:
        return t0.floor("h")
    peak = city.rolling(3, center=True, min_periods=1).mean().idxmax()
    print(f"Replay: worst held-out hour {peak} (city mean {city[peak]:.0f} ug/m3); starting 12 h earlier")
    return (peak - pd.Timedelta(hours=12)).floor("h")


CLOCK = {"now": pd.Timestamp(_start) if _start else default_replay_start()}
_cache = {}
_live = {"at": 0.0, "status": "not fetched yet"}


def latest_origin(pm):
    """Most recent hour for which at least half the stations have reported."""
    counts = pm.dropna(subset=["pm25"]).groupby("time")["station"].nunique()
    ok = counts[counts >= max(1, pm.station.nunique() // 2)]
    return ok.index.max()


def refresh_live():
    global PM, W
    if not LIVE or time.time() - _live["at"] < REFRESH_S:
        return
    from src.fetch import recent_measurements, weather_forecast
    try:
        pm, w = recent_measurements(), weather_forecast(past_days=10)   # covers the ~4-day data delay of the public feed
        PM, W = pm, w
        CLOCK["now"] = latest_origin(pm)
        _cache.clear()
        _live.update(at=time.time(), status=f"ok, latest reading {CLOCK['now']}")
    except Exception as e:                      # keep serving the last good data
        _live.update(at=time.time(), status=f"refresh failed: {e}")
    print("live refresh:", _live["status"])


def forecast_now():
    refresh_live()
    t = CLOCK["now"]
    if t not in _cache:
        _cache.clear()
        _cache[t] = FC.forecast(PM, W, t)
    return _cache[t]


def observed(t):
    obs = PM[PM.time == t].set_index("station")["pm25"].dropna()
    return obs.to_dict()


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.get("/api/meta")
@locked
def meta():
    refresh_live()
    return jsonify({"profiles": PROFILES, "modes": list(MODES), "now": str(CLOCK["now"]),
                    "live": LIVE, "live_status": _live["status"],
                    "stations": {k: v for k, v in STATIONS.items()}, "colors": COLORS,
                    "areas": {a: AREAS[a] for a in ROUTERS}})


@app.post("/api/tick")
@locked
def tick():
    """Advance the replay clock (simulated live stream of held-out data)."""
    if LIVE:
        _live["at"] = 0.0                       # in live mode the button forces a fresh download
        refresh_live()
        return jsonify({"now": str(CLOCK["now"]), "status": _live["status"]})
    CLOCK["now"] += pd.Timedelta(hours=int(request.args.get("hours", 1)))
    return jsonify({"now": str(CLOCK["now"])})


@app.get("/api/forecast")
@locked
def forecast():
    t0 = time.perf_counter()
    fc = forecast_now()
    t = CLOCK["now"]
    future_obs = {}
    for st in fc:  # truth for the next 24 h (replay only) so judges can see forecast vs reality
        s = PM[(PM.station == st) & (PM.time > t) & (PM.time <= t + pd.Timedelta(hours=24))].sort_values("time")
        future_obs[st] = [round(x, 1) if pd.notna(x) else None for x in s.pm25.tolist()]
    out = {st: {**v, "now_cat": aqi_category(v["median"][0])[0], "observed_next": future_obs.get(st, []),
                "coords": STATIONS.get(st)} for st, v in fc.items()}
    return jsonify({"now": str(t), "stations": out, "latency_ms": round(1000 * (time.perf_counter() - t0), 1)})


def _station_values(fc, hour, key="median"):
    return {st: v[key][hour] for st, v in fc.items()}


@app.get("/api/route")
@locked
def route():
    t0 = time.perf_counter()
    src = tuple(map(float, request.args["from"].split(",")))
    dst = tuple(map(float, request.args["to"].split(",")))
    mode = request.args.get("mode", "walk")
    profile = request.args.get("profile", "general")
    hour = int(request.args.get("hour", 0))
    fc = forecast_now()
    sensitive = profile in ("asthma", "elderly", "child")
    key = "p90" if sensitive else "median"

    # both points must fall inside the same routing area
    a_src, a_dst = area_of(*src, ROUTERS), area_of(*dst, ROUTERS)
    if a_src is None or a_dst is None or a_src != a_dst:
        names = " or ".join(AREAS[a]["label"] for a in ROUTERS)
        msg = ("Start and destination must be in the same highlighted area." if a_src and a_dst
               else f"Please pick both points inside a highlighted area ({names}).")
        return jsonify({"error": msg}), 400
    router = ROUTERS[a_src]

    router.set_pollution(_station_values(fc, hour, key), mode)
    r = router.routes(src, dst)
    saving = 100 * (1 - r["cleanest"]["dose_ug"] / r["fastest"]["dose_ug"]) if r["fastest"]["dose_ug"] else 0

    # best departure time: inhaled dose on the recommended (cleanest) route for each hour in the next 24 h
    path = r["cleanest"]["_path"]
    timeline = []
    for hh in range(0, 25):
        d, c = router.dose_along(path, _station_values(fc, hh, key), mode)
        timeline.append({"hour": hh, "time": str(CLOCK["now"] + pd.Timedelta(hours=hh)),
                         "dose_ug": round(d, 2), "avg_pm25": round(c, 1)})
    for v in r.values():
        v.pop("_path", None)
    best = min(timeline, key=lambda x: x["dose_ug"])
    best_label = pd.Timestamp(best["time"]).strftime("%a %H:%M")
    med_avg = r["balanced"]["avg_pm25"]
    lang = request.args.get("lang", "en")
    adv = guidance(profile, med_avg, med_avg, best_label if best["hour"] != hour else None, saving, lang=lang)
    # facts handed to the optional GenAI layer (/api/explain) - the model may only rephrase these
    facts = {"traveller": PROFILES[profile]["label"], "mode": mode, "aqi": adv["aqi"], "category": adv["category"],
             "rule_based_advice": guidance(profile, med_avg, med_avg, best_label if best["hour"] != hour else None,
                                           saving)["text"],
             "fastest_route": {k: r["fastest"][k] for k in ("minutes", "km", "dose_ug")},
             "cleanest_route": {k: r["cleanest"][k] for k in ("minutes", "km", "dose_ug")},
             "cleanest_saves_percent": round(saving, 1),
             "best_time_to_leave": best_label if best["hour"] != hour else "now"}
    return jsonify({"routes": r, "saving_pct": round(saving, 1), "advisory": adv, "facts": facts,
                    "timeline": timeline, "best_departure": best, "basis": key,
                    "latency_ms": round(1000 * (time.perf_counter() - t0), 1)})


# ------------------------------------------------------------------ GenAI layer (optional)
@app.get("/api/genai_status")
def genai():
    return jsonify(genai_status())


# The local model takes ~1 minute per answer on CPU, so it runs in ONE background worker:
# the app stays responsive and requests queue instead of freezing the server.
GENAI_POOL = ThreadPoolExecutor(max_workers=1)
JOBS = {}                 # job id -> {"future", "started"}
BULLETINS = {}            # (replay hour, lang) -> job id


def _submit(facts, lang, task):
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"future": GENAI_POOL.submit(llm_explain, facts, lang, task), "started": time.time()}
    return jid


def _job_state(jid):
    j = JOBS.get(jid)
    if not j:
        return {"status": "unknown"}
    f = j["future"]
    if not f.done():
        return {"status": "working", "elapsed_s": round(time.time() - j["started"])}
    out = f.result() if not f.exception() else None
    return {"status": "done", **(out or {"text": None})}


@app.post("/api/explain")
def explain():
    body = request.get_json(force=True, silent=True) or {}
    if not genai_status().get("available"):
        return jsonify({"status": "unavailable"})
    return jsonify({"status": "working", "job": _submit(body.get("facts", {}), body.get("lang", "en"), "travel")})


@app.get("/api/explain/<jid>")
def explain_result(jid):
    return jsonify(_job_state(jid))


@app.get("/api/officials/bulletin")
def officials_bulletin():
    """AI situation bulletin for officials, written once per replay hour in the background."""
    lang = request.args.get("lang", "en")
    st = genai_status()
    if not st.get("available") or lang not in st.get("languages", []):
        return jsonify({"status": "unavailable"})
    key = (str(CLOCK["now"]), lang)
    if key not in BULLETINS:
        with STATE:
            d = officials.build(forecast_now(), CLOCK["now"], "en")
        facts = {"forecast_time": d["now"], "summary": d["summary"],
                 "alerts": [{k: a[k] for k in ("station", "kind", "level", "start", "end")} for a in d["alerts"][:6]],
                 "station_peaks": [{"station": s["station"], "peak_pm25": s["peak"], "at": s["peak_time"],
                                    "category": s["peak_cat"]} for s in d["stations"]]}
        BULLETINS[key] = _submit(facts, lang, "bulletin")
    return jsonify(_job_state(BULLETINS[key]))


# ------------------------------------------------------------------ health officials' dashboard
@app.get("/officials")
def officials_page():
    return send_from_directory("web", "officials.html")


@app.get("/api/officials")
@locked
def officials_api():
    return jsonify(officials.build(forecast_now(), CLOCK["now"], request.args.get("lang", "en")))


@app.get("/api/officials.csv")
@locked
def officials_csv():
    csv = officials.to_csv(forecast_now(), CLOCK["now"])
    name = f"cleanair_forecast_{pd.Timestamp(CLOCK['now']):%Y%m%d_%H%M}.csv"
    return Response(csv, mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}"})


@app.get("/api/metrics")
def metrics():
    p = REPORTS / "metrics.json"
    return jsonify(json.loads(p.read_text()) if p.exists() else {})


if __name__ == "__main__":
    if genai_status().get("available"):
        print("GenAI:", genai_status()["provider"], "- loading the model in the background ...")
        GENAI_POOL.submit(warm_up)
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)), threaded=True)
