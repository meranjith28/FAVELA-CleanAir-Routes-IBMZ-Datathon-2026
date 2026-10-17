"""Automatic data download - run on the LinuxONE server (needs outbound internet).

Sources
  * OpenAQ v3 API  - measured hourly PM2.5 from CPCB reference monitors in Chennai (free key:
                     https://explore.openaq.org/register  ->  export OPENAQ_API_KEY=...)
  * Open-Meteo     - hourly weather history + forecast (no key)
  * Open-Meteo Air Quality (CAMS model) - modelled PM2.5, used ONLY as a fallback if OpenAQ is unavailable
  * OpenStreetMap  - road network (see src/routing.py)

    python3 -m src.fetch stations                         # discover Chennai PM2.5 monitors -> stations.json
    python3 -m src.fetch history 2023-01-01 2026-10-16    # hourly PM2.5 -> data/processed/pm.csv
    python3 -m src.fetch weather 2023-01-01 2026-10-16    # hourly weather -> data/processed/weather.csv
    python3 -m src.fetch cams 2023-01-01 2026-10-16       # fallback: modelled PM2.5 -> pm.csv
    python3 -m src.fetch all 2023-01-01 2026-10-16        # everything above (+ OSM graph)
"""
import json
import os
import sys
import time

import pandas as pd
import requests

from .config import CITY_CENTER, PROC, STATIONS_FILE

OPENAQ = "https://api.openaq.org/v3"
TZ = "Asia/Kolkata"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "CleanAirRoutes/1.0 (IBM Z Datathon 2026)"


# ---------------------------------------------------------------- helpers
def _get(url, params=None, headers=None, tries=6, pause=0.0):
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=90)
        except requests.RequestException as e:
            wait = 5 * (i + 1)
            print(f"   network error ({e.__class__.__name__}), retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", 0) or 0) or 10 * (i + 1)
            print(f"   HTTP {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        if pause:
            time.sleep(pause)
        return r.json()
    raise RuntimeError(f"giving up on {url}")


def _oaq(path, **params):
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("OPENAQ_API_KEY not set. Get a free key at https://explore.openaq.org/register "
                         "or run `python3 -m src.fetch cams ...` for the modelled fallback.")
    # pause keeps us under OpenAQ's free-tier rate limit (~60 requests/minute)
    return _get(OPENAQ + path, params=params, headers={"X-API-Key": key}, pause=1.1)


def short_name(name):
    """'Velachery Res. Area, Chennai - CPCB' -> 'Velachery Res. Area'"""
    n = str(name).split(",")[0]
    for sep in (" - ", " -"):
        n = n.split(sep)[0]
    return n.strip()


def _to_ist(series_utc):
    return pd.to_datetime(series_utc, utc=True, format="ISO8601").dt.tz_convert(TZ).dt.tz_localize(None)


# ---------------------------------------------------------------- stations
def find_stations(radius_m=25000, monitors_only=True):
    """Find PM2.5 reference monitors within radius of Chennai; save name, ids and true coordinates."""
    lat, lon = CITY_CENTER
    js = _oaq("/locations", coordinates=f"{lat},{lon}", radius=radius_m, limit=1000)
    found = {}
    for loc in js.get("results", []):
        if monitors_only and loc.get("isMonitor") is False:
            continue                                   # skip low-cost sensors
        pm25 = [s for s in loc.get("sensors", []) if (s.get("parameter") or {}).get("name") == "pm25"]
        coords = loc.get("coordinates") or {}
        if not pm25 or coords.get("latitude") is None:
            continue
        # A station can have several PM2.5 sensors over its life (e.g. CPCB re-registered all
        # Chennai sensors in Feb 2025). Record every one with its active period; we merge them.
        sensors = [{"id": s["id"], "first": None, "last": None} for s in pm25]
        if len(sensors) > 1:
            detail = _oaq(f"/locations/{loc['id']}/sensors").get("results", [])
            dates = {d["id"]: ((d.get("datetimeFirst") or {}).get("utc"), (d.get("datetimeLast") or {}).get("utc"))
                     for d in detail}
            for s in sensors:
                s["first"], s["last"] = dates.get(s["id"], (None, None))
        name = short_name(loc.get("name"))
        if name in found:
            name = f"{name} ({loc['id']})"
        found[name] = {"lat": coords["latitude"], "lon": coords["longitude"],
                       "location_id": loc["id"], "sensors": sensors,
                       "provider": (loc.get("provider") or {}).get("name"),
                       "first": ((loc.get("datetimeFirst") or {}).get("utc")),
                       "last": ((loc.get("datetimeLast") or {}).get("utc"))}
    if not found:
        raise SystemExit("No PM2.5 monitors found - try a larger radius or monitors_only=False")
    STATIONS_FILE.write_text(json.dumps(found, indent=2))
    print(f"{len(found)} PM2.5 stations -> {STATIONS_FILE}")
    for n, s in found.items():
        print(f"  {n:<32} {s['lat']:.4f},{s['lon']:.4f}  {s['provider']}  last {s['last']}")
        for x in s["sensors"]:
            print(f"      sensor {x['id']}: {x['first']} -> {x['last']}")
    return found


def station_hours(meta, start, end, window_days=90):
    """Hourly PM2.5 for one station, merged across all its sensors.
    Sensors whose known active period doesn't overlap [start, end] are skipped."""
    sensors = meta.get("sensors") or [{"id": meta["sensor_id"], "first": None, "last": None}]
    t0 = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    t1 = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
    frames = []
    for s in sensors:
        if s.get("last") and pd.Timestamp(s["last"]) < t0:
            continue
        if s.get("first") and pd.Timestamp(s["first"]) > t1:
            continue
        df = sensor_hours(s["id"], start, end, window_days)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["time", "pm25"])
    # where two sensors overlap, average them
    return pd.concat(frames).groupby("time", as_index=False)["pm25"].mean()


def load_station_meta():
    if not STATIONS_FILE.exists():
        return find_stations()
    return json.loads(STATIONS_FILE.read_text())


# ---------------------------------------------------------------- measured PM2.5 (OpenAQ)
def sensor_hours(sensor_id, start, end, window_days=90):
    """All hourly averages for one sensor, paged and chunked to keep each query fast."""
    rows = []
    t0, t_end = pd.Timestamp(start), pd.Timestamp(end)
    while t0 < t_end:
        t1 = min(t0 + pd.Timedelta(days=window_days), t_end)
        page = 1
        while True:
            js = _oaq(f"/sensors/{sensor_id}/hours", datetime_from=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      datetime_to=t1.strftime("%Y-%m-%dT%H:%M:%SZ"), limit=1000, page=page)
            res = js.get("results", [])
            for r in res:
                p = r.get("period") or {}
                rows.append(((p.get("datetimeFrom") or {}).get("utc"), r.get("value")))
            if len(res) < 1000:
                break
            page += 1
        t0 = t1
    df = pd.DataFrame(rows, columns=["utc", "pm25"]).dropna()
    if df.empty:
        return pd.DataFrame(columns=["time", "pm25"])
    df["time"] = _to_ist(df["utc"])
    return df[["time", "pm25"]]


def fetch_history(start, end):
    from .data import clean_pm
    meta = load_station_meta()
    frames = []
    for name, s in meta.items():
        print(f"  {name}: downloading ...", flush=True)
        df = station_hours(s, start, end)
        print(f"  {name}: {len(df)} hours")
        if len(df):
            frames.append(df.assign(station=name))
    if not frames:
        raise SystemExit("No measurements returned.")
    pm = clean_pm(pd.concat(frames, ignore_index=True))
    pm["source"] = "openaq_measured"
    pm.to_csv(PROC / "pm.csv", index=False)
    summary = pm.groupby("station").agg(start=("time", "min"), end=("time", "max"),
                                        hours=("pm25", "count"), mean=("pm25", "mean"))
    summary["mean"] = summary["mean"].round(1)
    print(summary.to_string())
    return pm


def recent_measurements(hours=96):
    """Last few days for every station - the live input for the forecaster."""
    end = pd.Timestamp.utcnow().tz_localize(None).ceil("h")
    start = end - pd.Timedelta(hours=hours)
    frames = []
    for name, s in load_station_meta().items():
        try:
            df = station_hours(s, start, end, window_days=30)
            frames.append(df.assign(station=name))
        except Exception as e:
            print(f"  live fetch failed for {name}: {e}")
    from .data import clean_pm
    return clean_pm(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------- weather (Open-Meteo)
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"]


def _weather(js):
    w = pd.DataFrame(js["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w


def fetch_weather(start, end):
    """Archive API in yearly chunks (big ranges can time out)."""
    lat, lon = CITY_CENTER
    frames = []
    for y0 in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="365D"):
        y1 = min(y0 + pd.Timedelta(days=364), pd.Timestamp(end))
        js = _get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon, "start_date": y0.date(), "end_date": y1.date(),
            "hourly": ",".join(WEATHER_VARS), "timezone": TZ})
        frames.append(_weather(js).dropna(subset=WEATHER_VARS, how="all"))
    # archive lags a few days behind - top up the most recent days from the forecast API
    frames.append(weather_forecast(past_days=10))
    w = pd.concat(frames).drop_duplicates("time", keep="first").sort_values("time")
    w = w[w.time <= pd.Timestamp(end) + pd.Timedelta(days=2)]
    w.to_csv(PROC / "weather.csv", index=False)
    print(f"weather: {len(w)} hours {w.time.min()} -> {w.time.max()}")
    return w


def weather_forecast(past_days=3, forecast_days=2):
    lat, lon = CITY_CENTER
    js = _get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days, "forecast_days": forecast_days, "timezone": TZ})
    return _weather(js)


# ---------------------------------------------------------------- fallback: modelled PM2.5 (CAMS)
def fetch_cams(start, end):
    """Modelled (not measured) PM2.5 at each station's coordinates. Use only if OpenAQ is unavailable,
    and say so in the pitch."""
    from .config import STATIONS
    frames = []
    for name, (lat, lon) in STATIONS.items():
        js = _get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": lat, "longitude": lon, "hourly": "pm2_5", "start_date": start, "end_date": end,
            "timezone": TZ})
        df = pd.DataFrame(js["hourly"]).rename(columns={"pm2_5": "pm25"})
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df.assign(station=name))
        print(f"  {name}: {df.pm25.notna().sum()} hours (modelled)")
    pm = pd.concat(frames, ignore_index=True).dropna(subset=["pm25"])
    pm["source"] = "cams_model"
    pm.to_csv(PROC / "pm.csv", index=False)
    return pm


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "stations"), sys.argv[2:]
    if cmd == "stations":
        find_stations()
    elif cmd == "history":
        fetch_history(*args)
    elif cmd == "weather":
        fetch_weather(*args)
    elif cmd == "cams":
        fetch_cams(*args)
    elif cmd == "all":
        if os.getenv("OPENAQ_API_KEY"):
            find_stations()
            fetch_history(*args)
        else:
            print("No OPENAQ_API_KEY - using modelled CAMS PM2.5 as a fallback")
            fetch_cams(*args)
        fetch_weather(*args)
        from .config import AREAS
        from .routing import fetch_osm
        for area in AREAS:
            fetch_osm(area)
    else:
        print(__doc__)
