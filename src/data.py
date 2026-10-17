"""Data loading: CPCB CCR exports, Open-Meteo weather, and a synthetic fallback.

Usage:
    python -m src.data cpcb            # parse everything in data/raw -> data/processed/pm.csv
    python -m src.data weather 2023-01-01 2026-10-16   # fetch hourly weather
    python -m src.data synthetic       # offline test data (pm.csv + weather.csv)
"""
import sys
import numpy as np
import pandas as pd
import requests

from .config import RAW, PROC, STATIONS, CITY_CENTER

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m"]


# ---------------------------------------------------------------- CPCB
def _find(cols, *keys):
    for c in cols:
        lc = str(c).lower().replace(" ", "")
        if all(k in lc for k in keys):
            return c
    return None


def match_station(name):
    """Map a free-text CPCB station name to our STATIONS key."""
    n = str(name).lower()
    # longest key first so 'Manali Village' wins over 'Manali'
    for key in sorted(STATIONS, key=len, reverse=True):
        if key.lower() in n:
            return key
    return None


def read_cpcb_file(path):
    """Read one CCR export (xlsx/csv) or a compiled CSV with a station column."""
    path = pd.io.common.stringify_path(path)
    station_hint = None
    if path.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, header=None)
        # CCR exports have metadata rows on top; find the header row
        is_hdr = raw.apply(lambda r: r.astype(str).str.contains("From Date|Timestamp", case=False).any(), axis=1)
        hdr = int(is_hdr.idxmax())
        meta = " ".join(raw.iloc[:hdr].astype(str).values.ravel())
        station_hint = match_station(meta)
        df = pd.read_excel(path, header=hdr)
    else:
        df = pd.read_csv(path, low_memory=False)
    tcol = _find(df.columns, "from") or _find(df.columns, "timestamp") or _find(df.columns, "datetime") or _find(df.columns, "date")
    pcol = _find(df.columns, "pm2.5") or _find(df.columns, "pm25")
    if tcol is None or pcol is None:
        raise ValueError(f"{path}: could not find time / PM2.5 columns in {list(df.columns)[:15]}")
    scol = _find(df.columns, "station")
    out = pd.DataFrame({
        "time": pd.to_datetime(df[tcol], dayfirst=True, errors="coerce"),
        "pm25": pd.to_numeric(df[pcol], errors="coerce"),
    })
    p10 = _find(df.columns, "pm10")
    if p10 is not None:
        out["pm10"] = pd.to_numeric(df[p10], errors="coerce")
    if scol is not None:
        out["station"] = df[scol].map(match_station)
    else:
        out["station"] = station_hint or match_station(path)
    return out.dropna(subset=["time", "station"])


def clean_pm(df):
    """Hourly regularise per station, drop impossible values, fill short gaps."""
    df = df[(df.pm25 > 0) & (df.pm25 < 999)].copy()
    df["time"] = df["time"].dt.floor("h")
    out = []
    for st, g in df.groupby("station"):
        g = g.groupby("time")["pm25"].mean().asfreq("h")
        g = g.interpolate(limit=3)                     # fill gaps up to 3 h only
        out.append(g.rename("pm25").reset_index().assign(station=st))
    return pd.concat(out, ignore_index=True)


def load_cpcb():
    files = [p for p in RAW.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]
    if not files:
        raise SystemExit(f"No CPCB files in {RAW}. Download hourly data from the CCR portal first.")
    frames = []
    for p in files:
        try:
            frames.append(read_cpcb_file(str(p)))
            print(f"  read {p.name}: {len(frames[-1])} rows")
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")
    pm = clean_pm(pd.concat(frames))
    pm.to_csv(PROC / "pm.csv", index=False)
    print(pm.groupby("station").agg(start=("time", "min"), end=("time", "max"),
                                    n=("pm25", "count"), mean=("pm25", "mean")))
    return pm


# ---------------------------------------------------------------- weather
def _weather_frame(js):
    h = js["hourly"]
    w = pd.DataFrame(h).rename(columns={"time": "time"})
    w["time"] = pd.to_datetime(w["time"])
    return w


def fetch_weather_history(start, end):
    """Hourly historical weather for Chennai from Open-Meteo archive (IST)."""
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": CITY_CENTER[0], "longitude": CITY_CENTER[1],
        "start_date": start, "end_date": end, "hourly": ",".join(WEATHER_VARS),
        "timezone": "Asia/Kolkata"}, timeout=120)
    r.raise_for_status()
    w = _weather_frame(r.json())
    w.to_csv(PROC / "weather.csv", index=False)
    print(f"weather: {len(w)} hours {w.time.min()} -> {w.time.max()}")
    return w


def fetch_weather_forecast():
    """Next 48 h weather forecast (+ last 2 days) - the 'live' covariates."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": CITY_CENTER[0], "longitude": CITY_CENTER[1],
        "hourly": ",".join(WEATHER_VARS), "past_days": 2, "forecast_days": 2,
        "timezone": "Asia/Kolkata"}, timeout=60)
    r.raise_for_status()
    return _weather_frame(r.json())


# ---------------------------------------------------------------- synthetic
def make_synthetic(days=540, seed=7):
    """Realistic-looking Chennai-like data so the pipeline runs before real data arrives."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-03-01", periods=days * 24, freq="h")
    hr, doy = t.hour.values, t.dayofyear.values
    temp = 29 + 4 * np.sin(2 * np.pi * (doy - 60) / 365) + 3 * np.sin(2 * np.pi * (hr - 9) / 24) + rng.normal(0, .8, len(t))
    rh = 70 - 12 * np.sin(2 * np.pi * (hr - 9) / 24) + rng.normal(0, 4, len(t))
    monsoon = ((doy > 280) | (doy < 15)).astype(float)       # NE monsoon Oct-Dec
    rain = np.clip(rng.gamma(.3, 4, len(t)) * (rng.random(len(t)) < .05 + .2 * monsoon), 0, 60)
    wind = np.clip(3.5 + 2 * np.sin(2 * np.pi * (hr - 15) / 24) + rng.normal(0, 1, len(t)), .2, None)
    wdir = (180 + 90 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 30, len(t))) % 360
    weather = pd.DataFrame({"time": t, "temperature_2m": temp, "relative_humidity_2m": rh,
                            "precipitation": rain, "wind_speed_10m": wind, "wind_direction_10m": wdir})
    # city-wide driver: winter high, rush-hour peaks, washed out by rain, diluted by wind
    winter = 1 + .6 * np.cos(2 * np.pi * (doy - 15) / 365)
    rush = 1 + .35 * np.exp(-((hr - 9) ** 2) / 4) + .45 * np.exp(-((hr - 20) ** 2) / 6)
    ar = np.zeros(len(t))
    for i in range(1, len(t)):
        ar[i] = .93 * ar[i - 1] + rng.normal(0, .08)
    base = 32 * winter * rush * np.exp(ar) / (1 + .12 * wind) * np.exp(-.04 * pd.Series(rain).rolling(6, 1).sum().values)
    offsets = {"Alandur": 1.0, "Velachery": 1.1, "Perungudi": 0.95, "Arumbakkam": 1.05,
               "Royapuram": 1.25, "Kodungaiyur": 1.35, "Manali": 1.45, "Manali Village": 1.4}
    rows = []
    for st, off in offsets.items():
        noise = np.exp(rng.normal(0, .12, len(t)))
        pm = np.clip(base * off * noise, 4, 400)
        gaps = rng.random(len(t)) < .01
        pm[gaps] = np.nan
        rows.append(pd.DataFrame({"time": t, "station": st, "pm25": pm}))
    pm = pd.concat(rows, ignore_index=True)
    pm.to_csv(PROC / "pm.csv", index=False)
    weather.to_csv(PROC / "weather.csv", index=False)
    print(f"synthetic: {len(pm)} pm rows, {len(weather)} weather rows")
    return pm, weather


def load_processed():
    pm = pd.read_csv(PROC / "pm.csv", parse_dates=["time"])
    w = pd.read_csv(PROC / "weather.csv", parse_dates=["time"])
    return pm, w


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if cmd == "cpcb":
        load_cpcb()
    elif cmd == "weather":
        fetch_weather_history(sys.argv[2], sys.argv[3])
    else:
        make_synthetic()
