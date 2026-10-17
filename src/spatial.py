"""Estimate PM2.5 between monitoring stations (inverse-distance weighting) + leave-one-station-out test."""
import numpy as np
import pandas as pd

from .config import STATIONS


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def idw(points, st_coords, values, power=2.0):
    """points (n,2) lat/lon; st_coords (m,2); values (m,) -> (n,) estimates."""
    points, st_coords, values = np.asarray(points, float), np.asarray(st_coords, float), np.asarray(values, float)
    d = haversine_km(points[:, None, 0], points[:, None, 1], st_coords[None, :, 0], st_coords[None, :, 1])
    d = np.maximum(d, 0.05)
    wts = 1.0 / d ** power
    return (wts * values[None, :]).sum(1) / wts.sum(1)


POWERS = [0.0, 0.5, 1.0, 2.0, 3.0]   # 0 = plain city average, higher = trust the nearest station more


def loso_eval(pm, powers=POWERS, max_hours=2000):
    """Leave-one-station-out: hide each station, estimate it from the others with several
    distance-weighting powers, and pick the power with the lowest error (evidence-based choice)."""
    wide = pm.pivot_table(index="time", columns="station", values="pm25")
    wide = wide[[s for s in wide.columns if s in STATIONS]]
    wide = wide[wide.notna().sum(1) >= max(3, wide.shape[1] - 2)]     # hours with most stations reporting
    if len(wide) > max_hours:
        wide = wide.sample(max_hours, random_state=0)
    errs = {p: [] for p in powers}
    per_station = {}
    for st in wide.columns:
        rows = wide[wide[st].notna()]
        st_err = {}
        for p in powers:
            e = []
            for _, row in rows.iterrows():
                others = row.drop(st).dropna()
                if len(others) < 2:
                    continue
                est = idw(np.array([STATIONS[st]]), [STATIONS[s] for s in others.index], others.values, power=p)[0]
                e.append(abs(est - row[st]))
            errs[p] += e
            st_err[p] = round(float(np.mean(e)), 2) if e else None
        per_station[st] = st_err
    by_power = {p: {"MAE": round(float(np.mean(v)), 2), "n": len(v)} for p, v in errs.items() if v}
    best = min(by_power, key=lambda p: by_power[p]["MAE"])
    return {"by_power": by_power, "best_power": best, "per_station": per_station,
            "hours_sampled": int(len(wide)),
            "overall": {"IDW_MAE": by_power[best]["MAE"], "citymean_MAE": by_power[0.0]["MAE"],
                        "best_power": best}}


def best_power(default=2.0):
    """Power chosen by the last training run (reports/metrics.json)."""
    import json
    from .config import REPORTS
    try:
        return float(json.loads((REPORTS / "metrics.json").read_text())["loso"]["best_power"])
    except Exception:
        return default
