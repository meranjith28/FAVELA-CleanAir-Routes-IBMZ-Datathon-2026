"""Multi-horizon PM2.5 forecasting (one model per horizon) + baselines + spatial validation.

    python -m src.model train          # trains all horizons, writes reports/metrics.json
    python -m src.model train --snapml # use IBM Snap ML booster if installed (LinuxONE)
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import HORIZONS, LAGS, MODELS, REPORTS, STATIONS, TEST_DAYS, TEST_START, TEST_END
VAL_DAYS = 60   # validation slice (end of training period) used to pick the hybrid weight
from .data import load_processed
from .spatial import idw, loso_eval
from .advisory import aqi_category

STATION_IDS = {s: i for i, s in enumerate(STATIONS)}
WCOLS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_u", "wind_v", "wind_speed_10m"]


def prep_weather(w):
    w = w.copy()
    rad = np.deg2rad(w["wind_direction_10m"])
    w["wind_u"] = -w["wind_speed_10m"] * np.sin(rad)
    w["wind_v"] = -w["wind_speed_10m"] * np.cos(rad)
    w["rain6"] = w["precipitation"].rolling(6, min_periods=1).sum()
    return w[["time"] + WCOLS + ["rain6"]]


def build_features(pm, w, h):
    """Features known at forecast origin t, target = PM2.5 at t+h.
    Weather at t+h is used as a known covariate (comes from weather forecast at run time)."""
    w = prep_weather(w)
    df = pm.merge(w, on="time", how="left").sort_values(["station", "time"]).reset_index(drop=True)
    city = pm.groupby("time")["pm25"].mean().rename("city_mean")
    df = df.merge(city, on="time", how="left")
    g = df.groupby("station")["pm25"]
    df["pm_now"] = df["pm25"]
    for l in LAGS:
        df[f"lag{l}"] = g.shift(l)
    df["roll6"] = g.transform(lambda s: s.rolling(6, min_periods=3).mean())
    df["roll24"] = g.transform(lambda s: s.rolling(24, min_periods=12).mean())
    df["trend3"] = df["pm_now"] - df["lag3"]
    df["same_hour_yday"] = g.shift(24 - h) if h < 24 else df["pm_now"]   # value at target hour, 1 day earlier
    # future (target-time) covariates
    tt = df["time"] + pd.Timedelta(hours=h)
    wf = w.drop_duplicates("time").set_index("time")
    for c in WCOLS + ["rain6"]:
        df[f"f_{c}"] = tt.map(wf[c]).values
    df["t_hour_sin"], df["t_hour_cos"] = np.sin(2 * np.pi * tt.dt.hour / 24), np.cos(2 * np.pi * tt.dt.hour / 24)
    df["t_doy_sin"], df["t_doy_cos"] = np.sin(2 * np.pi * tt.dt.dayofyear / 365), np.cos(2 * np.pi * tt.dt.dayofyear / 365)
    df["t_dow"] = tt.dt.dayofweek
    df["station_id"] = df["station"].map(STATION_IDS).fillna(-1).astype(int)
    df["target"] = g.shift(-h)
    feats = (["pm_now", "city_mean", "roll6", "roll24", "trend3", "same_hour_yday", "station_id", "t_dow",
              "t_hour_sin", "t_hour_cos", "t_doy_sin", "t_doy_cos"]
             + [f"lag{l}" for l in LAGS] + WCOLS + [f"f_{c}" for c in WCOLS + ["rain6"]])
    return df, feats


def _metrics(y, p):
    cat_acc = float(np.mean([aqi_category(a)[0] == aqi_category(b)[0] for a, b in zip(y, p)]))
    return {"MAE": round(float(mean_absolute_error(y, p)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "AQI_cat_acc": round(cat_acc, 3)}


def _booster(use_snap):
    if use_snap:
        try:
            from snapml import BoostingMachineRegressor  # IBM Snap ML (s390x wheels available)
            return BoostingMachineRegressor(num_round=400, learning_rate=0.05, max_depth=6), "snapml_boost"
        except ImportError:
            print("snapml not installed - falling back to scikit-learn")
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0), "hgb"


def test_window(pm):
    """Held-out period. Default: Chennai's pollution season (Dec-Feb). Falls back to the last
    TEST_DAYS if the data does not cover that winter. Override with TEST_START / TEST_END env vars."""
    t0, t1 = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END) + pd.Timedelta(hours=23)
    covered = pm[(pm.time >= t0) & (pm.time <= t1)]
    if covered.pm25.notna().sum() < 0.5 * (t1 - t0).total_seconds() / 3600:      # < half an hour/station-hour
        t1 = pm["time"].max()
        t0 = t1 - pd.Timedelta(days=TEST_DAYS)
    return t0, t1


def train(use_snap=False):
    pm, w = load_processed()
    t0, t1 = test_window(pm)
    # Blocked hold-out: drop a buffer around the test window from training so no training row's
    # lag features (up to 48 h back) or target (up to 24 h ahead) touch the test period.
    buf = pd.Timedelta(hours=72)
    report = {"test_start": str(t0), "test_end": str(t1),
              "test_label": f"{t0:%d %b %Y} - {t1:%d %b %Y}", "horizons": {}, "platform": _platform()}
    print(f"Test (held-out) period: {report['test_label']}")
    for h in HORIZONS:
        df, feats = build_features(pm, w, h)
        df = df.dropna(subset=feats + ["target"])
        te = df[(df.time >= t0) & (df.time <= t1)]
        tr = df[(df.time < t0 - buf) | (df.time > t1 + buf)]
        cut = t0 - buf                      # the hybrid weight is validated just before the test window
        Xtr, ytr, Xte, yte = tr[feats].values, tr["target"].values, te[feats].values, te["target"].values
        res = {"n_train": len(tr), "n_test": len(te)}
        # baselines
        res["persistence"] = _metrics(yte, te["pm_now"].values)
        res["same_hour_yesterday"] = _metrics(yte, te["same_hour_yday"].values)
        # linear model (on log scale) - reported as a reference only
        ridge = Ridge(alpha=1.0).fit(Xtr, np.log1p(ytr))
        res["ridge"] = _metrics(yte, np.expm1(ridge.predict(Xte)))

        # --- hybrid weight: choose the model/persistence mix on a VALIDATION slice of the
        #     training period (never the test set), so the final score stays honest
        vcut = cut - pd.Timedelta(days=VAL_DAYS)
        val = tr[(tr.time > vcut) & (tr.time <= cut)]
        fit = tr[(tr.time <= vcut - buf) | (tr.time > cut)]
        g_val, _ = _booster(use_snap)
        g_val.fit(fit[feats].values, np.log1p(fit["target"].values))
        pv = np.expm1(g_val.predict(val[feats].values))
        grid = np.linspace(0, 1, 21)
        maes = [mean_absolute_error(val["target"], wt * pv + (1 - wt) * val["pm_now"]) for wt in grid]
        w_model = float(grid[int(np.argmin(maes))])
        res["hybrid_model_weight"] = round(w_model, 2)

        # gradient boosting (main model) on the full training period, log target
        tic = time.time()
        gbm, name = _booster(use_snap)
        gbm.fit(Xtr, np.log1p(ytr))
        res["train_seconds"] = round(time.time() - tic, 1)
        p_gbm = np.expm1(gbm.predict(Xte))
        res[name] = _metrics(yte, p_gbm)
        # hybrid = model + persistence with the validation-chosen weight  (the forecast we use)
        p_hyb = w_model * p_gbm + (1 - w_model) * te["pm_now"].values
        res["hybrid"] = _metrics(yte, p_hyb)
        # p90 quantile model -> "worst-case" guidance for sensitive users
        #    Trained on the log scale: the 90th percentile of log(PM) is log of the 90th percentile,
        #    so the band is proportional - it widens automatically when pollution is high (winter).
        q90_lin = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300,
                                                learning_rate=0.05, random_state=0).fit(Xtr, ytr)
        q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300,
                                            learning_rate=0.05, random_state=0).fit(Xtr, np.log1p(ytr))
        # Conformal calibration on the VALIDATION slice: how much must the band be stretched so that
        # 90% of validation values fall under it? (Never uses the test period.)
        q90_v = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300,
                                              learning_rate=0.05, random_state=0).fit(
            fit[feats].values, np.log1p(fit["target"].values))
        pv_hyb = w_model * pv + (1 - w_model) * val["pm_now"].values
        band_v = np.maximum(np.expm1(q90_v.predict(val[feats].values)), pv_hyb)
        p90_scale = float(max(1.0, np.quantile(val["target"].values / np.maximum(band_v, 1.0), 0.9)))
        res["p90_scale"] = round(p90_scale, 3)
        band_te = np.maximum(np.expm1(q90.predict(Xte)), p_hyb)
        res["p90_coverage_linear"] = round(float(np.mean(yte <= np.maximum(q90_lin.predict(Xte), p_hyb))), 3)
        res["p90_coverage_uncalibrated"] = round(float(np.mean(yte <= band_te)), 3)
        res["p90_coverage"] = round(float(np.mean(yte <= band_te * p90_scale)), 3)
        res["skill_vs_persistence_%"] = round(100 * (1 - res["hybrid"]["MAE"] / res["persistence"]["MAE"]), 1)
        res["best"] = min(["persistence", "same_hour_yesterday", "ridge", name, "hybrid"], key=lambda k: res[k]["MAE"])
        joblib.dump({"gbm": gbm, "q90": q90, "q90_log": True, "p90_scale": p90_scale,
                     "feats": feats, "name": name, "w_model": w_model},
                    MODELS / f"h{h}.joblib")
        report["horizons"][h] = res
        print(f"h={h:>2}: persistence {res['persistence']['MAE']:5.2f} | {name} {res[name]['MAE']:5.2f} "
              f"| hybrid {res['hybrid']['MAE']:5.2f} (model weight {w_model:.2f}) "
              f"| skill {res['skill_vs_persistence_%']:5.1f}% | p90 cover {res['p90_coverage_linear']} -> {res['p90_coverage']} (scale x{res['p90_scale']})")
    # spatial interpolation: pick the distance-weighting power by leave-one-station-out
    test_pm = pm[(pm.time >= t0) & (pm.time <= t1)]
    report["loso"] = loso_eval(test_pm)
    print("LOSO by power:", {p: v["MAE"] for p, v in report["loso"]["by_power"].items()},
          "-> using power", report["loso"]["best_power"])
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _platform():
    import platform
    return {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
            "node": platform.node()}


# ---------------------------------------------------------------- inference
class Forecaster:
    def __init__(self):
        self.models = {h: joblib.load(MODELS / f"h{h}.joblib") for h in HORIZONS}

    def forecast(self, pm, w, origin):
        """Return per-station hourly forecast for origin+0..24 (median and p90)."""
        hist = pm[(pm.time <= origin) & (pm.time > origin - pd.Timedelta(days=4))].copy()
        # A station that skipped the latest reading(s) still gets a forecast from its last value
        # within the past 6 hours (CPCB stations often drop an hour or two).
        hist = hist.sort_values(["station", "time"])
        hist["pm25"] = hist.groupby("station")["pm25"].ffill(limit=6)
        filled = []
        for st, g in hist.groupby("station"):
            g = g.set_index("time").reindex(pd.date_range(g.time.min(), origin, freq="h"))
            g["station"] = st
            g["pm25"] = g["pm25"].ffill(limit=6)
            filled.append(g.rename_axis("time").reset_index())
        hist = pd.concat(filled, ignore_index=True) if filled else hist
        out, rows = {}, {}
        for h, m in self.models.items():
            df, feats = build_features(hist, w, h)
            last = df[df.time == origin].dropna(subset=["pm_now"])
            X = last[feats].astype(float).values
            if not len(X):
                continue
            wm = m.get("w_model", 1.0)
            p = wm * np.expm1(m["gbm"].predict(X)) + (1 - wm) * last["pm_now"].values
            q = m["q90"].predict(X)
            p90 = np.maximum(np.expm1(q) if m.get("q90_log") else q, p) * m.get("p90_scale", 1.0)
            for st, now, a, b in zip(last.station, last.pm_now, p, p90):
                rows.setdefault(st, {0: (float(now), float(now))})[h] = (float(a), float(b))
        for st, pts in rows.items():
            hs = sorted(pts)
            med = np.interp(range(25), hs, [pts[k][0] for k in hs])
            hi = np.interp(range(25), hs, [pts[k][1] for k in hs])
            out[st] = {"median": np.round(med, 1).tolist(), "p90": np.round(hi, 1).tolist()}
        return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(use_snap="--snapml" in sys.argv)
