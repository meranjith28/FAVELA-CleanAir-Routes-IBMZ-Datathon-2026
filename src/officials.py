"""Health officials' view: 24-hour outlook per station, city summary and ready-to-send public alerts.

NAQI breakpoints are defined for 24-hour averages; here they are applied to hourly forecasts as an
early-warning guide, which is stated on the dashboard."""
import pandas as pd

from .advisory import COLORS, aqi_category, cat_label, localize_time

LEVELS = [("Very Poor", 121), ("Poor", 91), ("Moderate", 61)]        # PM2.5 ug/m3 thresholds (NAQI)

ACTIONS = {
    "en": {
        "Moderate": "Asthma patients, elders and children should limit long outdoor exertion. Schools can shift outdoor activities to cleaner hours.",
        "Poor": "Schools: move outdoor activities indoors. Hospitals and clinics: prepare for more respiratory cases. Outdoor workers: take regular breaks and wear well-fitted masks.",
        "Very Poor": "Everyone should avoid outdoor exertion. Schools: cancel outdoor activities. Issue a public health advisory and check local traffic and industrial emission sources.",
    },
    "ta": {
        "Moderate": "ஆஸ்துமா நோயாளிகள், முதியவர்கள் மற்றும் குழந்தைகள் நீண்ட நேர வெளிப்புற உழைப்பைக் குறைக்க வேண்டும். பள்ளிகள் வெளிப்புறச் செயல்பாடுகளைக் குறைந்த மாசுபாடு உள்ள நேரத்திற்கு மாற்றலாம்.",
        "Poor": "பள்ளிகள்: வெளிப்புறச் செயல்பாடுகளை உள்ளரங்கிற்கு மாற்றவும். மருத்துவமனைகள்: சுவாசப் பிரச்சினை நோயாளிகள் அதிகரிக்கலாம், தயாராக இருக்கவும். வெளிப்புறத் தொழிலாளர்கள்: அடிக்கடி ஓய்வு எடுத்து, சரியாகப் பொருந்தும் முகக்கவசம் அணியவும்.",
        "Very Poor": "அனைவரும் வெளிப்புற உழைப்பைத் தவிர்க்க வேண்டும். பள்ளிகள்: வெளிப்புறச் செயல்பாடுகளை ரத்து செய்யவும். பொது சுகாதார எச்சரிக்கை வெளியிட்டு, உள்ளூர் போக்குவரத்து மற்றும் தொழிற்சாலை உமிழ்வுகளைச் சரிபார்க்கவும்.",
    },
}

TEMPLATE = {
    "en": {"expected": "{st} area: {lvl} air expected {start} to {end}. Peak PM2.5 about {peak:.0f} µg/m³ around {peak_t}. {act}",
           "possible": "{st} area: air may reach {lvl} levels {start} to {end} (worst likely case, PM2.5 up to {peak:.0f} µg/m³). Sensitive groups should plan ahead: {act}"},
    "ta": {"expected": "{st} பகுதி: {start} முதல் {end} வரை காற்றின் தரம் '{lvl}' நிலையில் இருக்கும் என எதிர்பார்க்கப்படுகிறது. அதிகபட்ச PM2.5 சுமார் {peak:.0f} µg/m³ ({peak_t}). {act}",
           "possible": "{st} பகுதி: {start} முதல் {end} வரை காற்றின் தரம் '{lvl}' நிலையை அடையக்கூடும் (மோசமான நிலைக் கணிப்பு, PM2.5 {peak:.0f} µg/m³ வரை). உணர்திறன் மிக்கவர்கள் முன்கூட்டியே திட்டமிடவும்: {act}"},
}


def _level(v):
    for name, thr in LEVELS:
        if v >= thr:
            return name
    return None


def _windows(series, hours):
    """Contiguous runs where the level is Moderate or worse -> [(i0, i1, worst_level, peak_idx)]."""
    out, i = [], 1                      # hour 0 is 'now'; alerts look ahead
    while i < len(series):
        if _level(series[i]):
            j = i
            while j + 1 < len(series) and _level(series[j + 1]):
                j += 1
            seg = series[i:j + 1]
            k = i + max(range(len(seg)), key=lambda x: seg[x])
            rank = {"Moderate": 0, "Poor": 1, "Very Poor": 2}
            worst = max((_level(v) for v in seg), key=lambda n: rank[n])
            out.append((i, j, worst, k))
            i = j + 1
        else:
            i += 1
    return out


def build(fc, now, lang="en"):
    now = pd.Timestamp(now)
    times = [now + pd.Timedelta(hours=h) for h in range(25)]
    fmt = lambda t: localize_time(t.strftime("%a %H:%M"), lang)
    stations, alerts = [], []
    for st, v in sorted(fc.items()):
        med, hi = v["median"], v["p90"]
        cats = [aqi_category(x)[0] for x in med]
        peak_i = max(range(1, 25), key=lambda h: med[h])
        stations.append({"station": st, "median": med, "p90": hi,
                         "colors": [COLORS[c] for c in cats], "cats": cats,
                         "peak": round(med[peak_i], 1), "peak_time": fmt(times[peak_i]),
                         "peak_cat": cat_label(cats[peak_i], lang)})
        rank = {"Moderate": 1, "Poor": 2, "Very Poor": 3}
        for i0, i1, lvl, k in _windows(med, times):
            alerts.append({"station": st, "kind": "expected", "level": lvl, "rank": rank[lvl],
                           "start": fmt(times[i0]), "end": fmt(times[i1] + pd.Timedelta(hours=1)),
                           "color": COLORS[lvl],
                           "text": TEMPLATE[lang]["expected"].format(
                               st=st, lvl=cat_label(lvl, lang), start=fmt(times[i0]),
                               end=fmt(times[i1] + pd.Timedelta(hours=1)), peak=med[k], peak_t=fmt(times[k]),
                               act=ACTIONS[lang][lvl])})
        # worst-case (p90) reaches Poor while the median does not -> early heads-up for sensitive groups
        med_poor = any((_level(x) in ("Poor", "Very Poor")) for x in med[1:])
        if not med_poor:
            for i0, i1, lvl, k in _windows(hi, times):
                if lvl in ("Poor", "Very Poor"):
                    alerts.append({"station": st, "kind": "possible", "level": lvl, "rank": rank[lvl] - 0.5,
                                   "start": fmt(times[i0]), "end": fmt(times[i1] + pd.Timedelta(hours=1)),
                                   "color": COLORS[lvl],
                                   "text": TEMPLATE[lang]["possible"].format(
                                       st=st, lvl=cat_label(lvl, lang), start=fmt(times[i0]),
                                       end=fmt(times[i1] + pd.Timedelta(hours=1)), peak=hi[k],
                                       act=ACTIONS[lang]["Moderate"])})
    alerts.sort(key=lambda a: (-a["rank"], a["start"]))
    poor_st = [s for s in stations if any(_level(x) in ("Poor", "Very Poor") for x in s["median"][1:])]
    worst = max(stations, key=lambda s: s["peak"]) if stations else None
    first_poor = None
    for h in range(1, 25):
        if any(_level(s["median"][h]) in ("Poor", "Very Poor") for s in stations):
            first_poor = fmt(times[h]); break
    summary = {"stations": len(stations), "stations_poor": len(poor_st),
               "first_poor": first_poor,
               "worst_station": worst["station"] if worst else None,
               "worst_peak": worst["peak"] if worst else None,
               "worst_time": worst["peak_time"] if worst else None,
               "worst_cat": worst["peak_cat"] if worst else None,
               "n_alerts": len(alerts)}
    return {"now": str(now), "hours": [t.strftime("%H") for t in times], "day_labels": [fmt(t) for t in times],
            "stations": stations, "alerts": alerts, "summary": summary, "lang": lang}


def to_csv(fc, now):
    now = pd.Timestamp(now)
    rows = []
    for st, v in fc.items():
        for h in range(25):
            cat, aqi = aqi_category(v["median"][h])
            rows.append({"station": st, "time": now + pd.Timedelta(hours=h), "hours_ahead": h,
                         "pm25_forecast": v["median"][h], "pm25_worst_case_p90": v["p90"][h],
                         "aqi": aqi, "category": cat})
    return pd.DataFrame(rows).to_csv(index=False)
