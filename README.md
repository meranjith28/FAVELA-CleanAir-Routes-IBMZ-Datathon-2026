# CleanAir Routes — [FAVELA_IBM Z Datathon 2026 Project]

**Forecasts PM2.5 across Chennai for the next 24 hours, then tells people which way *and when* to travel so they breathe in the least pollution** — with worst-case guidance for asthma patients, elders and children. Trained and served on **IBM LinuxONE (IBM Z, s390x)**.

Theme: **Real-Time AI for Critical Decisions** (forecast → decide → act before exposure). Event code: `SSF2026`.

## The problem
People cannot tell when air quality will worsen or which routes expose them to more pollution, so sensitive groups — children, elders, asthma patients — make travel decisions blind. Official stations report what the air *was*; nobody turns that into "*leave at 7 pm and take this street*".

## What it does
1. **Forecasts PM2.5 1–24 h ahead** at every monitoring station (one model per horizon).
2. **Gives a calibrated worst case (p90)** — sensitive users are advised on the pollution level that is exceeded only ~1 time in 10.
3. **Estimates pollution on every street segment** between stations, raised on busy arterial roads.
4. **Scores each route by inhaled dose** = concentration × breathing rate × time (walk or cycle).
5. **Returns three routes** — fastest, cleanest, balanced (cleanest within +25 % time) — and the **best hour to leave** in the next 24 h.
6. **Advises in plain language** using India's National AQI categories, tailored to who is travelling — in **English and Tamil**.
7. **Health officials' dashboard** (`/officials`): a stations × 24-hour heat table, a city summary and **ready-to-send public alerts** (schools, hospitals, outdoor workers) in English and Tamil, with CSV export.
8. **GenAI on LinuxONE:** IBM **Granite 3.3 2B** (big-endian GGUF) runs locally through llama.cpp and rewrites the advice in friendly language — using only the computed facts, never inventing numbers.
9. **IBM z17 AI accelerator:** a neural forecaster is run on the on-chip **Integrated Accelerator for AI (NNPA)** via `torch_nnpa`, benchmarked against the CPU (`scripts/nnpa_benchmark.py`).

## Architecture
```
CPCB / TNPCB stations (via OpenAQ API) ─┐
Open-Meteo weather (history + forecast) ┼─► features ─► 5 horizon models + calibrated p90 ─► 24 h forecast per station
                                        │                                                          │
OpenStreetMap roads (120,566 junctions) ┴─► road graph ─► pollution per street (+ arterial uplift) ◄┘
                                                              │
                                     dose-weighted shortest paths ─► fastest / cleanest / balanced + best time
                                                              │
                                      NAQI advisory (+ optional GenAI) ─► Flask API ─► map web app
                  everything trained and served on IBM LinuxONE Community Cloud (IBM Z, s390x), containerised
```

## Data
| Source | What we use |
|---|---|
| **CPCB / TNPCB monitoring stations**, via the [OpenAQ v3 API](https://docs.openaq.org) | Hourly PM2.5 — Manali, Manali Village, Kodungaiyur, Royapuram, Arumbakkam, Velachery, Perungudi (Feb 2025 → Oct 2026) |
| **US Consulate Chennai monitor** (AirNow, via OpenAQ) | Hourly PM2.5, Jan 2024 → Oct 2026 |
| [Open-Meteo](https://open-meteo.com) | Hourly temperature, humidity, rain, wind (history + forecast) |
| [OpenStreetMap](https://www.openstreetmap.org) via Overpass API | Road network covering all station areas |

All data is downloaded automatically on the LinuxONE server (`src/fetch.py`, `src/routing.py`). Stations that changed sensor IDs (CPCB re-registered all Chennai sensors in Feb 2025) are merged automatically.

## Model
- **Features:** current PM2.5, 1–48 h lags, rolling means, trend, same hour yesterday, city-wide mean, station, hour/day/season of the target time, and weather (current + forecast for the target hour).
- **Main model:** gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`) per horizon, on log PM2.5.
- **Hybrid:** blended with the "no change" forecast; the blend weight per horizon is chosen on a validation slice, never on the test set.
- **Worst case:** a 90th-percentile quantile model on the log scale, then **conformally calibrated** on the validation slice.
- **Between stations:** inverse-distance weighting; the weighting strength is chosen automatically by leave-one-station-out testing.

## Results (held-out data the model never trained on)
Errors are mean absolute error in µg/m³. "Skill" = improvement over the "no change" forecast.

**Winter test — 1 Dec 2025 to 28 Feb 2026 (Chennai's polluted season, blocked hold-out with 72 h buffers)**

| Horizon | No-change error | Our error | Skill | Worst-case coverage (target 0.90) |
|---|---|---|---|---|
| +1 h | 10.69 | 10.34 | 3.3 % | 0.92 |
| +3 h | 18.41 | 16.36 | 11.1 % | 0.91 |
| +6 h | 23.18 | 19.24 | **17.0 %** | 0.91 |
| +12 h | 26.17 | 22.66 | 13.4 % | 0.91 |
| +24 h | 24.10 | 23.62 | 2.0 % | 0.85 |

**Monsoon test — mid-July to mid-October 2026 (earlier run, last 90 days held out)**

| Horizon | No-change error | Our error | Skill |
|---|---|---|---|
| +1 h | 4.46 | 4.33 | 2.9 % |
| +3 h | 7.28 | 6.30 | 13.5 % |
| +6 h | 8.81 | 7.16 | 18.7 % |
| +12 h | 9.41 | 7.43 | **21.0 %** |
| +24 h | 8.47 | 7.73 | 8.7 % |

- The forecast beats "no change" at every horizon in both seasons, most strongly 3–12 h ahead — the window that matters for "when should I leave".
- Calibration lifted winter worst-case coverage from 0.76–0.88 to 0.85–0.92, so the safety advice for sensitive users is trustworthy.
- In winter, leave-one-station-out testing found that the plain city average estimates a hidden station best; the app uses that automatically.

**Performance on IBM LinuxONE:** each horizon model trains in about 2 seconds; a full route answer (3 routes + 24-hour best-time profile) on the 120,566-junction city graph takes about 0.3 seconds.

## IBM Z technologies used
| Technology | How CleanAir Routes uses it |
|---|---|
| **IBM LinuxONE (s390x)** | All data download, training, forecasting, routing and serving run on the LinuxONE Community Cloud VM |
| **Integrated Accelerator for AI (NNPA, IBM z17)** | Neural PM2.5 forecaster inference on the on-chip accelerator via `torch_nnpa`; speed and accuracy compared with the CPU in `reports/nnpa_benchmark.json` |
| **IBM Granite 3.3 2B (big-endian GGUF) + llama.cpp** | Local GenAI on LinuxONE — travel explanations on request and an hourly situation bulletin for officials. It runs in a background worker (about 1 minute per answer on 3 vCPUs), so the app stays responsive; no external AI service, no user data leaves the machine |
| **Containers on LinuxONE** | The app runs in a Docker container built on IBM's LinuxONE Jupyter image, bound to localhost only and reached through an encrypted SSH tunnel |

## Run on IBM LinuxONE
```bash
pip3 install -r requirements.txt               # optional: pip3 install snapml
export OPENAQ_API_KEY=xxxx                     # free key: https://explore.openaq.org/register
bash scripts/run_all.sh real 2024-01-01        # download stations, PM2.5, weather, roads, then train
python3 app.py                                 # replay demo  -> http://localhost:5000
LIVE=1 python3 app.py                          # live mode: latest readings + weather forecast, refreshed every 15 min
LOCAL_GGUF=/path/granite-3.3-2b-instruct-be.Q4_K_M.gguf python3 app.py   # + GenAI advice with IBM Granite
python3 scripts/nnpa_benchmark.py 1 6 24       # neural forecaster: CPU vs z17 AI accelerator (NNPA)
bash scripts/platform_evidence.sh              # architecture / CPU / library / model evidence
```
Access from a laptop through an encrypted SSH tunnel (port 5000 is never exposed to the internet):
`ssh -i key.pem -N -o ServerAliveInterval=30 -L 5000:localhost:5000 linux1@<vm-ip>`

**Container** (built on the Jupyter image, which already has the s390x libraries):
```bash
docker build --build-arg BASE=$(docker inspect -f '{{.Config.Image}}' notebook) -t cleanair-routes .
docker run -d --name cleanair --restart unless-stopped -p 127.0.0.1:5001:5000 \
  -v ~/models:/models:ro -e LOCAL_GGUF=/models/granite-3.3-2b-instruct-be.Q4_K_M.gguf cleanair-routes
```

**Offline smoke test:** `bash scripts/run_all.sh synthetic`

## Demo mode vs live mode
- **Replay (demo):** "Now" is a moment inside the held-out winter, just before the worst city-wide pollution spike. The model only sees readings up to "Now"; the chart then shows what really happened, so forecasts can be checked against reality. `+1 h` steps time forward like a live feed.
- **Live:** downloads the latest readings and weather every 15 minutes. The free public feed currently arrives ~4 days late, so true real-time forecasting needs the CPCB/TNPCB real-time feed (TNPCB already receives it from CPCB by API).

## Limitations (honest)
- Only ~8 stations across the city, so street-level estimates are coarse; differences between routes come mostly from avoiding busy arterial roads and from timing.
- The arterial-road uplift (up to +20 %) is a stated assumption, not measured.
- The CPCB stations' hourly history starts Feb 2025, so the model saw only one partial winter before the winter test.
- Public data arrives with a delay; the demo therefore replays real past data.

## Project structure
```
app.py                 Flask API + replay/live clock
src/fetch.py           automatic data download (OpenAQ, Open-Meteo, CAMS fallback)
src/data.py            cleaning, CPCB file loader, synthetic test data
src/model.py           features, horizon models, hybrid, calibrated p90, validation
src/spatial.py         between-station estimation + leave-one-station-out test
src/routing.py         OpenStreetMap download (tiled, retrying) + dose-aware routing
src/advisory.py        India NAQI categories, advice in English + Tamil, GenAI layer (Granite via llama.cpp)
src/officials.py       officials' dashboard: 24 h outlook, city summary, public alerts
web/                   map web app (Leaflet + Chart.js, bundled locally)
scripts/               full pipeline, platform evidence, NNPA benchmark
Dockerfile             container for LinuxONE
```

## Open-source used
pandas · NumPy · scikit-learn · NetworkX · Flask · Requests · joblib · openpyxl · PyTorch · torch_nnpa · llama-cpp-python · IBM Granite 3.3 · Leaflet · Chart.js.
Data: OpenAQ (CC BY 4.0), Open-Meteo (CC BY 4.0), OpenStreetMap (ODbL), CPCB / TNPCB / AirNow station data.
