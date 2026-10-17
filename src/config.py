"""Central configuration for CleanAir Routes (Chennai)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"          # put CPCB CCR exports here (xlsx/csv)
PROC = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for _p in (RAW, PROC, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

CITY_CENTER = (13.0827, 80.2707)

# Chennai CPCB CAAQMS stations. Coordinates are APPROXIMATE -
# verify against the CPCB CCR station list before the final run.
STATIONS = {
    "Alandur":      (12.9975, 80.2006),
    "Velachery":    (13.0050, 80.2398),
    "Perungudi":    (12.9532, 80.2357),
    "Arumbakkam":   (13.0732, 80.2109),
    "Royapuram":    (13.1040, 80.2932),
    "Kodungaiyur":  (13.1320, 80.2640),
    "Manali":       (13.1644, 80.2639),
    "Manali Village": (13.1662, 80.2584),
}

# If `python3 -m src.fetch stations` has run, use the real station list + exact coordinates from OpenAQ
STATIONS_FILE = PROC / "stations.json"
if STATIONS_FILE.exists():
    import json as _json
    STATIONS = {k: (v["lat"], v["lon"]) for k, v in _json.loads(STATIONS_FILE.read_text()).items()}

# Demo routing area: Guindy - Alandur - Velachery - Perungudi (south, west, north, east)
DEMO_BBOX = (12.955, 80.195, 13.025, 80.255)

# Routing areas (one road graph each, keeps every route fast). Add a line to cover more of the city.
AREAS = {
    # one box around all 7 CPCB/TNPCB stations (Perungudi -> Manali, Arumbakkam -> coast) + ~1.5 km margin
    "chennai": {"label": "Chennai · all monitoring stations", "bbox": (12.935, 80.195, 13.185, 80.305)},
}
# More boxes can be added, e.g. "west": {"label": "Avadi · Ambattur", "bbox": (13.050, 80.080, 13.140, 80.220)}

HORIZONS = [1, 3, 6, 12, 24]       # forecast horizons (hours ahead), one model each
LAGS = [1, 2, 3, 6, 12, 24, 48]
TEST_DAYS = 90                     # fallback: held-out last N days if the winter isn't in the data
# Held-out test period = Chennai's pollution season (blocked hold-out). Override with env vars.
import os as _os
TEST_START = _os.getenv("TEST_START", "2025-12-01")
TEST_END = _os.getenv("TEST_END", "2026-02-28")

# Travel modes: speed (km/h) and breathing rate (m3/h) for inhaled-dose estimate
MODES = {
    "walk":  {"speed_kmh": 4.8,  "vent_m3h": 1.4},
    "cycle": {"speed_kmh": 14.0, "vent_m3h": 2.0},
}

# Heuristic near-road uplift by road class (set all to 1.0 to disable).
# Busy arterials tend to have higher roadside PM2.5 than quiet streets.
ROAD_UPLIFT = {
    "trunk": 1.20, "trunk_link": 1.20, "primary": 1.15, "primary_link": 1.15,
    "secondary": 1.08, "secondary_link": 1.08, "tertiary": 1.03,
    "residential": 1.00, "unclassified": 1.00, "living_street": 0.97,
    "footway": 0.95, "pedestrian": 0.95, "path": 0.93, "cycleway": 0.95,
}

# Sensitive-user profiles: AQI-category threshold at which we warn
PROFILES = {
    "general":  {"label": "General public",          "warn_at": 300},
    "asthma":   {"label": "Asthma / respiratory",    "warn_at": 100},
    "elderly":  {"label": "Elderly",                 "warn_at": 100},
    "child":    {"label": "Children",                "warn_at": 100},
    "outdoor":  {"label": "Commuter / outdoor worker", "warn_at": 200},
}
