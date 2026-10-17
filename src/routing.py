"""Road graphs (OpenStreetMap via Overpass) + exposure-aware routing.

    python3 -m src.routing fetch          # download roads for every area in config.AREAS
    python3 -m src.routing fetch west     # just one area
    python3 -m src.routing synthetic      # offline grid graphs for testing
"""
import pickle
import sys

import networkx as nx
import numpy as np
import requests

from .config import AREAS, MODES, PROC, ROAD_UPLIFT, STATIONS
from .spatial import best_power, haversine_km, idw

HWY = "|".join(ROAD_UPLIFT)
LEGACY_GRAPH = PROC / "graph.pkl"          # single-area file from earlier versions = the "south" area


def graph_path(area):
    return PROC / f"graph_{area}.pkl"


OVERPASS = ["https://overpass-api.de/api/interpreter",          # main server first
            "https://overpass.private.coffee/api/interpreter",  # public mirrors as fallback
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
HEADERS = {"User-Agent": "CleanAirRoutes/1.0 (IBM Z Datathon 2026)", "Accept": "application/json"}
TILE_DEG = 0.05            # ~5.5 km tiles: small requests rarely time out on busy servers


def _overpass(q, rounds=3):
    """POST a query, retrying across servers with growing waits when they are busy."""
    import time
    last = None
    for rnd in range(rounds):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=300)
                if r.status_code == 200:
                    OVERPASS.remove(url); OVERPASS.insert(0, url)     # use the server that answered first next time
                    return r.json()["elements"]
                last = f"{url.split('/')[2]} HTTP {r.status_code}"
            except Exception as ex:
                last = f"{url.split('/')[2]} {ex.__class__.__name__}"
            print(f"     busy: {last}", flush=True)
        wait = 20 * (rnd + 1)
        print(f"     all servers busy - waiting {wait}s and retrying ...", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"Overpass unavailable ({last})")


def fetch_osm(area="south"):
    s, w, n, e = AREAS[area]["bbox"]
    ny, nx_ = max(1, round((n - s) / TILE_DEG)), max(1, round((e - w) / TILE_DEG))
    lats, lons = np.linspace(s, n, ny + 1), np.linspace(w, e, nx_ + 1)
    print(f"{area}: downloading roads from OpenStreetMap in {ny * nx_} tiles ...", flush=True)
    nodes, ways = {}, {}
    for i in range(ny):
        for j in range(nx_):
            bb = f"{lats[i]:.4f},{lons[j]:.4f},{lats[i + 1]:.4f},{lons[j + 1]:.4f}"
            q = f'[out:json][timeout:180];way["highway"~"^({HWY})$"]({bb});(._;>;);out body;'
            els = _overpass(q)
            for el in els:
                if el["type"] == "node":
                    nodes[el["id"]] = (el["lat"], el["lon"])
                elif el["type"] == "way":
                    ways[el["id"]] = el                 # ways crossing tile edges are de-duplicated by id
            print(f"   tile {i * nx_ + j + 1}/{ny * nx_}: {len(ways)} roads so far", flush=True)
    G = nx.Graph()
    for el in ways.values():
        hw = el["tags"].get("highway", "unclassified")
        name = el["tags"].get("name", "")
        nds = [x for x in el["nodes"] if x in nodes]
        for a, b in zip(nds[:-1], nds[1:]):
            la, lb = nodes[a], nodes[b]
            G.add_node(a, lat=la[0], lon=la[1]); G.add_node(b, lat=lb[0], lon=lb[1])
            G.add_edge(a, b, length_km=float(haversine_km(la[0], la[1], lb[0], lb[1])), highway=hw, name=name)
    return _finish(G, area)


def synthetic_graph(area="south", n=40, seed=3):
    rng = np.random.default_rng(seed)
    s, w, nn, e = AREAS[area]["bbox"]
    lats, lons = np.linspace(s, nn, n), np.linspace(w, e, n)
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, lat=float(lats[i] + rng.normal(0, 2e-4)), lon=float(lons[j] + rng.normal(0, 2e-4)))
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < n and j + dj < n:
                    hw = "primary" if (i % 10 == 0 or j % 10 == 0) else ("secondary" if (i % 5 == 0 or j % 5 == 0) else "residential")
                    a, b = i * n + j, (i + di) * n + j + dj
                    A, B = G.nodes[a], G.nodes[b]
                    G.add_edge(a, b, length_km=float(haversine_km(A["lat"], A["lon"], B["lat"], B["lon"])), highway=hw, name=hw)
    return _finish(G, area)


def _finish(G, area):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    p = graph_path(area)
    with open(p, "wb") as f:
        pickle.dump(G, f)
    print(f"{area}: graph {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {p}")
    return G


def available_areas():
    """Areas whose road graph has been downloaded."""
    out = [a for a in AREAS if graph_path(a).exists()]
    if "south" in AREAS and "south" not in out and LEGACY_GRAPH.exists():
        out.insert(0, "south")
    return out


def area_of(lat, lon, areas=None):
    for a in (areas or AREAS):
        s, w, n, e = AREAS[a]["bbox"]
        if s <= lat <= n and w <= lon <= e:
            return a
    return None


class Router:
    def __init__(self, area="south", G=None):
        if G is None:
            p = graph_path(area)
            if not p.exists() and area == "south":
                p = LEGACY_GRAPH
            with open(p, "rb") as f:
                G = pickle.load(f)
        self.area = area
        self.G = G
        self.power = best_power()          # distance weighting chosen by leave-one-station-out
        self.node_ids = np.array(list(G.nodes))
        self.node_xy = np.array([[G.nodes[n]["lat"], G.nodes[n]["lon"]] for n in self.node_ids])
        self.edges = list(G.edges)
        mids = []
        for a, b in self.edges:
            A, B = G.nodes[a], G.nodes[b]
            mids.append(((A["lat"] + B["lat"]) / 2, (A["lon"] + B["lon"]) / 2))
        self.mid = np.array(mids)
        self.uplift = np.array([ROAD_UPLIFT.get(G.edges[e]["highway"], 1.0) for e in self.edges])
        self.length = np.array([G.edges[e]["length_km"] for e in self.edges])

    def nearest(self, lat, lon):
        d = (self.node_xy[:, 0] - lat) ** 2 + ((self.node_xy[:, 1] - lon) * np.cos(np.radians(lat))) ** 2
        return self.node_ids[int(np.argmin(d))]

    def _conc(self, points, uplift, station_values):
        sts = [s for s in station_values if s in STATIONS]
        return idw(points, [STATIONS[s] for s in sts], [station_values[s] for s in sts], power=self.power) * uplift

    def set_pollution(self, station_values, mode="walk"):
        """station_values: {station: pm25}. Writes time_h, conc and dose (ug inhaled) onto every edge."""
        conc = self._conc(self.mid, self.uplift, station_values)
        m = MODES[mode]
        t = self.length / m["speed_kmh"]
        dose = conc * m["vent_m3h"] * t
        for (a, b), c, tt, d in zip(self.edges, conc, t, dose):
            ed = self.G.edges[a, b]
            ed["conc"], ed["time_h"], ed["dose"] = float(c), float(tt), float(d)
        self._mean_conc, self._vent = float(np.average(conc, weights=self.length)), m["vent_m3h"]

    def dose_along(self, path, station_values, mode="walk"):
        """Inhaled dose for a FIXED route under another hour's pollution (used for 'best time to leave')."""
        G = self.G
        mids, up, L = [], [], []
        for a, b in zip(path[:-1], path[1:]):
            A, B, ed = G.nodes[a], G.nodes[b], G.edges[a, b]
            mids.append(((A["lat"] + B["lat"]) / 2, (A["lon"] + B["lon"]) / 2))
            up.append(ROAD_UPLIFT.get(ed["highway"], 1.0)); L.append(ed["length_km"])
        if not mids:
            return 0.0, 0.0
        conc = self._conc(np.array(mids), np.array(up), station_values)
        m = MODES[mode]
        t = np.array(L) / m["speed_kmh"]
        dose = float((conc * m["vent_m3h"] * t).sum())
        return dose, float(np.average(conc, weights=L))

    def _summ(self, path):
        e = [self.G.edges[a, b] for a, b in zip(path[:-1], path[1:])]
        t = sum(x["time_h"] for x in e); d = sum(x["dose"] for x in e); L = sum(x["length_km"] for x in e)
        coords = [[self.G.nodes[n]["lat"], self.G.nodes[n]["lon"]] for n in path]
        return {"minutes": round(60 * t, 1), "km": round(L, 2), "dose_ug": round(d, 2),
                "avg_pm25": round(d / (self._vent * t), 1) if t else 0, "coords": coords,
                "seg_pm25": [round(x["conc"], 1) for x in e], "_path": path}

    def _sp(self, a, b, weight):
        """Shortest path searching from both ends at once - much faster on city-sized graphs."""
        return nx.bidirectional_dijkstra(self.G, a, b, weight=weight)[1]

    def routes(self, src, dst, max_detour=0.25):
        a, b = self.nearest(*src), self.nearest(*dst)
        fast = self._sp(a, b, "time_h")
        clean = self._sp(a, b, "dose")
        F, C = self._summ(fast), self._summ(clean)
        # balanced: least dose among routes within max_detour of the fastest time
        best = C if C["minutes"] <= F["minutes"] * (1 + max_detour) else F
        k = self._vent * self._mean_conc
        for alpha in (0.5, 1, 2):
            if best is C:
                break
            p = self._sp(a, b, lambda u, v, d, al=alpha: d["dose"] + al * k * d["time_h"])
            S = self._summ(p)
            if S["minutes"] <= F["minutes"] * (1 + max_detour) and S["dose_ug"] < best["dose_ug"]:
                best = S
        return {"fastest": F, "cleanest": C, "balanced": best}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    areas = sys.argv[2:] or list(AREAS)
    for a in areas:
        if cmd == "fetch":
            try:
                fetch_osm(a)
            except Exception as ex:
                print(f"{a}: FAILED ({ex}) - other areas still usable")
        else:
            synthetic_graph(a)
