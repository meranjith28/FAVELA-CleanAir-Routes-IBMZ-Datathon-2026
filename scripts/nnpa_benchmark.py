"""Neural PM2.5 forecaster on the IBM z17 on-chip AI accelerator (NNPA) vs the CPU.

Trains a small neural network (PyTorch) on the same features and the same held-out winter test
period as the main model, then runs the test-set predictions on:
  * the CPU, and
  * the Integrated Accelerator for AI (NNPA) through torch_nnpa,
and compares speed and accuracy. Results -> reports/nnpa_benchmark.json

    python3 scripts/nnpa_benchmark.py            # default horizon +6 h
    python3 scripts/nnpa_benchmark.py 1 6 24     # several horizons
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import REPORTS                     # noqa: E402
from src.data import load_processed                # noqa: E402
from src.model import build_features, test_window  # noqa: E402

try:
    import torch_nnpa  # noqa: F401  registers the "nnpa" device on IBM z16 / z17
    HAVE_NNPA = True
except Exception as e:
    HAVE_NNPA, NNPA_ERR = False, str(e)

torch.manual_seed(0)
np.random.seed(0)


def split(h):
    pm, w = load_processed()
    t0, t1 = test_window(pm)
    buf = pd.Timedelta(hours=72)
    df, feats = build_features(pm, w, h)
    df = df.dropna(subset=["target", "pm_now"])
    te = df[(df.time >= t0) & (df.time <= t1)]
    tr = df[(df.time < t0 - buf) | (df.time > t1 + buf)]
    med = tr[feats].median()
    Xtr, Xte = tr[feats].fillna(med).values.astype(np.float32), te[feats].fillna(med).values.astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    # clip standardised inputs so unseen winter extremes can't push the network into wild extrapolation
    Xtr, Xte = np.clip((Xtr - mu) / sd, -5, 5), np.clip((Xte - mu) / sd, -5, 5)
    # learn the CHANGE from the current level (log scale): when unsure the net falls back to "no change"
    ytr = (np.log1p(tr["target"].values) - np.log1p(tr["pm_now"].values)).astype(np.float32)
    return (Xtr.astype(np.float32), ytr, Xte.astype(np.float32), te["target"].values,
            te["pm_now"].values, f"{t0:%d %b %Y} - {t1:%d %b %Y}")


def to_pm(delta, pm_now):
    """change prediction -> PM2.5, clipped to a physically plausible range"""
    d = np.clip(np.asarray(delta, dtype=np.float64).ravel(), -3, 3)
    return np.expm1(np.clip(np.log1p(pm_now) + d, 0, np.log1p(999)))


def make_net(n_in):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


def train_cpu(net, X, y, epochs=12, bs=512):
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X, y = torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.l1_loss(net(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return net


def timed(fn, repeats=20):
    fn()                                   # warm-up (first NNPA call includes setup)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return out, 1000 * float(np.median(ts))


def run(h):
    Xtr, ytr, Xte, yte, now_te, label = split(h)
    net = make_net(Xtr.shape[1])
    t = time.perf_counter()
    train_cpu(net, Xtr, ytr)
    train_s = time.perf_counter() - t
    net.eval()
    xb = torch.from_numpy(Xte)
    x1 = xb[:8]                                         # one request = ~8 stations

    with torch.no_grad():
        cpu_pred, cpu_ms = timed(lambda: net(xb))
        _, cpu_ms_1 = timed(lambda: net(x1), repeats=200)
    cpu_pm = to_pm(cpu_pred.numpy(), now_te)
    res = {"horizon_h": h, "test_period": label, "n_train": len(Xtr), "n_test": len(Xte),
           "train_seconds_cpu": round(train_s, 1),
           "mae_persistence": round(float(np.mean(np.abs(now_te - yte))), 2),
           "cpu": {"mae": round(float(np.mean(np.abs(cpu_pm - yte))), 2),
                   "batch_ms": round(cpu_ms, 2), "single_request_ms": round(cpu_ms_1, 3)}}

    if HAVE_NNPA:
        try:
            net_n = make_net(Xtr.shape[1]); net_n.load_state_dict(net.state_dict()); net_n.eval()
            net_n = net_n.to("nnpa")
            xn, x1n = xb.to("nnpa"), x1.to("nnpa")
            with torch.no_grad():
                nn_pred, nn_ms = timed(lambda: net_n(xn).to("cpu"))
                _, nn_ms_1 = timed(lambda: net_n(x1n).to("cpu"), repeats=200)
            nn_pm = to_pm(nn_pred.float().numpy(), now_te)
            diff = np.abs(nn_pm - cpu_pm)
            res["nnpa"] = {"mae": round(float(np.mean(np.abs(nn_pm - yte))), 2),
                           "batch_ms": round(nn_ms, 2), "single_request_ms": round(nn_ms_1, 3),
                           "mean_abs_diff_vs_cpu_ug_m3": round(float(diff.mean()), 4),
                           "p99_abs_diff_vs_cpu_ug_m3": round(float(np.quantile(diff, 0.99)), 3),
                           "max_abs_diff_vs_cpu_ug_m3": round(float(diff.max()), 3)}
            res["nnpa_speedup_batch"] = round(cpu_ms / nn_ms, 2) if nn_ms else None
        except Exception as e:
            res["nnpa"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        res["nnpa"] = {"error": f"torch_nnpa not available here: {NNPA_ERR}"}
    return res


if __name__ == "__main__":
    horizons = [int(a) for a in sys.argv[1:]] or [6]
    report = {"platform": {"machine": platform.machine(), "torch": torch.__version__, "nnpa_available": HAVE_NNPA},
              "model": "MLP 128-64-1 predicting the change in log PM2.5 from now (clipped inputs/outputs), trained on CPU; inference on CPU and on NNPA", "runs": []}
    print(f"machine={platform.machine()}  torch={torch.__version__}  NNPA={'yes' if HAVE_NNPA else 'no'}")
    for h in horizons:
        r = run(h)
        report["runs"].append(r)
        n = r["nnpa"]
        print(f"\n+{h} h  (test {r['test_period']}, {r['n_test']} rows; trained in {r['train_seconds_cpu']} s)")
        print(f"  no-change MAE {r['mae_persistence']:6.2f}")
        print(f"  CPU   MAE {r['cpu']['mae']:6.2f} | full test set {r['cpu']['batch_ms']:8.2f} ms | one request {r['cpu']['single_request_ms']:.3f} ms")
        if "error" in n:
            print(f"  NNPA  {n['error']}")
        else:
            print(f"  NNPA  MAE {n['mae']:6.2f} | full test set {n['batch_ms']:8.2f} ms | one request {n['single_request_ms']:.3f} ms"
                  f"\n        difference vs CPU: average {n['mean_abs_diff_vs_cpu_ug_m3']} | 99% of rows within {n['p99_abs_diff_vs_cpu_ug_m3']} | max {n['max_abs_diff_vs_cpu_ug_m3']} ug/m3")
    (REPORTS / "nnpa_benchmark.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {REPORTS / 'nnpa_benchmark.json'}")
