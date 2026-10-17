# Team game plan (Oct 17 → submit by 09:30 IST Oct 18)

## Roles (4–6 people)
| Who | Job right now |
|---|---|
| A — Data | Download hourly PM2.5 (+PM10) from CPCB CCR for all Chennai stations, 2023 → today. Verify station lat/lon in `src/config.py`. |
| B — Platform | LinuxONE VM: finish the 2 mandatory IBM labs, clone repo, `pip install`, run `scripts/run_all.sh synthetic`, try `pip install snapml`, `podman build`. Screenshot everything. |
| C — Model | Once real data lands: run `real` pipeline, check metrics.json, tune (more lags, PM10 feature, Diwali/holiday flag). |
| D — App/Demo | Pick 2–3 demo trips in the demo area (e.g. Guindy → Velachery school run), find a replay date with a pollution spike, polish UI. |
| E/F — Pitch | Architecture slide, video script, recording, Hack2skill submission, GitHub README + open-source links. |

## Timeline (IST)
| Time | Milestone |
|---|---|
| 15:00 | Repo on GitHub; synthetic pipeline runs on LinuxONE |
| 18:00 | Real CPCB + weather data loaded; first real metrics |
| 21:00 | OSM graph for demo area; app running on LinuxONE via SSH tunnel |
| 00:00 | Model tuned; demo scenarios chosen; platform evidence screenshots |
| 03:00 | Video recorded (rough cut) |
| 07:00 | Final video ≤ 3:00, README final |
| 09:30 | **Submitted** (hard deadline 10:00) |

## 3-minute video script
- **0:00–0:20 Problem.** "An asthmatic child in Velachery walks to school at 8 am — exactly when PM2.5 peaks. Nobody tells her family that leaving at 7:15, or taking the inner street, halves what she breathes in."
- **0:20–0:40 Data.** Show CPCB hourly data chart: daily peaks, winter spike, gap between stations.
- **0:40–1:50 Demo.** Pick asthma profile → set trip → three routes appear → "cleanest cuts dose by X %" → best-time chart → press +1 h to show the real-time replay, forecast vs actual.
- **1:50–2:35 IBM Z.** Terminal: `uname -m` = s390x, `platform_evidence.sh`, training run, podman container, metrics table (skill vs persistence, p90 coverage, leave-one-station-out). Why LinuxONE: always-on public-health service, secure data, scales city-wide.
- **2:35–3:00 Impact + next.** More cities, low-cost sensors, SMS alerts for schools/clinics.

## Submission checklist
- [ ] Title: `[TeamName_IBM Z Datathon 2026 Project]`
- [ ] Video ≤ 3 min showing LinuxONE instance + model
- [ ] GitHub link; open-source links in README
- [ ] Optional: architecture flowchart, prototype link
