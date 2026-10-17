#!/usr/bin/env bash
# Full pipeline.
#   scripts/run_all.sh synthetic                      offline smoke test
#   scripts/run_all.sh real 2023-01-01 2026-10-17     download real data on the server, then train
# For measured data set a free OpenAQ key first:  export OPENAQ_API_KEY=xxxx
# (without it, the modelled CAMS PM2.5 fallback is used)
set -e
cd "$(dirname "$0")/.."
if [ "$1" = "real" ]; then
  START=${2:-2023-01-01}; END=${3:-$(date +%F)}
  python3 -m src.fetch all "$START" "$END"     # stations + PM2.5 + weather + OSM roads
else
  python3 -m src.data synthetic
  python3 -m src.routing synthetic
fi
python3 -m src.model train ${SNAPML:+--snapml}
echo "Done. Demo on history replay:  python3 app.py"
echo "      Live mode (latest data):  LIVE=1 python3 app.py"
