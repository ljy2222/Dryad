#!/usr/bin/env bash
# =====================================================================
# Complete dataset build pipeline (model_data/):
#   1. build_datasets.py — raw_data/*.pkl -> full.pkl (per dataset)
#   2. split.py          — full.pkl -> train.pkl + test.pkl (80/20, seed=5)
#   3. pkl2pcap.py       — test.pkl -> .pcap (UNIV1, for hardware switch testing)
# All three Python tools are called by this script.
#
# Prerequisite: model_data/raw_data/ must contain the 6 raw source files
#   (one-time setup: git checkout + git lfs pull, then mv sources raw_data/).
# Usage:  ./build_all.sh   (run from anywhere)
# =====================================================================
set -e
cd "$(dirname "$0")"            # -> model_data/

DATASETS="univ1 iscx bot_iot"

# ---- Step 1: build_datasets.py — raw_data/ -> full.pkl ----
# Runs only if any full.pkl is missing (to rebuild, delete full.pkl first).
need_build=0
for ds in $DATASETS; do [ -f "$ds/full.pkl" ] || need_build=1; done
if [ "$need_build" -eq 1 ]; then
    echo "[1/3] build_datasets.py: raw_data/ -> full.pkl"
    python3 build_datasets.py
else
    echo "[1/3] full.pkl already exists — skipping build_datasets.py"
    echo "       (to rebuild from raw_data/, delete full.pkl then re-run)"
fi

# ---- Step 2: split.py — full.pkl -> train.pkl + test.pkl (80/20, seed=5) ----
echo "[2/3] split.py: full.pkl -> train.pkl + test.pkl"
for ds in $DATASETS; do
    python3 split.py "$ds"
done

# ---- Step 3: pkl2pcap.py — test.pkl -> .pcap (UNIV1, for hardware testing) ----
echo "[3/3] pkl2pcap.py: univ1/test.pkl -> univ1/test.pcap (100 packets)"
python3 pkl2pcap.py -i ./univ1/test.pkl -o ./univ1/test.pcap -m 100

echo ""
echo "[done] Build complete:"
echo "  model_data/{univ1,iscx,bot_iot}/{full,train,test}.pkl"
echo "  model_data/univ1/test.pcap + test_ground_truth.txt (hardware test)"
