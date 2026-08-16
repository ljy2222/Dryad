#!/usr/bin/env bash
# DryadV2 pipeline: EDT → Distillation → Pruning → GA Optimization
# Prerequisite: model_data/{univ1,iscx,bot_iot}/{full,train,test}.pkl exist
# Usage: ./run_all.sh
set -e
cd "$(dirname "$0")"

echo "================================================================"
echo " DryadV2 Pipeline:  EDT → Distillation(RF) → Pruning → GA"
echo "================================================================"

# Step 1: EDT training
echo ""
echo "[1/4] EDT training (train_edt.py)"
echo "----------------------------------------------------------------"
python3 train_edt.py univ1
python3 train_edt.py iscx
python3 train_edt.py bot_iot

# Step 2: Distilled EDT training (RF teacher)
echo ""
echo "[2/4] Distilled EDT training (train_edt_distill.py, RF teacher)"
echo "----------------------------------------------------------------"
python3 train_edt_distill.py univ1
python3 train_edt_distill.py iscx
python3 train_edt_distill.py bot_iot

# Step 3: Hard/soft pruning on distilled EDT
echo ""
echo "[3/4] Hard/soft pruning on distilled EDT (prune_edt.py, depths 4/8/12)"
echo "----------------------------------------------------------------"
python3 prune_edt.py --tree ./json_models/univ1/univ1_edt_rf.json   --depths 4,8,12
python3 prune_edt.py --tree ./json_models/iscx/iscx_edt_rf.json     --depths 4,8,12
python3 prune_edt.py --tree ./json_models/bot_iot/bot_iot_edt_rf.json --depths 4,8,12

# Step 4: GA optimization on distilled EDT (UNIV1)
echo ""
echo "[4/4] GA optimization on distilled EDT (optimization.py, UNIV1)"
echo "----------------------------------------------------------------"
python3 optimization.py --tree ./json_models/univ1/univ1_edt_rf.json

echo ""
echo "================================================================"
echo " [done] Pipeline complete."
echo "  EDT models:      json_models/{univ1,iscx,bot_iot}/*_edt.json"
echo "  Distilled EDTs:  json_models/{univ1,iscx,bot_iot}/*_edt_rf.json"
echo "  GA optimized:    json_models/univ1/genetic_optimized_tree.json"
echo "================================================================"
