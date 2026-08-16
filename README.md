# DryadV2

## Quick Start

```bash
# 1. Build datasets (one-time, from raw_data/)
cd model_data && ./build_all.sh && cd ..

# 2. Run the full pipeline (EDT → Distillation → Pruning → GA)
./run_all.sh
```

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `train_edt.py` | Train EDT (decision tree with per-node value counter) per dataset |
| 2 | `train_edt_distill.py` | Distill EDT from RF teacher (soft-label replication, α=0.75) |
| 3 | `prune_edt.py` | Hard/soft prune distilled EDT at depths 4/8/12 |
| 4 | `optimization.py` | Genetic algorithm optimization on distilled EDT (UNIV1) |

## Directory Structure

```
code/Dryad/
├── run_all.sh              # One-click pipeline (steps 1-4)
├── train_edt.py            # EDT training
├── train_edt_distill.py    # RF teacher distillation
├── prune_edt.py            # Hard/soft pruning
├── optimization.py         # GA optimization
├── json_models/            # Trained models (per dataset)
│   ├── univ1/              # univ1_edt.json, univ1_edt_rf.json
│   ├── iscx/               # iscx_edt.json, iscx_edt_rf.json
│   └── bot_iot/            # bot_iot_edt.json, bot_iot_edt_rf.json
└── model_data/
    ├── build_all.sh        # Dataset build (raw_data → full/train/test)
    ├── build_datasets.py   # Raw sources → full.pkl
    ├── split.py            # full.pkl → train.pkl + test.pkl
    ├── pkl2pcap.py         # test.pkl → .pcap (hardware testing)
    ├── raw_data/           # Original source files
    ├── univ1/              # full.pkl, train.pkl, test.pkl
    ├── iscx/               # full.pkl, train.pkl, test.pkl
    └── bot_iot/            # full.pkl, train.pkl, test.pkl
```
