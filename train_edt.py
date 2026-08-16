# -*- coding:utf-8 -*-
"""Train an EDT (decision tree with per-node value counter) for UNIV1/ISCX/Bot-IoT,
save the soft-pruned tree as JSON into json_models/, and print test metrics."""
import sys, os, json, copy, time, pickle, gc
import numpy as np
from collections import Counter
import sklearn.tree as st
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "model_data")
OUT = os.path.join(ROOT, "json_models")
os.makedirs(OUT, exist_ok=True)

UNIV1_FEATURES = ['Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
                   'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)']

MAX_DEPTH = None  # no depth limit


def sklearn2json(model, feature_list, class_names, node_index=0):
    """sklearn tree -> json; each node keeps `value` (per-class sample counts)."""
    jm = {}
    jm['value'] = [c for c in model.tree_.value[node_index, 0]]
    if model.tree_.children_left[node_index] == -1:  # leaf
        return jm
    feature = feature_list[model.tree_.feature[node_index]]
    threshold = model.tree_.threshold[node_index]
    jm['name'] = '{} <= {}'.format(feature, threshold)
    jm['feature'] = '{}'.format(feature)
    jm['threshold'] = '{}'.format(threshold)
    left = model.tree_.children_right[node_index]
    right = model.tree_.children_left[node_index]
    jm['children'] = [sklearn2json(model, feature_list, class_names, right),
                      sklearn2json(model, feature_list, class_names, left)]
    return jm


def hard_prune(jm, limit_depth):
    """BFS; attach leafcount/tobedel bookkeeping needed by soft_prune."""
    q, dq = [jm], [0]
    while q:
        n = q.pop(0); d = dq.pop(0)
        n["tobedel"] = 0
        n["leafcount"] = [0, 0]
        if "children" not in n:
            n["leafcount"][0] = 1
        else:
            if d == limit_depth:
                del n["children"]
                n["leafcount"][0] = 1
            else:
                for c in n["children"]:
                    q.append(c); dq.append(d + 1)
    return jm


def soft_prune(jm):
    """Merge subtrees whose leaves all share the parent's majority class."""
    classNameStack, st_node = [], jm
    nodeStack = []
    while nodeStack or st_node:
        while st_node:
            nodeStack.append(st_node)
            st_node = st_node["children"][0] if "children" in st_node else None
        cur = nodeStack.pop()
        if "children" in cur and len(cur["children"]) > 0:
            cur["leafcount"][0] = cur["children"][0]["leafcount"][0] + cur["children"][0]["leafcount"][1]
            cur["leafcount"][1] = cur["children"][1]["leafcount"][0] + cur["children"][1]["leafcount"][1]
            cls = int(np.argmax(cur['value']))
            flag, count = 1, cur["leafcount"][0] + cur["leafcount"][1]
            for cn in classNameStack[-count:]:
                if cn != cls:
                    flag = 0; break
            if flag == 1:
                cur["tobedel"] = 1
                del cur["children"]
        else:
            classNameStack.append(int(np.argmax(cur['value'])))
        if nodeStack and nodeStack[-1]["children"][0] is cur:
            st_node = nodeStack[-1]["children"][1]
        else:
            st_node = None
    return jm


def load_dataset(name):
    """Load a dataset from model_data/<name>/{train,test}.pkl ({"X","Y"} dict)."""
    base = os.path.join(DATA, name)
    with open(f"{base}/train.pkl", "rb") as f: tr = pickle.load(f)
    with open(f"{base}/test.pkl", "rb") as f: te = pickle.load(f)
    Xtr, Ytr = np.asarray(tr["X"]), np.asarray(tr["Y"])
    Xte, Yte = np.asarray(te["X"]), np.asarray(te["Y"])
    feats = UNIV1_FEATURES if name == "univ1" else [f"f{i}" for i in range(Xtr.shape[1])]
    classes = [str(i) for i in range(int(max(Ytr.max(), Yte.max())) + 1)]
    return Xtr, Ytr, Xte, Yte, feats, classes


DATASETS = {
    "univ1": lambda: load_dataset("univ1"),
    "iscx": lambda: load_dataset("iscx"),
    "bot_iot": lambda: load_dataset("bot_iot"),
}


def train_one(name, max_depth=None):
    print(f"\n{'='*60}\n[{name}] loading data ...", flush=True)
    t0 = time.time()
    Xtr, Ytr, Xte, Yte, feats, classes = DATASETS[name]()
    print(f"[{name}] Xtr {Xtr.shape} Ytr {Counter(Ytr)} | Xte {Xte.shape} Yte {Counter(Yte)}", flush=True)
    print(f"[{name}] train/test split done in {time.time()-t0:.1f}s", flush=True)

    print(f"[{name}] training DecisionTree (max_depth={max_depth}) ...", flush=True)
    t1 = time.time()
    model = st.DecisionTreeClassifier(max_depth=max_depth, random_state=5, max_features=0.3, min_samples_leaf=10)
    model.fit(Xtr, Ytr)
    print(f"[{name}] trained in {time.time()-t1:.1f}s; n_nodes={model.tree_.node_count}", flush=True)

    # EDT json (with per-node value) + soft pruning
    jm = sklearn2json(model, feats, classes)
    jm = hard_prune(jm, limit_depth=max_depth if max_depth else 10**9)
    jm = soft_prune(copy.deepcopy(jm))

    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    out = os.path.join(OUT, name, f"{name}_edt.json")
    with open(out, 'w') as f:
        json.dump(jm, f)
    print(f"[{name}] saved EDT json -> {out} ({os.path.getsize(out)//1024} KB)", flush=True)

    # metrics on test (soft-prune does not change predictions)
    pred = model.predict(Xte)
    acc = accuracy_score(Yte, pred)
    f1m = f1_score(Yte, pred, average='macro', zero_division=0)
    precm = precision_score(Yte, pred, average='macro', zero_division=0)
    recm = recall_score(Yte, pred, average='macro', zero_division=0)
    print(f"[{name}] TEST metrics: Accuracy={acc:.4f} | macro-F1={f1m:.4f} | "
          f"macro-Precision={precm:.4f} | macro-Recall={recm:.4f}", flush=True)
    print(classification_report(Yte, pred, zero_division=0), flush=True)
    del Xtr, Ytr, Xte, Yte, model, jm; gc.collect()
    return acc, f1m, precm, recm


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    depth = MAX_DEPTH
    if '--depth' in sys.argv:
        i = sys.argv.index('--depth'); depth = int(sys.argv[i+1])
    targets = ['univ1', 'iscx', 'bot_iot'] if which == 'all' else [which]
    res = {}
    for ds in targets:
        try:
            res[ds] = train_one(ds, max_depth=depth)
        except Exception as e:
            import traceback; traceback.print_exc()
            res[ds] = None
    print("\n==== SUMMARY ====")
    for ds in targets:
        if res[ds]:
            a, f1, p, r = res[ds]
            print(f"{ds:8s} Acc={a:.4f} F1(macro)={f1:.4f} Prec(macro)={p:.4f} Rec(macro)={r:.4f}")
        else:
            print(f"{ds:8s} FAILED")
