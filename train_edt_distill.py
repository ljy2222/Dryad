# -*- coding:utf-8 -*-
"""Distill an EDT from an RF teacher.

RF teacher (sklearn RandomForest, 100 trees, 2-fold cross-fit) produces soft
labels; the EDT is trained via soft-label replication (alpha=0.75).
"""
import os, sys, json, copy, time, gc
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_edt import (sklearn2json, hard_prune, soft_prune,
                       load_dataset, OUT)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data")
TRAIN_CAP = None
DEPTH = None
TFF_KFOLD = 2


def stratified_subsample(X, y, cap, seed=5):
    if cap is None or len(X) <= cap:
        return X, y
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        take = max(1, int(round(len(ci) * cap / len(X))))
        idx.append(rng.choice(ci, min(take, len(ci)), replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]


def onehot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


def rf_soft_labels(X, y, n_classes):
    """RandomForest teacher, 2-fold cross-fit -> soft labels."""
    sl = np.zeros((len(X), n_classes))
    kf = KFold(n_splits=TFF_KFOLD, shuffle=True, random_state=5)
    for tr, te in kf.split(X):
        clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                     max_features='sqrt', n_jobs=-1, random_state=5)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        full = np.zeros((len(te), n_classes))
        for j, c in enumerate(clf.classes_):
            full[:, int(c)] = proba[:, j]
        sl[te] = full
        print(f"    RF fold done (train {len(tr)} -> soft for {len(te)})", flush=True)
    return sl


def train_distilled(name, teacher='rf'):
    print(f"\n{'='*60}\n[{name}|rf] loading ...", flush=True)
    Xtr, Ytr, Xte, Yte, feats, classes = load_dataset(name)
    n_classes = len(classes)
    Xtr = np.asarray(Xtr, dtype=np.float32); Ytr = np.asarray(Ytr).astype(int)
    Xte = np.asarray(Xte, dtype=np.float32); Yte = np.asarray(Yte).astype(int)
    Xs, Ys = stratified_subsample(Xtr, Ytr, TRAIN_CAP)
    print(f"[{name}|rf] train {Xs.shape} {Counter(Ys)} | test {Xte.shape}", flush=True)

    t = time.time()
    soft = rf_soft_labels(Xs, Ys, n_classes)
    print(f"[{name}|rf] soft labels ready in {time.time()-t:.1f}s", flush=True)

    # Soft-label distillation: y_i = 0.75*hard + 0.25*RF_soft, replicated per class
    ALPHA = 0.75
    hybrid = ALPHA * onehot(Ys, n_classes) + (1 - ALPHA) * np.clip(soft, 0, 1)
    n = len(Xs)
    rep_X = np.repeat(Xs, n_classes, axis=0)
    rep_y = np.tile(np.arange(n_classes), n)
    rep_w = hybrid.reshape(-1)
    keep = rep_w > 1e-9
    rep_X, rep_y, rep_w = rep_X[keep], rep_y[keep], rep_w[keep]
    print(f"[{name}|rf] replicated rows = {len(rep_X)}", flush=True)
    t = time.time()
    model = DTC(max_depth=DEPTH, random_state=5, min_samples_leaf=3)
    model.fit(rep_X, rep_y, sample_weight=rep_w)
    print(f"[{name}|rf] DT trained in {time.time()-t:.1f}s; n_nodes={model.tree_.node_count}", flush=True)

    # Replace soft-label leaf values with true-label counts so est_acc reflects
    # real training accuracy (tree structure unchanged, only leaf statistics updated).
    leaf_ids = model.apply(Xs)
    tree = model.tree_
    leaf_mask = (tree.children_left == -1)
    for c in range(n_classes):
        counts_c = np.bincount(leaf_ids[Ys == c], minlength=tree.node_count)
        tree.value[leaf_mask, 0, c] = counts_c[leaf_mask]

    jm = sklearn2json(model, feats, classes)
    jm = hard_prune(jm, limit_depth=10**9)
    jm = soft_prune(copy.deepcopy(jm))
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    out = os.path.join(OUT, name, f"{name}_edt_rf.json")
    with open(out, 'w') as f:
        json.dump(jm, f)
    print(f"[{name}|rf] saved -> {out} ({os.path.getsize(out)//1024} KB)", flush=True)

    pred = model.predict(Xte)
    acc = accuracy_score(Yte, pred)
    f1m = f1_score(Yte, pred, average='macro', zero_division=0)
    pm = precision_score(Yte, pred, average='macro', zero_division=0)
    rm = recall_score(Yte, pred, average='macro', zero_division=0)
    print(f"[{name}|rf] TEST: Acc={acc:.4f} | macro-F1={f1m:.4f} | "
          f"macro-Prec={pm:.4f} | macro-Rec={rm:.4f}", flush=True)
    print(classification_report(Yte, pred, zero_division=0), flush=True)
    del Xtr, Ytr, Xte, Yte, Xs, Ys, soft, model, jm; gc.collect()
    return acc, f1m, pm, rm


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    dsets = ['univ1', 'iscx', 'bot_iot'] if which == 'all' else [which]
    res = {}
    for ds in dsets:
        try:
            res[ds] = train_distilled(ds)
        except Exception:
            import traceback; traceback.print_exc(); res[ds] = None
    print("\n==== DISTILL SUMMARY ====")
    for ds in dsets:
        if res[ds]:
            a, f1, p, r = res[ds]
            print(f"{ds:8s} Acc={a:.4f} F1(macro)={f1:.4f} P={p:.4f} R={r:.4f}")
        else:
            print(f"{ds:8s} FAILED")
