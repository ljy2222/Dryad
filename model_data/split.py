# -*- coding:utf-8 -*-
"""How to split a full-dataset pkl into a train pkl and a test pkl.

Each per-dataset folder under model_data/ (univ1/, iscx/, bot_iot/) holds:
  - full.pkl : the complete dataset as {"X": np.ndarray, "Y": np.ndarray}
               (here "complete" = the union of the dataset's train and test
               samples, i.e. train + test concatenated).
  - train.pkl: {"X", "Y"} training split (the dataset's original train split).
  - test.pkl : {"X", "Y"} test split (the dataset's original test split).

This script documents and reproduces the standard stratified 80/20 split that
turns full.pkl into a (new) train.pkl / test.pkl pair. The train.pkl / test.pkl
shipped in each folder are the dataset's *original* splits; running this script
on full.pkl regenerates an equivalent stratified 80/20 split (same sizes,
different sample assignment) with a fixed seed for reproducibility.

Usage:
  python split.py <dataset_folder>          # split <folder>/full.pkl -> train/test
  python split.py univ1 --test_size 0.2     # relative to model_data/
"""
import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["X"]), np.asarray(d["Y"])


def save_pkl(path, X, Y):
    with open(path, "wb") as f:
        pickle.dump({"X": np.asarray(X), "Y": np.asarray(Y)}, f, protocol=4)


def split_full(folder, test_size=0.2, random_state=5):
    """Split <folder>/full.pkl into <folder>/train.pkl + test.pkl.

    Stratified when every class has >=2 samples; otherwise falls back to a
    plain random split (a class with a single sample cannot be stratified)."""
    full = os.path.join(folder, "full.pkl")
    X, Y = load_pkl(full)
    counts = np.bincount(Y.astype(int))
    stratify = Y if counts.min() >= 2 else None
    Xtr, Xte, Ytr, Yte = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=stratify)
    save_pkl(os.path.join(folder, "train.pkl"), Xtr, Ytr)
    save_pkl(os.path.join(folder, "test.pkl"), Xte, Yte)
    name = os.path.basename(os.path.normpath(folder))
    n_classes = int(Y.max()) + 1 if len(Y) else 0
    print(f"[stats] {name:8s} | features={X.shape[1]:2d} | classes={n_classes:2d} "
          f"| full={X.shape[0]:8d} | train={Xtr.shape[0]:8d} | test={Xte.shape[0]:8d} "
          f"(split: test_size={test_size}, seed={random_state}, "
          f"{'stratified' if stratify is not None else 'random (rare class)'})")


if __name__ == "__main__":
    ds = sys.argv[1]
    folder = ds if os.path.isdir(ds) else os.path.join(HERE, ds)
    ts = 0.2
    if "--test_size" in sys.argv:
        ts = float(sys.argv[sys.argv.index("--test_size") + 1])
    split_full(folder, test_size=ts)
