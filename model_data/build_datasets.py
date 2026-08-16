# -*- coding:utf-8 -*-
"""Build per-dataset folders (univ1/, iscx/, bot_iot/) from the raw source
files stored in model_data/raw_data/. Each folder gets full.pkl / train.pkl /
test.pkl in the {"X","Y"} dict format.

  UNIV1   : raw_data/x_train.pkl, y_train.pkl, x_test.pkl, y_test.pkl (lists)
  ISCX    : raw_data/ids_k4_max_min_scale.pkl        (dict{train,valid,test: (X,Y)})
  Bot-IoT : raw_data/bot_iot_k4_max_min_scale.pkl    (dict{train,valid,test: (X,Y)})

full.pkl = train + test concatenated.
"""
import os
import sys
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_data")

SOURCES = ["x_train.pkl", "y_train.pkl", "x_test.pkl", "y_test.pkl",
           "ids_k4_max_min_scale.pkl", "bot_iot_k4_max_min_scale.pkl"]


def save(path, X, Y):
    with open(path, "wb") as f:
        pickle.dump({"X": np.asarray(X), "Y": np.asarray(Y)}, f, protocol=4)
    print(f"  saved {os.path.relpath(path, HERE)} "
          f"X={np.asarray(X).shape} Y={np.asarray(Y).shape}")


def build_univ1():
    out = os.path.join(HERE, "univ1"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(RAW, "x_train.pkl"), "rb") as f: xtr = pickle.load(f)
    with open(os.path.join(RAW, "y_train.pkl"), "rb") as f: ytr = pickle.load(f)
    with open(os.path.join(RAW, "x_test.pkl"), "rb") as f: xte = pickle.load(f)
    with open(os.path.join(RAW, "y_test.pkl"), "rb") as f: yte = pickle.load(f)
    Xtr, Ytr = np.asarray(xtr), np.asarray(ytr)
    Xte, Yte = np.asarray(xte), np.asarray(yte)
    save(os.path.join(out, "train.pkl"), Xtr, Ytr)
    save(os.path.join(out, "test.pkl"), Xte, Yte)
    save(os.path.join(out, "full.pkl"),
         np.concatenate([Xtr, Xte]), np.concatenate([Ytr, Yte]))


def build_dict_dataset(src_name, folder_name):
    with open(os.path.join(RAW, src_name), "rb") as f: d = pickle.load(f)
    Xtr, Ytr = np.asarray(d["train"][0]), np.asarray(d["train"][1])
    Xte, Yte = np.asarray(d["test"][0]), np.asarray(d["test"][1])
    out = os.path.join(HERE, folder_name); os.makedirs(out, exist_ok=True)
    save(os.path.join(out, "train.pkl"), Xtr, Ytr)
    save(os.path.join(out, "test.pkl"), Xte, Yte)
    save(os.path.join(out, "full.pkl"),
         np.concatenate([Xtr, Xte]), np.concatenate([Ytr, Yte]))
    del d


if __name__ == "__main__":
    missing = [s for s in SOURCES if not os.path.exists(os.path.join(RAW, s))]
    if missing:
        sys.exit(f"Missing raw sources in raw_data/: {missing}. "
                 "Populate raw_data/ first (one-time: git checkout + git lfs pull + "
                 "mv sources raw_data/).")
    print("Building UNIV1 ..."); build_univ1()
    print("Building ISCX ..."); build_dict_dataset("ids_k4_max_min_scale.pkl", "iscx")
    print("Building Bot-IoT ..."); build_dict_dataset("bot_iot_k4_max_min_scale.pkl", "bot_iot")
    print("Done.")
