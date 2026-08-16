# -*- coding:utf-8 -*-
"""Hard/soft pruning on a (distilled) EDT JSON at multiple depths.

Loads an EDT JSON (e.g., json_models/univ1/univ1_edt_cnn.json), applies
hard_prune + soft_prune at depths 4/8/12, and reports the tree structure
(depth, nodes, leaves, rules) + estimated training-set accuracy
(eq_acc_est = sum(max(leaf.values)) / sum(sum(leaf.values))).

Usage:
  python3 prune_edt.py --tree ./json_models/univ1/univ1_edt_cnn.json --depths 4,8,12
"""
import os, sys, json, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_edt import hard_prune, soft_prune


def tree_stats(jm):
    """Return (max_depth, n_nodes, n_leaves, n_rules, est_accuracy)."""
    n_nodes = n_leaves = max_d = 0
    leaf_vals = []
    stack = [(jm, 0)]
    while stack:
        n, d = stack.pop()
        n_nodes += 1
        max_d = max(max_d, d)
        if "children" not in n:
            n_leaves += 1
            leaf_vals.append(n["value"])
        else:
            for c in n["children"]:
                stack.append((c, d + 1))
    n_rules = n_leaves + (n_nodes - n_leaves) * 2
    total = sum(sum(v) for v in leaf_vals)
    correct = sum(max(v) for v in leaf_vals)
    est_acc = correct / total if total > 0 else 0.0
    return max_d, n_nodes, n_leaves, n_rules, est_acc


def main():
    ap = argparse.ArgumentParser(description="Hard/soft prune an EDT JSON at multiple depths")
    ap.add_argument("--tree", default="./json_models/univ1/univ1_edt_cnn.json",
                    help="path to the (distilled) EDT JSON")
    ap.add_argument("--depths", default="4,8,12", help="comma-separated pruning depths")
    args = ap.parse_args()

    with open(args.tree) as f:
        jm = json.load(f)
    print(f"Loaded EDT: {args.tree}")

    # unpruned (full tree)
    d, nn, nl, nr, acc = tree_stats(jm)
    print(f"\n{'prune_depth':>10} {'max_depth':>10} {'nodes':>8} {'leaves':>8} {'rules':>8} {'est_acc':>10}")
    print(f"{'unpruned':>10} {d:10d} {nn:8d} {nl:8d} {nr:8d} {acc:10.4f}")

    # hard + soft prune at each depth
    for lim in [int(x) for x in args.depths.split(",")]:
        pruned = hard_prune(copy.deepcopy(jm), limit_depth=lim)
        pruned = soft_prune(pruned)
        d, nn, nl, nr, acc = tree_stats(pruned)
        print(f"{lim:10d} {d:10d} {nn:8d} {nl:8d} {nr:8d} {acc:10.4f}")
    print()


if __name__ == "__main__":
    main()
