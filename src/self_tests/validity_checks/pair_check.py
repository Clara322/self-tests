import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

PAIRS = {"control": ("unreachable", "joint_limit"),
         "limit": ("joint_limit", "limit_blocked"),
         "colliding": ("blocked", "local_optimum"),
         "partial": ("fixable", "unfixable")}
GROUPS = {
    "all features": None,
    "distance only": ["best_dist", "mean_dist_recent", "dist_drop_recent",
                      "dist_drop_total", "cost_drop_recent"],
    "search shape only": ["spread", "std_j1", "std_j2", "std_j3", "spread_drop",
                          "tip_scatter_recent"],
}
ROUNDS = (1, 3, 5, 10, 20)
REPEATS = 10
HOLDOUT = 0.25
NOT_FEATURES = {"goal_id", "type", "split", "round"}


def fit_score(train, test, cols, seed):
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    clf.fit(train[cols], train["type"])
    return float((clf.predict(test[cols]) == test["type"]).mean())


def resplit(goals_by_type, rng):
    held = set()
    for ids in goals_by_type.values():
        ids = rng.permutation(ids)
        held.update(ids[:max(1, round(HOLDOUT * len(ids)))])
    return held


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--pair", default="silent", choices=PAIRS)
    a = ap.parse_args()

    types = PAIRS[a.pair]
    df = pd.read_csv(a.file)
    dev = df[df["type"].isin(types) & df["split"].isin(["train", "val"])]
    all_cols = [c for c in df.columns if c not in NOT_FEATURES]
    goals_by_type = {t: dev.loc[dev["type"] == t, "goal_id"].unique() for t in types}
    rng = np.random.default_rng(0)
    splits = [resplit(goals_by_type, rng) for _ in range(REPEATS)]

    print(f"{a.pair} pair {types}, {a.file}: accuracy, chance = 0.50\n")
    print(f"{'features':20s} | " + " ".join(f"r{k:<5d}" for k in ROUNDS))
    for name, cols in GROUPS.items():
        cols = cols or all_cols
        row = []
        for k in ROUNDS:
            dk = dev[dev["round"] == k]
            accs = Parallel(n_jobs=-1)(
                delayed(fit_score)(dk[~dk["goal_id"].isin(h)], dk[dk["goal_id"].isin(h)], cols, r)
                for r, h in enumerate(splits))
            row.append(np.mean(accs))
        print(f"{name:20s} | " + " ".join(f"{v:.2f}  " for v in row))
    print("(treat differences under ~0.07 as noise)")

    print("\nmedian feature value per type (10th-90th percentile in brackets)")
    show = ["best_dist", "dist_drop_total", "std_j2", "spread", "tip_scatter_recent"]
    for k in (1, 10):
        dk = dev[dev["round"] == k]
        print(f"\n  round {k}")
        print(f"  {'feature':20s} " + " ".join(f"{t:>24s}" for t in types))
        for f in show:
            cells = []
            for t in types:
                v = dk.loc[dk["type"] == t, f]
                cells.append(f"{v.median():.3f} ({v.quantile(.1):.3f}-{v.quantile(.9):.3f})")
            print(f"  {f:20s} " + " ".join(f"{c:>24s}" for c in cells))