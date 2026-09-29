import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

ORDER = ["slow", "unreachable", "joint_limit", "limit_blocked",
         "blocked", "local_optimum", "fixable", "unfixable"]
SHORT = {"slow": "slow", "unreachable": "unreach", "joint_limit": "j_limit",
         "limit_blocked": "l_block", "blocked": "blocked", "local_optimum": "loc_opt",
         "fixable": "fixable", "unfixable": "unfix"}
PAIR = {"joint_limit": "limit_blocked", "limit_blocked": "joint_limit",
        "blocked": "local_optimum", "local_optimum": "blocked",
        "fixable": "unfixable", "unfixable": "fixable"}
NOT_FEATURES = {"goal_id", "type", "split", "round"}


def out_of_fold(dk, cols, seed=0):
    pred = pd.Series(index=dk.index, dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(dk, dk["type"], groups=dk["goal_id"]):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        clf.fit(dk.iloc[tr][cols], dk.iloc[tr]["type"])
        pred.iloc[te] = clf.predict(dk.iloc[te][cols])
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 10, 20])
    a = ap.parse_args()

    df = pd.read_csv(a.file)
    df = df[df["split"].isin(["train", "val"])]
    types = [t for t in ORDER if t in set(df["type"])]
    cols = [c for c in df.columns if c not in NOT_FEATURES]

    for k in a.rounds:
        dk = df[df["round"] == k].reset_index(drop=True)
        pred = out_of_fold(dk, cols)
        m = pd.crosstab(dk["type"], pred, normalize="index").reindex(index=types, columns=types, fill_value=0)
        acc = float((pred == dk["type"]).mean())
        within = float(np.mean([p == PAIR.get(t) for t, p in zip(dk["type"], pred) if p != t] or [0]))

        print(f"\n=== round {k}   overall accuracy {acc:.0%}   "
              f"(chance {1/len(types):.0%}; {within:.0%} of errors land on the pair-mate) ===")
        print(f"{'true / predicted':16s}" + "".join(f"{SHORT[t]:>9s}" for t in types))
        for t in types:
            cells = []
            for p in types:
                v = m.loc[t, p]
                s = f"{100 * v:.0f}" if v >= 0.005 else "."
                if p == t:
                    s = f"[{s}]"
                elif p == PAIR.get(t):
                    s = f"*{s}"
                cells.append(f"{s:>9s}")
            print(f"{SHORT[t]:16s}" + "".join(cells))
    print("\n[ ] = correct   * = mistaken for its pair-mate   . = under 0.5%   "
          f"n = {len(dk)} goals (train+val), each predicted once, out of fold")