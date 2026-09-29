import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

import features as F

PAIRS = {
    "control (unreachable vs joint_limit)": ("unreachable", "joint_limit"),
    "limit (joint_limit vs limit_blocked)": ("joint_limit", "limit_blocked"),
    "colliding (blocked vs local_optimum)": ("blocked", "local_optimum"),
    "partial (fixable vs unfixable)": ("fixable", "unfixable"),
}
FEATURES = [f for f in F.FEATURE_NAMES if f != "round"]   # round is constant within a fit
REPEATS = 10          # random re-splits for the default mode (--repeats to change)
HOLDOUT = 0.25        # fraction of goals held out in each re-split
TARGET = 0.80         # "history can tell them apart" threshold for the crossing round
COLOURS = ["#6da7ec", "#2a78d6", "#104281"]    # one hue, light -> dark = small -> large spread
MARKERS = ["o", "s", "^"]


def spread_of(path):
    m = re.search(r"s(\d{3})", path)
    return int(m.group(1)) / 100 if m else path


def _fit_score(train, test, seed):
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    clf.fit(train[FEATURES], train["type"])
    return float((clf.predict(test[FEATURES]) == test["type"]).mean())


def _resplit(goals_by_type, rng):
    held = set()
    for ids in goals_by_type.values():
        ids = rng.permutation(ids)
        held.update(ids[:max(1, round(HOLDOUT * len(ids)))])
    return held


def curves(df, use_test):
    rounds = sorted(df["round"].unique())
    out = []
    for pair, types in PAIRS.items():
        d = df[df["type"].isin(types)]
        dev = d[d["split"].isin(["train", "val"])]
        if use_test:
            for k in rounds:
                acc = _fit_score(dev[dev["round"] == k], d[(d["split"] == "test") & (d["round"] == k)], seed=0)
                out.append(dict(pair=pair, round=k, mean=acc, lo=acc, hi=acc,
                                n_test=int(((d["split"] == "test") & (d["round"] == k)).sum())))
            continue
        goals_by_type = {t: dev.loc[dev["type"] == t, "goal_id"].unique() for t in types}
        rng = np.random.default_rng(0)
        splits = [_resplit(goals_by_type, rng) for _ in range(REPEATS)]   
        for k in rounds:
            dk = dev[dev["round"] == k]
            accs = Parallel(n_jobs=-1)(            
                delayed(_fit_score)(dk[~dk["goal_id"].isin(h)], dk[dk["goal_id"].isin(h)], r)
                for r, h in enumerate(splits))
            out.append(dict(pair=pair, round=k, mean=float(np.mean(accs)),
                            lo=float(np.percentile(accs, 10)), hi=float(np.percentile(accs, 90)),
                            n_test=int(dk["goal_id"].isin(splits[0]).sum())))
    return pd.DataFrame(out)


def crossing(c):
    above = c.loc[c["mean"] >= TARGET, "round"]
    return int(above.min()) if len(above) else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--test", action="store_true", help="score once on the frozen test split")
    ap.add_argument("--repeats", type=int, default=REPEATS, help="re-splits per point (more = smoother, slower)")
    a = ap.parse_args()
    REPEATS = a.repeats

    results = []
    for path in a.files:
        print(f"working on {path} ...", flush=True)
        c = curves(pd.read_csv(path), a.test)
        c["spread"] = spread_of(path)
        results.append(c)
    res = pd.concat(results)
    tag = "_test" if a.test else ""
    res.to_csv(f"pair_curves{tag}.csv", index=False)

    show = [1, 3, 5, 10, 15, 20]
    what = "frozen test split" if a.test else f"mean over {REPEATS} re-splits of train+val"
    print(f"accuracy telling the pair apart ({what}); chance = 0.50\n")
    print(f"{'pair':38s} {'spread':>6s} | " + " ".join(f"r{k:<4d}" for k in show) + f" | first round >= {TARGET:.0%}")
    for pair in PAIRS:
        for spread, c in res[res["pair"] == pair].groupby("spread"):
            c = c.set_index("round")
            vals = " ".join(f"{c.loc[k, 'mean']:.2f} " if k in c.index else "  -   " for k in show)
            x = crossing(c.reset_index())
            print(f"{pair:38s} {spread:6.2f} | {vals}| {x if x else 'never'}")
        print()
    n = int(res["n_test"].iloc[0])
    print(f"about {n} goals scored per point; one point is uncertain by roughly "
          f"+-{100 * 1.96 * np.sqrt(0.25 / max(n, 1)):.0f} percentage points near 50%")

    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 4.5), sharey=True)
    for ax, pair in zip(axes, PAIRS):
        for i, (spread, c) in enumerate(res[res["pair"] == pair].groupby("spread")):
            ax.plot(c["round"], c["mean"], color=COLOURS[i % 3], marker=MARKERS[i % 3],
                    markersize=5, linewidth=2, label=f"spread {spread:.2f}")
            if not a.test:
                ax.fill_between(c["round"], c["lo"], c["hi"], color=COLOURS[i % 3], alpha=0.15, linewidth=0)
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
        ax.axhline(TARGET, color="#bbbbbb", linestyle=":", linewidth=1)
        ax.text(20, 0.505, "chance", color="#666666", ha="right", va="bottom", fontsize=9)
        ax.text(20, TARGET + 0.005, f"{TARGET:.0%}", color="#666666", ha="right", va="bottom", fontsize=9)
        ax.set_title(pair, fontsize=11)
        ax.set_xlabel("round (history seen so far)")
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0.3, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee")
    axes[0].set_ylabel("accuracy telling the pair apart")
    axes[0].legend(frameon=False, loc="lower right")
    band = "" if a.test else "  (shaded: 10th-90th percentile over re-splits)"
    fig.suptitle(f"Can history tell each pair apart? — {what}{band}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"pair_curves{tag}.png", dpi=150)
    print(f"\nwrote pair_curves{tag}.png and pair_curves{tag}.csv")