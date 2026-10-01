import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from self_tests import arm
import generator as G
import features as F

CHECKPOINT = 10
GO_LOOK_STD = 1.0
PAIRS = {
    "control (unreachable vs joint_limit)": ("unreachable", "joint_limit"),
    "limit (joint_limit vs limit_blocked)": ("joint_limit", "limit_blocked"),
    "colliding (blocked vs local_optimum)": ("blocked", "local_optimum"),
    "partial (fixable vs unfixable)": ("fixable", "unfixable"),
}
REPEATS = 10
HOLDOUT = 0.25


DIRECTION = ["align_j1_recent", "align_j2_recent", "align_j3_recent", "aniso_recent"]


def geometry(target):
    R = float(np.linalg.norm(target))
    aim = target if R < G.REACH else target * (0.999 * G.REACH / R)
    lo, hi = G.lever_range(aim)
    return {"geo_dist": R, "geo_reachable": float(R < G.REACH),
            "geo_lever_lo": lo, "geo_lever_hi": hi}


def repeat_probe(cfg, target, mean, rng, n):
    tips, hit = arm.execute(np.repeat(mean[None], n, axis=0), cfg, rng)
    dev = tips - tips.mean(0)
    scatter = float(np.linalg.norm(dev, axis=-1).mean())
    out = {"rep_scatter": scatter,
           "rep_success": float(arm.success(tips, hit, target).mean()),
           "rep_hit": float(hit.mean())}
    if scatter < 1e-9:                     
        out.update(rep_align_j1=0.0, rep_align_j2=0.0, rep_align_j3=0.0, rep_anisotropy=0.0)
        return out
    w, v = np.linalg.eigh(np.cov(dev.T))
    axis = v[:, -1]                       
    for j, d in enumerate(F.swing_directions(mean)):
        out[f"rep_align_j{j + 1}"] = float(abs(axis @ d))
    out["rep_anisotropy"] = float(1 - w[0] / max(w[-1], 1e-12))
    return out


def posture_probe(cfg, target, rng, n, best_so_far):
    R = float(np.linalg.norm(target))
    aim = target if R < G.REACH else target * (0.999 * G.REACH / R)
    lo, hi = G.lever_range(aim)
    d = rng.uniform(lo, hi, n)
    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    cand = np.array([arm.pose_reaching(aim, di, si) for di, si in zip(d, sign)])
    tips, hit = arm.execute(cand, cfg, rng)
    dist = np.linalg.norm(tips - target, axis=-1)
    ok = arm.success(tips, hit, target)
    clear = dist[~hit]
    best_clear = float(clear.min()) if len(clear) else 3.0
    folded = d <= np.median(d)
    return {"post_best_clear": best_clear,
            "post_gain_clear": best_so_far - best_clear,
            "post_hit": float(hit.mean()),
            "post_success": float(ok.mean()),
            "post_success_folded": float(ok[folded].mean()),
            "post_success_open": float(ok[~folded].mean())}


def go_look_probe(cfg, target, mean, rng, n, best_so_far):
    cand = mean + GO_LOOK_STD * rng.standard_normal((n, 3))
    tips, hit = arm.execute(cand, cfg, rng)
    dist = np.linalg.norm(tips - target, axis=-1)
    clear = dist[~hit]
    best_clear = float(clear.min()) if len(clear) else 3.0
    return {"look_best_clear": best_clear,
            "look_gain_clear": best_so_far - best_clear,
            "look_best_any": float(dist.min()),
            "look_hit": float(hit.mean()),
            "look_success": float(arm.success(tips, hit, target).mean())}


def build(goals_path, splits_path, n, seed=0):
    _, records = G.load_goals(goals_path)
    splits = G.load_splits(splits_path)
    where = {gid: s for s, ids in splits["splits"].items() for gid in ids}
    rows = []
    for i, (gtype, (target, cfg, m0, s0), d) in enumerate(records):
        rng = np.random.default_rng(seed + 7919 * i)
        history = F.practise(cfg, target, m0, s0, rng, rounds=CHECKPOINT)
        mean = history[-1]["mean"]
        row = {"goal_id": d["id"], "type": gtype, "split": where[d["id"]]}
        row.update({f"h1_{k}": v for k, v in F.features_at(history, 1).items() if k != "round"})
        row.update({f"h10_{k}": v for k, v in F.features_at(history, CHECKPOINT).items() if k != "round"})
        row.update(geometry(target))
        prng = np.random.default_rng(10**7 + i)
        row.update(repeat_probe(cfg, target, mean, prng, n))
        best_so_far = min(h["best_dist"] for h in history)
        row.update(go_look_probe(cfg, target, mean, prng, n, best_so_far))
        row.update(posture_probe(cfg, target, prng, n, best_so_far))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_score(train, test, cols, seed):
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    clf.fit(train[cols], train["type"])
    return float((clf.predict(test[cols]) == test["type"]).mean())


def evaluate_test(df, sets):
    out = []
    for pair, types in PAIRS.items():
        d = df[df["type"].isin(types)]
        dev, test = d[d["split"].isin(["train", "val"])], d[d["split"] == "test"]
        for name, cols in sets.items():
            acc = fit_score(dev, test, cols, seed=0)
            n = len(test)
            out.append(dict(pair=pair, features=name, accuracy=acc, n_test=n,
                            ci=1.96 * np.sqrt(max(acc * (1 - acc), 0.25 / n) / n)))
    return pd.DataFrame(out)


def feature_sets(df):
    direction = {f"h1_{c}" for c in DIRECTION} | {f"h10_{c}" for c in DIRECTION}
    h1 = [c for c in df.columns if c.startswith("h1_") and c not in direction]
    h10 = [c for c in df.columns if c.startswith("h10_") and c not in direction]
    h10_dir = [f"h10_{c}" for c in DIRECTION]
    geo = [c for c in df.columns if c.startswith("geo_")]
    rep = [c for c in df.columns if c.startswith("rep_")]
    look = [c for c in df.columns if c.startswith("look_")]
    post = [c for c in df.columns if c.startswith("post_")]
    return {"hist r1": h1, "hist r10": h10,
            "r10+dir": h10 + h10_dir, "r10+geo": h10 + geo, "r10+dir+geo": h10 + h10_dir + geo,
            "repeat": rep, "broad": look, "posture": post}


def evaluate(df):
    sets = feature_sets(df)
    out = []
    for pair, types in PAIRS.items():
        dev = df[df["type"].isin(types) & df["split"].isin(["train", "val"])]
        rng = np.random.default_rng(0)
        held = []
        for _ in range(REPEATS):
            h = set()
            for t in types:
                ids = rng.permutation(dev.loc[dev["type"] == t, "goal_id"].values)
                h.update(ids[:max(1, round(HOLDOUT * len(ids)))])
            held.append(h)
        for name, cols in sets.items():
            accs = Parallel(n_jobs=-1)(
                delayed(fit_score)(dev[~dev["goal_id"].isin(h)], dev[dev["goal_id"].isin(h)], cols, r)
                for r, h in enumerate(held))
            out.append(dict(pair=pair, features=name, accuracy=float(np.mean(accs)),
                            lo=float(np.percentile(accs, 10)), hi=float(np.percentile(accs, 90))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("goals")
    ap.add_argument("splits")
    ap.add_argument("--n", type=int, default=16, help="attempts each probe may spend")
    ap.add_argument("--test", action="store_true",
                    help="final score on the frozen test split -- run this ONCE")
    a = ap.parse_args()

    print(f"practising to round {CHECKPOINT} and probing ({a.n} attempts per probe) ...", flush=True)
    df = build(a.goals, a.splits, a.n)
    df.to_csv("probe_features.csv", index=False)
    res = evaluate_test(df, feature_sets(df)) if a.test else evaluate(df)
    res.to_csv("probe_results_test.csv" if a.test else "probe_results.csv", index=False)

    names = list(dict.fromkeys(res["features"]))
    if a.test:
        n = int(res["n_test"].iloc[0])
        print(f"\nFROZEN TEST SPLIT: fit on train+val, scored once on {n} test goals per pair; chance 0.50")
        print(f"(one number per cell; uncertain by about +-{100 * res['ci'].max():.0f} points at worst)\n")
    else:
        print(f"\naccuracy telling each pair apart (mean over {REPEATS} re-splits of train+val); chance 0.50\n")
    print(f"{'pair':38s}" + "".join(f"{n:>13s}" for n in names))
    for pair in PAIRS:
        r = res[res["pair"] == pair].set_index("features")
        print(f"{pair:38s}" + "".join(f"{r.loc[n, 'accuracy']:13.2f}" for n in names))
    print("\ncolumns: history after 1 and 10 rounds (as before) | r10 plus scatter direction, plus geometry"
          " (what the posture probe knows), plus both | each probe alone (16 attempts)")

    show = ["rep_scatter", "rep_align_j1", "rep_align_j2", "look_best_clear",
            "post_best_clear", "post_success", "post_success_folded"]
    print(f"\nmedian probe readings per type\n{'type':15s}" + "".join(f"{s:>16s}" for s in show))
    for t in G.RULES:
        d = df[df["type"] == t]
        if len(d):
            print(f"{t:15s}" + "".join(f"{d[s].median():16.3f}" for s in show))