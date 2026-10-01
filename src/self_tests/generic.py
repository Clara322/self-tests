import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from self_tests import arm
import generator as G
import features as F

CHECKPOINT = 10
RECENT = 3
GO_LOOK_STD = 1.0
PAIRS = {
    "control (unreachable vs joint_limit)": ("unreachable", "joint_limit"),
    "limit (joint_limit vs limit_blocked)": ("joint_limit", "limit_blocked"),
    "colliding (blocked vs local_optimum)": ("blocked", "local_optimum"),
    "partial (fixable vs unfixable)": ("fixable", "unfixable"),
}
REPEATS = 10
HOLDOUT = 0.25


def _axis(cov):
    w, v = np.linalg.eigh(cov)
    return v[:, -1], float(np.sqrt(max(w.sum(), 0.0))), float(1 - w[0] / max(w[-1], 1e-12))


def describe(cmds, tips, hits, target):
    phi = np.arctan2(target[1], target[0])
    rot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])  
    t = tips @ rot.T
    tg = rot @ target
    dist = np.linalg.norm(tips - target, axis=-1)
    ok = arm.success(tips, hits, target)
    out = {"success": float(ok.mean()), "hit": float(hits.mean()),
           "dist_min": float(dist.min()), "dist_mean": float(dist.mean())}

    c = t.mean(0) - tg
    out.update(offset_x=float(c[0]), offset_y=float(c[1]))
    Y = t - t.mean(0)
    if np.linalg.norm(Y, axis=-1).mean() < 1e-12:
        out.update(spread=0.0, elong=0.0, along_target=0.0)
    else:
        ax, s, e = _axis(np.cov(Y.T))
        out.update(spread=s, elong=e, along_target=float(abs(ax[0])))

    X = cmds - cmds.mean(0)
    for j in range(3):
        out[f"cmd_spread_j{j+1}"] = float(X[:, j].std())

    if np.all(X.std(0) < 1e-9):                       
        out.update({f"sens_j{j+1}": 0.0 for j in range(3)})
        R = Y
        cols = None
    else:
        lam = 1e-6 * len(X)
        J = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ Y).T  
        for j in range(3):
            out[f"sens_j{j+1}"] = float(np.linalg.norm(J[:, j]))
        R = Y - X @ J.T
        cols = J
    if np.linalg.norm(R, axis=-1).mean() < 1e-12:
        out.update(resid_spread=0.0, resid_elong=0.0)
        out.update({f"resid_along_j{j+1}": 0.0 for j in range(3)})
    else:
        ax, s, e = _axis(np.cov(R.T))
        out.update(resid_spread=s, resid_elong=e)
        for j in range(3):
            if cols is None or np.linalg.norm(cols[:, j]) < 1e-12:
                out[f"resid_along_j{j+1}"] = 0.0
            else:
                out[f"resid_along_j{j+1}"] = float(abs(ax @ cols[:, j] / np.linalg.norm(cols[:, j])))
    return out


def practise_raw(cfg, target, mean, std, rng, rounds=CHECKPOINT):
    batches = []
    for _ in range(rounds):
        cand = mean + std * rng.standard_normal((F.BATCH, 3))
        tip, hit = arm.execute(cand, cfg, rng)
        cost = arm.cost(tip, hit, target, cfg)
        elite = cand[cost.argsort()[:F.N_ELITE]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), 0.01)
        batches.append((cand, tip, hit))
    return batches, mean


def history_summary(batches, target, k):
    first = describe(*batches[0], target)
    recent = [describe(*b, target) for b in batches[max(0, k - RECENT):k]]
    now = {key: float(np.mean([r[key] for r in recent])) for key in first}
    out = {f"now_{key}": v for key, v in now.items()}
    out.update({f"chg_{key}": now[key] - first[key] for key in first})
    return out


def repeat_attempts(cfg, mean, rng, n):
    cmds = np.repeat(mean[None], n, axis=0)
    tips, hit = arm.execute(cmds, cfg, rng)
    return cmds, tips, hit


def broad_attempts(cfg, mean, rng, n):
    cmds = mean + GO_LOOK_STD * rng.standard_normal((n, 3))
    tips, hit = arm.execute(cmds, cfg, rng)
    return cmds, tips, hit


def posture_attempts(cfg, target, rng, n):
    R = float(np.linalg.norm(target))
    aim = target if R < G.REACH else target * (0.999 * G.REACH / R)
    lo, hi = G.lever_range(aim)
    d = rng.uniform(lo, hi, n)
    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    cmds = np.array([arm.pose_reaching(aim, di, si) for di, si in zip(d, sign)])
    tips, hit = arm.execute(cmds, cfg, rng)
    return cmds, tips, hit


OFF_D = 0.25           
OFF_D_CHECK = 0.50     
OFF_D_EXPLORE = 0.10    


def offtarget_aims(target, D, rng, k, grid=720):
    target = np.asarray(target, float)
    theta = np.linspace(0, 2 * np.pi, grid, endpoint=False)
    ring = target + D * np.stack([np.cos(theta), np.sin(theta)], -1)
    r = np.linalg.norm(ring, axis=-1)
    reach = 0.999 * G.REACH
    if np.linalg.norm(target) < G.REACH:
        idx = np.flatnonzero(r <= reach)            
    else:
        idx = np.arange(grid)
        ring = ring * np.minimum(1.0, reach / r)[:, None]
    pos = ((rng.uniform() + np.arange(k) / k) % 1.0) * len(idx)
    return ring[idx[pos.astype(int)]]


def offtarget_attempts(cfg, target, rng, n, D):
    cmds = []
    for p in offtarget_aims(target, D, rng, n // 2):
        lo, hi = G.lever_range(p)
        for sign in (1.0, -1.0):
            cmds.append(arm.pose_reaching(p, rng.uniform(lo, hi), sign))
    cmds = np.array(cmds)
    tips, hit = arm.execute(cmds, cfg, rng)
    return cmds, tips, hit


def build(goals_path, splits_path, n, seed=0):
    _, records = G.load_goals(goals_path)
    splits = G.load_splits(splits_path)
    where = {gid: s for s, ids in splits["splits"].items() for gid in ids}
    rows = []
    for i, (gtype, (target, cfg, m0, s0), d) in enumerate(records):
        rng = np.random.default_rng(seed + 7919 * i)            
        batches, mean = practise_raw(cfg, target, m0, s0, rng)
        row = {"goal_id": d["id"], "type": gtype, "split": where[d["id"]]}
        row.update({f"h1_{k}": v for k, v in history_summary(batches, target, 1).items()})
        row.update({f"h10_{k}": v for k, v in history_summary(batches, target, CHECKPOINT).items()})
        prng = np.random.default_rng(10**7 + i)          
        row.update({f"rep_{k}": v for k, v in describe(*repeat_attempts(cfg, mean, prng, n), target).items()})
        row.update({f"broad_{k}": v for k, v in describe(*broad_attempts(cfg, mean, prng, n), target).items()})
        row.update({f"post_{k}": v for k, v in describe(*posture_attempts(cfg, target, prng, n), target).items()})
        for prefix, D, s in (("off_", OFF_D, 2 * 10**7), ("off50_", OFF_D_CHECK, 3 * 10**7),
                             ("off10_", OFF_D_EXPLORE, 4 * 10**7)):
            cmds, tips, hit = offtarget_attempts(cfg, target, np.random.default_rng(s + i), n, D)
            row.update({f"{prefix}{k}": v for k, v in describe(cmds, tips, hit, target).items()})
            row[f"chk_{prefix}aimed_min"] = float(np.linalg.norm(arm.fingertip(cmds) - target, axis=-1).min())
            row[f"chk_{prefix}landed_min"] = float(np.linalg.norm(tips - target, axis=-1).min())
        rows.append(row)
    return pd.DataFrame(rows)


def fit_score(train, test, cols, seed):
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1)
    clf.fit(train[cols], train["type"])
    return float((clf.predict(test[cols]) == test["type"]).mean())


def feature_sets(df):
    pick = lambda p: [c for c in df.columns if c.startswith(p)]
    return {"hist r1": pick("h1_"), "hist r10": pick("h10_"),
            "repeat": pick("rep_"), "broad": pick("broad_"), "posture": pick("post_"),
            "off-target": pick("off_"), "off-target 0.5": pick("off50_"),
            "off-target 0.1": pick("off10_")}


def evaluate(df, use_test):
    sets = feature_sets(df)
    out = []
    for pair, types in PAIRS.items():
        d = df[df["type"].isin(types)]
        dev = d[d["split"].isin(["train", "val"])]
        if use_test:
            test = d[d["split"] == "test"]
            for name, cols in sets.items():
                out.append(dict(pair=pair, features=name, accuracy=fit_score(dev, test, cols, 0)))
            continue
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
            out.append(dict(pair=pair, features=name, accuracy=float(np.mean(accs))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("goals")
    ap.add_argument("splits")
    ap.add_argument("--n", type=int, default=16, help="attempts each probe may spend")
    ap.add_argument("--test", action="store_true", help="score once on the frozen test split")
    a = ap.parse_args()

    print("practising, probing, and describing every batch with the same rule ...", flush=True)
    df = build(a.goals, a.splits, a.n)
    tag = a.goals.replace("goals_", "").replace(".json", "")
    df.to_csv(f"generic_features_{tag}.csv", index=False)
    res = evaluate(df, a.test)
    res.to_csv(f"generic_results_{tag}{'_test' if a.test else ''}.csv", index=False)

    where = "FROZEN TEST SPLIT, one score per cell (about +-13)" if a.test else f"mean over {REPEATS} re-splits of train+val"
    print(f"\naccuracy telling each pair apart, same description for history and probes ({where}); chance 0.50\n")
    names = list(dict.fromkeys(res["features"]))
    print(f"{'pair':38s}" + "".join(f"{n:>10s}" for n in names))
    for pair in PAIRS:
        r = res[res["pair"] == pair].set_index("features")
        print(f"{pair:38s}" + "".join(f"{r.loc[n, 'accuracy']:10.2f}" for n in names))