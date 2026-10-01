import csv
import sys
import numpy as np

from self_tests import arm
import generator as G

CHECKPOINTS = tuple(range(1, 21))
BATCH = 16
N_ELITE = 4
RECENT = 3


def swing_directions(q):
    pts = arm.joint_positions(q)
    tip = pts[-1]
    dirs = []
    for j in range(3):
        v = tip - pts[j]
        perp = np.array([-v[1], v[0]])
        n = np.linalg.norm(perp)
        dirs.append(perp / n if n > 1e-9 else np.zeros(2))
    return dirs


def scatter_direction(tips, q):
    dev = tips - tips.mean(0)
    if np.linalg.norm(dev, axis=-1).mean() < 1e-9:
        return np.zeros(3), 0.0
    w, v = np.linalg.eigh(np.cov(dev.T))
    axis = v[:, -1]
    align = np.array([abs(axis @ d) for d in swing_directions(q)])
    return align, float(1 - w[0] / max(w[-1], 1e-12))


def practise(cfg, target, mean, std, rng, rounds=max(CHECKPOINTS)):
    history = []
    for _ in range(rounds):
        cand = mean + std * rng.standard_normal((BATCH, 3))
        tip, hit = arm.execute(cand, cfg, rng)
        align, aniso = scatter_direction(tip, mean)
        cost = arm.cost(tip, hit, target, cfg)
        elite = cand[cost.argsort()[:N_ELITE]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), 0.01)
        dist = np.linalg.norm(tip - target, axis=-1)
        history.append({
            "success_rate": float(arm.success(tip, hit, target).mean()),
            "hit_rate": float(hit.mean()),
            "best_dist": float(dist.min()),
            "mean_dist": float(dist.mean()),
            "best_cost": float(cost.min()),
            "tip_scatter": float(np.linalg.norm(tip - tip.mean(0), axis=-1).mean()),
            "std": std.copy(),
            "align": align,
            "aniso": aniso,
            "mean": mean.copy(),
        })
    return history


def features_at(history, k):
    h = history[:k]
    recent = h[-RECENT:]
    first = h[0]
    last = h[-1]
    take = lambda key, rows: float(np.mean([r[key] for r in rows]))

    return {
        "round": k,
        # Is it succeeding?
        "success_recent": take("success_rate", recent),
        "success_all": take("success_rate", h),
        # How close is it, and is that still improving?
        "best_dist": last["best_dist"],
        "mean_dist_recent": take("mean_dist", recent),
        "dist_drop_recent": h[-min(RECENT + 1, len(h))]["best_dist"] - last["best_dist"],
        "dist_drop_total": first["best_dist"] - last["best_dist"],
        "cost_drop_recent": h[-min(RECENT + 1, len(h))]["best_cost"] - last["best_cost"],
        # Is it hitting things
        "hit_recent": take("hit_rate", recent),
        "hit_all": take("hit_rate", h),
        # How wide is the search now, and per joint
        "spread": float(last["std"].mean()),
        "std_j1": float(last["std"][0]),
        "std_j2": float(last["std"][1]),
        "std_j3": float(last["std"][2]),
        "spread_drop": float(first["std"].mean() - last["std"].mean()),
        # How inconsistent are the outcomes
        "tip_scatter_recent": take("tip_scatter", recent),
        # Which way do the outcomes scatter, compared with each joint's swing
        "align_j1_recent": float(np.mean([r["align"][0] for r in recent])),
        "align_j2_recent": float(np.mean([r["align"][1] for r in recent])),
        "align_j3_recent": float(np.mean([r["align"][2] for r in recent])),
        "aniso_recent": take("aniso", recent),
    }


FEATURE_NAMES = [k for k in features_at([{
    "success_rate": 0., "hit_rate": 0., "best_dist": 0., "mean_dist": 0.,
    "best_cost": 0., "tip_scatter": 0., "std": np.zeros(3),
    "align": np.zeros(3), "aniso": 0., "mean": np.zeros(3)}] * max(CHECKPOINTS),
    max(CHECKPOINTS))]


def build(goals_path, splits_path, out_path, seed=0):
    payload, records = G.load_goals(goals_path)
    splits = G.load_splits(splits_path)
    where = {gid: name for name, ids in splits["splits"].items() for gid in ids}

    rows = []
    for i, (gtype, (target, cfg, m0, s0), d) in enumerate(records):
        rng = np.random.default_rng(seed + 7919 * i)      # one run per goal
        history = practise(cfg, target, m0, s0, rng)
        for k in CHECKPOINTS:
            row = {"goal_id": d["id"], "type": gtype, "split": where[d["id"]]}
            row.update(features_at(history, k))
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["goal_id", "type", "split"] + FEATURE_NAMES)
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:] or ["goals_v1.json", "splits_v1.json", "features.csv"]
    rows = build(*a[:3])
    print(f"{len(rows)} rows ({len(rows)//len(CHECKPOINTS)} goals x {len(CHECKPOINTS)} checkpoints)"
          f" -> {a[2]}")
    print(f"{len(FEATURE_NAMES)} features: {', '.join(FEATURE_NAMES)}")

    import collections
    by = collections.defaultdict(list)
    for r in rows:
        if r["round"] == 10:
            by[r["type"]].append(r)
    show = ["success_recent", "best_dist", "dist_drop_recent", "hit_recent",
            "std_j2", "tip_scatter_recent"]
    print(f"\nmeans at round 10:\n{'type':15s}" + "".join(f"{s:>19s}" for s in show))
    for t, rs in by.items():
        print(f"{t:15s}" + "".join(f"{np.mean([r[s] for r in rs]):19.3f}" for s in show))