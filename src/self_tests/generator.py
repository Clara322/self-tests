import dataclasses
import json
import math
import numpy as np
from self_tests import arm
from self_tests.arm import ArmConfig, L, REACH, R_SUCCESS

NO_LIM = np.tile([-np.inf, np.inf], (3, 1))
CHECKPOINT = 10                       # where the plateau is first measured
HORIZON = 20                          # how long the symptom must last
SPREAD = 0.10                         # default start spread for every type except slow
SLOW_SPREAD = 0.03
PLATEAU_BAND = (0.25, 0.45)           # where the partial-pair goals must stall
OBSTACLE_RADIUS = (0.20, 0.35)        # shared by blocked and local_optimum
HELDOUT_LIMIT = (-1.30, -0.90)        # joint-2 interval
HELDOUT_JOINT = 2                     # noise on joint 3

TAG = "v3"                            # version tag in the output file names
LIMIT_MARGIN = 0.05                   # limit_blocked: gap (rad) between the allowed range
                                      # and the nearest pose that reaches the target

PAIRS = {
    "silent":    (("unreachable", "joint_limit", "limit_blocked"),
                  "stall",   (0.30, 0.50, 0.70, 0.90, 1.10)),
    "colliding": (("blocked", "local_optimum"),   "stall",   (0.00, 0.01, 0.10, 0.30, 0.60)),
    "partial":   (("fixable", "unfixable"),       "plateau", (0.25, 0.30, 0.35, 0.40, 0.45)),
}
PAIR_OF = {t: name for name, (types, _, _) in PAIRS.items() for t in types}


def _half_normal_quantile(p):
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def noise_for_plateau(p, lever):
    return R_SUCCESS / (_half_normal_quantile(p) * lever)


def lever_range(point):
    R = float(np.linalg.norm(point))
    return (max(abs(L[1] - L[2]), abs(R - L[0])), min(L[1] + L[2], R + L[0]))


def ceiling(cfg, target, n_d=60, n_eval=1500, seed=9):
    R = float(np.linalg.norm(target))
    if R > REACH:
        return 0.0
    lo, hi = lever_range(target)
    if lo > hi:
        return 0.0
    best = 0.0
    for sign in (1.0, -1.0):
        for d in np.linspace(lo + 1e-6, hi - 1e-6, n_d):
            q = arm.pose_reaching(target, d, sign)
            if np.linalg.norm(arm.fingertip(q) - target) > 1e-6:
                continue
            best = max(best, arm.competence(q, cfg, target, np.random.default_rng(seed), n=n_eval))
    return best


def _lim(lo, hi):
    lim = NO_LIM.copy(); lim[1] = [lo, hi]; return lim


def _reachable_target(rng, lo=0.8, hi=2.0):
    a = rng.uniform(-np.pi, np.pi); r = rng.uniform(lo, hi)
    return np.array([r * np.cos(a), r * np.sin(a)])


def _reaching_q2(target, n=200):
    lo, hi = lever_range(target)
    out = []
    for sign in (1.0, -1.0):
        for d in np.linspace(lo + 1e-6, hi - 1e-6, n):
            q = arm.pose_reaching(target, d, sign)
            if np.linalg.norm(arm.fingertip(q) - target) < 1e-6:
                out.append((q[1] + np.pi) % (2 * np.pi) - np.pi)
    return np.array(out)


def _elbow(rng):
    return 1.0 if rng.random() < 0.5 else -1.0


def sample_goal(rng, gtype, spread=SPREAD):
    s0 = np.full(3, spread)

    if gtype == "slow":
        t = _reachable_target(rng, 1.0, 1.8)
        d = rng.uniform(*lever_range(t))
        u = rng.standard_normal(3); u /= np.linalg.norm(u)
        return t, ArmConfig(), arm.pose_reaching(t, d, 1.0) + rng.uniform(0.45, 0.55) * u, \
               np.full(3, SLOW_SPREAD)

    if gtype == "unreachable":
        a = rng.uniform(-np.pi, np.pi); r = rng.uniform(2.45, 2.8)
        t = np.array([r * np.cos(a), r * np.sin(a)])
        near = t * (rng.uniform(1.0, 0.999 * REACH) / r)
        return t, ArmConfig(), arm.pose_reaching(near, rng.uniform(*lever_range(near)), _elbow(rng)), s0

    if gtype == "joint_limit":
        t = _reachable_target(rng, 1.0, 1.8)
        d = rng.uniform(*lever_range(t))
        qa, qb = arm.pose_reaching(t, d, 1.0), arm.pose_reaching(t, d, -1.0)
        lo, hi = sorted([qb[1] - rng.uniform(0.2, 0.6), qb[1] + rng.uniform(0.2, 0.6)])
        if lo <= qa[1] <= hi:                       
            return None
        return t, ArmConfig(limits=_lim(lo, hi)), qa, s0

    if gtype == "limit_blocked":
        t = _reachable_target(rng, 1.0, 1.8)
        d = rng.uniform(*lever_range(t))
        qa, qb = arm.pose_reaching(t, d, 1.0), arm.pose_reaching(t, d, -1.0)
        side = _elbow(rng)                                  
        near = qb[1] + side * rng.uniform(0.2, 0.6)          
        far = near + side * rng.uniform(0.4, 1.2)
        lo, hi = sorted([near, far])
        q2 = _reaching_q2(t)
        q2 = np.concatenate([q2 - 2 * np.pi, q2, q2 + 2 * np.pi])   
        if np.any((q2 > lo - LIMIT_MARGIN) & (q2 < hi + LIMIT_MARGIN)):
            return None                                     
        if lo >= HELDOUT_LIMIT[0] and hi <= HELDOUT_LIMIT[1]:
            return None                                     
        return t, ArmConfig(limits=_lim(lo, hi)), qa, s0

    if gtype == "blocked":
        t = _reachable_target(rng, 1.0, 1.8)
        obs = (float(t[0]), float(t[1]), float(rng.uniform(*OBSTACLE_RADIUS)))
        return t, ArmConfig(obstacle=obs), \
               arm.pose_reaching(t, rng.uniform(*lever_range(t)), _elbow(rng)), s0

    if gtype == "local_optimum":
        t = _reachable_target(rng, 1.0, 1.8)
        d, sign = rng.uniform(*lever_range(t)), _elbow(rng)
        qa, qb = arm.pose_reaching(t, d, sign), arm.pose_reaching(t, d, -sign)
        pts = arm.joint_positions(qa)
        link = rng.integers(1, 3)                          
        c = pts[link] + rng.uniform(0.3, 0.7) * (pts[link + 1] - pts[link])
        rho = rng.uniform(*OBSTACLE_RADIUS)
        obs = (float(c[0]), float(c[1]), float(rho))
        if not arm.collides(pts, obs):                      
            return None
        if arm.collides(arm.joint_positions(qb), obs):      
            return None
        if np.linalg.norm(t - c) < rho + 0.05:             
            return None
        return t, ArmConfig(obstacle=obs), qa, s0

    if gtype in ("fixable", "unfixable"):
        p = rng.uniform(*PLATEAU_BAND)
        if gtype == "fixable":
            t = _reachable_target(rng, 0.85, 1.15)          
            d0 = rng.uniform(0.60, 0.90)
            lo, hi = lever_range(t)
            if not (lo <= d0 <= hi) or lo > 0.30:
                return None
            cfg = ArmConfig(noise_std=np.array([0.0, noise_for_plateau(p, d0), 0.0]))
        else:
            t = _reachable_target(rng, 0.85, 1.8)
            d0 = rng.uniform(*lever_range(t))
            cfg = ArmConfig(noise_std=np.array([noise_for_plateau(p, np.linalg.norm(t)), 0.0, 0.0]))
        return t, cfg, arm.pose_reaching(t, d0, 1.0), s0

    raise ValueError(gtype)


def _cem(cfg, t, rng, mean, std, rounds=CHECKPOINT, batch=16, n_elite=4):
    dists = []
    for _ in range(rounds):
        cand = mean + std * rng.standard_normal((batch, 3))
        tip, hit = arm.execute(cand, cfg, rng)
        c = arm.cost(tip, hit, t, cfg)
        el = cand[c.argsort()[:n_elite]]
        mean, std = el.mean(0), np.maximum(el.std(0), 0.01)
        dists.append(float(np.linalg.norm(tip - t, axis=-1).min()))
    return mean, dists


def _practise(cfg, t, rng, mean, std, rounds=HORIZON, batch=16, n_elite=4):
    means, best, successes = [], [], []
    for _ in range(rounds):
        cand = mean + std * rng.standard_normal((batch, 3))
        tip, hit = arm.execute(cand, cfg, rng)
        c = arm.cost(tip, hit, t, cfg)
        el = cand[c.argsort()[:n_elite]]
        mean, std = el.mean(0), np.maximum(el.std(0), 0.01)
        means.append(mean.copy())
        best.append(float(np.linalg.norm(tip - t, axis=-1).min()))
        successes.append(int(arm.success(tip, hit, t).sum()))
    return means, np.array(best), np.array(successes)

# --- gate 1: does the goal show the symptom its label claims? -----------------
# plateau: competence of the mean at CHECKPOINT and at HORIZON
# headroom: ceiling minus plateau -- what is still winnable
# stuck: not one successful attempt in any round up to HORIZON
RULES = {
    "slow":          dict(plateau=(0.00, 0.00), headroom=(0.50, 1.00), progressing=True),
    "unreachable":   dict(plateau=(0.00, 0.00), headroom=(0.00, 0.02), stuck=True),
    "joint_limit":   dict(plateau=(0.00, 0.00), headroom=(0.50, 1.00), stuck=True),
    "limit_blocked": dict(plateau=(0.00, 0.00), headroom=(0.00, 0.02), stuck=True),
    "blocked":       dict(plateau=(0.00, 0.00), headroom=(0.00, 0.02), stuck=True),
    "local_optimum": dict(plateau=(0.00, 0.00), headroom=(0.50, 1.00), stuck=True),
    "fixable":       dict(plateau=PLATEAU_BAND, headroom=(0.30, 1.00)),
    "unfixable":     dict(plateau=PLATEAU_BAND, headroom=(0.00, 0.08)),
}


def _competence(q, cfg, t):
    return float(arm.competence(q, cfg, t, np.random.default_rng(9), n=1500))


def _symptom_once(gtype, goal, seed):
    """Returns (ok, plateau at CHECKPOINT, stall distance)."""
    target, cfg, m0, s0 = goal
    rule = RULES[gtype]
    lo, hi = rule["plateau"]
    inside = lambda p: lo - 1e-9 <= p <= hi + 1e-9
    slow = gtype == "slow"
    means, best, succ = _practise(cfg, target, np.random.default_rng(seed), m0, s0,
                                  rounds=CHECKPOINT if slow else HORIZON)
    p_ck = _competence(means[CHECKPOINT - 1], cfg, target)
    stall = float(best[CHECKPOINT - 3:CHECKPOINT].mean())
    if not inside(p_ck):
        return False, p_ck, stall
    if slow:       
        return best[-1] < best[len(best) // 2], p_ck, stall
    if rule.get("stuck") and succ.sum() > 0:
        return False, p_ck, stall
    if not inside(_competence(means[-1], cfg, target)):
        return False, p_ck, stall
    return True, p_ck, stall


def validate(gtype, goal, seed=0, n_seeds=1):
    target, cfg, m0, s0 = goal
    assert cfg.noise_std[HELDOUT_JOINT] == 0.0, "noise on joint 3 is held out"
    lo, hi = cfg.limits[1]
    assert not (np.isfinite(lo) and lo >= HELDOUT_LIMIT[0] and hi <= HELDOUT_LIMIT[1]), \
        "joint-2 interval is held out"

    plateaus, stalls = [], []
    for k in range(n_seeds):
        ok, p, s = _symptom_once(gtype, goal, seed * 1000 + k)
        if not ok:
            return False, p, float("nan"), s
        plateaus.append(p); stalls.append(s)
    plateau, stall = float(np.mean(plateaus)), float(np.mean(stalls))

    ceil = ceiling(cfg, target)
    h_lo, h_hi = RULES[gtype]["headroom"]
    return h_lo <= ceil - plateau <= h_hi, plateau, ceil, stall


# --- gate 2: does the goal stall for the reason its label claims? -------------
MECHANISM = {"slow": None, "unreachable": None,
             "blocked": "obstacle", "local_optimum": "obstacle",
             "joint_limit": "limits", "limit_blocked": "limits", "fixable": "noise", "unfixable": "noise"}


def mechanisms_present(cfg):
    out = set()
    if cfg.obstacle is not None:            out.add("obstacle")
    if np.isfinite(np.asarray(cfg.limits)).any(): out.add("limits")
    if np.any(np.asarray(cfg.noise_std) > 0):     out.add("noise")
    return out


def _without(cfg, mech):
    if mech == "obstacle": return dataclasses.replace(cfg, obstacle=None)
    if mech == "limits":   return dataclasses.replace(cfg, limits=NO_LIM.copy())
    if mech == "noise":    return dataclasses.replace(cfg, noise_std=np.zeros(3))
    return cfg


def folding_gain(cfg, target, m0, s0, seed=0):
    """Competence gained by folding the wrist at the fingertip already reached."""
    mean, _ = _cem(cfg, target, np.random.default_rng(seed), m0, s0)
    before = _competence(mean, cfg, target)
    tip = arm.fingertip(mean)
    lo, hi = lever_range(tip)
    if lo > hi: return 0.0
    best = before
    for sign in (1.0, -1.0):
        q = arm.pose_reaching(tip, lo + 1e-6, sign)
        if np.linalg.norm(arm.fingertip(q) - tip) < 1e-4:
            best = max(best, _competence(q, cfg, target))
    return best - before


FOLD = {"fixable": (0.15, 1.00), "unfixable": (0.00, 0.08)} 

def verify_cause(gtype, goal, seed=0, need=0.5):
    """Removing the named mechanism must make the goal easy; nothing else may be present."""
    target, cfg, m0, s0 = goal
    mech = MECHANISM[gtype]
    present = mechanisms_present(cfg)
    assert present == ({mech} if mech else set()), \
        f"{gtype} declares {mech} but carries {present or 'nothing'}"
    if gtype in FOLD:                     
        lo, hi = FOLD[gtype]
        g = folding_gain(cfg, target, m0, s0, seed)
        return lo <= g <= hi, g
    if mech is None:
        return True, float("nan")
    free = _without(cfg, mech)
    mean, _ = _cem(free, target, np.random.default_rng(seed), m0, s0)
    return _competence(mean, free, target) >= need, float("nan")


# --- gate 3: keep the sampled goals spread out ---------------------------------
MIN_SEPARATION = 0.5        
SCALE = np.array([0.5, 0.05, 0.05, 0.05, 0.5, 1.0, 0.1,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def fingerprint(target, cfg, m0):
    """Rotation-invariant description of a goal (everything measured from the target ray)."""
    t = np.asarray(target, float)
    phi = np.arctan2(t[1], t[0])
    obs = cfg.obstacle if cfg.obstacle is not None else (0.0, 0.0, 0.0)
    o = np.array(obs[:2], float)
    r_o = float(np.linalg.norm(o))
    lim = np.asarray(cfg.limits, float)
    lim = np.where(np.isfinite(lim), lim, 0.0).ravel()
    q = np.asarray(m0, float).copy()
    q[0] -= phi
    return np.concatenate([[float(np.linalg.norm(t))],
                           np.asarray(cfg.noise_std, float),
                           [r_o, (np.arctan2(o[1], o[0]) - phi) if r_o > 0 else 0.0, obs[2]],
                           lim, q])


def far_enough(fp, kept_fps, min_sep=MIN_SEPARATION):
    if min_sep <= 0 or not len(kept_fps):
        return True
    d = np.linalg.norm((np.asarray(kept_fps) - fp) / SCALE, axis=1)
    return float(d.min()) >= min_sep


PILOT = 150

def _pilot_shares(gtype, edges, spread, rng):
    counts, found, tries = np.zeros(len(edges) - 1), 0, 0
    while found < PILOT and tries < 30 * PILOT:
        tries += 1
        g = sample_goal(rng, gtype, spread)
        if g is None:
            continue
        ok, plateau, stall = _symptom_once(gtype, g, seed=10**6 + tries)
        if not ok:
            continue
        found += 1
        b = _bin(_match_value(gtype, plateau, stall), edges)
        if b is not None:
            counts[b] += 1
    return counts / max(found, 1)


def _pair_quotas(types, edges, n, spread, rng):
    shares = np.min([_pilot_shares(t, edges, spread, rng) for t in types], axis=0)
    if shares.sum() == 0:
        raise RuntimeError(f"{types} never stall in the same place at spread {spread}")
    exact = n * shares / shares.sum()
    quota = np.floor(exact).astype(int)
    for k in np.argsort(-(exact - quota))[:n - quota.sum()]:
        quota[k] += 1
    return [int(q) for q in quota]


def _bin(value, edges):
    if not (edges[0] <= value <= edges[-1]):
        return None
    return int(min(np.searchsorted(edges, value, side="right") - 1, len(edges) - 2))


def _match_value(gtype, plateau, stall):
    _, quantity, _ = PAIRS[PAIR_OF[gtype]]
    return plateau if quantity == "plateau" else stall


def _sample_type(gtype, n, rng, spread, n_seeds, max_tries, quota=None):
    edges = PAIRS[PAIR_OF[gtype]][2] if gtype in PAIR_OF else None
    filled = [[] for _ in quota] if quota else [[]]

    kept_fps, tries = [], 0
    rejected = {"symptom": 0, "off_band": 0, "bin_full": 0, "cause": 0, "too_close": 0}
    sep = MIN_SEPARATION
    done = lambda: all(len(f) >= q for f, q in zip(filled, quota)) if quota else len(filled[0]) >= n
    while not done() and tries < n * max_tries:
        tries += 1
        if tries % (5 * n) == 0:              
            sep *= 0.5
        g = sample_goal(rng, gtype, spread)
        if g is None:
            continue
        fp = fingerprint(g[0], g[1], g[2])
        if not far_enough(fp, kept_fps, sep):
            rejected["too_close"] += 1
            continue
        ok, plateau, ceil, stall = validate(gtype, g, seed=tries, n_seeds=n_seeds)
        if not ok:
            rejected["symptom"] += 1
            continue
        b = 0
        if quota:
            b = _bin(_match_value(gtype, plateau, stall), edges)
            if b is None:
                rejected["off_band"] += 1
                continue
            if len(filled[b]) >= quota[b]:
                rejected["bin_full"] += 1
                continue
        if not verify_cause(gtype, g, seed=tries)[0]:
            rejected["cause"] += 1
            continue
        filled[b].append((g, plateau, ceil, stall))
        kept_fps.append(fp)
    return filled, dict(tries=tries, rejected=rejected, sep=sep)


def sample_goals(n_per_type, seed=0, spread=SPREAD, n_seeds=1, max_tries=250):
    rng = np.random.default_rng(seed)
    pilot_rng = np.random.default_rng(seed + 1)     
    quotas = {}
    for types, _, edges in PAIRS.values():
        q = _pair_quotas(types, edges, n_per_type, spread, pilot_rng)
        quotas.update({t: q for t in types})

    bins, stats = {}, {}
    for gtype in RULES:
        bins[gtype], stats[gtype] = _sample_type(gtype, n_per_type, rng, spread, n_seeds,
                                                 max_tries, quotas.get(gtype))
        stats[gtype]["quota"] = quotas.get(gtype)

    for types, _, _ in PAIRS.values():
        for k in range(len(bins[types[0]])):
            m = min(len(bins[t][k]) for t in types)
            for t in types:
                stats[t].setdefault("trimmed", 0)
                stats[t]["trimmed"] += len(bins[t][k]) - m
                bins[t][k] = bins[t][k][:m]

    out = {}
    for gtype, bs in bins.items():
        kept = [g for b in bs for g in b]
        out[gtype] = [(g, p, c) for g, p, c, _ in kept]
        stats[gtype].update(kept=len(kept), per_bin=[len(b) for b in bs],
                            plateau=[p for _, p, _, _ in kept], ceiling=[c for _, _, c, _ in kept],
                            stall=[s for _, _, _, s in kept])
        stats[gtype].setdefault("trimmed", 0)
    return out, stats


def files_for(spread, tag=TAG):
    """goals_s010_v3.json / splits_s010_v3.json for spread 0.10, and so on."""
    name = f"s{round(spread * 100):03d}" + (f"_{tag}" if tag else "")
    return f"goals_{name}.json", f"splits_{name}.json"


def _goal_to_dict(gtype, goal, plateau, ceil, index):
    target, cfg, m0, s0 = goal
    limits = [[None if not np.isfinite(a) else float(a) for a in pair]
              for pair in np.asarray(cfg.limits)]
    return {
        "id": f"{gtype}-{index:04d}",
        "type": gtype,
        "target": np.asarray(target).tolist(),
        "noise_std": np.asarray(cfg.noise_std).tolist(),
        "limits": limits,                      
        "obstacle": [float(x) for x in cfg.obstacle] if cfg.obstacle is not None else None,
        "penalty": float(cfg.penalty),
        "init_mean": np.asarray(m0).tolist(),
        "init_std": np.asarray(s0).tolist(),
        "plateau": float(plateau),             
        "ceiling": float(ceil),
    }


def _dict_to_goal(d):
    limits = np.array([[-np.inf if pair[0] is None else pair[0],
                        np.inf if pair[1] is None else pair[1]] for pair in d["limits"]])
    cfg = ArmConfig(noise_std=np.array(d["noise_std"]), limits=limits,
                    obstacle=tuple(d["obstacle"]) if d["obstacle"] else None,
                    penalty=d["penalty"])
    return (np.array(d["target"]), cfg, np.array(d["init_mean"]), np.array(d["init_std"]))


def save_goals(path, goals, seed, n_per_type, spread=SPREAD, n_seeds=1, stats=None):
    records, i = [], 0
    for gtype, kept in goals.items():
        for k, (goal, plateau, ceil) in enumerate(kept):
            d = _goal_to_dict(gtype, goal, plateau, ceil, i)
            if stats is not None:
                d["stall"] = float(stats[gtype]["stall"][k])  
            records.append(d)
            i += 1
    payload = {"version": 3, "seed": seed, "n_per_type": n_per_type,
               "spread": spread, "slow_spread": SLOW_SPREAD, "n_seeds": n_seeds,
               "checkpoint": CHECKPOINT, "horizon": HORIZON,
               "plateau_band": list(PLATEAU_BAND), "obstacle_radius": list(OBSTACLE_RADIUS),
               "limit_margin": LIMIT_MARGIN,
               "match_groups": {k: {"types": list(t), "matched_on": q, "bin_edges": list(e)}
                         for k, (t, q, e) in PAIRS.items()},
               "heldout_limit": list(HELDOUT_LIMIT), "heldout_joint": HELDOUT_JOINT,
               "goals": records}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


def load_goals(path):
    with open(path) as f:
        payload = json.load(f)
    return payload, [(d["type"], _dict_to_goal(d), d) for d in payload["goals"]]


def split_goals(records, seed=0, fracs=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    by_type = {}
    for d in records:
        by_type.setdefault(d["type"], []).append(d["id"])
    names = ("train", "val", "test")
    splits = {k: [] for k in names}
    for gtype, ids in by_type.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        exact = np.array(fracs, dtype=float) * n
        counts = np.floor(exact).astype(int)
        for k in np.argsort(-(exact - counts))[:n - counts.sum()]:
            counts[k] += 1
        if n >= len(names):
            while counts.min() == 0:
                counts[counts.argmin()] += 1
                counts[counts.argmax()] -= 1
        cuts = np.cumsum(counts)
        splits["train"] += ids[:cuts[0]]
        splits["val"] += ids[cuts[0]:cuts[1]]
        splits["test"] += ids[cuts[1]:]
    tr, va, te = (set(splits[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), "a goal is in two splits"
    assert len(tr | va | te) == len(records), "a goal is missing from the splits"
    for name in names:
        types = {i.rsplit("-", 1)[0] for i in splits[name]}
        assert types == set(by_type), f"{name} split is missing a goal type"
    return splits


def save_splits(path, splits, goals_path, seed):
    payload = {"goals_file": goals_path, "seed": seed,
               "sizes": {k: len(v) for k, v in splits.items()}, "splits": splits}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


def load_splits(path):
    with open(path) as f:
        return json.load(f)


def goals_in(records, splits, name):
    ids = set(splits["splits"][name] if "splits" in splits else splits[name])
    return [r for r in records if r[2]["id"] in ids]


def _report(stats):
    q = lambda v: f"{np.percentile(v, 10):.2f}/{np.median(v):.2f}/{np.percentile(v, 90):.2f}" if v else "-"
    print(f"{'type':14s} {'kept':>4s} {'tries':>6s} | {'sympt':>5s} {'band':>5s} {'full':>5s} "
          f"{'cause':>5s} {'close':>5s} {'trim':>4s} | {'per bin':14s} | {'plateau':>14s} "
          f"{'stall':>14s} {'ceiling':>14s}")
    for t, s in stats.items():
        r = s["rejected"]
        print(f"{t:14s} {s['kept']:4d} {s['tries']:6d} | {r['symptom']:5d} {r['off_band']:5d} "
              f"{r['bin_full']:5d} {r['cause']:5d} {r['too_close']:5d} {s['trimmed']:4d} | "
              f"{str(s['per_bin']):14s} | {q(s['plateau']):>14s} {q(s['stall']):>14s} "
              f"{q(s['ceiling']):>14s}")
    print("(plateau / stall / ceiling columns: 10th percentile / median / 90th percentile)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=14, help="goals per type")
    ap.add_argument("spreads", type=float, nargs="*", default=[SPREAD], help="start spreads to sweep")
    ap.add_argument("--seeds", type=int, default=1, help="practice runs the symptom must survive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=TAG, help="version tag in the file names")
    a = ap.parse_args()

    for spread in a.spreads:
        goals_path, splits_path = files_for(spread, a.tag)
        print(f"\n=== spread {spread:.2f}  ({a.n} per type, symptom checked on {a.seeds} run(s)) ===")
        goals, stats = sample_goals(a.n, seed=a.seed, spread=spread, n_seeds=a.seeds)
        _report(stats)
        save_goals(goals_path, goals, seed=a.seed, n_per_type=a.n, spread=spread,
                   n_seeds=a.seeds, stats=stats)
        payload, records = load_goals(goals_path)
        splits = split_goals([d for _, _, d in records], seed=a.seed)
        save_splits(splits_path, splits, goals_path, seed=a.seed)
        print(f"saved {len(records)} goals -> {goals_path}, "
              f"{ {k: len(v) for k, v in splits.items()} } -> {splits_path}")