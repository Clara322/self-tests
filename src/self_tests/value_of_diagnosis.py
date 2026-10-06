import argparse
import os
from enum import Enum
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from self_tests import arm
from self_tests.arm import R_SUCCESS
import generator as G
import generic as Q

CHECKPOINT = 10
RESPONSE_SEEDS = range(100, 105)
EVAL_SEED = 10_000
N_EVAL = 200
COMPARE = 16
EPS_ABANDON = 0.03


class Responses(Enum):
    CONTINUE = "continue"
    ABANDON = "abandon"
    RESTART = "restart"
    SWITCH = "switch"
    REOPTIMISE = "reoptimise"


def cem(n_params, cfg, target, rng, mean, std, batch_size=16, n_elite=4, num_iter=20, repeats=1,
        best_policy=None, best_cost=np.inf, best_dist=np.inf):
    i = 0
    history = []
    while i < num_iter:
        candidate_policies = mean + std * rng.standard_normal((batch_size, n_params))
        executed = np.repeat(candidate_policies, repeats, axis=0)
        tip, hit = arm.execute(executed, cfg, rng)
        costs = arm.cost(tip, hit, target, cfg).reshape(batch_size, repeats).mean(axis=-1)
        elite_inds = costs.argsort()[:n_elite]
        elite_policies = candidate_policies[elite_inds]
        mean = elite_policies.mean(axis=0)
        std = np.maximum(elite_policies.std(axis=0), 0.01)
        dist = np.linalg.norm(tip - target, axis=-1)
        cand_dist = dist.reshape(batch_size, repeats).mean(axis=1)
        j = int(np.argmin(costs))
        record = {
            "policies": candidate_policies, "tips": tip, "hits": hit, "costs": costs,
            "success_rate": float(((dist < R_SUCCESS) & ~hit).mean()),
            "best_dist": float(dist.min()), "mean_dist": float(dist.mean()),
            "hit_rate": float(hit.mean()), "spread": float(std.mean()),
            "mean_after": mean.copy(), "std_after": std.copy(),
            "round_best_policy": candidate_policies[j].copy(), "round_best_cost": float(costs[j]),
        }
        if record["round_best_cost"] < best_cost:
            best_cost = record["round_best_cost"]
            best_policy = record["round_best_policy"].copy()
            best_dist = float(cand_dist[j])
        if best_policy is not None:
            record["best_policy"] = best_policy.copy()
            record["best_cost"] = best_cost
            record["best_dist_so_far"] = best_dist
        history.append(record)
        i += 1
    return mean, std, history


def checkpoint(cfg, target, init_mean, init_std, seed):
    rng = np.random.default_rng(seed)
    mean, std, history = cem(3, cfg, target, rng, init_mean, init_std, num_iter=CHECKPOINT)
    return {"mean": mean, "std": std, "last": history[-1],
            "explore_std": np.max([r["std_after"] for r in history], axis=0)}


def run_cem(cfg, target, rng, mean, std, budget, batch_size=16, n_elite=4, repeats=1):
    per_round = batch_size * repeats
    rounds = budget // per_round
    mean, std, history = cem(3, cfg, target, rng, mean, std, batch_size=batch_size,
                             n_elite=n_elite, num_iter=rounds, repeats=repeats)
    return mean, history, rounds * per_round


def mean_cost(q, cfg, target, rng, n):
    tip, hit = arm.execute(np.repeat(np.asarray(q, float)[None], n, axis=0), cfg, rng)
    return float(arm.cost(tip, hit, target, cfg).mean())


def respond(response, checkpoint, cfg, target, seed, budget):
    rng = np.random.default_rng(seed)
    last = checkpoint["last"]
    if response == Responses.ABANDON:
        return checkpoint["mean"], 0
    elif response == Responses.CONTINUE:
        new_mean, history, spent = run_cem(cfg, target, rng, checkpoint["mean"], checkpoint["std"], budget)
        return new_mean, spent
    elif response == Responses.SWITCH:
        start = arm.mirror(last["best_policy"], target)
        new_mean, history, spent = run_cem(cfg, target, rng, start, checkpoint["explore_std"], budget)
        return new_mean, spent
    elif response == Responses.RESTART:
        start = rng.uniform(-np.pi, np.pi, 3)
        new_mean, history, spent = run_cem(cfg, target, rng, start, np.full(3, np.pi), budget)
        if last["best_cost"] < history[-1]["best_cost"]:
            return checkpoint["mean"], spent
        return new_mean, spent
    elif response == Responses.REOPTIMISE:
        new_mean, history, spent = run_cem(cfg, target, rng, checkpoint["mean"],
                                           np.full(3, 1.0), budget - COMPARE)
        better = new_mean
        if mean_cost(checkpoint["mean"], cfg, target, rng, COMPARE // 2) < mean_cost(new_mean, cfg, target, rng, COMPARE // 2):
            better = checkpoint["mean"]
        return better, spent + COMPARE
    raise ValueError(response)

FULL, PROBE = 128, 16                   
LAMBDA = EPS_ABANDON / FULL             
RESPONSES = [r.value for r in Responses]
REPEATS = 10
HOLDOUT = 0.25


def goal_outcomes(i, goal, gid, seed):
    target, cfg, m0, s0 = goal
    ckpt = checkpoint(cfg, target, m0, s0, seed + 7919 * i)     
    comp = lambda q: arm.competence(q, cfg, target, np.random.default_rng(EVAL_SEED), n=N_EVAL)
    base = comp(ckpt["mean"])
    rows = []
    for budget in (FULL, FULL - PROBE):
        for r in Responses:
            gains, used = [], 0
            for s in RESPONSE_SEEDS:
                policy, used = respond(r, ckpt, cfg, target, 1000 * i + s, budget)
                gains.append(comp(policy) - base)
            rows.append(dict(goal_id=gid, budget=budget, response=r.value,
                             gain=float(np.mean(gains)), spent=int(used)))
    return rows


TRIAL = [Responses.SWITCH, Responses.REOPTIMISE, Responses.RESTART]
SLICE = 16                           


def slice_best(response, ckpt, cfg, target, seed, n=SLICE):
    rng = np.random.default_rng(seed)
    last = ckpt["last"]
    if response == Responses.SWITCH:
        start, std = arm.mirror(last["best_policy"], target), ckpt["explore_std"]
    elif response == Responses.RESTART:
        start, std = rng.uniform(-np.pi, np.pi, 3), np.full(3, np.pi)
    else:                                 
        start, std = ckpt["mean"], np.full(3, 1.0)
    _, history, _ = run_cem(cfg, target, rng, start, std, n)
    return history[-1]["round_best_cost"]


def goal_trial(i, goal, gid, seed, n=SLICE):
    target, cfg, m0, s0 = goal
    ckpt = checkpoint(cfg, target, m0, s0, seed + 7919 * i)
    comp = lambda q: arm.competence(q, cfg, target, np.random.default_rng(EVAL_SEED), n=N_EVAL)
    base = comp(ckpt["mean"])
    level = ckpt["last"]["round_best_cost"]        
    gains, spent, picks = [], [], []
    for s in RESPONSE_SEEDS:
        seeds = [7_000_000 + 1000 * i + 10 * s + k for k in range(len(TRIAL))]
        improvement = [level - slice_best(r, ckpt, cfg, target, sd, n) for r, sd in zip(TRIAL, seeds)]
        k = int(np.argmax(improvement))
        if improvement[k] <= 0:
            gains.append(0.0); spent.append(n * len(TRIAL)); picks.append("abandon")
            continue
        policy, used = respond(TRIAL[k], ckpt, cfg, target, seeds[k], FULL - n * (len(TRIAL) - 1))
        gains.append(comp(policy) - base)
        spent.append(used + n * (len(TRIAL) - 1))
        picks.append(TRIAL[k].value)
    return dict(goal_id=gid, gain=float(np.mean(gains)), spent=float(np.mean(spent)),
                picks="/".join(picks))


def trial_outcomes(goals_path, cache, seed=0, n=SLICE):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    _, records = G.load_goals(goals_path)
    print(f"running trial and error ({n} attempts per candidate) on {len(records)} goals "
          f"(cached to {cache}) ...", flush=True)
    rows = Parallel(n_jobs=-1)(delayed(goal_trial)(i, goal, d["id"], seed, n)
                               for i, (_, goal, d) in enumerate(records))
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def outcomes(goals_path, cache, seed=0):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    _, records = G.load_goals(goals_path)
    print(f"running every response on {len(records)} goals (cached to {cache}) ...", flush=True)
    rows = Parallel(n_jobs=-1)(delayed(goal_outcomes)(i, goal, d["id"], seed)
                               for i, (_, goal, d) in enumerate(records))
    df = pd.DataFrame([r for rs in rows for r in rs])
    df.to_csv(cache, index=False)
    return df


def utility_table(out, lam=LAMBDA):
    out = out.copy()
    extra = np.where(out["budget"] == FULL, 0, PROBE)
    out["utility"] = out["gain"] - lam * (out["spent"] + extra)
    u = {}
    for (gid, b), g in out.groupby(["goal_id", "budget"]):
        u.setdefault(gid, {})[b] = g.set_index("response").loc[RESPONSES, "utility"].to_numpy()
    return u


def fit_classifier(train, cols, labels_train, seed):
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)
    clf.fit(train[cols], labels_train)
    return clf


def fit_predict(train, test, cols, labels_train, seed, proba=False):
    clf = fit_classifier(train, cols, labels_train, seed)
    pred = clf.predict(test[cols])
    return (pred, clf.predict_proba(test[cols]).max(1)) if proba else pred


THRESHOLDS = np.round(np.arange(0.30, 1.001, 0.05), 2)    

def choose_threshold(train, cols_h, cols_p, U, best, seed, folds=5):
    rng = np.random.default_rng(seed)
    ids = train["goal_id"].to_numpy()
    fold = np.empty(len(ids), int)
    for t in train["type"].unique():
        idx = np.flatnonzero(train["type"].to_numpy() == t)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % folds
    u_hist, u_probe, conf = np.zeros(len(ids)), np.zeros(len(ids)), np.zeros(len(ids))
    for f in range(folds):
        tr, te = train[fold != f], train[fold == f]
        yh = [best(g, FULL) for g in tr["goal_id"]]
        yp = [best(g, FULL - PROBE) for g in tr["goal_id"]]
        ph, c = fit_predict(tr, te, cols_h, yh, seed, proba=True)
        pp = fit_predict(tr, te, cols_p, yp, seed)
        m = fold == f
        u_hist[m] = [U(g, FULL)[RESPONSES.index(p)] for g, p in zip(te["goal_id"], ph)]
        u_probe[m] = [U(g, FULL - PROBE)[RESPONSES.index(p)] for g, p in zip(te["goal_id"], pp)]
        conf[m] = c
    value = [np.mean(np.where(conf < t, u_probe, u_hist)) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(value))])

GATE_PROBE = "off-target 0.1"
GATE_FOLDS = 5


def _folds(frame, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(frame), int)
    types = frame["type"].to_numpy()
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % k
    return fold


def oof_gains(train, cols_h, cols_p, U, best, seed):
    train = train.reset_index(drop=True)
    fold = _folds(train, GATE_FOLDS, 10_000 + seed)
    u_h, u_p = np.zeros(len(train)), np.zeros(len(train))
    for f in range(GATE_FOLDS):
        tr, te = train[fold != f], train[fold == f]
        ph = fit_predict(tr, te, cols_h, [best(g, FULL) for g in tr["goal_id"]], seed)
        pp = fit_predict(tr, te, cols_p, [best(g, FULL - PROBE) for g in tr["goal_id"]], seed)
        m = fold == f
        u_h[m] = [U(g, FULL)[RESPONSES.index(p)] for g, p in zip(te["goal_id"], ph)]
        u_p[m] = [U(g, FULL - PROBE)[RESPONSES.index(p)] for g, p in zip(te["goal_id"], pp)]
    return u_h, u_p, u_p - u_h


def fit_gate(X, y, seed):
    reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1)
    return reg.fit(X, y)


def selective(train, test, cols_h, cols_p, U, best, seed):
    train = train.reset_index(drop=True)
    _, _, gain = oof_gains(train, cols_h, cols_p, U, best, seed)
    return fit_gate(train[cols_h], gain, seed).predict(test[cols_h]) > 0


CHOICE_POLICIES = ["choose probe", "choose probe or skip"]
CHOICE_DIFFS = [("choose probe", "history"), ("choose probe", "history + off-target 0.1"),
                ("choose probe or skip", "history"), ("choose probe or skip", "selective self-test")]


def choose_probe(train, test, sets, U, best, seed, existing):
    """Select among existing probes from history, optionally skipping the test."""
    train = train.reset_index(drop=True)
    probes = ("off-target 0.1", "broad", "repeat")
    h = sets["hist r10"]
    k = min(GATE_FOLDS, int(train.groupby("type").size().min()))
    if k < 2:
        raise ValueError("Probe selection needs at least two training goals per cause.")
    fold = _folds(train, k, 10_000 + seed)
    uh, up, prob = np.zeros(len(train)), np.zeros((len(train), 3)), np.zeros((len(train), len(RESPONSES)))

    def probabilities(model, frame):
        p = np.zeros((len(frame), len(RESPONSES)))
        p[:, [RESPONSES.index(label) for label in model.classes_]] = model.predict_proba(frame[h])
        return p

    for f in range(k):
        tr, te = train[fold != f], train[fold == f]
        labels = [best(g, FULL - PROBE) for g in tr["goal_id"]]
        ph = fit_predict(tr, te, h, [best(g, FULL) for g in tr["goal_id"]], seed)
        uh[fold == f] = [U(g, FULL)[RESPONSES.index(p)] for g, p in zip(te["goal_id"], ph)]
        prob[fold == f] = probabilities(fit_classifier(tr, h, labels, seed), te)
        for j, name in enumerate(probes):
            pp = fit_predict(tr, te, h + sets[name], labels, seed)
            up[fold == f, j] = [U(g, FULL - PROBE)[RESPONSES.index(p)] for g, p in zip(te["goal_id"], pp)]
    model = fit_gate(np.c_[train[h], prob], up - uh[:, None], seed)
    classifier = fit_classifier(train, h, [best(g, FULL - PROBE) for g in train["goal_id"]], seed)
    gain = model.predict(np.c_[test[h], probabilities(classifier, test)])
    selected, test_now = gain.argmax(1), gain.max(1) > 0
    values = np.column_stack([existing["history + " + p] for p in probes])
    chosen = values[np.arange(len(test)), selected]
    return {"choose probe": chosen.tolist(),
            "choose probe or skip": np.where(test_now, chosen, existing["history"]).tolist(),
            "_chosen_probe": np.array(probes)[selected].tolist(), "_probed_choice": test_now.tolist()}


def score_split(df, u, train, test, seed, trial=None, probe_selection=False):
    U = lambda gid, b: u[gid][b]
    best = lambda gid, b: RESPONSES[int(np.argmax(U(gid, b)))]
    res = {}
    ids = test["goal_id"].tolist()
    res["oracle"] = [U(g, FULL).max() for g in ids]
    for k, r in enumerate(RESPONSES):
        res[f"always {r}"] = [U(g, FULL)[k] for g in ids]
    k_fixed = int(np.argmax(np.mean([U(g, FULL) for g in train["goal_id"]], axis=0)))
    res["best fixed"] = [U(g, FULL)[k_fixed] for g in ids]
    res["_best_fixed_name"] = RESPONSES[k_fixed]

    sets = Q.feature_sets(df)
    y_full = [best(g, FULL) for g in train["goal_id"]]
    y_probe = [best(g, FULL - PROBE) for g in train["goal_id"]]
    pred = fit_predict(train, test, sets["hist r10"], y_full, seed)
    res["history"] = [U(g, FULL)[RESPONSES.index(p)] for g, p in zip(ids, pred)]
    res["_pick_history"] = list(pred)
    for name, tv in (trial or {}).items():
        res[name] = [tv[g] for g in ids]
    for probe in ("repeat", "broad", "posture", "off-target", "off-target 0.5", "off-target 0.1"):
        cols = sets["hist r10"] + sets[probe]
        pred = fit_predict(train, test, cols, y_probe, seed)
        res[f"history + {probe}"] = [U(g, FULL - PROBE)[RESPONSES.index(p)] for g, p in zip(ids, pred)]
        if probe == "posture":
            res["_pick_posture"] = list(pred)
        if probe == "off-target":
            res["_pick_off"] = list(pred)
        if probe == "off-target 0.1":
            res["_pick_off10"] = list(pred)

    cols_h, cols_p = sets["hist r10"], sets["hist r10"] + sets["posture"]
    t = choose_threshold(train.reset_index(drop=True), cols_h, cols_p, U, best, seed)
    ph, conf = fit_predict(train, test, cols_h, y_full, seed, proba=True)
    pp = fit_predict(train, test, cols_p, y_probe, seed)
    probe_now = conf < t
    res["probe when unsure"] = [U(g, FULL - PROBE)[RESPONSES.index(b)] if q else U(g, FULL)[RESPONSES.index(a)]
                                for g, a, b, q in zip(ids, ph, pp, probe_now)]
    res["_probed"] = list(probe_now)
    res["_threshold"] = t

    cols_p = sets["hist r10"] + sets[GATE_PROBE]
    ph = fit_predict(train, test, cols_h, y_full, seed)
    pp = fit_predict(train, test, cols_p, y_probe, seed)
    uh = np.array([U(g, FULL)[RESPONSES.index(p)] for g, p in zip(ids, ph)])
    up = np.array([U(g, FULL - PROBE)[RESPONSES.index(p)] for g, p in zip(ids, pp)])
    go = selective(train, test, cols_h, cols_p, U, best, seed)
    res["selective self-test"] = list(np.where(go, up, uh))
    res["selective (upper bound)"] = list(np.maximum(up, uh))
    res["_probed_sel"] = list(go)
    if probe_selection:
        res.update(choose_probe(train, test, sets, U, best, seed, res))
    return res


POLICIES = ["oracle", "best fixed", "history", "history + broad", "trial and error", "trial and error (32)",
            "history + posture", "history + off-target", "history + off-target 0.5", "history + off-target 0.1",
            "selective self-test", "selective (upper bound)",
            "probe when unsure", "history + repeat"] + \
           [f"always {r}" for r in RESPONSES]


NAMES = {"history + broad": "history + random", "history + posture": "history + self-test",
         "history + off-target": "off-target test", "history + off-target 0.5": "off-target (0.5)",
         "history + off-target 0.1": "off-target (0.1)"}
MAIN = ["best fixed", "history", "history + broad", "trial and error", "trial and error (32)", "history + posture",
        "history + off-target", "history + off-target 0.5", "history + off-target 0.1"]
MIXES = {
    "equal mix (as generated)": {},
    "mostly trapped stalls": {"local_optimum": 3, "joint_limit": 3, "blocked": 3, "limit_blocked": 3},
    "mostly local causes": {"slow": 3, "fixable": 3, "unfixable": 3, "unreachable": 3},
}


def report(df, u, results, ids_types, title):
    oracle = np.mean(results["oracle"])
    print(f"\n{title}")
    print(f"{'policy':22s} {'utility':>8s} {'share of oracle':>16s}")
    for p in active_policies(results):
        v = np.mean(results[p])
        print(f"{NAMES.get(p, p):22s} {v:8.3f} {100 * v / oracle:15.0f}%")

    types_all = np.array(ids_types)
    main = MAIN
    print(f"\nshare of the oracle under other mixes of stall types\n{'mix':28s}"
          + "".join(f"{NAMES.get(p, p):>21s}" for p in main))
    for name, w in MIXES.items():
        wt = np.array([w.get(t, 1.0) for t in types_all])
        o = np.sum(wt * np.array(results["oracle"]))
        print(f"{name:28s}" + "".join(f"{100 * np.sum(wt * np.array(results[p])) / o:20.0f}%" for p in main))
    print(f"\n'best fixed' picked on training goals: {', '.join(sorted(set(results['_best_fixed_name'])))}")
    th = results["_threshold"]
    print(f"'probe when unsure': probes when history's confidence is below "
          f"{', '.join(f'{t:.2f}' for t in sorted(set(th)))} (chosen on training goals only); "
          f"probed {100 * np.mean(results['_probed']):.0f}% of goals")

    types = list(dict.fromkeys(ids_types))
    show = ["oracle"] + MAIN
    print(f"\nmean utility by goal type (0 = abandon at once; negative = spent attempts for nothing)"
          f"\n{'type':15s}" + "".join(f"{NAMES.get(s, s):>21s}" for s in show))
    for t in types:
        m = np.array([x == t for x in ids_types])
        print(f"{t:15s}" + "".join(f"{np.mean(np.array(results[s])[m]):21.3f}" for s in show)
              + f"   probed {100 * np.mean(np.array(results['_probed'])[m]):3.0f}%")


CURVE = (10, 25, 50, None)                    
PRICES = (0.0, EPS_ABANDON, 0.10, 0.25, 0.50) 
EXTRA_SPLITS = 3                              


def make_splits(dev, test, n):
    if test is not None:
        return [(dev, test)]
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(n):
        held = set()
        for t in dev["type"].unique():
            ids = rng.permutation(dev.loc[dev["type"] == t, "goal_id"].values)
            held.update(ids[:max(1, round(HOLDOUT * len(ids)))])
        pairs.append((dev[~dev["goal_id"].isin(held)], dev[dev["goal_id"].isin(held)]))
    return pairs


def active_policies(results=None, probe_selection=False):
    if probe_selection or (results is not None and "choose probe or skip" in results):
        return POLICIES + CHOICE_POLICIES
    return POLICIES


def pooled(df, u, trial, pairs, probe_selection=False):
    policies = active_policies(probe_selection=probe_selection)
    res = {p: [] for p in policies}
    res.update(_best_fixed_name=[], _threshold=[], _probed=[], _pick_history=[], _pick_posture=[], _pick_off=[], _pick_off10=[],
               _probed_sel=[])
    types = []
    if probe_selection:
        res.update(_chosen_probe=[], _probed_choice=[])
    for r, (tr, te) in enumerate(pairs):
        one = score_split(df, u, tr.reset_index(drop=True), te.reset_index(drop=True), r, trial, probe_selection)
        for p in policies:
            res[p] += list(one[p])
        if probe_selection:
            res['_chosen_probe'] += one['_chosen_probe']
            res['_probed_choice'] += one['_probed_choice']
        res["_best_fixed_name"].append(one["_best_fixed_name"])
        res["_threshold"].append(one["_threshold"])
        res["_probed"] += one["_probed"]
        res["_pick_history"] += one["_pick_history"]
        res["_pick_posture"] += one["_pick_posture"]
        res["_pick_off"] += one["_pick_off"]
        res["_pick_off10"] += one["_pick_off10"]
        res["_probed_sel"] += one["_probed_sel"]
        types += te["type"].tolist()
    return res, types


def share(res, p):
    return 100 * np.mean(res[p]) / np.mean(res["oracle"])


def bootstrap(res, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    o = np.array(res["oracle"])
    arr = {p: np.array(res[p]) for p in MAIN}
    idx = rng.integers(0, len(o), (n, len(o)))
    out = {}
    for p in MAIN:
        s = 100 * arr[p][idx].mean(1) / o[idx].mean(1)
        out[p] = np.percentile(s, [2.5, 97.5])
    for key, t in (("diff", "trial and error"), ("diff32", "trial and error (32)")):
        d = 100 * (arr["history + posture"][idx].mean(1) - arr[t][idx].mean(1)) / o[idx].mean(1)
        out[key] = np.percentile(d, [2.5, 97.5])
    for a, b in OFF_DIFFS:
        d = 100 * (arr[a][idx].mean(1) - arr[b][idx].mean(1)) / o[idx].mean(1)
        out[(a, b)] = np.percentile(d, [2.5, 97.5])
    return out


OFF_DIFFS = [("history + off-target", "history"), ("history + off-target", "trial and error"),
             ("history + posture", "history + off-target"), ("history + off-target 0.5", "history")]


def subsample(train, n, seed):
    if n is None:
        return train
    rng = np.random.default_rng(seed)
    keep = []
    for t in train["type"].unique():
        ids = train.loc[train["type"] == t, "goal_id"].values
        keep += list(rng.permutation(ids)[:n])
    return train[train["goal_id"].isin(keep)]


def trial_utilities(tr_outs, lam):
    return {name: dict(zip(t["goal_id"], t["gain"] - lam * t["spent"])) for name, t in tr_outs.items()}


def picks_summary(picks):
    vals, counts = np.unique(np.array(picks), return_counts=True)
    order = np.argsort(-counts)
    return ", ".join(f"{vals[i]} {100 * counts[i] / len(picks):.0f}%" for i in order[:3])


def run_checks(df, out, tr_outs, pairs, mode, show=None, pick_cols=None, probe_selection=False):
    show = show or MAIN
    if probe_selection:
        show = list(show) + ["choose probe", "choose probe or skip"]
    pick_cols = pick_cols or [("history", "_pick_history"), ("history + self-test", "_pick_posture"),
                              ("off-target test", "_pick_off")]
    u = utility_table(out)
    trial = trial_utilities(tr_outs, LAMBDA)
    head = lambda: "".join(f"{NAMES.get(p, p):>21s}" for p in show)

    print(f"\n=== CHECK 1: how much labelled training does diagnosis need? ({mode}) ===")
    print("trial and error needs none; the others learn from this many training stalls per cause")
    print(f"{'training stalls per cause':28s}" + head())
    for n in CURVE:
        sub = [(subsample(tr, n, r), te) for r, (tr, te) in enumerate(pairs)]
        res, _ = pooled(df, u, trial, sub, probe_selection)
        print(f"{('all' if n is None else str(n)):28s}" + "".join(f"{share(res, p):20.0f}%" for p in show))

    print(f"\n=== CHECK 2: a cause never seen in training ({mode}) ===")
    print("train on the other seven causes, test on the held-out one (mean utility; trial and error is unaffected)")
    types = [t for t in G.RULES if t in set(df["type"])]
    print(f"{'held-out cause':16s}{'oracle':>9s}" + head())
    tot = {p: [] for p in show + ["oracle"]}
    picks = {}
    for t in types:
        sub = [(tr[tr["type"] != t], te[te["type"] == t]) for tr, te in pairs]
        res, _ = pooled(df, u, trial, sub, probe_selection)
        for p in show + ["oracle"]:
            tot[p] += list(res[p])
        picks[t] = [res[k] for _, k in pick_cols]
        print(f"{t:16s}{np.mean(res['oracle']):9.3f}" + "".join(f"{np.mean(res[p]):21.3f}" for p in show))
    print(f"{'all, as share':16s}{'100%':>9s}" + "".join(f"{share(tot, p):20.0f}%" for p in show))
    print("\nwhat the learned strategies chose for the cause they never saw")
    print(f"{'held-out cause':16s}" + "".join(f"  {lab + ' chose':38s}" for lab, _ in pick_cols))
    for t in types:
        print(f"{t:16s}" + "".join(f"  {picks_summary(p):38s}" for p in picks[t]))

    print(f"\n=== CHECK 3: what if trying costs more? ({mode}) ===")
    print("price = competence lost by spending the whole 128-attempt budget (0.03 is the default)."
          "\nThe learned policies re-learn under each price; trial and error's rule stays fixed.")
    print(f"{'price of full budget':28s}" + head() + "   best (not oracle)")
    for price in PRICES:
        lam = price / FULL
        u_l = utility_table(out, lam)
        trial_l = trial_utilities(tr_outs, lam)
        res, _ = pooled(df, u_l, trial_l, pairs, probe_selection)
        sh = {p: share(res, p) for p in show}
        win = max(sh, key=sh.get)
        print(f"{price:<28.2f}" + "".join(f"{sh[p]:20.0f}%" for p in show) + f"   {NAMES.get(win, win)}")

FOCUS = ["best fixed",              
         "history",                
         "history + broad",         
         "trial and error",         
         "history + off-target 0.1", 
         "selective self-test",      
         "selective (upper bound)"] 
FOCUS_DIFFS = [("history + off-target 0.1", "history"),          
               ("history + off-target 0.1", "history + broad"),
               ("history + off-target 0.1", "trial and error"),
               ("selective self-test", "history"),                
               ("selective self-test", "history + off-target 0.1")]
FOCUS_NAMES = {"history + broad": "history + random", "history + off-target 0.1": "self-test (0.10)",
               "selective self-test": "selective self-test", "selective (upper bound)": "selective, upper bound"}
VERDICT_ROWS = {0, 3, 4}       


def intervals(res, policies, diffs, n=2000, seed=0):
    """95% bootstrap intervals for shares of the oracle and for differences between strategies."""
    rng = np.random.default_rng(seed)
    o = np.array(res["oracle"])
    idx = rng.integers(0, len(o), (n, len(o)))
    arr = {p: np.array(res[p]) for p in set(policies) | {x for d in diffs for x in d}}
    out = {p: np.percentile(100 * arr[p][idx].mean(1) / o[idx].mean(1), [2.5, 97.5]) for p in policies}
    for a_, b_ in diffs:
        d = 100 * (arr[a_][idx].mean(1) - arr[b_][idx].mean(1)) / o[idx].mean(1)
        out[(a_, b_)] = np.percentile(d, [2.5, 97.5])
    return out


def verdict(point, lo, hi):
    if point <= 0:
        return "NOT REPLICATED (not positive)"
    if lo > 0:
        return "SUPPORTED (positive, interval excludes 0)"
    return "SUGGESTIVE (positive, interval includes 0)"


def report_focus(df, res, ids_types, mode, optimistic):
    focus, differences = list(FOCUS), list(FOCUS_DIFFS)
    if "choose probe or skip" in res:
        focus += CHOICE_POLICIES
        differences += CHOICE_DIFFS
    nm = lambda p: FOCUS_NAMES.get(p, p)
    print("\n1. SANITY: no self-test attempt is AIMED closer than 0.10 to a reachable target")
    sanity = df.groupby("type").agg(
        success=("off10_success", "mean"), closest_aimed=("chk_off10_aimed_min", "min"),
        closest_landed=("chk_off10_landed_min", "min"))
    print(sanity.round(3).to_string())
    print("   closest_aimed must be 0.10 for every reachable cause (out-of-reach aims sit on the reach\n"
          "   boundary, so theirs is lower; they cannot succeed anyway). success = share of attempts that\n"
          "   still reached the target, through noise or a clipped joint; report it alongside the result.")

    ci = intervals(res, focus, differences)
    note = "; re-splits overlap, so these are optimistic" if optimistic else ""
    print(f"\n2. MAIN RESULT ({mode})\n   share of the best possible utility, 95% interval (bootstrap over goals{note})")
    print(f"   {'best possible':24s} {np.mean(res['oracle']):.3f} utility = 100%")
    for p in focus:
        print(f"   {nm(p):24s} {share(res, p):4.0f}%   [{ci[p][0]:.0f}%, {ci[p][1]:.0f}%]")

    print("\n3. DIFFERENCES (points of the best possible)")
    o = np.mean(res["oracle"])
    for k, (a_, b_) in enumerate(differences):
        pt = 100 * (np.mean(res[a_]) - np.mean(res[b_])) / o
        lo, hi = ci[(a_, b_)]
        line = f"   {nm(a_) + ' minus ' + nm(b_):48s} {pt:+5.1f}   [{lo:+.0f}, {hi:+.0f}]"
        if k in VERDICT_ROWS or k >= len(FOCUS_DIFFS):
            line += f"   -> {verdict(pt, lo, hi)}" if not optimistic else "   (verdict on --test runs only)"
        print(line)
    print("   Rule (for each comparison marked with a verdict): supported if positive with an interval\n"
          "   excluding 0 on at least one test set; suggestive if positive but the interval includes 0;\n"
          "   not replicated if negative on either set. Applies to goal sets generated after\n"
          "   30 Sept 2026, 19:00 (seed 9 onwards); earlier sets are exploratory for these strategies.")

    print("\n4. WHERE IT COMES FROM: mean utility per stall cause (0 = abandon at once)")
    show = ["oracle"] + focus
    print(f"   {'cause':15s}" + "".join(f"{('best possible' if p == 'oracle' else nm(p)):>24s}" for p in show))
    for t in dict.fromkeys(ids_types):
        m = np.array([x == t for x in ids_types])
        print(f"   {t:15s}" + "".join(f"{np.mean(np.array(res[p])[m]):24.3f}" for p in show))

    print("\n5. SELECTIVE SELF-TEST: share of stalls it chose to test (predicted gain > 0)")
    probed = np.array(res["_probed_sel"])
    for t in dict.fromkeys(ids_types):
        m = np.array([x == t for x in ids_types])
        print(f"   {t:15s} {100 * probed[m].mean():5.0f}%")
    print(f"   {'all stalls':15s} {100 * probed.mean():5.0f}%")
    if '_probed_choice' in res:
        print(f"\n6. PROBE CHOICE: tested {100 * np.mean(res['_probed_choice']):.1f}% of situations")
        print("   proposed probe (before deciding whether to skip): " + picks_summary(res['_chosen_probe']))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("goals")
    ap.add_argument("splits")
    ap.add_argument("--test", action="store_true", help="fit on train+val, score once on test")
    ap.add_argument("--checks", action="store_true", help="add the three robustness checks (focused report)")
    ap.add_argument("--all", action="store_true", help="print every strategy and table (the full earlier output)")
    ap.add_argument("--no-checks", action="store_true", help="with --all: skip the three robustness checks")
    ap.add_argument("--probe-selection", action="store_true", help="compare learned probe choice/skip with the existing baselines")
    a = ap.parse_args()

    goals_file = Path(a.goals)
    tag = goals_file.stem.replace("goals_", "")
    cache = lambda prefix: str(goals_file.with_name(f"{prefix}_{tag}.csv"))
    out = outcomes(a.goals, cache("outcomes"))
    u = utility_table(out)
    tr_outs = {"trial and error": trial_outcomes(a.goals, cache("trial")),
               "trial and error (32)": trial_outcomes(a.goals, cache("trial32"), n=2 * SLICE)}
    trial = trial_utilities(tr_outs, LAMBDA)
    print("describing history and tests (same rule as generic.py) ...", flush=True)
    df = Q.build(a.goals, a.splits, 16)

    dev = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True) if a.test else None
    mode = "FROZEN TEST SPLIT, fit on train+val, scored once" if a.test else f"train+val, {REPEATS} re-splits pooled"
    res, types = pooled(df, u, trial, make_splits(dev, test, REPEATS), a.probe_selection)

    if not a.all:
        report_focus(df, res, types, mode, optimistic=not a.test)
        if a.checks:
            NAMES.update(FOCUS_NAMES)
            run_checks(df, out, tr_outs, make_splits(dev, test, EXTRA_SPLITS),
                       "test split" if a.test else f"train+val, {EXTRA_SPLITS} re-splits",
                       show=[p for p in FOCUS if p != "selective (upper bound)"],
                       pick_cols=[("history", "_pick_history"), ("self-test (0.10)", "_pick_off10")],
                       probe_selection=a.probe_selection)
        raise SystemExit

    print("\nwhat the 16 test attempts did, per goal type (mean over all goals)")
    print("  success = share of attempts that reached the target; aimed/landed = closest any attempt")
    print("  was aimed at / landed to the target. The off-target tests must show success 0.")
    sanity = df.groupby("type").agg(
        posture_success=("post_success", "mean"), posture_any=("post_success", lambda v: (v > 0).mean()),
        off_success=("off_success", "mean"), off_aimed=("chk_off_aimed_min", "min"),
        off_landed=("chk_off_landed_min", "min"), off50_success=("off50_success", "mean"),
        off50_aimed=("chk_off50_aimed_min", "min"), off50_landed=("chk_off50_landed_min", "min"))
    print(sanity.round(3).to_string())
    print("  posture_any = share of goals where at least one posture attempt reached the target")

    best = {g: RESPONSES[int(np.argmax(u[g][FULL]))] for g in u}
    df["best"] = df["goal_id"].map(best)
    print("\nbest response per goal type (share of goals):")
    print((pd.crosstab(df["type"], df["best"], normalize="index") * 100).round(0).to_string())

    report(df, u, res, types, mode)
    if a.probe_selection:
        report_focus(df, res, types, mode, optimistic=not a.test)
    ci = bootstrap(res)
    print("\n95% intervals (bootstrap over goals" + ("" if a.test else "; re-splits overlap, so these are optimistic") + ")")
    for p in MAIN:
        print(f"  {NAMES.get(p, p):22s} {share(res, p):4.0f}%   [{ci[p][0]:.0f}%, {ci[p][1]:.0f}%]")
    print(f"  {'self-test minus trial and error':34s} [{ci['diff'][0]:+.0f}, {ci['diff'][1]:+.0f}] points")
    print(f"  {'self-test minus trial and error (32)':34s} [{ci['diff32'][0]:+.0f}, {ci['diff32'][1]:+.0f}] points")
    for a_, b_ in OFF_DIFFS:
        lab = f"{NAMES.get(a_, a_)} minus {NAMES.get(b_, b_)}"
        print(f"  {lab:34s} [{ci[(a_, b_)][0]:+.0f}, {ci[(a_, b_)][1]:+.0f}] points")

    if not a.no_checks:
        pairs = make_splits(dev, test, EXTRA_SPLITS)
        run_checks(df, out, tr_outs, pairs, "test split" if a.test else f"train+val, {EXTRA_SPLITS} re-splits",
                   probe_selection=a.probe_selection)