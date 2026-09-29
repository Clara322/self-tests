import matplotlib.pyplot as plt
import numpy as np
from self_tests import arm
from self_tests.goals import GOALS
from self_tests.arm import R_SUCCESS
from self_tests.utils.plots import plot_competence
from self_tests.utils.plots import plot_run
from enum import Enum

SEED = 0
CHECKPOINT = 10
BUDGETS = (128, 112)
RESPONSE_SEEDS = range(100, 105)
EVAL_SEED = SEED + 10_000
N_EVAL = 200
COMPARE = 16
EPS_ABANDON = 0.03

class Responses(Enum):
    CONTINUE = "continue"
    ABANDON = "abandon"
    RESTART = "restart"
    SWITCH = "switch"
    REOPTIMISE = "reoptimise"

def cem(n_params, cfg, target, rng, mean, std, batch_size=16, n_elite=4, num_iter=20, repeats = 1, best_policy=None, best_cost=np.inf, best_dist=np.inf):
    i = 0
    history = []
    while (i < num_iter):
        # Sample
        candidate_policies = mean + std * rng.standard_normal((batch_size, n_params))
        # Evaluate policy
        executed = np.repeat(candidate_policies, repeats, axis=0)
        tip, hit = arm.execute(executed, cfg, rng)
        costs = arm.cost(tip, hit, target, cfg).reshape(batch_size, repeats).mean(axis=-1)

        elite_inds = costs.argsort()[:n_elite]
        elite_policies = candidate_policies[elite_inds]
        # Update mean and variance
        mean = elite_policies.mean(axis=0)
        std = np.maximum(elite_policies.std(axis=0), 0.01)

        dist = np.linalg.norm(tip - target, axis=-1)
        cand_dist = dist.reshape(batch_size, repeats).mean(axis=1)
        j = int(np.argmin(costs))

        record = {
            "policies": candidate_policies,
            "tips": tip,
            "hits": hit,
            "costs": costs,
            "success_rate": float(((dist < R_SUCCESS) & ~hit).mean()),
            "best_dist": float(dist.min()),
            "mean_dist": float(dist.mean()),
            "hit_rate": float(hit.mean()),
            "spread": float(std.mean()),
            "mean_after": mean.copy(),
            "std_after": std.copy(),
            "round_best_policy": candidate_policies[j].copy(),
            "round_best_cost": float(costs[j]),
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

def evaluate(history, cfg, target, means, rng_eval, n=200):
    return [arm.competence(m, cfg, target, rng_eval, n=n) for m in means]

def print_history(history, comps):
    for i, x in enumerate(history):
        print(f"round {i:2d} best dist {x['best_dist']:.3f} spread {x['spread']:.3f} successes {x['success_rate']:.2f}  competence {comps[i]:.2f}")

def checkpoint(cfg, target, init_mean, init_std, seed):
    rng = np.random.default_rng(seed)
    mean, std, history = cem(3, cfg, target, rng, init_mean, init_std, num_iter=CHECKPOINT)
    return {"mean": mean, "std": std, "last": history[-1], "explore_std": np.max([r["std_after"] for r in history], axis=0)}

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
        if (last["best_cost"] < history[-1]["best_cost"]):
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


def run_goals():
    runs = {}
    for i, (name, (target, goal_cfg, mean, std)) in enumerate(GOALS.items()):
        print("New GOAL", name)
        rng = np.random.default_rng(SEED + i)
        rng_eval = np.random.default_rng(SEED + 10_000)
        mean, std, history = cem(n_params=3, cfg=goal_cfg, target=target, rng=rng, mean=mean, std=std)
        comps = evaluate(history, goal_cfg, target,
                        [r["best_policy"] for r in history], rng_eval)
        runs[name] = comps
        print_history(history, comps)

        tail = history[-5:]
        stall = np.mean([r["best_dist_so_far"] for r in tail])
        print(f"{name:15s} stall {stall:.3f}  competence {np.mean(comps[-5:]):.2f} hits {np.mean([r['hit_rate'] for r in tail]):.2f}")

        plot_run(history, target, save="run.png")

    plot_competence(runs, save="competence.png")

def response_gains(ckpt, cfg, target, budget, seeds=RESPONSE_SEEDS):
    def comp(q):
        return arm.competence(q, cfg, target, np.random.default_rng(EVAL_SEED), n=N_EVAL)
 
    base = comp(ckpt["mean"])
    gains, spent = {}, {}
    for r in Responses:
        vals = []
        for s in seeds:
            policy, used = respond(r, ckpt, cfg, target, s, budget)
            vals.append(comp(policy) - base)
        gains[r], spent[r] = np.array(vals), used
    return base, gains, spent
 
 
def response_table(budget):
    print(f"\n=== budget {budget} attempts, {len(RESPONSE_SEEDS)} seeds ===")
    header = (f"{'goal':15s} {'comp':>5s}  "
              + "  ".join(f"{r.value:>13s}" for r in Responses)
              + "   best          expected")
    print(header)
    print("-" * len(header))
 
    wins = {r: 0 for r in Responses}
    for i, (name, (target, goal_cfg, mean, std)) in enumerate(GOALS.items()):
        ckpt = checkpoint(goal_cfg, target, mean, std, seed=SEED + i)
        base, gains, spent = response_gains(ckpt, goal_cfg, target, budget)
 
        best = max(Responses, key=lambda r: (round(gains[r].mean(), 6), -spent[r]))
        if gains[best].mean() <= EPS_ABANDON:
            best = Responses.ABANDON
        wins[best] += 1
 
        cells = "  ".join(
            f"{gains[r].mean():+.2f}±{gains[r].std(ddof=1) / np.sqrt(len(gains[r])):.2f}".rjust(13)
            for r in Responses)

        EXPECTED = {
            "easy": {Responses.ABANDON},
            "unreachable": {Responses.ABANDON},
            "joint_limit": {Responses.SWITCH},
            "local_optimum": {Responses.RESTART, Responses.SWITCH},
            "blocked": {Responses.ABANDON},
            "fixable": {Responses.REOPTIMISE},
            "unfixable": {Responses.ABANDON},
            "limit_blocked": {Responses.ABANDON},
        }

        exp = EXPECTED.get(name, set())
        mark = "ok" if best in exp else "MISMATCH"
        exp_s = "/".join(sorted(r.value for r in exp)) or "?"
        print(f"{name:15s} {base:5.2f}  {cells}   {best.value:12s}  {exp_s} {mark}")
 
    print("\nattempts spent: " + ", ".join(f"{r.value} {spent[r]}" for r in Responses))
    print("wins:           " + ", ".join(f"{r.value} {wins[r]}/{len(GOALS)}" for r in Responses))
 
def main() -> None:
    # run_goals()
    for budget in BUDGETS:
        response_table(budget)
