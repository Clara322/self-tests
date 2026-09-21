import matplotlib.pyplot as plt
import numpy as np
from self_tests import arm
from self_tests.goals import GOALS
from self_tests.arm import R_SUCCESS
from self_tests.utils.plots import plot_competence
from self_tests.utils.plots import plot_run

SEED = 0

def cem(n_params, cfg, target, rng, mean, std, batch_size=16, n_elite=4, num_iter=20):
    i = 0
    best_policy, best_cost, best_dist = None, np.inf, np.inf
    history = []
    while (i < num_iter):
        # Sample
        candidate_policies = mean + std * rng.standard_normal((batch_size, n_params))
        # Evaluate policy
        tip, hit = arm.execute(candidate_policies, cfg, rng)
        costs = arm.cost(tip, hit, target, cfg)
        elite_inds = costs.argsort()[:n_elite]
        elite_policies = candidate_policies[elite_inds]
        # Update mean and variance
        mean = elite_policies.mean(axis=0)
        std = np.maximum(elite_policies.std(axis=0), 0.01)

        dist = np.linalg.norm(tip - target, axis=-1)
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
            "round_best_policy": candidate_policies[j].copy(),
            "round_best_cost": float(costs[j]),
        }

        if record["round_best_cost"] < best_cost:
            best_cost = record["round_best_cost"]
            best_policy = record["round_best_policy"].copy()
            best_dist = float(dist[j])

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

def main() -> None:
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