import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from self_tests.arm import REACH, R_SUCCESS, joint_positions

def plot_competence(runs, ax=None, band=(0.4, 0.7), save=None):
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]

    if band:
        ax.axhspan(*band, color="grey", alpha=0.12, zorder=0)
    for name, comps in runs.items():
        ax.plot(comps, marker="o", ms=3, lw=1.8, label=name)

    ax.set_xlabel("round")
    ax.set_ylabel("competence")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    if save:
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
    return ax


def plot_arm(q, target=None, cfg=None, ax=None, color="C0", alpha=1.0, lw=3):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    q = np.atleast_2d(np.asarray(q, float))

    for pose in q:
        pts = joint_positions(pose)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, alpha=alpha, lw=lw, ms=4)

    if target is not None:
        ax.add_patch(Circle(target, R_SUCCESS, fill=False, color="k", lw=1.2))
        ax.plot(*target, "k+", ms=10)
    if cfg is not None and cfg.obstacle is not None:
        ax.add_patch(Circle(cfg.obstacle[:2], cfg.obstacle[2],
                            color="firebrick", alpha=0.35))

    ax.add_patch(Circle((0, 0), REACH, fill=False, color="grey",
                        ls=":", lw=0.8))
    ax.plot(0, 0, "ks", ms=6)
    ax.set_xlim(-REACH - 0.2, REACH + 0.2)
    ax.set_ylim(-REACH - 0.2, REACH + 0.2)
    ax.set_aspect("equal")
    return ax


def plot_run(history, target, cfg=None, every=4, save=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    shown = list(range(0, len(history), every))
    for i in shown:
        shade = 0.25 + 0.75 * i / max(len(history) - 1, 1)
        plot_arm(history[i]["best_policy"], target, cfg, ax=a1,
                 color="C0", alpha=shade, lw=2)
    a1.set_title(f"search mean, rounds {shown[0]}–{shown[-1]} (dark = later)")

    a2.plot([r["best_dist"] for r in history], label="best distance")
    a2.plot([r["mean_dist"] for r in history], label="mean distance")
    a2.plot([r["spread"] for r in history], label="search spread")
    a2.plot([r["success_rate"] for r in history], label="success rate")
    if any(r["hit_rate"] > 0 for r in history):
        a2.plot([r["hit_rate"] for r in history], label="collision rate")
    a2.axhline(R_SUCCESS, color="k", ls=":", lw=0.8)
    a2.set_xlabel("round")
    a2.set_yscale("log")
    a2.legend(fontsize=9)
    a2.grid(alpha=0.25)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_scatter(q, cfg, target, rng, n=300, ax=None):
    from self_tests import arm

    ax = plot_arm(q, target, cfg, ax=ax, alpha=0.5, lw=2)
    tips, _ = arm.execute(np.repeat(np.asarray(q, float)[None], n, 0), cfg, rng)
    ax.plot(tips[:, 0], tips[:, 1], ".", ms=3, color="C3", alpha=0.5)

    pts = joint_positions(q)
    ax.set_xlim(target[0] - 0.5, target[0] + 0.5)
    ax.set_ylim(target[1] - 0.5, target[1] + 0.5)
    for j, label in enumerate(["base", "elbow", "wrist"]):
        d = pts[-1] - pts[j]
        d = d / np.linalg.norm(d) * 0.4
        ax.plot([pts[-1, 0] - d[0], pts[-1, 0] + d[0]],
                [pts[-1, 1] - d[1], pts[-1, 1] + d[1]],
                ls="--", lw=0.8, label=f"{label}→tip")
    ax.legend(fontsize=8)
    return ax