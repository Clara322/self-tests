from self_tests.arm import R_SUCCESS, REACH, ArmConfig, collides, execute, fingertip, joint_positions, lever_arms, mirror
import numpy as np

def _checks(seed=0):
    rng = np.random.default_rng(seed)

    # 1. Workspace fills a disc of radius 2.4
    q = rng.uniform(-np.pi, np.pi, (20000, 3))
    r = np.linalg.norm(fingertip(q), axis=-1)
    assert r.max() <= REACH + 1e-9 and r.max() > 0.98 * REACH
    print(f"workspace: max reach {r.max():.3f} (limit {REACH})")

    # 2. Noise cloud grows with noise (noise on the elbow)
    q0 = np.array([0.3, 1.0, -0.5])
    for s in [0.01, 0.05, 0.1]:
        cfg = ArmConfig(noise_std=np.array([0.0, s, 0.0]))
        tip, _ = execute(np.repeat(q0[None], 500, 0), cfg, rng)
        print(f"elbow noise {s:.2f} rad -> tip spread {tip.std(0).mean():.4f}")

    # 3. Lever-arm rule: tip moves by d * lever arm
    d = 1e-4
    for i in range(3):
        dq = np.zeros(3); dq[i] = d
        moved = np.linalg.norm(fingertip(q0 + dq) - fingertip(q0))
        assert abs(moved - d * lever_arms(q0)[i]) < 1e-6
    print("lever arms at q0:", np.round(lever_arms(q0), 3))

    # 4. Hidden joint limit: commanded elbow-down pose lands where the clipped pose would
    cfg = ArmConfig(limits=np.array([[-np.inf, np.inf], [0.2, 2.5], [-np.inf, np.inf]]))
    q_cmd = np.array([0.5, -1.0, 0.4])
    tip, _ = execute(q_cmd[None], cfg, rng)
    assert np.allclose(tip[0], fingertip([0.5, 0.2, 0.4]))
    print("hidden limit: command clipped silently")

    # 5. Mirror keeps the distance to the target
    target = np.array([1.2, 0.7])
    assert np.isclose(np.linalg.norm(fingertip(q0) - target),
                      np.linalg.norm(fingertip(mirror(q0, target)) - target))
    print("mirror: distance preserved")

    # 6. Collision check
    obs = (0.6, 0.0, 0.15)
    assert collides(joint_positions([0.0, 0.0, 0.0]), obs)          # straight arm through it
    assert not collides(joint_positions([np.pi / 2, 0.0, 0.0]), obs)
    print("collisions: ok")

    # 7. Fixable vs unfixable elbow noise: best and worst lever arm among poses reaching a target
    q = rng.uniform(-np.pi, np.pi, (400000, 3))
    for tgt in [np.array([1.0, 0.0]), np.array([2.2, 0.0])]:
        ok = np.linalg.norm(fingertip(q) - tgt, axis=-1) < 0.02
        lev = lever_arms(q[ok])[:, 1]
        print(f"target r={np.linalg.norm(tgt):.1f}: {ok.sum()} poses, elbow lever "
              f"{lev.min():.2f}-{lev.max():.2f} -> spread at 0.05 rad "
              f"{0.05 * lev.min():.3f}-{0.05 * lev.max():.3f} (success radius {R_SUCCESS})")

    try:
        import matplotlib.pyplot as plt
        tips = fingertip(rng.uniform(-np.pi, np.pi, (5000, 3)))
        plt.figure(figsize=(4, 4)); plt.scatter(*tips.T, s=1)
        plt.gca().set_aspect("equal"); plt.title("workspace"); plt.savefig("workspace.png", dpi=120)
        print("saved workspace.png")
    except ImportError:
        pass

if __name__ == "__main__":
    _checks()
