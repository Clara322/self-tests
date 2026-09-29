from dataclasses import dataclass, field
import numpy as np

L = np.array([1.0, 0.8, 0.6])
REACH = L.sum()
R_SUCCESS = 0.05

@dataclass
class ArmConfig:
    noise_std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    limits: np.ndarray = field(default_factory=lambda: np.tile([-np.inf, np.inf], (3, 1)))
    obstacle: tuple | None = None # (cx, cy, radius)
    penalty: float = 1.0


def joint_positions(q):
    q = np.asarray(q, dtype=float)
    a = np.cumsum(q, axis=-1)
    steps = np.stack([np.cos(a), np.sin(a)], axis=-1) * L[:, None]
    pts = np.cumsum(steps, axis=-2)
    base = np.zeros(pts.shape[:-2] + (1, 2))
    return np.concatenate([base, pts], axis=-2)

def fingertip(q):
    return joint_positions(q)[..., -1, :]

def lever_arms(q):
    pts = joint_positions(q)
    return np.linalg.norm(pts[..., -1:, :] - pts[..., :3, :], axis=-1)

def mirror(q, target):
    q = np.asarray(q, dtype=float)
    phi = np.arctan2(target[1], target[0])
    return np.stack([2 * phi - q[..., 0], -q[..., 1], -q[..., 2]], axis=-1)

def collides(pts, obstacle):
    if obstacle is None:
        return np.zeros(pts.shape[:-2], dtype=bool)
    c = np.array(obstacle[:2])
    r = obstacle[2]
    A, B = pts[..., :-1, :], pts[..., 1:, :]
    AB = B - A
    t = np.clip(np.sum((c-A)*AB, -1)/np.sum(AB*AB, -1), 0.0, 1.0)
    closest = A + t[...,None]*AB
    return np.any(np.linalg.norm(closest -c, axis=-1) < r, axis=-1)

def execute(q_cmd, cfg: ArmConfig, rng):
    q_cmd = np.asarray(q_cmd, dtype=float)
    q = q_cmd + rng.normal(0.0, 1.0, q_cmd.shape) * cfg.noise_std
    q = np.clip(q, cfg.limits[:, 0], cfg.limits[:, 1])
    pts = joint_positions(q)
    return pts[..., -1, :], collides(pts, cfg.obstacle)

def cost(tip, hit, target, cfg: ArmConfig):
    return np.linalg.norm(tip-target, axis=-1) + cfg.penalty * hit

def success(tip, hit, target):
    return (np.linalg.norm(tip - target, axis=-1) < R_SUCCESS) & ~hit

def competence(q_cmd, cfg: ArmConfig, target, rng_eval, n=20):
    q_rep = np.repeat(np.asarray(q_cmd, dtype=float)[None], n, axis=0)
    tip, hit = execute(q_rep, cfg, rng_eval)
    return success(tip, hit, target).mean()

def pose_reaching(target, d, sign=1.0):
    target = np.asarray(target, float)
    R = np.linalg.norm(target)
    a = (L[0] ** 2 - d ** 2 + R ** 2) / (2 * R)
    h = np.sqrt(max(L[0] ** 2 - a ** 2, 0.0))
    u = target / R
    E = a * u + sign * h * np.array([-u[1], u[0]])
    dv = target - E
    dd = np.linalg.norm(dv)
    a2 = (L[1] ** 2 - L[2] ** 2 + dd ** 2) / (2 * dd)
    h2 = np.sqrt(max(L[1] ** 2 - a2 ** 2, 0.0))
    u2 = dv / dd
    W = E + a2 * u2 + sign * h2 * np.array([-u2[1], u2[0]])
    q0 = np.arctan2(E[1], E[0])
    q1 = np.arctan2((W - E)[1], (W - E)[0]) - q0
    q2 = np.arctan2((target - W)[1], (target - W)[0]) - q0 - q1
    return np.array([q0, q1, q2])
