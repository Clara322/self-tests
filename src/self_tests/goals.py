from self_tests.arm import ArmConfig, pose_reaching
import numpy as np
ZEROS, ONES = np.zeros(3), np.ones(3)

def noisy(joint, std):
    cfg = ArmConfig()
    cfg.noise_std = cfg.noise_std.copy()
    cfg.noise_std[joint] = std
    return cfg

def clipped(joint, low, high):
    cfg = ArmConfig()
    cfg.limits = cfg.limits.copy()
    cfg.limits[joint] = [low, high]
    return cfg

GOALS = {
    "easy":        (np.array([1.2, 0.7]),  ArmConfig(), ZEROS, ONES),
    "unreachable": (np.array([2.63, 0.0]), ArmConfig(), ZEROS, np.full(3, 0.3)),
    "fixable":   (np.array([1.0, 0.0]), noisy(1, 0.167),
              np.array([0.7151, -1.1156, -2.1309]), np.full(3, 0.3)),
    "unfixable": (np.array([1.0, 0.0]), noisy(0, 0.104),
              np.array([0.7151, -1.1156, -2.1309]), np.full(3, 0.3)),
    "local_optimum": (np.array([1.0, 0.0]), ArmConfig(obstacle=(0.868,0.709, 0.15)), pose_reaching(np.array([1.0, 0.0]), 1.0, sign=+1.0), np.full(3, 0.3)),
    "blocked": (np.array([1.0, 0.0]), ArmConfig(obstacle=(1.0, 0.0, 0.23)), pose_reaching(np.array([1.0, 0.0]), 1.0, sign=+1.0), np.full(3, 0.3)),
    "joint_limit": (np.array([0.8332, 0.0]), clipped(1, -0.787, np.inf), np.array([0.3474, -1.087, -2.702]),np.full(3, 0.3)),
}