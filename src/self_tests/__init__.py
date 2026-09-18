import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

def evaluate_policy(env_name, theta):
    """Run one episode with a linear policy: action = 1 if theta @ obs > 0 else 0."""
    env = gym.make(env_name)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = 1 if np.dot(theta, obs) > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    return total_reward

def cem(env_name, n_params, batch_size=200, n_elite=40, num_iter=50):
    # Sample from a Gaussian
    # evaluate cross entropy
    # set mean and variance 
    mean = np.zeros(n_params)
    std = np.ones(n_params)
    i = 0
    history = {"iters": [], "mean_rewards": [], "elite_mean_rewards": [], "max_rewards": []}
    while (i < num_iter):
        candidate_policies = mean + std * np.random.randn(batch_size, n_params)
        rewards = np.array([evaluate_policy(env_name, th) for th in candidate_policies])
        elite_inds = rewards.argsort()[-n_elite:]
        elite_policies = candidate_policies[elite_inds]

        history["iters"].append(i)
        history["mean_rewards"].append(rewards.mean())
        history["elite_mean_rewards"].append(rewards[elite_inds].mean())
        history["max_rewards"].append(rewards.max())

        mean = elite_policies.mean(axis=0)
        std = np.maximum(elite_policies.std(axis=0), 0.01)
        i += 1


    return mean, history

def main() -> None:

    best_theta, history = cem('CartPole-v1', n_params=4)
    scores = [evaluate_policy('CartPole-v1', best_theta) for _ in range(100)]
    print(f"Mean: {np.mean(scores):.0f} ± {np.std(scores):.0f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["iters"], history["mean_rewards"], 'b-', alpha=0.7, label='Population mean')
    ax.plot(history["iters"], history["elite_mean_rewards"], 'r-', linewidth=2, label='Elite mean')
    ax.plot(history["iters"], history["max_rewards"], 'g--', alpha=0.5, label='Best in batch')
    ax.axhline(y=500, color='k', linestyle=':', alpha=0.3, label='Max possible (500)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Reward')
    ax.legend()
    plt.show()