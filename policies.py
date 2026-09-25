import random
from tennis_env import TennisCourtEnv, UP, DOWN, LEFT, RIGHT, PICK

# episode runner for a given policy
def run_episode(env, policy, show=False):
    obs, info = env.reset()
    total_reward, balls = 0.0 ,0
    while True:
        action = policy(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        balls += info["picked"]
        if show:
            env.render()
        if terminated or truncated:
            return total_reward, balls

# random policy 
def random_policy(env):
    return random.randint(0, 4)

# greedy policy 
def greedy_collector(env):
    if env.robot in env.balls:
        return PICK
    r, c = env.robot
    br, bc = min(env.balls, key=lambda b: abs(b[0] - r) + abs(b[1] - c))
    if br > r: return DOWN
    if br < r: return UP
    if bc > c: return RIGHT
    return LEFT

# run and print results
for mode in ["seen", "collected"]:
    env = TennisCourtEnv(reward_mode=mode)
    for name, policy in [("random", random_policy), ("greedy collector", greedy_collector)]:
        results = [run_episode(env, policy) for _ in range(200)]
        avg_r = sum(r for r, b in results) / len(results)
        avg_b = sum(b for r, b in results) / len(results)
        print(f"reward={mode:10s} policy={name:17s} avg reward={avg_r:6.2f} avg balls={avg_b:.2f}")