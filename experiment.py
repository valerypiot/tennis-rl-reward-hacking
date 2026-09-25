from tennis_env import TennisCourtEnv
from q_learning import train, evaluate


for mode in ["seen", "collected"]:
    env = TennisCourtEnv(reward_mode=mode)
    Q = train(env)
    avg_reward, avg_balls = evaluate(env, Q)
    print(f"Reward mode: {mode:10s} -> average reward {avg_reward:6.2f}, balls collected {avg_balls:.2f} / 3")

# training
print("\n Robot trained with the 'seen' reward ---\n")
env = TennisCourtEnv(reward_mode="seen")
Q = train(env)
evaluate(env, Q, episodes=1, show=True)