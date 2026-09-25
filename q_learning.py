import random
from tennis_env import ACTION_NAMES, TennisCourtEnv

N_ACTIONS = 5

def train(env, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1, seed=0):
    random.seed(seed) 
    Q = {} # list of 5 numbers one per action

    def q_values(state):
        if state not in Q:
            Q[state] = [0.0] * N_ACTIONS
        return Q[state]

    for episode in range(episodes):
        state, info = env.reset()
        while True:
            # choose action
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1) # explore
            else:
                values = q_values(state)
                action = values.index(max(values))

            # action 
            next_state, reward, terminated, truncated, info = env.step(action)
            # 3. Update the Q-value: move it a bit towards (reward + discounted future value)
            future = 0.0 if terminated else max(q_values(next_state))
            target = reward + gamma * future
            q_values(state)[action] += alpha * (target - q_values(state)[action])

            state = next_state
            if terminated or truncated:
                break
    return Q

def evaluate(env, Q, episodes=100, show=False):
    total_reward, total_balls = 0.0, 0
    for _ in range(episodes):
        state, info = env.reset()
        while True:
            values = Q.get(state, [0.0] * N_ACTIONS)
            action = values.index(max(values)) # no exploration now
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_balls += info["picked"]
            if show:
                print("action:", ACTION_NAMES[action], "| reward:",
                round(reward, 2))
                env.render()
            if terminated or truncated:
                break
        show = False # only show the first episode
    return total_reward / episodes, total_balls / episodes