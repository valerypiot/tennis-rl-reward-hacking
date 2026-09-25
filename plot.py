import matplotlib.pyplot as plt
from tennis_env import TennisCourtEnv
from q_learning import train, evaluate

modes = ["seen", "collected"]
results = [evaluate(env, train(env)) for env in [TennisCourtEnv(m) for m in modes]]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, i, title in zip(axes, [0, 1], ["Average reward", "Balls collected"]):
    bars = ax.bar(modes, [r[i] for r in results], color=["tab:red", "tab:green"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_title(title)

fig.suptitle("High reward does not mean the task is done")
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()