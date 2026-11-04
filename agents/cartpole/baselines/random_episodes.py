"""Run random CartPole episodes and plot how long each one lasts."""

import gymnasium as gym
import matplotlib.pyplot as plt

EPISODES = 20
STEP_LIMIT = 500


def main():
    env = gym.make("CartPole-v1")
    episode_lengths = []

    for episode in range(EPISODES):
        env.reset()
        total_reward = 0.0

        while True:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_lengths.append(total_reward)
        print(f"Episode {episode + 1}: survived {int(total_reward)} steps")

    env.close()

    plt.plot(range(1, EPISODES + 1), episode_lengths, marker="o")
    plt.axhline(STEP_LIMIT, linestyle="--", label=f"{STEP_LIMIT}-step limit")
    plt.xlabel("Episode")
    plt.ylabel("Steps survived")
    plt.title("Random agent playing CartPole")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
