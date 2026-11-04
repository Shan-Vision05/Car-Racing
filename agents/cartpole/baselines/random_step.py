"""Step CartPole with random actions and print each transition."""

import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    state, info = env.reset()

    print(f"Initial state: {state}")
    print(f"Initial info: {info}")
    print("Taking 10 random actions...")

    for step in range(10):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {step + 1}: Action: {action}, Next State: {next_state}, "
            f"Reward: {reward}, Terminated: {terminated}, "
            f"Truncated: {truncated}, Info: {info}"
        )

        if terminated or truncated:
            print(
                f"Episode ended at step {step + 1} "
                f"(terminated={terminated}, truncated={truncated})"
            )
            break

    env.close()


if __name__ == "__main__":
    main()
