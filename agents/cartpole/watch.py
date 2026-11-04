"""Watch a trained CartPole policy in a window."""

import sys
import time

import gymnasium as gym

from agents.cartpole.model import (
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    greedy_action,
    load_policy_for_inference,
)

EPISODES = 5
STEP_DELAY_SEC = 0.02


def resolve_checkpoint():
    if CHECKPOINT_BEST.is_file():
        return CHECKPOINT_BEST
    if CHECKPOINT_LAST.is_file():
        return CHECKPOINT_LAST
    return None


def main():
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        print(
            "No checkpoint found. "
            "Train first: python -m agents.cartpole.train"
        )
        sys.exit(1)

    model = load_policy_for_inference(checkpoint)
    model.eval()
    print(f"Loaded policy from {checkpoint}")

    env = gym.make("CartPole-v1", render_mode="human")

    print(f"Watching {EPISODES} episodes (close the window to quit early).")

    for episode in range(1, EPISODES + 1):
        state, _info = env.reset()
        score = 0.0

        while True:
            action = greedy_action(model, state)
            state, reward, terminated, truncated, _info = env.step(action)
            score += reward
            time.sleep(STEP_DELAY_SEC)

            if terminated or truncated:
                break

        print(f"Episode {episode}: score {int(score)}")

    env.close()


if __name__ == "__main__":
    main()
