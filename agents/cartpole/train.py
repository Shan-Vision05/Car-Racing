"""Train a CartPole actor-critic with policy gradient and GAE."""

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from agents.cartpole.model import (
    CHECKPOINT_BEST,
    CHECKPOINT_DIR,
    CHECKPOINT_LAST,
    build_actor_critic,
)

torch.manual_seed(42)
np.random.seed(42)

env = gym.make("CartPole-v1")

model = build_actor_critic()
optimizer = torch.optim.Adam(
    [
        {"params": model.actor.parameters(), "lr": 1e-3},
        {"params": model.critic.parameters(), "lr": 1e-3},
    ]
)

GAMMA = 0.99
GAE_LAM = 0.95
EPISODES = 800
EVAL_EPISODES = 20
EVAL_EVERY = 25
BATCH_EPISODES = 8
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
SOLVED_SCORE = 475.0
SOLVED_STREAK = 3

best_eval_score = float("-inf")
solved_streak = 0
stop_early = False
last_loss = None


def discounted_returns(rewards):
    returns = []
    total = 0.0

    for reward in reversed(rewards):
        total = reward + GAMMA * total
        returns.insert(0, total)

    return returns


def compute_gae(rewards, values):
    """Advantage from rewards and a detached value prediction per step."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    value_list = values.detach().tolist()

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * next_value - value_list[step]
        gae = delta + GAMMA * GAE_LAM * gae
        advantages.append(gae)
        next_value = value_list[step]

    advantages.reverse()
    return torch.tensor(advantages, dtype=torch.float32)


def normalize_advantages(advantages):
    if advantages.numel() < 2:
        return advantages * 0.0

    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    if std < 1e-8:
        return advantages - mean

    return (advantages - mean) / (std + 1e-8)


def evaluate_policy(episodes=EVAL_EPISODES):
    scores = []

    for _ in range(episodes):
        state, info = env.reset()
        score = 0.0

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                logits, _ = model(state_tensor)
                action = torch.argmax(logits).item()

            state, reward, terminated, truncated, info = env.step(action)
            score += reward

            if terminated or truncated:
                break

        scores.append(score)

    return float(np.mean(scores))


def run_episode():
    state, info = env.reset()

    log_probs = []
    entropies = []
    values = []
    rewards = []

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, value = model(state_tensor)
        distribution = Categorical(logits=logits)

        action = distribution.sample()
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        values.append(value)

        state, reward, terminated, truncated, info = env.step(action.item())
        rewards.append(float(reward))

        if terminated or truncated:
            break

    values_tensor = torch.stack(values)
    mc_returns = torch.tensor(
        discounted_returns(rewards),
        dtype=torch.float32,
    )
    advantages = compute_gae(rewards, values_tensor)

    return (
        torch.stack(log_probs),
        advantages,
        values_tensor,
        mc_returns,
        torch.stack(entropies),
        len(rewards),
    )


def actor_critic_loss(log_probs, advantages, values, mc_returns, entropies):
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, mc_returns)
    entropy_bonus = -ENTROPY_COEF * entropies.mean()
    return policy_loss + VALUE_COEF * value_loss + entropy_bonus


def apply_update(log_probs, advantages, values, mc_returns, entropies):
    global last_loss

    advantages = normalize_advantages(advantages)
    loss = actor_critic_loss(
        log_probs, advantages, values, mc_returns, entropies
    )
    last_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), MAX_GRAD_NORM)
    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), MAX_GRAD_NORM)
    optimizer.step()


def save_if_best(mean_score):
    global best_eval_score, solved_streak, stop_early

    if mean_score > best_eval_score:
        best_eval_score = mean_score
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_BEST)
        print(f"  New best eval {mean_score:.1f} -> saved {CHECKPOINT_BEST.name}")

    if mean_score >= SOLVED_SCORE:
        solved_streak += 1
    else:
        solved_streak = 0

    if solved_streak >= SOLVED_STREAK:
        print(
            f"  Reached {SOLVED_SCORE}+ eval mean "
            f"{SOLVED_STREAK} times in a row — stopping early."
        )
        stop_early = True


def main():
    global stop_early

    batch_log_probs = []
    batch_advantages = []
    batch_values = []
    batch_returns = []
    batch_entropies = []

    for episode in range(1, EPISODES + 1):
        (
            log_probs,
            advantages,
            values,
            mc_returns,
            entropies,
            train_length,
        ) = run_episode()

        batch_log_probs.append(log_probs)
        batch_advantages.append(advantages)
        batch_values.append(values)
        batch_returns.append(mc_returns)
        batch_entropies.append(entropies)

        if episode % BATCH_EPISODES == 0:
            apply_update(
                torch.cat(batch_log_probs),
                torch.cat(batch_advantages),
                torch.cat(batch_values),
                torch.cat(batch_returns),
                torch.cat(batch_entropies),
            )
            batch_log_probs.clear()
            batch_advantages.clear()
            batch_values.clear()
            batch_returns.clear()
            batch_entropies.clear()

        if episode % EVAL_EVERY == 0:
            mean_score = evaluate_policy()
            save_if_best(mean_score)

            loss_str = f"{last_loss:.3f}" if last_loss is not None else "n/a"
            actor_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Episode {episode:4d} | "
                f"Training length: {train_length:3d} | "
                f"Evaluation mean: {mean_score:.1f} | "
                f"Loss: {loss_str} | "
                f"LR: {actor_lr:.1e}"
            )

            if stop_early:
                break

    if batch_log_probs:
        apply_update(
            torch.cat(batch_log_probs),
            torch.cat(batch_advantages),
            torch.cat(batch_values),
            torch.cat(batch_returns),
            torch.cat(batch_entropies),
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_LAST)
    print(f"Saved final weights to {CHECKPOINT_LAST}")
    if best_eval_score > float("-inf"):
        print(
            f"Best evaluation mean: {best_eval_score:.1f} "
            f"(use {CHECKPOINT_BEST} for watching)"
        )

    env.close()


if __name__ == "__main__":
    main()
