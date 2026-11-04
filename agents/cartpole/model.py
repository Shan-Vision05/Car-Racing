from pathlib import Path

import torch
import torch.nn as nn

PACKAGE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"
CHECKPOINT_BEST = CHECKPOINT_DIR / "best.pt"
CHECKPOINT_LAST = CHECKPOINT_DIR / "last.pt"


class ActorCritic(nn.Module):
    """Separate actor and critic so value error cannot drown the policy."""

    def __init__(self, obs_size=4, hidden=128, n_actions=2):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value


def build_actor_critic():
    return ActorCritic()


def is_legacy_sequential_checkpoint(state_dict):
    return "0.weight" in state_dict and "actor.0.weight" not in state_dict


def load_policy_for_inference(checkpoint_path):
    """Load ActorCritic, or an older policy-only Sequential checkpoint."""
    state_dict = torch.load(checkpoint_path, weights_only=True)

    if is_legacy_sequential_checkpoint(state_dict):
        policy = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        policy.load_state_dict(state_dict)
        return policy

    model = build_actor_critic()
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def greedy_action(model, state):
    state_tensor = torch.tensor(state, dtype=torch.float32)

    if isinstance(model, ActorCritic):
        logits, _ = model(state_tensor)
    else:
        logits = model(state_tensor)

    return torch.argmax(logits).item()
