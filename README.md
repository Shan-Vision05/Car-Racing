# Gymnasium Agents

Reinforcement-learning agents for [Gymnasium](https://gymnasium.farama.org/), one environment at a time. CartPole is the first. The next one is a car-racing policy, which is why this repo exists.

Each environment lives in its own package under `agents/`. Shared code can move to a common module once a second agent needs it.

## Layout

```
agents/
  cartpole/
    model.py              actor-critic network and checkpoint loading
    train.py              policy-gradient training
    watch.py              render a saved policy
    baselines/            random rollouts, before any learning
    checkpoints/          local weights (best.pt, last.pt), not committed
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch. Run every command below from the repo root.

## CartPole

Random baseline, ten steps:

```bash
python -m agents.cartpole.baselines.random_step
```

Random baseline, twenty episodes, then a plot:

```bash
python -m agents.cartpole.baselines.random_episodes
```

Train the actor-critic. It evaluates every 25 episodes, keeps the best weights, and stops early once the greedy policy holds a mean of 475 for three evals in a row.

```bash
python -m agents.cartpole.train
```

Watch the saved policy. This opens a window and prefers `best.pt`, then `last.pt`. Train first; those files stay on your machine.

```bash
python -m agents.cartpole.watch
```

## Adding an agent

Add `agents/<env>/` with the same split: `model.py` for the network, `train.py` and `watch.py` as entry points, and `checkpoints/` for weights. Keep environment-specific code inside that package.
