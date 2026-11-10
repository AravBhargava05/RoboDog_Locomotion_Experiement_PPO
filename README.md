# RoboDog · locomotion, speed, and effort

A PPO experiment in HalfCheetah simulation: keep useful forward progress while reducing control effort.

**Python · PPO · MO-Gymnasium · MuJoCo**

![Recorded HalfCheetah locomotion demonstration](assets/locomotion.gif)

[Watch the full demonstration ↗](https://aravsportfolio.com/?project=robot-dog-locomotion) · [Training code](morl_halfcheetah_PPO.py)

## The experiment

1. Train a PPO policy for each effort penalty: **0, 0.03, 0.1, 0.3, 1, 3**.
2. Evaluate every policy on the same held-out seeds using the original vector objectives.
3. Select **FAST** by highest mean forward progress.
4. Select **EFFICIENT** by lowest mean control cost among policies retaining at least **85%** of FAST’s progress.

Training uses `forward − weight × control_cost`. In `mo-halfcheetah-v5`, control cost is the sum of squared actions; it is an effort proxy, not a battery-energy measurement. If no policy moves forward on average, the script reports that rather than assigning an efficiency winner. One policy can win both categories.

## Run

Tested with **Python 3.12** on macOS ARM64. Use a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Quick check: training, evaluation, model reload, plots, and MP4 recording
python morl_halfcheetah_PPO.py --smoke-test

# Full sweep: 2 million requested timesteps per policy, 4 vectorized environments
python morl_halfcheetah_PPO.py --output-dir runs/full
```

The smoke test checks the pipeline; its policies are not converged. Use a new output directory for each run. PPO completes whole rollout batches, so actual timesteps can exceed the requested count; both are recorded.

## Outputs

Each run saves model checkpoints, training logs, per-episode evaluation data, `metrics.csv`, a speed/effort scatter plot, and deterministic **FAST / EFFICIENT** MP4 rollouts. `config.json` records parameters and package versions; `results.json` records measurements and selection. Use `--skip-video` when rendering is unavailable.

The code implements the experiment shown in the portfolio. The original demo checkpoint is not included, so retraining can produce different motion and results. [Historical notes](docs/experiment-notes.md) preserve the earlier write-up; its example scores are not newly reproduced benchmarks.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover the real environment’s reward sign and time limit, the 85% selection rule, invalid configurations, training, checkpoint reload, and repeatable evaluation. See [validation notes](docs/validation.md).

[MO-HalfCheetah environment](https://mo-gymnasium.farama.org/environments/mo-halfcheetah/) · [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
