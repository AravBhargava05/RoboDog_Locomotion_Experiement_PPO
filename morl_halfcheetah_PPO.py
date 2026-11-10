"""HalfCheetah PPO sweep: forward progress subject to an effort tradeoff.

python morl_halfcheetah_PPO.py --smoke-test
python morl_halfcheetah_PPO.py --output-dir runs/full
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch


ENV_ID = "mo-halfcheetah-v5"


@dataclass(frozen=True)
class Config:
    energy_weights: tuple[float, ...] = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    eval_episodes: int = 10
    max_episode_steps: int = 1000
    baseline_speed_frac: float = 0.85
    seed: int = 0
    video_episodes: int = 2
    skip_video: bool = False
    torch_threads: int = 1
    smoke_test: bool = False

    def validate(self) -> None:
        for name in (
            "total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs",
            "eval_episodes", "max_episode_steps", "video_episodes", "torch_threads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.energy_weights or any(
            not math.isfinite(w) or w < 0 for w in self.energy_weights
        ):
            raise ValueError("energy weights must be finite, nonnegative, and nonempty")
        if len(set(self.energy_weights)) != len(self.energy_weights):
            raise ValueError("energy weights must be distinct")
        if not 0 < self.baseline_speed_frac <= 1:
            raise ValueError("baseline_speed_frac must be in (0, 1]")
        rollout_size = self.n_envs * self.n_steps
        if rollout_size <= 1 or self.batch_size <= 1 or rollout_size % self.batch_size:
            raise ValueError("batch_size must exceed 1 and divide n_envs * n_steps")


def objectives(reward: np.ndarray) -> tuple[float, float]:
    """Read v5's [forward velocity, negative action-squared cost] contract."""
    values = np.asarray(reward, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected two finite reward objectives; got {reward!r}")
    if values[1] > 1e-6:
        raise ValueError("mo-halfcheetah-v5's control-cost objective must be nonpositive")
    return float(values[0]), max(0.0, -float(values[1]))


class ScalarizedHalfCheetah(gym.Wrapper):
    """Convert the objectives to forward - energy_weight * control_cost."""

    def __init__(self, env: gym.Env, energy_weight: float):
        super().__init__(env)
        if not math.isfinite(energy_weight) or energy_weight < 0:
            raise ValueError("energy_weight must be finite and nonnegative")
        self.energy_weight = energy_weight

    def step(self, action):
        obs, vector_reward, terminated, truncated, info = self.env.step(action)
        forward, cost = objectives(vector_reward)
        info = {**info, "forward_objective": forward, "control_cost": cost}
        return obs, forward - self.energy_weight * cost, terminated, truncated, info


def make_training_env(energy_weight: float, max_episode_steps: int) -> gym.Env:
    return ScalarizedHalfCheetah(
        mo_gym.make(ENV_ID, max_episode_steps=max_episode_steps), energy_weight
    )


def evaluate_policy(model: PPO, cfg: Config) -> dict[str, Any]:
    """Measure original vector objectives on the same held-out seeds per policy."""
    env = mo_gym.make(ENV_ID, max_episode_steps=cfg.max_episode_steps)
    episodes = []
    try:
        for episode in range(cfg.eval_episodes):
            seed = cfg.seed + 10_000 + episode
            obs, _ = env.reset(seed=seed)
            forward_total = cost_total = 0.0
            for step in range(cfg.max_episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                forward, cost = objectives(reward)
                forward_total += forward
                cost_total += cost
                if terminated or truncated:
                    break
            episodes.append({
                "seed": seed, "forward": forward_total,
                "control_cost": cost_total, "steps": step + 1,
            })
    finally:
        env.close()
    forward_values = [ep["forward"] for ep in episodes]
    cost_values = [ep["control_cost"] for ep in episodes]
    return {
        "forward_mean": float(np.mean(forward_values)),
        "forward_std": float(np.std(forward_values)),
        "forward_median": float(np.median(forward_values)),
        "control_cost_mean": float(np.mean(cost_values)),
        "control_cost_std": float(np.std(cost_values)),
        "control_cost_median": float(np.median(cost_values)),
        "episodes": episodes,
    }


def select_policies(results: list[dict], baseline_speed_frac: float) -> dict:
    """Minimize measured effort among policies meeting the progress constraint."""
    if not results or not 0 < baseline_speed_frac <= 1:
        raise ValueError("Nonempty results and a fraction in (0, 1] are required")
    for row in results:
        if not all(math.isfinite(row[k]) for k in ("forward_mean", "control_cost_mean")):
            raise ValueError("Selection requires finite measured objectives")
        if row["control_cost_mean"] < 0:
            raise ValueError("Control cost must be nonnegative")
    fast = max(range(len(results)), key=lambda i: results[i]["forward_mean"])
    progress = results[fast]["forward_mean"]
    if progress <= 0:
        return {
            "fast_index": fast, "efficient_index": None, "minimum_forward": None,
            "note": "No policy achieved positive mean forward progress; train longer.",
        }
    minimum = baseline_speed_frac * progress
    eligible = [i for i, row in enumerate(results) if row["forward_mean"] >= minimum]
    efficient = min(eligible, key=lambda i: (
        results[i]["control_cost_mean"], -results[i]["forward_mean"]
    ))
    return {
        "fast_index": fast, "efficient_index": efficient, "minimum_forward": minimum,
        "note": "The same policy can win both categories if no lower-cost policy qualifies.",
    }


def record_policy(model_path: Path, path: Path, cfg: Config) -> None:
    """Stream deterministic rollouts to MP4 without retaining all frames in RAM."""
    import imageio.v2 as imageio

    model = PPO.load(model_path, device="cpu")
    env = mo_gym.make(
        ENV_ID, max_episode_steps=cfg.max_episode_steps, render_mode="rgb_array"
    )
    try:
        fps = env.metadata.get("render_fps", 20)
        with imageio.get_writer(path, fps=fps) as writer:
            for episode in range(cfg.video_episodes):
                obs, _ = env.reset(seed=cfg.seed + 20_000 + episode)
                writer.append_data(env.render())
                for _ in range(cfg.max_episode_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    writer.append_data(env.render())
                    if terminated or truncated:
                        break
    finally:
        env.close()


def plot_tradeoff(results: list[dict], selected: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x_mid = (min(r["forward_mean"] for r in results) +
             max(r["forward_mean"] for r in results)) / 2
    y_mid = (min(r["control_cost_mean"] for r in results) +
             max(r["control_cost_mean"] for r in results)) / 2
    for i, row in enumerate(results):
        label = f"wE={row['energy_weight']:g}"
        if i == selected["fast_index"]:
            label += " · FAST"
        if i == selected["efficient_index"]:
            label += " · EFFICIENT"
        ax.scatter(row["forward_mean"], row["control_cost_mean"], s=65)
        right = row["forward_mean"] >= x_mid
        high = row["control_cost_mean"] >= y_mid
        ax.annotate(label, (row["forward_mean"], row["control_cost_mean"]),
                    xytext=(-8 if right else 8, -10 if high else 8),
                    ha="right" if right else "left", va="top" if high else "bottom",
                    textcoords="offset points", fontsize=8)
    if selected["minimum_forward"] is not None:
        ax.axvline(selected["minimum_forward"], linestyle="--", color="0.5",
                   label="Minimum qualifying progress")
        ax.legend(loc="best")
    ax.set(xlabel="Mean summed forward objective per episode (higher is better)",
           ylabel="Mean summed control cost per episode (lower is better)",
           title="HalfCheetah · forward progress vs control effort")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_experiment(cfg: Config, output_dir: Path) -> dict:
    cfg.validate()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("models", "logs", "plots", "videos"):
        (output_dir / name).mkdir()
    packages = {
        name: version(name) for name in (
            "mo-gymnasium", "gymnasium", "mujoco", "stable-baselines3", "torch", "numpy"
        )
    }
    write_json(output_dir / "config.json", {
        **asdict(cfg), "environment": ENV_ID, "versions": packages,
    })
    torch.set_num_threads(cfg.torch_threads)
    results = []
    for index, weight in enumerate(cfg.energy_weights):
        tag = f"ppo_{index:02d}_wE{weight:g}"
        factory = partial(make_training_env, weight, cfg.max_episode_steps)
        vec_env = VecMonitor(DummyVecEnv([factory for _ in range(cfg.n_envs)]),
                             filename=str(output_dir / "logs" / tag))
        try:
            model = PPO(
                "MlpPolicy", vec_env, seed=cfg.seed, device="cpu",
                n_steps=cfg.n_steps, batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs, learning_rate=3e-4,
                policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                verbose=0,
            )
            print(f"Training wE={weight:g}: {cfg.total_timesteps:,} requested steps", flush=True)
            model.learn(total_timesteps=cfg.total_timesteps)
            model_path = Path("models") / f"{tag}.zip"
            model.save(output_dir / model_path)
            result = {
                "energy_weight": weight, "model": model_path.as_posix(),
                "actual_timesteps": model.num_timesteps, **evaluate_policy(model, cfg),
            }
        finally:
            vec_env.close()
        results.append(result)
        write_json(output_dir / "evaluations.json", {"policies": results})
        print(f"  forward={result['forward_mean']:.2f}, cost={result['control_cost_mean']:.2f}",
              flush=True)
    selected = select_policies(results, cfg.baseline_speed_frac)
    report = {
        "environment": ENV_ID, "smoke_test": cfg.smoke_test,
        "baseline_speed_frac": cfg.baseline_speed_frac,
        "selection": selected, "policies": results, "videos": {},
    }
    plot_tradeoff(results, selected, output_dir / "plots" / "forward_vs_control_cost.png")
    columns = ("energy_weight", "forward_mean", "forward_std", "control_cost_mean",
               "control_cost_std", "actual_timesteps", "model")
    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    # Persist the measurements before rendering so a graphics error cannot lose them.
    write_json(output_dir / "results.json", report)
    if not cfg.skip_video:
        for label, index_key in (("fast", "fast_index"), ("efficient", "efficient_index")):
            index = selected[index_key]
            if index is None:
                continue
            path = Path("videos") / f"{label}.mp4"
            record_policy(output_dir / results[index]["model"], output_dir / path, cfg)
            report["videos"][label] = path.as_posix()
            write_json(output_dir / "results.json", report)
    print(f"Saved experiment to {output_dir}. {selected['note']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--energy-weights", type=float, nargs="+", default=Config.energy_weights)
    parser.add_argument("--timesteps", type=int, default=Config.total_timesteps)
    parser.add_argument("--n-envs", type=int, default=Config.n_envs)
    parser.add_argument("--n-steps", type=int, default=Config.n_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--n-epochs", type=int, default=Config.n_epochs)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-episode-steps", type=int, default=Config.max_episode_steps)
    parser.add_argument("--baseline-speed-frac", type=float, default=Config.baseline_speed_frac)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--video-episodes", type=int, default=Config.video_episodes)
    parser.add_argument("--torch-threads", type=int, default=Config.torch_threads)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny two-policy run to check training, evaluation and recording")
    args = parser.parse_args()
    cfg = Config(
        energy_weights=tuple(args.energy_weights), total_timesteps=args.timesteps,
        n_envs=args.n_envs, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, eval_episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps, baseline_speed_frac=args.baseline_speed_frac,
        seed=args.seed, video_episodes=args.video_episodes, skip_video=args.skip_video,
        torch_threads=args.torch_threads, smoke_test=args.smoke_test,
    )
    if args.smoke_test:
        cfg = Config(
            energy_weights=(0.0, 0.3), total_timesteps=256, n_envs=2, n_steps=64,
            batch_size=64, n_epochs=2, eval_episodes=2, max_episode_steps=64,
            video_episodes=1, seed=args.seed, baseline_speed_frac=args.baseline_speed_frac,
            torch_threads=args.torch_threads, skip_video=args.skip_video, smoke_test=True,
        )
        print("SMOKE TEST: pipeline validation only; these policies are not converged.", flush=True)
    output = args.output_dir or Path("runs/smoke" if args.smoke_test else "runs/full")
    try:
        run_experiment(cfg, output)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\nChoose a new --output-dir for each run.\n")


if __name__ == "__main__":
    main()
