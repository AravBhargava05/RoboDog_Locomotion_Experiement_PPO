import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401

import imageio.v2 as imageio
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


@dataclass
class Config:
    env_ids: Tuple[str, ...] = ("FetchPickAndPlace-v4", "FetchPickAndPlace-v3")
    seed: int = 0

    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.98
    tau: float = 0.05
    buffer_size: int = 1_000_000

    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"

    eval_episodes: int = 50
    max_steps: int = 50

    videos_per_checkpoint: int = 20
    fps: int = 30
    pause_frames_between_eps: int = 8

    video_dir: str = "pickplace/videos"
    model_dir: str = "pickplace/models"


def make_env(env_ids: Tuple[str, ...], seed: int, render_mode: str | None = None):
    last_err = None
    for env_id in env_ids:
        try:
            env = gym.make(env_id, render_mode=render_mode)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env, env_id
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not create any env from {env_ids}. Last error: {last_err}")


def evaluate_success(model: SAC, env, episodes: int, max_steps: int, seed: int) -> Dict[str, float]:
    successes = []
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        ep_ret = 0.0
        done = False
        t = 0
        ep_success = 0.0

        while not done and t < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            t += 1
            if isinstance(info, dict) and "is_success" in info:
                ep_success = float(info["is_success"])

        successes.append(ep_success)
        returns.append(ep_ret)
        lengths.append(t)

    return {
        "success_rate": float(np.mean(successes)),
        "return_mean": float(np.mean(returns)),
        "ep_len_mean": float(np.mean(lengths)),
    }


def record_episodes_mp4(
    model: SAC,
    env_ids: Tuple[str, ...],
    out_path: str,
    seed: int,
    n_episodes: int,
    max_steps: int,
    fps: int,
    pause_frames: int,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env, env_id = make_env(env_ids, seed=seed, render_mode="rgb_array")

    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + 10_000 + ep)
            done = False
            t = 0
            last_frame = None

            while not done and t < max_steps:
                frame = env.render()
                last_frame = frame
                writer.append_data(frame)

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                t += 1

            if pause_frames and last_frame is not None:
                for _ in range(pause_frames):
                    writer.append_data(last_frame)
    finally:
        writer.close()
        env.close()


def main():
    cfg = Config()
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.video_dir, exist_ok=True)

    env, env_id = make_env(cfg.env_ids, seed=cfg.seed)
    print(f"Using env: {env_id}")

    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        buffer_size=cfg.buffer_size,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=cfg.n_sampled_goal,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE
            if cfg.goal_selection_strategy == "future"
            else cfg.goal_selection_strategy,
        ),
        verbose=1,
        seed=cfg.seed,
    )

    pre_path = os.path.join(cfg.video_dir, f"{env_id}_pre_{cfg.videos_per_checkpoint}.mp4")
    print(f"Recording pre-training -> {pre_path}")
    record_episodes_mp4(
        model,
        cfg.env_ids,
        pre_path,
        seed=cfg.seed + 123,
        n_episodes=cfg.videos_per_checkpoint,
        max_steps=cfg.max_steps,
        fps=cfg.fps,
        pause_frames=cfg.pause_frames_between_eps,
    )

    half = cfg.total_timesteps // 2
    print(f"Training first half: {half} steps")
    model.learn(total_timesteps=half)

    mid_path = os.path.join(cfg.video_dir, f"{env_id}_mid_{cfg.videos_per_checkpoint}.mp4")
    print(f"Recording mid-training -> {mid_path}")
    record_episodes_mp4(
        model,
        cfg.env_ids,
        mid_path,
        seed=cfg.seed + 456,
        n_episodes=cfg.videos_per_checkpoint,
        max_steps=cfg.max_steps,
        fps=cfg.fps,
        pause_frames=cfg.pause_frames_between_eps,
    )

    print(f"Training second half: {cfg.total_timesteps - half} steps")
    model.learn(total_timesteps=cfg.total_timesteps - half)

    end_path = os.path.join(cfg.video_dir, f"{env_id}_end_{cfg.videos_per_checkpoint}.mp4")
    print(f"Recording end -> {end_path}")
    record_episodes_mp4(
        model,
        cfg.env_ids,
        end_path,
        seed=cfg.seed + 789,
        n_episodes=cfg.videos_per_checkpoint,
        max_steps=cfg.max_steps,
        fps=cfg.fps,
        pause_frames=cfg.pause_frames_between_eps,
    )

    model_path = os.path.join(cfg.model_dir, f"sac_her_{env_id}.zip")
    model.save(model_path)
    print(f"Saved model: {model_path}")

    eval_stats = evaluate_success(model, env, episodes=cfg.eval_episodes, max_steps=cfg.max_steps, seed=cfg.seed)
    print("\nEval:")
    print(f"  success_rate = {eval_stats['success_rate']:.3f}")
    print(f"  return_mean  = {eval_stats['return_mean']:.1f}")
    print(f"  ep_len_mean  = {eval_stats['ep_len_mean']:.1f}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
