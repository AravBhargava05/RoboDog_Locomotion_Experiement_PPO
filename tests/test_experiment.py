from dataclasses import replace
from pathlib import Path
import sys

import mo_gymnasium as mo_gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morl_halfcheetah_PPO import (
    ENV_ID, Config, evaluate_policy, make_training_env, objectives,
    run_experiment, select_policies,
)


def test_scalarization_matches_real_environment():
    raw = mo_gym.make(ENV_ID, max_episode_steps=8)
    wrapped = make_training_env(0.3, 8)
    try:
        np.testing.assert_array_equal(raw.reset(seed=42)[0], wrapped.reset(seed=42)[0])
        action = np.full(raw.action_space.shape, 0.5, dtype=np.float32)
        obs, reward, terminated, truncated, _ = raw.step(action)
        actual, scalar, term, trunc, info = wrapped.step(action)
        np.testing.assert_array_equal(obs, actual)
        forward, cost = objectives(reward)
        assert cost == pytest.approx(np.square(action).sum())
        assert scalar == pytest.approx(forward - 0.3 * cost)
        assert info["control_cost"] == pytest.approx(cost)
        assert (term, trunc) == (terminated, truncated)
    finally:
        raw.close()
        wrapped.close()


def test_gym_contract_and_time_limit():
    env = make_training_env(0.1, 4)
    try:
        check_env(env)
        env.reset(seed=1)
        for index in range(4):
            _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert not terminated
            assert truncated == (index == 3)
    finally:
        env.close()


@pytest.mark.parametrize("reward", [[1, 2], [1], [np.nan, -1], [0, -np.inf]])
def test_invalid_objectives_fail_loudly(reward):
    with pytest.raises(ValueError):
        objectives(np.array(reward))


def test_selection_rejects_a_slow_low_cost_policy():
    rows = [
        {"forward_mean": 100, "control_cost_mean": 50},
        {"forward_mean": 90, "control_cost_mean": 30},
        {"forward_mean": 80, "control_cost_mean": 1},
    ]
    result = select_policies(rows, 0.85)
    assert result["fast_index"] == 0
    assert result["efficient_index"] == 1
    assert result["minimum_forward"] == 85


def test_nonmoving_policies_do_not_earn_an_efficiency_claim():
    result = select_policies([
        {"forward_mean": -1, "control_cost_mean": 2},
        {"forward_mean": 0, "control_cost_mean": 0},
    ], 0.85)
    assert result["efficient_index"] is None
    assert result["minimum_forward"] is None


@pytest.mark.parametrize("changes", [
    {"energy_weights": ()}, {"energy_weights": (0, -1)},
    {"energy_weights": (float("nan"),)}, {"energy_weights": (0.1, 0.1)},
    {"baseline_speed_frac": 0}, {"baseline_speed_frac": 1.1},
    {"eval_episodes": 0}, {"max_episode_steps": 0}, {"seed": -1},
    {"n_envs": 1, "n_steps": 1}, {"batch_size": 7},
])
def test_bad_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_real_training_save_reload_and_shared_evaluation_seeds(tmp_path):
    cfg = Config(
        energy_weights=(0.0, 0.3), total_timesteps=128,
        n_envs=2, n_steps=32, batch_size=32, n_epochs=1,
        eval_episodes=2, max_episode_steps=16, skip_video=True, smoke_test=True,
    )
    output = tmp_path / "run"
    report = run_experiment(cfg, output)
    assert len(report["policies"]) == 2
    for row in report["policies"]:
        model = PPO.load(output / row["model"], device="cpu")
        repeated = evaluate_policy(model, cfg)
        assert repeated["episodes"] == row["episodes"]
        assert [ep["seed"] for ep in row["episodes"]] == [10_000, 10_001]
        assert all(ep["steps"] == 16 for ep in row["episodes"])
        assert row["actual_timesteps"] >= cfg.total_timesteps
    assert (output / "metrics.csv").is_file()
    assert (output / "plots/forward_vs_control_cost.png").stat().st_size > 1000
    assert (output / "results.json").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(cfg, output)
