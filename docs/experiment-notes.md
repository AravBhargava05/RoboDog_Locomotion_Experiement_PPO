# RoboDog: Mission-Constrained Reinforcement Learning (Speed vs Energy)

This project trains a simulated “robot dog” (MO-HalfCheetah) to run fast while using less energy. Instead of optimizing a single reward, the environment returns a *vector reward* with two objectives:

1) forward progress (speed)
2) energy cost (effort)

I trained multiple policies with different energy penalties and then selected two final behaviors for my portfolio:
- FAST: highest speed
- ENERGY EFFICIENT: lowest energy use while still staying fast enough to be useful (mission constraint)

This mirrors real robotics tradeoffs: you often cannot pick “minimum energy” if it means the robot barely moves, so you define a baseline performance requirement and optimize efficiency subject to that requirement.

---

## What I built

### 1) Convert multi-objective reward into a trainable scalar
The environment gives a 2D reward per step:
- vec_r[0] = forward
- vec_r[1] = energy cost (or sometimes negative cost, depending on the env)

My wrapper turns that into a scalar reward used for PPO training:

reward = forward - wE * energy

where:
- wE is the energy weight (a hyperparameter I sweep)
- energy is forced to be positive (handles env variants where vec_r[1] is negative)

This wrapper is implemented as a Gym wrapper that intercepts step() and replaces the reward while leaving observations/actions unchanged.

---

### 2) Train a sweep of policies (one model per energy weight)
I trained multiple PPO agents with different values of wE:

energy_weights = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)

Each weight produces a different behavior:
- low wE: run aggressively, spend energy freely
- high wE: conserve energy, move more cautiously (or even slow down too much)

To speed up training, I used vectorized environments:
- n_envs = 4 parallel simulators
- VecMonitor for stable logging/rollout management

---

### 3) Evaluate policies on true objectives (not the scalar training reward)
After training each model, I evaluate it using the original vector reward, measuring:

- forward_mean: total forward progress per episode (summed over steps)
- energy_mean: total energy used per episode (summed over steps)

This is important: the selection is based on *measured behavior*, not on which wE “should” be best.

---

### 4) Mission constraint: “energy efficient” must still be fast
In real robotics, you rarely want “minimum energy” if it means the robot stops or crawls. So I used a baseline constraint:

- Define max forward among all policies
- Require the efficient policy to achieve at least a fraction of max speed
  Example: baseline_speed_frac = 0.85 (85% of max speed)

Then select:
- FAST = policy with max forward_mean
- ENERGY EFFICIENT = policy with minimum energy_mean among those with forward_mean >= baseline

This makes the experiment feel realistic and prevents “min energy” from becoming a degenerate policy that just doesn’t move.

---

## Outputs (what the code saves)

### Models
Each trained policy is saved as:
morl_halfcheetah_v5/models/ppo_wE?.zip

### Plot
A speed vs energy scatter plot is saved as:
morl_halfcheetah_v5/plots/pareto_forward_vs_energy.png

This makes the tradeoff visible: different wE values trace out different points on the frontier.

### Videos
The script records deterministic rollout videos for the selected policies:
- FAST video (max speed)
- ENERGY EFFICIENT video (mission-constrained)

Videos are saved to:
morl_halfcheetah_v5/videos/

---

## Example results (from one run)

Chosen models (mission-constrained):
- FAST:        wE1e-04  forward_med=4505.5  energy_med=1420.2  eng/fwd=0.3149
- EFFICIENT:   wE3e-01  forward_med=4194.2  energy_med=1129.1  eng/fwd=0.2690
Baseline: forward >= 85% of max speed

Interpretation:
- The FAST policy is the “sprinter”: highest forward progress, higher energy usage.
- The ENERGY EFFICIENT policy keeps most of the speed but improves energy-per-forward.

---

## Why this matters for robotics

Robots run on limited power. Speed is useful, but efficiency determines:
- battery life
- thermal limits
- actuator wear
- mission endurance

This experiment demonstrates a practical way to handle competing objectives:
1) train multiple policies with different tradeoff weights
2) evaluate on real metrics (speed and energy)
3) choose a mission-appropriate policy using a baseline constraint

That same pattern applies to:
- drones (speed vs battery)
- quadrupeds (stability vs agility vs energy)
- manipulators (precision vs time vs torque/energy)

---

## How to reproduce

1) Install dependencies:
- mo-gymnasium
- gymnasium
- stable-baselines3
- mujoco (and licenses/setups if required)

2) Run training:
python robodog.py

3) Watch outputs:
- plots in morl_halfcheetah_v5/plots/
- videos in morl_halfcheetah_v5/videos/
- models in morl_halfcheetah_v5/models/
