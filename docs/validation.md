# Validation

Checked on November 10, 2026 with Python 3.12 on macOS ARM64 using the versions in `requirements.txt`.

- **20 tests passed**, including real MuJoCo environment transitions, Gymnasium/SB3 interface checks, checkpoint round trips, and the progress-constrained selection rule.
- `--smoke-test` completed a two-policy training/evaluation run and recorded a playable MP4. Neither tiny policy achieved positive mean progress, and the selector correctly left the efficiency category unassigned.
- A longer integration run trained **two policies for 65,536 timesteps each**, evaluated each on **three shared held-out seeds with 1,000-step episodes**, saved both checkpoints, wrote the comparison plot and metrics, and recorded both selected videos.

```bash
python -m pytest -q
python morl_halfcheetah_PPO.py --smoke-test --output-dir runs/smoke
python morl_halfcheetah_PPO.py --timesteps 65536 --energy-weights 0 0.3 \
  --eval-episodes 3 --n-steps 512 --video-episodes 1 --output-dir runs/short
```

| Short integration run | Mean summed forward objective | Mean summed control cost |
| :--- | ---: | ---: |
| `wE = 0` | 153.97 | 1417.70 |
| `wE = 0.3` | 76.17 | 346.28 |

The lower-cost policy did not meet the 85% progress threshold, so `wE = 0` correctly won both categories. This confirms the constraint is enforced, not that an efficiency improvement was found. Each exported video contains 1,001 frames at the environment’s 20 FPS.

These are integration checks, not a convergence study or a reproduction of the historical portfolio scores. The full six-weight, two-million-timestep-per-policy sweep has not been run as part of this repair. Generated checkpoints, logs, and videos are excluded from version control; the original portfolio demonstration remains in the README.
