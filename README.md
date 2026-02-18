# Multi-Agent Regatta Reinforcement Learning

This project presents a 2D America's Cup match simulator developed as part of the **Artificial Intelligence** course for the Master's degree in Computer Science at the **Università di Bologna**, utilizing Multi-Agent Reinforcement Learning (MARL) to solve complex control problems in competitive regatta scenarios.

By implementing the **Proximal Policy Optimization (PPO)** algorithm within a [Gymnasium](https://gymnasium.farama.org/) and [PettingZoo](https://pettingzoo.farama.org/) framework, two autonomous agents are trained to navigate a simplified match race environment, competing to reach a target buoy while respecting sailing physics, wind dynamics, and right-of-way rules.

### Authors

- Gianluca di Giacomo
- Alessandro Tomaiuolo
- Pietro Sami

---

## Features

- **Multi-agent parallel environment** built on the PettingZoo `ParallelEnv` API
- **Realistic sailing physics** — polar speed diagram, wind shadow cones, stochastic wind shifts
- **PPO training** via Stable-Baselines3 with SuperSuit vectorized wrappers
- **Web dashboard** (Flask) for configuring, training, simulating, and batch-testing — all from the browser
- **Video replay** — every simulation is rendered to MP4 with trajectories, wind arrows, and shadow cones
- **Batch evaluation** with detailed statistics: win rates, VMG, positional analysis, triple-turn counts, and more

---

## Project Structure

```
progettoAI/
├── app.py              # Flask web server (train / simulate / test endpoints)
├── main.py             # Standalone CLI training script (self-contained env)
├── sailing_env.py      # Reusable PettingZoo parallel environment
├── test.py             # Standalone CLI batch-testing script
├── templates/
│   └── index.html      # Single-page web dashboard
├── static/             # Generated models & simulation videos
├── README.md
└── LICENSE
```

---

## Environment Details

### Overview

`MultiAgentSailingZoo` is a PettingZoo **parallel** environment where two sailboats (`boat_0`, `boat_1`) race upwind toward a target buoy on a 2D field.

### Observation Space

Each agent receives a **14-dimensional** continuous vector (values normalised to `[-1, 1]`):

| Index | Feature                         |
|------:|---------------------------------|
| 0–1   | Own position (x, y)             |
| 2     | Own speed                       |
| 3–4   | Own heading (cos, sin)          |
| 5–6   | Apparent wind angle (cos, sin)  |
| 7–8   | Angle to target (cos, sin)      |
| 9     | Distance to target              |
| 10–11 | Relative opponent position      |
| 12–13 | Opponent heading (cos, sin)     |

### Action Space

Discrete(3):

| Action | Meaning                     |
|-------:|-----------------------------|
| 0      | Turn left (−15°)            |
| 1      | Sail straight               |
| 2      | Turn right (+15°)           |

### Sailing Physics

- **Polar speed diagram** — boat speed depends on the angle between heading and wind; sailing directly into the wind (< 20°) yields zero speed, with optimal speeds around beam reach (≈ 90°).
- **Wind shadow** — each boat casts a 20° cone downwind; an opponent inside the cone suffers a 40% wind-speed reduction.
- **Stochastic wind shifts** — every 25 steps the wind direction changes by a random offset of up to ±10°.

### Reward Structure

| Component                 | Value                          |
|---------------------------|--------------------------------|
| VMG (Velocity Made Good)  | `+0.7 × VMG` per step         |
| Time penalty              | `−0.05` per step               |
| Dead-zone penalty         | `−0.5` when heading < 20° off wind |
| Proximity penalty         | scaled up to `−5.0` when boats are close |
| Illegal overtake          | `−3.0` for outside boat crossing |
| Target reached (1st)      | `+100`                         |
| Target reached (2nd)      | `+50` (with scaled penalty for the leader) |
| Collision                 | `−200`                         |
| Out of bounds             | `−200`                         |
| Personal-best improvement | `+0.5 × improvement` for non-winners |

---

## Installation

### Prerequisites

- Python 3.9+

### Install Dependencies

```bash
pip install flask numpy gymnasium pettingzoo stable-baselines3 supersuit imageio imageio-ffmpeg matplotlib torch
```

---

## Usage

### Web Interface

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

From the dashboard you can:

1. **Configure** — set field size, max steps per episode, and total training timesteps.
2. **Train** — start PPO training with a live progress bar; cancel at any time.
3. **Simulate** — run a single episode with the trained model and watch the generated MP4 video with post-race statistics.
4. **Test** — run *N* episodes for batch evaluation with win rates, VMG, and positional analysis.

### CLI Training (standalone)

```bash
python main.py
```

Trains a PPO agent and saves the model to `ppo_sailing_marl.zip` in the project root.

### CLI Testing (standalone)

```bash
python test.py
```

Loads the saved model and runs 100 episodes, printing a detailed report with win rates, positional conversion rates, VMG, polar efficiency, and path efficiency.

---

## PPO Hyperparameters

The default PPO configuration (used by both `app.py` and `main.py`):

| Parameter        | Value   |
|------------------|---------|
| Learning rate    | 3 × 10⁻⁴ |
| Rollout steps    | 1 024   |
| Batch size       | 128     |
| Epochs per update| 10      |
| Gamma (γ)        | 0.99    |
| GAE lambda       | 0.95    |
| Clip range       | 0.2     |
| Entropy coeff.   | 0.01    |
| Vec. envs        | 8       |
| Reward normalisation | VecNormalize (clip = 10) |

Training is parallelised across 8 vectorised copies of the environment using SuperSuit's `concat_vec_envs_v1`.

---

## Evaluation Metrics

After training, the test suite collects the following per-episode statistics:

- **Win rate** — percentage of episodes ending with a boat reaching the target.
- **VMG (Velocity Made Good)** — average component of speed directed toward the target.
- **Triple turns** — count of three consecutive identical turn actions (an indicator of oscillation / indecision).
- **Positional analysis** — tracks whether each boat started on the *inside* (closer to the target's x-axis) or *outside*, and correlates starting position with win probability.

---

## License

See [LICENSE](LICENSE) for details.
