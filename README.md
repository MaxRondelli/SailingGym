# Regatta Simulator — Multi-Agent Sailing RL

A multi-agent reinforcement learning project where two autonomous sailboats compete in a regatta. Agents are trained with **PPO** (Proximal Policy Optimization) using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [PettingZoo](https://pettingzoo.farama.org/) in a custom sailing environment, and the entire workflow (training, simulation, testing) is accessible through a **Flask web interface**.

---

## Features

- **Custom sailing environment** with realistic physics: polar speed diagrams, wind shadow (dirty air), stochastic wind shifts, collision detection, and overtaking penalties.
- **Multi-agent parallel training** — two boats (`boat_0`, `boat_1`) share a single PPO policy and learn competitive racing tactics.
- **Web-based dashboard** to configure parameters, launch training, run simulations, and evaluate models — no command-line needed.
- **Video generation** of race simulations rendered as MP4.
- **Batch testing** with detailed statistics: win rates, VMG, positional advantage analysis (inside/outside start), and more.

---

## Project Structure

```
progettoAI/
├── app.py              # Flask web server (training, simulation, testing endpoints)
├── sailing_env.py      # PettingZoo parallel environment for multi-agent sailing
├── main.py             # Standalone training / experimentation script
├── test.py             # Standalone model validation script (CLI)
├── templates/
│   └── index.html      # Web UI (single-page dashboard)
├── static/             # Generated videos and saved models
└── README.md
```

---

## Environment Details

### Observation Space (14-dimensional, continuous)

| Index | Feature |
|-------|---------|
| 0–1 | Boat position (x, y) normalized by field size |
| 2 | Boat speed / max speed |
| 3–4 | Heading (cos, sin) |
| 5–6 | Apparent wind angle (cos, sin) |
| 7–8 | Angle to target (cos, sin) |
| 9 | Distance to target (normalized) |
| 10–11 | Relative position of opponent (dx, dy) |
| 12–13 | Opponent heading (cos, sin) |

### Action Space (Discrete, 3 actions)

| Action | Effect |
|--------|--------|
| 0 | Turn left (−15°) |
| 1 | Sail straight |
| 2 | Turn right (+15°) |

### Physics & Rules

- **Polar speed diagram** — boat speed depends on the angle between heading and wind direction; sailing directly into the wind (< 20°) yields zero speed.
- **Wind shadow** — a boat downwind of an opponent within a 20° cone loses 40% wind speed.
- **Stochastic wind** — wind direction shifts randomly every 25 steps.
- **Collisions** — boats that come within `2 × boat_radius` are penalized and terminated.
- **Overtaking** — trajectory-intersection detection penalizes boats that cut across from the outside position.

### Reward Shaping

| Component | Value |
|-----------|-------|
| VMG (velocity made good) toward target | `+vmg × 0.7` per step |
| Time penalty | `−0.05` per step |
| Dead-zone penalty (sailing into the wind) | `−0.5` |
| Proximity penalty (near collision) | scaled up to `−5.0` |
| Illegal overtake from outside | `−3.0` |
| First to reach target | `+100.0` |
| Second to reach target | `+50.0` |
| Collision / Out of bounds | `−200.0` |

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

### CLI Testing (standalone)

```bash
python test.py
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `field_size` | 400 | Size of the square sailing field (pixels) |
| `max_steps` | 250 | Maximum steps per episode before timeout |
| `total_timesteps` | 500,000 | Total PPO training timesteps |
| `num_episodes` (test) | 100 | Number of episodes for batch evaluation |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| RL Algorithm | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO) |
| Multi-Agent Framework | [PettingZoo](https://pettingzoo.farama.org/) + [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) |
| Environment | [Gymnasium](https://gymnasium.farama.org/) |
| Web Server | [Flask](https://flask.palletsprojects.com/) |
| Rendering | [Matplotlib](https://matplotlib.org/) |
| Video Export | [imageio](https://imageio.readthedocs.io/) + FFmpeg |
| Deep Learning | [PyTorch](https://pytorch.org/) |

---

## License

This project is provided for educational and research purposes.
