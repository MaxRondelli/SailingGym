# Multi-Agent Regatta Reinforcement Learning

This project presents a 2D America's Cup match simulator developed as part of the Artificial Intelligence course for the Master's degree in Computer Science at the Università di Bologna, utilizing Multi-Agent Reinforcement Learning (MARL) to solve complex control problems in competitive regatta scenarios. 
By implementing the Proximal Policy Optimization (PPO) algorithm within a Gymnasium and Petting Zoo framework, autonomous agents are trained to navigate a simplified match race environment. 


Gianluca di Giacomo
Alessandro Tomaiuolo
Pietro Sami

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
