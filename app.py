import os
import json
import threading
import time
import uuid

import numpy as np
import imageio
from flask import Flask, render_template, request, jsonify, send_from_directory

from sailing_env import MultiAgentSailingZoo

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

training_state = {
    "running": False,
    "progress": 0,           # 0‑100
    "message": "",
    "done": False,
    "model_path": None,
    "error": None,
    "cancel_requested": False,
    # store params so simulate uses the same ones
    "params": {}
}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    """Start training in a background thread."""
    if training_state["running"]:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.json
    max_steps   = int(data.get("max_steps", 250))
    field_size  = int(data.get("field_size", 400))
    variable_wind = bool(data.get("variable_wind", True))
    total_timesteps = int(data.get("total_timesteps", 500_000))

    training_state.update({
        "running": True,
        "progress": 0,
        "message": "Initializing...",
        "done": False,
        "model_path": None,
        "error": None,
        "cancel_requested": False,
        "params": {
            "max_steps": max_steps,
            "field_size": field_size,
            "variable_wind": variable_wind,
        }
    })

    thread = threading.Thread(
        target=_train_worker,
        args=(max_steps, field_size, variable_wind, total_timesteps),
        daemon=True
    )
    thread.start()
    return jsonify({"status": "started"})


@app.route("/train/status")
def train_status():
    return jsonify({
        "running":  training_state["running"],
        "progress": training_state["progress"],
        "message":  training_state["message"],
        "done":     training_state["done"],
        "error":    training_state["error"],
    })


@app.route("/train/cancel", methods=["POST"])
def cancel_training():
    if not training_state["running"]:
        return jsonify({"error": "No training in progress"}), 400
    training_state["cancel_requested"] = True
    return jsonify({"status": "cancel_requested"})


@app.route("/simulate", methods=["POST"])
def simulate():
    """Run one episode, produce an mp4 and stats."""
    model_path = training_state.get("model_path")
    if not model_path or not os.path.exists(model_path + ".zip"):
        return jsonify({"error": "No trained model found. Train first."}), 400

    params = training_state["params"]
    max_steps     = params.get("max_steps", 250)
    field_size    = params.get("field_size", 400)
    variable_wind = params.get("variable_wind", True)

    from stable_baselines3 import PPO
    model = PPO.load(model_path)

    env = MultiAgentSailingZoo(
        field_size=field_size,
        max_steps=max_steps,
        variable_wind=variable_wind,
        render_mode="rgb_array"
    )
    observations, infos = env.reset()

    frames = [env.render()]
    step = 0
    last_infos = infos

    while env.agents and step < max_steps:
        actions = {}
        for agent_id in env.agents:
            obs = observations[agent_id]
            action, _ = model.predict(obs, deterministic=True)
            actions[agent_id] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        if infos:
            last_infos = infos
        frames.append(env.render())
        step += 1

    # Pad end if there was a winner
    if env.winner:
        for _ in range(15):
            frames.append(frames[-1])

    # Save video
    video_name = f"simulation_{uuid.uuid4().hex[:8]}.mp4"
    video_path = os.path.join(STATIC_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=15)

    env.close()

    # Build stats
    winner = env.winner
    collision = False
    if not winner:
        p0 = np.array([env.boat_states["boat_0"]['x'], env.boat_states["boat_0"]['y']])
        p1 = np.array([env.boat_states["boat_1"]['x'], env.boat_states["boat_1"]['y']])
        if np.linalg.norm(p0 - p1) < (env.boat_radius * 2.1):
            collision = True

    if winner:
        outcome = f"Target reached by {winner}"
    elif collision:
        outcome = "Collision"
    elif step >= max_steps:
        outcome = "Timeout"
    else:
        outcome = "Out of bounds"

    stats = {
        "outcome": outcome,
        "winner": winner,
        "steps": step,
        "agents": {}
    }
    for a in env.possible_agents:
        ai = last_infos.get(a, {})
        stats["agents"][a] = {
            "avg_vmg":          round(ai.get("avg_vmg", 0), 2),
            "polar_efficiency": round(ai.get("polar_efficiency", 0) * 100, 1),
            "path_efficiency":  round(ai.get("path_efficiency", 0) * 100, 1),
            "max_speed":        round(ai.get("max_speed", 0), 1),
            "is_winner":        ai.get("is_winner", False),
        }

    return jsonify({"video_url": f"/static/{video_name}", "stats": stats})


# ── Test state ────────────────────────────────────────────────────────────────
test_state = {
    "running": False,
    "progress": 0,
    "message": "",
    "done": False,
    "error": None,
    "results": None,
}


@app.route("/test", methods=["POST"])
def run_test():
    """Run N test episodes in a background thread."""
    if test_state["running"]:
        return jsonify({"error": "Test already in progress"}), 409

    model_path = training_state.get("model_path")
    if not model_path or not os.path.exists(model_path + ".zip"):
        return jsonify({"error": "No trained model found. Train first."}), 400

    data = request.json
    num_episodes = int(data.get("num_episodes", 100))

    test_state.update({
        "running": True,
        "progress": 0,
        "message": "Starting tests...",
        "done": False,
        "error": None,
        "results": None,
    })

    params = training_state["params"]
    thread = threading.Thread(
        target=_test_worker,
        args=(model_path, num_episodes, params),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"})


@app.route("/test/status")
def test_status():
    return jsonify({
        "running":  test_state["running"],
        "progress": test_state["progress"],
        "message":  test_state["message"],
        "done":     test_state["done"],
        "error":    test_state["error"],
        "results":  test_state["results"],
    })


def _test_worker(model_path, num_episodes, params):
    try:
        from stable_baselines3 import PPO

        model = PPO.load(model_path)

        max_steps     = params.get("max_steps", 250)
        field_size    = params.get("field_size", 400)
        variable_wind = params.get("variable_wind", True)

        env = MultiAgentSailingZoo(
            field_size=field_size,
            max_steps=max_steps,
            variable_wind=variable_wind,
        )

        counts = {
            "wins_boat_0": 0,
            "wins_boat_1": 0,
            "collisions": 0,
            "out_of_bounds": 0,
            "timeouts": 0,
        }

        position_stats = {
            "boat_0": {"inside": 0, "outside": 0, "win_inside": 0, "win_outside": 0},
            "boat_1": {"inside": 0, "outside": 0, "win_inside": 0, "win_outside": 0},
        }

        metrics = {
            "boat_0": {"vmg": [], "polar": [], "path": []},
            "boat_1": {"vmg": [], "polar": [], "path": []},
        }

        for i in range(num_episodes):
            observations, infos = env.reset()

            current_inside_agent = None
            for a in env.possible_agents:
                if env.boat_states[a]["is_inside"]:
                    position_stats[a]["inside"] += 1
                    current_inside_agent = a
                else:
                    position_stats[a]["outside"] += 1

            terminated = False
            truncated = False
            last_infos = infos

            while not (terminated or truncated):
                actions = {}
                for agent_id in env.agents:
                    obs = observations[agent_id]
                    action, _ = model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = action.item()
                    actions[agent_id] = action

                observations, rewards, terminations, truncations, infos = env.step(actions)
                if infos:
                    last_infos = infos
                terminated = all(terminations.values())
                truncated = all(truncations.values())

            # Outcome
            if env.winner:
                if env.winner == "boat_0":
                    counts["wins_boat_0"] += 1
                else:
                    counts["wins_boat_1"] += 1

                if env.winner == current_inside_agent:
                    position_stats[env.winner]["win_inside"] += 1
                else:
                    position_stats[env.winner]["win_outside"] += 1
            else:
                if any(truncations.values()):
                    counts["timeouts"] += 1
                else:
                    p0 = np.array([env.boat_states["boat_0"]["x"], env.boat_states["boat_0"]["y"]])
                    p1 = np.array([env.boat_states["boat_1"]["x"], env.boat_states["boat_1"]["y"]])
                    if np.linalg.norm(p0 - p1) < (env.boat_radius * 2.1):
                        counts["collisions"] += 1
                    else:
                        counts["out_of_bounds"] += 1

            for a in env.possible_agents:
                ai = last_infos.get(a, {})
                metrics[a]["vmg"].append(ai.get("avg_vmg", 0))
                metrics[a]["polar"].append(ai.get("polar_efficiency", 0))
                metrics[a]["path"].append(ai.get("path_efficiency", 0))

            pct = int((i + 1) / num_episodes * 100)
            test_state["progress"] = pct
            test_state["message"] = f"Episode {i+1}/{num_episodes}"

        env.close()

        # Build results
        success_rate = round(
            (counts["wins_boat_0"] + counts["wins_boat_1"]) / max(1, num_episodes) * 100, 1
        )

        agents_stats = {}
        for a in ["boat_0", "boat_1"]:
            ps = position_stats[a]
            agents_stats[a] = {
                "avg_vmg":          round(float(np.mean(metrics[a]["vmg"])), 2),
                "polar_efficiency": round(float(np.mean(metrics[a]["polar"])) * 100, 1),
                "path_efficiency":  round(float(np.mean(metrics[a]["path"])) * 100, 1),
                "start_inside":     ps["inside"],
                "start_outside":    ps["outside"],
                "win_inside":       ps["win_inside"],
                "win_outside":      ps["win_outside"],
            }

        test_state.update({
            "running": False,
            "progress": 100,
            "message": "Tests complete!",
            "done": True,
            "results": {
                "num_episodes":  num_episodes,
                "counts":        counts,
                "success_rate":  success_rate,
                "agents":        agents_stats,
            },
        })

    except Exception as e:
        test_state.update({
            "running": False,
            "progress": 0,
            "message": f"Error: {e}",
            "done": False,
            "error": str(e),
        })


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


# ── Training worker ───────────────────────────────────────────────────────────

def _train_worker(max_steps, field_size, variable_wind, total_timesteps):
    try:
        import supersuit as ss
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
        from stable_baselines3.common.callbacks import BaseCallback

        class ProgressCallback(BaseCallback):
            def __init__(self, total, state_dict):
                super().__init__()
                self.total = total
                self.state = state_dict

            def _on_step(self) -> bool:
                if self.state.get("cancel_requested"):
                    return False
                pct = min(int(self.num_timesteps / self.total * 100), 99)
                self.state["progress"] = pct
                self.state["message"] = f"Training... {self.num_timesteps:,}/{self.total:,} steps"
                return True

        training_state["message"] = "Building environments..."
        training_state["progress"] = 0

        env = MultiAgentSailingZoo(
            field_size=field_size,
            max_steps=max_steps,
            variable_wind=variable_wind
        )
        env = ss.black_death_v3(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=1,
                                     base_class="stable_baselines3")
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
        )

        training_state["message"] = "Training started..."
        cb = ProgressCallback(total_timesteps, training_state)
        model.learn(total_timesteps=total_timesteps, callback=cb)

        if training_state.get("cancel_requested"):
            training_state.update({
                "running": False,
                "progress": 0,
                "message": "Training cancelled.",
                "done": False,
                "error": "cancelled",
                "cancel_requested": False,
            })
            return

        model_path = os.path.join(STATIC_DIR, "ppo_sailing_marl")
        model.save(model_path)

        training_state.update({
            "running": False,
            "progress": 100,
            "message": "Training complete!",
            "done": True,
            "model_path": model_path,
        })

    except Exception as e:
        training_state.update({
            "running": False,
            "progress": 0,
            "message": f"Error: {e}",
            "done": False,
            "error": str(e),
        })


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5000)
