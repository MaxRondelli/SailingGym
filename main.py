import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
import imageio
from typing import Optional
import os
from stable_baselines3.common.monitor import Monitor

import functools
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pettingzoo import ParallelEnv

class MultiAgentSailingZoo(ParallelEnv):
    metadata = {"render_modes": ["rgb_array", "human"], "name": "sailing_v1"}

    def __init__(self, field_size=400, render_mode=None):
        super().__init__()

        self.field_size = field_size
        self.render_mode = render_mode

        # Parametri Fisici
        self.max_speed = 15.0
        self.target_radius = 10.0
        self.boat_radius = 5.0
        self.dt = 1.0
        self.max_steps = 250

        # Agenti
        self.possible_agents = ["boat_0", "boat_1"]
        self.agents = self.possible_agents[:]

        # Spazi
        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self._act_space = spaces.Discrete(3) # 0: SX, 1: Dritto, 2: DX
        self.observation_spaces = {agent: self._obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._act_space for agent in self.possible_agents}

        # Variabili Stato
        self.boat_states = {}
        self.target = []
        self.wind_direction = None
        self.wind_speed = None
        self.step_count = 0
        self.trajectories = {a: [] for a in self.agents}
        # self.previous_distances = {a: 0.0 for a in self.agents}
        # self.best_distances = {a: 0.0 for a in self.agents}
        self.winner = None

        # --- Accumulatori per le statistiche ---
        self.stat_cumulative_vmg = {}
        self.stat_cumulative_polar = {} # Somma della velocità massima teorica
        self.stat_total_dist = {}
        self.stat_initial_dist = {}

        self.np_random = None
        self.fig = None
        self.ax = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.winner = None

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        elif self.np_random is None:
            self.np_random = np.random.RandomState()

        # Generazione target
        self.target = np.array([
            self.np_random.uniform(self.field_size - 200, self.field_size - 200),
            self.np_random.uniform(self.field_size - 100, self.field_size - 100)
        ])

        # Vento
        self.wind_direction = np.pi/2
        self.wind_speed = self.np_random.uniform(10, 18)
        self.wind_change_steps = 25

        # Barche
        start_x = self.np_random.uniform(100, 300)
        start_y = self.np_random.uniform(50, 50)

        self.boat_states = {}
        self.trajectories = {a: [] for a in self.agents}
        self.previous_distances = {}

        self.best_distances = {}

        heading = self.np_random.uniform(0, 2*np.pi)

        for i, agent in enumerate(self.agents):
            self.boat_states[agent] = {
                'x': start_x + (i * 20),
                'y': start_y,
                'speed': 0.0,
                'heading': heading,
                'finished': False,
                'max_speed_hit': 0.0,
                'is_inside': False
            }
            self.trajectories[agent].append(np.array([self.boat_states[agent]['x'], self.boat_states[agent]['y']]))

            # Calcolo distanza iniziale
            pos = np.array([self.boat_states[agent]['x'], self.boat_states[agent]['y']])
            dist = np.linalg.norm(pos - self.target)

            self.stat_initial_dist[agent] = dist

            # --- FIX QUI: Assegna il valore iniziale per l'agente corrente ---
            self.best_distances[agent] = dist

            # Azzera accumulatori stats
            self.stat_total_dist[agent] = 0.0
            self.stat_cumulative_vmg[agent] = 0.0
            self.stat_cumulative_polar[agent] = 0.0

        target_x = self.target[0]

        # Calcoliamo la distanza X per ogni agente
        x_distances = {}
        for agent in self.agents:
            boat_x = self.boat_states[agent]['x']
            # Distanza assoluta sull'asse X
            dist_x = abs(target_x - boat_x)
            x_distances[agent] = dist_x
            print(boat_x)

        # Troviamo la distanza minima tra tutti gli agenti
        min_dist_x = min(x_distances.values())

        # Assegniamo True a chi ha la distanza minima (o a entrambi se pari)
        for agent in self.agents:
            if x_distances[agent] == min_dist_x:
                self.boat_states[agent]['is_inside'] = True
            else:
                self.boat_states[agent]['is_inside'] = False

        observations = {a: self._get_single_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        rewards = {a: 0.0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        self.step_count += 1

        # 1. Applicazione Azioni e Fisica
        for agent in self.agents:
            if agent not in actions: continue # Se l'agente è morto/finito, salta

            action = actions[agent]
            state = self.boat_states[agent]

            if state['finished']: continue

            # Rotazione
            if action == 0: state['heading'] -= np.radians(15)
            elif action == 2: state['heading'] += np.radians(15)
            state['heading'] = state['heading'] % (2 * np.pi)

            # Velocità (Polar Diagram)
            apparent_wind = self.wind_direction - state['heading']
            current_wind_speed = self.wind_speed # Vento reale percepito (potrebbe diminuire per ombra)

            # --- Calcolo velocità TEORICA (senza ombra) per le stats ---
            # "Quanto veloce sarei potuto andare?"
            theoretical_max_speed = self._get_polar_speed(apparent_wind, self.wind_speed)
            self.stat_cumulative_polar[agent] += theoretical_max_speed

            # Identifica avversari
            opponents = [opponent for opponent in self.possible_agents if opponent != agent and not self.boat_states[opponent]['finished']]

            # Calcolo del cono d'ombra
            for opp in opponents:
                opp_state = self.boat_states[opp]

                # Calcolo distanza tra due punti
                dx = state['x'] - opp_state['x']
                dy = state['y'] - opp_state['y']
                dist = np.hypot(dx,dy)

                # Calcola l'angolo del vento rispetto al vettore che unisce le barche
                wind_vec = -np.array([np.cos(self.wind_direction), np.sin(self.wind_direction)])
                pos_vec = np.array([dx, dy])

                # Proiezione: positiva (sono in ombra), negativa (non sono in ombra)
                proj = np.dot(wind_vec, pos_vec)

                if proj > 0 and dist < (self.boat_radius * 10): # Controlla se sono nel cono e abbastanza vicino affinchè influisca
                    # Calcolo angolo di deviazione per vedere se sono nel cono d'ombra
                    cos_angle = np.clip(proj / (dist + 1e-6), -1.0, 1.0)
                    angle_diff = np.arccos(cos_angle)
                    if angle_diff < np.radians(20):
                        current_wind_speed = current_wind_speed * 0.6
                        rewards[agent] -= 2

            # Calcolo velocità effettiva
            state['speed'] = self._get_polar_speed(apparent_wind, current_wind_speed)

            if state['speed'] > state['max_speed_hit']:
                state['max_speed_hit'] = state['speed']

            # Movimento
            displacement = state['speed'] * 0.514 * self.dt
            state['x'] += displacement * np.cos(state['heading'])
            state['y'] += displacement * np.sin(state['heading'])

            self.trajectories[agent].append(np.array([state['x'], state['y']]))

            # --- Aggiornamento Stats Cumulative ---
            self.stat_total_dist[agent] += displacement

        # Calcolo Collisioni
        collision = False
        active_agents_list = [a for a in self.agents if not self.boat_states[a]['finished']]
        if len(active_agents_list) >= 2:
            p0 = np.array([self.boat_states[active_agents_list[0]]['x'], self.boat_states[active_agents_list[0]]['y']])
            p1 = np.array([self.boat_states[active_agents_list[1]]['x'], self.boat_states[active_agents_list[1]]['y']])
            if np.linalg.norm(p0 - p1) < (self.boat_radius * 2):
                collision = True

        # Aggiornamento Vento
        if (self.step_count % self.wind_change_steps) == 0:
              self.wind_change_range = self.np_random.uniform(-np.radians(10),np.radians(10))
              self.wind_direction += self.wind_change_range
              self.wind_direction = max(np.radians(0), min(np.pi, self.wind_direction))


        # Calcolo Rewards e Stati
        for agent in self.agents:
            state = self.boat_states[agent]
            pos = np.array([state['x'], state['y']])
            dist_to_target = np.linalg.norm(pos - self.target)

            if state['finished']:
                terminations[agent] = True
                continue


            # VMG Reward
            to_target_angle = np.arctan2(self.target[1] - state['y'], self.target[0] - state['x'])
            angle_error = np.abs(to_target_angle - state['heading'])
            vmg = state['speed'] * np.cos(angle_error)
            self.stat_cumulative_vmg[agent] += vmg
            rewards[agent] = rewards[agent] + (vmg * 0.5)

            # Penalità tempo
            rewards[agent] -= 0.05

            # Penalità Deadzone
            apparent_wind = self.wind_direction - state['heading']
            wind_angle_rel = np.abs(np.degrees(apparent_wind)) % 360
            if wind_angle_rel > 180: wind_angle_rel = 360 - wind_angle_rel
            if wind_angle_rel < 20:
                rewards[agent] -= 0.5

            displacement = state['speed'] * 0.514 * self.dt
            dx_me = np.cos(state['heading']) * displacement
            dy_me = np.sin(state['heading']) * displacement

            for opp in opponents:
                  displacement_opp = self.boat_states[opp]['speed'] * 0.514 * self.dt
                  dx_opp = np.cos(self.boat_states[opp]['heading']) * displacement_opp
                  dy_opp = np.sin(self.boat_states[opp]['heading']) * displacement_opp
                  det = dx_me * dy_opp - dx_opp * dy_me


                  x_opp = self.boat_states[opp]['x']
                  y_opp = self.boat_states[opp]['y']


                  dx_pos = x_opp - state['x']
                  dy_pos = y_opp - state['y']

                  opp_pos = np.array([x_opp,y_opp])
                  opp_dist_to_target = np.linalg.norm(opp_pos - self.target)

                  if det != 0:

                    # numero di step affinche' si scontrano
                    t = (dx_pos * dy_opp - dy_pos * dx_opp ) / det
                    u = (dx_pos * dy_me - dy_pos * dx_me ) / det

                    if 0 <= t <= 10 and 0 <= u <= 10:

                      # scoraggia la barca più lontana a provare a sorpassare
                      if not state['is_inside']:
                          rewards[agent] -= 3
                      else:
                          rewards[agent] -= 0.05


            if collision:
                rewards[agent] -= 200.0
                state['finished'] = True
                terminations[agent] = True

            # Target Raggiunto
            if dist_to_target < self.target_radius:
                rewards[agent] += 50.0
                state['finished'] = True
                terminations[agent] = True
                self.winner = agent

                # Penalità all'avversario
                for opp in self.possible_agents:
                    if opp != agent:
                        rewards[opp] -= 50.0
                        terminations[opp] = True
                        if opp in self.boat_states:
                            self.boat_states[opp]['finished'] = True

            if dist_to_target < self.best_distances[agent]:
              self.best_distances[agent] = dist_to_target

            # Out of bounds
            if not (0 <= state['x'] <= self.field_size and 0 <= state['y'] <= self.field_size):
                rewards[agent] -= 200.0
                state['finished'] = True
                terminations[agent] = True

            # Max steps
            if self.step_count >= self.max_steps:
                truncations[agent] = True


        # --- CALCOLO STATISTICHE FINALI ---
        for a in self.possible_agents:
            # Info base
            infos[a]['is_winner'] = (self.winner == a)
            infos[a]['max_speed'] = float(self.boat_states[a]['max_speed_hit']) if a in self.boat_states else 0.0

            steps = max(1, self.step_count)

            # 1. Avg VMG
            # Nota: stat_cumulative_vmg è accumulato ad ogni step
            infos[a]['avg_vmg'] = float(self.stat_cumulative_vmg[a] / steps)

            # 2. Polar Efficiency (Velocità Reale / Velocità Teorica Polare)
            avg_theo = self.stat_cumulative_polar[a] / steps
            # Velocità media reale = distanza totale percorsa / tempo totale
            avg_real_speed_knots = (self.stat_total_dist[a] / (steps * self.dt)) / 0.514

            if avg_theo > 0.001:
                # Se stat_cumulative_polar sommava nodi, usiamo nodi su nodi
                infos[a]['polar_efficiency'] = float(avg_real_speed_knots / avg_theo)
            else:
                infos[a]['polar_efficiency'] = 0.0

            # 3. Path Efficiency (Distanza Lineare / Distanza Effettiva Percorsa)
            # 1.0 = percorso perfetto in linea retta (impossibile controvento), < 1.0 = zigzag
            dist_traveled = self.stat_total_dist[a]
            if dist_traveled > 1.0: # Evita divisioni con distanze minime
                infos[a]['path_efficiency'] = float(self.stat_initial_dist[a] / dist_traveled)
            else:
                infos[a]['path_efficiency'] = 0.0

        # Output console a fine gara
        # Filtra agenti attivi per il rendering o logiche interne
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        if not self.agents: # Se tutti hanno finito
            reason = "Target Raggiunto" if self.winner else ("Collisione" if collision else "Timeout/Out of Bounds")
            print(f"\n{'='*40}")
            print(f"🏁 FINE REGATA - Step: {self.step_count} | Causa: {reason}")
            for a in self.possible_agents:
                status = "🏆" if self.winner == a else "❌"
                vmg = infos[a]['avg_vmg']
                pol = infos[a]['polar_efficiency'] * 100
                pat = infos[a]['path_efficiency'] * 100
                dist_tot = self.stat_total_dist[a]
                print(f" {a}: {status} | VMG Avg: {vmg:.2f} | Eff. Polare: {pol:.1f}% | Eff. Rotta: {pat:.1f}% | Dist: {dist_tot:.1f}m")
            print(f"{'='*40}\n")

        observations = {a: self._get_single_obs(a) for a in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def _get_polar_speed(self, apparent_wind_angle, wind_speed):
        angle_deg = np.abs(np.degrees(apparent_wind_angle) % 360)
        if angle_deg > 180:
            angle_deg = 360 - angle_deg

        if angle_deg < 20:
            speed_ratio = 0.0
        elif angle_deg < 50:
            speed_ratio = 0.2 + (angle_deg - 20) * 0.02
        elif angle_deg < 90:
            speed_ratio = 0.4 + (angle_deg - 50) * 0.0075
        elif angle_deg < 120:
            speed_ratio = 0.7
        elif angle_deg < 150:
            speed_ratio = 0.7 - (angle_deg - 120) * 0.003
        else:
            speed_ratio = 0.6 - (angle_deg - 150) * 0.005

        return min(speed_ratio * wind_speed, self.max_speed)

    def _get_single_obs(self, agent_id):
        if agent_id not in self.boat_states:
             return np.zeros(14, dtype=np.float32)

        me = self.boat_states[agent_id]
        pos = np.array([me['x'], me['y']])
        dist_to_target = np.linalg.norm(pos - self.target)
        angle_to_target = np.arctan2(self.target[1] - pos[1], self.target[0] - pos[0])
        apparent_wind = self.wind_direction - me['heading']

        opp_id = "boat_1" if agent_id == "boat_0" else "boat_0"
        opp = self.boat_states[opp_id]
        opp_pos = np.array([opp['x'], opp['y']])
        rel_pos = opp_pos - pos

        obs = np.array([
            me['x'] / self.field_size,
            me['y'] / self.field_size,
            me['speed'] / self.max_speed,
            np.cos(me['heading']),
            np.sin(me['heading']),
            np.cos(apparent_wind),
            np.sin(apparent_wind),
            np.cos(angle_to_target),
            np.sin(angle_to_target),
            dist_to_target / (self.field_size * np.sqrt(2)),
            rel_pos[0] / self.field_size,
            rel_pos[1] / self.field_size,
            np.cos(opp['heading']),
            np.sin(opp['heading'])
        ], dtype=np.float32)
        return obs

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        if self.render_mode == 'rgb_array' or self.render_mode == 'human':
            return self._render_frame()

    def _render_frame(self):
        if self.fig is not None:
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, self.field_size)
        self.ax.set_ylim(0, self.field_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # --- 1. DISEGNO DEL VENTO ---
        wind_arrow_center_x = 40
        wind_arrow_center_y = self.field_size - 40
        arrow_length = 30
        dx = arrow_length * np.cos(self.wind_direction)
        dy = arrow_length * np.sin(self.wind_direction)
        wind_degrees = int(np.degrees(self.wind_direction) % 360)

        self.ax.arrow(wind_arrow_center_x, wind_arrow_center_y, -dx, -dy,
                      head_width=10, head_length=10, fc='cyan', ec='blue', alpha=0.8, width=2)
        self.ax.text(wind_arrow_center_x, wind_arrow_center_y + 25,
                     f"Wind: {self.wind_speed:.1f} ({wind_degrees}°)", color='blue',
                     ha='center', fontsize=9, weight='bold')

        # --- 2. Target ---
        target_circle = patches.Circle(self.target, self.target_radius,
                                       color='red', alpha=0.3, label='Target')
        self.ax.add_patch(target_circle)

        # --- DISEGNO DEBUG WIND SHADOW ---

        # Coni d'ombra per capire dove "colpiscono"
        for agent_id, state in self.boat_states.items():
            if state['finished']: continue

            # Calcolo vertice cono (centro barca)
            start_point = np.array([state['x'], state['y']])

            # Calcolo direzione vento (DOVE va il vento)
            wind_vec_x = -np.cos(self.wind_direction)
            wind_vec_y = -np.sin(self.wind_direction)

            shadow_length = self.boat_radius * 10  # Lunghezza ombra
            cone_width_angle = np.radians(20)      # L'angolo usato nella logica step()

            # Calcolo i due lati del triangolo d'ombra
            angle_wind = np.arctan2(wind_vec_y, wind_vec_x)

            x1 = state['x'] + shadow_length * np.cos(angle_wind - cone_width_angle)
            y1 = state['y'] + shadow_length * np.sin(angle_wind - cone_width_angle)

            x2 = state['x'] + shadow_length * np.cos(angle_wind + cone_width_angle)
            y2 = state['y'] + shadow_length * np.sin(angle_wind + cone_width_angle)

            # Disegno poligono
            shadow_poly = patches.Polygon([start_point, [x1, y1], [x2, y2]],
                                          closed=True, color='gray', alpha=0.2)
            self.ax.add_patch(shadow_poly)

        future_steps = 30  # Quanti step in avanti visualizzare

        for agent in self.agents:
            if self.boat_states[agent]['finished']: continue

            s = self.boat_states[agent]

            # Calcolo vettore spostamento per singolo step (coerente con step())
            step_dist = s['speed'] * 0.514 * self.dt
            dx = np.cos(s['heading']) * step_dist
            dy = np.sin(s['heading']) * step_dist

            # Calcolo punto finale della proiezione
            end_x = s['x'] + dx * future_steps
            end_y = s['y'] + dy * future_steps

            # 1. linea tratteggiata (La Proiezione)
            self.ax.plot([s['x'], end_x], [s['y'], end_y],
                         linestyle='--', color='gray', alpha=0.5, linewidth=1)

        # --- 3. Rendering Agenti ---
        colors = ['green', 'orange', 'purple', 'blue'] # Colori per diversi agenti


        for i, (agent_id, state) in enumerate(self.boat_states.items()):
            color = colors[i % len(colors)]

            # Traiettoria
            if agent_id in self.trajectories and len(self.trajectories[agent_id]) > 1:
                traj = np.array(self.trajectories[agent_id])
                self.ax.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', alpha=0.5, linewidth=1)

            # Poligono Barca
            boat_size = 15
            boat_points = np.array([
                [boat_size, 0],
                [-boat_size/2, boat_size/2],
                [-boat_size/2, -boat_size/2]
            ])

            rotation_matrix = np.array([
                [np.cos(state['heading']), -np.sin(state['heading'])],
                [np.sin(state['heading']), np.cos(state['heading'])]
            ])
            rotated_points = boat_points @ rotation_matrix.T
            final_points = rotated_points + np.array([state['x'], state['y']])

            boat = patches.Polygon(final_points, closed=True, color=color,
                                   edgecolor='black', linewidth=1, label=agent_id)
            self.ax.add_patch(boat)

            # Etichetta ID vicino alla barca
            self.ax.text(state['x'], state['y'] + 10, agent_id, fontsize=8, color=color, weight='bold', ha='center')

        # Titolo e Legenda
        plt.suptitle(f"STEP: {self.step_count}", y=0.96, fontsize=12, weight='bold')
        self.ax.legend(loc='lower right', fontsize=8)

        # Output per rgb_array
        self.fig.canvas.draw()
        image = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]

        if self.render_mode == "human":
            plt.show(block=False)
            plt.pause(0.001)

        plt.close(self.fig)
        return image

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

"""## Train e test"""

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

def train():
    # 1. Inizializza Ambiente PettingZoo
    env = MultiAgentSailingZoo()

    env = ss.black_death_v3(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # 3. Concatenazione per SB3 (Parallel processing CPU)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128, #cambiato per parameter sharing
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print("--- Inizio Addestramento (Sailing MARL) ---")
    model.learn(total_timesteps=500000)

    print("--- Salvataggio Modello ---")
    model.save("ppo_sailing_marl")
    print("Modello salvato come 'ppo_sailing_marl.zip'")

train()

import imageio
import numpy as np
from stable_baselines3 import PPO

print("="*70)
print("🎬 MULTI-AGENT SAILING VIDEO")
print("="*70)

print("\n1. Loading model...")
try:
    model = PPO.load("ppo_sailing_marl")
    print("Loaded: ppo_sailing_marl")
except:
    print("No model found! Using random actions for demo.")
    model = None

for i in range(10):
    print(f"\n--- Episode {i+1} ---")

    # Inizializza environment
    env = MultiAgentSailingZoo(render_mode='rgb_array')
    observations, infos = env.reset()

    frames = []
    frames.append(env.render())

    step = 0

    # Loop finché ci sono agenti attivi
    while env.agents and step < 250:
        actions = {}

        # Calcola azioni per gli agenti attivi
        for agent_id in env.agents:
            obs = observations[agent_id]

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Azione casuale se non c'è modello
                action = env.action_space(agent_id).sample()

            actions[agent_id] = action

        # Step dell'ambiente
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Aggiungi frame
        frames.append(env.render())
        step += 1

    # Statistiche finali (prendiamo boat_0 come riferimento)
    ref_agent = "boat_0"
    dist = 0
    if dist < 10.0:
        print(f"   ✓ Target reached (or close)!")
        # Aggiungi qualche frame extra alla fine se vince
        for _ in range(15): frames.append(frames[-1])
    else:
        print(f"   ✗ Final Distance {ref_agent}: {dist:.1f}m")

    # Salvataggio video
    videourl = f'multi_sailing_demo_{i}.mp4'
    print(f"3. Saving video ({len(frames)} frames) to {videourl}...")
    imageio.mimsave(videourl, frames, fps=15)

    env.close()

print("\n" + "="*70)
print("✓ Videos created!")
print("🎥 Download the .mp4 files from the Colab file browser to watch.")

import glob
import ipywidgets as widgets
from IPython.display import Video, display

# Trova i video
video_files = glob.glob("*.mp4")

# Crea una lista di oggetti Video
video_widgets = []
for path in video_files:
    # Creiamo l'oggetto Video
    vid = Video(path, width=600, height=400, embed=True, html_attributes="autoplay muted loop playsinline")

    out = widgets.Output()
    with out:
        display(vid)
    video_widgets.append(out)

# Crea una griglia (GridBox)
# 'repeat(3, 320px)' significa: 3 colonne, larghe 320px l'una
grid = widgets.GridBox(video_widgets, layout=widgets.Layout(grid_template_columns="repeat(5, 400px)"))

display(grid)

"""## Statistiche"""

import numpy as np
from stable_baselines3 import PPO  # ✅ Ora attivo

# --- CARICAMENTO MODELLO ---
try:
    model = PPO.load("ppo_sailing_marl")
    print("✅ Modello 'ppo_sailing_marl' caricato correttamente!")
except Exception as e:
    print(f"❌ Errore nel caricamento del modello: {e}")
    print("Assicurati di aver fatto model.save('ppo_sailing_marl') prima di eseguire questo script.")
    exit()

# 1. Setup Statistiche
num_episodes = 1000

counts = {
    "Vittorie boat_0": 0,
    "Vittorie boat_1": 0,
    "Collisioni": 0,
    "Fuori Campo": 0,
    "Timeout (250 step)": 0
}

# --- NUOVO: Struttura dati espansa per tracciare vittorie per posizione ---
position_stats = {
    "boat_0": {"Inside": 0, "Outside": 0, "Win_Inside": 0, "Win_Outside": 0},
    "boat_1": {"Inside": 0, "Outside": 0, "Win_Inside": 0, "Win_Outside": 0}
}
# -------------------------------------------------------------------------

metrics = {
    "boat_0": {"vmg": [], "polar": [], "path": []},
    "boat_1": {"vmg": [], "polar": [], "path": []}
}

print(f"🚀 Inizio validazione modello su {num_episodes} episodi...")

for i in range(num_episodes):
    observations, infos = env.reset()

    # --- Rilevamento Posizione Iniziale ---
    current_inside_agent = None # Variabile temporanea per questo episodio

    for a in env.possible_agents:
        if env.boat_states[a]['is_inside']:
            position_stats[a]["Inside"] += 1
            current_inside_agent = a # Memorizziamo chi è interno
        else:
            position_stats[a]["Outside"] += 1
    # ---------------------------------------

    terminated = False
    truncated = False
    last_infos = infos

    while not (terminated or truncated):
        actions = {}

        # --- LOGICA DI PREVISIONE PPO ---
        for agent_id in env.agents:
            obs = observations[agent_id]
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            actions[agent_id] = action
        # --------------------------------

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if infos:
            last_infos = infos

        terminated = all(terminations.values())
        truncated = all(truncations.values())

    # --- ASSEGNAZIONE ESITO E STATISTICHE TATTICHE ---
    if env.winner:
        # Conteggio globale
        if env.winner == "boat_0": counts["Vittorie boat_0"] += 1
        else: counts["Vittorie boat_1"] += 1

        # --- NUOVO: Controllo tattico (Ha vinto partendo da dentro o fuori?) ---
        if env.winner == current_inside_agent:
            # Il vincitore era quello partito INTERNO
            position_stats[env.winner]["Win_Inside"] += 1
        else:
            # Il vincitore era quello partito ESTERNO
            position_stats[env.winner]["Win_Outside"] += 1
        # -----------------------------------------------------------------------

    else:
        if any(truncations.values()):
            counts["Timeout (250 step)"] += 1
        else:
            p0 = np.array([env.boat_states["boat_0"]['x'], env.boat_states["boat_0"]['y']])
            p1 = np.array([env.boat_states["boat_1"]['x'], env.boat_states["boat_1"]['y']])
            if np.linalg.norm(p0 - p1) < (env.boat_radius * 2.1):
                counts["Collisioni"] += 1
            else:
                counts["Fuori Campo"] += 1

    # --- RACCOLTA METRICHE ---
    for a in env.possible_agents:
        agent_info = last_infos.get(a, {})
        metrics[a]["vmg"].append(agent_info.get("avg_vmg", 0))
        metrics[a]["polar"].append(agent_info.get("polar_efficiency", 0))
        metrics[a]["path"].append(agent_info.get("path_efficiency", 0))

    if (i + 1) % 10 == 0:
        print(f"✅ Completati {i + 1} episodi...")

# --- REPORT FINALE ---
print("\n" + "="*80)
print(f"{'REPORT VALIDAZIONE PPO':^80}")
print("="*80)

print(f"{'Esito Gara':<25} | {'Frequenza':<10}")
print("-" * 80)
for esito, conteggio in counts.items():
    icona = "🟢" if "Vittorie" in esito else "🔴"
    print(f"{icona} {esito:<22} | {conteggio:<10}")

print("-" * 80)
# --- NUOVA TABELLA CON VITTORIE SPECIFICHE ---
print(f"{'Agente':<10} | {'Start IN':<10} | {'Vittorie (da IN)':<18} | {'Start OUT':<10} | {'Vittorie (da OUT)':<18}")
print("-" * 80)
for a in env.possible_agents:
    s_in = position_stats[a]["Inside"]
    w_in = position_stats[a]["Win_Inside"]
    s_out = position_stats[a]["Outside"]
    w_out = position_stats[a]["Win_Outside"]

    # Calcolo percentuali di conversione (Vittorie / Start)
    perc_win_in = (w_in / s_in * 100) if s_in > 0 else 0.0
    perc_win_out = (w_out / s_out * 100) if s_out > 0 else 0.0

    # Formattazione stringa con percentuale tra parentesi
    str_w_in = f"{w_in} ({perc_win_in:.0f}%)"
    str_w_out = f"{w_out} ({perc_win_out:.0f}%)"

    print(f"{a:<10} | {s_in:<10} | {str_w_in:<18} | {s_out:<10} | {str_w_out:<18}")

print("-" * 80)
print(f"{'Agente':<10} | {'VMG Medio':<12} | {'Eff. Polare':<12} | {'Eff. Rotta':<12}")
print("-" * 80)
for a in env.possible_agents:
    avg_vmg = np.mean(metrics[a]["vmg"])
    avg_polar = np.mean(metrics[a]["polar"]) * 100
    avg_path = np.mean(metrics[a]["path"]) * 100
    print(f"{a:<10} | {avg_vmg:<12.2f} | {avg_polar:>10.1f}% | {avg_path:>10.1f}%")

print("="*80)
success_rate = ((counts["Vittorie boat_0"] + counts["Vittorie boat_1"]) / num_episodes) * 100
print(f"🎯 Percentuale Successo Totale: {success_rate:.1f}%")
print("="*80 + "\n")

