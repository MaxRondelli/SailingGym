import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pettingzoo import ParallelEnv


class MultiAgentSailingZoo(ParallelEnv):
    metadata = {"render_modes": ["rgb_array", "human"], "name": "sailing_v1"}

    def __init__(self, field_size=400, max_steps=250, variable_wind=True, render_mode=None):
        super().__init__()

        self.field_size = field_size
        self.render_mode = render_mode
        self.variable_wind = variable_wind

        # Parametri Fisici
        self.max_speed = 15.0
        self.target_radius = 20.0
        self.boat_radius = 5.0
        self.dt = 1.0
        self.max_steps = max_steps

        # Agenti
        self.possible_agents = ["boat_0", "boat_1"]
        self.agents = self.possible_agents[:]

        # Spazi
        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self._act_space = spaces.Discrete(3)
        self.observation_spaces = {agent: self._obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._act_space for agent in self.possible_agents}

        # Variabili Stato
        self.boat_states = {}
        self.target = []
        self.wind_direction = None
        self.wind_speed = None
        self.step_count = 0
        self.trajectories = {a: [] for a in self.agents}
        self.winner = None

        # Accumulatori per le statistiche
        self.stat_cumulative_vmg = {}
        self.stat_cumulative_polar = {}
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

        # Generazione target (scaled with field size)
        self.target = np.array([
            self.np_random.uniform(self.field_size - 200, self.field_size - 200),
            self.np_random.uniform(self.field_size - 50, self.field_size - 50)
        ])

        # Vento
        self.wind_direction = np.pi / 2
        self.wind_speed = self.np_random.uniform(10, 18)
        self.wind_change_steps = 25

        # Barche
        start_x = self.np_random.uniform(100, 300)
        start_y = self.np_random.uniform(50, 50)

        self.boat_states = {}
        self.trajectories = {a: [] for a in self.agents}
        self.previous_distances = {}
        self.best_distances = {}

        heading = self.np_random.uniform(0, 2 * np.pi)

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
            self.trajectories[agent].append(
                np.array([self.boat_states[agent]['x'], self.boat_states[agent]['y']]))

            pos = np.array([self.boat_states[agent]['x'], self.boat_states[agent]['y']])
            dist = np.linalg.norm(pos - self.target)

            self.stat_initial_dist[agent] = dist
            self.best_distances[agent] = dist

            self.stat_total_dist[agent] = 0.0
            self.stat_cumulative_vmg[agent] = 0.0
            self.stat_cumulative_polar[agent] = 0.0

        target_x = self.target[0]

        x_distances = {}
        for agent in self.agents:
            boat_x = self.boat_states[agent]['x']
            dist_x = abs(target_x - boat_x)
            x_distances[agent] = dist_x

        min_dist_x = min(x_distances.values())

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

        for agent in self.agents:
            if agent not in actions:
                continue
            action = actions[agent]
            state = self.boat_states[agent]
            if state['finished']:
                continue

            if action == 0:
                state['heading'] -= np.radians(15)
            elif action == 2:
                state['heading'] += np.radians(15)
            state['heading'] = state['heading'] % (2 * np.pi)

            apparent_wind = self.wind_direction - state['heading']
            current_wind_speed = self.wind_speed

            theoretical_max_speed = self._get_polar_speed(apparent_wind, self.wind_speed)
            self.stat_cumulative_polar[agent] += theoretical_max_speed

            opponents = [opp for opp in self.possible_agents if
                         opp != agent and not self.boat_states[opp]['finished']]

            for opp in opponents:
                opp_state = self.boat_states[opp]
                dx = state['x'] - opp_state['x']
                dy = state['y'] - opp_state['y']
                dist = np.hypot(dx, dy)

                wind_vec = -np.array([np.cos(self.wind_direction), np.sin(self.wind_direction)])
                pos_vec = np.array([dx, dy])
                proj = np.dot(wind_vec, pos_vec)

                if proj > 0 and dist < (self.boat_radius * 10):
                    cos_angle = np.clip(proj / (dist + 1e-6), -1.0, 1.0)
                    angle_diff = np.arccos(cos_angle)
                    if angle_diff < np.radians(20):
                        current_wind_speed = current_wind_speed * 0.6
                        rewards[agent] -= 2

            state['speed'] = self._get_polar_speed(apparent_wind, current_wind_speed)

            if state['speed'] > state['max_speed_hit']:
                state['max_speed_hit'] = state['speed']

            displacement = state['speed'] * 0.514 * self.dt
            state['x'] += displacement * np.cos(state['heading'])
            state['y'] += displacement * np.sin(state['heading'])

            self.trajectories[agent].append(np.array([state['x'], state['y']]))
            self.stat_total_dist[agent] += displacement

        # Collisioni
        collision = False
        active_agents_list = [a for a in self.agents if not self.boat_states[a]['finished']]
        if len(active_agents_list) >= 2:
            p0 = np.array([self.boat_states[active_agents_list[0]]['x'],
                           self.boat_states[active_agents_list[0]]['y']])
            p1 = np.array([self.boat_states[active_agents_list[1]]['x'],
                           self.boat_states[active_agents_list[1]]['y']])
            if np.linalg.norm(p0 - p1) < (self.boat_radius * 2):
                collision = True

        # Aggiornamento Vento (only if variable_wind is True)
        if self.variable_wind and (self.step_count % self.wind_change_steps) == 0:
            self.wind_change_range = self.np_random.uniform(-np.radians(10), np.radians(10))
            self.wind_direction += self.wind_change_range
            self.wind_direction = max(np.radians(0), min(np.pi, self.wind_direction))

        # Rewards e Stati
        for agent in self.agents:
            state = self.boat_states[agent]
            pos = np.array([state['x'], state['y']])
            dist_to_target = np.linalg.norm(pos - self.target)

            if state['finished']:
                terminations[agent] = True
                continue

            to_target_angle = np.arctan2(self.target[1] - state['y'],
                                         self.target[0] - state['x'])
            angle_error = np.abs(to_target_angle - state['heading'])
            vmg = state['speed'] * np.cos(angle_error)
            self.stat_cumulative_vmg[agent] += vmg
            rewards[agent] = rewards[agent] + (vmg * 0.5)
            rewards[agent] -= 0.05

            apparent_wind = self.wind_direction - state['heading']
            wind_angle_rel = np.abs(np.degrees(apparent_wind)) % 360
            if wind_angle_rel > 180:
                wind_angle_rel = 360 - wind_angle_rel
            if wind_angle_rel < 20:
                rewards[agent] -= 0.5

            displacement = state['speed'] * 0.514 * self.dt
            dx_me = np.cos(state['heading']) * displacement
            dy_me = np.sin(state['heading']) * displacement

            opponents = [opp for opp in self.possible_agents if
                         opp != agent and not self.boat_states[opp]['finished']]

            for opp in opponents:
                displacement_opp = self.boat_states[opp]['speed'] * 0.514 * self.dt
                dx_opp = np.cos(self.boat_states[opp]['heading']) * displacement_opp
                dy_opp = np.sin(self.boat_states[opp]['heading']) * displacement_opp
                det = dx_me * dy_opp - dx_opp * dy_me

                x_opp = self.boat_states[opp]['x']
                y_opp = self.boat_states[opp]['y']

                dx_pos = x_opp - state['x']
                dy_pos = y_opp - state['y']

                opp_pos = np.array([x_opp, y_opp])
                opp_dist_to_target = np.linalg.norm(opp_pos - self.target)

                if det != 0:
                    t = (dx_pos * dy_opp - dy_pos * dx_opp) / det
                    u = (dx_pos * dy_me - dy_pos * dx_me) / det

                    if 0 <= t <= 10 and 0 <= u <= 10:
                        if not state['is_inside']:
                            rewards[agent] -= 3
                        else:
                            rewards[agent] -= 0.05

            if collision:
                rewards[agent] -= 200.0
                state['finished'] = True
                terminations[agent] = True

            if dist_to_target < self.target_radius:
                rewards[agent] += 50.0
                state['finished'] = True
                terminations[agent] = True
                self.winner = agent

                for opp in self.possible_agents:
                    if opp != agent:
                        rewards[opp] -= 50.0
                        terminations[opp] = True
                        if opp in self.boat_states:
                            self.boat_states[opp]['finished'] = True

            if dist_to_target < self.best_distances[agent]:
                self.best_distances[agent] = dist_to_target

            if not (0 <= state['x'] <= self.field_size and 0 <= state['y'] <= self.field_size):
                rewards[agent] -= 200.0
                state['finished'] = True
                terminations[agent] = True

            if self.step_count >= self.max_steps:
                truncations[agent] = True

        # Statistiche finali
        for a in self.possible_agents:
            infos[a]['is_winner'] = (self.winner == a)
            infos[a]['max_speed'] = float(
                self.boat_states[a]['max_speed_hit']) if a in self.boat_states else 0.0

            steps = max(1, self.step_count)
            infos[a]['avg_vmg'] = float(self.stat_cumulative_vmg[a] / steps)

            avg_theo = self.stat_cumulative_polar[a] / steps
            avg_real_speed_knots = (self.stat_total_dist[a] / (steps * self.dt)) / 0.514

            if avg_theo > 0.001:
                infos[a]['polar_efficiency'] = float(avg_real_speed_knots / avg_theo)
            else:
                infos[a]['polar_efficiency'] = 0.0

            dist_traveled = self.stat_total_dist[a]
            if dist_traveled > 1.0:
                infos[a]['path_efficiency'] = float(self.stat_initial_dist[a] / dist_traveled)
            else:
                infos[a]['path_efficiency'] = 0.0

        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

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

        # Vento
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

        # Target
        target_circle = patches.Circle(self.target, self.target_radius,
                                       color='red', alpha=0.3, label='Target')
        self.ax.add_patch(target_circle)

        # Wind Shadow cones
        for agent_id, state in self.boat_states.items():
            if state['finished']:
                continue

            start_point = np.array([state['x'], state['y']])
            wind_vec_x = -np.cos(self.wind_direction)
            wind_vec_y = -np.sin(self.wind_direction)
            shadow_length = self.boat_radius * 10
            cone_width_angle = np.radians(20)
            angle_wind = np.arctan2(wind_vec_y, wind_vec_x)

            x1 = state['x'] + shadow_length * np.cos(angle_wind - cone_width_angle)
            y1 = state['y'] + shadow_length * np.sin(angle_wind - cone_width_angle)
            x2 = state['x'] + shadow_length * np.cos(angle_wind + cone_width_angle)
            y2 = state['y'] + shadow_length * np.sin(angle_wind + cone_width_angle)

            shadow_poly = patches.Polygon([start_point, [x1, y1], [x2, y2]],
                                          closed=True, color='gray', alpha=0.2)
            self.ax.add_patch(shadow_poly)

        # Proiezioni future
        future_steps = 30
        for agent in self.agents:
            if self.boat_states[agent]['finished']:
                continue
            s = self.boat_states[agent]
            step_dist = s['speed'] * 0.514 * self.dt
            ddx = np.cos(s['heading']) * step_dist
            ddy = np.sin(s['heading']) * step_dist
            end_x = s['x'] + ddx * future_steps
            end_y = s['y'] + ddy * future_steps
            self.ax.plot([s['x'], end_x], [s['y'], end_y],
                         linestyle='--', color='gray', alpha=0.5, linewidth=1)

        # Agenti
        colors = ['green', 'orange', 'purple', 'blue']
        for i, (agent_id, state) in enumerate(self.boat_states.items()):
            color = colors[i % len(colors)]

            if agent_id in self.trajectories and len(self.trajectories[agent_id]) > 1:
                traj = np.array(self.trajectories[agent_id])
                self.ax.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', alpha=0.5,
                             linewidth=1)

            boat_size = 15
            boat_points = np.array([
                [boat_size, 0],
                [-boat_size / 2, boat_size / 2],
                [-boat_size / 2, -boat_size / 2]
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
            self.ax.text(state['x'], state['y'] + 10, agent_id, fontsize=8, color=color,
                         weight='bold', ha='center')

        wind_label = "Variable" if self.variable_wind else "Fixed"
        plt.suptitle(f"STEP: {self.step_count} | Wind: {wind_label}", y=0.96, fontsize=12,
                     weight='bold')
        self.ax.legend(loc='lower right', fontsize=8)

        self.fig.canvas.draw()
        image = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]

        plt.close(self.fig)
        self.fig = None
        return image

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
