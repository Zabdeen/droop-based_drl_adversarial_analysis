import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List

# Hyperparameters for DQN
GAMMA = 0.99
LR = 0.001
MEMORY_SIZE = 1000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# System parameters
N_DG = 3
N_BACKUP_DG = 30
P_backup_gen_capacity = [1000] * N_BACKUP_DG
P_gen_capacity = [6400, 3200, 1500] + P_backup_gen_capacity  # in kW
P_gen_battery = [1500, 0, 200]  # in kW

V_nom_mag = 12470
f_ref = 60

m_p = 0.01
m_q = 0.01

Kp = 0.5
Ki = 2

class DG:
    def __init__(self, index, P_max, V_nom_mag, f_ref):
        self.index = index
        self.P_max = P_max   # maximum power generation
        self.V_nom_mag = V_nom_mag
        self.f_ref = f_ref

        self.P = np.zeros(3)
        self.Q = np.zeros(3)
        self.V_mag = np.full(3, V_nom_mag)
        self.f = f_ref
        self.integral_error_freq = 0
        self.integral_error_volt = np.zeros(3)

    def primary_control(self, droop_params, P_load, Q_load, delta_freq, delta_v):
        m_p, m_q = droop_params
        self.f = self.f_ref - m_p * (np.sum(P_load) / self.P_max) + delta_freq
        self.V_mag = self.V_nom_mag - m_q * (np.sum(Q_load) / self.P_max) + delta_v

    def secondary_control(self, shared_data, Kp, Ki, reconstruct=False):
        if reconstruct:
            neighbor_indices = [(self.index - 1) % N_DG, (self.index + 1) % N_DG]
            neighbor_freqs = [shared_data[i]['f'] for i in neighbor_indices]
            neighbor_voltages = [shared_data[i]['V_mag'] for i in neighbor_indices]

            freq_error = ((neighbor_freqs[0] + neighbor_freqs[1]) / 2) - self.f
            voltage_error = ((neighbor_voltages[0] + neighbor_voltages[1]) / 2) - self.V_mag
        else:
            freq_error = self.f_ref - self.f
            voltage_error = np.sum([shared_data[i]['V_mag'] for i in range(N_DG)], axis=0) / N_DG - self.V_mag

        frequency_error_history[self.index].append(freq_error)
        voltage_error_history[self.index].append(voltage_error)
        self.integral_error_freq += freq_error * dt
        self.integral_error_volt += voltage_error * dt
        delta_freq = Kp * freq_error + Ki * self.integral_error_freq
        delta_v = Kp * voltage_error + Ki * self.integral_error_volt
        return delta_freq, delta_v

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.update_target_model()
        self.losses = []
        self.update_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward
            if not done:
                next_target = self.target_model(next_state).max(0)[0].item()
                target += GAMMA * next_target

            target_f = self.model(state).detach().clone()
            target_f[action] = target

            states.append(state)
            targets_f.append(target_f)

        states = torch.stack(states)
        targets_f = torch.stack(targets_f)

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = nn.MSELoss()(outputs, targets_f)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        self.update_counter += 1
        if self.update_counter % 10 == 0:
            self.update_target_model()

t_sim = 200
dt = 0.1

t = np.arange(0, t_sim, dt)
P_loads_history = []
Q_loads_history = []
P_loads_history_one = []
Q_loads_history_one = []
f_history = np.zeros((N_DG, len(t)))
V_history = np.zeros((N_DG, len(t), 3))
V_peak_history = np.zeros((N_DG, len(t)))
delta_f_history = np.zeros((N_DG, len(t)))
delta_v_history = np.zeros((N_DG, len(t), 3))
delta_v_peak_history = np.zeros((N_DG, len(t)))
P_max_history = np.zeros((N_DG, len(t)))
loading_ratio_history = np.zeros((N_DG, len(t)))
frequency_error_history = [[] for _ in range(N_DG)]
voltage_error_history = [[] for _ in range(N_DG)]
f_nadir_history = np.zeros((N_DG, len(t)))
phase_angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])

DGs = [DG(i, P_gen_capacity[i] + P_gen_battery[i], V_nom_mag, f_ref) for i in range(N_DG)]

total_P_loads = 1000  # kW
total_Q_loads = 20  # Kvar

P_loads = [np.full(3, (total_P_loads * P_gen_capacity[i] / sum(P_gen_capacity))) for i in range(N_DG)]
Q_loads = [np.full(3, (total_Q_loads * P_gen_capacity[i] / sum(P_gen_capacity))) for i in range(N_DG)]

shared_data = [{'P_load': P_loads[i] if i < N_DG else np.zeros(3),
                'Q_load': Q_loads[i] if i < N_DG else np.zeros(3),
                'f': f_ref,
                'V_mag': np.full(3, V_nom_mag)} for i in range(N_DG + N_BACKUP_DG)]

total_active_capacity = sum([sum(P_gen_capacity), sum(P_gen_battery)])
if total_P_loads > total_active_capacity:
    # Distribute the excess load among backup generators
    excess_load = total_P_loads - total_active_capacity
    for i in range(N_BACKUP_DG):
        shared_data[N_DG + i]['P_load'] = np.full(3, excess_load / N_BACKUP_DG)

recorded_states = []

reward_history = np.zeros(len(t))

state_dim = 5 * N_DG
action_dim = 2
agent = DQNAgent(state_dim, action_dim)

stop_manipulation = [False] * N_DG

# Simulation loop
for i, t_step in enumerate(t):

    if t_step >= 10:
      total_P_loads = 2000   # Load change
      total_Q_loads = 40

    P_loads = [(1 + np.random.uniform(-0.01, 0.01)) * np.full(3, (total_P_loads * P_gen_capacity[j] / sum(P_gen_capacity))) for j in range(N_DG)]
    Q_loads = [(1 + np.random.uniform(-0.01, 0.01)) * np.full(3, (total_Q_loads * P_gen_capacity[j] / sum(P_gen_capacity))) for j in range(N_DG)]
    P_loads_history.append(P_loads)
    P_loads_history_one.append([x[0] for x in P_loads])
    Q_loads_history.append(Q_loads)
    Q_loads_history_one.append([x[0] for x in Q_loads])

    total_active_power = sum([DGs[i].P_max for i in range(N_DG)])
    if total_active_power < total_P_loads:
        required_power_from_backup = total_P_loads - total_active_power
        backup_power_per_dg = required_power_from_backup / N_BACKUP_DG
        for i in range(N_DG, N_DG + N_BACKUP_DG):
            P_loads[i] = np.full(3, backup_power_per_dg)

    state = []
    for dg in DGs:
        state.extend([dg.f, *dg.V_mag, int(stop_manipulation[dg.index])])

    # Centralized DQN agent decides the control mode for all DGs
    action = agent.act(state)
    reconstruct = bool(action) # boolean for action

    total_reward = 0

    for dg in DGs:
        delta_freq, delta_v = dg.secondary_control(shared_data, Kp, Ki, reconstruct=reconstruct)
        delta_f_history[dg.index, i] = delta_freq
        delta_v_history[dg.index, i] = np.max(np.abs(delta_v))

        dg.primary_control((m_p, m_q), shared_data[dg.index]['P_load'], shared_data[dg.index]['Q_load'], delta_freq, delta_v)

        shared_data[dg.index]['f'] = dg.f
        shared_data[dg.index]['V_mag'] = dg.V_mag
        shared_data[dg.index]['Q_load'] = Q_loads[dg.index]
        shared_data[dg.index]['P_load'] = P_loads[dg.index]
        f_history[dg.index, i] = dg.f
        V_history[dg.index, i] = dg.V_mag
        P_max_history[dg.index, i] = dg.P_max
        loading_ratio_history[dg.index, i] = np.sum(P_loads[dg.index]) / dg.P_max
        delta_v_peak_history[dg.index, i] = np.max(delta_v_history[dg.index, i])

        reward = -abs((f_ref - dg.f) + (V_nom_mag - dg.V_mag[0]))
        total_reward += reward

    reward_history[i] = total_reward

    next_state = []
    for dg in DGs:
        next_state.extend([dg.f, *dg.V_mag, int(stop_manipulation[dg.index])])

    done = t_step == t[-1]

    agent.remember(state, action, total_reward, next_state, done)
    agent.replay()

    for j, dg in enumerate(DGs):
        V_peak_history[j, i] = np.max(np.abs(dg.V_mag))

plt.figure(figsize=(15, 5))
plt.plot(t, reward_history, label='Centralized Agent')
plt.xlabel('Time (s)')
plt.ylabel('Reward')
plt.title('Reward values for RL agent over time (Load perturbation at t == 10)')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(t, f_history.T)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz) (Load perturbation at t == 10)')
plt.title('Frequency Values')
plt.legend([f'DG {i + 1}' for i in range(N_DG)])
plt.grid()
