import torch
import numpy as np
from torch import nn, optim

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr % self.max_size] = data
        else:
            self.storage.append(data)
        self.ptr += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in ind]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),                                   
            torch.stack(actions),                                     
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),  
            torch.stack(next_states),                                 
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)     
        )

    def __len__(self):
        return len(self.storage)


LOG_STD_MIN = -20
LOG_STD_MAX = 2
LOG_2PI = float(np.log(2 * np.pi))

class Actor(nn.Module):

    def __init__(self, N, state_dim):
        super(Actor, self).__init__()
        self.N = N
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_mean = nn.Linear(64, N)
        self.fc_log_std = nn.Linear(64, N)

    def forward(self, x):
        x = x.view(-1, self.state_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        z = mean + std * noise 
        phase = z
        log_prob = -0.5 * (noise.pow(2) + 2 * log_std + LOG_2PI)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  

        return phase, log_prob

class Critic(nn.Module):
    def __init__(self, N, state_dim):
        super(Critic, self).__init__()
        self.N = N
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim + 2 * N, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        x = x.view(-1, self.state_dim)
        a_cat = torch.cat([torch.real(action), torch.imag(action)], dim=1)
        x = torch.cat([x, a_cat], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class SACAgent:
    def __init__(self,
                 N,
                 device,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 max_buffer_size=1e6,
                 alpha=0.5,          
                 auto_entropy_tuning=True,
                 target_entropy=None,
                 lr_alpha=1e-3
                 ):
        self.N = N
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.state_dim = 4 * N * N + 2

        # Actor (stochastic Gaussian policy in phase space)
        self.actor = Actor(N, self.state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Two Critics & Targets 
        self.critic1 = Critic(N, self.state_dim).to(device)
        self.critic2 = Critic(N, self.state_dim).to(device)
        self.critic1_target = Critic(N, self.state_dim).to(device)
        self.critic2_target = Critic(N, self.state_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size)

        # SAC temperature (alpha)
        self.auto_entropy_tuning = auto_entropy_tuning
        if target_entropy is None: #-|A|
            target_entropy = -float(N)
        self.target_entropy = target_entropy

        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=device, requires_grad=True
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=device, requires_grad=False
            )
            self.alpha_optimizer = None

    @torch.no_grad()
    def select_action(self, state, noise=True):

        state = state.to(self.device)
        if noise:
            phase, _ = self.actor.sample(state)   
        else:
            mean, _ = self.actor(state)
            phase = mean

        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        theta  = (theta_real + 1j * theta_imag).to(torch.complex64)
        thetaH = (theta_real - 1j * theta_imag).to(torch.complex64)
        return theta, thetaH

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # replay에서 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)   
        rewards     = rewards.to(self.device)    
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # 현재 temperature
        alpha = self.log_alpha.exp()

        # ── 1) Critic target 계산 ──────────────────────────────
        with torch.no_grad():
            # 다음 상태에서 policy로 action 샘플 및 log_pi 계산
            next_phase, next_log_pi = self.actor.sample(next_states)  # [B,N], [B,1]
            next_theta_real = torch.cos(next_phase)
            next_theta_imag = torch.sin(next_phase)
            next_actions = (next_theta_real + 1j * next_theta_imag).to(torch.complex64)

            target_Q1 = self.critic1_target(next_states, next_actions)
            target_Q2 = self.critic2_target(next_states, next_actions)
            target_Q_min = torch.min(target_Q1, target_Q2)

            # V(s') = min(Q1,Q2) - alpha * log_pi
            target_V = target_Q_min - alpha * next_log_pi
            target_Q = rewards + (1 - dones) * self.gamma * target_V

        # ── 2) Critic1 update ──────────────────────────────────
        current_Q1 = self.critic1(states, actions)
        critic1_loss = nn.MSELoss()(current_Q1, target_Q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # ── 3) Critic2 update ──────────────────────────────────
        current_Q2 = self.critic2(states, actions)
        critic2_loss = nn.MSELoss()(current_Q2, target_Q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ── 4) Actor update ────────────────────────────────────
        # 현재 상태들에서 policy로 action 샘플 및 log_pi 계산
        phase, log_pi = self.actor.sample(states)  # [B,N], [B,1]
        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        pi_actions = (theta_real + 1j * theta_imag).to(torch.complex64)

        Q1_pi = self.critic1(states, pi_actions)
        Q2_pi = self.critic2(states, pi_actions)
        Q_pi = torch.min(Q1_pi, Q2_pi)

        actor_loss = (alpha * log_pi - Q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── 5) Temperature(alpha) update ───────────────────────
        if self.auto_entropy_tuning:
            # log_pi는 detach 없이 사용 (gradient는 log_alpha 쪽으로만)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ── 6) Soft update target critics ──────────────────────
        with torch.no_grad():
            for param, target_param in zip(self.critic1.parameters(),
                                           self.critic1_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(self.critic2.parameters(),
                                           self.critic2_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
