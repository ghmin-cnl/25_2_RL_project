import torch
import numpy as np
from torch import nn, optim

class OUNoise:
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.x0 = x0
        self.X = None  # internal state (np.ndarray)

    def _ensure_state(self, shape):
        if (self.X is None) or (self.X.shape != tuple(shape)):
            if self.x0 is None:
                self.X = np.zeros(shape, dtype=np.float32)
            else:
                self.X = np.broadcast_to(self.x0, shape).astype(np.float32).copy()

    def reset(self, shape=None, x0=None):
        """Reset OU state."""
        if x0 is not None:
            self.x0 = x0
        if shape is None:
            self.X = None
        else:
            self._ensure_state(shape)

    def __call__(self, shape):
        self._ensure_state(shape)
        mu = np.broadcast_to(self.mu, shape).astype(np.float32)
        randn = np.random.randn(*shape).astype(np.float32)
        dx = self.theta * (mu - self.X) * self.dt + self.sigma * np.sqrt(self.dt) * randn
        self.X = self.X + dx
        return self.X


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
            torch.stack(states),                                      # [B, state_dim]
            torch.stack(actions),                                     # [B, N] (complex)
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),  # [B,1]
            torch.stack(next_states),                                 # [B, state_dim]
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)     # [B,1]
        )

    def __len__(self):
        return len(self.storage)

class Actor(nn.Module):
    def __init__(self, N, state_dim):
        super(Actor, self).__init__()
        self.N = N
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, N)

    def forward(self, x):
        x = x.view(-1, self.state_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        phase = self.fc3(x)  # [B, N]

        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        theta  = (theta_real + 1j * theta_imag).to(torch.complex64)
        thetaH = (theta_real - 1j * theta_imag).to(torch.complex64)
        return theta, thetaH

    def get_phase(self, x):
        x = x.view(-1, self.state_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        phase = self.fc3(x)
        return phase

class Critic(nn.Module):
    def __init__(self, N, state_dim):
        super(Critic, self).__init__()
        self.N = N
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim + 2 * N, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        """
        x:      [B, state_dim]
        action: [B, N] (complex)
        """
        x = x.view(-1, self.state_dim)
        a_cat = torch.cat([torch.real(action), torch.imag(action)], dim=1)
        x = torch.cat([x, a_cat], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class DDPGAgent:
    def __init__(self,
                 N,
                 device,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 max_buffer_size=1e6,
                 use_ou_noise=True,
                 ou_rho=0.90,
                 ou_sigma=0.20,
                 ):
        self.N = N
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # state = flatten(4 x N x N) + 2 (prev SNRt, prev SNRc)
        self.state_dim = 4 * N * N + 2

        # Actor & Target
        self.actor = Actor(N, self.state_dim).to(device)
        self.actor_target = Actor(N, self.state_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic & Target (DDPG는 단일 critic)
        self.critic = Critic(N, self.state_dim).to(device)
        self.critic_target = Critic(N, self.state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size)

        # OU Noise (exploration용, phase에 더함)
        self.use_ou_noise = use_ou_noise
        if self.use_ou_noise:
            theta_ou = float(-np.log(max(min(ou_rho, 0.9999), 1e-6)))  # theta = -log(rho)
            self.ou_noise = OUNoise(mu=0.0, theta=theta_ou,
                                    sigma=float(ou_sigma), dt=1.0, x0=None)
            self.ou_noise.reset()
        else:
            self.ou_noise = None

    def reset_noise(self):
        if self.use_ou_noise and (self.ou_noise is not None):
            # shape=None → X=None, 이후 첫 호출에서 batch shape에 맞게 재생성
            self.ou_noise.reset(shape=None)

    @torch.no_grad()
    def select_action(self, state, noise=True):
        state = state.to(self.device)
        phase = self.actor.get_phase(state)  # [B, N]

        # exploration noise (phase 공간)
        if noise and self.use_ou_noise and (self.ou_noise is not None):
            phase_np = phase.cpu().numpy()
            noise_val = self.ou_noise(phase_np.shape)
            phase = torch.from_numpy(phase_np + noise_val).to(self.device)

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
        rewards     = rewards.to(self.device)    # [B,1]
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ── 1) Critic target 계산 ──────────────────────────────
        with torch.no_grad():
            # actor_target에서 next action 얻기 (deterministic policy)
            next_phase = self.actor_target.get_phase(next_states)  # [B, N]
            next_theta_real = torch.cos(next_phase)
            next_theta_imag = torch.sin(next_phase)
            next_actions = (next_theta_real + 1j * next_theta_imag).to(torch.complex64)

            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # ── 2) Critic update ───────────────────────────────────
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── 3) Actor update (매 step) ──────────────────────────
        phase = self.actor.get_phase(states)
        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        current_actions = (theta_real + 1j * theta_imag).to(torch.complex64)

        actor_loss = -self.critic(states, current_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── 4) Soft update targets ─────────────────────────────
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
