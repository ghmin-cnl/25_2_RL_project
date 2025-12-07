import numpy as np
import torch
from torch import nn, optim

LOG_STD_MIN = -20
LOG_STD_MAX = 2
LOG_2PI = float(np.log(2 * np.pi))

class A2CActor(nn.Module):
    def __init__(self, N, state_dim):
        super(A2CActor, self).__init__()
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

        eps = torch.randn_like(mean)
        phase = mean + std * eps  # [B, N]

        # log_prob of diagonal Gaussian
        log_prob = -0.5 * ((eps ** 2) + 2 * log_std + LOG_2PI)  # [B, N]
        log_prob = log_prob.sum(dim=-1, keepdim=True)           # [B, 1]

        # entropy of diagonal Gaussian
        entropy = 0.5 * (1.0 + LOG_2PI + 2 * log_std)
        entropy = entropy.sum(dim=-1, keepdim=True)             # [B, 1]

        return phase, log_prob, entropy


class ValueCritic(nn.Module):
    def __init__(self, state_dim):
        super(ValueCritic, self).__init__()
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, self.state_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.fc3(x)
        return v


class A2CAgent:
    def __init__(self,
                 N,
                 device,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.005,          
                 value_loss_coef=0.5,
                 entropy_coef=1e-3):

        self.N = N
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # main_RL_onpolicy에서 사용한 state_dim과 동일하게 맞춤
        self.state_dim = 4 * N * N + 2

        # Actor & Critic
        self.actor = A2CActor(N, self.state_dim).to(device)
        self.critic = ValueCritic(self.state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)


    @torch.no_grad()
    def select_action(self, state_batch):
        state_batch = state_batch.to(self.device)

        # policy에서 phase 샘플링
        phase, log_prob, _ = self.actor.sample(state_batch)  # [B, N], [B,1], [B,1]

        # phase -> complex theta
        theta_real = torch.cos(phase)
        theta_imag = torch.sin(phase)
        theta = (theta_real + 1j * theta_imag).to(torch.complex64)
        thetaH = (theta_real - 1j * theta_imag).to(torch.complex64)

        # V(s)
        value = self.critic(state_batch)  # [B, 1]

        return theta, thetaH, log_prob, value

    def update(self, rollout):
        # 텐서 꺼내서 device로 이동
        states = rollout["states"].to(self.device)          # [T, state_dim]
        actions = rollout["actions"].to(self.device)        # [T, N] complex
        rewards = rollout["rewards"].to(self.device)        # [T, 1]
        next_states = rollout["next_states"].to(self.device)# [T, state_dim]
        dones = rollout["dones"].to(self.device)            # [T, 1]

        T = states.size(0)

        # ----- 1) V(s), V(s')로 target / advantage 계산 -----
        with torch.no_grad():
            values = self.critic(states)            # [T,1]
            next_values = self.critic(next_states)  # [T,1]

            targets = rewards + self.gamma * (1.0 - dones) * next_values  # [T,1]
            advantages = targets - values                                  # [T,1]

            # optional: advantage 정규화
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # ----- 2) Critic 업데이트 (value loss) -----
        value_preds = self.critic(states)  # [T,1]
        value_loss = (targets - value_preds).pow(2).mean()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # ----- 3) Actor 업데이트 (policy gradient + entropy bonus) -----
        # 저장된 action(theta)의 phase 복원
        action_real = torch.real(actions)
        action_imag = torch.imag(actions)
        phase_taken = torch.atan2(action_imag, action_real)  # [T, N]

        mean, log_std = self.actor(states)  # [T, N], [T, N]
        std = log_std.exp()

        # log_prob of the *taken* actions (phase_taken)
        # z = (phase_taken - mean) / std
        z = (phase_taken - mean) / (std + 1e-8)
        log_prob = -0.5 * (z.pow(2) + 2 * log_std + LOG_2PI)  # [T, N]
        log_prob = log_prob.sum(dim=-1, keepdim=True)         # [T, 1]

        # entropy (diagonal Gaussian)
        entropy = 0.5 * (1.0 + LOG_2PI + 2 * log_std)         # [T, N]
        entropy = entropy.sum(dim=-1, keepdim=True)           # [T, 1]

        # policy gradient loss (maximize logπ * A + entropy → minimize -(...))
        actor_loss = -(log_prob * advantages.detach() + self.entropy_coef * entropy).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 끝에서 loss 등을 모니터링하고 싶으면 return 해도 되고, 안 해도 됨
        # 여기서는 그냥 끝냄
