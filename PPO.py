import numpy as np
import torch
from torch import nn, optim

LOG_STD_MIN = -20
LOG_STD_MAX = 2
LOG_2PI = float(np.log(2 * np.pi))


class PPOActor(nn.Module):

    def __init__(self, N, state_dim):
        super(PPOActor, self).__init__()
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

class PPOAgent:
    def __init__(self,
                 N,
                 device,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.005,           # 여기서는 사용 안하지만 인터페이스용
                 use_ou_noise=False,  # on-policy라 사용 안함
                 ou_rho=0.90,
                 ou_sigma=0.20,
                 value_loss_coef=0.5,
                 entropy_coef=1e-3,
                 clip_eps=0.2,
                 gae_lambda=0.95,
                 ppo_epochs=10,
                 mini_batch_size=256):
        self.N = N
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # main_RL_onpolicy에서 사용한 state_dim과 동일하게 맞춤
        self.state_dim = 4 * N * N + 2

        # Actor & Critic
        self.actor = PPOActor(N, self.state_dim).to(device)
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

    def _compute_gae(self, rewards, dones, values, next_values):
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0

        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values  # [T,1]

        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        # 텐서 꺼내서 device로 이동
        states = rollout["states"].to(self.device)          # [T, state_dim]
        actions = rollout["actions"].to(self.device)        # [T, N] complex
        rewards = rollout["rewards"].to(self.device)        # [T, 1]
        next_states = rollout["next_states"].to(self.device)# [T, state_dim]
        dones = rollout["dones"].to(self.device)            # [T, 1]
        old_log_probs = rollout["log_probs"].to(self.device)# [T, 1]
        old_values = rollout["values"].to(self.device)      # [T, 1]

        T = states.size(0)

        # ----- 1) Advantage, Return 계산 (GAE) -----
        with torch.no_grad():
            # old_values는 rollout에서 받은 값 사용
            values = old_values                              # [T,1]
            next_values = self.critic(next_states)           # [T,1]

            advantages, returns = self._compute_gae(
                rewards, dones, values, next_values
            )

            # advantage 정규화
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # ----- 2) PPO update (여러 epoch, minibatch) -----
        # T 개의 step을 하나의 batch로 보고 셔플
        indices = torch.randperm(T, device=self.device)

        for _ in range(self.ppo_epochs):
            for start in range(0, T, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]            # [B, state_dim]
                mb_actions = actions[mb_idx]          # [B, N] complex
                mb_old_log_probs = old_log_probs[mb_idx]  # [B,1]
                mb_advantages = advantages[mb_idx]    # [B,1]
                mb_returns = returns[mb_idx]          # [B,1]

                # ----- 2-1) 새 log_prob, entropy 계산 -----
                # 저장된 complex action으로부터 phase 복원
                action_real = torch.real(mb_actions)
                action_imag = torch.imag(mb_actions)
                phase_taken = torch.atan2(action_imag, action_real)  # [B, N]

                mean, log_std = self.actor(mb_states)  # [B, N], [B, N]
                std = log_std.exp()

                z = (phase_taken - mean) / (std + 1e-8)
                log_prob = -0.5 * (z.pow(2) + 2 * log_std + LOG_2PI)  # [B, N]
                log_prob = log_prob.sum(dim=-1, keepdim=True)         # [B, 1]

                # entropy (diagonal Gaussian)
                entropy = 0.5 * (1.0 + LOG_2PI + 2 * log_std)         # [B, N]
                entropy = entropy.sum(dim=-1, keepdim=True)           # [B, 1]

                # ----- 2-2) PPO clipped objective -----
                ratio = torch.exp(log_prob - mb_old_log_probs)        # [B,1]

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                             1.0 + self.clip_eps) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ----- 2-3) Critic (value) 업데이트 -----
                value_pred = self.critic(mb_states)  # [B,1]
                value_loss = (mb_returns - value_pred).pow(2).mean() * self.value_loss_coef

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()
