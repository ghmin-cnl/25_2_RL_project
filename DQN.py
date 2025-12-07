import numpy as np
import torch
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
            torch.stack(states),                                      # [B, state_dim]
            torch.stack(actions),                                     # [B, N] (complex)
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),  # [B,1]
            torch.stack(next_states),                                 # [B, state_dim]
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)     # [B,1]
        )

    def __len__(self):
        return len(self.storage)


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class DQNAgent:
    def __init__(self,
                 N,
                 device,
                 lr=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 max_buffer_size=1e6,
                 num_actions=64,    
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=1e-4):

        self.N = N
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # main_RL.py에서 사용하는 state_dim에 맞춤
        self.state_dim = 4 * N * N + 2

        # 각 action은 N 차원의 complex 위상 벡터
        # 여기서는 단순히 균일 랜덤 위상으로 초기화
        phases = np.random.uniform(low=-np.pi, high=np.pi, size=(num_actions, N)).astype(np.float32)
        theta_real = np.cos(phases)
        theta_imag = np.sin(phases)
        theta_complex = theta_real + 1j * theta_imag
        self.action_codebook = torch.from_numpy(theta_complex).to(torch.complex64).to(device)  # [A, N]
        self.num_actions = num_actions

        # Q-network & target Q-network
        self.q_net = QNetwork(self.state_dim, num_actions).to(device)
        self.q_target = QNetwork(self.state_dim, num_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.actor = self.q_net

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size)

        # Epsilon-greedy 파라미터
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.total_it = 0


    @torch.no_grad()
    def select_action(self, state, noise=True):
        self.total_it += 1

        state = state.to(self.device)
        batch_size = state.size(0)

        # epsilon 업데이트
        if noise:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - self.epsilon_decay * self.total_it
            )
        else:
            # 평가 시에는 greedy
            self.epsilon = 0.0

        # Q(s, :) 계산
        q_values = self.q_net(state)  # [B, num_actions]

        # epsilon-greedy로 action index 선택
        if noise and np.random.rand() < self.epsilon:
            # random action
            action_idx = torch.randint(low=0, high=self.num_actions,
                                       size=(batch_size,), device=self.device)
        else:
            # greedy action
            action_idx = torch.argmax(q_values, dim=1)  # [B]

        # 코드북에서 complex theta 뽑기
        # action_codebook: [A, N]
        theta = self.action_codebook[action_idx]        # [B, N] complex
        thetaH = torch.conj(theta)                      # [B, N] complex

        return theta, thetaH

    def _soft_update_target(self):
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def _action_to_index(self, actions_complex):
        # 코드북: [A, N], actions: [B, N]
        # 거리: ||a - c||^2
        # → |a-c|^2 = (Re)^2 + (Im)^2
        A = self.num_actions
        B = actions_complex.size(0)

        # [B, 1, N]
        actions_exp = actions_complex.unsqueeze(1)
        # [1, A, N]
        codebook_exp = self.action_codebook.unsqueeze(0)  # [1, A, N]

        diff = actions_exp - codebook_exp                 # [B, A, N], complex
        diff_real = torch.real(diff)
        diff_imag = torch.imag(diff)
        dist2 = diff_real.pow(2) + diff_imag.pow(2)       # [B, A, N]
        dist2 = dist2.sum(dim=-1)                         # [B, A]

        # 가장 가까운 코드북 index
        _, indices = torch.min(dist2, dim=1)              # [B]
        return indices

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # replay에서 샘플링
        states, actions_complex, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)                   # [B, state_dim]
        actions_complex = actions_complex.to(self.device) # [B, N] complex
        rewards = rewards.to(self.device)                 # [B,1]
        next_states = next_states.to(self.device)         # [B, state_dim]
        dones = dones.to(self.device)                     # [B,1]

        # ----- 1) complex action -> discrete index -----
        action_indices = self._action_to_index(actions_complex)  # [B]

        # ----- 2) target Q 계산 -----
        with torch.no_grad():
            # Q_target(s', a')에서 a'는 greedy action
            q_next = self.q_target(next_states)           # [B, A]
            max_q_next, _ = q_next.max(dim=1, keepdim=True)  # [B,1]
            target_q = rewards + (1.0 - dones) * self.gamma * max_q_next  # [B,1]

        # ----- 3) 현재 Q(s, a) -----
        q_values = self.q_net(states)                    # [B, A]
        # gather로 선택한 action의 Q만 가져오기
        action_indices_unsq = action_indices.unsqueeze(1)  # [B,1]
        current_q = torch.gather(q_values, 1, action_indices_unsq)  # [B,1]

        # ----- 4) DQN loss (MSE) -----
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ----- 5) target network soft update -----
        self._soft_update_target()
