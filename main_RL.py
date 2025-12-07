import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time  

from processing import (load_and_process_data, generate_w, cal_loss, linear_to_db)
#from DQN import DQNAgent as Agent
#from DDPG import DDPGAgent as Agent
#from TD3 import TD3Agent as Agent
from SAC import SACAgent as Agent

seeds = [0, 1, 2, 3, 4]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

torch.set_default_dtype(torch.float32)

K, M, N = 1, 8, 32
lr = 1e-3
sigma2 = 1e-2
TaudB = 10
Tau = 10 ** (TaudB / 10)

batch_size = 256
Episode = 1000

gamma = 0.99
tau_soft = 0.01
PdB = 10
P = 10 ** (PdB / 10)

alpha_corr = 1.0
beta_w = 0.99
lambda_ = 0.5

SNR_com_th_dB = TaudB
eps_corr = 1e-8


# Train/Test 데이터
Train_Files = [
    f"\data\\train\\Train_Channel_{i:03d}.mat"
    for i in range(1, 41)
]

Test_File = "\data\\test\\Test_Channel.mat"


train_sets = []
for f in Train_Files:
    R_sep_scaled, origin, originH, num = load_and_process_data(f, N, M)
    train_sets.append((
        R_sep_scaled.to(device),
        origin.to(device),
        originH.to(device),
        num
    ))

R_test_sep_scaled, origin_dataset_test, origin_datasetH_test, test_num = \
    load_and_process_data(Test_File, N, M, is_test=True)

R_test_sep_scaled = R_test_sep_scaled.to(device).float()
origin_dataset_test = origin_dataset_test.to(device)
origin_datasetH_test = origin_datasetH_test.to(device)

state_dim = 4 * N * N + 2


if __name__ == "__main__":
    agent = Agent(
        N=N,
        device=device,
        lr=lr,
        gamma=gamma,
        tau=tau_soft,
        use_ou_noise=True,
        ou_rho=0.9,
        ou_sigma=0.2
    )

    start_time = time.time()

    episode_reward_log = []
    episode_snrt_log = []
    episode_snrc_log = []
    train_losses = []
    episode_violation_rate_log = []

    for e in range(Episode):

        agent.reset_noise()
        agent.actor.train()

        total_reward = 0.0
        step_count = 0

        SNRt_ep_list = []
        SNRc_ep_list = []

        violation_count = 0
        total_constraint_steps = 0

        for (R_sep_scaled, origin_dataset, origin_datasetH, total_num) in train_sets:

            prev_SNRt = 0.0
            prev_SNRc = 0.0

            for t in range(5, total_num):

                R_t5 = R_sep_scaled[t - 5]

                state_vec = torch.cat([
                    R_t5.reshape(-1),
                    torch.tensor([prev_SNRt, prev_SNRc],
                                 dtype=torch.float32, device=device)
                ], dim=0)

                state_batch = state_vec.unsqueeze(0)  # shape: (1, state_dim)

                theta_pred, thetaH_pred = agent.select_action(state_batch, noise=True)

                y_chan = origin_dataset[t].unsqueeze(0)
                yH_chan = origin_datasetH[t].unsqueeze(0)

                _, hc_batch, ht_batch, Ht_batch = cal_loss(
                    y_chan, yH_chan, theta_pred, thetaH_pred, M, N
                )

                ht_i = ht_batch[0]
                hc_i = hc_batch[0]
                Ht_i = Ht_batch[0]

                w = generate_w(ht_i, hc_i, P, Tau, sigma2)

                SNRt_lin = torch.linalg.norm(Ht_i @ w) ** 2 / sigma2
                SNRc_lin = torch.abs(w.conj().T @ hc_i) ** 2 / sigma2

                SNRt_dB = linear_to_db(SNRt_lin)
                SNRc_dB = linear_to_db(SNRc_lin)

                SNRt_ep_list.append(SNRt_dB.item())
                SNRc_ep_list.append(SNRc_dB.item())

                if SNRc_dB.item() < SNR_com_th_dB:
                    violation_count += 1
                total_constraint_steps += 1

                norm_ht2 = torch.linalg.norm(ht_i) ** 2
                norm_hc2 = torch.linalg.norm(hc_i) ** 2
                corr_num = torch.abs(torch.conj(ht_i).T @ hc_i) ** 2
                rho_t = corr_num / (norm_ht2 * norm_hc2 + eps_corr)

                penalty = torch.nn.functional.relu(SNR_com_th_dB - SNRc_dB)

                reward_t = (
                    alpha_corr * rho_t +
                    beta_w * SNRt_dB +
                    (1.0 - beta_w) * SNRc_dB -
                    lambda_ * penalty
                )

                R_next = R_sep_scaled[t - 4]

                next_state_vec = torch.cat([
                    R_next.reshape(-1),
                    torch.tensor([SNRt_dB.item(), SNRc_dB.item()],
                                 dtype=torch.float32, device=device)
                ], dim=0)

                agent.replay_buffer.add((
                    state_vec.detach().cpu(),
                    theta_pred[0].detach().cpu(),
                    float(reward_t.item()),
                    next_state_vec.detach().cpu(),
                    0.0  # done
                ))
                agent.train(batch_size)

                total_reward += reward_t.item()
                step_count += 1


                prev_SNRt = float(SNRt_dB.item())
                prev_SNRc = float(SNRc_dB.item())

        avg_reward = total_reward / max(step_count, 1)
        avg_snrt = np.mean(SNRt_ep_list)
        avg_snrc = np.mean(SNRc_ep_list)


        violation_rate = violation_count / max(total_constraint_steps, 1)
        episode_violation_rate_log.append(violation_rate)

        episode_reward_log.append(avg_reward)
        episode_snrt_log.append(avg_snrt)
        episode_snrc_log.append(avg_snrc)
        train_losses.append(-avg_reward)

        print(f"Ep {e+1:03d} | Reward={avg_reward:.4f} | "
              f"SNRt={avg_snrt:.2f} dB | SNRc={avg_snrc:.2f} dB | "
              f"Viol={violation_rate:.3f}")


    print("\n=== Test evaluation ===")
    agent.actor.eval()

    SNRt_test_list = []
    SNRc_test_list = []
    reward_test_list = []

    prev_SNRt_eval = 0.0
    prev_SNRc_eval = 0.0

    violation_test_count = 0
    total_test_steps = 0

    for t in range(5, test_num):

        R_t5 = R_test_sep_scaled[t - 5]

        state_vec = torch.cat([
            R_t5.reshape(-1),
            torch.tensor([prev_SNRt_eval, prev_SNRc_eval],
                         dtype=torch.float32, device=device)
        ], dim=0)

        state_batch = state_vec.unsqueeze(0)

        theta_eval, thetaH_eval = agent.select_action(state_batch, noise=False)

        y_chan = origin_dataset_test[t].unsqueeze(0)
        yH_chan = origin_datasetH_test[t].unsqueeze(0)

        _, hc_batch, ht_batch, Ht_batch = cal_loss(
            y_chan, yH_chan, theta_eval, thetaH_eval, M, N
        )

        ht_i = ht_batch[0]
        hc_i = hc_batch[0]
        Ht_i = Ht_batch[0]

        w_te = generate_w(ht_i, hc_i, P, Tau, sigma2)

        SNRt_lin = torch.linalg.norm(Ht_i @ w_te) ** 2 / sigma2
        SNRc_lin = torch.abs(w_te.conj().T @ hc_i) ** 2 / sigma2

        SNRt_test_list.append(SNRt_lin)
        SNRc_test_list.append(SNRc_lin)

        SNRt_dB = linear_to_db(SNRt_lin)
        SNRc_dB = linear_to_db(SNRc_lin)

        if SNRc_dB.item() < SNR_com_th_dB:
            violation_test_count += 1
        total_test_steps += 1

        norm_ht2 = torch.linalg.norm(ht_i) ** 2
        norm_hc2 = torch.linalg.norm(hc_i) ** 2
        corr_num = torch.abs(torch.conj(ht_i).T @ hc_i) ** 2
        rho_t = corr_num / (norm_ht2 * norm_hc2 + eps_corr)

        penalty = torch.nn.functional.relu(SNR_com_th_dB - SNRc_dB)

        reward_eval = (
            alpha_corr * rho_t +
            beta_w * SNRt_dB +
            (1 - beta_w) * SNRc_dB -
            lambda_ * penalty
        )
        reward_test_list.append(reward_eval.item())

        prev_SNRt_eval = float(SNRt_dB.item())
        prev_SNRc_eval = float(SNRc_dB.item())

    SNRt_test = torch.stack(SNRt_test_list)
    SNRc_test = torch.stack(SNRc_test_list)
    reward_test_mean = np.mean(reward_test_list)

    violation_test_rate = violation_test_count / max(total_test_steps, 1)

    print("\n=== Test 결과 ===")
    print(f"SNRt_test = {linear_to_db(SNRt_test.mean()):.2f} dB")
    print(f"SNRc_test = {linear_to_db(SNRc_test.mean()):.2f} dB")
    print(f"Reward_test = {reward_test_mean:.4f}")
    print(f"Viol_test = {violation_test_rate:.3f}")

    total_time_sec = time.time() - start_time
    total_time_min = total_time_sec / 60.0
    print(f"\n총 소요 시간: {total_time_sec:.2f} sec ({total_time_min:.2f} min)")

    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    agent_name = agent.__class__.__name__.replace("Agent", "")
    save_name = f"{agent_name}-alpha=0.5_seed3_result.pt"
    save_path = os.path.join(result_dir, save_name)

    torch.save({
        'reward_ep': np.array(episode_reward_log),
        'snrt_ep': np.array(episode_snrt_log),
        'snrc_ep': np.array(episode_snrc_log),
        'viol_ep': np.array(episode_violation_rate_log),         
        'snrt_test': linear_to_db(SNRt_test.mean()).cpu().numpy(),
        'snrc_test': linear_to_db(SNRc_test.mean()).cpu().numpy(),
        'viol_test': np.array(violation_test_rate),              
        'total_time_sec': np.array(total_time_sec),          
    }, save_path)

    print(f"\n== Test 결과가 '{save_path}' 에 저장되었습니다. ==")
    
    model_save_name = f"{agent_name}-alpha=0.5_seed3_agent.pt"
    model_save_path = os.path.join(result_dir, model_save_name)
    torch.save(agent, model_save_path)
    print(f"== 학습된 에이전트가 '{model_save_path}' 에 저장되었습니다. ==")
