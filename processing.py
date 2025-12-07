import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset


def load_and_process_data(file_path, N, M, is_test=False):
    chan_theta_mat = scio.loadmat(file_path)
    G0 = chan_theta_mat['G_all']
    hrc0 = chan_theta_mat['hrc_all']
    hrt0 = chan_theta_mat['hrt_all']
    GH0 = chan_theta_mat['GH_all']
    hrcH0 = chan_theta_mat['hrcH_all']
    hrtH0 = chan_theta_mat['hrtH_all']

    total_num = G0.shape[2]
    G   = torch.zeros([total_num, N, M], dtype=torch.complex64)
    hrc = torch.zeros([total_num, N, 1], dtype=torch.complex64)
    hrt = torch.zeros([total_num, N, 1], dtype=torch.complex64)
    GH  = torch.zeros([total_num, M, N], dtype=torch.complex64)
    hrcH = torch.zeros([total_num, 1, N], dtype=torch.complex64)
    hrtH = torch.zeros([total_num, 1, N], dtype=torch.complex64)

    for i in range(total_num):
        G[i, :, :]   = torch.from_numpy(G0[:, :, i])
        hrc[i, :, :] = torch.from_numpy(hrc0[:, i].reshape(N, 1))
        hrt[i, :, :] = torch.from_numpy(hrt0[:, i].reshape(N, 1))
        GH[i, :, :]  = torch.from_numpy(GH0[:, :, i])
        hrcH[i, :, :] = torch.from_numpy(hrcH0[:, i].reshape(1, N))
        hrtH[i, :, :] = torch.from_numpy(hrtH0[:, i].reshape(1, N))

    # R_com, R_rad 계산
    psi_com   = torch.matmul(torch.diag_embed(hrcH.squeeze(), dim1=1), G)
    psiH_com  = torch.matmul(GH, torch.diag_embed(hrc.squeeze(), dim1=1))
    R_com     = torch.matmul(psi_com, psiH_com).reshape(total_num, 1, N, N)
    R_com_sep = torch.cat([torch.real(R_com), torch.imag(R_com)], axis=1)

    psi_rad   = torch.matmul(torch.diag_embed(hrtH.squeeze(), dim1=1), G)
    psiH_rad  = torch.matmul(GH, torch.diag_embed(hrt.squeeze(), dim1=1))
    R_rad     = torch.matmul(psi_rad, psiH_rad).reshape(total_num, 1, N, N)
    R_rad_sep = torch.cat([torch.real(R_rad), torch.imag(R_rad)], axis=1)

    # R_rad와 R_com을 합친 뒤 평균/표준편차 스케일링
    R_sep = torch.cat([R_rad_sep, R_com_sep], axis=1)
    R_sep_mean = torch.mean(R_sep, dim=2, keepdim=True)
    R_sep_std  = torch.std(R_sep, dim=2, keepdim=True)
    R_sep_scaled = (R_sep - R_sep_mean) / (R_sep_std + 1e-10)

    # origin_dataset (실제 채널 구성)
    hrc_vec = hrc.view(total_num, -1)
    hrt_vec = hrt.view(total_num, -1)
    G_vec   = G.view(total_num, -1)
    origin_dataset = torch.cat((torch.real(G_vec), torch.imag(G_vec),
                                torch.real(hrt_vec), torch.imag(hrt_vec),
                                torch.real(hrc_vec), torch.imag(hrc_vec)), axis=-1)

    hrcH_vec = hrcH.view(total_num, -1)
    hrtH_vec = hrtH.view(total_num, -1)
    GH_vec   = GH.view(total_num, -1)
    origin_datasetH = torch.cat((torch.real(GH_vec), torch.imag(GH_vec),
                                 torch.real(hrtH_vec), torch.imag(hrtH_vec),
                                 torch.real(hrcH_vec), torch.imag(hrcH_vec)), axis=-1)

    return R_sep_scaled, origin_dataset, origin_datasetH, total_num


class MyData(Dataset):
    def __init__(self, R, chans, chans2):
        self.data   = R.to(torch.float32)
        self.label  = chans.to(torch.float32)
        self.label2 = chans2.to(torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.label2[index]

    def __len__(self):
        return len(self.data)


def generate_w(ht, hc, Pt, Tau, sigma):
    temp1 = torch.abs(hc.conj().T @ ht) ** 2
    u1 = hc / torch.linalg.norm(hc)
    u2_ = ht - (u1.conj().T @ ht) * u1
    u2 = u2_ / torch.linalg.norm(u2_)
    x1_ = u1.conj().T @ ht
    x1 = torch.sqrt(Tau * sigma / torch.linalg.norm(hc) ** 2) * x1_ / torch.linalg.norm(x1_)

    x2_ = u2.conj().T @ ht
    temp2 = Pt - Tau * sigma / torch.linalg.norm(hc) ** 2
    if temp2 < 0:
        temp2sqrt = 1j * torch.sqrt(Tau * sigma / torch.linalg.norm(hc) ** 2 - Pt)
    else:
        temp2sqrt = torch.sqrt(temp2)

    x2 = temp2sqrt * x2_ / torch.linalg.norm(x2_)
    if Pt * temp1 > Tau * sigma * torch.linalg.norm(ht) ** 2:
        w = np.sqrt(Pt) * ht / torch.linalg.norm(ht)
    else:
        w = x1 * u1 + x2 * u2
        if torch.linalg.norm(w) ** 2 > Pt:
            w = np.sqrt(Pt) * w / torch.linalg.norm(w)
    return w


def cal_loss(chan, chanH, theta, thetaH, M, N):
    chan  = chan.to(torch.float32)
    chanH = chanH.to(torch.float32)

    # G, GH
    G0 = chan[:, :2 * N * M]
    G  = G0[:, :N * M] + 1j * G0[:, N * M:]
    G  = G.view([-1, N, M]).to(torch.complex64)

    G0H = chanH[:, :2 * N * M]
    GH  = G0H[:, :N * M] + 1j * G0H[:, N * M:]
    GH  = GH.view([-1, M, N]).to(torch.complex64)

    # hrt, hrtH
    hrt0 = chan[:, 2 * N * M : 2 * (N*M + N)]
    hrt  = hrt0[:, :N] + 1j * hrt0[:, N:]
    hrt  = hrt.view([-1, N, 1]).to(torch.complex64)

    hrt0H = chanH[:, 2 * N * M : 2 * (N*M + N)]
    hrtH  = hrt0H[:, :N] + 1j * hrt0H[:, N:]
    hrtH  = hrtH.view([-1, 1, N]).to(torch.complex64)

    # hrc, hrcH
    hrc0 = chan[:, 2 * (N * M + N):]
    hrc  = hrc0[:, :N] + 1j * hrc0[:, N:]
    hrc  = hrc.view([-1, N, 1]).to(torch.complex64)

    hrc0H = chanH[:, 2 * (N * M + N):]
    hrcH  = hrc0H[:, :N] + 1j * hrc0H[:, N:]
    hrcH  = hrcH.view([-1, 1, N]).to(torch.complex64)

    theta  = theta.to(torch.complex64)
    thetaH = thetaH.to(torch.complex64)

    # GH * diag(theta)
    temp1 = torch.matmul(GH, torch.diag_embed(theta, dim1=1))
    temp2 = torch.matmul(torch.diag_embed(thetaH, dim1=1), G)
    hc    = torch.matmul(temp1, hrc)
    ht    = torch.matmul(temp1, hrt)
    htH   = torch.matmul(hrtH, temp2)
    Ht    = torch.matmul(ht, htH)

  
    reward_vector = torch.matmul(Ht, hc)
    reward_vector = reward_vector[:, :, 0]
    reward = torch.linalg.norm(reward_vector, axis=-1).to(torch.float32)
    a = 0.8 * torch.linalg.norm(Ht, axis=[1, 2])


    loss = -torch.mean(reward + a)
    return loss, hc, ht, Ht


def linear_to_db(value):
    return 10 * torch.log10(value + 1e-10)
