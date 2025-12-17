import numpy as np
import time


#N_max là số phần tử tối đa 1 hàng
def build_interlayer_and_correlation(M, N, N_max, d_element_spacing, d_layer_spacing_transmit, d_layer_spacing_receive, wavelength):
    #tạo ma trận rỗng
    W_T = np.zeros((M, M), dtype=complex)
    Corr_T = np.zeros((M, M), dtype=float)
    U_R = np.zeros((N, N), dtype=complex)
    Corr_R = np.zeros((N, N), dtype=float)

    #xác định vị trí phần tử TX (phát) trên siêu bề mặt dùng để xác định khoảng cách giữa các phần tử
    #-> phục vụ tính ma trận liên tầng W_T và ma trận tương quan Corr_T
    for mm1 in range(1, M+1):
        m_z = int(np.ceil(mm1 / N_max))
        m_x = (mm1 - 1) % N_max + 1
        for mm2 in range(1, M+1):
            n_z = int(np.ceil(mm2 / N_max))
            n_x = (mm2 - 1) % N_max + 1
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing
            #ma trận liên tầng W_T
            d_temp2 = np.sqrt(d_layer_spacing_transmit**2 + d_temp**2)
            W_T[mm2-1, mm1-1] = wavelength / (4 * np.pi * d_temp2) * np.exp(-1j * 2 * np.pi * d_temp2 / wavelength)
            #ma trận tương quan Corr_T
            Corr_T[mm2-1, mm1-1] = sinc(2 * d_temp / wavelength)

    #tương tự với phía RX (thu)
    for nn1 in range(1, N+1):
        m_z = int(np.ceil(nn1 / N_max))
        m_x = (nn1 - 1) % N_max + 1
        for nn2 in range(1, N+1):
            n_z = int(np.ceil(nn2 / N_max))
            n_x = (nn2 - 1) % N_max + 1
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_receive**2 + d_temp**2)
            U_R[nn2-1, nn1-1] = wavelength / (4 * np.pi * d_temp2) * np.exp(-1j * 2 * np.pi * d_temp2 / wavelength)
            Corr_R[nn2-1, nn1-1] = sinc(2 * d_temp / wavelength)

    return W_T, Corr_T, U_R, Corr_R


#buid ma trận W_T_1 và U_R_1 dùng để kết nối TX/RX với luồng dữ liệu (streams)
def build_W1_U1(
    M, N, N_max, d_element_spacing,
    d_layer_spacing_transmit, d_layer_spacing_receive,
    wavelength, S
):
    # W_T_1: (M x S)
    W_T_1 = np.zeros((M, S), dtype=complex)
    # U_R_1: (S x N)
    U_R_1 = np.zeros((S, N), dtype=complex)

    # ---- TX side: giống build_interlayer_and_correlation ----
    for mm in range(1, M+1):
        m_z = int(np.ceil(mm / N_max))
        m_x = (mm - 1) % N_max + 1
        for s in range(1, S+1):
            # khoảng cách giữa phần tử Tx-mm và stream s
            # *** DÙNG CÙNG LOGIC KHOẢNG CÁCH NHƯ W_T ***
            d_temp = np.sqrt((m_x - s)**2 + (m_z - s)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_transmit**2 + d_temp**2)
            W_T_1[mm-1, s-1] = wavelength/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/wavelength)

    # ---- RX side: giống build_interlayer_and_correlation ----
    for nn in range(1, N+1):
        n_z = int(np.ceil(nn / N_max))
        n_x = (nn - 1) % N_max + 1
        for s in range(1, S+1):
            d_temp = np.sqrt((n_x - s)**2 + (n_z - s)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_receive**2 + d_temp**2)
            U_R_1[s-1, nn-1] = wavelength/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/wavelength)

    return W_T_1, U_R_1

#tính ma trận tương quan (nêu ở trên) Corr_T và Corr_R
def sinc(x):
    y = np.ones_like(x, dtype=float)
    idx = np.abs(x) > 1e-12
    y[idx] = np.sin(np.pi * x[idx]) / (np.pi * x[idx])
    return y


#phần này chưa phase_tx và phase_rx là 2 biến dùng để tối ưu NMSE
#tính ma trận P (ma trận tín hiệu đi từ TX qua các layers đến streams)
def build_P_from_tx_phases(phase_tx, W_T_1, W_T, L):
    # phase_tx: M x L (each column is layer)
    M, Lp = phase_tx.shape
    assert Lp == L
    P = np.diag(phase_tx[:,0]) @ W_T_1
    for l in range(1, L):
        P = np.diag(phase_tx[:, l]) @ (W_T @ P)
    return P

#tính ma trận Q(ma trận tín hiệu đi từ streams qua các layers đến RX)
def build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K):
    # phase_rx: N x K
    N, Kp = phase_rx.shape
    assert Kp == K
    Q = U_R_1 @ np.diag(phase_rx[:,0])
    for k in range(1, K):
        Q = Q @ U_R @ np.diag(phase_rx[:, k])
    return Q

#tính ma trận tín hiệu tổng hợp 
def compute_factor_and_nmse(Q, G, P, H_true, eps=1e-12):
    H_SIM = Q @ G @ P

    #vecto hoá tín hiệu, đưa ma trận thành vecto 1 cột để dễ tính beta
    H_sim_vec = H_SIM.reshape(-1, 1)
    H_true_vec = H_true.reshape(-1, 1)

    #tính beta
    #beta là hệ số bù trừ công suất/pha để kênh thực tế H_SIM so khớp được với kênh lý tưởng H_true.
    #Giúp so sánh NMSE công bằng, bất kể H_SIM có cường độ mạnh hay yếu.
    denom = (H_sim_vec.conj().T @ H_sim_vec).item()
    if np.abs(denom) < eps:
        beta = 0+0j
    else:
        beta = (H_sim_vec.conj().T @ H_true_vec).item() / denom
    
    #tính nmse
    #NMSE = mức độ lệch giữa kênh thực tế và kênh lý tưởng.
    #NMSE càng nhỏ → H_SIM càng gần H_true → nhiễu càng ít → kênh được chéo hoá tốt.
    num = np.linalg.norm(beta * H_SIM - H_true)**2
    den = np.linalg.norm(H_true)**2 + eps
    nmse = num / den
    return beta, float(np.real_if_close(nmse)), H_SIM

#triển khai thuật toán ssa vào bài toán tối ưu
def ssa_local_search_phase_search(
    G, H_true, W_T, W_T_1, U_R, U_R_1,
    M, L, N, K, pop=30, iters=200, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    n_vars = M*L + N*K
    salps = np.random.rand(pop, n_vars) * 2 * np.pi

    nmse_vals = np.zeros(pop)
    for i in range(pop):
        ph = salps[i]
        phase_tx = np.exp(1j * ph[:M*L].reshape(M, L))
        phase_rx = np.exp(1j * ph[M*L:].reshape(N, K))
        P = build_P_from_tx_phases(phase_tx, W_T_1, W_T, L)
        Q = build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K)
        _, nmse_vals[i], _ = compute_factor_and_nmse(Q, G, P, H_true)

    best_idx = np.argmin(nmse_vals)
    best = salps[best_idx].copy()
    best_nmse = nmse_vals[best_idx]

    for t in range(1, iters+1):
        c1 = 2 * np.exp(-(4*t/iters)**2)
        new_salps = salps.copy()

        for i in range(pop):
            if i == 0:
                c2 = np.random.rand(n_vars)
                c3 = np.random.rand(n_vars)
                step = c1 * c2
                direction = np.where(c3 < 0.5, 1.0, -1.0)
                new = np.mod(best + step * direction, 2*np.pi)
            else:
                new = np.mod(0.5 * (salps[i] + salps[i-1]), 2*np.pi)
            new_salps[i] = new

        nmse_new = np.zeros(pop)
        for i in range(pop):
            ph = new_salps[i]
            phase_tx = np.exp(1j * ph[:M*L].reshape(M, L))
            phase_rx = np.exp(1j * ph[M*L:].reshape(N, K))
            P = build_P_from_tx_phases(phase_tx, W_T_1, W_T, L)
            Q = build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K)
            _, nmse_new[i], _ = compute_factor_and_nmse(Q, G, P, H_true)

        salps = new_salps
        idx = np.argmin(nmse_new)

        if nmse_new[idx] < best_nmse:
            best_nmse = nmse_new[idx]
            best = salps[idx].copy()

            # ===== LOCAL SEARCH CHỈ Ở ĐÂY =====
            best, best_nmse = local_search_phase(
                best, best_nmse,
                G, H_true,
                W_T, W_T_1, U_R, U_R_1,
                M, L, N, K
            )

    phase_tx = np.exp(1j * best[:M*L].reshape(M, L))
    phase_rx = np.exp(1j * best[M*L:].reshape(N, K))
    P = build_P_from_tx_phases(phase_tx, W_T_1, W_T, L)
    Q = build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K)
    beta, nmse_final, H_sim = compute_factor_and_nmse(Q, G, P, H_true)

    return best, nmse_final, beta, H_sim

def local_search_phase(best, best_nmse, G, H_true,
                       W_T, W_T_1, U_R, U_R_1,
                       M, L, N, K,
                       sigma=0.1, n_trials=10):
    n_vars = len(best)
    best_local = best.copy()
    best_local_nmse = best_nmse

    for _ in range(n_trials):
        candidate = best_local + sigma * np.random.randn(n_vars)
        candidate = np.mod(candidate, 2*np.pi)

        phase_tx = np.exp(1j * candidate[:M*L].reshape(M, L))
        phase_rx = np.exp(1j * candidate[M*L:].reshape(N, K))

        P = build_P_from_tx_phases(phase_tx, W_T_1, W_T, L)
        Q = build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K)

        _, nmse_cand, _ = compute_factor_and_nmse(Q, G, P, H_true)

        if nmse_cand < best_local_nmse:
            best_local_nmse = nmse_cand
            best_local = candidate.copy()

    return best_local, best_local_nmse

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# --- 1. Tải Dữ Liệu Từ Các Tệp .mat (Bây giờ đã ở cùng thư mục) ---
try:
    # Tải dữ liệu NMSE
    nmse_k1_data = sio.loadmat('NMSE_K_1.mat')
    nmse_k2_data = sio.loadmat('NMSE_K_2.mat')
    nmse_k5_data = sio.loadmat('NMSE_K_5.mat')
    nmse_k10_data = sio.loadmat('NMSE_K_10.mat')

    # Tải dữ liệu Capacity
    capacity_k1_data = sio.loadmat('Capacity_K_1.mat')
    capacity_k2_data = sio.loadmat('Capacity_K_2.mat')
    capacity_k5_data = sio.loadmat('Capacity_K_5.mat')
    capacity_k10_data = sio.loadmat('Capacity_K_10.mat')
    capacity_max_data = sio.loadmat('Capacity_max.mat')

    # Trích xuất vectors dữ liệu
    NMSE_K_1 = nmse_k1_data['NMSE_K_1'].flatten()
    NMSE_K_2 = nmse_k2_data['NMSE_K_2'].flatten()
    NMSE_K_5 = nmse_k5_data['NMSE_K_5'].flatten()
    NMSE_K_10 = nmse_k10_data['NMSE_K_10'].flatten()
    
    Capacity_K_1 = capacity_k1_data['Capacity_K_1'].flatten()
    Capacity_K_2 = capacity_k2_data['Capacity_K_2'].flatten()
    Capacity_K_5 = capacity_k5_data['Capacity_K_5'].flatten()
    Capacity_K_10 = capacity_k10_data['Capacity_K_10'].flatten()
    Capacity_max = capacity_max_data['Capacity_max'].flatten()

except Exception as e:
    print(f"Lỗi khi tải hoặc xử lý tệp .mat: {e}")
    print("Vui lòng kiểm tra lại tên biến trong các file .mat hoặc đường dẫn.")
    exit()

# Trục x (Số lớp L)
L_values = np.arange(1, len(NMSE_K_1) + 1)

# --- 2. Vẽ Đồ Thị NMSE ---
plt.figure(figsize=(8, 6))

# Kiểu đường: K=1 ('r-.'), K=2 ('g:'), K=5 ('b--'), K=10 ('m-')
plt.plot(L_values, NMSE_K_1, 'r-.', linewidth=3, label='K = 1')
plt.plot(L_values, NMSE_K_2, 'g:', linewidth=3, label='K = 2')
plt.plot(L_values, NMSE_K_5, 'b--', linewidth=3, label='K = 5')
plt.plot(L_values, NMSE_K_10, 'm-', linewidth=3, label='K = 10')

plt.xlabel('Number of transmit metasurface layers, L', fontsize=14)
plt.ylabel('Fitting NMSE, $\\Delta$', fontsize=14)
plt.xticks(L_values) 
plt.tick_params(labelsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--')
plt.title('Fitting NMSE vs. Number of Transmit Metasurface Layers')
plt.show()


# --- 3. Vẽ Đồ Thị Capacity ---
plt.figure(figsize=(8, 6))

# Kiểu đường: K=1 ('r-.'), K=2 ('g:'), K=5 ('b--'), K=10 ('m-'), Optimal ('k-.')
plt.plot(L_values, Capacity_K_1, 'r-.', linewidth=3, label='K = 1')
plt.plot(L_values, Capacity_K_2, 'g:', linewidth=3, label='K = 2')
plt.plot(L_values, Capacity_K_5, 'b--', linewidth=3, label='K = 5')
plt.plot(L_values, Capacity_K_10, 'm-', linewidth=3, label='K = 10')
plt.plot(L_values, Capacity_max, 'k-.', linewidth=3, label='Optimal')

plt.xlabel('Number of transmit metasurface layers, L', fontsize=14)
plt.ylabel('Channel capacity, C [bps/Hz]', fontsize=14)
plt.xticks(L_values) 
plt.tick_params(labelsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--')
plt.title('Channel Capacity vs. Number of Transmit Metasurface Layers')
plt.show()
