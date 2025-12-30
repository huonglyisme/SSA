import numpy as np
import matplotlib.pyplot as plt
import time


#N_max là số phần tử tối đa 1 hàng
def build_interlayer_and_correlation(
    M, N, N_max,
    d_element_spacing,
    d_layer_spacing_transmit,
    d_layer_spacing_receive,
    wavelength
):
    W_T = np.zeros((M, M), dtype=complex)
    Corr_T = np.zeros((M, M), dtype=float)
    U_R = np.zeros((N, N), dtype=complex)
    Corr_R = np.zeros((N, N), dtype=float)

    for mm1 in range(1, M+1):
        m_z = int(np.ceil(mm1 / N_max))
        m_x = (mm1 - 1) % N_max + 1
        for mm2 in range(1, M+1):
            n_z = int(np.ceil(mm2 / N_max))
            n_x = (mm2 - 1) % N_max + 1
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_transmit**2 + d_temp**2)
            W_T[mm2-1, mm1-1] = wavelength/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/wavelength)
            Corr_T[mm2-1, mm1-1] = sinc(2 * d_temp / wavelength)

    for nn1 in range(1, N+1):
        m_z = int(np.ceil(nn1 / N_max))
        m_x = (nn1 - 1) % N_max + 1
        for nn2 in range(1, N+1):
            n_z = int(np.ceil(nn2 / N_max))
            n_x = (nn2 - 1) % N_max + 1
            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_receive**2 + d_temp**2)
            U_R[nn2-1, nn1-1] = wavelength/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/wavelength)
            Corr_R[nn2-1, nn1-1] = sinc(2 * d_temp / wavelength)

    return W_T, Corr_T, U_R, Corr_R


#buid ma trận W_T_1 và U_R_1 dùng để kết nối TX/RX với luồng dữ liệu (streams)
def build_W1_U1(
    M, N, N_max,
    d_element_spacing,
    d_layer_spacing_transmit,
    d_layer_spacing_receive,
    wavelength,
    S
):
    W_T_1 = np.zeros((M, S), dtype=complex)
    U_R_1 = np.zeros((S, N), dtype=complex)

    for mm in range(1, M+1):
        m_z = int(np.ceil(mm / N_max))
        m_x = (mm - 1) % N_max + 1
        for s in range(1, S+1):
            d_temp = np.sqrt((m_x - s)**2 + (m_z - s)**2) * d_element_spacing
            d_temp2 = np.sqrt(d_layer_spacing_transmit**2 + d_temp**2)
            W_T_1[mm-1, s-1] = wavelength/(4*np.pi*d_temp2) * np.exp(-1j*2*np.pi*d_temp2/wavelength)

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
    M, Lp = phase_tx.shape
    assert Lp == L
    P = np.diag(phase_tx[:,0]) @ W_T_1
    for l in range(1, L):
        P = np.diag(phase_tx[:, l]) @ (W_T @ P)
    return P

#tính ma trận Q(ma trận tín hiệu đi từ streams qua các layers đến RX)
def build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K):
    N, Kp = phase_rx.shape
    assert Kp == K
    Q = U_R_1 @ np.diag(phase_rx[:,0])
    for k in range(1, K):
        Q = Q @ U_R @ np.diag(phase_rx[:, k])
    return Q

#tính ma trận tín hiệu tổng hợp 
def compute_factor_and_nmse(Q, G, P, H_true, eps=1e-12):
    H_SIM = Q @ G @ P

    H_sim_vec = H_SIM.reshape(-1, 1)
    H_true_vec = H_true.reshape(-1, 1)

    denom = (H_sim_vec.conj().T @ H_sim_vec).item()
    if np.abs(denom) < eps:
        beta = 0+0j
    else:
        beta = (H_sim_vec.conj().T @ H_true_vec).item() / denom

    nmse = np.linalg.norm(beta * H_SIM - H_true)**2 / (np.linalg.norm(H_true)**2 + eps)
    return beta, float(np.real_if_close(nmse)), H_SIM

#triển khai thuật toán ssa vào bài toán tối ưu
def ssa_basic_phase_search(
    G, H_true,
    W_T, W_T_1,
    U_R, U_R_1,
    M, L, N, K,
    pop=30,
    iters=200,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    n_vars = M*L + N*K
    salps = np.random.rand(pop, n_vars) * 2*np.pi

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

    nmse_history = [best_nmse]

    for t in range(1, iters+1):
        c1 = 2 * np.exp(-(4*t/iters)**2)
        new_salps = salps.copy()

        for i in range(pop):
            if i == 0:
                c2 = np.random.rand(n_vars)
                c3 = np.random.rand(n_vars)
                direction = np.where(c3 < 0.5, 1.0, -1.0)
                new = np.mod(best + c1 * c2 * direction, 2*np.pi)
            else:
                new = np.mod(0.5*(salps[i] + salps[i-1]), 2*np.pi)
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

        nmse_history.append(best_nmse)

    phase_tx = np.exp(1j * best[:M*L].reshape(M, L))
    phase_rx = np.exp(1j * best[M*L:].reshape(N, K))
    P = build_P_from_tx_phases(phase_tx, W_T_1, W_T, L)
    Q = build_Q_from_rx_phases(phase_rx, U_R_1, U_R, K)

    beta, nmse_final, H_sim = compute_factor_and_nmse(Q, G, P, H_true)

    return best, nmse_final, beta, H_sim, nmse_history

def compute_capacity(H, rho):
    S = H.shape[0]
    I = np.eye(S)
    return float(np.real_if_close(
        np.log2(np.linalg.det(I + (rho/S) * (H @ H.conj().T)))
    ))

# ================== SIMULATION PARAMETERS ==================
S = 4
M = S
N = S
N_max = 2

L = 3        # số lớp TX metasurface (ví dụ)
K = 2        # số lớp RX metasurface (ví dụ)

rho = 10     # SNR cho capacity
# ===========================================================

wavelength = 0.01
d_element_spacing = wavelength / 2
d_layer_spacing_transmit = 2 * wavelength
d_layer_spacing_receive = 2 * wavelength

W_T, Corr_T, U_R, Corr_R = build_interlayer_and_correlation(
    M, N, N_max,
    d_element_spacing,
    d_layer_spacing_transmit,
    d_layer_spacing_receive,
    wavelength
)

W_T_1, U_R_1 = build_W1_U1(
    M, N, N_max,
    d_element_spacing,
    d_layer_spacing_transmit,
    d_layer_spacing_receive,
    wavelength,
    S
)

G = np.random.randn(S, S) + 1j*np.random.randn(S, S)
H_true = np.eye(S)

best, nmse_final, beta, H_sim, nmse_hist = ssa_basic_phase_search(
    G, H_true,
    W_T, W_T_1,
    U_R, U_R_1,
    M, L, N, K,
    pop=30,
    iters=200,
    seed=0
)

rho = 10
capacity = compute_capacity(H_sim, rho)

print("Final NMSE:", nmse_final)
print("Capacity:", capacity)

plt.figure()
plt.semilogy(nmse_hist, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("NMSE")
plt.grid(True, which="both")
plt.title("SSA NMSE Convergence")
plt.show()

