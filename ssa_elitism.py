import numpy as np
import matplotlib.pyplot as plt

# --- 1. MÔ HÌNH VẬT LÝ (Chương 2 & 5) ---
def build_H_sim_final(ph, M, N, L, K, W, W_in, W_out, G):
    """Tính toán H_SIM = QGP dựa trên logic Chương 2.3 [cite: 520]"""
    p_tx = np.exp(1j * ph[:M*L].reshape(M, L))
    p_rx = np.exp(1j * ph[M*L:].reshape(N, K))
    
    # Ma trận P (Transmit): S -> M qua L lớp [cite: 510]
    P = np.diag(p_tx[:, 0]) @ W_in
    for l in range(1, L):
        P = np.diag(p_tx[:, l]) @ (W @ P)
        
    # Ma trận Q (Receive): N -> S qua K lớp [cite: 511]
    Q = W_out @ np.diag(p_rx[:, 0])
    for k in range(1, K):
        Q = Q @ W @ np.diag(p_rx[:, k])
        
    return Q @ G @ P

# --- 2. THUẬT TOÁN ELITISM SSA (Thuật toán 3 & 5 [cite: 594, 650]) ---
def ssa_elitism_final(M, N, L, K, G, H_target, W, W_in, W_out, pop=30, iters=50):
    n_vars = M*L + N*K
    # Khởi tạo: Kết hợp Uniform và một chút Gaussian để tránh kẹt (Hybrid nhẹ) [cite: 655-656]
    salps = np.random.uniform(0, 2*np.pi, (pop, n_vars))
    elite_size = 3 
    
    def fitness_func(ph):
        H_sim = build_H_sim_final(ph, M, N, L, K, W, W_in, W_out, G)
        # NMSE chuẩn hóa theo công thức (2.1) [cite: 518]
        H_sim_vec = H_sim.flatten()
        H_true_vec = H_target.flatten()
        # Tính beta tối ưu để bù trừ biên độ 
        beta = np.vdot(H_sim_vec, H_true_vec) / (np.linalg.norm(H_sim_vec)**2 + 1e-12)
        nmse = np.linalg.norm(beta * H_sim - H_target)**2 / (np.linalg.norm(H_target)**2 + 1e-12)
        return nmse

    for t in range(iters):
        # Đánh giá Fitness và Sắp xếp [cite: 596, 661]
        fit_vals = np.array([fitness_func(s) for s in salps])
        idx = np.argsort(fit_vals)
        salps = salps[idx]
        
        # ELITISM: Lưu tinh hoa [cite: 597, 662]
        elites = salps[:elite_size].copy()
        
        c1 = 2 * np.exp(-(4 * t / iters)**2) # Hệ số c1 (Algorithm 1) [cite: 555]
        best_salp = salps[0]
        
        for i in range(pop):
            if i == 0: # Leader cập nhật [cite: 555]
                c2 = np.random.rand(n_vars)
                c3 = np.random.rand(n_vars)
                step = c1 * (2*np.pi * c2)
                salps[i] = np.mod(best_salp + np.where(c3 >= 0.5, 1, -1) * step, 2*np.pi)
            else: # Follower cập nhật theo Newton [cite: 555, 663]
                salps[i] = np.mod((salps[i] + salps[i-1]) / 2, 2*np.pi)
        
        # ELITISM: Ghi đè bảo toàn tinh hoa [cite: 601, 669]
        salps[:elite_size] = elites

    # Kết quả sau tối ưu
    best_ph = salps[0]
    best_nmse = fitness_func(best_ph)
    H_final = build_H_sim_final(best_ph, M, N, L, K, W, W_in, W_out, G)
    # Tính Capacity với SNR = 20dB để tránh bệt 0
    capacity = np.log2(np.abs(np.linalg.det(np.eye(H_target.shape[0]) + 100 * (H_final @ H_final.conj().T))) + 1e-12)
    
    return best_nmse, capacity

# --- 3. CHẠY MÔ PHỎNG VÀ VẼ ĐỒ THỊ ---
M = N = 16; S = 2; wavelength = 0.01; d = wavelength/2
L_range = [1, 2, 3, 5, 8, 10]
K_list = [1, 2, 5, 10]

# Khởi tạo kênh G vật lý (N x M) và ma trận lan truyền W (M x M)
G_phys = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) * 0.5
W = (np.random.randn(M, M) + 1j*np.random.randn(M, M)) * 0.1 # Ma trận lan truyền ngẫu nhiên ổn định
W_in = np.eye(M, S, dtype=complex) # Khởi tạo đơn giản để giữ năng lượng
W_out = np.eye(S, N, dtype=complex)
H_target = np.eye(S)

res_nmse = {k: [] for k in K_list}
res_cap = {k: [] for k in K_list}

for K in K_list:
    print(f"Tính toán Elitism SSA cho K = {K}...")
    for L in L_range:
        nmse, cap = ssa_elitism_final(M, N, L, K, G_phys, H_target, W, W_in, W_out)
        res_nmse[K].append(nmse)
        res_cap[K].append(cap)

# Vẽ đồ thị
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
styles = {1: 'r-o', 2: 'g-s', 5: 'b-^', 10: 'm-d'}

for K in K_list:
    ax1.plot(L_range, res_nmse[K], styles[K], linewidth=2, label=f'K = {K}')
    ax2.plot(L_range, res_cap[K], styles[K], linewidth=2, label=f'K = {K}')

ax1.set_title("Fitting NMSE vs. Layers L (Elitism SSA Final)"); ax1.set_xlabel("Layers L"); ax1.set_ylabel("NMSE"); ax1.grid(True); ax1.legend()
ax2.set_title("Channel Capacity vs. Layers L"); ax2.set_xlabel("Layers L"); ax2.set_ylabel("Capacity [bps/Hz]"); ax2.grid(True); ax2.legend()
plt.tight_layout(); plt.show()
