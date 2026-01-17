import numpy as np
import matplotlib.pyplot as plt

# --- 1. MÔ HÌNH VẬT LÝ (Dựa trên Chương 2 & 5) ---
def build_H_sim_hybrid(ph, M, N, L, K, W, W_in, W_out, G):
    """Tính toán H_SIM = QGP dựa trên tích chuỗi ma trận [cite: 520, 638]"""
    p_tx = np.exp(1j * ph[:M*L].reshape(M, L))
    p_rx = np.exp(1j * ph[M*L:].reshape(N, K))
    
    # Transmit SIM: Tích lũy pha qua L lớp [cite: 510]
    P = np.diag(p_tx[:, 0]) @ W_in
    for l in range(1, L):
        P = np.diag(p_tx[:, l]) @ (W @ P)
        
    # Receive SIM: Tích lũy pha qua K lớp [cite: 511]
    Q = W_out @ np.diag(p_rx[:, 0])
    for k in range(1, K):
        Q = Q @ W @ np.diag(p_rx[:, k])
        
    return Q @ G @ P

# --- 2. THUẬT TOÁN HYBRID SSA (Algorithm 2 & 5) ---
def ssa_hybrid_optimization(M, N, L, K, G, H_target, W, W_in, W_out, pop=30, iters=50):
    n_vars = M*L + N*K
    salps = np.random.uniform(0, 2*np.pi, (pop, n_vars))
    
    # BƯỚC HYBRID: Khởi tạo 1/3 quần thể quanh 0 (Transparent Mode) [cite: 562, 701, 708]
    for i in range(pop // 3):
        salps[i] = np.random.normal(0, 0.1, n_vars) 
        
    def fitness_func(ph):
        H_sim = build_H_sim_hybrid(ph, M, N, L, K, W, W_in, W_out, G)
        # NMSE chuẩn hóa với hệ số beta tối ưu [cite: 518, 639]
        H_sim_vec = H_sim.flatten()
        H_true_vec = H_target.flatten()
        beta = np.vdot(H_sim_vec, H_true_vec) / (np.linalg.norm(H_sim_vec)**2 + 1e-12)
        return np.linalg.norm(beta * H_sim - H_target)**2 / (np.linalg.norm(H_target)**2 + 1e-12)

    # Khởi tạo nghiệm tốt nhất
    fit_vals = np.array([fitness_func(s) for s in salps])
    best_idx = np.argmin(fit_vals)
    best_salp = salps[best_idx].copy()

    for t in range(iters):
        c1 = 2 * np.exp(-(4 * t / iters)**2) # Hệ số c1 hội tụ [cite: 555]
        
        for i in range(pop):
            if i == 0: # Leader cập nhật theo Leader-Follower [cite: 543]
                c2 = np.random.rand(n_vars)
                c3 = np.random.rand(n_vars)
                step = c1 * (2*np.pi * c2)
                salps[i] = np.mod(best_salp + np.where(c3 >= 0.5, 1, -1) * step, 2*np.pi)
            else: # Follower cập nhật [cite: 544]
                salps[i] = np.mod((salps[i] + salps[i-1]) / 2, 2*np.pi)
        
        # Cập nhật nghiệm tốt nhất sau mỗi vòng lặp
        current_fit = np.array([fitness_func(s) for s in salps])
        if np.min(current_fit) < fitness_func(best_salp):
            best_salp = salps[np.argmin(current_fit)].copy()

    # Tính Capacity cuối cùng
    H_final = build_H_sim_hybrid(best_salp, M, N, L, K, W, W_in, W_out, G)
    SNR = 100 # SNR = 20dB để đồ thị hiển thị rõ [cite: 471]
    capacity = np.log2(np.abs(np.linalg.det(np.eye(H_target.shape[0]) + SNR * (H_final @ H_final.conj().T))) + 1e-12)
    
    return fitness_func(best_salp), capacity

# --- 3. CHẠY MÔ PHỎNG (Thông số từ Chương 5.1) ---
M = N = 16; S = 2; N_max = 4
L_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K_list = [1, 2, 5, 10]

# Khởi tạo ma trận ổn định để tín hiệu truyền qua được nhiều lớp
G_phys = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) * 0.5
W_prop = (np.random.randn(M, M) + 1j*np.random.randn(M, M)) * 0.1 
W_in = np.eye(M, S, dtype=complex) # Selection matrix
W_out = np.eye(S, N, dtype=complex)
H_target = np.eye(S)

res_nmse = {k: [] for k in K_list}
res_cap = {k: [] for k in K_list}

for K in K_list:
    print(f"Tính toán Hybrid SSA cho K = {K}...")
    for L in L_range:
        nmse, cap = ssa_hybrid_optimization(M, N, L, K, G_phys, H_target, W_prop, W_in, W_out)
        res_nmse[K].append(nmse)
        res_cap[K].append(cap)

# Vẽ đồ thị chuẩn theo Figure 5.1 & 5.6 trong báo cáo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
styles = {1: 'r-o', 2: 'g-s', 5: 'b-^', 10: 'm-d'}

for K in K_list:
    ax1.plot(L_range, res_nmse[K], styles[K], linewidth=2, label=f'K = {K}')
    ax2.plot(L_range, res_cap[K], styles[K], linewidth=2, label=f'K = {K}')

ax1.set_title("Hybrid SSA: NMSE vs. Layers L"); ax1.set_xlabel("Layers L"); ax1.set_ylabel("NMSE"); ax1.grid(True); ax1.legend()
ax2.set_title("Hybrid SSA: Capacity vs. Layers L"); ax2.set_xlabel("Layers L"); ax2.set_ylabel("Capacity [bps/Hz]"); ax2.grid(True)
ax2.axhline(y=8.9, color='k', linestyle='--', label='Optimal')
ax2.legend()
plt.tight_layout(); plt.show()
