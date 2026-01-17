import numpy as np
import matplotlib.pyplot as plt

# --- 1. MÔ HÌNH VẬT LÝ CHUẨN (Chương 2 & 5) ---
def build_H_sim_final(ph, M, N, L, K, W, W_in, W_out, G):
    p_tx = np.exp(1j * ph[:M*L].reshape(M, L))
    p_rx = np.exp(1j * ph[M*L:].reshape(N, K))
    
    # Transmit SIM (P): S -> M qua L lớp 
    P = np.diag(p_tx[:, 0]) @ W_in
    for l in range(1, L):
        P = np.diag(p_tx[:, l]) @ (W @ P)
        
    # Receive SIM (Q): N -> S qua K lớp 
    Q = W_out @ np.diag(p_rx[:, 0])
    for k in range(1, K):
        Q = Q @ W @ np.diag(p_rx[:, k])
        
    return Q @ G @ P

# --- 2. THUẬT TOÁN LOCAL SEARCH (Hill Climbing - Algorithm 4) ---
def local_search_refine(best_ph, best_nmse, M, N, L, K, G, H_target, W, W_in, W_out, sigma=0.01, trials=15):
    curr_ph = best_ph.copy()
    curr_nmse = best_nmse
    
    for _ in range(trials):
        # Rung lắc nhẹ xung quanh nghiệm tốt nhất 
        candidate = curr_ph + sigma * np.random.randn(len(curr_ph))
        candidate = np.mod(candidate, 2*np.pi)
        
        H_sim = build_H_sim_final(candidate, M, N, L, K, W, W_in, W_out, G)
        H_sim_vec, H_true_vec = H_sim.flatten(), H_target.flatten()
        beta = np.vdot(H_sim_vec, H_true_vec) / (np.linalg.norm(H_sim_vec)**2 + 1e-12)
        nmse = np.linalg.norm(beta * H_sim - H_target)**2 / (np.linalg.norm(H_target)**2 + 1e-12)
        
        if nmse < curr_nmse: # Chỉ cập nhật nếu thực sự tốt hơn 
            curr_nmse = nmse
            curr_ph = candidate.copy()
    return curr_ph, curr_nmse

# --- 3. QUY TRÌNH TỐI ƯU HÓA TỔNG HỢP (Algorithm 5) ---
def ssa_local_search_optimization(M, N, L, K, G, H_target, W, W_in, W_out, pop=25, iters=40):
    n_vars = M*L + N*K
    salps = np.random.uniform(0, 2*np.pi, (pop, n_vars))
    
    def get_nmse(ph):
        H_sim = build_H_sim_final(ph, M, N, L, K, W, W_in, W_out, G)
        H_sim_vec, H_true_vec = H_sim.flatten(), H_target.flatten()
        beta = np.vdot(H_sim_vec, H_true_vec) / (np.linalg.norm(H_sim_vec)**2 + 1e-12)
        return np.linalg.norm(beta * H_sim - H_target)**2 / (np.linalg.norm(H_target)**2 + 1e-12)

    best_nmse = float('inf')
    best_ph = salps[0].copy()

    for t in range(iters):
        c1 = 2 * np.exp(-(4 * t / iters)**2) # [cite: 555]
        
        for i in range(pop):
            score = get_nmse(salps[i])
            if score < best_nmse:
                best_nmse = score
                best_ph = salps[i].copy()
                # KÍCH HOẠT LOCAL SEARCH (Chương 5.3.3) [cite: 603, 681]
                best_ph, best_nmse = local_search_refine(best_ph, best_nmse, M, N, L, K, G, H_target, W, W_in, W_out)

        # Cập nhật vị trí bầy đàn [cite: 555]
        for i in range(pop):
            if i == 0:
                c2, c3 = np.random.rand(n_vars), np.random.rand(n_vars)
                step = c1 * (2*np.pi * c2)
                salps[i] = np.mod(best_ph + np.where(c3 >= 0.5, 1, -1) * step, 2*np.pi)
            else:
                salps[i] = np.mod((salps[i] + salps[i-1]) / 2, 2*np.pi)

    H_final = build_H_sim_final(best_ph, M, N, L, K, W, W_in, W_out, G)
    capacity = np.log2(np.abs(np.linalg.det(np.eye(S) + 100 * (H_final @ H_final.conj().T))) + 1e-12)
    return best_nmse, capacity

# --- 4. CẤU HÌNH VÀ VẼ ĐỒ THỊ ---
M = N = 16; S = 2; N_max = 4
L_range = [1, 2, 3, 5, 8, 10]
K_list = [1, 2, 5, 10]

# Khởi tạo ma trận ổn định biên độ
G_phys = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) * 0.5
W_prop = (np.random.randn(M, M) + 1j*np.random.randn(M, M)) * 0.1 
W_in = np.eye(M, S, dtype=complex)
W_out = np.eye(S, N, dtype=complex)
H_target = np.eye(S)

results_nmse = {k: [] for k in K_list}
results_cap = {k: [] for k in K_list}

for K in K_list:
    print(f"Tính toán Local Search SSA cho K = {K}...")
    for L in L_range:
        nmse, cap = ssa_local_search_optimization(M, N, L, K, G_phys, H_target, W_prop, W_in, W_out)
        results_nmse[K].append(nmse)
        results_cap[K].append(cap)

# Hiển thị đồ thị (Figure 5.1 & 5.7) [cite: 780, 840]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
styles = {1: 'r-o', 2: 'g-s', 5: 'b-^', 10: 'm-d'}

for K in K_list:
    ax1.plot(L_range, results_nmse[K], styles[K], linewidth=2, label=f'K = {K}')
    ax2.plot(L_range, results_cap[K], styles[K], linewidth=2, label=f'K = {K}')

ax1.set_title("Local Search SSA: NMSE vs. Layers L"); ax1.set_xlabel("Layers L"); ax1.set_ylabel("NMSE"); ax1.grid(True); ax1.legend()
ax2.set_title("Local Search SSA: Capacity vs. Layers L"); ax2.set_xlabel("Layers L"); ax2.set_ylabel("Capacity [bps/Hz]"); ax2.grid(True)
ax2.axhline(y=8.9, color='k', linestyle='--', label='Optimal'); ax2.legend()
plt.tight_layout(); plt.show()
