import numpy as np
import matplotlib.pyplot as plt

# --- 1. Khởi tạo hệ thống (Tối ưu hóa tốc độ nhân ma trận) ---
def build_matrices_fast(M, N, S, d_spacing, wavelength):
    # Khởi tạo ma trận lan truyền dựa trên khoảng cách vật lý
    W = np.zeros((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            if i == j: continue
            dist = np.abs(i - j) * d_spacing
            W[i, j] = (wavelength / (4 * np.pi * dist)) * np.exp(-1j * 2 * np.pi * dist / wavelength)
    
    # Ma trận kết nối luồng dữ liệu (S streams) vào Antenna (M/N)
    W_in = np.random.randn(M, S) + 1j * np.random.randn(M, S)
    W_out = np.random.randn(S, N) + 1j * np.random.randn(S, N)
    return W, W_in, W_out

def build_H_eff(phases_tx, phases_rx, W, W_in, W_out, G, L, K):
    # Tích lũy các lớp Metasurface phía phát (Transmit)
    P = np.diag(np.exp(1j * phases_tx[:, 0])) @ W_in
    for l in range(1, L):
        P = np.diag(np.exp(1j * phases_tx[:, l])) @ (W @ P)
        
    # Tích lũy các lớp Metasurface phía thu (Receive)
    Q = W_out @ np.diag(np.exp(1j * phases_rx[:, 0]))
    for k in range(1, K):
        Q = Q @ W @ np.diag(np.exp(1j * phases_rx[:, k]))
        
    return Q @ G @ P  # Kết quả ma trận (S x S)

def get_fitness(phases, M, N, L, K, W, W_in, W_out, G, H_target):
    # Tách phase của Tx và Rx từ vector phẳng của SSA
    p_tx = phases[:M*L].reshape(M, L)
    p_rx = phases[M*L:].reshape(N, K)
    H_sim = build_H_eff(p_tx, p_rx, W, W_in, W_out, G, L, K)
    
    # Tính NMSE (Hàm mục tiêu cần cực tiểu hóa)
    nmse = np.linalg.norm(H_sim - H_target)**2 / np.linalg.norm(H_target)**2
    return nmse

# --- 2. Thuật toán SSA rút gọn (Chạy nhanh để lấy kết quả) ---
def ssa_core(M, N, L, K, G, H_target, W, W_in, W_out):
    dim = M*L + N*K
    n_salps = 5  # Giảm số lượng cá thể để chạy nhanh
    max_iter = 10
    salps = np.random.uniform(0, 2*np.pi, (n_salps, dim))
    best_score = float('inf')
    best_pos = None

    for t in range(max_iter):
        for i in range(n_salps):
            score = get_fitness(salps[i], M, N, L, K, W, W_in, W_out, G, H_target)
            if score < best_score:
                best_score = score
                best_pos = salps[i].copy()
        
        # Cập nhật vị trí Salps (Logic Leader-Follower)
        c1 = 2 * np.exp(-(4 * t / max_iter)**2)
        for i in range(n_salps):
            if i == 0:
                salps[i] = best_pos + c1 * ((np.random.rand(dim)-0.5)*2*np.pi)
            else:
                salps[i] = (salps[i] + salps[i-1]) / 2
        salps = np.clip(salps, 0, 2*np.pi)
    
    # Tính Capacity cuối cùng từ kết quả tốt nhất
    p_tx = best_pos[:M*L].reshape(M, L)
    p_rx = best_pos[M*L:].reshape(N, K)
    H_final = build_H_eff(p_tx, p_rx, W, W_in, W_out, G, L, K)
    cap = np.log2(np.abs(np.linalg.det(np.eye(H_target.shape[0]) + 10 * (H_final @ H_final.conj().T))))
    return best_score, cap

# --- 3. Chạy mô phỏng và Vẽ đồ thị ---
M = N = 16
S = 2
d = 0.005
wv = 0.01
L_range = [1, 3, 5, 7, 10]
K_list = [1, 2, 5, 10]
G_channel = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) * 0.1
H_target = np.eye(S)

results = {k: {'nmse': [], 'cap': []} for k in K_list}

for K in K_list:
    print(f"Tính toán K={K}...")
    for L in L_range:
        W, Win, Wout = build_matrices_fast(M, N, S, d, wv)
        nmse, cap = ssa_core(M, N, L, K, G_channel, H_target, W, Win, Wout)
        results[K]['nmse'].append(nmse)
        results[K]['cap'].append(cap)

# Vẽ đồ thị
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {1: 'red', 2: 'green', 5: 'blue', 10: 'magenta'}

for K in K_list:
    ax1.plot(L_range, results[K]['nmse'], '-o', color=colors[K], label=f'K={K}')
    ax2.plot(L_range, results[K]['cap'], '-s', color=colors[K], label=f'K={K}')

ax1.set_title("Fitting NMSE (SSA Optimized)"); ax1.set_xlabel("Layers L"); ax1.grid(True); ax1.legend()
ax2.set_title("Channel Capacity"); ax2.set_xlabel("Layers L"); ax2.grid(True); ax2.legend()
plt.show()
