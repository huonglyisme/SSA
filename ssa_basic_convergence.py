import numpy as np
import matplotlib.pyplot as plt

# --- 1. PHYSICAL MODEL LOGIC FROM REPORT ---
def build_H_sim_final(ph, M, N, L, K, W, W_in, W_out, G):
    p_tx = np.exp(1j * ph[:M*L].reshape(M, L))
    p_rx = np.exp(1j * ph[M*L:].reshape(N, K))
    
    # Transmit matrix P: S -> M through L layers [cite: 510]
    P = np.diag(p_tx[:, 0]) @ W_in
    for l in range(1, L):
        P = np.diag(p_tx[:, l]) @ (W @ P)
        
    # Receive matrix Q: N -> S through K layers [cite: 511]
    Q = W_out @ np.diag(p_rx[:, 0])
    for k in range(1, K):
        Q = Q @ W @ np.diag(p_rx[:, k])
        
    return Q @ G @ P

# --- 2. ELITISM SSA CONVERGENCE TRACKING ---
def run_one_convergence(M, N, L, K, G, H_target, W, W_in, W_out, pop=50, iters=200):
    n_vars = M*L + N*K
    # Initialization: Uniform [0, 2pi]
    salps = np.random.uniform(0, 2*np.pi, (pop, n_vars))
    elite_size = 3 # [cite: 597, 662]
    
    history = []
    
    def get_nmse(ph):
        H_sim = build_H_sim_final(ph, M, N, L, K, W, W_in, W_out, G)
        H_sim_vec = H_sim.flatten()
        H_true_vec = H_target.flatten()
        # Optimal beta compensation [cite: 82, 518]
        beta = np.vdot(H_sim_vec, H_true_vec) / (np.linalg.norm(H_sim_vec)**2 + 1e-12)
        nmse = np.linalg.norm(beta * H_sim - H_target)**2 / (np.linalg.norm(H_target)**2 + 1e-12)
        return nmse

    for t in range(iters):
        fit_vals = np.array([get_nmse(s) for s in salps])
        idx = np.argsort(fit_vals)
        salps = salps[idx]
        
        # Best NMSE at this iteration
        history.append(fit_vals[0])
        
        elites = salps[:elite_size].copy() # Elite preservation [cite: 149, 156]
        
        c1 = 2 * np.exp(-(4 * t / iters)**2) # Balance coefficient [cite: 114, 555]
        best_salp = salps[0]
        
        for i in range(pop):
            if i == 0: # Leader update
                c2 = np.random.rand(n_vars)
                c3 = np.random.rand(n_vars)
                step = c1 * (2*np.pi * c2)
                salps[i] = np.mod(best_salp + np.where(c3 >= 0.5, 1, -1) * step, 2*np.pi)
            else: # Follower update
                salps[i] = np.mod((salps[i] + salps[i-1]) / 2, 2*np.pi) # [cite: 114, 663]
        
        # Elitism override [cite: 160, 669]
        salps[:elite_size] = elites

    return history

# --- 3. EXECUTION ---
# Setup parameters similar to codessa_elitism.py
M = N = 16; S = 2; wavelength = 0.01; L = 10; K = 10
np.random.seed(42)

G_phys = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) * 0.5
W = (np.random.randn(M, M) + 1j*np.random.randn(M, M)) * 0.1
W_in = np.eye(M, S, dtype=complex)
W_out = np.eye(S, N, dtype=complex)
H_target = np.eye(S)

conv_history = run_one_convergence(M, N, L, K, G_phys, H_target, W, W_in, W_out, pop=60, iters=300)

# --- 4. PLOTTING THE SINGLE CONVERGENCE LINE ---
plt.figure(figsize=(9, 6))
plt.semilogy(range(1, 301), conv_history, 'b-', linewidth=2.5, label='Elitism SSA')

plt.title(f"NMSE Convergence Plot (L={L}, K={K})", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Best Fitting NMSE (Log Scale)", fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# Ensure no overlapping labels
plt.tight_layout()
plt.savefig('one_line_convergence.png')
print("Saved one_line_convergence.png")

