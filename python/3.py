import numpy as np
import matplotlib.pyplot as plt

def get_hermite_dft_basis(N=32):
    # 1. 原始 DFT 分解
    n = np.arange(N)
    A = np.exp(-2j * np.pi * np.outer(n, n) / N) / np.sqrt(N)
    ev, evec = np.linalg.eig(A)

    # 2. 构造辅助“形状矩阵” S (离散谐振子算符)
    # S = 二阶差分 + 二次势能
    D2 = np.diag(np.ones(N-1), -1) - 2*np.diag(np.ones(N)) + np.diag(np.ones(N-1), 1)
    # 修正边界以符合周期性
    D2[0, -1] = D2[-1, 0] = 1
    
    # 二次势能项 (离散化的 x^2)
    x2 = np.diag((n - (N-1)/2)**2)
    S = -D2 + 0.1 * x2 # 系数 0.1 用于平衡动能和势能

    # 3. 对简并子空间进行二次对角化
    target_evs = [1+0j, 0+1j, -1+0j, 0-1j]
    final_vectors = []

    for teev in target_evs:
        # 找出属于当前特征值的子空间 (容差 1e-5)
        mask = np.isclose(ev, teev, atol=1e-5)
        V_sub = evec[:, mask]
        
        # 二次对角化核心步骤：投影 S 到子空间
        S_small = np.conj(V_sub.T) @ S @ V_sub
        _, W = np.linalg.eigh(S_small) # 使用 eigh 因为 S 是实对称的
        
        # 合成最终基底
        V_final_sub = V_sub @ W
        final_vectors.append(V_final_sub[:, 0]) # 选该子空间中“能量”最低的一个

    # 绘图展示
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, vec in enumerate(final_vectors):
        axes[i].plot(np.real(vec), label='Real')
        axes[i].set_title(f"Refined Mode $\lambda={target_evs[i]}$")
    plt.show()

get_hermite_dft_basis(32)