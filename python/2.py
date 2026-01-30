import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import make_interp_spline

def plot_dft_eigenvectors_rows(N=16):
    # 1. 构造归一化的 DFT 矩阵
    n = np.arange(N)
    k = n.reshape((N, 1))
    A = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)

    # 2. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # 3. 清理并按目标顺序分类：1, i, -1, -i
    target_evs = [1+0j, 0+1j, -1+0j, 0-1j]
    groups = defaultdict(list)
    for i in range(N):
        ev_clean = target_evs[np.argmin(np.abs(eigenvalues[i] - target_evs))]
        groups[ev_clean].append(eigenvectors[:, i])

    # 4. 绘图设置：4行布局
    titles = [r"$\lambda = 1$ (Even)", r"$\lambda = i$ (Odd)", 
              r"$\lambda = -1$ (Even)", r"$\lambda = -i$ (Odd)"]
    max_cols = max(len(v) for v in groups.values())
    
    fig, axes = plt.subplots(4, max_cols, figsize=(4 * max_cols, 10), sharex=True, sharey=True)

    # 插值点：用于生成光滑曲线
    n_smooth = np.linspace(0, N-1, 200)

    for row, ev in enumerate(target_evs):
        vecs = groups[ev]
        # 按节点数（频率）排序
        vecs.sort(key=lambda v: np.sum(np.abs(np.diff(np.sign(v.real)))))
        
        for col in range(max_cols):
            ax = axes[row, col]
            if col < len(vecs):
                vec = vecs[col]
                # 相位对齐
                max_idx = np.argmax(np.abs(vec))
                vec = vec * np.conj(vec[max_idx]) / np.abs(vec[max_idx])
                
                # 创建光滑曲线 (使用 B-spline 插值)
                spline_real = make_interp_spline(n, vec.real, k=3)
                spline_imag = make_interp_spline(n, vec.imag, k=3)
                
                ax.plot(n_smooth, spline_real(n_smooth), 'b-', lw=2, label='Real' if col==0 else "")
                ax.plot(n_smooth, spline_imag(n_smooth), 'r--', lw=1.5, alpha=0.6, label='Imag' if col==0 else "")
                
                # 淡淡地标出原始采样点以对齐物理意义
                ax.scatter(n, vec.real, color='blue', s=10, alpha=0.3)
                
                if col == 0:
                    ax.set_ylabel(titles[row], fontsize=14, fontweight='bold')
                ax.axhline(0, color='black', lw=0.5, alpha=0.5)
                ax.grid(True, alpha=0.2)
            else:
                ax.axis('off')

    plt.suptitle(f"DFT Eigenmode Profiles (N={N}): Continuous View\nRows represent rotational symmetry classes", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 建议 N=16 效果较好
plot_dft_eigenvectors_rows(N=8)