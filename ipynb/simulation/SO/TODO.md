# Source Optimization TODO

Lanczos 降阶在固定介质、变化光源场景下的 ROM 思路（光源优化 / 逆向设计）。

### 1. 为什么 Lanczos 在 Source Optimization 中是神技？

在光源优化中，你通常需要寻找一个最优的电流分布 $J$ 或场分布 $\mathbf{E}$，以实现在目标区域的特定能量集中或模式匹配。

* **传统做法**：每修改一次光源，都要重新跑一遍迭代求解器（如 GMRES），计算成本是 $O(N)$。
* **你的 Lanczos 方案**：
1. **预计算阶段（离线）**：运行 Lanczos 算法，提取系统的 $K$ 个低阶本征模态（Eigenmodes）$\phi_k$ 和对应的本征值 $\lambda_k$。
2. **优化阶段（在线）**：场 $\mathbf{E}$ 被表示为基函数的线性组合 $E = \sum_k c_k \phi_k$。此时，波动方程变成了**代数方程**：

$$\mathcal{L} E = s \quad \Rightarrow \quad \lambda_k c_k = \langle \phi_k, s \rangle$$

优化光源 $J$ 时，计算场 $\mathbf{E}$ 只需要做 $K$ 次内积加权。如果 $K \ll N$，计算速度会提升几个数量级，且**目标函数关于光源的梯度计算也变得极为简单**。

---

### 2. 物理层面的考量：哪些模态是关键？

波动方程算子 $\mathcal{L}$ 的谱分布很有趣：

* **近谐振模态**：本征值 $\lambda$ 接近 0 的模态。这些模态对光源极其敏感，能量最容易激发。
* **远场/消逝模态**：本征值远离 0。

在光源优化中，你往往只需要提取**绝对值最小**（Smallest Magnitude, 'SM'）的那一部分本征值对应的模态，因为它们构成了响应函数的主成分。

---

### 3. Python 伪代码实现框架

你可以利用 `scipy.sparse.linalg.LinearOperator` 配合 `eigsh`（它内部就是 Lanczos 算法的变体）来实现这个优化加速器：

```python
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.fft import fftn, ifftn

# 1. 定义 Matrix-free 算子 (FFT切换法)
def create_wave_operator(shape, dx, epsilon_r, k0):
    N = np.prod(shape)
    # 预计算 k 空间
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(shape[2], d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_sq = KX**2 + KY**2 + KZ**2

    def matvec(v):
        E_r = v.reshape(shape)
        # 频域项: \nabla^2
        E_k = fftn(E_r)
        term_diff = ifftn(-k_sq * E_k)
        # 空域项: k0^2 * eps
        term_space = (k0**2 * epsilon_r) * E_r
        return (term_diff + term_space).flatten()

    return LinearOperator((N, N), matvec=matvec, dtype=complex)

# 2. 预计算本征模态 (离线阶段)
def precompute_modes(operator, num_modes=50):
    print(f"提取前 {num_modes} 个本征模态...")
    # sigma=0 表示寻找最接近 0 (谐振点) 的本征值
    vals, vecs = eigsh(operator, k=num_modes, which='LM', sigma=0)
    return vals, vecs

# 3. 快速求解光源响应 (在线阶段)
def fast_solve_for_source(vals, vecs, source_dist):
    # 将源投影到本征基上
    s_flat = source_dist.flatten()
    projections = vecs.conj().T @ s_flat # 获取各模态的激发系数
    coeffs = projections / vals          # 除以本征值 (即你想要的"非迭代"步骤)
    E_flat = vecs @ coeffs               # 重新组合成物理场
    return E_flat

```

---

### 4. 这种方案在 Source Optimization 中的独特优势

1. **导数计算（Adjoint Method）的简化**：在做梯度下降优化光源时，你通常需要计算伴随场。由于你已经有了本征基，伴随场的计算同样是解析的，不再需要额外的 PDE 求解。
2. **物理可解释性**：你可以直接看到光源激发了介质的哪些本征模式。例如，如果某个模式对你的目标增益贡献最大，你可以有针对性地设计光源的形状去匹配这个模式。
3. **约束处理**：如果光源有总功率约束或空间分布约束，在基函数空间（$K$ 维）做优化比在网格空间（$N$ 维）做优化要稳定得多。

### 5. 注意事项

* **复数对称性**：波动方程算子在没有损耗（$\epsilon_r$ 为实数）时是 Hermitian 的，Lanczos 表现完美。如果有损耗，算子是非对称的，你需要使用 **Arnoldi 方法**（`scipy.sparse.linalg.eigs`），计算量会稍大一些。
* **模态截断误差**：如果你选取的模态数 $K$ 太少，场的高频细节可能会丢失。但在光源优化中，我们往往关注的是宏观的场强分布，几十到几百个模态通常已经足够。

**这个思路在光子晶体设计、天线阵列合成和微波成像中非常先进。你是否已经准备好基于这个架构去构建你的光源优化算法了？**
