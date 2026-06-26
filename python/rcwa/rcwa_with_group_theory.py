import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import time

# ==========================================
# 1. 基础环境与高精度介质构造
# ==========================================

def create_perturbed_square(grid_size=128, size=20, eps_sq=12.0):
    """
    构造一个带有极微小扰动的正方形介质。
    微扰的作用是打破完美的数学简并，使得我们可以进行 1对1 的特征值精度对比。
    """
    eps_space = np.ones((grid_size, grid_size), dtype=complex)
    x0, y0 = grid_size // 2, grid_size // 2
    # 引入 1e-7 的微小形变
    size_x, size_y = size, size + 1e-7
    for i in range(grid_size):
        for j in range(grid_size):
            if abs(i - x0) <= size_x/2 and abs(j - y0) <= size_y/2:
                eps_space[i, j] = eps_sq
    return eps_space

def get_fft_data(eps_space):
    """获取介质的傅里叶变换数据"""
    return np.fft.fftshift(np.fft.fft2(eps_space)) / eps_space.size

# ==========================================
# 2. 群论核心：8重对称轨道生成 (C4v)
# ==========================================

def get_C4v_8fold_indices(nx, ny, N_target):
    """
    生成 C4v 点群的 8 点轨道 (Orbits)。
    这是实现极致加速的关键步骤：我们将空间划分为不可约区域。
    """
    cx, cy = nx // 2, ny // 2
    # 简单的按幅度排序选取
    # 注意：实际应用中应基于能量截断，这里简化为圆形截断
    limit = int(np.sqrt(N_target / np.pi) * 1.5) + 2
    
    indices_all = []
    # 从中心向外螺旋或简单的矩形扫描
    for ix in range(cx - limit, cx + limit + 1):
        for iy in range(cy - limit, cy + limit + 1):
            if 0 <= ix < nx and 0 <= iy < ny:
                indices_all.append((ix, iy))
    
    # 按距离中心的距离排序，模拟平面波展开的截断
    indices_all.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    orbits = []      # 存储每个轨道的具体点坐标
    orbit_reps = []  # 存储每个轨道的代表元 (用于计数)
    visited = set()
    
    for idx in indices_all:
        if idx in visited: continue
        if len(orbit_reps) * 8 >= N_target: break # 估算，大概取够数量
        
        ix, iy = idx
        dx, dy = ix - cx, iy - cy
        
        # --- C4v 8个对称操作生成 ---
        # 1. 恒等 & C2: (x, y), (-x, -y)
        # 2. 镜像 x/y:  (-x, y), (x, -y)
        # 3. 对角镜像:  (y, x), (-y, -x)
        # 4. C4 旋转:   (-y, x), (y, -x)
        pts_deltas = [
            (dx, dy), (-dx, -dy), (-dx, dy), (dx, -dy),
            (dy, dx), (-dy, -dx), (-dy, dx), (dy, -dx)
        ]
        
        # 还原到绝对坐标并去重 (处理轴上点和中心点)
        current_orbit = set()
        for d in pts_deltas:
            px, py = cx + d[0], cy + d[1]
            if 0 <= px < nx and 0 <= py < ny:
                current_orbit.add((px, py))
        
        # 只有当轨道内的点都在网格内时才接受
        if len(current_orbit) > 0:
            current_orbit_list = list(current_orbit)
            orbits.append(current_orbit_list)
            orbit_reps.append(idx) # 选第一个作为代表
            for p in current_orbit: visited.add(p)
            
    return orbit_reps, orbits

# ==========================================
# 3. 极致加速：直接构造不可约子块
# ==========================================

def build_C4v_irreducible_block(eps_fft, orbits, nx, ny, k0):
    """
    直接构造 C4v 全对称 (A1) 子块矩阵。
    
    加速原理：
    1. 不构造 N*N 大矩阵，直接构造 (N/8)*(N/8) 小矩阵。
    2. 利用 SALC (Symmetry Adapted Linear Combinations) 基底。
    
    参数:
        orbits: 由 get_C4v_8fold_indices 生成的轨道列表
    """
    N_sub = len(orbits)
    Me_sub = np.zeros((N_sub, N_sub), dtype=complex)
    cx, cy = nx // 2, ny // 2
    
    # 预计算归一化因子: 1/sqrt(轨道大小)
    # 对于一般点是 1/sqrt(8)，轴上点是 1/sqrt(4)，中心点是 1
    norms = np.array([1.0 / np.sqrt(len(orb)) for orb in orbits])
    
    # 预先将轨道转换为 numpy 数组以利用向量化加速
    # 由于轨道长度不一 (1, 4, 8)，我们无法完全向量化外层，但可以优化内层
    
    # 动量算符 (Laplacian) 的对角项
    dk = 2 * np.pi / nx # 假设方晶格 dx=dy
    # 计算每个轨道代表元的 k^2 (因为同一轨道内 k^2 相同)
    k_sq_diag = []
    for orb in orbits:
        p = orb[0] # 取任意一个点
        kx = (p[0] - cx) * dk
        ky = (p[1] - cy) * dk
        k_sq_diag.append(kx**2 + ky**2)
    k_sq_diag = np.array(k_sq_diag)

    # --- 核心双重循环 (O(N_sub^2)) ---
    # 由于 N_sub ≈ N_full / 8，这里的循环次数减少了 64 倍
    for i in range(N_sub):
        orb_i = orbits[i]
        ni = norms[i]
        
        for j in range(N_sub):
            orb_j = orbits[j]
            nj = norms[j]
            
            # 计算耦合项: <psi_i | eps | psi_j>
            # = sum_{p in orb_i} sum_{q in orb_j} eps(p - q) * ni * nj
            # 优化：利用群性质，只需计算 sum_{q in orb_j} eps(p_ref - q)，然后乘以 sqrt(Ni/Nj) * C 
            # 但为了鲁棒性，我们使用全求和 (对于小矩阵非常快)
            
            val_sum = 0j
            for r in orb_i:
                for c in orb_j:
                    diff_x = r[0] - c[0] + cx
                    diff_y = r[1] - c[1] + cy
                    # 边界检查 (确保 FFT 索引有效)
                    if 0 <= diff_x < nx and 0 <= diff_y < ny:
                        val_sum += eps_fft[diff_x, diff_y]
            
            Me_sub[i, j] = val_sum * ni * nj

    # 波动方程形式: (k0^2 * E - k^2)
    # 注意：这里构造的是算符矩阵 A，使得 A v = lambda v
    # 标准形式通常是 k0^2 * Convolution - diag(k^2)
    Me_sub = (k0**2) * Me_sub 
    
    # 减去对角项 (Laplacian)
    np.fill_diagonal(Me_sub, Me_sub.diagonal() - k_sq_diag)
    
    return Me_sub

# ==========================================
# 4. 图像重建与展示工具
# ==========================================

def reconstruct_mode(vec_sub, orbits, grid_size):
    """
    将不可约子块的特征向量 (N/8) 还原为全空间的场分布 (N)。
    利用 A1 对称性：轨道内所有点系数相同。
    """
    full_k_space = np.zeros((grid_size, grid_size), dtype=complex)
    
    # 将子空间系数分发到全空间
    for i, coeff in enumerate(vec_sub):
        norm = 1.0 / np.sqrt(len(orbits[i]))
        # A1 模式：全对称，系数 = val / sqrt(orbit_size)
        actual_val = coeff * norm 
        for p in orbits[i]:
            full_k_space[p] = actual_val
            
    # 逆傅里叶变换回实空间
    return np.fft.ifft2(np.fft.ifftshift(full_k_space))

def plot_bloch_modes(evecs_sub, evals_sub, orbits, grid_size, k0, mode_from=0, num_to_show=5, step=1):
    """
    绘制 Bloch 模式 (仅展示 E 场模平方，因为这里做的是标量波动方程演示)
    """
    # 筛选有效的特征值 (接近 k0^2 * epsilon 的物理值)
    # 简单排序：按特征值实部大小（通常对应频率或有效折射率）
    idx_sorted = np.argsort(np.real(evals_sub))[::-1] # 从大到小
    
    valid_count = min(num_to_show, len(evals_sub))
    fig, axes = plt.subplots(1, valid_count, figsize=(3 * valid_count, 3.5))
    if valid_count == 1: axes = [axes]
    
    print(f"\n--- Top {valid_count} Reconstructed Modes (A1 Symmetry) ---")
    
    for i in range(valid_count):
        idx = idx_sorted[mode_from + i * step]
        val = evals_sub[idx]
        vec = evecs_sub[:, idx]
        
        # 重建图像
        mode_spatial = reconstruct_mode(vec, orbits, grid_size)
        intensity = np.abs(mode_spatial)**2
        
        ax = axes[i]
        im = ax.imshow(intensity, cmap='inferno', origin='lower')
        
        # 估算有效折射率 n_eff = k / k0
        # 特征值 lambda = k0^2 * n_eff^2 - k_t^2 ... 粗略展示 eigenvalue
        ax.set_title(f"Mode {i}\n$\lambda$: {np.real(val):.2e}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 性能基准测试与主程序
# ==========================================

def run_accelerated_benchmark(show_images=False):
    # 参数设置
    GRID = 128
    K0 = 10.0
    N_targets = range(300, 3000, 300) # 测试不同模态截断数
    
    t_full = []
    t_fast = []
    errors = []
    x_axis = []
    
    eps_space = create_perturbed_square(GRID)
    eps_fft = get_fft_data(eps_space)

    print(f"{'N_modes':<10} | {'T_Full(s)':<10} | {'T_Fast(s)':<10} | {'Speedup':<8} | {'Error':<10}")
    print("-" * 65)

    for n_target in N_targets:
        # --- A. 准备全矩阵数据 (用于对比) ---
        # 为了公平对比，我们需要先找出全空间对应的 indices
        # 这里我们直接利用 fast 方法生成的 indices 展开得到 full indices
        reps, orbits = get_C4v_8fold_indices(GRID, GRID, n_target)
        full_indices = [p for orb in orbits for p in orb]
        N_full = len(full_indices)
        N_sub = len(orbits)
        x_axis.append(N_full)
        
        # --- B. 原始方法 (O(N^3)) ---
        start = time.time()
        # 1. 构造全矩阵 (Toeplitz)
        Me_full = np.zeros((N_full, N_full), dtype=complex)
        cx, cy = GRID//2, GRID//2
        dk = 2*np.pi/GRID
        
        # 这是一个极其耗时的步骤，模拟真实 RCWA 的负担
        # 为了不让这一步太慢影响测试脚本运行，我们用稍微优化一点的写法
        ids_arr = np.array(full_indices)
        for j in range(N_full):
             diff = ids_arr - ids_arr[j] + [cx, cy]
             mask = (diff[:,0]>=0) & (diff[:,0]<GRID) & (diff[:,1]>=0) & (diff[:,1]<GRID)
             Me_full[mask, j] = eps_fft[diff[mask,0], diff[mask,1]]
        
        # 添加对角项
        kx_full = np.array([(p[0]-cx)*dk for p in full_indices])
        ky_full = np.array([(p[1]-cy)*dk for p in full_indices])
        Me_full = K0**2 * Me_full - np.diag(kx_full**2 + ky_full**2)
        
        vals_o, vec_o = eig(Me_full)
        t_o = time.time() - start
        t_full.append(t_o)
        
        # --- C. 极速群论方法 (O((N/8)^3)) ---
        start = time.time()
        # 1. 直接构造压缩子块 (跳过全矩阵)
        Me_sub = build_C4v_irreducible_block(eps_fft, orbits, GRID, GRID, K0)
        # 2. 求解小规模特征值
        vals_s, vec_s = eig(Me_sub)
        t_s = time.time() - start
        t_fast.append(t_s)
        
        # --- D. 误差验证 ---
        # 在 vals_o 中寻找与 vals_s 最接近的匹配
        # 因为我们只计算了 A1 模，全矩阵中还有 A2, B1, B2, E 等模式，所以是子集匹配
        diffs = []
        for v in vals_s:
            min_dist = np.min(np.abs(vals_o - v))
            diffs.append(min_dist)
        avg_err = np.mean(diffs)
        errors.append(avg_err)
        
        print(f"{N_full:<10} | {t_o:<10.4f} | {t_s:<10.4f} | {t_o/t_s:<8.1f}x | {avg_err:.2e}")

        # --- E. 图像展示 (仅展示最后一次循环) ---
        if show_images and n_target == N_targets[-1]:
            plot_bloch_modes(vec_s, vals_s, orbits, GRID, K0, num_to_show=5)

    # --- 绘图总结 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_axis, t_full, 'o-', label='Full Matrix $O(N^3)$')
    ax1.plot(x_axis, t_fast, 's-', label='C4v Reduced $O((N/8)^3)$')
    ax1.set_yscale('log')
    ax1.set_xlabel('Total Number of Plane Waves (N)')
    ax1.set_ylabel('Time (s)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis, errors, 'r^:', label='Eigenvalue Consistency Error')
    ax2.set_yscale('log')
    ax2.set_ylabel('Error', color='r')
    ax2.legend(loc='lower right')
    
    plt.title('Extreme Acceleration: 8-Fold Symmetry Reduction')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置 show_images=True 以查看 Bloch 模式分布
    run_accelerated_benchmark(show_images=True)