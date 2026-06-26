import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, eig, diag

# --- 1. 介质与模式准备 ---
def create_square_media(grid_size=64, square_size=20, eps_bg=2.0, eps_sq=5.0):
    eps_space = np.full((grid_size, grid_size), eps_bg)
    start = (grid_size - square_size) // 2
    end = start + square_size
    eps_space[start:end, start:end] = eps_sq
    return eps_space

def get_top_modes(epsilon_space, N, k0_vec=(0, 0), d_space=(1, 1)):
    nx, ny = epsilon_space.shape
    eps_fft = np.fft.fftshift(np.fft.fft2(epsilon_space)) / epsilon_space.size
    amplitude = np.abs(eps_fft)
    flat_idx = np.argpartition(amplitude.ravel(), -N)[-N:]
    idx_2d = [np.unravel_index(i, amplitude.shape) for i in flat_idx]
    
    dkx = 2 * np.pi / (nx * d_space[0])
    dky = 2 * np.pi / (ny * d_space[1])
    
    kx_list, ky_list, indices = [], [], []
    for ix, iy in idx_2d:
        kx_list.append(k0_vec[0] + (ix - nx//2) * dkx)
        ky_list.append(k0_vec[1] + (iy - ny//2) * dky)
        indices.append((ix, iy))
        
    return np.array(kx_list), np.array(ky_list), indices, eps_fft

def build_toeplitz(fft_coeffs, ids, nx, ny):
    """通用：将 FFT 系数转化为卷积矩阵 (Toeplitz 形式)"""
    N = len(ids)
    cx, cy = nx // 2, ny // 2
    mat = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            ix_diff = ids[i][0] - ids[j][0] + cx
            iy_diff = ids[i][1] - ids[j][1] + cy
            if 0 <= ix_diff < nx and 0 <= iy_diff < ny:
                mat[i, j] = fft_coeffs[ix_diff, iy_diff]
    return mat

def get_RCWA_matrices_0(k0, kx, ky, ids, nx, ny, eps_fft, inv_eps_fft):
    """
    为了对比，在一个函数内同时计算 E 形式和 H 形式的 M
    """
    N = len(kx)
    I = np.eye(N)
    Kx, Ky = np.diag(kx)/k0, np.diag(ky)/k0
    
    E = build_toeplitz(eps_fft, ids, nx, ny)
    P = inv(build_toeplitz(inv_eps_fft, ids, nx, ny)) # Li's Rule
    E_inv = inv(E)

    # 构造核心算符
    # Wh: H -> dE/dz
    Wh = np.block([[Kx@E_inv@Ky, I-Kx@E_inv@Kx], [Ky@E_inv@Ky-I, -Ky@E_inv@Kx]])
    # Ve: E -> dH/dz
    Ve = np.block([[Kx@Ky, E-Kx**2], [Ky**2-E, -Ky@Kx]])
    
    Me = (k0**2) * (Wh @ Ve)
    Mh = (k0**2) * (Ve @ Wh)
    
    return Me, Mh

def get_RCWA_matrices(k0, k_vectors, indices, nx, ny, eps_fft, inv_eps_fft):
    """
    精确的 RCWA 耦合矩阵构造 (使用 Li's Rule)
    Me: 电场耦合矩阵 (2N x 2N)
    Mh: 磁场耦合矩阵 (2N x 2N)
    """
    kx, ky = k_vectors
    N = len(kx)
    I = np.eye(N)
    
    # 归一化横向波矢
    Kx = np.diag(kx) / k0
    Ky = np.diag(ky) / k0
    
    # 1. 构造卷积矩阵
    E = build_toeplitz(eps_fft, indices, nx, ny)          # 用于 Laurent Rule
    # 核心修正：P_mat 才是处理不连续性的正确算符
    P_mat = inv(build_toeplitz(inv_eps_fft, indices, nx, ny)) 

    # 2. 构造 P 矩阵 (磁场 H 到电场 E 的映射算符)
    # 在这里，所有的 E_inv 应该被 P_mat 替换
    P = np.block([
        [Kx @ P_mat @ Ky,          I - Kx @ P_mat @ Kx],
        [Ky @ P_mat @ Ky - I,     -Ky @ P_mat @ Kx]
    ])

    # 3. 构造 Q 矩阵 (电场 E 到磁场 H 的映射算符)
    Q = np.block([
        [Kx @ Ky,                  E - Kx**2],
        [Ky**2 - E,               -Ky @ Kx]
    ])

    # 4. 构造主矩阵
    Me = (k0**2) * (P @ Q)
    Mh = (k0**2) * (Q @ P)
    
    return Me, Mh

def test_convergence(N_modes = 40):
    k0 = 10.0
    grid_size = 64

    # 构造高折射率对比介质 (n=1.0 vs n=3.5)
    eps_space = np.ones((grid_size, grid_size), dtype=complex)
    eps_space[22:42, 22:42] = 12.25  # Si 的介电常数
    
    # 获取频谱信息
    # (假设 get_top_modes 函数已定义)
    kx_v, ky_v, ids, eps_f = get_top_modes(eps_space, N_modes)
    _, _, _, inv_eps_f = get_top_modes(1.0/eps_space, N_modes)

    # 计算矩阵
    Me, Mh = get_RCWA_matrices(k0, (kx_v, ky_v), ids, grid_size, grid_size, eps_f, inv_eps_f)
    
    # 求解 kz
    vals_e = np.sort(np.abs(np.sqrt(eig(Me)[0] + 0j)))[::-1]
    
    print(f"模式数 N={N_modes} 时，前 5 个传播常数 kz:")
    print(vals_e[:5])

for mode in range(5, 10, 2):
    test_convergence(mode)

# --- 测试脚本 ---
k0 = 2 * np.pi / 0.5  # 波长 0.5um
grid_size = 64
N_modes = 40

# 1. 构造带有吸收的介质 (复数 epsilon)
# 假设是一个吸收很强的金属块: eps = -10 + 2j
eps_bg = 1.0 + 0j
eps_metal = -10.0 + 2.0j 
eps_space = np.full((grid_size, grid_size), eps_bg, dtype=complex)
eps_space[22:42, 22:42] = eps_metal

# 2. 获取 FFT 和模式
from __main__ import get_top_modes # 假设之前的函数在作用域内
kx_v, ky_v, ids, eps_fft = get_top_modes(eps_space, N_modes)
_, _, _, inv_eps_fft = get_top_modes(1.0/eps_space, N_modes)

# 3. 计算两种矩阵
Me, Mh = get_RCWA_matrices(k0, (kx_v, ky_v), ids, grid_size, grid_size, eps_fft, inv_eps_fft)
def reconstruct_mode(coeffs, ids, grid_size):
    """将倒空间特征向量系数重构为实空间分布"""
    # coeffs 长度为 2*N，前 N 个是 Ex(或Hx)，后 N 个是 Ey(或Hy)
    N = len(ids)
    field_x_fft = np.zeros((grid_size, grid_size), dtype=complex)
    field_y_fft = np.zeros((grid_size, grid_size), dtype=complex)
    
    for i, (ix, iy) in enumerate(ids):
        field_x_fft[ix, iy] = coeffs[i]
        field_y_fft[ix, iy] = coeffs[i + N]
    
    # 执行逆傅里叶变换回到实空间
    field_x = np.fft.ifft2(np.fft.ifftshift(field_x_fft))
    field_y = np.fft.ifft2(np.fft.ifftshift(field_y_fft))
    
    # 返回总强度分布 (E^2 或 H^2)
    return np.sqrt(np.abs(field_x)**2 + np.abs(field_y)**2)

# --- 核心计算逻辑修复 ---

# 4. 求解特征值并排序 (非常重要：确保 E 和 H 模式一一对应)
evals_e, evecs_e = eig(Me)
evals_h, evecs_h = eig(Mh)

# 按特征值的实部（传播常数）排序
idx_e = np.argsort(np.abs(np.sqrt(evals_e + 0j)))[::-1] # 从传播模到截止模
idx_h = np.argsort(np.abs(np.sqrt(evals_h + 0j)))[::-1]


def plot_bloch_modes(evecs_e, idx_e, evals_e, evecs_h, idx_h, evals_h, ids, grid_size, mode_from = 0,num_to_show=5,step=2):
    """
    在一张画布上对比显示前 N 个模式
    第一行：E-form 结果
    第二行：H-form 结果
    """
    fig, axes = plt.subplots(2, num_to_show, figsize=(4 * num_to_show, 8))
    
    # 确保当 num_to_show=1 时，axes 也是 2D 数组
    if num_to_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(0,num_to_show):
        # --- 计算 E 形式空间分布 ---
        ve = evecs_e[:, idx_e[mode_from + i*step]]
        mode_e_spatial = reconstruct_mode(ve, ids, grid_size)
        kz_e = np.sqrt(evals_e[idx_e[mode_from +i*step]] + 0j)
        
        ax_e = axes[0, i]
        im_e = ax_e.imshow(mode_e_spatial, cmap='viridis', origin='lower')
        ax_e.set_title(f"E-Mode {mode_from +i*step}\n$k_z$: {kz_e:.2f}")
        plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
        
        # --- 计算 H 形式空间分布 ---
        vh = evecs_h[:, idx_h[mode_from +i*step]]
        mode_h_spatial = reconstruct_mode(vh, ids, grid_size)
        kz_h = np.sqrt(evals_h[idx_h[mode_from +i*step]] + 0j)
        
        ax_h = axes[1, i]
        im_h = ax_h.imshow(mode_h_spatial, cmap='viridis', origin='lower')
        ax_h.set_title(f"H-Mode {mode_from +i*step}\n$k_z$: {kz_h:.2f}")
        plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)

        # 隐藏坐标轴标签以增加清晰度
        if i == 0:
            axes[0, i].set_ylabel("E-form ($|E|^2$)", fontsize=12, fontweight='bold')
            axes[1, i].set_ylabel("H-form ($|H|^2$)", fontsize=12, fontweight='bold')
        else:
            ax_e.set_yticks([])
            ax_h.set_yticks([])
        ax_e.set_xticks([])
        ax_h.set_xticks([])

    plt.tight_layout()
    plt.show()

# --- 调用执行 ---
print(f"eigen count={len(evals_e)}")
plot_bloch_modes(evecs_e, idx_e, evals_e, evecs_h, idx_h, evals_h, ids, grid_size, mode_from = 0, num_to_show=N_modes//2, step=2)


def kz(eigenvalues):
    kz = np.sqrt(eigenvalues + 0j)
    # 3. 物理符号修正 (确保波向 +z 方向传播且能量不发散)
    # 传播波: Re(kz) > 0; 衰减波: Im(kz) > 0
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz
def phase_propgation(kz, dz):
    return np.exp(1j * kz * dz)
def field(eigenvectors, phase_factor, coefficients, is_spectrum_domain = True):
    E_at_dz = eigenvectors @ diag(phase_factor) @ coefficients
    if not is_spectrum_domain: E_at_dz = np.fft.ifft2(E_at_dz) # TODO : normalize
    return E_at_dz