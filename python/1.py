import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def solve_laplacian_3d_refined(f, Lx, Ly, Lz):
    Nx, Ny, Nz = f.shape
    
    # 1. 计算波数
    # x, y 使用 FFT 波数 (周期性)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    # z 使用 DCT 波数 (Neumann)
    kz = np.pi * np.arange(Nz) / Lz
    
    # 2. 构造 3D 波数矩阵
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # 3. 执行前向变换
    # FFT (x,y) -> DCT (z)
    f_hat = np.fft.fftn(f, axes=(0, 1))
    f_tilde = dct(f_hat, axis=2, type=2, norm='ortho')
    
    # 4. 频域应用算子: -(kx^2 + ky^2 + kz^2)
    laplace_kernel = -(KX**2 + KY**2 + KZ**2)
    u_tilde = laplace_kernel * f_tilde
    
    # 5. 执行逆变换
    # IDCT (z) -> IFFT (x,y)
    u_hat = idct(u_tilde, axis=2, type=2, norm='ortho')
    u = np.fft.ifftn(u_hat, axes=(0, 1))
    
    return u.real

# --- 参数与网格设置 ---
Nx, Ny, Nz = 64, 64, 64
Lx, Ly, Lz = 2*np.pi, 2*np.pi, 1.0

# x, y 周期性采样
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
# z 必须使用 Cell-Centered 采样以匹配 DCT-II
z = (np.arange(Nz) + 0.5) * (Lz / Nz)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 测试函数 f: 满足 x,y 周期, z 方向导数为0 (cos(pi*z/Lz))
f = np.sin(X) * np.cos(Y) * np.cos(np.pi * Z / Lz)

# --- 计算 ---
u_numeric = solve_laplacian_3d_refined(f, Lx, Ly, Lz)

# 理论值推导: -(1^2 + 1^2 + (pi/Lz)^2) * f
analytical_factor = -(1**2 + 1**2 + (np.pi/Lz)**2)
u_analytical = analytical_factor * f

# --- 误差检查 ---
error = np.max(np.abs(u_numeric - u_analytical))
print(f"修正后的最大绝对误差: {error:.2e}")

# --- 可视化 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 取 z = Lz/2 处的切面进行对比 (即 index = Nz//2)
z_idx = Nz // 2

im0 = axes[0].imshow(f[:, :, z_idx].T, extent=[0, Lx, 0, Ly], origin='lower')
axes[0].set_title(f"原始函数 $f(x,y,z_{{{z_idx}}})$")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u_numeric[:, :, z_idx].T, extent=[0, Lx, 0, Ly], origin='lower')
axes[1].set_title("数值解 $u = \\nabla^2 f$")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(u_numeric - u_analytical)[:, :, z_idx].T, 
                    extent=[0, Lx, 0, Ly], origin='lower', cmap='hot')
axes[2].set_title("绝对误差切面")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()