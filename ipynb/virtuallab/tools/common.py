#!/usr/bin/env python3
from virtuallab_ascii_to_ndarray import show_side_by_side, show_amplitude_phase, show_single
from virtuallab_fin_to_ndarray import parse_file, show_complex_plot
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../virtuallab/tools/refractiveindex/refractiveindex')
from download_material import *

def plot_cutlines_with_error(c1, c2, label1="Data 1", label2="Data 2"):
    """
    绘制两条 cutline，并将误差曲线叠加在同一张图上

    左侧 y 轴：原始曲线
    右侧 y 轴：误差曲线

    Parameters
    ----------
    c1, c2 : 1D ndarray
        两条 cutline
    """
    x = np.arange(len(c1))
    error = c1 - c2

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()                        # 主坐标轴（原始曲线）
    ax2 = ax1.twinx()                      # 右侧坐标轴（误差）

    # -----------------------------
    # Plot 原始曲线（左轴）
    # -----------------------------
    ax1.plot(x, c1, linewidth=2.0, label=label1)
    ax1.plot(x, c2, linewidth=2.0, linestyle='--', label=label2)
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.set_xlabel("Pixel index", fontsize=12)

    # -----------------------------
    # Plot 误差曲线（右轴）
    # -----------------------------
    ax2.plot(x, error, linewidth=2.0, color="black", alpha=0.7, label="Error")
    ax2.set_ylabel("Error", fontsize=12)

    # -----------------------------
    # 论文风格美化
    # -----------------------------
    # 坐标轴加粗
    for ax in [ax1, ax2]:
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)

    # 网格
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # 图例：原始曲线放左侧，误差放右侧
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Cutline Comparison with Error", fontsize=14)
    plt.tight_layout()
    plt.show()

def get_cutline(arr: np.ndarray, index: int, direction: str = "horizontal"):
    if direction == "horizontal":
        return arr[index, :]
    elif direction == "vertical":
        return arr[:, index]
    else:
        raise ValueError("direction 必须是 'horizontal' 或 'vertical'")
import numpy as np

def shift(F, dx, dy=None):
    F = np.asarray(F)
    dim = F.ndim

    if dim == 1:
        # 1D shift
        N = F.shape[0]
        k = np.fft.fftfreq(N) * N
        phase = np.exp(-1j * 2 * np.pi * k * dx / N)
        return np.fft.ifft(np.fft.fft(F) * phase)

    elif dim == 2:
        # 2D shift
        if dy is None:
            raise ValueError("For 2D input, shift(F, dx, dy) must provide dy.")

        F_freq = np.fft.fft2(F)
        ny, nx = F_freq.shape

        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny

        phase_x = np.exp(-1j * 2 * np.pi * kx * dx / nx)
        phase_y = np.exp(-1j * 2 * np.pi * ky * dy / ny)

        phase_2d = np.outer(phase_y, phase_x)

        return np.fft.ifft2(F_freq * phase_2d)

    else:
        raise ValueError("shift() only supports 1D or 2D arrays.")

