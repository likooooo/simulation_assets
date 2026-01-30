#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.animation as animation

def get_rms(result, target):
    """计算 RMS"""
    result = np.array(result)
    target = np.array(target)
    return np.sqrt(np.mean(np.abs((result - target) ** 2)))

def cost_function(param, calc_result, target):
    """计算 cost (RMS)"""
    result = calc_result(param)
    return get_rms(result, target)

def search_best_coef(calc_result, target, norminal_list):
    """粗搜索：在 norminal_list 中找到最优参数"""
    cost_list = []

    for norminal in norminal_list:
        cost = cost_function(norminal, calc_result, target)
        cost_list.append(cost)

    # 找到最优点
    min_idx = np.argmin(cost_list)
    best_param = norminal_list[min_idx]
    best_cost = cost_list[min_idx]

    # 绘制结果
    plt.figure(figsize=(6, 4))
    plt.plot(norminal_list, cost_list, marker='o')
    plt.scatter(best_param, best_cost, color='red', zorder=5, label=f'Best param = {best_param:.3f}')
    plt.xlabel('Norminal')
    plt.ylabel('Cost (RMS)')
    plt.title('Parameter Search Result')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_param, best_cost

def finer_search(calc_result, target, search_range):
    """精细搜索：在 search_range 内找到更精确的最优参数"""
    res = minimize_scalar(
        lambda x: cost_function(x, calc_result, target),
        bounds=search_range,
        method='bounded'
    )
    return res.x, res.fun

def finer_search_with_plot(calc_result, target, search_range, tol=1e-5, max_iter=50):
    """黄金分割搜索 + 绘制过程"""
    a, b = search_range
    phi = (np.sqrt(5) - 1) / 2  # 黄金比例系数
    history = []  # 记录搜索过程

    # 初始化两个点
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = cost_function(x1, calc_result, target)
    f2 = cost_function(x2, calc_result, target)

    for i in range(max_iter):
        history.append((a, b, x1, x2, f1, f2))

        if abs(b - a) < tol:
            break

        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = a + phi * (b - a)
            f2 = cost_function(x2, calc_result, target)
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = b - phi * (b - a)
            f1 = cost_function(x1, calc_result, target)

    # 最优点
    best_param = (a + b) / 2
    best_cost = cost_function(best_param, calc_result, target)

    # 绘制 cost 曲线 + 搜索过程
    xs = np.linspace(search_range[0], search_range[1], 200)
    ys = [cost_function(x, calc_result, target) for x in xs]

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, label='Cost Function')
    for step, (a_k, b_k, x1_k, x2_k, f1_k, f2_k) in enumerate(history):
        plt.axvline(x1_k, color='orange', linestyle='--', alpha=0.3)
        plt.axvline(x2_k, color='green', linestyle='--', alpha=0.3)
        plt.plot(x1_k, f1_k, 'o', color='orange', alpha=0.5)
        plt.plot(x2_k, f2_k, 'o', color='green', alpha=0.5)
        plt.text(x1_k, f1_k, f'{step}', fontsize=8, color='orange')
        plt.text(x2_k, f2_k, f'{step}', fontsize=8, color='green')

    plt.axvline(best_param, color='red', linestyle='-', label=f'Best param={best_param:.5f}')
    plt.xlabel('Parameter')
    plt.ylabel('Cost (RMS)')
    plt.title('Golden Section Search Process')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_param, best_cost

def finer_search_with_animation(calc_result, target, search_range, tol=1e-5, max_iter=50):
    """黄金分割搜索 + 动态可视化"""
    a, b = search_range
    phi = (np.sqrt(5) - 1) / 2
    history = []

    # 初始化两个点
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = cost_function(x1, calc_result, target)
    f2 = cost_function(x2, calc_result, target)

    for i in range(max_iter):
        history.append((a, b, x1, x2, f1, f2))
        if abs(b - a) < tol:
            break
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = a + phi * (b - a)
            f2 = cost_function(x2, calc_result, target)
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = b - phi * (b - a)
            f1 = cost_function(x1, calc_result, target)

    best_param = (a + b) / 2
    best_cost = cost_function(best_param, calc_result, target)

    # 生成 cost 曲线
    xs = np.linspace(search_range[0], search_range[1], 400)
    ys = [cost_function(x, calc_result, target) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, label="Cost Function", color='blue')
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Cost (RMS)")
    ax.set_title("Golden Section Search Animation")
    ax.grid(True)

    # 动画元素
    line_a = ax.axvline(search_range[0], color='gray', linestyle='--', alpha=0.7)
    line_b = ax.axvline(search_range[1], color='gray', linestyle='--', alpha=0.7)
    point_x1, = ax.plot([], [], 'o', color='orange')
    point_x2, = ax.plot([], [], 'o', color='green')
    text_step = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        point_x1.set_data([], [])
        point_x2.set_data([], [])
        text_step.set_text('')
        return line_a, line_b, point_x1, point_x2, text_step

    def update(frame):
        a_k, b_k, x1_k, x2_k, f1_k, f2_k = history[frame]
        line_a.set_xdata(a_k)
        line_b.set_xdata(b_k)
        point_x1.set_data(x1_k, f1_k)
        point_x2.set_data(x2_k, f2_k)
        text_step.set_text(f"Step {frame}")
        return line_a, line_b, point_x1, point_x2, text_step

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), init_func=init,
        blit=True, repeat=False, interval=800
    )

    plt.legend()
    plt.show()

    return best_param, best_cost

def search(calc_result, target, norminal_list):
    # 先粗搜索
    best_param, best_cost = search_best_coef(calc_result, target, norminal_list)
    print(f"粗搜索结果: 最佳参数={best_param:.4f}, 最小 RMS={best_cost:.6f}")

    # # 再细搜索（假设我观察图形后，选择区间在 2.5 到 3.5）
    # fine_param, fine_cost = finer_search(calc_result, target, (2.5, 3.5))
    # print(f"细搜索结果: 最佳参数={fine_param:.6f}, 最小 RMS={fine_cost:.6f}")
    
    best_param, best_cost = \
        [finer_search_with_plot, finer_search_with_animation][1](calc_result, target, (2.5, 3.5))
    print(f"最佳参数: {best_param:.6f}, 最小 RMS: {best_cost:.6f}")

if __name__ == "__main__":
    # 模拟目标函数
    def calc_result(x):
        return np.array([x, x + 1, x + 2])  # 模拟计算结果
    target = np.array([3, 4, 5])
    norminal_list = np.linspace(0, 5, 20)
    search(calc_result, target, norminal_list)