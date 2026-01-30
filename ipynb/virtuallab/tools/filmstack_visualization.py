import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Tuple, List
import colorsys

class MockMaterial:
    def __init__(self, name, nk):
        self.name = name
        self.nk = nk

class MockFilm:
    def __init__(self, material, thickness):
        self.material = material
        self.d = thickness
        self.name = material.name
        self.nk = material.nk


# 计算角度 (Snell's Law)
def calculate_angles(layer_list, initial_theta):
    angles = []
    n0 = layer_list[0].nk.real 
    sne_const = n0 * np.sin(initial_theta)
    for layer in layer_list:
        sin_th = sne_const / layer.nk
        theta = np.arcsin(sin_th)
        angles.append(theta)
    return angles


def plot_periodic_structure(layers, angles, color_map, angle_deg = -1, title = "Multilayer Ray Tracing",visual_width = -1, inf_display_height = 100):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 提取薄膜层（去掉头尾的 Air 和 Si）
    films = layers[1:-1] 
    total_film_thickness = sum(f.d for f in films)
    
    # --- Z 轴方向修正逻辑 ---
    # 目标：Air/Film 1 界面是 Z=0。薄膜和基底区域 Z 轴为正。
    
    layer_coords = []
    
    # Air 层的上表面 (top) 设为 -inf_display_height
    current_top = -inf_display_height 
    z_zero_level = 0 
    
    # 1. 计算 Air 层的坐标
    # Air 层：从 -H 到 0 (Z轴为负)
    top = current_top
    bottom = z_zero_level # Air 层的下表面是 Z=0
    layer_coords.append((top, bottom)) 
    current_top = bottom # 下一层 Film 1 的 top = 0

    # 2. 计算 Films 和 Substrate 的坐标
    # Films/Substrate 层：从 0 开始向下延伸 (Z轴为正)
    for layer in layers[1:]:
        height = inf_display_height if layer.d == float('inf') else layer.d
        top = current_top
        bottom = current_top + height # 注意：这里是加法，因为 Z 轴正方向是向下
        layer_coords.append((top, bottom))
        current_top = bottom 

    # --- Visual Width 的默认值 ---
    if -1 == visual_width: 
        # 使用总膜层厚度的 1.5 倍作为默认宽度
        visual_width = max(500, total_film_thickness * 1.5) 

    x_min = -visual_width / 2
    x_max = visual_width / 2
    
    
    current_ray_x = 0 
    
    legend_handles = {}

    for i, (layer, theta) in enumerate(zip(layers, angles)):
        top_y, bottom_y = layer_coords[i]
        mat_name = layer.name
        
        # 1. 绘制层背景 (全宽)
        # 注意：矩形的高度计算仍然是 abs(top_y - bottom_y)
        rect_height = abs(top_y - bottom_y)
        # 矩形的左下角坐标是 (x_min, min(top_y, bottom_y))
        rect_y_start = min(top_y, bottom_y)
        
        rect = patches.Rectangle(
            (x_min, rect_y_start), 
            visual_width, 
            rect_height,
            linewidth=0,
            facecolor=color_map.get(mat_name, "#CCCCCC"),
            alpha=0.85
        )
        ax.add_patch(rect)
        if mat_name not in legend_handles: legend_handles[mat_name] = rect
        
        # 绘制层级线 (仍然是 bottom_y, 因为它是下一层的 top_y)
        ax.axhline(bottom_y, color='k', lw=0.5, alpha=0.2)
        
        # 2. 标注厚度 (右侧)
        d_text = "Inf" if layer.d == float('inf') else f"{layer.d:.1f} nm"
        label_y = (top_y + bottom_y) / 2
        ax.text(x_max * 0.95, label_y, f"{mat_name}\n{d_text}", 
                va='center', ha='right', fontsize=8, color='#444')

        # 3. 角度标注内容 (Re(theta) in degrees)
        theta_real = np.real(theta)
        deg_val = np.degrees(theta_real)
        angle_text = f"{deg_val:.1f}°"
        
        # 4. 光线追踪逻辑
        
        # === Case A: 第一层 (Air) ===
        if i == 0:
            # 入射角画在底部 (bottom_y = 0)
            arrow_len = min(x_max*0.3, inf_display_height * 0.8)
            # 逆推起点 (Air在负Z轴，所以要向上推)
            start_x = -arrow_len * np.sin(theta_real)
            start_y = bottom_y - arrow_len * np.cos(theta_real) # 注意：从0向上（负Z）减
            
            ax.annotate("", xy=(0, bottom_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2))
            
            ax.text(start_x, start_y, f"Incidence\n{angle_text}", 
                    color="darkred", fontsize=9, va="bottom", ha="center")
            
            # 下一层的起点 X 坐标
            current_ray_x = 0 
            
        # === Case B: 中间层 & 基底 ===
        else:
            # 这里的任务是：从 (current_ray_x, top_y) 走到 y = bottom_y
            # Y 轴向下是正向，layer_h = bottom_y - top_y
            
            layer_h = bottom_y - top_y # 垂直距离
            
            # 计算这一层需要的总水平位移
            total_dx = layer_h * np.tan(theta_real)
            
            # 我们需要分段画线
            seg_start_x = current_ray_x
            seg_start_y = top_y
            
            # 剩余需要走的垂直距离
            dist_y_remaining = layer_h
            
            segments = 0
            while dist_y_remaining > 0.1 and segments < 10:
                segments += 1
                
                # 计算如果走完剩余垂直距离，X 会去哪里
                # 因为 tan(theta_real) 已经在计算 total_dx 时包含了方向，直接加即可
                theoretical_end_x = seg_start_x + dist_y_remaining * np.tan(theta_real)
                
                # 检查是否越界
                hit_boundary = False
                boundary_x = 0
                
                if theoretical_end_x > x_max:
                    hit_boundary = True
                    boundary_x = x_max
                elif theoretical_end_x < x_min:
                    hit_boundary = True
                    boundary_x = x_min
                
                if not hit_boundary:
                    # 如果不越界，直接画到底
                    # Y 轴向下是正向，所以 bottom_y > top_y
                    ax.plot([seg_start_x, theoretical_end_x], 
                            [seg_start_y, bottom_y], 
                            color="red", lw=1.5)
                    
                    if segments == 1:
                        mid_x = (seg_start_x + theoretical_end_x)/2
                        mid_y = (seg_start_y + bottom_y)/2
                        ax.text(mid_x, mid_y, angle_text, color="darkred", 
                                fontsize=8, ha="left", va="bottom", weight='bold')
                    
                    current_ray_x = theoretical_end_x
                    dist_y_remaining = 0 
                    
                else:
                    # 如果越界
                    # dx_to_wall = boundary_x - seg_start_x
                    # dy_moved = dx_to_wall / tan(theta)
                    dx_to_wall = boundary_x - seg_start_x
                    dy_moved = dx_to_wall / np.tan(theta_real) if np.tan(theta_real) != 0 else 0
                    
                    hit_y = seg_start_y + dy_moved # Y 轴向下是正向，所以是加

                    ax.plot([seg_start_x, boundary_x], [seg_start_y, hit_y], color="red", lw=1.5)
                    
                    if segments == 1:
                        ax.text((seg_start_x+boundary_x)/2, (seg_start_y+hit_y)/2, 
                                angle_text, color="darkred", fontsize=8, ha="left")

                    # 瞬移到另一侧
                    if boundary_x == x_max:
                        seg_start_x = x_min 
                    else:
                        seg_start_x = x_max 
                    
                    seg_start_y = hit_y
                    
                    # 更新剩余垂直距离
                    dist_y_remaining -= dy_moved
                    
                    current_ray_x = seg_start_x 

    # --- 图像修饰 ---
    ax.set_xlim(x_min, x_max)
    # y轴范围：从 Air 顶部 (最小负数) 到 基底底部 (最大正数)
    ax.set_ylim(layer_coords[0][0], layer_coords[-1][1]) 
    
    ax.set_xlabel("Lateral Position [Periodic Boundary]")
    ax.set_ylabel("Z Position")
    if -1 != angle_deg:
        ax.set_title(f"{title} (Incidence: {angle_deg}°)")
    else :
        ax.set_title(f"{title}")

    # 合并图例
    handles, labels = plt.gca().get_legend_handles_labels()
    for name, rect in legend_handles.items():
        handles.append(rect)
        labels.append(name)
        
    ax.legend(handles, labels, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.show()
    
def nk_to_color(
    n: float, 
    k: float, 
    n_min: float = 1.0, 
    n_max: float = 5.0, 
    k_max: float = 3.0
) -> Tuple[float, float, float]:
    """
    根据材料的折射率 (n) 和消光系数 (k) 返回一个 RGB 颜色。
    
    该函数将 n 映射到色相 (Hue)，k 映射到饱和度 (Saturation) 或明度 (Value)，
    以实现较好的可视化区分度。
    
    Args:
        n: 材料的折射率（实部）。
        k: 材料的消光系数（虚部）。
        n_min: n 值的预期最小值，用于色相归一化。
        n_max: n 值的预期最大值，用于色相归一化。
        k_max: k 值的预期最大值，用于亮度/饱和度归一化。

    Returns:
        一个 RGB 颜色三元组 (R, G, B)，每个分量在 [0.0, 1.0] 范围内。
    """
    
    # 1. 归一化 n (映射到色相 H)
    # 使用 np.clip 确保 n_norm 在 [0, 1] 范围内
    n_norm = np.clip((n - n_min) / (n_max - n_min), 0.0, 1.0)
    
    # 将 n_norm 映射到色相 H (0.0 到 1.0)
    # 不同的映射函数可以产生不同的色谱，这里使用线性映射
    H = n_norm
    
    # 2. 归一化 k (映射到饱和度 S 和明度 V)
    # k 越大，材料吸收越强 (金属、半导体)，k 越小，材料越透明 (介质)。
    k_norm = np.clip(k / k_max, 0.0, 1.0)
    
    # 策略：
    # - 吸收强 (k 大) 的材料，颜色应该更“浓烈”或“深沉”。
    # - 吸收弱 (k 小) 的材料，颜色应该更“浅”或“亮”。
    
    # 使用 K 来控制饱和度和亮度，实现对比度：
    # - 介质 (k -> 0): 饱和度 S 设为中高 (0.6)，亮度 V 设为高 (0.9)。
    # - 强吸收体 (k -> k_max): 饱和度 S 设为高 (1.0)，亮度 V 设为中低 (0.5)，使颜色变深。
    
    S = 0.6 + 0.4 * k_norm  # k 越大，饱和度 S 越高
    V = 0.9 - 0.4 * k_norm  # k 越大，亮度 V 越低
    
    # 3. 从 HSV 转换到 RGB
    r, g, b = colorsys.hsv_to_rgb(H, S, V)
    
    return (r, g, b)

def plot_tmm_filmstack(meterial_film_list, meterial_film_name_list, angle_deg = 45, visual_width = -1, inf_display_height = 100):
    degree = np.pi / 180
    th_0 = angle_deg * degree

    def make_mock_material(i):
        return MockMaterial(meterial_film_name_list[i], meterial_film_list[i].nk)

    top       = make_mock_material(0)
    substrate = make_mock_material(-1)

    layers = [MockFilm(top, float('inf'))] 
    for i in range(1, len(meterial_film_list) - 1):
        layers.append(MockFilm(make_mock_material(i), meterial_film_list[i].depth))
    layers.append(MockFilm(substrate, float('inf')))

    dir_list = calculate_angles(layers, th_0)
    color_map = dict()
    nmin, nmax = min([np.real(l.nk) for l in  layers]), max([np.real(l.nk) for l in  layers])
    k_max = max([np.imag(l.nk) for l in  layers])
    for layer in layers:
        if layer.name in color_map: continue
        color_map[layer.name] = nk_to_color(np.real(layer.nk), np.imag(layer.nk), nmin, nmax, k_max)

    plot_periodic_structure(layers, dir_list, color_map=color_map, angle_deg=angle_deg, title="TMM filmstack",visual_width=visual_width, inf_display_height=inf_display_height)