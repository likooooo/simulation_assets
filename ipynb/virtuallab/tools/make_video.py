#!/usr/bin/env python3
from typing import Callable, List, Dict, Any
from matplotlib import cm
from tqdm import tqdm  
import numpy as np
import cv2
def make_video(
    output_filename: str,
    frame_callback: Callable[..., np.ndarray],
    sweep_table: List[Dict[str, Any]],
    frame_duration: float = 0.1,
    normalize_to_uint8: bool = True,
    use_rdbu_colormap: bool = True,
    draw_params: bool = True
):
    """
    生成一个基于参数扫描的视频（包含自适应字体大小）。
    """

    if not sweep_table:
        print("Error: sweep_table 为空。")
        return

    fps = 1.0 / frame_duration

    print("正在生成第一帧以确定视频规格...")
    first_args = sweep_table[0]
    first_frame = frame_callback(**first_args)

    height, width = first_frame.shape[:2]
    print(f"视频分辨率: {width}x{height}, FPS: {fps:.2f}, 总帧数: {len(sweep_table)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("Error: 无法打开 VideoWriter。")
        return

    # Matplotlib colormap
    cmap = cm.get_cmap("RdBu_r")

    # --- 预先计算字体参数 (基于视频高度) ---
    if draw_params:
        # 设定基准：假设图像高度为 600px 时，字体大小为 0.6
        # 设定最小值：0.4，防止在极小图像上文字无法渲染
        scale_factor = height / 600.0
        font_scale = max(0.4, scale_factor * 0.6)
        
        # 字体粗细也需要自适应 (至少为 1)
        thickness = max(1, int(font_scale * 2))
        
        # 计算单行文字的高度，用于行间距
        # 使用 "Ay" 这种包含上下伸出部分的字符来测算最大高度
        (text_w, text_h), baseline = cv2.getTextSize("Ay", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        line_height = int(text_h * 1.6)  # 行间距设为文字高度的 1.6 倍
        margin_x = int(10 * max(1, scale_factor)) # 左边距也自适应

    for i, args in enumerate(tqdm(sweep_table, desc="渲染视频中")):

        img = frame_callback(**args)

        # --- 浮点图像 RdBu_r 映射处理 ---
        if (img.dtype == np.float32 or img.dtype == np.float64) and use_rdbu_colormap:
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                img_norm = (img - vmin) / (vmax - vmin)
            else:
                img_norm = np.zeros_like(img)

            img_color = cmap(img_norm)
            img_color = (img_color[:, :, :3] * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

        else:
            if normalize_to_uint8 and (img.dtype == np.float32 or img.dtype == np.float64):
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img

        # --- 在左上角写 sweep_table 参数 (自适应大小) ---
        if draw_params:
            text_lines = [f"{k}={v}" for k, v in args.items()]
            
            # 初始 Y 坐标 (第一行的位置)
            y_pos = line_height 

            for line in text_lines:
                # 1. 绘制黑色描边 (thickness + 2, 黑色)
                cv2.putText(
                    img_bgr,
                    line,
                    (margin_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),       # 黑色
                    thickness + 3,   # 描边要比字体粗
                    cv2.LINE_AA
                )
                
                # 2. 绘制白色文字 (thickness, 白色)
                cv2.putText(
                    img_bgr,
                    line,
                    (margin_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255), # 白色
                    thickness,
                    cv2.LINE_AA
                )
                
                y_pos += line_height

        video_writer.write(img_bgr)

    video_writer.release()
    print(f"\n视频已成功保存至: {output_filename}")
    
# ----------------------------------------------------------------
# 下面是一个使用示例
# ----------------------------------------------------------------

def demo_usage():
    # 1. 定义 frame_callback
    # 假设我们要画一个随时间移动和变大的圆，并在 2D 平面上显示波纹
    def my_renderer(x_center, y_center, radius, frequency):
        # 创建一个 500x500 的画布
        h, w = 500, 500
        x = np.linspace(-10, 10, w)
        y = np.linspace(-10, 10, h)
        xv, yv = np.meshgrid(x, y)
        
        # 生成一个干涉波纹图案 (返回 0.0 - 1.0 的 float 矩阵)
        dist = np.sqrt((xv - x_center)**2 + (yv - y_center)**2)
        z = 0.5 + 0.5 * np.sin(dist * frequency)
        
        # 我们可以在 2D 数组上画个圆 (稍微 hack 一下，直接操作像素)
        # 将 float 转回 0-255 用于绘图测试
        img_uint8 = (z * 255).astype(np.uint8)
        
        # 使用 OpenCV 画一个实心圆
        cv2.circle(img_uint8, (int(250 + x_center*20), int(250 + y_center*20)), int(radius), (0), -1)
        
        return img_uint8 # 返回 2D ndarray

    # 2. 准备 Sweep Table
    # 我们生成 100 帧的参数
    sweep_data = []
    for i in range(100):
        sweep_data.append({
            "x_center": 5 * np.sin(i * 0.1),  # x 坐标正弦摆动
            "y_center": 5 * np.cos(i * 0.1),  # y 坐标余弦摆动
            "radius": 20 + i * 0.5,           # 半径逐渐变大
            "frequency": 1.0 + i * 0.05       # 波纹频率变高
        })

    # 3. 生成视频
    # 每帧持续 0.04秒 (即 25 FPS)
    make_video(
        output_filename="sweep_demo.mp4",
        frame_callback=my_renderer,
        sweep_table=sweep_data,
        frame_duration=0.04
    )

if __name__ == "__main__":
    demo_usage()