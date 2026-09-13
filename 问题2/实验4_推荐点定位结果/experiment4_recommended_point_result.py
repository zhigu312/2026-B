import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Circle

# Import from experiment 2 for geometry functions
EXP2_DIR = Path(__file__).resolve().parent.parent / "实验二_定位性能空间分布"
sys.path.append(str(EXP2_DIR))
import experiment2_worst_case_heatmap as exp2

OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "图5_最终推荐第二检测点下的二次交会定位结果.png"
PDF_PATH = OUTPUT_DIR / "图5_最终推荐第二检测点下的二次交会定位结果.pdf"

def configure_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42

def unit_vector(angle_rad: float) -> np.ndarray:
    return np.array([math.cos(angle_rad), math.sin(angle_rad)])

def draw_figure() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot()
    
    p = exp2.P
    s1 = np.array([p.s1_x, p.s1_y], dtype=float)
    s2 = np.array([470.438289, 290.520546], dtype=float)
    
    alpha = math.radians(p.angle_error_deg)
    
    # 第一次有效域
    first_bearing = math.radians(p.first_bearing_deg)
    first_region = exp2.bounded_angle_triangle(s1, first_bearing, alpha, p.max_detection_radius)
    
    # 最坏情况：距离 1500，首次测向误差 +1度， 二测误差 +1度
    target_distance = 1500.0
    first_offset_rad = math.radians(1.0)
    target_angle = first_bearing + first_offset_rad
    target = s1 + target_distance * unit_vector(target_angle)
    
    second_offset_rad = math.radians(1.0)
    delta = target - s2
    true_second_bearing = math.atan2(delta[1], delta[0])
    measured_second_bearing = true_second_bearing + second_offset_rad
    second_region = exp2.bounded_angle_triangle(s2, measured_second_bearing, alpha, p.max_detection_radius)
    
    intersection = exp2.clip_convex_polygon(first_region, second_region)
    
    fig, ax = plt.subplots(figsize=(10.6, 9.0), constrained_layout=True)
    
    # 绘制第一个扇形
    angles_1 = np.linspace(first_bearing - alpha, first_bearing + alpha, 100)
    arc_1_outer = s1 + p.max_detection_radius * np.stack([np.cos(angles_1), np.sin(angles_1)], axis=-1)
    arc_1_inner = s1 + p.near_radius * np.stack([np.cos(angles_1[::-1]), np.sin(angles_1[::-1])], axis=-1)
    poly_1 = np.vstack([arc_1_outer, arc_1_inner])
    
    ax.add_patch(Polygon(poly_1, closed=True, facecolor="#5AA9E6", edgecolor="#1769AA", linewidth=1.5, alpha=0.3, zorder=2))
    
    # 绘制第二个扇形
    angles_2 = np.linspace(measured_second_bearing - alpha, measured_second_bearing + alpha, 100)
    arc_2_outer = s2 + p.max_detection_radius * np.stack([np.cos(angles_2), np.sin(angles_2)], axis=-1)
    arc_2_inner = s2 + p.near_radius * np.stack([np.cos(angles_2[::-1]), np.sin(angles_2[::-1])], axis=-1)
    poly_2 = np.vstack([arc_2_outer, arc_2_inner])
    
    ax.add_patch(Polygon(poly_2, closed=True, facecolor="#F3A83B", edgecolor="#D88927", linewidth=1.5, alpha=0.25, zorder=3))
    
    # 绘制交集区域
    if len(intersection) > 2:
        ax.add_patch(Polygon(intersection, closed=True, facecolor="#C43B3B", edgecolor="#8F2929", linewidth=2.0, alpha=0.7, zorder=5))
        
        # 旋转卡壳直径
        max_dist = 0.0
        pA, pB = None, None
        for i in range(len(intersection)):
            for j in range(i+1, len(intersection)):
                dist = np.linalg.norm(intersection[i] - intersection[j])
                if dist > max_dist:
                    max_dist = dist
                    pA, pB = intersection[i], intersection[j]
        if pA is not None and pB is not None:
            ax.plot([pA[0], pB[0]], [pA[1], pB[1]], color="#111827", linewidth=2.0, linestyle="--", zorder=7)
            # 添加直径标注
            midpoint = (pA + pB) / 2
            # Offset slightly
            normal = np.array([-(pB[1]-pA[1]), pB[0]-pA[0]])
            normal = normal / np.linalg.norm(normal)
            ax.annotate(rf"$D_{{wc}}={max_dist:.2f}\ \mathrm{{m}}$", 
                        xy=midpoint, xytext=midpoint + 60*normal, 
                        fontsize=12, color="#8F2929", fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color="#8F2929"), zorder=9)

    # 标注点 S1, S2, 目标 G
    ax.scatter(*s1, s=70, facecolor="#111827", edgecolor="white", zorder=8)
    ax.annotate(r"$S_1$", xy=s1, xytext=(-20, -10), textcoords="offset points", fontsize=12)
    
    ax.scatter(*s2, s=150, marker="*", facecolor="#C43B3B", edgecolor="white", zorder=8)
    ax.annotate(r"推荐第二检测点 $S_2$", xy=s2, xytext=(10, -20), textcoords="offset points", fontsize=11, color="#8F2929")
    
    ax.scatter(*target, s=80, marker="x", color="#111827", linewidth=2.0, zorder=8)
    ax.annotate("最坏情形真实目标 $G$", xy=target, xytext=(10, 10), textcoords="offset points", fontsize=11)
    
    # 候选圆盘 (q0, safe_radius)
    c0 = p.max_detection_radius / (2.0 * math.cos(alpha))
    q0 = s1 + c0 * unit_vector(first_bearing)
    safe_radius = p.guaranteed_radius - c0
    ax.add_patch(Circle(q0, safe_radius, fill=False, edgecolor="#087A4C", linewidth=2.0, linestyle="--", zorder=2))
    
    # 限制范围
    # focus more tightly if we want to show the detail of R2, but we also want to show S1 and S2
    ax.set_xlim(-100, 1600)
    ax.set_ylim(-100, 1600)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.35)
    ax.set_axisbelow(True)

    ax.set_title("图5: 最终推荐第二检测点下的二次交会定位结果", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("横坐标 x / m", fontsize=11.5)
    ax.set_ylabel("纵坐标 y / m", fontsize=11.5)
    
    legend_handles = [
        Patch(facecolor="#5AA9E6", edgecolor="#1769AA", alpha=0.3, label=r"首次目标可行域 $\Omega_1$"),
        Patch(facecolor="#F3A83B", edgecolor="#D88927", alpha=0.25, label=r"最坏情形二次测向可行域 $\Omega_2$"),
        Patch(facecolor="#C43B3B", edgecolor="#8F2929", alpha=0.7, label=r"二次定位区域 $\mathcal{R}_2$"),
        Line2D([0], [0], color="#111827", linewidth=2.0, linestyle="--", label="定位区域最坏直径"),
        Line2D([0], [0], color="#087A4C", linewidth=2.0, linestyle="--", label=r"保证接收候选区域边界 $\partial\mathcal{C}_{\rm safe}$")
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10.5, frameon=True, framealpha=0.96)
    
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    draw_figure()
    print("Saved Figure 5.")
