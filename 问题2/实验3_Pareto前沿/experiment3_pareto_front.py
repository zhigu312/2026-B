"""问题二实验三：第二检测位置的定位精度—移动时间 Pareto 前沿。

直接读取实验二的圆内网格结果，以 D_wc 和移动时间为两个
需同时最小化的目标。膝点由归一化 Pareto 前沿上各点到两端连线
的垂直距离最大准则选取，不引入人工权重。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = OUTPUT_DIR.parent / "实验二_定位性能空间分布" / "实验二_Dwc空间网格.csv"
PNG_PATH = OUTPUT_DIR / "图5_第二检测位置的定位精度_移动时间Pareto前沿.png"
PDF_PATH = OUTPUT_DIR / "图5_第二检测位置的定位精度_移动时间Pareto前沿.pdf"
DOUBLE_PNG_PATH = OUTPUT_DIR / "图5_第二检测位置选择与Pareto前沿_双子图.png"
DOUBLE_PDF_PATH = OUTPUT_DIR / "图5_第二检测位置选择与Pareto前沿_双子图.pdf"
HEATMAP_PNG_PATH = OUTPUT_DIR / "最坏二测定位直径热力图_膝点标注.png"
HEATMAP_PDF_PATH = OUTPUT_DIR / "最坏二测定位直径热力图_膝点标注.pdf"
CSV_PATH = OUTPUT_DIR / "实验三_Pareto候选点.csv"
REPORT_PATH = OUTPUT_DIR / "实验三结果说明.md"
MOVE_SPEED = 5.0
TOLERANCE = 1.0e-9


def load_candidates() -> np.ndarray:
    """读取实验二的圆内候选点，返回 [x, y, T_move, D_wc]。"""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"未找到实验二结果：{SOURCE_CSV}")
    records: list[list[float]] = []
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["inside_safe_circle"] != "1" or not row["D_wc_m"]:
                continue
            x = float(row["S2_x_m"])
            y = float(row["S2_y_m"])
            distance = math.hypot(x, y)
            records.append([x, y, distance / MOVE_SPEED, float(row["D_wc_m"])])
    if not records:
        raise ValueError("实验二结果中没有可用的圆内候选点。")
    return np.asarray(records, dtype=float)


def classify_pareto(objectives: np.ndarray) -> np.ndarray:
    """对两个最小化目标进行严格 Pareto 分类；相同目标的镜像点均保留。"""
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    pareto = np.zeros(len(objectives), dtype=bool)
    best_previous_d = math.inf
    position = 0
    while position < len(order):
        end = position + 1
        time_value = objectives[order[position], 0]
        while end < len(order) and abs(objectives[order[end], 0] - time_value) <= TOLERANCE:
            end += 1
        group = order[position:end]
        group_minimum = float(np.min(objectives[group, 1]))
        if group_minimum < best_previous_d - TOLERANCE:
            pareto[group[np.abs(objectives[group, 1] - group_minimum) <= TOLERANCE]] = True
            best_previous_d = group_minimum
        position = end
    return pareto


def unique_front(objectives: np.ndarray, pareto: np.ndarray) -> np.ndarray:
    """将目标空间中重合的镜像点合并，仅用于连线和膝点识别。"""
    front = objectives[pareto]
    rounded = np.round(front, decimals=9)
    return np.unique(rounded, axis=0)[np.argsort(np.unique(rounded, axis=0)[:, 0])]


def select_knee(front: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回膝点和各前沿点在归一化目标空间内的垂距。"""
    if len(front) < 3:
        return front[len(front) // 2], np.zeros(len(front))
    spans = np.ptp(front, axis=0)
    if np.any(spans <= TOLERANCE):
        return front[len(front) // 2], np.zeros(len(front))
    normalized = (front - np.min(front, axis=0)) / spans
    start, end = normalized[0], normalized[-1]
    chord = end - start
    distances = np.abs(
        chord[1] * (normalized[:, 0] - start[0])
        - chord[0] * (normalized[:, 1] - start[1])
    ) / np.linalg.norm(chord)
    return front[int(np.argmax(distances))], distances


def configure_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def draw_figure(data: np.ndarray, pareto: np.ndarray, front: np.ndarray, knee: np.ndarray) -> None:
    configure_plot()
    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.965, bottom=0.19)

    dominated = ~pareto
    ax.scatter(data[dominated, 2], data[dominated, 3], s=13, color="#AAB4BE",
               alpha=0.42, edgecolors="none", rasterized=True, zorder=2)
    ax.plot(front[:, 0], front[:, 1], color="#D88927", linewidth=2.0, zorder=4)
    ax.scatter(data[pareto, 2], data[pareto, 3], s=34, facecolor="#F3A83B",
               edgecolor="white", linewidth=0.55, zorder=5)
    ax.scatter(knee[0], knee[1], marker="*", s=210, facecolor="#C43B3B",
               edgecolor="white", linewidth=1.0, zorder=7)

    ax.annotate(
        rf"膝点  $T_{{\rm move}}={knee[0]:.2f}\,\rm s$" + "\n" + rf"$D_{{\rm wc}}={knee[1]:.2f}\,\rm m$",
        xy=knee, xytext=(34, 46), textcoords="offset points", ha="left", va="bottom",
        fontsize=9.2, color="#8F2929",
        arrowprops=dict(arrowstyle="->", color="#A63232", lw=1.1, shrinkA=3, shrinkB=5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D8B1B1", alpha=0.94),
        zorder=8,
    )

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#AAB4BE",
               markeredgecolor="none", markersize=6, alpha=0.65, label="被支配候选点"),
        Line2D([0], [0], marker="o", color="#D88927", markerfacecolor="#F3A83B",
               markeredgecolor="white", linewidth=1.8, markersize=7, label="Pareto 前沿点"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#C43B3B",
               markeredgecolor="white", markersize=12, label="最终膝点"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=9.0, columnspacing=2.0, handlelength=2.2)
    ax.set_xlabel(r"移动时间 $T_{\rm move}$ / s", fontsize=10.8)
    ax.set_ylabel(r"最坏二测定位直径 $D_{\rm wc}$ / m", fontsize=10.8)
    ax.grid(True, linestyle=(0, (2, 4)), linewidth=0.55, color="#D4DBE2", alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7A8591")
    ax.tick_params(axis="both", labelsize=9.2, colors="#4B5563")
    ax.margins(x=0.035, y=0.07)
    fig.savefig(PNG_PATH, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_double_figure(data: np.ndarray, pareto: np.ndarray, front: np.ndarray,
                       knee: np.ndarray) -> None:
    """将候选点的物理空间位置与目标空间 Pareto 前沿并列展示。"""
    configure_plot()
    knee_mask = (
        np.isclose(data[:, 2], knee[0], atol=TOLERANCE)
        & np.isclose(data[:, 3], knee[1], atol=TOLERANCE)
    )
    dominated = ~pareto

    # 实验一与实验二使用的保证接收圆几何参数。
    alpha = math.radians(1.0)
    c0 = 1500.0 / (2.0 * math.cos(alpha))
    q0 = np.array([c0 / math.sqrt(2.0), c0 / math.sqrt(2.0)])
    safe_radius = 1000.0 - c0

    fig, (ax_space, ax_front) = plt.subplots(1, 2, figsize=(13.2, 5.7))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.925, bottom=0.19, wspace=0.22)

    # (a) 物理位置空间。
    ax_space.scatter(data[dominated, 0], data[dominated, 1], s=10,
                     color="#CBD2D9", alpha=0.48, edgecolors="none", zorder=2)
    ax_space.scatter(data[pareto, 0], data[pareto, 1], s=25,
                     facecolor="#F3A83B", edgecolor="white", linewidth=0.4, zorder=4)
    ax_space.add_patch(plt.Circle(q0, safe_radius, fill=False, edgecolor="#244F4B",
                                  linewidth=1.7, zorder=5))
    line_end = q0 + (safe_radius + 95.0) * np.array([1.0, 1.0]) / math.sqrt(2.0)
    ax_space.plot([0.0, line_end[0]], [0.0, line_end[1]], color="white",
                  linewidth=3.0, zorder=5)
    ax_space.plot([0.0, line_end[0]], [0.0, line_end[1]], color="#173B67",
                  linewidth=1.3, linestyle=(0, (6, 3)), zorder=6)
    ax_space.scatter(0.0, 0.0, s=56, facecolor="#202830", edgecolor="white",
                     linewidth=0.8, zorder=7)
    ax_space.annotate(r"$S_1$", (0.0, 0.0), xytext=(7, -18),
                      textcoords="offset points", fontsize=9.5, color="#26323D")
    ax_space.scatter(q0[0], q0[1], marker="D", s=50, facecolor="#B44444",
                     edgecolor="white", linewidth=0.7, zorder=7)
    ax_space.annotate(r"$q_0$", q0, xytext=(7, -16), textcoords="offset points",
                      fontsize=9.2, color="#9C3535")
    ax_space.scatter(data[knee_mask, 0], data[knee_mask, 1], marker="*", s=190,
                     facecolor="#C43B3B", edgecolor="white", linewidth=0.9, zorder=8)
    for index, point in enumerate(data[knee_mask, :2], start=1):
        offset = (-64, 18) if point[0] > point[1] else (12, -27)
        ax_space.annotate(rf"$S_2^{{({index})}}$", point, xytext=offset,
                          textcoords="offset points", fontsize=9.3, color="#9A2F2F",
                          arrowprops=dict(arrowstyle="-", color="#B04A4A", lw=0.8))
    ax_space.set_aspect("equal", adjustable="box")
    ax_space.set_xlim(-40.0, q0[0] + safe_radius + 45.0)
    ax_space.set_ylim(-40.0, q0[1] + safe_radius + 45.0)
    ax_space.set_xlabel(r"第二检测点横坐标 $x_2$ / m", fontsize=10.2)
    ax_space.set_ylabel(r"第二检测点纵坐标 $y_2$ / m", fontsize=10.2)
    ax_space.set_title("(a) 候选位置及推荐膝点", fontsize=11.0, pad=9)

    # (b) 目标空间。
    ax_front.scatter(data[dominated, 2], data[dominated, 3], s=11,
                     color="#AAB4BE", alpha=0.40, edgecolors="none", rasterized=True, zorder=2)
    ax_front.plot(front[:, 0], front[:, 1], color="#D88927", linewidth=1.9, zorder=4)
    ax_front.scatter(data[pareto, 2], data[pareto, 3], s=29,
                     facecolor="#F3A83B", edgecolor="white", linewidth=0.5, zorder=5)
    ax_front.scatter(knee[0], knee[1], marker="*", s=190, facecolor="#C43B3B",
                     edgecolor="white", linewidth=0.9, zorder=7)
    ax_front.annotate(
        rf"$T_{{\rm move}}={knee[0]:.2f}\,\rm s$" + "\n" + rf"$D_{{\rm wc}}={knee[1]:.2f}\,\rm m$",
        xy=knee, xytext=(28, 42), textcoords="offset points", fontsize=8.9,
        color="#8F2929", arrowprops=dict(arrowstyle="->", color="#A63232", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#D8B1B1", alpha=0.94),
    )
    ax_front.set_xlabel(r"移动时间 $T_{\rm move}$ / s", fontsize=10.2)
    ax_front.set_ylabel(r"最坏二测定位直径 $D_{\rm wc}$ / m", fontsize=10.2)
    ax_front.set_title("(b) 定位精度—移动时间 Pareto 前沿", fontsize=11.0, pad=9)
    ax_front.margins(x=0.035, y=0.07)

    for axis in (ax_space, ax_front):
        axis.grid(True, linestyle=(0, (2, 4)), linewidth=0.5, color="#D4DBE2", alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#7A8591")
        axis.tick_params(axis="both", labelsize=8.8, colors="#4B5563")

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#AAB4BE",
               markeredgecolor="none", markersize=6, alpha=0.65, label="被支配候选点"),
        Line2D([0], [0], marker="o", color="#D88927", markerfacecolor="#F3A83B",
               markeredgecolor="white", linewidth=1.8, markersize=7, label="Pareto 前沿点"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#C43B3B",
               markeredgecolor="white", markersize=12, label="最终膝点及其对称位置"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=3, frameon=False, fontsize=9.2, columnspacing=2.4, handlelength=2.3)
    fig.savefig(DOUBLE_PNG_PATH, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(DOUBLE_PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_knee_heatmap(data: np.ndarray, knee: np.ndarray) -> None:
    """单独绘制 D_wc 空间热力图，并标明 Pareto 膝点的两个对称位置。"""
    configure_plot()
    alpha = math.radians(1.0)
    c0 = 1500.0 / (2.0 * math.cos(alpha))
    q0 = np.array([c0 / math.sqrt(2.0), c0 / math.sqrt(2.0)])
    safe_radius = 1000.0 - c0
    knee_mask = (
        np.isclose(data[:, 2], knee[0], atol=TOLERANCE)
        & np.isclose(data[:, 3], knee[1], atol=TOLERANCE)
    )

    coordinates = np.unique(np.concatenate((data[:, 0], data[:, 1])))
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    field = np.full(x_grid.shape, np.nan)
    index = {round(value, 6): i for i, value in enumerate(coordinates)}
    for x, y, _, diameter in data:
        field[index[round(y, 6)], index[round(x, 6)]] = diameter
    inside = (x_grid - q0[0]) ** 2 + (y_grid - q0[1]) ** 2 <= safe_radius**2 + 1.0e-6

    # 将圆外网格赋为最近圆内样本值，等值线绘制后再裁剪至几何圆内。
    extended = field.copy()
    valid_points = np.column_stack((x_grid[inside], y_grid[inside]))
    valid_values = field[inside]
    for row, column in np.argwhere(~inside):
        point = np.array([x_grid[row, column], y_grid[row, column]])
        direction = point - q0
        distance = float(np.linalg.norm(direction))
        projected = q0 if distance <= 1.0e-12 else q0 + (safe_radius - 1.0e-7) * direction / distance
        nearest = int(np.argmin(np.sum((valid_points - projected) ** 2, axis=1)))
        extended[row, column] = valid_values[nearest]

    minimum, maximum = float(np.nanmin(field)), float(np.nanmax(field))
    fig, ax = plt.subplots(figsize=(8.6, 7.25))
    fig.subplots_adjust(left=0.11, right=0.88, top=0.97, bottom=0.13)
    levels = np.linspace(minimum, maximum, 28)
    filled = ax.contourf(x_grid, y_grid, extended, levels=levels, cmap="viridis",
                         norm=PowerNorm(gamma=0.78, vmin=minimum, vmax=maximum),
                         antialiased=True)
    contours = ax.contour(x_grid, y_grid, extended,
                          levels=np.linspace(minimum, maximum, 7)[1:-1],
                          colors="white", linewidths=0.65, alpha=0.62)
    ax.clabel(contours, inline=True, fontsize=7.4, fmt="%.0f", colors="white")
    clipping_circle = Circle(q0, safe_radius, transform=ax.transData)
    filled.set_clip_path(clipping_circle)
    contours.set_clip_path(clipping_circle)
    ax.add_patch(Circle(q0, safe_radius, fill=False, edgecolor="#183C3A",
                        linewidth=1.9, zorder=6))

    line_end = q0 + (safe_radius + 95.0) * np.array([1.0, 1.0]) / math.sqrt(2.0)
    ax.plot([0.0, line_end[0]], [0.0, line_end[1]], color="white", linewidth=3.0, zorder=7)
    ax.plot([0.0, line_end[0]], [0.0, line_end[1]], color="#173B67", linewidth=1.35,
            linestyle=(0, (6, 3)), zorder=8)
    ax.scatter(0.0, 0.0, s=60, facecolor="#202830", edgecolor="white", linewidth=0.9, zorder=10)
    ax.annotate(r"$S_1=(0,0)$", (0.0, 0.0), xytext=(7, -21), textcoords="offset points",
                fontsize=9.3, color="#26323D")
    ax.scatter(q0[0], q0[1], marker="D", s=55, facecolor="#B44444",
               edgecolor="white", linewidth=0.8, zorder=10)
    ax.annotate(r"$q_0$", q0, xytext=(8, -17), textcoords="offset points",
                fontsize=9.3, color="#9C3535")
    ax.scatter(data[knee_mask, 0], data[knee_mask, 1], marker="*", s=205,
               facecolor="#D13A35", edgecolor="white", linewidth=1.0, zorder=11)
    for index_number, point in enumerate(data[knee_mask, :2], start=1):
        offset = (13, -34)
        ax.annotate(rf"膝点 $S_2^{{({index_number})}}$", point, xytext=offset,
                    textcoords="offset points", fontsize=9.0, color="#A42E2B",
                    arrowprops=dict(arrowstyle="-", color="#B94A46", lw=0.85))

    colorbar = fig.colorbar(filled, ax=ax, pad=0.028, fraction=0.048)
    colorbar.set_label(r"最坏二测定位直径 $D_{\rm wc}(S_2)$ / m", fontsize=9.8)
    colorbar.ax.tick_params(labelsize=8.5)
    legend = [
        Line2D([0], [0], color="#183C3A", linewidth=1.9,
               label=r"保证接收圆边界 $\partial\mathcal{C}_{\rm safe}$"),
        Line2D([0], [0], color="#173B67", linewidth=1.35, linestyle=(0, (6, 3)),
               label=r"首次示向中心线 $\hat\theta_1=45^\circ$"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#D13A35",
               markeredgecolor="white", markersize=11, label="Pareto膝点及其对称位置"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, frameon=False, fontsize=8.2, columnspacing=1.25, handlelength=2.2)
    upper = q0[0] + safe_radius + 65.0
    ax.set_xlim(-55.0, upper)
    ax.set_ylim(-55.0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"第二检测点横坐标 $x_2$ / m", fontsize=10.2)
    ax.set_ylabel(r"第二检测点纵坐标 $y_2$ / m", fontsize=10.2)
    ax.grid(True, linestyle=(0, (2, 4)), linewidth=0.5, color="#D7DDE3", alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7A8591")
    ax.tick_params(axis="both", labelsize=8.7, colors="#4B5563")
    fig.savefig(HEATMAP_PNG_PATH, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(HEATMAP_PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_results(data: np.ndarray, pareto: np.ndarray, knee: np.ndarray, distances: np.ndarray,
                 front: np.ndarray) -> tuple[np.ndarray, float]:
    knee_objective_mask = np.isclose(data[:, 2], knee[0], atol=TOLERANCE) & np.isclose(data[:, 3], knee[1], atol=TOLERANCE)
    knee_positions = data[knee_objective_mask, :2]
    knee_front_index = int(np.argmin(np.linalg.norm(front - knee, axis=1)))
    knee_distance = float(distances[knee_front_index])
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["S2_x_m", "S2_y_m", "T_move_s", "D_wc_m", "classification"])
        for row, is_pareto, is_knee in zip(data, pareto, knee_objective_mask):
            classification = "knee" if is_knee else ("Pareto" if is_pareto else "dominated")
            writer.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", classification])
    return knee_positions, knee_distance


def save_report(data: np.ndarray, pareto: np.ndarray, front: np.ndarray, knee: np.ndarray,
                knee_positions: np.ndarray, knee_distance: float) -> None:
    positions = "\n".join(
        rf"\[S_2^{{({i + 1})}}=({point[0]:.6f},\ {point[1]:.6f})\ \mathrm{{m}}.\]"
        for i, point in enumerate(knee_positions)
    )
    text = rf"""# 实验三：定位精度—移动时间 Pareto 前沿

## 实验方法

对实验二保证接收圆内的 {len(data)} 个候选点，同时最小化

\[
T_{{\mathrm{{move}}}}(S_2)=\frac{{\lVert S_2-S_1\rVert}}{{5}},
\qquad D_{{\mathrm{{wc}}}}(S_2).
\]

若另一候选点在两个目标上均不差且至少一项更优，则当前点被支配。从非支配解中去除目标空间的重合点后，共得到 {len(front)} 个 Pareto 目标点。对两个目标分别做极差归一化，以 Pareto 点到前沿两端连线的垂距最大处作为膝点。

## 实验结果

膝点对应

\[
T_{{\mathrm{{move}}}}={knee[0]:.6f}\ \mathrm{{s}},
\qquad D_{{\mathrm{{wc}}}}={knee[1]:.6f}\ \mathrm{{m}},
\]

其归一化垂距为 {knee_distance:.6f}。由于模型关于首次示向中心线镜像对称，该目标值可对应以下等价位置：

{positions}

该点位于 Pareto 前沿的明显转折处：在此之前增加移动时间可显著降低最坏定位直径；越过该点后，继续移动带来的定位改善迅速减弱。因此，该点是不设主观权重时兼顾定位精度与移动代价的推荐第二检测位置。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_candidates()
    objectives = data[:, 2:4]
    pareto = classify_pareto(objectives)
    front = unique_front(objectives, pareto)
    knee, distances = select_knee(front)
    knee_positions, knee_distance = save_results(data, pareto, knee, distances, front)
    draw_figure(data, pareto, front, knee)
    draw_double_figure(data, pareto, front, knee)
    draw_knee_heatmap(data, knee)
    save_report(data, pareto, front, knee, knee_positions, knee_distance)
    print(f"候选点数: {len(data)}")
    print(f"Pareto位置数: {int(np.sum(pareto))}，唯一目标点数: {len(front)}")
    print(f"膝点: T_move={knee[0]:.6f} s, D_wc={knee[1]:.6f} m")
    for index, point in enumerate(knee_positions, start=1):
        print(f"膝点位置{index}: S2=({point[0]:.6f}, {point[1]:.6f}) m")
    print(f"PNG: {PNG_PATH}\nPDF: {PDF_PATH}\nCSV: {CSV_PATH}\n说明: {REPORT_PATH}")
    print(f"双子图PNG: {DOUBLE_PNG_PATH}\n双子图PDF: {DOUBLE_PDF_PATH}")
    print(f"膝点热力图PNG: {HEATMAP_PNG_PATH}\n膝点热力图PDF: {HEATMAP_PDF_PATH}")


if __name__ == "__main__":
    main()
