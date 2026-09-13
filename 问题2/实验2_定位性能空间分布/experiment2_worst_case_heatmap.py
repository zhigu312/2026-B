"""问题二实验二：保证接收候选圆内的最坏二测定位直径分布。

对候选圆盘内的第二检测点 S2 做二维网格搜索。对每个 S2，在首次
目标可行域中确定性离散真实目标 G，并枚举第二次测向误差；每个情景
均重新构造两个有界角域的交集并计算其几何直径，最后取最大值得到
D_wc(S2)。

角域的 1500 m 圆弧采用外切弦三角形保守近似，单个角域的最大径向
放宽量仅为 1500(sec(1°)-1)≈0.228 m，不会把 GDOP 或概率指标混入
D_wc 的定义。
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "图4_保证接收区域内二次定位最坏直径分布.png"
PDF_PATH = OUTPUT_DIR / "图4_保证接收区域内二次定位最坏直径分布.pdf"
CSV_PATH = OUTPUT_DIR / "实验二_Dwc空间网格.csv"


@dataclass(frozen=True)
class Parameters:
    s1_x: float = 0.0
    s1_y: float = 0.0
    first_bearing_deg: float = 45.0
    angle_error_deg: float = 1.0
    near_radius: float = 5.0
    max_detection_radius: float = 1500.0
    guaranteed_radius: float = 1000.0


P = Parameters()


def unit_vector(angle_rad: float) -> np.ndarray:
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def safe_circle_geometry(p: Parameters) -> tuple[np.ndarray, np.ndarray, float, float]:
    s1 = np.array([p.s1_x, p.s1_y], dtype=float)
    theta = math.radians(p.first_bearing_deg)
    alpha = math.radians(p.angle_error_deg)
    c0 = p.max_detection_radius / (2.0 * math.cos(alpha))
    q0 = s1 + c0 * unit_vector(theta)
    safe_radius = p.guaranteed_radius - c0
    if safe_radius <= 0.0:
        raise ValueError("保证接收候选圆为空。")
    return s1, q0, c0, safe_radius


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def bounded_angle_triangle(
    center: np.ndarray,
    bearing_rad: float,
    half_angle_rad: float,
    radial_cap: float,
) -> np.ndarray:
    """构造包含半径 radial_cap 圆扇形的外切弦三角形（逆时针）。"""

    boundary_radius = radial_cap / math.cos(half_angle_rad)
    lower = center + boundary_radius * unit_vector(bearing_rad - half_angle_rad)
    upper = center + boundary_radius * unit_vector(bearing_rad + half_angle_rad)
    return np.vstack((center, lower, upper))


def line_intersection(
    p: np.ndarray,
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """返回直线 pq 与 ab 的交点；调用者保证二者不平行。"""

    r = q - p
    s = b - a
    denominator = cross2(r, s)
    t = cross2(a - p, s) / denominator
    return p + t * r


def clip_convex_polygon(
    subject: np.ndarray,
    clipper: np.ndarray,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Sutherland–Hodgman 算法求两个逆时针凸多边形的交集。"""

    output = [point.copy() for point in subject]
    for edge_index in range(len(clipper)):
        if not output:
            return np.empty((0, 2), dtype=float)

        edge_start = clipper[edge_index]
        edge_end = clipper[(edge_index + 1) % len(clipper)]
        edge = edge_end - edge_start
        input_vertices = output
        output = []
        previous = input_vertices[-1]
        previous_inside = cross2(edge, previous - edge_start) >= -tolerance

        for current in input_vertices:
            current_inside = cross2(edge, current - edge_start) >= -tolerance
            if current_inside:
                if not previous_inside:
                    segment = current - previous
                    if abs(cross2(segment, edge)) > tolerance:
                        output.append(
                            line_intersection(previous, current, edge_start, edge_end)
                        )
                output.append(current.copy())
            elif previous_inside:
                segment = current - previous
                if abs(cross2(segment, edge)) > tolerance:
                    output.append(
                        line_intersection(previous, current, edge_start, edge_end)
                    )
            previous = current
            previous_inside = current_inside

    if len(output) <= 1:
        return np.asarray(output, dtype=float).reshape((-1, 2))

    cleaned = [output[0]]
    for point in output[1:]:
        if np.linalg.norm(point - cleaned[-1]) > 1.0e-8:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= 1.0e-8:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def polygon_diameter(vertices: np.ndarray) -> float:
    """用问题一的旋转卡壳思想计算凸多边形几何直径。"""

    vertex_count = len(vertices)
    if vertex_count <= 1:
        return 0.0
    if vertex_count == 2:
        return float(np.linalg.norm(vertices[1] - vertices[0]))

    # Sutherland–Hodgman 通常保持逆时针方向；此处仍显式校正以增强稳健性。
    signed_area_twice = float(
        np.sum(
            vertices[:, 0] * np.roll(vertices[:, 1], -1)
            - vertices[:, 1] * np.roll(vertices[:, 0], -1)
        )
    )
    polygon = vertices if signed_area_twice >= 0.0 else vertices[::-1]

    def squared_distance(i: int, j: int) -> float:
        difference = polygon[i] - polygon[j]
        return float(np.dot(difference, difference))

    maximum_squared = 0.0
    opposite = 1
    for i in range(vertex_count):
        next_i = (i + 1) % vertex_count
        edge = polygon[next_i] - polygon[i]
        while True:
            next_opposite = (opposite + 1) % vertex_count
            current_area = cross2(edge, polygon[opposite] - polygon[i])
            next_area = cross2(edge, polygon[next_opposite] - polygon[i])
            if next_area <= current_area + 1.0e-11:
                break
            opposite = next_opposite

        # 平行边会产生面积相等的反足点，同时检查相邻点以覆盖退化情形。
        next_opposite = (opposite + 1) % vertex_count
        for first in (i, next_i):
            maximum_squared = max(
                maximum_squared,
                squared_distance(first, opposite),
                squared_distance(first, next_opposite),
            )
    return math.sqrt(maximum_squared)


def verify_diameter_kernel() -> None:
    """用直接枚举交叉核对旋转卡壳实现。"""

    generator = np.random.default_rng(20260912)
    alpha = math.radians(P.angle_error_deg)
    for _ in range(250):
        first_center = generator.uniform(-200.0, 200.0, size=2)
        second_center = generator.uniform(-200.0, 200.0, size=2)
        first = bounded_angle_triangle(
            first_center,
            generator.uniform(-math.pi, math.pi),
            alpha,
            P.max_detection_radius,
        )
        second = bounded_angle_triangle(
            second_center,
            generator.uniform(-math.pi, math.pi),
            alpha,
            P.max_detection_radius,
        )
        intersection = clip_convex_polygon(first, second)
        if len(intersection) <= 1:
            brute_force = 0.0
        else:
            differences = intersection[:, None, :] - intersection[None, :, :]
            brute_force = float(np.sqrt(np.max(np.sum(differences**2, axis=-1))))
        calipers = polygon_diameter(intersection)
        if abs(brute_force - calipers) > 1.0e-8:
            raise AssertionError(
                f"旋转卡壳核验失败：brute={brute_force}, calipers={calipers}"
            )


def target_scenarios(p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成确定性目标位置与第二次测向误差情景。"""

    # 径向样本在近端、候选圆所在距离带和1500 m边界均有覆盖。
    radii = np.array(
        [5.001, 20.0, 60.0, 150.0, 300.0, 450.0, 600.0,
         750.0, 900.0, 1050.0, 1200.0, 1350.0, 1500.0],
        dtype=float,
    )
    angular_offsets = np.deg2rad(np.linspace(-p.angle_error_deg, p.angle_error_deg, 7))
    second_errors = np.deg2rad(np.linspace(-p.angle_error_deg, p.angle_error_deg, 5))

    s1 = np.array([p.s1_x, p.s1_y], dtype=float)
    center_angle = math.radians(p.first_bearing_deg)
    points: list[np.ndarray] = []
    metadata: list[tuple[float, float]] = []
    for radius in radii:
        for offset in angular_offsets:
            points.append(s1 + radius * unit_vector(center_angle + float(offset)))
            metadata.append((radius, math.degrees(float(offset))))
    return np.asarray(points), np.asarray(metadata), second_errors


def dense_target_scenarios(p: Parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成用于复核低值点的高密度确定性情景。"""

    radii = np.unique(
        np.concatenate(
            (
                np.linspace(5.001, p.max_detection_radius, 61),
                np.array([20.0, 60.0, 150.0, 300.0, 450.0, 600.0,
                          750.0, 900.0, 1050.0, 1200.0, 1350.0]),
            )
        )
    )
    angular_offsets = np.deg2rad(np.linspace(-p.angle_error_deg, p.angle_error_deg, 13))
    second_errors = np.deg2rad(np.linspace(-p.angle_error_deg, p.angle_error_deg, 9))
    s1 = np.array([p.s1_x, p.s1_y], dtype=float)
    center_angle = math.radians(p.first_bearing_deg)

    points: list[np.ndarray] = []
    metadata: list[tuple[float, float]] = []
    for radius in radii:
        for offset in angular_offsets:
            points.append(s1 + radius * unit_vector(center_angle + float(offset)))
            metadata.append((radius, math.degrees(float(offset))))
    return np.asarray(points), np.asarray(metadata), second_errors


def evaluate_candidate(
    s2: np.ndarray,
    first_region: np.ndarray,
    targets: np.ndarray,
    target_metadata: np.ndarray,
    second_errors: np.ndarray,
    p: Parameters,
) -> tuple[float, float, float, float]:
    """返回 D_wc 及造成最大值的目标半径、角偏差和二测误差。"""

    alpha = math.radians(p.angle_error_deg)
    worst_diameter = -math.inf
    worst_radius = math.nan
    worst_angle_offset = math.nan
    worst_second_error = math.nan

    for target, (radius, first_offset_deg) in zip(targets, target_metadata):
        delta = target - s2
        target_distance = float(np.hypot(delta[0], delta[1]))

        # near 分支本身将目标限制在5 m圆内，直径不超过10 m。
        if target_distance <= p.near_radius:
            diameter = 2.0 * p.near_radius
            if diameter > worst_diameter:
                worst_diameter = diameter
                worst_radius = float(radius)
                worst_angle_offset = float(first_offset_deg)
                worst_second_error = math.nan
            continue

        true_second_bearing = math.atan2(delta[1], delta[0])
        for second_error in second_errors:
            measured_second_bearing = true_second_bearing + float(second_error)
            second_region = bounded_angle_triangle(
                s2,
                measured_second_bearing,
                alpha,
                p.max_detection_radius,
            )
            intersection = clip_convex_polygon(first_region, second_region)
            diameter = polygon_diameter(intersection)
            if diameter > worst_diameter:
                worst_diameter = diameter
                worst_radius = float(radius)
                worst_angle_offset = float(first_offset_deg)
                worst_second_error = math.degrees(float(second_error))

    return (
        worst_diameter,
        worst_radius,
        worst_angle_offset,
        worst_second_error,
    )


def calculate_heatmap(
    grid_size: int,
    p: Parameters,
) -> dict[str, np.ndarray | float]:
    """在候选圆盘内计算 D_wc 网格。"""

    if grid_size < 21 or grid_size % 2 == 0:
        raise ValueError("grid_size 应为不小于21的奇数，以保证网格包含 q0。")

    s1, q0, c0, safe_radius = safe_circle_geometry(p)
    alpha = math.radians(p.angle_error_deg)
    first_region = bounded_angle_triangle(
        s1,
        math.radians(p.first_bearing_deg),
        alpha,
        p.max_detection_radius,
    )
    targets, metadata, second_errors = target_scenarios(p)

    coordinates = np.linspace(q0[0] - safe_radius, q0[0] + safe_radius, grid_size)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    inside = (x_grid - q0[0]) ** 2 + (y_grid - q0[1]) ** 2 <= safe_radius**2 + 1.0e-9

    dwc = np.full_like(x_grid, np.nan, dtype=float)
    worst_radius = np.full_like(x_grid, np.nan, dtype=float)
    worst_first_offset = np.full_like(x_grid, np.nan, dtype=float)
    worst_second_error = np.full_like(x_grid, np.nan, dtype=float)

    candidate_indices = np.argwhere(inside)
    started = time.perf_counter()
    progress_step = max(1, len(candidate_indices) // 10)
    for counter, (row, column) in enumerate(candidate_indices, start=1):
        s2 = np.array([x_grid[row, column], y_grid[row, column]])
        result = evaluate_candidate(
            s2,
            first_region,
            targets,
            metadata,
            second_errors,
            p,
        )
        dwc[row, column] = result[0]
        worst_radius[row, column] = result[1]
        worst_first_offset[row, column] = result[2]
        worst_second_error[row, column] = result[3]
        if counter % progress_step == 0 or counter == len(candidate_indices):
            elapsed = time.perf_counter() - started
            print(
                f"进度 {counter:4d}/{len(candidate_indices)} "
                f"({100.0 * counter / len(candidate_indices):5.1f}%), "
                f"耗时 {elapsed:6.1f} s"
            )

    runtime = time.perf_counter() - started
    return {
        "s1": s1,
        "q0": q0,
        "c0": c0,
        "safe_radius": safe_radius,
        "x": x_grid,
        "y": y_grid,
        "inside": inside,
        "dwc": dwc,
        "worst_radius": worst_radius,
        "worst_first_offset": worst_first_offset,
        "worst_second_error": worst_second_error,
        "runtime": runtime,
        "scenario_count": float(len(targets) * len(second_errors)),
    }


def dense_validate_low_point(
    results: dict[str, np.ndarray | float],
    p: Parameters,
) -> None:
    """用更密的目标与误差网格复核热力图最低网格点及其镜像点。"""

    dwc = np.asarray(results["dwc"])
    x_grid = np.asarray(results["x"])
    y_grid = np.asarray(results["y"])
    best_flat = int(np.nanargmin(dwc))
    best_row, best_column = np.unravel_index(best_flat, dwc.shape)
    best_point = np.array([x_grid[best_row, best_column], y_grid[best_row, best_column]])
    mirror_point = best_point[::-1]

    s1 = np.asarray(results["s1"])
    alpha = math.radians(p.angle_error_deg)
    first_region = bounded_angle_triangle(
        s1,
        math.radians(p.first_bearing_deg),
        alpha,
        p.max_detection_radius,
    )
    dense_targets, dense_metadata, dense_errors = dense_target_scenarios(p)
    best_result = evaluate_candidate(
        best_point,
        first_region,
        dense_targets,
        dense_metadata,
        dense_errors,
        p,
    )
    mirror_result = evaluate_candidate(
        mirror_point,
        first_region,
        dense_targets,
        dense_metadata,
        dense_errors,
        p,
    )
    results["dense_best_dwc"] = float(best_result[0])
    results["dense_mirror_dwc"] = float(mirror_result[0])
    results["dense_scenario_count"] = float(len(dense_targets) * len(dense_errors))
    results["dense_worst_radius"] = float(best_result[1])
    results["dense_worst_first_offset"] = float(best_result[2])
    results["dense_worst_second_error"] = float(best_result[3])


def configure_plot() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def extend_field_to_square(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    inside: np.ndarray,
    values: np.ndarray,
    q0: np.ndarray,
    safe_radius: float,
) -> np.ndarray:
    """将圆外网格赋为最近边界值，仅用于平滑绘图后再严格裁剪到圆内。"""

    extended = values.copy()
    valid_points = np.column_stack((x_grid[inside], y_grid[inside]))
    valid_values = values[inside]
    for row, column in np.argwhere(~inside):
        point = np.array([x_grid[row, column], y_grid[row, column]])
        direction = point - q0
        distance = float(np.linalg.norm(direction))
        if distance <= 1.0e-12:
            projected = q0
        else:
            projected = q0 + (safe_radius - 1.0e-7) * direction / distance
        nearest = int(np.argmin(np.sum((valid_points - projected) ** 2, axis=1)))
        extended[row, column] = valid_values[nearest]
    return extended


def draw_heatmap(results: dict[str, np.ndarray | float], p: Parameters) -> None:
    configure_plot()
    s1 = np.asarray(results["s1"])
    q0 = np.asarray(results["q0"])
    safe_radius = float(results["safe_radius"])
    x_grid = np.asarray(results["x"])
    y_grid = np.asarray(results["y"])
    inside = np.asarray(results["inside"], dtype=bool)
    dwc = np.asarray(results["dwc"])
    extended_dwc = extend_field_to_square(
        x_grid,
        y_grid,
        inside,
        dwc,
        q0,
        safe_radius,
    )
    valid_values = dwc[np.isfinite(dwc)]
    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))

    fig, ax = plt.subplots(figsize=(8.7, 7.4))
    fig.subplots_adjust(left=0.10, right=0.88, top=0.97, bottom=0.12)

    levels = np.linspace(minimum, maximum, 28)
    normalization = PowerNorm(gamma=0.78, vmin=minimum, vmax=maximum)
    filled = ax.contourf(
        x_grid,
        y_grid,
        extended_dwc,
        levels=levels,
        cmap="viridis",
        norm=normalization,
        antialiased=True,
        extend="neither",
    )
    contour_levels = np.linspace(minimum, maximum, 7)[1:-1]
    contours = ax.contour(
        x_grid,
        y_grid,
        extended_dwc,
        levels=contour_levels,
        colors="white",
        linewidths=0.65,
        alpha=0.62,
    )
    ax.clabel(contours, inline=True, fontsize=7.5, fmt="%.0f", colors="white")

    # 绘图插值只在圆内展示，候选域边界仍由严格几何圆给出。
    clipping_circle = Circle(q0, safe_radius, transform=ax.transData)
    filled.set_clip_path(clipping_circle)
    contours.set_clip_path(clipping_circle)

    # 候选圆边界。
    ax.add_patch(
        Circle(
            q0,
            safe_radius,
            fill=False,
            edgecolor="#183C3A",
            linewidth=1.9,
            zorder=6,
        )
    )

    # 中心示向线采用白色衬底加深蓝虚线，确保在任意热力颜色上均可见。
    theta = math.radians(p.first_bearing_deg)
    line_length = float(np.linalg.norm(q0 - s1) + safe_radius + 85.0)
    line_end = s1 + line_length * unit_vector(theta)
    ax.plot(
        [s1[0], line_end[0]],
        [s1[1], line_end[1]],
        color="white",
        linewidth=3.0,
        alpha=0.88,
        zorder=7,
    )
    center_line = ax.plot(
        [s1[0], line_end[0]],
        [s1[1], line_end[1]],
        color="#173B67",
        linewidth=1.35,
        linestyle=(0, (6, 3)),
        zorder=8,
    )[0]

    ax.scatter(
        *s1,
        s=60,
        facecolor="#202830",
        edgecolor="white",
        linewidth=0.9,
        zorder=10,
    )
    ax.annotate(
        r"$S_1=(0,0)$",
        xy=s1,
        xytext=(-15, -24),
        textcoords="offset points",
        fontsize=9.5,
        color="#26323D",
    )
    ax.scatter(
        *q0,
        s=56,
        marker="D",
        facecolor="#B44444",
        edgecolor="white",
        linewidth=0.8,
        zorder=10,
    )
    ax.annotate(
        r"$q_0$",
        xy=q0,
        xytext=(8, -17),
        textcoords="offset points",
        fontsize=9.5,
        color="#9C3535",
    )

    # 用最小值位置提示低值区，不把离散网格点宣称为连续全局最优。
    best_flat = int(np.nanargmin(dwc))
    best_row, best_column = np.unravel_index(best_flat, dwc.shape)
    best_point = np.array([x_grid[best_row, best_column], y_grid[best_row, best_column]])
    mirror_point = best_point[::-1]
    ax.scatter(
        [best_point[0], mirror_point[0]],
        [best_point[1], mirror_point[1]],
        marker="*",
        s=92,
        facecolor="#F6F2E8",
        edgecolor="#252525",
        linewidth=0.75,
        zorder=11,
    )

    colorbar = fig.colorbar(filled, ax=ax, pad=0.028, fraction=0.048)
    colorbar.set_label(r"最坏二测定位直径 $D_{\rm wc}(S_2)$ / m", fontsize=9.8)
    colorbar.ax.tick_params(labelsize=8.5, width=0.7, length=3)
    colorbar.outline.set_linewidth(0.7)

    line_legend = Line2D(
        [0],
        [0],
        color="#173B67",
        linewidth=1.35,
        linestyle=(0, (6, 3)),
        label=rf"首次示向中心线 $\hat\theta_1={p.first_bearing_deg:g}^\circ$",
    )
    boundary_legend = Line2D(
        [0],
        [0],
        color="#183C3A",
        linewidth=1.9,
        label=r"保证接收圆边界 $\partial\mathcal{C}_{\rm safe}$",
    )
    best_legend = Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="#F6F2E8",
        markeredgecolor="#252525",
        markersize=9,
        label=r"最小 $D_{\rm wc}$ 点及其对称点",
    )
    ax.legend(
        handles=[boundary_legend, line_legend, best_legend],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=2.2,
    )

    # 图号和图名由论文题注生成，图内不重复放置大标题。
    upper = float(q0[0] + safe_radius + 65.0)
    ax.set_xlim(-55.0, upper)
    ax.set_ylim(-55.0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"第二检测点横坐标 $x_2$ / m", fontsize=10.2)
    ax.set_ylabel(r"第二检测点纵坐标 $y_2$ / m", fontsize=10.2)
    ax.grid(True, linestyle=(0, (2, 4)), linewidth=0.5, color="#D7DDE3", alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7A8591")
    ax.spines["bottom"].set_color("#7A8591")
    ax.tick_params(axis="both", labelsize=8.7, colors="#4B5563")

    # 白色文字描边仅用于图例外的必要标记，保证缩小后仍清晰。
    center_line.set_path_effects(
        [path_effects.Stroke(linewidth=2.2, foreground="white"), path_effects.Normal()]
    )

    fig.savefig(PNG_PATH, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_grid(results: dict[str, np.ndarray | float]) -> None:
    x_grid = np.asarray(results["x"])
    y_grid = np.asarray(results["y"])
    inside = np.asarray(results["inside"], dtype=bool)
    dwc = np.asarray(results["dwc"])
    worst_radius = np.asarray(results["worst_radius"])
    worst_first_offset = np.asarray(results["worst_first_offset"])
    worst_second_error = np.asarray(results["worst_second_error"])

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "S2_x_m",
                "S2_y_m",
                "inside_safe_circle",
                "D_wc_m",
                "worst_target_radius_m",
                "worst_first_angle_offset_deg",
                "worst_second_measurement_error_deg",
            ]
        )
        for row in range(x_grid.shape[0]):
            for column in range(x_grid.shape[1]):
                writer.writerow(
                    [
                        f"{x_grid[row, column]:.6f}",
                        f"{y_grid[row, column]:.6f}",
                        int(inside[row, column]),
                        "" if not inside[row, column] else f"{dwc[row, column]:.6f}",
                        "" if not inside[row, column] else f"{worst_radius[row, column]:.6f}",
                        "" if not inside[row, column] else f"{worst_first_offset[row, column]:.6f}",
                        "" if not inside[row, column] else f"{worst_second_error[row, column]:.6f}",
                    ]
                )


def print_summary(results: dict[str, np.ndarray | float], grid_size: int) -> None:
    q0 = np.asarray(results["q0"])
    dwc = np.asarray(results["dwc"])
    x_grid = np.asarray(results["x"])
    y_grid = np.asarray(results["y"])
    valid = np.isfinite(dwc)
    best_flat = int(np.nanargmin(dwc))
    best_row, best_column = np.unravel_index(best_flat, dwc.shape)
    symmetry_difference = np.abs(dwc - dwc.T)
    maximum_symmetry_error = float(np.nanmax(symmetry_difference))

    print("\n实验二完成")
    print(f"候选网格: {grid_size} × {grid_size}")
    print(f"圆内候选点数: {int(np.sum(valid))}")
    print(f"每个候选点的确定性情景数: {int(float(results['scenario_count']))}")
    print(f"q0 = ({q0[0]:.6f}, {q0[1]:.6f}) m")
    print(f"D_wc 范围: [{np.nanmin(dwc):.6f}, {np.nanmax(dwc):.6f}] m")
    print(
        "最小网格点: "
        f"S2=({x_grid[best_row, best_column]:.6f}, "
        f"{y_grid[best_row, best_column]:.6f}) m, "
        f"D_wc={dwc[best_row, best_column]:.6f} m"
    )
    print(
        f"密集复核（{int(float(results['dense_scenario_count']))}个情景）: "
        f"D_wc={float(results['dense_best_dwc']):.6f} m, "
        f"镜像点={float(results['dense_mirror_dwc']):.6f} m"
    )
    print(
        "密集复核最坏情景: "
        f"r={float(results['dense_worst_radius']):.3f} m, "
        f"首次角偏差={float(results['dense_worst_first_offset']):.3f}°, "
        f"二测误差={float(results['dense_worst_second_error']):.3f}°"
    )
    print(f"关于 y=x 镜像的最大数值差: {maximum_symmetry_error:.6e} m")
    print(f"运行时间: {float(results['runtime']):.2f} s")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"CSV: {CSV_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算并绘制问题二实验二 D_wc 热力图。")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=51,
        help="候选圆外接正方形的单轴网格数，必须为不小于21的奇数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_diameter_kernel()
    results = calculate_heatmap(args.grid_size, P)
    dense_validate_low_point(results, P)
    save_grid(results)
    draw_heatmap(results, P)
    print_summary(results, args.grid_size)


if __name__ == "__main__":
    main()
