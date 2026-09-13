"""问题二实验一：验证首次测向后的保证接收候选圆。

固定 S1=(0, 0)、中心示向角 45°、角度误差上界 1°。
程序绘制首次目标可行域、其保守几何包络圆，以及保证第二次不返回
``no_signal`` 的第二检测点候选圆盘，并用确定性离散验证集合包含关系。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon


@dataclass(frozen=True)
class Parameters:
    """实验参数，距离单位为 m，角度单位为 °。"""

    s1_x: float = 0.0
    s1_y: float = 0.0
    bearing_deg: float = 45.0
    angle_error_deg: float = 1.0
    near_radius: float = 5.0
    max_detection_radius: float = 1500.0
    guaranteed_reception_radius: float = 1000.0


PARAM = Parameters()
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_PNG = OUTPUT_DIR / "图3_首次有效测向后的目标可行域与保证接收候选区域.png"
FIGURE_PDF = OUTPUT_DIR / "图3_首次有效测向后的目标可行域与保证接收候选区域.pdf"


def unit_vector(angle_rad: np.ndarray | float) -> np.ndarray:
    """返回数学角 angle_rad 对应的二维单位向量。"""

    return np.stack((np.cos(angle_rad), np.sin(angle_rad)), axis=-1)


def calculate_geometry(p: Parameters) -> dict[str, np.ndarray | float]:
    """计算扇形包络圆和保守保证接收候选圆盘。"""

    s1 = np.array([p.s1_x, p.s1_y], dtype=float)
    theta = np.deg2rad(p.bearing_deg)
    alpha = np.deg2rad(p.angle_error_deg)

    # 对完整扇形 0 <= r <= L 的严格包络：圆同时经过顶点与两侧外弧端点。
    c0 = p.max_detection_radius / (2.0 * np.cos(alpha))
    q0 = s1 + c0 * unit_vector(theta)
    safe_radius = p.guaranteed_reception_radius - c0
    if safe_radius <= 0.0:
        raise ValueError("保证接收半径不足以构造非空候选圆盘。")

    return {
        "s1": s1,
        "theta": theta,
        "alpha": alpha,
        "c0": c0,
        "q0": q0,
        "safe_radius": safe_radius,
    }


def verify_geometry(
    p: Parameters,
    geometry: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """离散验证外包关系，并核对理论最坏接收距离。"""

    s1 = np.asarray(geometry["s1"])
    q0 = np.asarray(geometry["q0"])
    theta = float(geometry["theta"])
    alpha = float(geometry["alpha"])
    c0 = float(geometry["c0"])
    safe_radius = float(geometry["safe_radius"])

    radii = np.linspace(p.near_radius, p.max_detection_radius, 1201)
    offsets = np.linspace(-alpha, alpha, 801)
    rr, dd = np.meshgrid(radii, offsets, indexing="ij")
    feasible_points = s1 + rr[..., None] * unit_vector(theta + dd)
    distances_to_q0 = np.linalg.norm(feasible_points - q0, axis=-1)
    sampled_max_envelope_distance = float(distances_to_q0.max())

    # 外弧端点位于包络圆上；将 S2 取在 q0 关于该端点的反方向，
    # 可使 ||S2-G|| 恰好等于 safe_radius+c0=1000 m。
    outer_endpoint = s1 + p.max_detection_radius * unit_vector(theta + alpha)
    direction = (outer_endpoint - q0) / np.linalg.norm(outer_endpoint - q0)
    critical_s2 = q0 - safe_radius * direction
    critical_distance = float(np.linalg.norm(critical_s2 - outer_endpoint))

    tolerance = 1.0e-8
    assert sampled_max_envelope_distance <= c0 + tolerance
    assert abs(critical_distance - p.guaranteed_reception_radius) <= tolerance

    return {
        "sampled_max_envelope_distance": sampled_max_envelope_distance,
        "envelope_margin": c0 - sampled_max_envelope_distance,
        "critical_reception_distance": critical_distance,
    }


def configure_chinese_font() -> None:
    """设置 Windows 常见中文字体，同时保留跨平台回退字体。"""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def draw_figure(
    p: Parameters,
    geometry: dict[str, np.ndarray | float],
) -> None:
    """绘制论文用理论核心图，并分别保存为 PNG 与 PDF。"""

    configure_chinese_font()

    s1 = np.asarray(geometry["s1"])
    q0 = np.asarray(geometry["q0"])
    theta = float(geometry["theta"])
    alpha = float(geometry["alpha"])
    c0 = float(geometry["c0"])
    safe_radius = float(geometry["safe_radius"])

    boundary_angles = np.linspace(theta - alpha, theta + alpha, 500)
    outer_arc = s1 + p.max_detection_radius * unit_vector(boundary_angles)
    inner_arc = s1 + p.near_radius * unit_vector(boundary_angles[::-1])
    sector_vertices = np.vstack((outer_arc, inner_arc))

    fig, ax = plt.subplots(figsize=(10.6, 9.0), constrained_layout=True)

    # 先画目标外包和 S2 候选盘，再把真实目标可行域覆盖在上层，避免混淆。
    envelope_circle = Circle(
        q0,
        c0,
        fill=False,
        edgecolor="#D97706",
        linewidth=2.3,
        linestyle=(0, (7, 4)),
        zorder=1,
    )
    safe_disk = Circle(
        q0,
        safe_radius,
        facecolor="#67C587",
        edgecolor="#087A4C",
        linewidth=2.4,
        alpha=0.30,
        zorder=2,
    )
    feasible_sector = Polygon(
        sector_vertices,
        closed=True,
        facecolor="#5AA9E6",
        edgecolor="#1769AA",
        linewidth=1.8,
        alpha=0.68,
        zorder=3,
    )
    ax.add_patch(envelope_circle)
    ax.add_patch(safe_disk)
    ax.add_patch(feasible_sector)

    # 扇形边界和外弧。
    for angle in (theta - alpha, theta + alpha):
        edge = s1 + p.max_detection_radius * unit_vector(angle)
        ax.plot(
            [s1[0], edge[0]],
            [s1[1], edge[1]],
            color="#1769AA",
            linewidth=1.6,
            zorder=4,
        )
    ax.plot(outer_arc[:, 0], outer_arc[:, 1], color="#1769AA", linewidth=2.0, zorder=4)

    # 中心示向线。
    center_endpoint = s1 + p.max_detection_radius * unit_vector(theta)
    ax.plot(
        [s1[0], center_endpoint[0]],
        [s1[1], center_endpoint[1]],
        color="#173B67",
        linewidth=1.8,
        linestyle=(0, (5, 4)),
        zorder=5,
    )

    # 标出 S1 与 q0。
    ax.scatter(*s1, s=72, color="#111827", edgecolor="white", linewidth=0.8, zorder=7)
    ax.scatter(
        *q0,
        s=165,
        marker="*",
        color="#B91C1C",
        edgecolor="white",
        linewidth=0.9,
        zorder=8,
    )
    ax.annotate(
        r"$S_1=(0,0)$",
        xy=s1,
        xytext=(-16, -31),
        textcoords="offset points",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.annotate(
        rf"$q_0=({q0[0]:.2f},\,{q0[1]:.2f})\ \mathrm{{m}}$",
        xy=q0,
        xytext=(18, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#B91C1C", "lw": 1.2},
        fontsize=11.2,
        color="#8B1111",
        zorder=9,
    )

    # 两个关键半径的图内标注。
    normal = unit_vector(theta + np.pi / 2.0)
    safe_radius_endpoint = q0 + safe_radius * normal
    ax.plot(
        [q0[0], safe_radius_endpoint[0]],
        [q0[1], safe_radius_endpoint[1]],
        color="#087A4C",
        linewidth=1.7,
        zorder=6,
    )
    safe_midpoint = (q0 + safe_radius_endpoint) / 2.0
    ax.annotate(
        rf"$r_{{\rm safe}}={safe_radius:.2f}\ \mathrm{{m}}$",
        xy=safe_midpoint,
        xytext=(-54, 12),
        textcoords="offset points",
        fontsize=10.8,
        color="#08633F",
        fontweight="bold",
    )

    c0_midpoint = (s1 + q0) / 2.0
    ax.annotate(
        rf"$c_0={c0:.2f}\ \mathrm{{m}}$",
        xy=c0_midpoint,
        xytext=(18, -30),
        textcoords="offset points",
        fontsize=10.8,
        color="#A55703",
        fontweight="bold",
    )

    # 角域与中心线标签。
    label_point = s1 + 1320.0 * unit_vector(theta)
    lower_angle = p.bearing_deg - p.angle_error_deg
    upper_angle = p.bearing_deg + p.angle_error_deg
    ax.annotate(
        rf"首次可行角域：${lower_angle:g}^\circ\leq\theta\leq{upper_angle:g}^\circ$",
        xy=label_point,
        xytext=(-176, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#1769AA", "lw": 1.1},
        fontsize=10.8,
        color="#125A93",
    )
    center_label = s1 + 1125.0 * unit_vector(theta)
    ax.annotate(
        rf"示向中心线 $\hat\theta_1={p.bearing_deg:g}^\circ$",
        xy=center_label,
        xytext=(25, -34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#173B67", "lw": 1.0},
        fontsize=10.5,
        color="#173B67",
    )

    # 推导结果框，明确“保证接收”而非“保证返回示向角”。
    formula_text = (
        rf"$c_0=\dfrac{{{p.max_detection_radius:g}}}"
        rf"{{2\cos{p.angle_error_deg:g}^\circ}}={c0:.3f}\ \mathrm{{m}}$"
        "\n"
        rf"$r_{{\rm safe}}={p.guaranteed_reception_radius:g}-c_0="
        rf"{safe_radius:.3f}\ \mathrm{{m}}$"
        "\n"
        rf"$S_2\in\mathcal{{C}}_{{\rm safe}}\Rightarrow"
        rf"\|S_2-G\|\leq{p.guaranteed_reception_radius:g}\ \mathrm{{m}}$"
    )
    ax.text(
        0.025,
        0.975,
        formula_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        linespacing=1.55,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "#9CA3AF",
            "alpha": 0.94,
        },
        zorder=10,
    )

    legend_handles = [
        Patch(
            facecolor="#5AA9E6",
            edgecolor="#1769AA",
            alpha=0.68,
            label=(
                rf"首次目标可行域 $\Omega_1$（${p.near_radius:g}<r"
                rf"\leq{p.max_detection_radius:g}\ \mathrm{{m}}$）"
            ),
        ),
        Line2D(
            [0],
            [0],
            color="#173B67",
            linewidth=1.8,
            linestyle=(0, (5, 4)),
            label="首次示向中心线",
        ),
        Line2D(
            [0],
            [0],
            color="#D97706",
            linewidth=2.3,
            linestyle=(0, (7, 4)),
            label=r"保守几何包络圆 $B(q_0,c_0)$",
        ),
        Patch(
            facecolor="#67C587",
            edgecolor="#087A4C",
            alpha=0.35,
            label=r"保证接收候选圆盘 $\mathcal{C}_{\rm safe}$",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#B91C1C",
            markeredgecolor="white",
            markersize=13,
            label=r"包络圆心 $q_0$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9.8,
        frameon=True,
        framealpha=0.96,
        edgecolor="#CBD5E1",
    )

    ax.set_title(
        "图3  首次有效测向后的目标可行域与保证接收候选区域",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("横坐标 x / m", fontsize=11.5)
    ax.set_ylabel("纵坐标 y / m", fontsize=11.5)
    ax.set_xlim(-280.0, 1370.0)
    ax.set_ylim(-280.0, 1370.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#64748B")

    fig.savefig(FIGURE_PNG, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    geometry = calculate_geometry(PARAM)
    checks = verify_geometry(PARAM, geometry)
    draw_figure(PARAM, geometry)

    q0 = np.asarray(geometry["q0"])
    print("实验一：保证接收候选圆")
    print(f"q0 = ({q0[0]:.6f}, {q0[1]:.6f}) m")
    print(f"几何包络圆半径 c0 = {float(geometry['c0']):.6f} m")
    print(f"保证接收候选圆盘半径 = {float(geometry['safe_radius']):.6f} m")
    print(
        "离散样本到 q0 的最大距离 = "
        f"{checks['sampled_max_envelope_distance']:.6f} m"
    )
    print(
        "构造的临界情形 ||S2-G|| = "
        f"{checks['critical_reception_distance']:.6f} m"
    )
    print(f"PNG: {FIGURE_PNG}")
    print(f"PDF: {FIGURE_PDF}")


if __name__ == "__main__":
    main()
