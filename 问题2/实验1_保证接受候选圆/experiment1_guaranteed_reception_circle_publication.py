"""问题二实验一的论文制图版本。

图内只呈现几何关系与关键尺度；完整公式和证明放在正文或图注中。
图号、图名应由 Word/LaTeX 的题注功能统一生成。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon
from matplotlib.ticker import MultipleLocator

from experiment1_guaranteed_reception_circle import (
    PARAM,
    Parameters,
    calculate_geometry,
    configure_chinese_font,
    unit_vector,
    verify_geometry,
)


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_PNG = OUTPUT_DIR / "图3_首次有效测向后的目标可行域与保证接收候选区域_论文版.png"
FIGURE_PDF = OUTPUT_DIR / "图3_首次有效测向后的目标可行域与保证接收候选区域_论文版.pdf"


def draw_publication_figure(
    p: Parameters,
    geometry: dict[str, np.ndarray | float],
) -> None:
    """绘制低饱和、色盲友好且适合论文排版的几何图。"""

    configure_chinese_font()
    plt.rcParams["hatch.linewidth"] = 0.28

    s1 = np.asarray(geometry["s1"])
    q0 = np.asarray(geometry["q0"])
    theta = float(geometry["theta"])
    alpha = float(geometry["alpha"])
    c0 = float(geometry["c0"])
    safe_radius = float(geometry["safe_radius"])

    colors = {
        "sector_fill": "#BCD7EA",
        "sector_edge": "#2F6B9A",
        "safe_fill": "#D6ECE7",
        "safe_edge": "#267A70",
        "envelope": "#C18A32",
        "centerline": "#294E70",
        "accent": "#A64040",
        "text": "#26323D",
        "grid": "#D7DDE3",
    }

    angles = np.linspace(theta - alpha, theta + alpha, 600)
    outer_arc = s1 + p.max_detection_radius * unit_vector(angles)
    inner_arc = s1 + p.near_radius * unit_vector(angles[::-1])
    sector_vertices = np.vstack((outer_arc, inner_arc))
    endpoint_minus = outer_arc[0]
    endpoint_plus = outer_arc[-1]
    center_endpoint = s1 + p.max_detection_radius * unit_vector(theta)
    normal = unit_vector(theta + np.pi / 2.0)

    fig, ax = plt.subplots(figsize=(8.4, 7.5))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.975, bottom=0.19)

    # 1. 几何包络圆：仅用细长虚线表示，使其处于视觉背景层。
    ax.add_patch(
        Circle(
            q0,
            c0,
            fill=False,
            edgecolor=colors["envelope"],
            linewidth=1.65,
            linestyle=(0, (7, 4)),
            zorder=1,
        )
    )

    # 2. 第二检测点候选域：浅青绿加稀疏纹理，兼顾彩色和灰度打印。
    ax.add_patch(
        Circle(
            q0,
            safe_radius,
            facecolor=colors["safe_fill"],
            edgecolor=colors["safe_edge"],
            linewidth=1.55,
            hatch="/",
            zorder=2,
        )
    )

    # 3. 首次目标可行域：蓝色窄扇形，绘于候选域之上以明确物理含义。
    ax.add_patch(
        Polygon(
            sector_vertices,
            closed=True,
            facecolor=colors["sector_fill"],
            edgecolor=colors["sector_edge"],
            linewidth=1.45,
            zorder=3,
        )
    )
    for angle in (theta - alpha, theta + alpha):
        endpoint = s1 + p.max_detection_radius * unit_vector(angle)
        ax.plot(
            [s1[0], endpoint[0]],
            [s1[1], endpoint[1]],
            color=colors["sector_edge"],
            linewidth=1.25,
            zorder=4,
        )
    ax.plot(
        outer_arc[:, 0],
        outer_arc[:, 1],
        color=colors["sector_edge"],
        linewidth=1.75,
        zorder=4,
    )

    # 中心示向线使用点划线，与所有区域边界区分。
    ax.plot(
        [s1[0], center_endpoint[0]],
        [s1[1], center_endpoint[1]],
        color=colors["centerline"],
        linewidth=1.4,
        linestyle=(0, (6, 3, 1.4, 3)),
        zorder=5,
    )

    # 标记包络圆与扇形外弧的两个接触点。
    ax.scatter(
        [endpoint_minus[0], endpoint_plus[0]],
        [endpoint_minus[1], endpoint_plus[1]],
        s=20,
        facecolor="white",
        edgecolor=colors["sector_edge"],
        linewidth=0.95,
        zorder=7,
    )
    ax.annotate(
        rf"${p.bearing_deg - p.angle_error_deg:g}^\circ$",
        xy=endpoint_minus,
        xytext=(9, -15),
        textcoords="offset points",
        fontsize=8.8,
        color=colors["sector_edge"],
    )
    ax.annotate(
        rf"${p.bearing_deg + p.angle_error_deg:g}^\circ$",
        xy=endpoint_plus,
        xytext=(-29, 9),
        textcoords="offset points",
        fontsize=8.8,
        color=colors["sector_edge"],
    )

    # S1 和 q0：只将 q0 设为暗红强调色。
    ax.scatter(
        *s1,
        s=54,
        facecolor="#202830",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    ax.scatter(
        *q0,
        s=67,
        marker="D",
        facecolor=colors["accent"],
        edgecolor="white",
        linewidth=0.8,
        zorder=9,
    )
    ax.annotate(
        r"$S_1=(0,0)$",
        xy=s1,
        xytext=(-17, -25),
        textcoords="offset points",
        fontsize=9.8,
        color=colors["text"],
    )
    ax.annotate(
        r"$q_0$",
        xy=q0,
        xytext=(10, -12),
        textcoords="offset points",
        fontsize=10.5,
        fontweight="bold",
        color=colors["accent"],
    )

    # 候选圆盘半径尺寸线保留唯一关键数值 249.89 m。
    safe_endpoint = q0 + safe_radius * normal
    ax.add_patch(
        FancyArrowPatch(
            q0,
            safe_endpoint,
            arrowstyle="<->",
            mutation_scale=9,
            linewidth=1.2,
            color=colors["safe_edge"],
            zorder=7,
        )
    )
    safe_midpoint = (q0 + safe_endpoint) / 2.0
    ax.annotate(
        rf"$r_{{\rm safe}}={safe_radius:.2f}\ \mathrm{{m}}$",
        xy=safe_midpoint,
        xytext=(-100, 20),
        textcoords="offset points",
        fontsize=9.3,
        color=colors["safe_edge"],
    )

    # 就地短标签减少图例负担，同时避免把两类区域混为一谈。
    ax.text(
        q0[0] + 132.0,
        q0[1] - 158.0,
        r"第二检测点候选域 $\mathcal{C}_{\rm safe}$",
        ha="center",
        va="center",
        fontsize=9.0,
        color=colors["safe_edge"],
        zorder=7,
    )

    envelope_label = q0 + c0 * unit_vector(np.deg2rad(107.0))
    ax.annotate(
        r"保守几何包络圆 $B(q_0,c_0)$",
        xy=envelope_label,
        xytext=(-4, 13),
        textcoords="offset points",
        ha="center",
        fontsize=9.0,
        color="#8D631D",
    )

    sector_label = s1 + 1260.0 * unit_vector(theta)
    ax.annotate(
        rf"目标可行域 $\Omega_1$："
        rf"$\hat\theta_1\pm{p.angle_error_deg:g}^\circ$",
        xy=sector_label,
        xytext=(-103, 34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": colors["sector_edge"], "lw": 0.8},
        fontsize=9.3,
        color=colors["sector_edge"],
    )

    center_label = s1 + 1050.0 * unit_vector(theta)
    ax.annotate(
        rf"中心线 $\hat\theta_1={p.bearing_deg:g}^\circ$",
        xy=center_label,
        xytext=(37, -40),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": colors["centerline"], "lw": 0.75},
        fontsize=8.9,
        color=colors["centerline"],
    )

    # 三项横排图例置于图外，正文中的图注负责说明完整数学含义。
    legend_handles = [
        Patch(
            facecolor=colors["sector_fill"],
            edgecolor=colors["sector_edge"],
            label=(
                r"目标可行域 $\Omega_1$"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=colors["envelope"],
            linewidth=1.65,
            linestyle=(0, (7, 4)),
            label=r"外包圆 $B(q_0,c_0)$",
        ),
        Patch(
            facecolor=colors["safe_fill"],
            edgecolor=colors["safe_edge"],
            hatch="/",
            label=r"候选域 $\mathcal{C}_{\rm safe}$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        fontsize=8.5,
        frameon=False,
        handlelength=2.3,
        handleheight=1.1,
        columnspacing=1.35,
    )

    # 科研论文规范：图内不重复写“图3”和图名，由排版软件生成下方题注。
    ax.set_xlabel(r"$x$ / m", fontsize=10.2, color=colors["text"])
    ax.set_ylabel(r"$y$ / m", fontsize=10.2, color=colors["text"])
    ax.set_xlim(-250.0, 1380.0)
    ax.set_ylim(-250.0, 1380.0)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(250.0))
    ax.yaxis.set_major_locator(MultipleLocator(250.0))
    ax.tick_params(
        axis="both",
        labelsize=8.8,
        colors="#4B5563",
        length=3.4,
        width=0.75,
    )
    ax.grid(
        True,
        linestyle=(0, (2, 4)),
        linewidth=0.52,
        color=colors["grid"],
        alpha=0.8,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7A8591")
    ax.spines["bottom"].set_color("#7A8591")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.savefig(FIGURE_PNG, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    geometry = calculate_geometry(PARAM)
    checks = verify_geometry(PARAM, geometry)
    draw_publication_figure(PARAM, geometry)

    q0 = np.asarray(geometry["q0"])
    print("论文版图已生成")
    print(f"q0 = ({q0[0]:.6f}, {q0[1]:.6f}) m")
    print(f"c0 = {float(geometry['c0']):.6f} m")
    print(f"r_safe = {float(geometry['safe_radius']):.6f} m")
    print(
        "网格覆盖校验最大距离 = "
        f"{checks['sampled_max_envelope_distance']:.6f} m"
    )
    print(f"PNG: {FIGURE_PNG}")
    print(f"PDF: {FIGURE_PDF}")


if __name__ == "__main__":
    main()
