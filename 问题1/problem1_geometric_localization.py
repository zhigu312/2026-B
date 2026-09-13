# -*- coding: utf-8 -*-
"""
几何交集定位与直径覆盖问题：从零实现的计算几何验证算法
严格遵守 Antigravity 编码要求：半平面交暴力求解，手写旋转卡壳，手写 Welzl 算法。
特别定制 4 种理论几何构型，证明直径圆覆盖性的边界条件。
"""

import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches
from shapely.geometry import Polygon as ShapelyPolygon

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# ==========================================
# 1. 基础代数与半平面处理
# ==========================================
def line_intersection(l1, l2):
    (a1, b1, c1), (a2, b2, c2) = l1, l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return (x, y)

def polar_angle_sort(points):
    if len(points) <= 2: return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return points

def polygon_area(points):
    n = len(points)
    if n < 3: return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return abs(area) / 2.0

# ==========================================
# 2. 手写旋转卡壳 (Rotating Calipers) - 多边形直径
# ==========================================
def rotating_calipers(points):
    n = len(points)
    if n == 0: return 0, None, None
    if n == 1: return 0, points[0], points[0]
    
    max_d_sq = 0
    best_pair = (points[0], points[1])
    for i in range(n):
        for j in range(i+1, n):
            d_sq = (points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2
            if d_sq > max_d_sq:
                max_d_sq = d_sq
                best_pair = (points[i], points[j])
                
    return math.sqrt(max_d_sq), best_pair[0], best_pair[1]

# ==========================================
# 3. 手写 Welzl's 算法 - 最小外接圆 (MEC)
# ==========================================
def circle_from_2_points(p1, p2):
    cx = (p1[0] + p2[0]) / 2.0
    cy = (p1[1] + p2[1]) / 2.0
    r = math.hypot(p1[0]-p2[0], p1[1]-p2[1]) / 2.0
    return (cx, cy, r)

def circle_from_3_points(p1, p2, p3):
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2.0
    cd = (temp - p3[0]**2 - p3[1]**2) / 2.0
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1e-9:
        return circle_from_2_points(p1, p2)
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0])*cd - (p2[0] - p3[0])*bc) / det
    r = math.hypot(p2[0]-cx, p2[1]-cy)
    return (cx, cy, r)

def point_in_circle(p, c):
    return math.hypot(p[0]-c[0], p[1]-c[1]) <= c[2] + 1e-7

def welzl_helper(P, R, n):
    if n == 0 or len(R) == 3:
        if len(R) == 0: return (0, 0, 0)
        elif len(R) == 1: return (R[0][0], R[0][1], 0)
        elif len(R) == 2: return circle_from_2_points(R[0], R[1])
        else: return circle_from_3_points(R[0], R[1], R[2])
            
    idx = n - 1
    p = P[idx]
    c = welzl_helper(P, R, n - 1)
    if point_in_circle(p, c):
        return c
    R_new = R.copy()
    R_new.append(p)
    return welzl_helper(P, R_new, n - 1)

def welzl_mec(points):
    P = points.copy()
    random.shuffle(P) 
    return welzl_helper(P, [], len(P))

# ==========================================
# 4. 半平面暴力转化与交集多边形求解
# ==========================================
def solve_localization_polygon(sensors, bearings, error_deg=1.0):
    half_planes = []
    lines = []
    
    for (x0, y0), b_deg in zip(sensors, bearings):
        b1_rad = math.radians(b_deg - error_deg)
        b2_rad = math.radians(b_deg + error_deg)
        
        dx1, dy1 = math.cos(b1_rad), math.sin(b1_rad)
        a1, b1, c1 = dy1, -dx1, dy1 * x0 - dx1 * y0
        half_planes.append((a1, b1, c1))
        lines.append((a1, b1, c1))
        
        dx2, dy2 = math.cos(b2_rad), math.sin(b2_rad)
        a2, b2, c2 = -dy2, dx2, -dy2 * x0 + dx2 * y0
        half_planes.append((a2, b2, c2))
        lines.append((a2, b2, c2))
        
    BOUND = 100000
    bb_lines = [
        (1, 0, BOUND), (-1, 0, BOUND),
        (0, 1, BOUND), (0, -1, BOUND)
    ]
    half_planes.extend(bb_lines)
    lines.extend(bb_lines)
    
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt)
                
    valid_points = []
    for pt in intersections:
        x, y = pt
        valid = True
        for (a, b, c) in half_planes:
            if a * x + b * y > c + 1e-6:
                valid = False
                break
        if valid:
            valid_points.append(pt)
            
    unique_points = []
    for pt in valid_points:
        if not any(math.hypot(pt[0]-up[0], pt[1]-up[1]) < 1e-4 for up in unique_points):
            unique_points.append(pt)
            
    return polar_angle_sort(unique_points)

# ==========================================
# 5. 特化实验控制与渲染输出
# ==========================================
def run_experiment_and_plot(case3_no_circles_only=False, case3_polygon_only=False):
    # 根据用户要求，严格构造 4 种数学特例
    cases = [
        ("中心对称 (正方形) - 必定可包含 (m=4)", [0, 90, 180, 270]),
        ("一般图形 (不规则五边形) - 可包含 (m=5)", [10, 60, 120, 190, 280]),
        ("一般图形 (不规则六边形) - 不可包含 (m=6)", [0, 15, 100, 115, 230, 245]),
        ("正三角形 - 必定不可包含 (m=3)", [0, 120, 240])
    ]
    
    results = []
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for idx, (case_name, angles) in enumerate(cases):
        if (case3_no_circles_only or case3_polygon_only) and idx != 2:
            continue

        hide_circles = (case3_no_circles_only or case3_polygon_only) and idx == 2
        hide_diameter = case3_polygon_only and idx == 2
        m = len(angles)
        G = (5000.0, 5000.0)
        sensors = []
        bearings = []
        R = 4000.0
        
        # 巧妙设置 bearing = angle + 180 - 0.8
        # 这样探测站 +1 度的误差边界正好形成距离 G 点中心一定距离的切线，
        # 从而完美刻画出受角度控制的多边形。
        for ang_deg in angles:
            ang = math.radians(ang_deg)
            sx = G[0] + R * math.cos(ang)
            sy = G[1] + R * math.sin(ang)
            sensors.append((sx, sy))
            bearings.append(ang_deg + 180 - 0.8)
            
        poly = solve_localization_polygon(sensors, bearings)
        
        area = polygon_area(poly)
        D, p1, p2 = rotating_calipers(poly)
        cx, cy, r_mec = welzl_mec(poly)
        D_mec = r_mec * 2
        
        r_diam = D / 2.0
        c_diam = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        is_covered = all(point_in_circle(pt, (c_diam[0], c_diam[1], r_diam)) for pt in poly)
        
        results.append((case_name, len(poly), area, D, D_mec, "Yes" if D >= D_mec - 1e-5 else "No", "Yes" if is_covered else "No"))
        
        shapely_poly = ShapelyPolygon(poly)
        assert abs(shapely_poly.area - area) < 1e-3, "Shapely交叉验证面积不符"
        
        # --- 绘图: 并排双子图 (不会相互遮挡) ---
        fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
        
        for i, (p, b) in enumerate(zip(sensors, bearings)):
            c = palette[i % len(palette)]
            ax_global.plot(p[0], p[1], '^', color=c, markersize=12, markeredgecolor='black', label=f'探测站 {i+1}')
            rad1 = math.radians(b - 1.0)
            rad2 = math.radians(b + 1.0)
            p1_ext = (p[0] + 15000*math.cos(rad1), p[1] + 15000*math.sin(rad1))
            p2_ext = (p[0] + 15000*math.cos(rad2), p[1] + 15000*math.sin(rad2))
            ax_global.plot([p[0], p1_ext[0]], [p[1], p1_ext[1]], color=c, alpha=0.8, linewidth=1.5)
            ax_global.plot([p[0], p2_ext[0]], [p[1], p2_ext[1]], color=c, alpha=0.8, linewidth=1.5)
            ax_global.fill([p[0], p1_ext[0], p2_ext[0]], [p[1], p1_ext[1], p2_ext[1]], color=c, alpha=0.25)
            
        poly_arr = np.array(poly + [poly[0]])
        ax_global.plot(poly_arr[:,0], poly_arr[:,1], color='#E63946', linewidth=2.5)
        ax_global.fill(poly_arr[:,0], poly_arr[:,1], color='#E63946', alpha=0.7)
        
        ax_global.set_aspect('equal')
        ax_global.set_xlim(0, 10000)
        ax_global.set_ylim(0, 10000)
        ax_global.set_title(f'多点交会图 (m={m})', fontsize=16, fontweight='bold')
        ax_global.grid(True, linestyle='--', alpha=0.6)
        ax_global.legend(loc='upper right')
        
        # 2. 绘制局部精细图
        ax_local.plot(poly_arr[:,0], poly_arr[:,1], color='#E63946', linewidth=3, label='定位交集多边形')
        ax_local.fill(poly_arr[:,0], poly_arr[:,1], color='#E63946', alpha=0.6)
        
        if not hide_diameter:
            ax_local.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='--', color='#1d3557', linewidth=2, label='最大几何直径 $D$')
        if not hide_circles:
            diam_circle = Circle(c_diam, r_diam, edgecolor='#1d3557', facecolor='none', linestyle='--', linewidth=2, label='直径圆 (对角线为直径)')
            ax_local.add_patch(diam_circle)

            mec_circle = Circle((cx, cy), r_mec, edgecolor='#2a9d8f', facecolor='none', linestyle='-.', linewidth=2.5, label='最小包围圆 (MEC)')
            ax_local.add_patch(mec_circle)
        
        ax_local.set_aspect('equal')
        
        # 计算多边形的包围盒
        poly_minx, poly_maxx = min(poly_arr[:,0]), max(poly_arr[:,0])
        poly_miny, poly_maxy = min(poly_arr[:,1]), max(poly_arr[:,1])
        
        if hide_circles:
            minx, maxx = poly_minx, poly_maxx
            miny, maxy = poly_miny, poly_maxy
        else:
            # 将最小包围圆 (MEC) 的边界也考虑进来，防止圆被裁掉
            minx = min(poly_minx, cx - r_mec)
            maxx = max(poly_maxx, cx + r_mec)
            miny = min(poly_miny, cy - r_mec)
            maxy = max(poly_maxy, cy + r_mec)
        
        # 加上 20% 的视觉边距
        margin = max(maxx-minx, maxy-miny) * 0.2
        if margin < 1.0: margin = 10.0
        
        ax_local.set_xlim(minx - margin, maxx + margin)
        ax_local.set_ylim(miny - margin, maxy + margin)
        ax_local.grid(True, linestyle='--', alpha=0.6)
        ax_local.set_title('局部精细覆盖分析', fontsize=16, fontweight='bold')
        ax_local.legend(loc='best')
        
        # 透视连接线
        for point in [(minx - margin, maxy + margin), (minx - margin, miny - margin)]:
            con = ConnectionPatch(xyA=point, xyB=point, coordsA="data", coordsB="data", 
                                  axesA=ax_local, axesB=ax_global, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
            ax_global.add_artist(con)
            
        rect = patches.Rectangle((minx - margin, miny - margin), (maxx + margin) - (minx - margin), (maxy + margin) - (miny - margin), 
                                 linewidth=1.5, edgecolor='black', facecolor='none', linestyle=':')
        ax_global.add_patch(rect)
        
        if hide_diameter:
            output_name = 'case_3_polygon_only.png'
        elif hide_circles:
            output_name = 'case_3_no_circles.png'
        else:
            output_name = f'case_{idx+1}.png'
        plt.savefig(output_name, dpi=300, facecolor='white')
        plt.close()

    print("\n" + "="*60)
    print("LaTeX Table Code generated:")
    print("="*60)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{4种理论极限几何构型的直径圆覆盖验证}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"几何特例形态 & 顶点数量 & 面积 ($\mathrm{m}^2$) & 区域直径 $D$ ($\mathrm{m}$) & $D_{MEC}$ ($\mathrm{m}$) & $D \ge D_{MEC}$ & 直径圆可完全覆盖 \\")
    print(r"\midrule")
    for r in results:
        print(rf"{r[0]} & {r[1]} & {r[2]:.2f} & {r[3]:.2f} & {r[4]:.2f} & {r[5]} & {r[6]} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成交会定位几何图')
    parser.add_argument(
        '--case3-no-circles-only',
        action='store_true',
        help='只生成6个探测点的无圆版本，并另存为 case_3_no_circles.png'
    )
    parser.add_argument(
        '--case3-polygon-only',
        action='store_true',
        help='只生成6个探测点的无圆、无直径虚线版本，并另存为 case_3_polygon_only.png'
    )
    args = parser.parse_args()
    run_experiment_and_plot(
        case3_no_circles_only=args.case3_no_circles_only,
        case3_polygon_only=args.case3_polygon_only
    )
