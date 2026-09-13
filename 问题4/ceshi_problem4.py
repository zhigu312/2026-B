"""B题问题4独立线上程序：DUSO方向感知搜索 + 几何定位清除。

本文件只复用 ceshi.py 经验证的接口、角域、MEC/Jung 与 20 m
清除逻辑；不修改问题3代码、Q表或 runs 目录。问题4的 no_signal
只用于“位置-方向-半径-类型”联合状态的非对称更新，不作为
全向源式的距离排除证据。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ceshi as base

Point = Tuple[float, float]
SCRIPT_DIR = Path(__file__).resolve().parent
P4_RUN_ROOT = SCRIPT_DIR / "runs_problem_4"
P4_POLICY_PATH = SCRIPT_DIR / "rl_policy_problem_4.json"
P4_LATTICE_SPACING_M = 940.0
P4_INNER_RING_RADIUS_M = 1050.0
P4_INNER_RING_COUNT = 7
P4_FAST_OUTER_RING_RADIUS_M = 1700.0
P4_FAST_OUTER_RING_COUNT = 12
P4_LEGACY_OUTER_RING_RADIUS_M = 1900.0
P4_LEGACY_OUTER_RING_COUNT = 12


def p4_directional_skeleton(
    certified: bool = False, legacy: bool = False
) -> List[Point]:
    """空间+方位覆盖骨架。

    v5默认采用中心+7个1050 m内环点+12个1700 m外环点。它不降低
    v4的方位采样密度，只把外环向任务区收缩，减少外环周向移动和
    内外环切换成本。稀疏到10个外环点虽更快，但大样本出现漏清，
    因而不作为正式方案。legacy=True可恢复v4的12点1900 m外环。
    """
    if not certified:
        inner = base.regular_ring(
            P4_INNER_RING_RADIUS_M,
            P4_INNER_RING_COUNT,
            180.0 / P4_INNER_RING_COUNT,
        )
        if legacy:
            outer_radius = P4_LEGACY_OUTER_RING_RADIUS_M
            outer_count = P4_LEGACY_OUTER_RING_COUNT
            outer_phase = 15.0
        else:
            outer_radius = P4_FAST_OUTER_RING_RADIUS_M
            outer_count = P4_FAST_OUTER_RING_COUNT
            # 沿用v4验证过的相位，避免同时改变半径与方位采样结构。
            outer_phase = 15.0
        outer = base.regular_ring(outer_radius, outer_count, outer_phase)
        return [(0.0, 0.0)] + base.nearest_neighbor_order(inner + outer)

    # 指导论文第7.3节的连续域覆盖证书：19个三角晶格点加6个外点。
    # h=940 m时，所有覆盖三角形最长边973.160 m < 最小接收半径，
    # 25点凸包内切半径1815.941 m > 目标圆半径1800 m。
    h = P4_LATTICE_SPACING_M
    lattice: List[Point] = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                lattice.append((
                    h * (q + 0.5 * r),
                    h * (math.sqrt(3.0) / 2.0 * r),
                ))
    outer = [
        (
            2.0 * h * math.cos(math.radians(30.0 + 60.0 * k)),
            2.0 * h * math.sin(math.radians(30.0 + 60.0 * k)),
        )
        for k in range(6)
    ]
    origin = (0.0, 0.0)
    others = [p for p in lattice + outer if base.distance(p, origin) > 1e-6]
    return [origin] + base.nearest_neighbor_order(others)


class DirectionAwareSearchAgent(base.SearchAgent):
    """问题4 DUSO扩展：保留清除硬门，仅改写发现与规划语义。"""

    def __init__(self, client, policy, **kwargs):
        kwargs.pop("problem", None)
        self.certified_directional_skeleton = kwargs.pop(
            "certified_directional_skeleton", False
        )
        self.legacy_directional_skeleton = kwargs.pop(
            "legacy_directional_skeleton", True
        )
        self.multiscale_directional_probes = kwargs.pop(
            "multiscale_directional_probes", False
        )
        self.tail_guard_enabled = kwargs.pop("tail_guard_enabled", True)
        self.tail_guard_max_channels = max(
            1, int(kwargs.pop("tail_guard_max_channels", 3))
        )
        self.tail_guard_min_route_m = max(
            0.0, float(kwargs.pop("tail_guard_min_route_m", 1400.0))
        )
        self.tail_guard_budget_s = max(
            120.0, float(kwargs.pop("tail_guard_budget_s", 900.0))
        )
        self.coverage_continuity_enabled = kwargs.pop(
            "coverage_continuity_enabled", False
        )
        self.coverage_continuity_near_m = max(
            0.0, float(kwargs.pop("coverage_continuity_near_m", 1050.0))
        )
        self.coverage_continuity_far_m = max(
            self.coverage_continuity_near_m,
            float(kwargs.pop("coverage_continuity_far_m", 1250.0)),
        )
        self.coverage_continuity_min_saving_m = max(
            0.0, float(kwargs.pop("coverage_continuity_min_saving_m", 260.0))
        )
        self.directional_coverage_crossing_threshold = max(
            0.0, min(1.0, float(kwargs.pop(
                "directional_coverage_crossing_threshold", 0.42
            )))
        )
        self.joint_scenario_enabled = kwargs.pop("joint_scenario_enabled", True)
        self.scenario_planning_weight = max(
            0.0, min(1.0, float(kwargs.pop("scenario_planning_weight", 0.0)))
        )
        self.max_scenarios = max(96, int(kwargs.pop("max_scenarios", 384)))
        self.scenario_direction_step_deg = max(
            15.0, min(90.0, float(kwargs.pop("scenario_direction_step_deg", 30.0)))
        )
        # 定向源的负观测不能直接推断距离，强制关闭P3负圆推断。
        kwargs["negative_range_inference"] = False
        kwargs.setdefault("joint_coverage_localization", True)
        kwargs.setdefault("joint_coverage_warmup", 2)
        kwargs.setdefault("joint_probe_variants", True)
        kwargs.setdefault("directed_plan_order", True)
        kwargs.setdefault("information_rate_planning", True)
        # v8更强调信息增益/移动米数，而不是单纯追求最大信息量：限制
        # 单次跨场跳跃，并避免为很小的理论增益接受过长绕行。
        kwargs.setdefault("route_tail_penalty", 0.48)
        kwargs.setdefault("rigo_route_credit_m", 12.0)
        super().__init__(client, policy, problem=4, **kwargs)
        self.scan_waypoints = p4_directional_skeleton(
            self.certified_directional_skeleton,
            self.legacy_directional_skeleton,
        )
        self.batch_size = len(self.scan_waypoints)
        self.directional_negative_updates = 0
        self.directional_positive_updates = 0
        self.azimuth_skeleton_complete = False
        self.tail_guard_activations = 0
        self.tail_guard_channels = 0
        self.tail_guard_optical_fallbacks = 0
        self.coverage_continuity_overrides = 0
        self._scenario_cache: Dict[tuple, list] = {}
        self.scenario_build_count = 0
        self.scenario_empty_fallbacks = 0

    @staticmethod
    def _scenario_visible(target: Point, sensor: Point, phi_deg: float) -> bool:
        source_to_sensor = base.normalize_angle(math.degrees(math.atan2(
            sensor[1] - target[1], sensor[0] - target[0]
        )))
        return abs(base.angle_difference(source_to_sensor, phi_deg)) <= 90.0 + 1e-9

    def _scenario_key(self, state) -> tuple:
        return (
            state.channel,
            tuple((round(o.position[0], 2), round(o.position[1], 2),
                   round(o.bearing_deg, 2)) for o in state.observations),
            tuple((round(p[0], 2), round(p[1], 2))
                  for p in state.no_signal_positions),
        )

    def directional_scenarios(self, state):
        """构造(G,R,type,phi)联合集合，并与全部正负观测保持一致。

        场景集只服务于补测点和路线评分；MEC/Jung清除硬门完全不变。
        因此离散误差最多使规划退回旧算法，不会放宽清除条件。
        """
        if not self.joint_scenario_enabled or not state.observations:
            return []
        key = self._scenario_key(state)
        cached = self._scenario_cache.get(key)
        if cached is not None:
            return cached
        polygon = base.localization_geometry(state.observations).vertices
        if not polygon:
            self._scenario_cache[key] = []
            return []

        samples = list(polygon)
        samples.extend((
            ((polygon[i][0] + polygon[(i + 1) % len(polygon)][0]) / 2.0,
             (polygon[i][1] + polygon[(i + 1) % len(polygon)][1]) / 2.0)
            for i in range(len(polygon))
        ))
        min_x, max_x = min(p[0] for p in polygon), max(p[0] for p in polygon)
        min_y, max_y = min(p[1] for p in polygon), max(p[1] for p in polygon)
        for ix in range(8):
            for iy in range(8):
                point = (
                    min_x + (ix + 0.5) * (max_x - min_x) / 8.0,
                    min_y + (iy + 0.5) * (max_y - min_y) / 8.0,
                )
                if base.point_in_convex_polygon(point, polygon):
                    samples.append(point)

        scenarios = []
        phi_values = [
            float(v) for v in range(0, 360, int(self.scenario_direction_step_deg))
        ]
        for target in samples:
            positive_distances = [
                base.distance(target, obs.position) for obs in state.observations
            ]
            lower = max(base.MIN_RECEIVE_RADIUS, max(positive_distances))
            if lower > 1500.0 + 1e-6:
                continue
            radius_values = sorted({
                round(lower, 3), round((lower + 1500.0) / 2.0, 3), 1500.0
            })
            for receive_radius in radius_values:
                radius_weight = max(5.0, 1500.0 - lower + 5.0) / len(radius_values)
                # 全向假设：负观测只能由超出固定接收半径解释。
                if all(
                    base.distance(target, p) > receive_radius + 1e-6
                    for p in state.no_signal_positions
                ):
                    scenarios.append((target, radius_weight * 0.70,
                                      receive_radius, "omni", None))
                # 定向假设：所有正观测必须可见；负观测可由超距或背向解释。
                for phi in phi_values:
                    if not all(
                        self._scenario_visible(target, obs.position, phi)
                        for obs in state.observations
                    ):
                        continue
                    if not all(
                        base.distance(target, p) > receive_radius + 1e-6
                        or not self._scenario_visible(target, p, phi)
                        for p in state.no_signal_positions
                    ):
                        continue
                    scenarios.append((target, radius_weight * 0.30 / len(phi_values),
                                      receive_radius, "directional", phi))

        # 确定性分层抽稀，避免仅保留列表前部的空间或方向状态。
        if len(scenarios) > self.max_scenarios:
            stride = len(scenarios) / self.max_scenarios
            scenarios = [
                scenarios[min(len(scenarios) - 1, int((i + 0.5) * stride))]
                for i in range(self.max_scenarios)
            ]
        self.scenario_build_count += 1
        if not scenarios:
            self.scenario_empty_fallbacks += 1
        self._scenario_cache[key] = scenarios
        if len(self._scenario_cache) > 2048:
            self._scenario_cache.clear()
        return scenarios

    def optimized_plan_order(self, plans, end_hint):
        """抑制滚动规划在覆盖点和远距离目标之间来回跳转。

        父类求解的是当前候选集合的开放有向路线，但实际每次只执行
        第一个动作并立即重规划，目标出口变化时可能产生跨场振荡。
        若首动作需要长跳，而附近存在尚未完成的公共覆盖点，则先完成
        该近邻覆盖点；不删除任何覆盖任务，也不改变清除硬门。
        """
        order = super().optimized_plan_order(plans, end_hint)
        if not self.coverage_continuity_enabled or not order:
            return order
        first = order[0]
        coverage = [
            index for index, plan in enumerate(plans)
            if plan.get("kind") == "coverage"
        ]
        if not coverage:
            return order
        # 已决定执行覆盖时，把“下一覆盖点”交给纯覆盖开放路径求解。
        # 这样定位候选的动态出口不会在每轮重规划时打乱覆盖子路径。
        if plans[first].get("kind") == "coverage":
            pure_order = base.optimized_visit_order(
                [plans[index]["point"] for index in coverage],
                self.client.position,
                end_hint,
            )
            preferred = coverage[pure_order[0]]
            first_distance = base.distance(
                self.client.position, plans[first]["point"]
            )
            preferred_distance = base.distance(
                self.client.position, plans[preferred]["point"]
            )
            if (
                preferred != first
                and first_distance - preferred_distance
                >= self.coverage_continuity_min_saving_m
            ):
                self.coverage_continuity_overrides += 1
                return [preferred] + [index for index in order if index != preferred]
            return order
        nearest = min(
            coverage,
            key=lambda index: base.distance(
                self.client.position, plans[index]["point"]
            ),
        )
        first_distance = base.distance(
            self.client.position, plans[first]["point"]
        )
        coverage_distance = base.distance(
            self.client.position, plans[nearest]["point"]
        )
        if (
            first_distance >= self.coverage_continuity_far_m
            and coverage_distance <= self.coverage_continuity_near_m
            and first_distance - coverage_distance
            >= self.coverage_continuity_min_saving_m
        ):
            self.coverage_continuity_overrides += 1
            return [nearest] + [index for index in order if index != nearest]
        return order

    def measure_channel(self, point: Point, channel: int):
        result = super().measure_channel(point, channel)
        if result:
            if result.get("measure_result") == "no_signal":
                # DUSO非对称更新：仅记录“该联合状态在此点不可见”。
                # 绝不将位置直接排除在以1000 m为半径的圆外。
                self.directional_negative_updates += 1
            elif result.get("measure_result") in ("direction", "near"):
                self.directional_positive_updates += 1
        return result

    def negative_feasible_hypotheses(self, state):
        """向父类暴露scenario的位置-半径边缘分布，绝不做P3负圆误剪。"""
        scenarios = self.directional_scenarios(state)
        if scenarios and self.scenario_planning_weight > 0.0:
            return [
                (target, weight, receive_radius)
                for target, weight, receive_radius, _, _ in scenarios
            ]
        # 联合离散集为空时退回只依赖正观测的旧安全规划。
        if not state.observations:
            return []
        polygon = base.localization_geometry(state.observations).vertices
        result = []
        samples = list(polygon)
        samples.extend((
            ((polygon[i][0] + polygon[(i + 1) % len(polygon)][0]) / 2.0,
             (polygon[i][1] + polygon[(i + 1) % len(polygon)][1]) / 2.0)
            for i in range(len(polygon))
        ))
        if polygon:
            min_x, max_x = min(p[0] for p in polygon), max(p[0] for p in polygon)
            min_y, max_y = min(p[1] for p in polygon), max(p[1] for p in polygon)
            for ix in range(7):
                for iy in range(7):
                    point = (
                        min_x + (ix + 0.5) * (max_x - min_x) / 7.0,
                        min_y + (iy + 0.5) * (max_y - min_y) / 7.0,
                    )
                    if base.point_in_convex_polygon(point, polygon):
                        samples.append(point)
        for point in samples:
            lower = max(
                base.MIN_RECEIVE_RADIUS,
                max(base.distance(point, obs.position) for obs in state.observations),
            )
            if lower <= 1500.0:
                result.append((point, max(2.0, 1500.0 - lower),
                               (lower + 1500.0) / 2.0))
        return result

    def planning_target(self, state):
        scenarios = self.directional_scenarios(state)
        if not scenarios or self.scenario_planning_weight <= 0.0:
            return super().planning_target(state)
        total = sum(item[1] for item in scenarios)
        center = (
            sum(item[0][0] * item[1] for item in scenarios) / total,
            sum(item[0][1] * item[1] for item in scenarios) / total,
        )
        sigma = math.sqrt(sum(
            item[1] * base.distance(item[0], center) ** 2 for item in scenarios
        ) / total)
        return center, sigma

    def rigo_expected_gain(self, state, point: Point) -> float:
        """位置优先的联合观测信息增益，方向/类型仅用于保持可见性。"""
        scenarios = self.directional_scenarios(state)
        if not scenarios or self.scenario_planning_weight <= 0.0:
            return super().rigo_expected_gain(state, point)
        total = sum(item[1] for item in scenarios)
        groups: Dict[tuple, float] = {}
        signal_weight = near_weight = directional_geometry = 0.0
        first = state.observations[0]
        for target, weight, receive_radius, source_type, phi in scenarios:
            probe_range = base.distance(point, target)
            visible = (
                source_type == "omni"
                or self._scenario_visible(target, point, float(phi))
            )
            if probe_range > receive_radius or not visible:
                outcome = ("no_signal",)
            elif probe_range <= 5.0:
                outcome = ("near",)
                near_weight += weight
                signal_weight += weight
            else:
                bearing = base.normalize_angle(math.degrees(math.atan2(
                    target[1] - point[1], target[0] - point[0]
                )))
                outcome = ("direction", int(round(bearing / 4.0)) % 90)
                signal_weight += weight
                crossing = abs(math.sin(math.radians(base.angle_difference(
                    first.bearing_deg, bearing
                ))))
                metric_error = max(
                    2.0,
                    probe_range * math.tan(math.radians(
                        base.FEASIBLE_BEARING_TOLERANCE_DEG
                    )),
                )
                directional_geometry += weight * crossing * math.log1p(
                    1500.0 / metric_error
                )
            groups[outcome] = groups.get(outcome, 0.0) + weight

        observation_entropy = -sum(
            (weight / total) * math.log(max(1e-12, weight / total))
            for weight in groups.values()
        )
        signal_probability = signal_weight / total
        no_signal_probability = 1.0 - signal_probability
        clear_probability = sum(
            item[1] for item in scenarios
            if base.distance(point, item[0]) <= base.CLEAR_RADIUS
        ) / total
        # 位置/清除优先；类型和phi只通过观测分组及可见概率间接参与。
        scenario_gain = max(0.0,
            1.55 * observation_entropy
            + directional_geometry / total
            + 5.0 * max(near_weight / total, clear_probability)
            + 0.35 * signal_probability
            - 0.45 * no_signal_probability
        )
        # 与位置几何基线做保守残差融合；权重为0即字节逻辑等价的旧规划。
        legacy_gain = super().rigo_expected_gain(state, point)
        return (
            (1.0 - self.scenario_planning_weight) * legacy_gain
            + self.scenario_planning_weight * scenario_gain
        )

    def coverage_measure_worthwhile(self, state, point: Point) -> bool:
        # 未发现频道必须扫，否则无法保证定向源发现完备性。
        if not state.observations:
            return True
        if state.cleared or self.localization_ready(state):
            return False
        # 已发现频道的公共复测仅在有较好交会角时执行；即使无信号
        # 也不会误删位置可行域。
        first = state.observations[0]
        estimate, _ = base.estimate_target(state.observations)
        if estimate is None:
            estimate, _ = self.first_ray_range_posterior(state)
        predicted = base.normalize_angle(math.degrees(math.atan2(
            estimate[1] - point[1], estimate[0] - point[0]
        )))
        crossing = abs(math.sin(math.radians(
            base.angle_difference(first.bearing_deg, predicted)
        )))
        return (
            base.distance(point, estimate) <= 1500.0
            and crossing >= self.directional_coverage_crossing_threshold
        )

    def probe_candidates(self, first, attempt: int, state=None):
        """定向源的可见侧短基线补测。

        P3的820 m前探容易越过距首测点仅200~700 m的定向源，使后续点
        全部落到辐射背面。这里用35/90/190/360/600 m递增步长，先在
        已知可见侧取得第二条示向线，再由交会估计加速接近。左右成对
        的小偏置用来抵抗首点恰好位于±90°方向边界的情况。
        """
        ux, uy = base.bearing_vector(first.bearing_deg)
        px, py = -uy, ux
        upper = min(
            1500.0,
            base.ray_distance_to_arena_exit(first.position, first.bearing_deg),
        )
        forward_schedule = (35.0, 90.0, 190.0, 360.0, 600.0, 820.0)
        lateral_schedule = (12.0, 25.0, 50.0, 80.0, 110.0, 140.0)
        index = min(max(0, attempt), len(forward_schedule) - 1)
        forward = min(upper, forward_schedule[index])
        lateral = lateral_schedule[index]
        center = (
            first.position[0] + forward * ux,
            first.position[1] + forward * uy,
        )
        candidates = {
            "probe_left": (center[0] + lateral * px, center[1] + lateral * py),
            "probe_right": (center[0] - lateral * px, center[1] - lateral * py),
            "probe_forward": center,
        }
        if self.multiscale_directional_probes:
            # 一次把论文的多尺度思想交给RIGO比较，避免必须按
            # 35/90/190/...逐次失败后才能到达有效中基线。
            # 横移量随前进距离增长，且限制在160 m内，尽量留在首测
            # 已知的可见侧；最终清除仍只由MEC/Jung硬门触发。
            for forward in (180.0, 350.0, 550.0):
                if forward > upper + 1e-6:
                    continue
                side = min(160.0, max(45.0, 0.28 * forward))
                c = (
                    first.position[0] + forward * ux,
                    first.position[1] + forward * uy,
                )
                candidates[f"directional_{int(forward)}_left"] = (
                    c[0] + side * px, c[1] + side * py
                )
                candidates[f"directional_{int(forward)}_right"] = (
                    c[0] - side * px, c[1] - side * py
                )
        return candidates

    def recover_clear_locally(self, state) -> bool:
        """保留原MEC/交点补救，再对定向背面导致的边界偏差做局部光学兜底。"""
        if super().recover_clear_locally(state):
            return True
        return self._directional_optical_recovery(state)

    def localize_and_clear(self, state) -> bool:
        if super().localize_and_clear(state):
            return True
        return self._directional_optical_recovery(state)

    def _directional_optical_recovery(self, state) -> bool:
        if not state.failed_clear_positions:
            return False
        center = state.failed_clear_positions[-1]
        # 离线反例中，多条同侧示向线使MEC中心距真源21.9 m，仅超出
        # 清除半径1.9 m。半径18 m的8点环能覆盖这类小偏差；只在清除失败
        # 后触发，不增加正常样本的虚拟时间。
        ring = base.regular_ring(18.0, 8, 22.5)
        candidates = [(center[0] + dx, center[1] + dy) for dx, dy in ring]
        order = base.optimized_visit_order(candidates, self.client.position)
        previous_phase = self.client.phase
        self.client.phase = "directional_optical_recovery"
        try:
            for index in order:
                if self.clear_at(state, candidates[index]):
                    return True
        finally:
            self.client.phase = previous_phase
        return False

    def batch_localize_and_clear(self, states, end_hint) -> None:
        """少量困难目标的长尾接管，避免末尾重复全场批路径。

        联合规划已经为每个困难频道执行过多次补测。若末尾只剩1~3个
        目标，父类再做五轮batch_probe会在目标分散时重复横穿场地。
        此处按当前位置重排目标，每个频道只追加一次信息率最高的补测，
        随即尝试MEC/局部恢复；仍失败则进入论文给出的有限光学完备路径。
        大于阈值的正常批任务仍完全沿用原实现。
        """
        active = [
            state for state in states
            if state.observations and not state.cleared
        ]
        exhausted = [
            state for state in active
            if len(self.used_probe_actions[state.channel]) >= 2
        ]
        # 按全部未清频道评估跨场风险。旧版只统计已用完3类动作的频道，
        # 会漏掉“已有2次补测但目标相距很远”的典型长尾。
        active_estimates = []
        for state in active:
            estimate, _ = base.estimate_target(state.observations)
            active_estimates.append(
                estimate or state.observations[-1].position
            )
        projected_route_m = base.optimized_route_distance(
            active_estimates, self.client.position, end_hint
        ) if active_estimates else 0.0
        if (
            not self.tail_guard_enabled
            or not active
            or len(active) > self.tail_guard_max_channels
            or not exhausted
            or projected_route_m < self.tail_guard_min_route_m
        ):
            return super().batch_localize_and_clear(states, end_hint)

        self.tail_guard_activations += 1
        self.tail_guard_channels += len(active)
        estimates = active_estimates
        order = base.optimized_visit_order(
            estimates, self.client.position, end_hint
        )
        guard_started_s = self.client.virtual_time_s

        for index in order:
            state = active[index]
            if state.cleared:
                continue
            # 每个频道至多增加一次最高信息率补测，不再按轮次重复全场巡回。
            plan = self.plan_probe(
                state, len(self.used_probe_actions[state.channel])
            )
            if plan is not None:
                self.client.phase = "tail_guard_probe"
                plan["next_route_point"] = None
                plan["future_coverage_points"] = []
                self.execute_probe(plan)
            if state.cleared:
                continue

            ready, certified_point, _ = base.guaranteed_clearance(
                state.observations
            )
            estimate, _ = base.estimate_target(state.observations)
            target = certified_point if ready else estimate
            if target is not None:
                self.client.phase = "tail_guard_clear"
                if self.clear_at(state, target):
                    continue
            self.client.phase = "tail_guard_local_recovery"
            if self.recover_clear_locally(state):
                continue

            # 光学动作与发射方向无关。该两车道有限路径覆盖首次正观测
            # 的完整角域，只在所有常规恢复失败后承担100%清除兜底。
            self.tail_guard_optical_fallbacks += 1
            self.client.phase = "tail_guard_optical_fallback"
            self.certified_optical_clear(state)

            # 一旦局部闭环累计耗时超过预算，余下频道直接走有限完备路径，
            # 用可预测的上界替代最多5轮跨场补测，专门压制万秒级尾部。
            if self.client.virtual_time_s - guard_started_s > self.tail_guard_budget_s:
                for rest_index in order:
                    rest = active[rest_index]
                    if rest.cleared:
                        continue
                    self.tail_guard_optical_fallbacks += 1
                    self.client.phase = "tail_guard_budget_fallback"
                    self.certified_optical_clear(rest)
                break

    def run(self) -> dict:
        summary = super().run()
        self.azimuth_skeleton_complete = (
            len(self.sweep_positions) >= len(self.scan_waypoints)
            or self.cleared_count >= 16
        )
        scenario_counts = []
        directional_fractions = []
        radius_spans = []
        for state in self.channels.values():
            if not state.observations:
                continue
            scenarios = self.directional_scenarios(state)
            if not scenarios:
                continue
            total = sum(item[1] for item in scenarios)
            scenario_counts.append(len(scenarios))
            directional_fractions.append(sum(
                item[1] for item in scenarios if item[3] == "directional"
            ) / total)
            radii = [item[2] for item in scenarios]
            radius_spans.append(max(radii) - min(radii))
        summary.update({
            "algorithm": "DUSO_GIRH_ACCR_problem4_v8_stable_rigo",
            "directional_negative_updates": self.directional_negative_updates,
            "directional_positive_updates": self.directional_positive_updates,
            "azimuth_skeleton_complete": self.azimuth_skeleton_complete,
            "certified_directional_skeleton": self.certified_directional_skeleton,
            "legacy_directional_skeleton": self.legacy_directional_skeleton,
            "directional_skeleton_nodes": len(self.scan_waypoints),
            "directional_skeleton_open_route_m": sum(
                base.distance(self.scan_waypoints[i], self.scan_waypoints[i + 1])
                for i in range(len(self.scan_waypoints) - 1)
            ),
            "multiscale_directional_probes": self.multiscale_directional_probes,
            "tail_guard_enabled": self.tail_guard_enabled,
            "tail_guard_max_channels": self.tail_guard_max_channels,
            "tail_guard_min_route_m": self.tail_guard_min_route_m,
            "tail_guard_budget_s": self.tail_guard_budget_s,
            "tail_guard_activations": self.tail_guard_activations,
            "tail_guard_channels": self.tail_guard_channels,
            "tail_guard_optical_fallbacks": self.tail_guard_optical_fallbacks,
            "coverage_continuity_enabled": self.coverage_continuity_enabled,
            "coverage_continuity_overrides": self.coverage_continuity_overrides,
            "directional_coverage_crossing_threshold": (
                self.directional_coverage_crossing_threshold
            ),
            "joint_scenario_enabled": self.joint_scenario_enabled,
            "scenario_planning_weight": self.scenario_planning_weight,
            "max_scenarios": self.max_scenarios,
            "scenario_direction_step_deg": self.scenario_direction_step_deg,
            "scenario_build_count": self.scenario_build_count,
            "scenario_empty_fallbacks": self.scenario_empty_fallbacks,
            "mean_final_scenario_count": (
                sum(scenario_counts) / len(scenario_counts)
                if scenario_counts else 0.0
            ),
            "mean_final_directional_fraction": (
                sum(directional_fractions) / len(directional_fractions)
                if directional_fractions else 0.0
            ),
            "mean_final_radius_span_m": (
                sum(radius_spans) / len(radius_spans) if radius_spans else 0.0
            ),
            "problem4_run_root": str(P4_RUN_ROOT),
            "problem4_policy_path": str(P4_POLICY_PATH),
        })
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B题问题4 DUSO独立演练程序")
    parser.add_argument("--base-url", default=base.DEFAULT_BASE_URL)
    parser.add_argument("--robot-id", default=base.DEFAULT_ROBOT_ID)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--formal", action="store_true", help="关闭随机探索且不改写策略")
    parser.add_argument("--max-probes", type=int, default=5)
    parser.add_argument("--joint-warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot", action="store_true", help="可选生成轨迹图；不影响算法决策")
    parser.add_argument("--show", action="store_true", help="生成图后弹窗显示")
    parser.add_argument(
        "--certified-skeleton", action="store_true",
        help="使用论文25点严格方位覆盖证书；更稳健但通常更慢",
    )
    parser.add_argument(
        "--legacy-skeleton", action="store_true",
        help="A/B回归：恢复v4的中心+7内环+12外环覆盖骨架",
    )
    parser.add_argument(
        "--short-skeleton", action="store_true",
        help="兼容参数：1700 m外环现已是默认正式方案",
    )
    parser.add_argument(
        "--multiscale-directional-probes", action="store_true",
        help="兼容参数：多尺度定向补测现已默认启用",
    )
    parser.add_argument(
        "--no-multiscale-directional-probes", action="store_true",
        help="A/B回归：关闭多尺度定向补测候选",
    )
    parser.add_argument(
        "--no-tail-guard", action="store_true",
        help="仅用于A/B回归：关闭少量困难目标长尾接管",
    )
    parser.add_argument("--tail-guard-max-channels", type=int, default=5)
    parser.add_argument("--tail-guard-min-route-m", type=float, default=1050.0)
    parser.add_argument("--tail-guard-budget-s", type=float, default=720.0)
    parser.add_argument("--coverage-continuity", action="store_true")
    parser.add_argument(
        "--no-coverage-continuity", action="store_true",
        help="A/B回归：关闭覆盖子路径连续性约束",
    )
    parser.add_argument("--coverage-continuity-near-m", type=float, default=1050.0)
    parser.add_argument("--coverage-continuity-far-m", type=float, default=1250.0)
    parser.add_argument("--coverage-continuity-min-saving-m", type=float, default=260.0)
    parser.add_argument("--directional-coverage-crossing", type=float, default=0.34)
    parser.add_argument(
        "--no-joint-scenario", action="store_true",
        help="A/B回归：关闭位置-半径-类型-方向联合scenario规划",
    )
    parser.add_argument("--max-scenarios", type=int, default=384)
    parser.add_argument("--scenario-direction-step-deg", type=float, default=30.0)
    parser.add_argument(
        "--scenario-planning-weight", type=float, default=0.18,
        help="联合scenario参与补测评分的保守残差权重；0可恢复影子模式",
    )
    return parser.parse_args()


def main() -> None:
    program_started_perf = time.perf_counter()
    program_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    args = parse_args()
    if not args.robot_id.strip():
        raise SystemExit("请使用 --robot-id 填入模拟器分配的团队号/机器狗编号")
    if args.seed is not None:
        random.seed(args.seed)
    if args.max_probes <= 0 or args.joint_warmup <= 0:
        raise SystemExit("--max-probes和--joint-warmup必须为正整数")
    output_dir = Path(args.output_dir) if args.output_dir else (
        P4_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    epsilon = 0.0 if args.formal else max(0.0, min(1.0, args.epsilon))
    policy = base.QLearningPolicy(
        P4_POLICY_PATH, epsilon=epsilon, alpha=0.18, training=not args.formal
    )
    client = base.SimulatorClient(args.base_url, args.robot_id)
    agent = DirectionAwareSearchAgent(
        client, policy, max_probes=args.max_probes,
        joint_coverage_warmup=args.joint_warmup,
        certified_directional_skeleton=args.certified_skeleton,
        # 修复旧版参数反转：文档声明默认1700 m，但实际一直默认1900 m。
        # 现在正式模式使用中心+7内环+12个1700 m外环；显式
        # --legacy-skeleton 才回退到1900 m旧路线。
        legacy_directional_skeleton=args.legacy_skeleton,
        multiscale_directional_probes=(
            args.multiscale_directional_probes
            or not args.no_multiscale_directional_probes
        ),
        tail_guard_enabled=not args.no_tail_guard,
        tail_guard_max_channels=args.tail_guard_max_channels,
        tail_guard_min_route_m=args.tail_guard_min_route_m,
        tail_guard_budget_s=args.tail_guard_budget_s,
        coverage_continuity_enabled=(
            args.coverage_continuity or not args.no_coverage_continuity
        ),
        coverage_continuity_near_m=args.coverage_continuity_near_m,
        coverage_continuity_far_m=args.coverage_continuity_far_m,
        coverage_continuity_min_saving_m=args.coverage_continuity_min_saving_m,
        directional_coverage_crossing_threshold=args.directional_coverage_crossing,
        joint_scenario_enabled=not args.no_joint_scenario,
        max_scenarios=args.max_scenarios,
        scenario_direction_step_deg=args.scenario_direction_step_deg,
        scenario_planning_weight=args.scenario_planning_weight,
    )
    summary: Optional[dict] = None
    try:
        summary = agent.run()
        print("\n问题4运行汇总：")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("收到中断信号，尝试主动退出")
        client.exit()
    except Exception:
        client.exit()
        raise
    finally:
        policy.save()
        if summary is None:
            summary = {
                "problem": 4, "algorithm": "DUSO_GIRH_ACCR_problem4_v8_stable_rigo",
                "cleared_count": agent.cleared_count,
                "virtual_time_s": client.virtual_time_s,
                "path_distance_m": sum(r["move_distance_m"] for r in client.records),
                "action_count": len(client.records),
            }
        program_runtime_s = time.perf_counter() - program_started_perf
        summary["program_started_at"] = program_started_at
        summary["program_runtime_s"] = program_runtime_s
        summary["program_runtime_minutes"] = program_runtime_s / 60.0
        base.save_outputs(output_dir, client.records, summary, agent.channels)
        if args.plot:
            base.draw_path(
                output_dir / "problem4_robot_path.png", client.records,
                summary, show=args.show,
            )
        else:
            print("本次未请求绘图；日志已保存，不需要matplotlib。")
        print(
            f"程序现实运行时间：{program_runtime_s:.3f} s "
            f"({program_runtime_s / 60.0:.3f} min)"
        )
        print(f"问题4独立输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
