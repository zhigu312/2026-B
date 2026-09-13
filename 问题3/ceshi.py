"""B题问题3/4：强化学习辅助的干扰源搜索、定位与清除演练程序。

核心思想：
1. 用覆盖航点保证全向源（问题3）的基础可发现性；问题4增加外圈航点处理定向源盲区。
2. 对已经测得示向度的频道，用Q-learning在多个二次测点方案中选择动作。
3. 用多条带±1度误差的测向线进行稳健交会估计，再调用20米光学清除。
4. 每轮演练保存Q表、CSV/JSON日志与增强二维轨迹图。

注意：正式测试前请先进行多轮演练，并把 --epsilon 调低至0或使用 --formal。
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARENA_RADIUS = 1800.0
MIN_RECEIVE_RADIUS = 1000.0
CLEAR_RADIUS = 20.0
BEARING_ERROR_DEG = 1.0
FEASIBLE_BEARING_TOLERANCE_DEG = 1.01
CLEARANCE_NUMERIC_MARGIN_M = 0.25
JUNG_CLEAR_DIAMETER_M = (CLEAR_RADIUS - CLEARANCE_NUMERIC_MARGIN_M) * math.sqrt(3.0)
P3_HEX_COVERAGE_RADIUS_M = 1128.0
OPTICAL_FALLBACK_RADIAL_STEP_M = 30.0
OPTICAL_FALLBACK_MAX_RANGE_M = 1500.0
DEFAULT_BASE_URL = "http://127.0.0.1:2026"
# 提交包不包含参赛团队号。运行时必须通过 --robot-id 显式传入。
DEFAULT_ROBOT_ID = ""

Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing_vector(deg: float) -> Point:
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)


def normalize_angle(deg: float) -> float:
    return deg % 360.0


def angle_difference(a: float, b: float) -> float:
    """返回[-180, 180)中的角差 a-b。"""
    return (a - b + 180.0) % 360.0 - 180.0


def ray_intersection(p1: Point, a1: float, p2: Point, a2: float) -> Optional[Point]:
    """求两条正向测向射线的交点；近似平行或交点位于射线后方时返回None。"""
    d1 = bearing_vector(a1)
    d2 = bearing_vector(a2)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-4:
        return None
    qx, qy = p2[0] - p1[0], p2[1] - p1[1]
    t = (qx * d2[1] - qy * d2[0]) / cross
    u = (qx * d1[1] - qy * d1[0]) / cross
    if t < -10.0 or u < -10.0:
        return None
    return p1[0] + t * d1[0], p1[1] + t * d1[1]


@dataclass
class Observation:
    position: Point
    bearing_deg: float
    virtual_time_s: float


@dataclass
class ChannelState:
    channel: int
    observations: List[Observation] = field(default_factory=list)
    no_signal_positions: List[Point] = field(default_factory=list)
    cleared: bool = False
    clear_position: Optional[Point] = None
    failed_clear_positions: List[Point] = field(default_factory=list)


@dataclass(frozen=True)
class FeasibleRegionSummary:
    """正测向角域的保守可行域及保证清除几何量。"""

    vertices: Tuple[Point, ...]
    center: Optional[Point]
    radius: float
    diameter: float
    area: float
    jung_certified: bool


class QLearningPolicy:
    """适合少量在线演练的轻量Q-learning宏动作策略。"""

    def __init__(
        self, path: Path, epsilon: float, alpha: float = 0.22,
        gamma: float = 0.85, training: bool = True,
    ):
        self.path = path
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = training
        self.q: Dict[str, Dict[str, float]] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.q = raw.get("q", raw)
                # 清除是环境终止操作而不是probe动作，清理旧版本遗留的高Q污染。
                for values in self.q.values():
                    if isinstance(values, dict):
                        values.pop("clear_estimate", None)
            except (OSError, ValueError, TypeError):
                print(f"警告：无法读取Q表 {path}，将从空策略开始")

    @staticmethod
    def state_key(problem: int, n_obs: int, uncertainty: float, last_signal: bool) -> str:
        obs_bin = min(n_obs, 3)
        if not math.isfinite(uncertainty):
            unc_bin = "inf"
        elif uncertainty <= 12:
            unc_bin = "low"
        elif uncertainty <= 30:
            unc_bin = "mid"
        else:
            unc_bin = "high"
        # v5隔离旧Q表：第二阶段由负观测可行域和RIGO重新定义动作价值。
        return f"v5|p{problem}|obs{obs_bin}|unc_{unc_bin}|sig{int(last_signal)}"

    def choose(self, state: str, actions: Sequence[str], heuristic: Dict[str, float]) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(actions))
        state_q = self.q.setdefault(state, {}) if self.training else self.q.get(state, {})
        return max(actions, key=lambda a: state_q.get(a, 0.0) + heuristic.get(a, 0.0))

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        if not self.training or action == "clear_estimate":
            return
        state_q = self.q.setdefault(state, {})
        old = state_q.get(action, 0.0)
        next_values = [
            value for name, value in self.q.get(next_state, {}).items()
            if name != "clear_estimate"
        ]
        target = reward + self.gamma * (max(next_values) if next_values else 0.0)
        state_q[action] = old + self.alpha * (target - old)

    def save(self) -> None:
        if not self.training:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "tabular_q_learning_macro_actions",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "q": self.q,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SimulatorClient:
    def __init__(self, base_url: str, robot_id: str, timeout: float = 6.0):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.timeout = timeout
        self.counter = 0
        self.position: Point = (0.0, 0.0)
        self.virtual_time_s = 0.0
        self.records: List[dict] = []
        self.entered = False
        self.closed = False
        self.phase = "idle"

    def _request_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}-{uuid.uuid4().hex[:8]}"

    def _base(self, request_id: str) -> dict:
        return {"arena_id": "default", "robot_id": self.robot_id, "request_id": request_id}

    def post(self, path: str, payload: dict, retries: int = 2) -> Optional[dict]:
        """网络故障重试时复用完全相同的payload和request_id。"""
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for attempt in range(retries + 1):
            request = Request(
                self.base_url + path,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    print(path, response.status, result)
                    return result
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(path, f"HTTP错误 {exc.code}: {detail}")
                return None
            except URLError as exc:
                print(path, f"连接异常（第{attempt + 1}次）: {exc.reason}")
                if attempt < retries:
                    time.sleep(0.35 * (attempt + 1))
        return None

    def enter(self) -> dict:
        result = self.post("/enter", self._base(self._request_id("enter")))
        if not result or result.get("accepted") is not True:
            raise RuntimeError("/enter失败：请确认演练已开始、接口已就绪、端口和参赛队号正确")
        self.entered = True
        self.closed = False
        self.virtual_time_s = float(result.get("virtual_time_s", 0.0))
        return result

    def exit(self) -> Optional[dict]:
        if not self.entered or self.closed:
            return None
        result = self.post("/exit", self._base(self._request_id("exit")), retries=0)
        self.closed = True
        return result

    def _action(self, path: str, position: Point, channel: int) -> Optional[dict]:
        old_pos = self.position
        old_time = self.virtual_time_s
        payload = self._base(self._request_id(path.strip("/")))
        payload.update({
            "position": {"x": round(position[0], 4), "y": round(position[1], 4)},
            "channel": int(channel),
        })
        result = self.post(path, payload)
        if not result or result.get("accepted") is not True:
            return None
        self.position = position
        self.virtual_time_s = float(result.get("virtual_time_s", old_time))
        rec = {
            "seq": len(self.records) + 1,
            "phase": self.phase,
            "path": path,
            "x": position[0],
            "y": position[1],
            "channel": channel,
            "measure_result": result.get("measure_result", ""),
            "clear_result": result.get("clear_result", ""),
            "svd_deg": result.get("svd_deg", ""),
            "virtual_time_s": self.virtual_time_s,
            "delta_virtual_time_s": self.virtual_time_s - old_time,
            "move_distance_m": distance(old_pos, position),
        }
        self.records.append(rec)
        return result

    def measure(self, position: Point, channel: int) -> Optional[dict]:
        return self._action("/measure", position, channel)

    def clear(self, position: Point, channel: int) -> Optional[dict]:
        return self._action("/clear", position, channel)


def regular_ring(radius: float, count: int, phase_deg: float = 0.0) -> List[Point]:
    return [
        (
            radius * math.cos(math.radians(phase_deg + 360.0 * i / count)),
            radius * math.sin(math.radians(phase_deg + 360.0 * i / count)),
        )
        for i in range(count)
    ]


def nearest_neighbor_order(points: Iterable[Point], start: Point = (0.0, 0.0)) -> List[Point]:
    remaining = list(points)
    ordered: List[Point] = []
    current = start
    while remaining:
        nxt = min(remaining, key=lambda p: distance(current, p))
        ordered.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return ordered


def coverage_waypoints(problem: int) -> List[Point]:
    # 问题3：中心+半径1128米的6点正六边形。连续圆域的最不利点位于
    # 场地边界两相邻外点的角平分线，距离约998.1米，小于最小接收半径。
    # 保留中心公共首测；离线消融表明无中心七边形虽路径短，但补测代价更高。
    if problem == 3:
        return [(0.0, 0.0)] + regular_ring(
            P3_HEX_COVERAGE_RADIUS_M, 6, 30.0
        )
    # 问题4：定向源可能只向圆域外辐射，因此增加目标区域外的12点环。
    inner = [(0.0, 0.0)] + regular_ring(1050.0, 7, 360.0 / 14.0)
    outer = regular_ring(2200.0, 12, 15.0)
    return [(0.0, 0.0)] + nearest_neighbor_order(inner[1:] + outer)


def optimized_visit_order(
    points: Sequence[Point], start: Point, end_hint: Optional[Point] = None
) -> List[int]:
    """最近邻初解加2-opt，优化开放路径；可兼顾路径终点后的下一覆盖航点。"""
    if not points:
        return []
    remaining = set(range(len(points)))
    order: List[int] = []
    current = start
    while remaining:
        nxt = min(
            remaining,
            key=lambda i: distance(current, points[i])
            + (0.10 * distance(points[i], end_hint) if end_hint is not None else 0.0),
        )
        order.append(nxt)
        remaining.remove(nxt)
        current = points[nxt]

    def route_cost(candidate: Sequence[int]) -> float:
        total = distance(start, points[candidate[0]])
        total += sum(
            distance(points[candidate[i]], points[candidate[i + 1]])
            for i in range(len(candidate) - 1)
        )
        if end_hint is not None:
            total += distance(points[candidate[-1]], end_hint)
        return total

    def two_opt(seed: List[int]) -> Tuple[List[int], float]:
        candidate_order = list(seed)
        candidate_cost = route_cost(candidate_order)
        improved = True
        while improved:
            improved = False
            for i in range(len(candidate_order) - 1):
                for j in range(i + 1, len(candidate_order)):
                    trial = (
                        candidate_order[:i]
                        + list(reversed(candidate_order[i:j + 1]))
                        + candidate_order[j + 1:]
                    )
                    cost = route_cost(trial)
                    if cost + 1e-6 < candidate_cost:
                        candidate_order, candidate_cost = trial, cost
                        improved = True
        return candidate_order, candidate_cost

    best_order, _ = two_opt(order)
    return best_order


def optimized_route_distance(
    points: Sequence[Point], start: Point, end_hint: Optional[Point] = None
) -> float:
    order = optimized_visit_order(points, start, end_hint)
    if not order:
        return distance(start, end_hint) if end_hint is not None else 0.0
    total = distance(start, points[order[0]])
    total += sum(
        distance(points[order[i]], points[order[i + 1]])
        for i in range(len(order) - 1)
    )
    if end_hint is not None:
        total += distance(points[order[-1]], end_hint)
    return total


def pair_candidates(observations: Sequence[Observation]) -> List[Point]:
    candidates: List[Point] = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            a, b = observations[i], observations[j]
            crossing = abs(math.sin(math.radians(angle_difference(a.bearing_deg, b.bearing_deg))))
            if crossing < 0.12:
                continue
            point = ray_intersection(a.position, a.bearing_deg, b.position, b.bearing_deg)
            if point is not None and math.hypot(*point) <= ARENA_RADIUS + 80.0:
                candidates.append(point)
    return candidates


def bearing_consistent(point: Point, observation: Observation, tolerance_deg: float = 1.08) -> bool:
    dx = point[0] - observation.position[0]
    dy = point[1] - observation.position[1]
    if math.hypot(dx, dy) < 1e-6:
        return True
    actual = normalize_angle(math.degrees(math.atan2(dy, dx)))
    return abs(angle_difference(actual, observation.bearing_deg)) <= tolerance_deg


def _clip_polygon(
    polygon: Sequence[Point], signed_inside,
    tolerance: float = 1e-8,
) -> List[Point]:
    """Sutherland-Hodgman半平面裁剪；signed_inside>=0为可行侧。"""
    if not polygon:
        return []
    output: List[Point] = []
    previous = polygon[-1]
    previous_value = float(signed_inside(previous))
    previous_inside = previous_value >= -tolerance
    for current in polygon:
        current_value = float(signed_inside(current))
        current_inside = current_value >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-15:
                ratio = previous_value / denominator
                output.append((
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                ))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    compact: List[Point] = []
    for point in output:
        if not compact or distance(point, compact[-1]) > 1e-7:
            compact.append(point)
    if len(compact) > 1 and distance(compact[0], compact[-1]) <= 1e-7:
        compact.pop()
    return compact


def _circumscribed_disk_polygon(
    center: Point, radius: float, sides: int,
) -> List[Point]:
    """返回外切正多边形，保证不因圆离散化排除真实目标。"""
    sides = max(24, int(sides))
    outer_radius = radius / math.cos(math.pi / sides)
    phase = math.pi / sides
    return [(
        center[0] + outer_radius * math.cos(phase + 2.0 * math.pi * i / sides),
        center[1] + outer_radius * math.sin(phase + 2.0 * math.pi * i / sides),
    ) for i in range(sides)]


def _clip_to_detection_disk(
    polygon: Sequence[Point], center: Point,
    radius: float = 1500.0, sides: int = 64,
) -> List[Point]:
    """用外切切线多边形施加“有信号则距离不超过1500m”的安全约束。"""
    result = list(polygon)
    for index in range(max(24, sides)):
        angle = 2.0 * math.pi * index / max(24, sides)
        nx, ny = math.cos(angle), math.sin(angle)
        result = _clip_polygon(
            result,
            lambda p, nx=nx, ny=ny: radius - (
                nx * (p[0] - center[0]) + ny * (p[1] - center[1])
            ),
        )
        if not result:
            break
    return result


def _positive_feasible_polygon(
    observations: Sequence[Observation],
) -> List[Point]:
    """角域半平面与场地/最大接收圆求交，得到包含真值的保守凸域。"""
    polygon = _circumscribed_disk_polygon((0.0, 0.0), ARENA_RADIUS, 128)
    for observation in observations:
        lower = bearing_vector(
            observation.bearing_deg - FEASIBLE_BEARING_TOLERANCE_DEG
        )
        upper = bearing_vector(
            observation.bearing_deg + FEASIBLE_BEARING_TOLERANCE_DEG
        )
        sx, sy = observation.position
        # 真值位于lower射线左侧、upper射线右侧。
        polygon = _clip_polygon(
            polygon,
            lambda p, d=lower, sx=sx, sy=sy: (
                d[0] * (p[1] - sy) - d[1] * (p[0] - sx)
            ),
        )
        polygon = _clip_polygon(
            polygon,
            lambda p, d=upper, sx=sx, sy=sy: -(
                d[0] * (p[1] - sy) - d[1] * (p[0] - sx)
            ),
        )
        if not polygon:
            break
        polygon = _clip_to_detection_disk(
            polygon, observation.position, radius=1500.0, sides=64
        )
        if not polygon:
            break
    return polygon


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2.0


def polygon_diameter(points: Sequence[Point]) -> float:
    return max((
        distance(points[i], points[j])
        for i in range(len(points)) for j in range(i + 1, len(points))
    ), default=0.0)


def point_in_convex_polygon(
    point: Point, polygon: Sequence[Point], tolerance: float = 1e-7,
) -> bool:
    """判断点是否位于逆时针凸多边形内（含边界）。"""
    if len(polygon) < 3:
        return False
    return all(
        (polygon[(index + 1) % len(polygon)][0] - polygon[index][0])
        * (point[1] - polygon[index][1])
        - (polygon[(index + 1) % len(polygon)][1] - polygon[index][1])
        * (point[0] - polygon[index][0]) >= -tolerance
        for index in range(len(polygon))
    )


def feasible_error_candidates(observations: Sequence[Observation]) -> List[Point]:
    """由每条±1度边界射线构造定位可行域候选顶点。"""
    candidates: List[Point] = []
    offsets = (-BEARING_ERROR_DEG, 0.0, BEARING_ERROR_DEG)
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            a, b = observations[i], observations[j]
            for da in offsets:
                for db in offsets:
                    point = ray_intersection(
                        a.position, a.bearing_deg + da,
                        b.position, b.bearing_deg + db,
                    )
                    if point is None or math.hypot(*point) > ARENA_RADIUS + 1e-6:
                        continue
                    if all(bearing_consistent(point, obs) for obs in observations):
                        candidates.append(point)
    # 可行域可能被目标圆边界截断；补入角域边界射线与目标圆的交点，
    # 防止仅使用两两射线交点时低估靠近圆周目标的不确定性。
    for observation in observations:
        for offset in (-BEARING_ERROR_DEG, BEARING_ERROR_DEG):
            angle = observation.bearing_deg + offset
            ux, uy = bearing_vector(angle)
            travel = ray_distance_to_arena_exit(observation.position, angle)
            point = (
                observation.position[0] + travel * ux,
                observation.position[1] + travel * uy,
            )
            if all(bearing_consistent(point, obs) for obs in observations):
                candidates.append(point)
    return candidates


def circle_from_two(a: Point, b: Point) -> Tuple[Point, float]:
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return center, distance(center, a)


def circle_from_three(a: Point, b: Point, c: Point) -> Optional[Tuple[Point, float]]:
    d = 2.0 * (
        a[0] * (b[1] - c[1])
        + b[0] * (c[1] - a[1])
        + c[0] * (a[1] - b[1])
    )
    if abs(d) < 1e-9:
        return None
    ux = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    uy = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
    ) / d
    center = (ux, uy)
    return center, distance(center, a)


def smallest_enclosing_circle(points: Sequence[Point]) -> Tuple[Optional[Point], float]:
    """小规模确定性增量算法，返回候选可行域的近似最小覆盖圆。"""
    if not points:
        return None, math.inf
    circle: Tuple[Point, float] = (points[0], 0.0)

    def outside(point: Point, candidate: Tuple[Point, float]) -> bool:
        return distance(point, candidate[0]) > candidate[1] + 1e-7

    for i, p in enumerate(points):
        if not outside(p, circle):
            continue
        circle = (p, 0.0)
        for j in range(i):
            q = points[j]
            if not outside(q, circle):
                continue
            circle = circle_from_two(p, q)
            for k in range(j):
                r = points[k]
                if not outside(r, circle):
                    continue
                triple = circle_from_three(p, q, r)
                if triple is not None:
                    circle = triple
                else:
                    pair_circles = [
                        circle_from_two(p, q),
                        circle_from_two(p, r),
                        circle_from_two(q, r),
                    ]
                    valid = [
                        candidate for candidate in pair_circles
                        if all(not outside(x, candidate) for x in (p, q, r))
                    ]
                    if valid:
                        circle = min(valid, key=lambda candidate: candidate[1])
    return circle


def _observation_key(
    observations: Sequence[Observation],
) -> Tuple[Tuple[float, float, float], ...]:
    return tuple((
        round(obs.position[0], 6),
        round(obs.position[1], 6),
        round(obs.bearing_deg, 4),
    ) for obs in observations)


@lru_cache(maxsize=8192)
def _localization_geometry_cached(
    key: Tuple[Tuple[float, float, float], ...],
) -> FeasibleRegionSummary:
    observations = [
        Observation((x, y), bearing, 0.0) for x, y, bearing in key
    ]
    polygon = _positive_feasible_polygon(observations)
    if polygon:
        center, radius = smallest_enclosing_circle(polygon)
        diameter = polygon_diameter(polygon)
        return FeasibleRegionSummary(
            vertices=tuple(polygon), center=center, radius=radius,
            diameter=diameter, area=polygon_area(polygon),
            jung_certified=(diameter <= JUNG_CLEAR_DIAMETER_M + 1e-8),
        )
    return FeasibleRegionSummary(
        vertices=(), center=None, radius=math.inf,
        diameter=math.inf, area=math.inf, jung_certified=False,
    )


def localization_geometry(
    observations: Sequence[Observation],
) -> FeasibleRegionSummary:
    """构造完整角域可行域，并返回MEC/Jung清除判据所需量。"""
    if not observations:
        return FeasibleRegionSummary(
            vertices=(), center=None, radius=math.inf,
            diameter=math.inf, area=math.inf, jung_certified=False,
        )
    return _localization_geometry_cached(_observation_key(observations))


def guaranteed_clearance(
    observations: Sequence[Observation],
) -> Tuple[bool, Optional[Point], str]:
    """用Jung充分条件或最小覆盖圆给出保守的20m保证清除判定。"""
    if len(observations) < 2:
        return False, None, "insufficient_observations"
    geometry = localization_geometry(observations)
    if geometry.center is None:
        return False, None, "empty_or_degenerate"
    if geometry.jung_certified:
        return True, geometry.center, "jung"
    if geometry.radius <= CLEAR_RADIUS - CLEARANCE_NUMERIC_MARGIN_M:
        return True, geometry.center, "mec"
    return False, geometry.center, "not_ready"


def estimate_target(observations: Sequence[Observation]) -> Tuple[Optional[Point], float]:
    """优先返回完整角域可行域的最小覆盖圆中心与半径。"""
    geometry = localization_geometry(observations)
    if geometry.center is not None and math.isfinite(geometry.radius):
        return geometry.center, geometry.radius

    # 极少数舍入矛盾案例沿用旧边界交点算法作为第一层安全回退。
    feasible = feasible_error_candidates(observations)
    if feasible:
        center, radius = smallest_enclosing_circle(feasible)
        if center is not None:
            return center, radius

    # 可行域因读数舍入出现空集时，退回稳健名义交点。
    candidates = pair_candidates(observations)
    if candidates:
        estimate = (
            statistics.median(p[0] for p in candidates),
            statistics.median(p[1] for p in candidates),
        )
        spread = max((distance(estimate, p) for p in candidates), default=0.0)
        mean_range = statistics.mean(
            distance(estimate, obs.position) for obs in observations
        )
        angular_error = mean_range * math.tan(math.radians(BEARING_ERROR_DEG))
        return estimate, spread + angular_error

    # 两条线没有可靠正向交点时，用最小二乘直线交会作为退化解。
    if len(observations) < 2:
        return None, math.inf
    a11 = a12 = a22 = b1 = b2 = 0.0
    for obs in observations:
        theta = math.radians(obs.bearing_deg)
        nx, ny = -math.sin(theta), math.cos(theta)
        rhs = nx * obs.position[0] + ny * obs.position[1]
        a11 += nx * nx
        a12 += nx * ny
        a22 += ny * ny
        b1 += nx * rhs
        b2 += ny * rhs
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-6:
        return None, math.inf
    estimate = ((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det)
    if math.hypot(*estimate) > ARENA_RADIUS + 100.0:
        return None, math.inf
    condition_penalty = min(200.0, 12.0 / math.sqrt(abs(det)))
    return estimate, condition_penalty


def ray_distance_to_arena_exit(position: Point, bearing_deg: float) -> float:
    """从检测点沿示向射线到目标圆边界的正向距离。"""
    ux, uy = bearing_vector(bearing_deg)
    projection = position[0] * ux + position[1] * uy
    discriminant = projection * projection + ARENA_RADIUS ** 2 - (
        position[0] ** 2 + position[1] ** 2
    )
    if discriminant <= 0:
        return MIN_RECEIVE_RADIUS
    return max(5.0, -projection + math.sqrt(discriminant))


def information_gain(old_uncertainty: float, new_uncertainty: float) -> float:
    """用定位区域面积比的对数表示信息增益，避免尺度过大。"""
    old_radius = 1500.0 if not math.isfinite(old_uncertainty) else max(2.0, old_uncertainty)
    new_radius = 1500.0 if not math.isfinite(new_uncertainty) else max(2.0, new_uncertainty)
    return max(-3.0, min(8.0, 2.0 * math.log(old_radius / new_radius)))


class SearchAgent:
    def __init__(
        self,
        client: SimulatorClient,
        policy: QLearningPolicy,
        problem: int,
        max_probes: int = 4,
        batch_size: Optional[int] = None,
        adaptive_coverage: bool = False,
        shared_probe: bool = False,
        shared_probe_limit: int = 2,
        joint_probe_variants: bool = False,
        optimize_coverage_endpoint: bool = False,
        directed_plan_order: bool = True,
        immediate_refine: bool = True,
        information_gated_coverage: bool = True,
        joint_coverage_localization: bool = True,
        joint_coverage_warmup: int = 1,
        information_rate_planning: bool = True,
        cost_aware_refine: bool = True,
        probabilistic_exit_planning: bool = True,
        negative_range_inference: bool = True,
        negative_range_min_samples: int = 1,
        coverage_information_rate_threshold: float = 0.08,
        enhanced_route_search: bool = False,
        route_variant_min_saving_m: float = 120.0,
        probe_lateral_min_m: float = 120.0,
        probe_lateral_max_m: float = 220.0,
        probe_lateral_scale: float = 0.16,
        probe_forward_scale: float = 0.56,
        probe_forward_max_m: float = 820.0,
        route_tail_penalty: float = 0.30,
        rigo_route_credit_m: float = 18.0,
        expanded_rigo_candidates: bool = True,
        certified_optical_fallback: bool = True,
    ):
        self.client = client
        self.policy = policy
        self.problem = problem
        self.channels = {c: ChannelState(c) for c in range(1, 21)}
        self.scan_waypoints = coverage_waypoints(problem)
        self.max_probes = max_probes
        # Q2主动测量作为Q3铺垫：问题3默认先走完整公共观测路线，再集中清除，
        # 避免每3/4个点就横跨圆域执行一次目标巡回。
        self.batch_size = batch_size or (
            len(self.scan_waypoints) if problem == 3 else 7
        )
        self.used_probe_actions: Dict[int, set] = {c: set() for c in range(1, 21)}
        self.adaptive_coverage = adaptive_coverage
        self.shared_probe = shared_probe
        self.shared_probe_limit = max(
            0, int(shared_probe_limit)
        )
        self.joint_probe_variants = joint_probe_variants
        self.optimize_coverage_endpoint = optimize_coverage_endpoint
        self.directed_plan_order = directed_plan_order
        self.immediate_refine = immediate_refine
        self.information_gated_coverage = information_gated_coverage
        self.joint_coverage_localization = joint_coverage_localization
        self.joint_coverage_warmup = max(1, joint_coverage_warmup)
        self.information_rate_planning = information_rate_planning
        self.cost_aware_refine = cost_aware_refine
        # 现有问题4回放仅2组，概率出口在其中1组反而增加86秒；问题3的
        # 12组证据更充分，因此先只在问题3启用，避免为追求平均值牺牲稳健性。
        self.probabilistic_exit_planning = (
            probabilistic_exit_planning and problem == 3
        )
        self.negative_range_inference = negative_range_inference
        self.negative_range_min_samples = max(1, negative_range_min_samples)
        self.coverage_information_rate_threshold = max(
            0.0, coverage_information_rate_threshold
        )
        self.enhanced_route_search = enhanced_route_search
        self.route_variant_min_saving_m = max(0.0, route_variant_min_saving_m)
        self.probe_lateral_min_m = max(70.0, probe_lateral_min_m)
        self.probe_lateral_max_m = max(
            self.probe_lateral_min_m, probe_lateral_max_m
        )
        self.probe_lateral_scale = max(0.05, probe_lateral_scale)
        self.probe_forward_scale = max(0.30, probe_forward_scale)
        self.probe_forward_max_m = max(500.0, probe_forward_max_m)
        self.route_tail_penalty = max(0.0, route_tail_penalty)
        self.rigo_route_credit_m = max(0.0, rigo_route_credit_m)
        self.expanded_rigo_candidates = expanded_rigo_candidates
        self.certified_optical_fallback = certified_optical_fallback
        self._planning_hypothesis_cache: Dict[tuple, list] = {}
        self.coverage_ring_adapted = False
        self.coverage_ring_phase_deg = 30.0 if problem == 3 else None
        self.coverage_cells = [
            (float(x), float(y))
            for x in range(-1800, 1801, 120)
            for y in range(-1800, 1801, 120)
            if x * x + y * y <= ARENA_RADIUS ** 2
        ]
        self.covered_cells: set = set()
        self.sweep_positions: List[Point] = []

    def add_direction(self, state: ChannelState, position: Point, result: dict) -> None:
        state.observations.append(Observation(
            position=position,
            bearing_deg=float(result["svd_deg"]),
            virtual_time_s=float(result.get("virtual_time_s", self.client.virtual_time_s)),
        ))

    def measure_channel(self, point: Point, channel: int) -> Optional[dict]:
        """统一记录无信号位置；负观测是Q2距离推断的重要约束。"""
        result = self.client.measure(point, channel)
        if result and result.get("measure_result") == "no_signal":
            negatives = self.channels[channel].no_signal_positions
            if not any(distance(point, old) < 1.0 for old in negatives):
                negatives.append(point)
        return result

    def first_ray_range_posterior(
        self, state: ChannelState
    ) -> Tuple[Point, float]:
        """由direction/no_signal共同估计目标在首条射线上的位置与距离标准差。

        同一信号源的真实可测半径固定但未知，范围为1000~1500米。候选点
        必须允许一个半径R，使全部direction点在R内、no_signal点在R外；
        可行R区间的宽度作为该候选的似然权重。
        """
        first = state.observations[0]
        upper = min(
            1500.0,
            ray_distance_to_arena_exit(first.position, first.bearing_deg),
        )
        ux, uy = bearing_vector(first.bearing_deg)
        fallback_t = 0.62 * upper
        if (
            not self.negative_range_inference
            or len(state.no_signal_positions) < self.negative_range_min_samples
        ):
            return (
                first.position[0] + fallback_t * ux,
                first.position[1] + fallback_t * uy,
            ), 0.28 * upper

        weighted: List[Tuple[float, float]] = []
        sample_count = max(60, int(upper / 12.0))
        for index in range(1, sample_count + 1):
            t = upper * index / sample_count
            point = (
                first.position[0] + t * ux,
                first.position[1] + t * uy,
            )
            positive_radius = max(
                MIN_RECEIVE_RADIUS,
                max(distance(point, obs.position) for obs in state.observations),
            )
            negative_radius = min(
                1500.0,
                min(distance(point, position) for position in state.no_signal_positions),
            )
            width = negative_radius - positive_radius
            if width > 0.0:
                # 小的基线权重防止可行区间在舍入误差下过度尖锐。
                weighted.append((t, width + 8.0))
        if not weighted:
            return (
                first.position[0] + fallback_t * ux,
                first.position[1] + fallback_t * uy,
            ), 0.32 * upper
        total_weight = sum(weight for _, weight in weighted)
        mean_t = sum(t * weight for t, weight in weighted) / total_weight
        variance = sum(
            weight * (t - mean_t) ** 2 for t, weight in weighted
        ) / total_weight
        return (
            first.position[0] + mean_t * ux,
            first.position[1] + mean_t * uy,
        ), math.sqrt(max(0.0, variance))

    def negative_feasible_hypotheses(
        self, state: ChannelState,
    ) -> List[Tuple[Point, float, float]]:
        """离散表示正角域减去无信号排除圆后的非凸规划可行域。

        每个假设同时满足：全部direction测点到源的距离不超过某个固定R，
        全部no_signal测点到源的距离大于同一个R，且1000<=R<=1500。
        该集合只用于主动测点和路径评分；最终清除仍由保守MEC/Jung判据把关。
        """
        if not state.observations:
            return []
        key = (
            state.channel,
            _observation_key(state.observations),
            tuple((round(p[0], 3), round(p[1], 3))
                  for p in state.no_signal_positions),
        )
        cached = self._planning_hypothesis_cache.get(key)
        if cached is not None:
            return cached
        polygon = localization_geometry(state.observations).vertices
        if not polygon:
            self._planning_hypothesis_cache[key] = []
            return []
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_y = min(point[1] for point in polygon)
        max_y = max(point[1] for point in polygon)
        samples: List[Point] = list(polygon)
        samples.extend((
            (polygon[index][0] + polygon[(index + 1) % len(polygon)][0]) / 2.0,
            (polygon[index][1] + polygon[(index + 1) % len(polygon)][1]) / 2.0,
        ) for index in range(len(polygon)))
        grid_size = 11
        for ix in range(grid_size):
            x = min_x + (ix + 0.5) * (max_x - min_x) / grid_size
            for iy in range(grid_size):
                y = min_y + (iy + 0.5) * (max_y - min_y) / grid_size
                point = (x, y)
                if point_in_convex_polygon(point, polygon):
                    samples.append(point)

        hypotheses: List[Tuple[Point, float, float]] = []
        for point in samples:
            lower = max(
                MIN_RECEIVE_RADIUS,
                max(distance(point, obs.position) for obs in state.observations),
            )
            upper = 1500.0
            if state.no_signal_positions:
                upper = min(
                    upper,
                    min(distance(point, p) for p in state.no_signal_positions),
                )
            if lower <= upper + 1e-6:
                width = max(2.0, upper - lower)
                hypotheses.append((point, width, (lower + upper) / 2.0))
        # 舍入或粗网格导致空集时，规划退回正观测几何，不影响清除安全。
        self._planning_hypothesis_cache[key] = hypotheses
        if len(self._planning_hypothesis_cache) > 4096:
            self._planning_hypothesis_cache.clear()
        return hypotheses

    def planning_target(self, state: ChannelState) -> Tuple[Optional[Point], float]:
        """返回负观测修正后的规划中心和离散后验离散度。"""
        hypotheses = self.negative_feasible_hypotheses(state)
        if not hypotheses:
            return estimate_target(state.observations)
        total = sum(weight for _, weight, _ in hypotheses)
        center = (
            sum(point[0] * weight for point, weight, _ in hypotheses) / total,
            sum(point[1] * weight for point, weight, _ in hypotheses) / total,
        )
        sigma = math.sqrt(sum(
            weight * distance(point, center) ** 2
            for point, weight, _ in hypotheses
        ) / total)
        return center, sigma

    def rigo_expected_gain(self, state: ChannelState, point: Point) -> float:
        """估计候选点的负观测分裂、测向交会与直接near综合信息增益。"""
        hypotheses = self.negative_feasible_hypotheses(state)
        if not hypotheses:
            return 0.0
        total = sum(weight for _, weight, _ in hypotheses)
        signal_weight = near_weight = directional_gain = 0.0
        for target, weight, receive_radius in hypotheses:
            probe_range = distance(point, target)
            if probe_range <= CLEAR_RADIUS:
                near_weight += weight
            if probe_range > receive_radius:
                continue
            signal_weight += weight
            predicted = normalize_angle(math.degrees(math.atan2(
                target[1] - point[1], target[0] - point[0]
            )))
            crossing = max((
                abs(math.sin(math.radians(angle_difference(
                    observation.bearing_deg, predicted
                )))) for observation in state.observations
            ), default=0.0)
            metric_error = max(
                2.0,
                probe_range * math.tan(math.radians(
                    FEASIBLE_BEARING_TOLERANCE_DEG
                )),
            )
            directional_gain += weight * crossing * math.log1p(
                1500.0 / metric_error
            )
        probability = max(1e-9, min(1.0 - 1e-9, signal_weight / total))
        binary_gain = -(
            probability * math.log(probability)
            + (1.0 - probability) * math.log(1.0 - probability)
        )
        return (
            binary_gain
            + directional_gain / total
            + 4.0 * near_weight / total
        )

    def rigo_information_rate(
        self, state: ChannelState, point: Point, already_at_point: bool = False,
    ) -> float:
        travel_s = 0.0 if already_at_point else (
            distance(self.client.position, point) / 5.0
        )
        return self.rigo_expected_gain(state, point) / max(6.0, travel_s + 6.0)

    def clear_at(self, state: ChannelState, position: Point) -> bool:
        result = self.client.clear(position, state.channel)
        if not result:
            return False
        if result.get("clear_result") == "success":
            state.cleared = True
            state.clear_position = position
            print(f"频道{state.channel}清除成功，位置≈({position[0]:.1f},{position[1]:.1f})")
            return True
        state.failed_clear_positions.append(position)
        return False

    def recover_clear_locally(self, state: ChannelState) -> bool:
        """清除未命中后在当前邻域修正，避免把失败目标留到末尾跨场补救。"""
        current = self.client.position
        result = self.measure_channel(current, state.channel)
        if result and result.get("measure_result") == "near":
            return self.clear_at(state, current)
        if result and result.get("measure_result") == "direction":
            self.add_direction(state, current, result)
            ready, refined, _ = guaranteed_clearance(state.observations)
            if ready and refined is not None:
                return self.clear_at(state, refined)

        estimate, _ = estimate_target(state.observations)
        if estimate is None:
            return False
        alternatives = sorted(
            pair_candidates(state.observations), key=lambda p: distance(p, estimate)
        )
        # 只尝试局部候选，防止恢复动作本身演变成新的跨场巡回。
        for point in alternatives:
            if distance(point, current) > 80.0 or distance(point, estimate) < 4.0:
                continue
            if self.clear_at(state, point):
                return True
        return False

    def probe_candidates(
        self, first: Observation, attempt: int,
        state: Optional[ChannelState] = None,
    ) -> Dict[str, Point]:
        ux, uy = bearing_vector(first.bearing_deg)
        px, py = -uy, ux
        # Q2主动测量：由射线与目标圆的交段估计距离尺度。测点既靠近潜在目标，
        # 又保持足够横向基线，使一次补测尽量同时完成“定位铺垫”和“接近清除”。
        upper = min(1500.0, ray_distance_to_arena_exit(first.position, first.bearing_deg))
        if (
            state is not None and self.negative_range_inference
            and len(state.no_signal_positions) >= self.negative_range_min_samples
        ):
            predicted, range_sigma = self.first_ray_range_posterior(state)
            predicted_range = max(80.0, distance(first.position, predicted))
            # 尽量在预计目标旁形成近似正交交会；横向偏置随距离后验误差增加，
            # 既避免共线，又把Q2→Q3的最后一段压到约160~300米。
            forward = min(upper, predicted_range)
            lateral = min(
                self.probe_lateral_max_m,
                max(self.probe_lateral_min_m, 0.70 * range_sigma),
            )
        else:
            forward = min(
                self.probe_forward_max_m,
                max(320.0, self.probe_forward_scale * upper),
            )
            lateral = min(
                self.probe_lateral_max_m,
                max(self.probe_lateral_min_m, self.probe_lateral_scale * upper),
            )
        if attempt >= 2:
            forward *= 0.90
            lateral *= 1.18
        return {
            "probe_left": (
                first.position[0] + forward * ux + lateral * px,
                first.position[1] + forward * uy + lateral * py,
            ),
            "probe_right": (
                first.position[0] + forward * ux - lateral * px,
                first.position[1] + forward * uy - lateral * py,
            ),
            "probe_forward": (
                first.position[0] + min(upper, forward) * ux,
                first.position[1] + min(upper, forward) * uy,
            ),
        }

    def localize_and_clear(self, state: ChannelState) -> bool:
        if state.cleared:
            return True
        if not state.observations:
            return False

        used_actions = set()
        last_signal = True
        for attempt in range(self.max_probes):
            estimate, uncertainty = estimate_target(state.observations)
            ready, guaranteed_point, _ = guaranteed_clearance(state.observations)
            if ready and guaranteed_point is not None:
                estimate = guaranteed_point
                break

            first = state.observations[0]
            candidates = self.probe_candidates(first, attempt, state)
            actions = [a for a in candidates if a not in used_actions]
            if not actions:
                actions = list(candidates)

            state_key = self.policy.state_key(
                self.problem, len(state.observations), uncertainty, last_signal
            )
            # 交替左右测点可提高交会角；问题4略偏好forward以减少跨越定向边界。
            heuristic = {
                "probe_left": 1.2 if attempt % 2 == 0 else 0.3,
                "probe_right": 1.2 if attempt % 2 == 1 else 0.3,
                "probe_forward": 0.9 if self.problem == 4 else -0.4,
            }
            action = self.policy.choose(state_key, actions, heuristic)
            used_actions.add(action)
            point = candidates[action]
            old_time = self.client.virtual_time_s
            old_uncertainty = uncertainty
            result = self.measure_channel(point, state.channel)
            if not result:
                return False

            measurement = result.get("measure_result")
            reward = -0.025 * (self.client.virtual_time_s - old_time)
            if measurement == "near":
                reward += 180.0
                success = self.clear_at(state, point)
                reward += 500.0 if success else -80.0
                next_key = self.policy.state_key(self.problem, len(state.observations), 0.0, True)
                self.policy.update(state_key, action, reward, next_key)
                return success
            if measurement == "direction":
                self.add_direction(state, point, result)
                last_signal = True
                new_estimate, new_uncertainty = estimate_target(state.observations)
                reward += 35.0
                if math.isfinite(old_uncertainty) and math.isfinite(new_uncertainty):
                    reward += max(-20.0, min(80.0, old_uncertainty - new_uncertainty))
                if new_estimate is not None:
                    reward += 10.0
            else:
                last_signal = False
                reward -= 28.0

            _, new_uncertainty = estimate_target(state.observations)
            next_key = self.policy.state_key(
                self.problem, len(state.observations), new_uncertainty, last_signal
            )
            self.policy.update(state_key, action, reward, next_key)

        estimate, uncertainty = estimate_target(state.observations)
        if estimate is None:
            return False

        success = self.clear_at(state, estimate)
        if success:
            return True

        # 清除失败后，优先在各两两交点尝试；它们代表±误差下可能的不同中心。
        alternatives = sorted(
            pair_candidates(state.observations), key=lambda p: distance(p, estimate)
        )
        for point in alternatives[:3]:
            if distance(point, estimate) < 8.0:
                continue
            if self.clear_at(state, point):
                return True

        # 最后在估计点进行一次测向。near可直接清除，direction则加入约束后重估。
        result = self.measure_channel(estimate, state.channel)
        if result and result.get("measure_result") == "near":
            return self.clear_at(state, estimate)
        if result and result.get("measure_result") == "direction":
            self.add_direction(state, estimate, result)
            refined, _ = estimate_target(state.observations)
            if refined is not None:
                return self.clear_at(state, refined)
        if self.certified_optical_fallback and self.problem == 3:
            return self.certified_optical_clear(state)
        return False

    def certified_optical_clear(self, state: ChannelState) -> bool:
        """有界光学兜底：仅在常规几何定位和局部恢复全部失败后触发。

        沿首次正观测角域建立两条蛇形车道，径向步长30 m，横向
        偏置为r*tan(epsilon)/2。它只是100%清除的终止兜底，不改变
        已经成功的正常样本路线。
        """
        if state.cleared or not state.observations:
            return state.cleared
        first = state.observations[0]
        ux, uy = bearing_vector(first.bearing_deg)
        px, py = -uy, ux
        radii = [
            OPTICAL_FALLBACK_RADIAL_STEP_M / 2.0
            + index * OPTICAL_FALLBACK_RADIAL_STEP_M
            for index in range(int(
                OPTICAL_FALLBACK_MAX_RANGE_M / OPTICAL_FALLBACK_RADIAL_STEP_M
            ))
        ]
        previous_phase = self.client.phase
        self.client.phase = "certified_optical_fallback"
        try:
            for sign, sequence in (
                (1.0, radii), (-1.0, list(reversed(radii)))
            ):
                for radius in sequence:
                    lateral = (
                        sign * radius
                        * math.tan(math.radians(FEASIBLE_BEARING_TOLERANCE_DEG))
                        / 2.0
                    )
                    point = (
                        first.position[0] + radius * ux + lateral * px,
                        first.position[1] + radius * uy + lateral * py,
                    )
                    if self.clear_at(state, point):
                        return True
        finally:
            self.client.phase = previous_phase
        return False

    @property
    def discovered_count(self) -> int:
        return sum(bool(state.observations) or state.cleared for state in self.channels.values())

    def coverage_gain_cells(self, point: Point) -> List[int]:
        # 910 + 120/sqrt(2) < 1000，网格全覆盖可保守代表连续圆域覆盖。
        return [
            index for index, cell in enumerate(self.coverage_cells)
            if index not in self.covered_cells and distance(point, cell) <= 910.0
        ]

    def mark_sweep(self, point: Point) -> int:
        gained = self.coverage_gain_cells(point)
        self.covered_cells.update(gained)
        self.sweep_positions.append(point)
        return len(gained)

    @property
    def coverage_ratio(self) -> float:
        return len(self.covered_cells) / max(1, len(self.coverage_cells))

    def expected_information_at(self, point: Point) -> float:
        value = 0.0
        for state in self.channels.values():
            if state.cleared or not state.observations or self.localization_ready(state):
                continue
            # 归一到旧交会角评分的约0~1尺度，避免信息项压过覆盖与路程。
            value += min(1.0, self.rigo_expected_gain(state, point) / 4.0)
        return value

    def select_coverage_waypoint(self, remaining: Sequence[Point]) -> Point:
        unresolved = sum(
            not state.cleared and not self.localization_ready(state)
            for state in self.channels.values()
        )
        best_point = remaining[0]
        best_score = -math.inf
        for point in remaining:
            new_cells = len(self.coverage_gain_cells(point))
            info = self.expected_information_at(point)
            time_cost = distance(self.client.position, point) / 5.0 + 5.0 * unresolved
            # 显式奖励：覆盖新区域、改善交会角；惩罚真实虚拟时间。
            score = 2.8 * new_cells + 95.0 * info - 0.55 * time_cost
            if score > best_score:
                best_point, best_score = point, score
        return best_point

    def optimized_ring_order(self, ring: Sequence[Point]) -> List[Point]:
        """保持覆盖环路长度不变，选择最利于衔接后续Q2巡回的终点。"""
        hints = []
        for state in self.channels.values():
            if state.cleared or not state.observations:
                continue
            estimate, _ = self.first_ray_range_posterior(state)
            hints.append(estimate)
        if not hints or len(ring) < 3:
            return list(ring)
        candidates = []
        count = len(ring)
        for start in range(count):
            for direction in (-1, 1):
                order = [ring[(start + direction * step) % count] for step in range(count)]
                # 各候选覆盖环本身等长，比较其终点接入预计目标巡回的代价。
                score = optimized_route_distance(hints, order[-1], None)
                candidates.append((score, order))
        return min(candidates, key=lambda item: item[0])[1]

    def adapt_problem3_coverage_ring(self, remaining: Sequence[Point]) -> List[Point]:
        """中心首测后旋转六边形，使公共覆盖点兼作高信息Q2测点。"""
        if (
            self.problem != 3 or self.coverage_ring_adapted
            or len(remaining) != 6
        ):
            return list(remaining)
        active = [
            state for state in self.channels.values()
            if state.observations and not state.cleared
        ]
        self.coverage_ring_adapted = True
        if not active:
            return list(remaining)
        best_phase = 30.0
        best_score = -math.inf
        best_ring = list(remaining)
        # 任意旋转都保持六边形连续覆盖证明成立；仅用信息增益改变朝向。
        for phase in range(0, 60, 5):
            ring = regular_ring(P3_HEX_COVERAGE_RADIUS_M, 6, float(phase))
            score = sum(
                max(self.rigo_expected_gain(state, point) for point in ring)
                for state in active
            )
            # 同分时偏向首段更短的开放访问顺序。
            score -= 0.0002 * optimized_route_distance(
                ring, self.client.position, None
            )
            if score > best_score + 1e-9:
                best_score, best_phase, best_ring = score, float(phase), ring
        self.coverage_ring_phase_deg = best_phase
        return best_ring

    def opportunistic_sweep(self, primary_channel: int) -> Tuple[int, int, float, float]:
        """在Q2/清除落点顺便搜索其他频道，让目标访问点替代固定覆盖点。"""
        point = self.client.position
        gain_preview = self.coverage_gain_cells(point)
        threshold = max(32, int(0.045 * len(self.coverage_cells)))
        if len(gain_preview) < threshold or self.coverage_ratio >= 1.0:
            return 0, 0, 0.0, 0.0
        before_time = self.client.virtual_time_s
        before_discovered = self.discovered_count
        before_uncertainty = {
            channel: estimate_target(state.observations)[1]
            for channel, state in self.channels.items()
            if state.observations and not state.cleared
        }
        previous_phase = self.client.phase
        self.client.phase = "adaptive_sweep"
        for channel, state in self.channels.items():
            if channel == primary_channel or state.cleared or self.localization_ready(state):
                continue
            result = self.measure_channel(point, channel)
            if not result:
                continue
            if result.get("measure_result") == "near":
                self.clear_at(state, point)
            elif result.get("measure_result") == "direction":
                self.add_direction(state, point, result)
        self.client.phase = previous_phase
        new_cells = self.mark_sweep(point)
        new_channels = self.discovered_count - before_discovered
        gain = 0.0
        for channel, old_uncertainty in before_uncertainty.items():
            new_uncertainty = estimate_target(self.channels[channel].observations)[1]
            gain += information_gain(old_uncertainty, new_uncertainty)
        elapsed = self.client.virtual_time_s - before_time
        return new_channels, new_cells, gain, elapsed

    def shared_probe_sweep(self, primary_channel: int) -> Tuple[int, float, float]:
        """在一个目标的Q2测点，顺便为其他目标取得高价值交叉示向线。"""
        point = self.client.position
        ranked = []
        for channel, state in self.channels.items():
            if (
                channel == primary_channel or state.cleared
                or not state.observations or self.localization_ready(state)
            ):
                continue
            first = state.observations[0]
            estimate, old_uncertainty = estimate_target(state.observations)
            if estimate is None:
                estimate, _ = self.first_ray_range_posterior(state)
            predicted = normalize_angle(math.degrees(math.atan2(
                estimate[1] - point[1], estimate[0] - point[0]
            )))
            crossing = abs(math.sin(math.radians(
                angle_difference(first.bearing_deg, predicted)
            )))
            predicted_range = distance(point, estimate)
            # 只共享高交会角、且当前点预计位于接收圆内部的频道。
            # 低价值顺带测量会累积固定6秒成本，v8主动将门槛收紧。
            range_score = max(0.0, 1.0 - predicted_range / 1450.0)
            score = 2.4 * crossing + 1.6 * range_score
            rate = self.rigo_information_rate(
                state, point, already_at_point=True
            )
            if (
                crossing >= 0.52
                and predicted_range <= 1380.0
                and rate >= 0.075
            ):
                ranked.append((score, channel, old_uncertainty))

        before_time = self.client.virtual_time_s
        total_gain = 0.0
        useful = 0
        previous_phase = self.client.phase
        self.client.phase = "shared_q2_measure"
        for _, channel, old_uncertainty in sorted(
            ranked, reverse=True
        )[:self.shared_probe_limit]:
            state = self.channels[channel]
            result = self.measure_channel(point, channel)
            if not result:
                continue
            if result.get("measure_result") == "near":
                useful += 1
                self.clear_at(state, point)
            elif result.get("measure_result") == "direction":
                self.add_direction(state, point, result)
                new_uncertainty = estimate_target(state.observations)[1]
                total_gain += information_gain(old_uncertainty, new_uncertainty)
                useful += 1
        self.client.phase = previous_phase
        return useful, total_gain, self.client.virtual_time_s - before_time

    def scan_one_waypoint(self, point: Point) -> List[int]:
        detected: List[int] = []
        unresolved = [
            c for c, state in self.channels.items()
            if (
                not state.cleared and not self.localization_ready(state)
                and self.coverage_measure_worthwhile(state, point)
            )
        ]
        # 连续频道扫描减少不必要的频道切换逻辑复杂度。
        for channel in unresolved:
            state = self.channels[channel]
            result = self.measure_channel(point, channel)
            if not result:
                raise RuntimeError(f"频道{channel}检测请求失败")
            measurement = result.get("measure_result")
            if measurement == "near":
                self.clear_at(state, point)
            elif measurement == "direction":
                self.add_direction(state, point, result)
                detected.append(channel)
        self.mark_sweep(point)
        return detected

    def coverage_measure_worthwhile(self, state: ChannelState, point: Point) -> bool:
        """未知频道必扫；已发现频道只保留预计能显著改善交会的免费补测。"""
        if not self.information_gated_coverage or not state.observations:
            return True
        if self.information_rate_planning:
            return self.rigo_information_rate(
                state, point, already_at_point=True
            ) >= self.coverage_information_rate_threshold
        first = state.observations[0]
        estimate, _ = estimate_target(state.observations)
        if estimate is None:
            estimate, _ = self.first_ray_range_posterior(state)
        predicted_range = distance(point, estimate)
        predicted = normalize_angle(math.degrees(math.atan2(
            estimate[1] - point[1], estimate[0] - point[0]
        )))
        crossing = abs(math.sin(math.radians(
            angle_difference(first.bearing_deg, predicted)
        )))
        return predicted_range <= 1400.0 and crossing >= 0.20

    def localization_ready(self, state: ChannelState) -> bool:
        ready, _, _ = guaranteed_clearance(state.observations)
        return ready

    def plan_probe(self, state: ChannelState, attempt: int) -> Optional[dict]:
        if state.cleared or not state.observations or self.localization_ready(state):
            return None
        estimate, uncertainty = estimate_target(state.observations)
        planning_estimate, _ = self.planning_target(state)
        first = state.observations[0]
        if estimate is not None and len(state.observations) >= 2:
            # 两线已有交点后，下一测点直接服务于接近和清除；不再回到首测点附近。
            latest = state.observations[-1]
            dx = estimate[0] - latest.position[0]
            dy = estimate[1] - latest.position[1]
            norm = max(1e-9, math.hypot(dx, dy))
            px, py = -dy / norm, dx / norm
            offset = min(140.0, max(45.0, 0.45 * uncertainty)) if math.isfinite(uncertainty) else 90.0
            candidates = {
                "probe_estimate": estimate,
                "probe_estimate_left": (
                    estimate[0] + offset * px, estimate[1] + offset * py
                ),
                "probe_estimate_right": (
                    estimate[0] - offset * px, estimate[1] - offset * py
                ),
            }
            heuristic = {
                "probe_estimate": 4.0,
                "probe_estimate_left": 0.6 if attempt % 2 == 0 else 0.2,
                "probe_estimate_right": 0.6 if attempt % 2 == 1 else 0.2,
            }
        else:
            candidates = self.probe_candidates(first, attempt, state)
            heuristic = {
                "probe_left": 1.3 if attempt % 2 == 0 else 0.2,
                "probe_right": 1.3 if attempt % 2 == 1 else 0.2,
                "probe_forward": 1.0 if self.problem == 4 else -0.5,
            }
            if self.problem == 3 and self.expanded_rigo_candidates:
                ux, uy = bearing_vector(first.bearing_deg)
                px, py = -uy, ux
                upper = min(
                    1500.0,
                    ray_distance_to_arena_exit(first.position, first.bearing_deg),
                )
                # 指导论文的多尺度候选集；仍由RIGO信息增益率、
                # 移动代价与固定接收半径的正负观测一致性共同筛选。
                for forward in (350.0, 700.0, 1050.0):
                    if forward > upper + 1e-6:
                        continue
                    for sign, side in ((1.0, "left"), (-1.0, "right")):
                        name = f"paper_{int(forward)}_{side}"
                        candidates[name] = (
                            first.position[0] + forward * ux + sign * 160.0 * px,
                            first.position[1] + forward * uy + sign * 160.0 * py,
                        )
                        heuristic[name] = 0.15
        unused = [a for a in candidates if a not in self.used_probe_actions[state.channel]]
        actions = unused or list(candidates)
        state_key = self.policy.state_key(
            self.problem, len(state.observations), uncertainty, True
        )
        expected_exit = planning_estimate or estimate
        if expected_exit is None:
            expected_exit, _ = self.first_ray_range_posterior(state)
        alternative_rigo = {
            name: self.rigo_expected_gain(state, point)
            for name, point in candidates.items()
        }
        # RIGO只调整同一目标的补测点变体，不绕过定位保证判据。
        for name in actions:
            heuristic[name] = heuristic.get(name, 0.0) + (
                60.0 * self.rigo_information_rate(state, candidates[name])
            )
        action = self.policy.choose(state_key, actions, heuristic)
        probe_point = candidates[action]
        # 单条示向线的目标位置仍有较大不确定性，按负观测修正后的出口
        # 同时估计成功进入Q3和仍停在Q2测点两种结局。
        if len(state.observations) >= 2:
            exit_probability = 0.96
        else:
            predicted_range = distance(probe_point, expected_exit)
            signal_probability = max(
                0.0, min(1.0, (1500.0 - predicted_range) / 500.0)
            )
            predicted_bearing = normalize_angle(math.degrees(math.atan2(
                expected_exit[1] - probe_point[1],
                expected_exit[0] - probe_point[0],
            )))
            crossing = abs(math.sin(math.radians(
                angle_difference(first.bearing_deg, predicted_bearing)
            )))
            geometry_probability = max(0.0, min(1.0, crossing / 0.32))
            exit_probability = signal_probability * (
                0.35 + 0.65 * geometry_probability
            )
        return {
            "state": state,
            "state_key": state_key,
            "action": action,
            "point": candidates[action],
            "old_uncertainty": uncertainty,
            "observation_count": len(state.observations),
            "expected_exit": expected_exit,
            "exit_probability": exit_probability,
            # 单条示向线时允许批处理器联合选择左/右/前向点；多线交会后仍直奔估计点。
            "alternatives": (
                {
                    name: candidates[name] for name in actions
                    if self.problem == 4 or name != "probe_forward"
                }
                if len(state.observations) == 1 else {action: candidates[action]}
            ),
            "alternative_rigo": alternative_rigo,
        }

    def optimized_plan_order(
        self, plans: Sequence[dict], end_hint: Optional[Point]
    ) -> List[int]:
        """优化入口为Q2点、出口为预计Q3清除点的有向任务巡回。"""
        if not plans:
            return []

        def expected_leg(left: dict, destination: Point) -> float:
            if not self.probabilistic_exit_planning:
                return distance(left["expected_exit"], destination)
            probability = max(0.0, min(1.0, left.get("exit_probability", 1.0)))
            return (
                probability * distance(left["expected_exit"], destination)
                + (1.0 - probability) * distance(left["point"], destination)
            )

        def route_cost(order: Sequence[int]) -> float:
            first_leg = distance(self.client.position, plans[order[0]]["point"])
            total = first_leg
            # 开放有向路径偶尔会为了较小的总期望值选择1.3~1.7km首跳，
            # 对超过850m的首段加二次风险成本，降低虚拟时间长尾与方差。
            total += self.route_tail_penalty * (
                max(0.0, first_leg - 850.0) ** 2 / 850.0
            )
            for left, right in zip(order, order[1:]):
                total += expected_leg(plans[left], plans[right]["point"])
            if end_hint is not None:
                total += expected_leg(plans[order[-1]], end_hint)
            return total

        def improve(seed: List[int]) -> Tuple[List[int], float]:
            best = list(seed)
            best_cost = route_cost(best)
            changed = True
            while changed:
                changed = False
                candidate_order = best
                candidate_cost = best_cost
                for i in range(len(best) - 1):
                    for j in range(i + 1, len(best)):
                        trial = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                        cost = route_cost(trial)
                        if cost + 1e-6 < candidate_cost:
                            candidate_order, candidate_cost = trial, cost
                if self.enhanced_route_search:
                    # 有向Q2入口/Q3出口不满足对称TSP假设；单点重插入和交换
                    # 能修复2-opt无法处理的“某个目标被夹在场地另一侧”长跳转。
                    for i in range(len(best)):
                        reduced = best[:i] + best[i + 1:]
                        for j in range(len(reduced) + 1):
                            trial = reduced[:j] + [best[i]] + reduced[j:]
                            cost = route_cost(trial)
                            if cost + 1e-6 < candidate_cost:
                                candidate_order, candidate_cost = trial, cost
                    for i in range(len(best) - 1):
                        for j in range(i + 1, len(best)):
                            trial = list(best)
                            trial[i], trial[j] = trial[j], trial[i]
                            cost = route_cost(trial)
                            if cost + 1e-6 < candidate_cost:
                                candidate_order, candidate_cost = trial, cost
                if candidate_cost + 1e-6 < best_cost:
                    best, best_cost, changed = candidate_order, candidate_cost, True
            return best, best_cost

        best_order: List[int] = []
        best_cost = math.inf
        # 有向入口/出口使“最近的第一个补测点”未必全局最优，因此枚举首任务。
        for first in range(len(plans)):
            remaining = set(range(len(plans)))
            remaining.remove(first)
            seed = [first]
            while remaining:
                nxt = min(
                    remaining,
                    key=lambda i: expected_leg(plans[seed[-1]], plans[i]["point"]),
                )
                seed.append(nxt)
                remaining.remove(nxt)
            order, cost = improve(seed)
            if cost + 1e-6 < best_cost:
                best_order, best_cost = order, cost
        return best_order

    def optimize_probe_variants(
        self, plans: List[dict], end_hint: Optional[Point]
    ) -> None:
        """坐标下降求解小规模广义TSP：同时选补测点变体与访问顺序。"""
        if len(plans) < 2:
            return
        points = [plan["point"] for plan in plans]
        best_cost = optimized_route_distance(points, self.client.position, end_hint)
        for _ in range(3):
            improved = False
            for index, plan in enumerate(plans):
                chosen_action = plan["action"]
                chosen_point = plan["point"]
                local_cost = best_cost
                for action, point in plan["alternatives"].items():
                    trial = list(points)
                    trial[index] = point
                    cost = optimized_route_distance(
                        trial, self.client.position, end_hint
                    )
                    if cost + 1e-6 < local_cost:
                        chosen_action, chosen_point, local_cost = action, point, cost
                if chosen_point != points[index]:
                    points[index] = chosen_point
                    plan["action"] = chosen_action
                    plan["point"] = chosen_point
                    best_cost = local_cost
                    improved = True
            if not improved:
                break

    def optimize_joint_probe_variants(
        self, nodes: List[dict], end_hint: Optional[Point]
    ) -> None:
        """让Q2左/右基线与当前联合路线对齐，减少横向偏置的累计绕行。"""
        if not self.joint_probe_variants:
            return
        for _ in range(3):
            order = self.optimized_plan_order(nodes, end_hint)
            positions = {node_index: rank for rank, node_index in enumerate(order)}
            changed = False
            for index, node in enumerate(nodes):
                alternatives = node.get("alternatives")
                if node.get("kind") != "probe" or not alternatives:
                    continue
                rank = positions[index]
                previous = (
                    self.client.position if rank == 0
                    else nodes[order[rank - 1]]["expected_exit"]
                )
                following = (
                    nodes[order[rank + 1]]["point"]
                    if rank + 1 < len(order) else end_hint
                )
                probability = max(
                    0.0, min(1.0, node.get("exit_probability", 1.0))
                )

                def local_cost(action: str, point: Point) -> float:
                    value = distance(previous, point)
                    if following is not None:
                        value += (
                            probability * distance(node["expected_exit"], following)
                            + (1.0 - probability) * distance(point, following)
                        )
                    information = node.get("alternative_rigo", {}).get(
                        action, 0.0
                    )
                    return value - self.rigo_route_credit_m * information

                current_cost = local_cost(node["action"], node["point"])
                action, point = min(
                    alternatives.items(),
                    key=lambda item: local_cost(item[0], item[1]),
                )
                if (
                    action != node["action"]
                    and local_cost(action, point) + self.route_variant_min_saving_m
                    < current_cost
                ):
                    node["action"] = action
                    node["point"] = point
                    changed = True
            if not changed:
                break

    def should_refine_now(self, state: ChannelState, estimate: Point, plan: dict) -> bool:
        """比较立即补测插入成本与未来公共点的预期信息收益。"""
        if not self.cost_aware_refine:
            return True
        next_point = plan.get("next_route_point")
        future_coverage = plan.get("future_coverage_points") or []
        if next_point is None or not future_coverage:
            return True
        current = self.client.position
        insertion_time = (
            distance(current, estimate)
            + distance(estimate, next_point)
            - distance(current, next_point)
        ) / 5.0 + 10.0
        # 估计点很近时立即完成总是更稳定。
        if distance(current, estimate) <= 260.0 or insertion_time <= 55.0:
            return True

        first = state.observations[0]
        best_future_rate = 0.0
        for point in future_coverage:
            predicted_range = distance(point, estimate)
            signal_probability = max(
                0.0, min(1.0, (1500.0 - predicted_range) / 500.0)
            )
            if signal_probability <= 0.0:
                continue
            predicted = normalize_angle(math.degrees(math.atan2(
                estimate[1] - point[1], estimate[0] - point[0]
            )))
            crossing = abs(math.sin(math.radians(
                angle_difference(first.bearing_deg, predicted)
            )))
            expected_radius = max(12.0, 35.0 / max(0.12, crossing))
            gain = max(0.0, 2.0 * math.log(1500.0 / expected_radius))
            best_future_rate = max(
                best_future_rate, gain * signal_probability / 6.0
            )
        # 只有未来公共点很可能提供高质量交会线，且当前插入代价明显偏大时才等待。
        return not (best_future_rate >= 0.22 and insertion_time > 95.0)

    def execute_probe(self, plan: dict) -> None:
        state: ChannelState = plan["state"]
        # 只有真正执行的动作才进入去重集合；联合规划中被新观测取消的预案不算执行。
        self.used_probe_actions[state.channel].add(plan["action"])
        old_time = self.client.virtual_time_s
        result = self.measure_channel(plan["point"], state.channel)
        if not result:
            return
        measurement = result.get("measure_result")
        reward = 0.0
        last_signal = False
        if measurement == "near":
            reward += 180.0
            success = self.clear_at(state, plan["point"])
            reward += 500.0 if success else -80.0
            last_signal = True
        elif measurement == "direction":
            self.add_direction(state, plan["point"], result)
            last_signal = True
            reward += 35.0
            estimate, _ = estimate_target(state.observations)
            # 首次补测仍不足以达到14米安全误差时，就在当前目标附近完成下一测。
            # 避免把1~3个困难目标留到batch_probe_2后再次横跨整个场地。
            if (
                self.immediate_refine and not self.localization_ready(state)
                and len(state.observations) >= 2 and estimate is not None
                and self.should_refine_now(state, estimate, plan)
            ):
                previous_phase = self.client.phase
                self.client.phase = "local_refine"
                refine = self.measure_channel(estimate, state.channel)
                if refine and refine.get("measure_result") == "near":
                    success = self.clear_at(state, estimate)
                    reward += 500.0 if success else -80.0
                elif refine and refine.get("measure_result") == "direction":
                    self.add_direction(state, estimate, refine)
                else:
                    reward -= 28.0
                self.client.phase = previous_phase
            # Q2测点已主动靠近目标；满足20米清除条件时立刻就近完成Q3清除，
            # 避免补测巡回结束后再次返回同一目标区域。
            if not state.cleared and self.localization_ready(state):
                estimate, _ = estimate_target(state.observations)
                if estimate is not None:
                    previous_phase = self.client.phase
                    self.client.phase = "active_probe_clear"
                    success = self.clear_at(state, estimate)
                    self.client.phase = previous_phase
                    reward += 500.0 if success else -80.0
        else:
            reward -= 28.0

        # 当前Q2/清除位置若能覆盖新的公共区域，就顺便扫其余频道。
        # 这会把原本单独的公共覆盖航点压缩进目标定位路线中。
        sweep_new_channels = sweep_new_cells = 0
        sweep_gain = 0.0
        if self.adaptive_coverage:
            (
                sweep_new_channels,
                sweep_new_cells,
                sweep_gain,
                _,
            ) = self.opportunistic_sweep(state.channel)
        shared_useful = 0
        shared_gain = 0.0
        if self.shared_probe and not self.adaptive_coverage:
            shared_useful, shared_gain, _ = self.shared_probe_sweep(state.channel)
        _, uncertainty = estimate_target(state.observations)
        primary_gain = information_gain(plan["old_uncertainty"], uncertainty)
        elapsed = self.client.virtual_time_s - old_time
        # 面积不确定性的对数下降量是信息增益；虚拟秒数直接作为成本。
        # 新频道与新覆盖区域只奖励首次获得，避免原地反复测量刷奖励。
        reward += (
            115.0 * (primary_gain + sweep_gain + shared_gain)
            + 75.0 * sweep_new_channels
            + 35.0 * shared_useful
            + 0.9 * sweep_new_cells
            - 0.14 * elapsed
        )
        next_key = self.policy.state_key(
            self.problem, len(state.observations), uncertainty, last_signal
        )
        self.policy.update(plan["state_key"], plan["action"], reward, next_key)

    def batch_localize_and_clear(
        self, states: Sequence[ChannelState], end_hint: Optional[Point]
    ) -> None:
        """分轮集中访问所有补测点，再按2-opt路径集中清除，避免跨圆域往返。"""
        active = [state for state in states if state.observations and not state.cleared]
        for attempt in range(self.max_probes):
            plans = [self.plan_probe(state, attempt) for state in active]
            plans = [plan for plan in plans if plan is not None]
            if not plans:
                break
            if self.joint_probe_variants:
                self.optimize_probe_variants(plans, end_hint)
            order = (
                self.optimized_plan_order(plans, end_hint)
                if self.directed_plan_order
                else optimized_visit_order(
                    [plan["point"] for plan in plans], self.client.position, end_hint
                )
            )
            self.client.phase = f"batch_probe_{attempt + 1}"
            for index in order:
                plan = plans[index]
                if (
                    not plan["state"].cleared
                    and not self.localization_ready(plan["state"])
                ):
                    self.execute_probe(plan)

        clear_tasks = []
        for state in active:
            if state.cleared:
                continue
            estimate, uncertainty = estimate_target(state.observations)
            if estimate is not None:
                clear_tasks.append((state, estimate, uncertainty))
        order = optimized_visit_order(
            [task[1] for task in clear_tasks], self.client.position, end_hint
        )
        self.client.phase = "batch_clear"
        failed: List[ChannelState] = []
        for index in order:
            state, estimate, uncertainty = clear_tasks[index]
            success = self.clear_at(state, estimate)
            if not success:
                failed.append(state)

        # 通常仅有极少数失败目标；保留原来的补测/备选交点逻辑以维持清除准确率。
        self.client.phase = "clear_recovery"
        for state in failed:
            self.localize_and_clear(state)

    def run_adaptive_coverage(self) -> None:
        """用Q2/Q3访问点替代部分固定覆盖点，并保留保守的全场覆盖判据。"""
        remaining = list(self.scan_waypoints)
        public_scans = 0
        first_batch_done = False
        while remaining and self.cleared_count < 16:
            # 已被Q2/Q3测量圆完全覆盖的固定航点不再访问。
            remaining = [p for p in remaining if self.coverage_gain_cells(p)]
            if not remaining or self.coverage_ratio >= 1.0:
                break
            waypoint = self.select_coverage_waypoint(remaining)
            remaining.remove(waypoint)
            public_scans += 1
            print(
                f"\n自适应覆盖 {public_scans}: {waypoint}，"
                f"当前保守覆盖率={self.coverage_ratio:.1%}"
            )
            self.client.phase = "adaptive_coverage"
            self.scan_one_waypoint(waypoint)

            # 三个公共站形成第一批可靠示向线后，立即让目标访问点承担后续覆盖。
            # 此后不频繁打断公共路径，避免生成多轮跨圆域目标巡回。
            if public_scans >= 3 and not first_batch_done:
                active = [
                    state for state in self.channels.values()
                    if state.observations and not state.cleared
                ]
                next_hint = (
                    self.select_coverage_waypoint(remaining) if remaining else None
                )
                self.batch_localize_and_clear(active, next_hint)
                first_batch_done = True

        # 处理覆盖途中及机会扫描中新发现的频道；第二批也可能继续贡献覆盖面积。
        pending = [
            state for state in self.channels.values()
            if state.observations and not state.cleared
        ]
        self.batch_localize_and_clear(pending, None)

    def run_joint_coverage_localization(self) -> None:
        """滚动联合规划固定覆盖任务、Q2补测任务和Q3清除任务。"""
        remaining_coverage = list(self.scan_waypoints)
        pending_plans: Dict[int, dict] = {}
        probe_attempts = {channel: 0 for channel in self.channels}
        failed_joint_clear = set()
        step = 0

        while self.cleared_count < 16:
            if (
                self.problem == 3 and len(self.sweep_positions) == 1
                and len(remaining_coverage) == 6
            ):
                remaining_coverage = self.adapt_problem3_coverage_ring(
                    remaining_coverage
                )
            # 公共点带来的新观测会让旧补测计划失效，必须按最新可行域重建。
            for channel, plan in list(pending_plans.items()):
                state = self.channels[channel]
                if (
                    state.cleared or self.localization_ready(state)
                    or len(state.observations) != plan["observation_count"]
                ):
                    pending_plans.pop(channel, None)

            nodes: List[dict] = [
                {
                    "kind": "coverage",
                    "point": point,
                    "expected_exit": point,
                    "coverage_point": point,
                }
                for point in remaining_coverage
            ]
            completed_coverage = len(self.scan_waypoints) - len(remaining_coverage)
            if completed_coverage < self.joint_coverage_warmup:
                order = self.optimized_plan_order(nodes, None)
                node = nodes[order[0]]
                step += 1
                point = node["coverage_point"]
                print(
                    f"\n联合规划预热{completed_coverage + 1}/"
                    f"{self.joint_coverage_warmup}：公共覆盖点 {point}"
                )
                self.client.phase = "joint_coverage_warmup"
                self.scan_one_waypoint(point)
                remaining_coverage.remove(point)
                continue
            for channel, state in self.channels.items():
                if state.cleared or not state.observations:
                    continue
                if self.localization_ready(state):
                    if channel in failed_joint_clear:
                        continue
                    estimate, _ = estimate_target(state.observations)
                    if estimate is not None:
                        nodes.append({
                            "kind": "clear",
                            "point": estimate,
                            "expected_exit": estimate,
                            "state": state,
                        })
                    continue
                if probe_attempts[channel] >= self.max_probes:
                    continue
                if channel not in pending_plans:
                    plan = self.plan_probe(state, probe_attempts[channel])
                    if plan is not None:
                        pending_plans[channel] = plan
                plan = pending_plans.get(channel)
                if plan is not None:
                    node = dict(plan)
                    node["kind"] = "probe"
                    nodes.append(node)

            if not nodes:
                break
            self.optimize_joint_probe_variants(nodes, None)
            order = self.optimized_plan_order(nodes, None)
            # 每执行一个任务后都会回到循环顶部重建状态与路径。
            node = nodes[order[0]]
            step += 1
            if node["kind"] == "coverage":
                point = node["coverage_point"]
                print(
                    f"\n联合规划步骤{step}：公共覆盖点 {point}，"
                    f"剩余覆盖点={len(remaining_coverage)}"
                )
                self.client.phase = "joint_coverage"
                self.scan_one_waypoint(point)
                remaining_coverage.remove(point)
            elif node["kind"] == "probe":
                state = node["state"]
                print(
                    f"\n联合规划步骤{step}：频道{state.channel}主动补测，"
                    f"剩余覆盖点={len(remaining_coverage)}"
                )
                self.client.phase = "joint_probe"
                node["next_route_point"] = (
                    nodes[order[1]]["point"] if len(order) > 1 else None
                )
                node["future_coverage_points"] = list(remaining_coverage)
                self.execute_probe(node)
                probe_attempts[state.channel] += 1
                pending_plans.pop(state.channel, None)
            else:
                state = node["state"]
                print(f"\n联合规划步骤{step}：频道{state.channel}直接清除")
                self.client.phase = "joint_clear"
                if not self.clear_at(state, node["point"]):
                    self.client.phase = "joint_local_recovery"
                    if not self.recover_clear_locally(state):
                        failed_joint_clear.add(state.channel)

        if remaining_coverage and self.cleared_count < 16:
            # 正常不会进入；作为异常规划结果的发现率兜底。
            for point in remaining_coverage:
                self.client.phase = "joint_coverage_fallback"
                self.scan_one_waypoint(point)

    def run(self) -> dict:
        enter = self.client.enter()
        print(
            f"进入成功，现实剩余时间={enter.get('remaining_real_duration_s')}秒，"
            f"问题{self.problem}，探索率epsilon={self.policy.epsilon:.3f}"
        )
        try:
            if self.joint_coverage_localization:
                self.run_joint_coverage_localization()
            elif self.adaptive_coverage:
                self.run_adaptive_coverage()
            else:
                waypoints = list(self.scan_waypoints)
                index = 0
                while index < len(waypoints):
                    waypoint = waypoints[index]
                    index += 1
                    print(f"\n覆盖航点 {index}/{len(waypoints)}: {waypoint}")
                    self.client.phase = "coverage_scan"
                    self.scan_one_waypoint(waypoint)
                    if (
                        self.problem == 3 and index == 1
                        and self.optimize_coverage_endpoint
                    ):
                        waypoints[1:] = self.optimized_ring_order(waypoints[1:])

                    # 不再逐频道立即往返定位。每若干公共航点集中补测、集中清除。
                    batch_due = index % self.batch_size == 0 or index == len(waypoints)
                    if batch_due:
                        next_waypoint = (
                            waypoints[index]
                            if index < len(waypoints)
                            else None
                        )
                        active = [
                            state for state in self.channels.values()
                            if state.observations and not state.cleared
                        ]
                        self.batch_localize_and_clear(active, next_waypoint)
                    if self.cleared_count >= 16:
                        print("已达到题目给出的最大干扰源数16，提前结束搜索")
                        break

            # 对最后仍有示向度但未成功清除的频道集中补救。
            pending = [
                state for state in self.channels.values()
                if state.observations and not state.cleared
            ]
            self.batch_localize_and_clear(pending, None)
        finally:
            self.client.exit()
            self.policy.save()

        phase_time: Dict[str, float] = {}
        phase_distance: Dict[str, float] = {}
        for record in self.client.records:
            phase = record.get("phase", "unknown")
            phase_time[phase] = phase_time.get(phase, 0.0) + float(
                record["delta_virtual_time_s"]
            )
            phase_distance[phase] = phase_distance.get(phase, 0.0) + float(
                record["move_distance_m"]
            )
        geometry_rows = [
            localization_geometry(state.observations)
            for state in self.channels.values() if state.observations
        ]
        finite_radii = [
            item.radius for item in geometry_rows if math.isfinite(item.radius)
        ]
        return {
            "problem": self.problem,
            "cleared_count": self.cleared_count,
            "observed_channels": [
                c for c, s in self.channels.items() if s.observations
            ],
            "cleared_channels": [
                c for c, s in self.channels.items() if s.cleared
            ],
            "virtual_time_s": self.client.virtual_time_s,
            "average_time_per_cleared_s": (
                self.client.virtual_time_s / self.cleared_count
                if self.cleared_count else None
            ),
            "path_distance_m": sum(r["move_distance_m"] for r in self.client.records),
            "max_single_move_m": max(
                (r["move_distance_m"] for r in self.client.records), default=0.0
            ),
            "cross_field_move_count": sum(
                r["move_distance_m"] > 850.0 for r in self.client.records
            ),
            "action_count": len(self.client.records),
            "mean_observations_per_cleared": (
                sum(len(state.observations) for state in self.channels.values())
                / max(1, self.cleared_count)
            ),
            "geometry_jung_certified_channels": sum(
                item.jung_certified for item in geometry_rows
            ),
            "geometry_mec_ready_channels": sum(
                item.radius <= CLEAR_RADIUS - CLEARANCE_NUMERIC_MARGIN_M
                for item in geometry_rows
            ),
            "mean_final_feasible_radius_m": (
                statistics.mean(finite_radii) if finite_radii else None
            ),
            "batch_size": self.batch_size,
            "adaptive_coverage": self.adaptive_coverage,
            "shared_probe": self.shared_probe,
            "shared_probe_limit": self.shared_probe_limit,
            "joint_probe_variants": self.joint_probe_variants,
            "optimize_coverage_endpoint": self.optimize_coverage_endpoint,
            "directed_plan_order": self.directed_plan_order,
            "immediate_refine": self.immediate_refine,
            "information_gated_coverage": self.information_gated_coverage,
            "joint_coverage_localization": self.joint_coverage_localization,
            "joint_coverage_warmup": self.joint_coverage_warmup,
            "information_rate_planning": self.information_rate_planning,
            "cost_aware_refine": self.cost_aware_refine,
            "probabilistic_exit_planning": self.probabilistic_exit_planning,
            "negative_range_inference": self.negative_range_inference,
            "negative_range_min_samples": self.negative_range_min_samples,
            "coverage_information_rate_threshold": (
                self.coverage_information_rate_threshold
            ),
            "route_aligned_probe_variants": self.joint_probe_variants,
            "route_tail_penalty": self.route_tail_penalty,
            "coverage_ring_phase_deg": self.coverage_ring_phase_deg,
            "probe_lateral_range_m": [
                self.probe_lateral_min_m, self.probe_lateral_max_m
            ],
            "expanded_rigo_candidates": self.expanded_rigo_candidates,
            "certified_optical_fallback": self.certified_optical_fallback,
            "conservative_coverage_ratio": self.coverage_ratio,
            "sweep_position_count": len(self.sweep_positions),
            "virtual_time_by_phase_s": phase_time,
            "distance_by_phase_m": phase_distance,
        }

    @property
    def cleared_count(self) -> int:
        return sum(state.cleared for state in self.channels.values())


def save_outputs(
    output_dir: Path,
    records: Sequence[dict],
    summary: dict,
    channels: Dict[int, ChannelState],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        with (output_dir / "robot_records.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    detail = {
        "summary": summary,
        "channels": {
            str(channel): {
                "cleared": state.cleared,
                "clear_position": state.clear_position,
                "failed_clear_positions": state.failed_clear_positions,
                "no_signal_positions": state.no_signal_positions,
                "observations": [
                    {
                        "position": obs.position,
                        "bearing_deg": obs.bearing_deg,
                        "virtual_time_s": obs.virtual_time_s,
                    }
                    for obs in state.observations
                ],
            }
            for channel, state in channels.items()
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def draw_path(
    output_file: Path,
    records: Sequence[dict],
    summary: dict,
    show: bool,
) -> None:
    if not records:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
    except ImportError:
        print("未安装matplotlib，搜索日志已保存，但跳过二维绘图。可执行：pip install matplotlib")
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    ax.add_patch(Circle(
        (0, 0), ARENA_RADIUS, fill=False, linestyle="--", linewidth=1.7,
        color="#555555", label="目标区域边界",
    ))

    points = [(0.0, 0.0)] + [(float(r["x"]), float(r["y"])) for r in records]
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    colors = list(range(1, len(segments) + 1))
    line = LineCollection(segments, cmap="viridis", linewidths=1.8, alpha=0.82)
    line.set_array(colors)
    ax.add_collection(line)
    colorbar = fig.colorbar(line, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("动作顺序")

    categories = {
        "direction": ("^", "#159947", "测得示向度"),
        "no_signal": ("x", "#8A8A8A", "无信号"),
        "near": ("D", "#F39C12", "距离过近"),
    }
    for result_name, (marker, color, label) in categories.items():
        selected = [
            r for r in records
            if r["path"] == "/measure" and r["measure_result"] == result_name
        ]
        if selected:
            ax.scatter(
                [r["x"] for r in selected], [r["y"] for r in selected],
                marker=marker, color=color, s=34, alpha=0.8, label=label, zorder=4,
            )

    clear_success = [
        r for r in records
        if r["path"] == "/clear" and r["clear_result"] == "success"
    ]
    clear_failed = [
        r for r in records
        if r["path"] == "/clear" and r["clear_result"] != "success"
    ]
    if clear_failed:
        ax.scatter(
            [r["x"] for r in clear_failed], [r["y"] for r in clear_failed],
            marker="X", facecolors="none", edgecolors="#C0392B", s=70,
            linewidths=1.4, label="清除未命中", zorder=5,
        )
    if clear_success:
        ax.scatter(
            [r["x"] for r in clear_success], [r["y"] for r in clear_success],
            marker="*", color="#D7191C", edgecolors="white", linewidths=0.7,
            s=210, label="已清除干扰源", zorder=7,
        )
        for r in clear_success:
            ax.annotate(
                f"Ch{r['channel']}", (r["x"], r["y"]),
                xytext=(7, 7), textcoords="offset points", fontsize=9,
                weight="bold", color="#A51414",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#E7B0B0", alpha=0.9),
            )

    # 为每个频道仅绘制最近三条测向射线，防止图面过密。
    direction_records: Dict[int, List[dict]] = {}
    for record in records:
        if record["path"] == "/measure" and record["measure_result"] == "direction":
            direction_records.setdefault(int(record["channel"]), []).append(record)
    cmap = plt.get_cmap("tab20")
    for channel, channel_records in direction_records.items():
        for record in channel_records[-3:]:
            ux, uy = bearing_vector(float(record["svd_deg"]))
            length = 650.0
            ax.plot(
                [record["x"], record["x"] + length * ux],
                [record["y"], record["y"] + length * uy],
                color=cmap((channel - 1) % 20), alpha=0.33, linewidth=1.0,
            )

    ax.scatter([0], [0], marker="P", s=110, color="black", label="起点", zorder=8)
    extent = max(
        2000.0,
        max(max(abs(x), abs(y)) for x, y in points) + 220.0,
    )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("东向坐标 x / m")
    ax.set_ylabel("北向坐标 y / m")
    ax.set_title(
        f"问题{summary['problem']}机器狗搜索与清除轨迹\n"
        f"清除 {summary['cleared_count']} 个 | 路程 {summary['path_distance_m']:.0f} m | "
        f"虚拟时间 {summary['virtual_time_s']:.1f} s"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#3B528B", lw=2, label="机器狗路径"))
    labels.append("机器狗路径")
    ax.legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.92)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=320, bbox_inches="tight", facecolor="white")
    print(f"二维轨迹图已保存：{output_file}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B题问题3/4强化学习辅助演练程序")
    parser.add_argument("--problem", type=int, choices=(3, 4), default=3)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--epsilon", type=float, default=0.16, help="演练探索率")
    parser.add_argument("--formal", action="store_true", help="正式模式：关闭随机探索")
    parser.add_argument("--max-probes", type=int, default=4)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="累计多少个公共航点后集中定位清除；默认问题3走完整8点、问题4每7点",
    )
    parser.add_argument(
        "--adaptive-coverage", action="store_true",
        help="实验模式：让Q2/Q3目标访问点替代部分公共覆盖点",
    )
    parser.add_argument(
        "--staged-coverage", action="store_true",
        help="回退到先完成公共覆盖、再集中定位清除的旧分阶段模式",
    )
    parser.add_argument(
        "--joint-warmup", type=int, default=1,
        help="联合规划前必须完成的公共观测点数量；默认仅中心点1个",
    )
    parser.add_argument(
        "--no-information-rate-planning", action="store_true",
        help="关闭默认的信息增益率公共复测门控，用于A/B对照",
    )
    parser.add_argument(
        "--coverage-information-rate-threshold", type=float, default=0.08,
        help="公共覆盖点顺路复测的信息增益率门槛",
    )
    parser.add_argument(
        "--route-tail-penalty", type=float, default=0.30,
        help="超过850 m首跳的二次长尾惩罚系数",
    )
    parser.add_argument(
        "--rigo-route-credit-m", type=float, default=18.0,
        help="信息增益折算为可接受绕行距离的系数",
    )
    parser.add_argument(
        "--no-cost-aware-refine", action="store_true",
        help="关闭默认的即时补测成本判断，恢复为定位不足就立即补测",
    )
    parser.add_argument(
        "--no-probabilistic-exit-planning", action="store_true",
        help="关闭默认的Q2/Q3概率出口路径成本，用于A/B对照",
    )
    parser.add_argument(
        "--no-negative-range-inference", action="store_true",
        help="关闭默认的无信号负观测距离推断，用于A/B对照",
    )
    parser.add_argument(
        "--shared-probe", action="store_true",
        help="兼容参数：受限共享Q2测量现已默认启用",
    )
    parser.add_argument(
        "--no-shared-probe", action="store_true",
        help="A/B回归：关闭受限共享Q2测量",
    )
    parser.add_argument(
        "--shared-probe-limit", type=int, default=2,
        help="每个Q2位置最多顺带测量的其他频道数，默认2",
    )
    parser.add_argument(
        "--joint-probe-variants", action="store_true",
        help="兼容参数：启用联合选择各目标的左/右Q2补测点",
    )
    parser.add_argument(
        "--no-route-aligned-probes", action="store_true",
        help="关闭默认的Q2左右测点路线对齐，用于A/B对照",
    )
    parser.add_argument(
        "--optimize-coverage-endpoint", action="store_true",
        help="实验模式：根据中心首测结果选择公共覆盖环的终点",
    )
    parser.add_argument(
        "--legacy-plan-order", action="store_true",
        help="回退到仅按Q2测点排序的旧巡回，用于A/B对照",
    )
    parser.add_argument(
        "--no-immediate-refine", action="store_true",
        help="关闭默认的目标邻域即时补测，用于与旧版第二轮补测做A/B对照",
    )
    parser.add_argument(
        "--no-information-gated-coverage", action="store_true",
        help="关闭默认的信息增益门控公共复测，用于A/B对照",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-show", action="store_true", help="保存图片但不弹出窗口")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-expanded-rigo-candidates", action="store_true",
        help="关闭指导论文多尺度RIGO候选点，用于A/B对照",
    )
    parser.add_argument(
        "--no-certified-optical-fallback", action="store_true",
        help="关闭常规定位失败后的有界光学兜底，用于A/B对照",
    )
    return parser.parse_args()


def main() -> None:
    program_started_perf = time.perf_counter()
    program_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    args = parse_args()
    if not args.robot_id.strip():
        raise SystemExit("请使用 --robot-id 填入模拟器分配的团队号/机器狗编号")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size必须为正整数")
    if args.max_probes <= 0:
        raise SystemExit("--max-probes必须为正整数")
    if args.joint_warmup <= 0:
        raise SystemExit("--joint-warmup必须为正整数")
    if args.seed is not None:
        random.seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else (
        script_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    q_path = script_dir / "rl_policy.json"
    epsilon = 0.0 if args.formal else max(0.0, min(1.0, args.epsilon))
    policy = QLearningPolicy(
        q_path,
        epsilon=epsilon,
        alpha=0.22,
        # 正式模式既不update也不save，Q文件字节级保持不变。
        training=not args.formal,
    )
    client = SimulatorClient(args.base_url, args.robot_id)
    agent = SearchAgent(
        client,
        policy,
        args.problem,
        max_probes=args.max_probes,
        batch_size=args.batch_size,
        adaptive_coverage=args.adaptive_coverage,
        shared_probe=(args.shared_probe or not args.no_shared_probe),
        shared_probe_limit=args.shared_probe_limit,
        joint_probe_variants=(
            args.joint_probe_variants or not args.no_route_aligned_probes
        ),
        optimize_coverage_endpoint=args.optimize_coverage_endpoint,
        directed_plan_order=not args.legacy_plan_order,
        immediate_refine=not args.no_immediate_refine,
        information_gated_coverage=not args.no_information_gated_coverage,
        joint_coverage_localization=(
            not args.staged_coverage and not args.adaptive_coverage
        ),
        joint_coverage_warmup=args.joint_warmup,
        information_rate_planning=not args.no_information_rate_planning,
        cost_aware_refine=not args.no_cost_aware_refine,
        probabilistic_exit_planning=not args.no_probabilistic_exit_planning,
        negative_range_inference=not args.no_negative_range_inference,
        coverage_information_rate_threshold=(
            args.coverage_information_rate_threshold
        ),
        route_tail_penalty=args.route_tail_penalty,
        rigo_route_credit_m=args.rigo_route_credit_m,
        expanded_rigo_candidates=not args.no_expanded_rigo_candidates,
        certified_optical_fallback=not args.no_certified_optical_fallback,
    )

    summary = None
    try:
        summary = agent.run()
        print("\n运行汇总：")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("收到中断信号，尝试主动退出")
        client.exit()
    except Exception as exc:
        print(f"运行失败：{exc}")
        client.exit()
        raise
    finally:
        policy.save()
        if summary is None:
            summary = {
                "problem": args.problem,
                "cleared_count": agent.cleared_count,
                "observed_channels": [c for c, s in agent.channels.items() if s.observations],
                "cleared_channels": [c for c, s in agent.channels.items() if s.cleared],
                "virtual_time_s": client.virtual_time_s,
                "path_distance_m": sum(r["move_distance_m"] for r in client.records),
                "max_single_move_m": max(
                    (r["move_distance_m"] for r in client.records), default=0.0
                ),
                "cross_field_move_count": sum(
                    r["move_distance_m"] > 850.0 for r in client.records
                ),
                "action_count": len(client.records),
                "batch_size": agent.batch_size,
                "adaptive_coverage": agent.adaptive_coverage,
                "shared_probe": agent.shared_probe,
                "shared_probe_limit": agent.shared_probe_limit,
                "joint_probe_variants": agent.joint_probe_variants,
                "optimize_coverage_endpoint": agent.optimize_coverage_endpoint,
                "directed_plan_order": agent.directed_plan_order,
                "immediate_refine": agent.immediate_refine,
                "information_gated_coverage": agent.information_gated_coverage,
                "joint_coverage_localization": agent.joint_coverage_localization,
                "joint_coverage_warmup": agent.joint_coverage_warmup,
                "information_rate_planning": agent.information_rate_planning,
                "cost_aware_refine": agent.cost_aware_refine,
                "probabilistic_exit_planning": agent.probabilistic_exit_planning,
                "negative_range_inference": agent.negative_range_inference,
                "negative_range_min_samples": agent.negative_range_min_samples,
                "coverage_information_rate_threshold": (
                    agent.coverage_information_rate_threshold
                ),
                "route_aligned_probe_variants": agent.joint_probe_variants,
                "route_tail_penalty": agent.route_tail_penalty,
                "probe_lateral_range_m": [
                    agent.probe_lateral_min_m, agent.probe_lateral_max_m
                ],
                "expanded_rigo_candidates": agent.expanded_rigo_candidates,
                "certified_optical_fallback": agent.certified_optical_fallback,
                "conservative_coverage_ratio": agent.coverage_ratio,
            }
        program_runtime_s = time.perf_counter() - program_started_perf
        summary["program_started_at"] = program_started_at
        summary["program_runtime_s"] = program_runtime_s
        summary["program_runtime_minutes"] = program_runtime_s / 60.0
        save_outputs(output_dir, client.records, summary, agent.channels)
        draw_path(output_dir / "robot_path_enhanced.png", client.records, summary, not args.no_show)
        print(
            f"程序现实运行时间：{program_runtime_s:.3f} s "
            f"({program_runtime_s / 60.0:.3f} min)"
        )


if __name__ == "__main__":
    main()
