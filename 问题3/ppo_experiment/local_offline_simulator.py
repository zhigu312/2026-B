#!/usr/bin/env python3
"""B题问题3/4离线模拟器。

物理判定、虚拟计时、请求字段和幂等规则按题面附件1/2实现。官方
未公开随机案例生成器，因此案例分布单独放在 CaseDistribution 中，
可在不改变规则的前提下做域随机化或后续校准。

用法：
  python local_offline_simulator.py self-test
  python local_offline_simulator.py serve --problem 3 --seed 2026 --port 2027
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit


Point = Tuple[float, float]


@dataclass(frozen=True)
class OfficialRules:
    arena_radius_m: float = 1800.0
    channel_min: int = 1
    channel_max: int = 20
    jammer_count_min: int = 10
    jammer_count_max: int = 16
    receive_radius_min_m: float = 1000.0
    receive_radius_max_m: float = 1500.0
    directional_width_deg: float = 180.0
    bearing_error_deg: float = 1.0
    near_radius_m: float = 5.0
    clear_radius_m: float = 20.0
    speed_mps: float = 5.0
    channel_switch_s: float = 1.0
    measure_s: float = 5.0
    clear_miss_s: float = 3.0
    clear_success_s: float = 5.0
    max_virtual_duration_s: float = 360000.0
    max_real_duration_s: int = 1200
    max_coordinate_abs_m: float = 2_000_000.0


RULES = OfficialRules()


@dataclass
class Jammer:
    channel: int
    x: float
    y: float
    receive_radius_m: float
    directional: bool = False
    direction_deg: Optional[float] = None
    cleared: bool = False
    error_seed: int = 0

    @property
    def position(self) -> Point:
        return self.x, self.y


@dataclass(frozen=True)
class CaseDistribution:
    """未公开部分的显式假设。

    official_estimate：频道无放回、圆盘面积均匀、半径和方向均匀。
    问题4默认使定向源约占30%，与本机已有2个演练案例的3/10一致。
    robust：问题4定向源数在1..n-1间随机，用于更强的分布外泛化。
    """

    mode: str = "official_estimate"
    directional_fraction: float = 0.30


def _normalize_angle(value: float) -> float:
    return value % 360.0


def _angle_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_case(
    problem: int,
    seed: int,
    distribution: CaseDistribution = CaseDistribution(),
    count: Optional[int] = None,
) -> List[Jammer]:
    if problem not in (3, 4):
        raise ValueError("problem必须是3或4")
    rng = random.Random(int(seed))
    n = count if count is not None else rng.randint(
        RULES.jammer_count_min, RULES.jammer_count_max
    )
    if not RULES.jammer_count_min <= n <= RULES.jammer_count_max:
        raise ValueError("干扰源数必须在10..16")
    channels = rng.sample(
        range(RULES.channel_min, RULES.channel_max + 1), n
    )
    if problem == 3:
        directional_channels = set()
    elif distribution.mode == "robust":
        directional_channels = set(rng.sample(channels, rng.randint(1, n - 1)))
    else:
        directional_count = max(
            1, min(n - 1, int(round(n * distribution.directional_fraction)))
        )
        directional_channels = set(rng.sample(channels, directional_count))

    result: List[Jammer] = []
    for channel in channels:
        # sqrt(U)保证圆盘上按面积均匀，而非半径均匀。
        radius = RULES.arena_radius_m * math.sqrt(rng.random())
        theta = rng.random() * 2.0 * math.pi
        directional = channel in directional_channels
        result.append(Jammer(
            channel=channel,
            x=radius * math.cos(theta),
            y=radius * math.sin(theta),
            receive_radius_m=rng.uniform(
                RULES.receive_radius_min_m, RULES.receive_radius_max_m
            ),
            directional=directional,
            direction_deg=rng.uniform(0.0, 360.0) if directional else None,
            error_seed=rng.getrandbits(64),
        ))
    return result


class LocalHTTPError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class InterfaceClosed(ConnectionError):
    pass


def _identifier_ok(value: object, max_bytes: int) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if len(value.encode("utf-8")) > max_bytes:
        return False
    return not any(unicodedata.category(c) in ("Cc", "Cf") for c in value)


class LocalSimulatorCore:
    """与官方四个HTTP端点具有相同状态转移的单案例内核。"""

    def __init__(
        self,
        problem: int,
        seed: int,
        robot_id: str = "offline-robot",
        distribution: CaseDistribution = CaseDistribution(),
        count: Optional[int] = None,
        jammers: Optional[Sequence[Jammer]] = None,
    ):
        if not _identifier_ok(robot_id, 64):
            raise ValueError("robot_id必须是1..64字节的可见字符串")
        self.problem = problem
        self.seed = int(seed)
        self.robot_id = robot_id
        self.distribution = distribution
        source = jammers if jammers is not None else generate_case(
            problem, seed, distribution, count
        )
        self.jammers: Dict[int, Jammer] = {
            j.channel: Jammer(**asdict(j)) for j in source
        }
        self.position: Point = (0.0, 0.0)
        self.current_channel = 1
        self.virtual_time_us = 0
        self.entered = False
        self.closed = False
        self.entered_monotonic: Optional[float] = None
        self.idempotency: Dict[str, Tuple[str, str, dict]] = {}
        self.lock = threading.Lock()

    @property
    def virtual_time_s(self) -> float:
        value = self.virtual_time_us / 1_000_000.0
        return int(value) if value.is_integer() else value

    def _base_response(self, accepted: bool = True) -> dict:
        return {
            "accepted": accepted,
            "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": self.virtual_time_s if accepted else 0,
        }

    def _is_timed_out(self) -> bool:
        if self.virtual_time_s >= RULES.max_virtual_duration_s:
            return True
        return bool(
            self.entered_monotonic is not None
            and time.monotonic() - self.entered_monotonic
            >= RULES.max_real_duration_s
        )

    def _bearing_error(self, jammer: Jammer, point: Point) -> float:
        # 同源同地点必然相同；不同地点在[-1,1]上呈统计变化。
        key = (
            f"{jammer.error_seed}|{jammer.channel}|"
            f"{point[0]:.12g}|{point[1]:.12g}"
        ).encode("ascii")
        raw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        unit = raw / float((1 << 64) - 1)
        return (2.0 * unit - 1.0) * RULES.bearing_error_deg

    @staticmethod
    def _canonical(path: str, payload: dict) -> Tuple[str, str]:
        body = json.dumps(
            payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        return path, body

    def _validate_base(self, payload: dict, allowed: set) -> Optional[dict]:
        required = {"arena_id", "robot_id", "request_id"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise LocalHTTPError(400, "missing_or_invalid_base_fields")
        if not _identifier_ok(payload.get("robot_id"), 64) or not _identifier_ok(
            payload.get("request_id"), 128
        ):
            raise LocalHTTPError(400, "invalid_identifier")
        if allowed - set(payload):
            raise LocalHTTPError(400, "missing_required_fields")
        if set(payload) - allowed:
            return self._base_response(False)
        if payload["arena_id"] != "default" or payload["robot_id"] != self.robot_id:
            return self._base_response(False)
        return None

    @staticmethod
    def _validate_action(payload: dict) -> Tuple[Point, int]:
        position = payload.get("position")
        if not isinstance(position, dict) or not {"x", "y"}.issubset(position):
            raise LocalHTTPError(400, "invalid_position")
        coordinates = []
        for name in ("x", "y"):
            value = position[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LocalHTTPError(400, "invalid_coordinate")
            value = float(value)
            if not math.isfinite(value) or abs(value) > RULES.max_coordinate_abs_m:
                raise LocalHTTPError(400, "invalid_coordinate")
            coordinates.append(value)
        channel = payload.get("channel")
        if (
            isinstance(channel, bool)
            or not isinstance(channel, (int, float))
            or not math.isfinite(float(channel))
            or not float(channel).is_integer()
            or not RULES.channel_min <= int(channel) <= RULES.channel_max
        ):
            raise LocalHTTPError(400, "invalid_channel")
        return (coordinates[0], coordinates[1]), int(channel)

    def _advance(self, seconds: float) -> None:
        # 附件2：内部按微秒累计。
        self.virtual_time_us += int(round(seconds * 1_000_000.0))

    def _in_coverage(self, jammer: Jammer, point: Point) -> bool:
        if _distance(jammer.position, point) > jammer.receive_radius_m + 1e-9:
            return False
        if not jammer.directional:
            return True
        source_to_point = _normalize_angle(math.degrees(math.atan2(
            point[1] - jammer.y, point[0] - jammer.x
        )))
        return abs(_angle_difference(source_to_point, jammer.direction_deg or 0.0)) \
            <= RULES.directional_width_deg / 2.0 + 1e-10

    def _measure_result(self, point: Point, channel: int) -> dict:
        jammer = self.jammers.get(channel)
        if jammer is None or jammer.cleared or not self._in_coverage(jammer, point):
            return {"measure_result": "no_signal"}
        if _distance(jammer.position, point) <= RULES.near_radius_m + 1e-9:
            return {"measure_result": "near"}
        true_bearing = _normalize_angle(math.degrees(math.atan2(
            jammer.y - point[1], jammer.x - point[0]
        )))
        measured = _normalize_angle(true_bearing + self._bearing_error(jammer, point))
        return {"measure_result": "direction", "svd_deg": round(measured, 2)}

    def _execute(self, path: str, payload: dict) -> dict:
        if path not in ("/enter", "/measure", "/clear", "/exit"):
            raise LocalHTTPError(404, "unknown_path")

        action_fields = {"arena_id", "robot_id", "request_id", "position", "channel"}
        base_fields = {"arena_id", "robot_id", "request_id"}
        rejected = self._validate_base(
            payload, action_fields if path in ("/measure", "/clear") else base_fields
        )
        if rejected is not None:
            return rejected
        if (
            path in ("/measure", "/clear")
            and isinstance(payload.get("position"), dict)
            and set(payload["position"]) - {"x", "y"}
        ):
            return self._base_response(False)
        action_data = (
            self._validate_action(payload)
            if path in ("/measure", "/clear") else None
        )

        request_id = payload["request_id"]
        signature = self._canonical(path, payload)
        old = self.idempotency.get(request_id)
        if old is not None:
            if old[:2] != signature:
                raise LocalHTTPError(409, "request_id_conflict")
            return dict(old[2])
        # 已接受请求的网络重试要在接口关闭/超时判定前命中幂等缓存。
        if self.closed or self._is_timed_out():
            self.closed = True
            raise InterfaceClosed("模拟器接口已关闭")

        if path == "/enter":
            if self.entered:
                return self._base_response(False)
            self.entered = True
            self.entered_monotonic = time.monotonic()
            response = self._base_response()
            response.update({
                "max_virtual_duration_s": int(RULES.max_virtual_duration_s),
                "max_real_duration_s": RULES.max_real_duration_s,
                "remaining_real_duration_s": RULES.max_real_duration_s,
            })
        else:
            if not self.entered:
                return self._base_response(False)
            if path == "/exit":
                response = self._base_response()
                response["exit_reason"] = "user_exit"
                self.closed = True
            else:
                assert action_data is not None
                point, channel = action_data
                move_s = _distance(self.position, point) / RULES.speed_mps
                self.position = point
                if path == "/measure":
                    switch_s = (
                        RULES.channel_switch_s
                        if channel != self.current_channel else 0.0
                    )
                    self.current_channel = channel
                    self._advance(move_s + switch_s + RULES.measure_s)
                    response = self._base_response()
                    response.update(self._measure_result(point, channel))
                else:
                    jammer = self.jammers.get(channel)
                    success = bool(
                        jammer is not None and not jammer.cleared
                        and _distance(jammer.position, point) <= RULES.clear_radius_m + 1e-9
                    )
                    if success:
                        jammer.cleared = True
                    self._advance(
                        move_s + (
                            RULES.clear_success_s if success else RULES.clear_miss_s
                        )
                    )
                    response = self._base_response()
                    response["clear_result"] = (
                        "success" if success else "no_target_in_range"
                    )
        self.idempotency[request_id] = (*signature, dict(response))
        return response

    def handle(self, path: str, payload: dict) -> dict:
        if not self.lock.acquire(blocking=False):
            raise LocalHTTPError(409, "concurrent_action")
        try:
            return self._execute(path, payload)
        finally:
            self.lock.release()

    def truth(self) -> dict:
        """仅用于离线训练评估；策略观测不会暴露这些字段。"""
        return {
            "problem": self.problem,
            "seed": self.seed,
            "jammer_count": len(self.jammers),
            "cleared_count": sum(j.cleared for j in self.jammers.values()),
            "jammers": [asdict(j) for j in sorted(
                self.jammers.values(), key=lambda item: item.channel
            )],
        }


class LocalSimulatorClient:
    """ceshi.py/ceshi_actor_critic.py可直接替换的内存客户端。"""

    def __init__(
        self,
        problem: int,
        seed: int,
        robot_id: str = "offline-robot",
        distribution: CaseDistribution = CaseDistribution(),
        count: Optional[int] = None,
    ):
        self.core = LocalSimulatorCore(
            problem, seed, robot_id, distribution, count
        )
        self.robot_id = robot_id
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
        return {
            "arena_id": "default", "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def post(self, path: str, payload: dict, retries: int = 0) -> dict:
        del retries
        return self.core.handle(path, payload)

    def enter(self) -> dict:
        result = self.post("/enter", self._base(self._request_id("enter")))
        if result.get("accepted") is not True:
            raise RuntimeError("/enter失败")
        self.entered = True
        self.virtual_time_s = float(result["virtual_time_s"])
        return result

    def exit(self) -> Optional[dict]:
        if not self.entered or self.closed:
            return None
        result = self.post("/exit", self._base(self._request_id("exit")))
        self.closed = True
        return result

    def _action(self, path: str, position: Point, channel: int) -> dict:
        old_position = self.position
        old_time = self.virtual_time_s
        # 与现有线上客户端一致：实际提交坐标保留4位小数。
        submitted = (round(position[0], 4), round(position[1], 4))
        payload = self._base(self._request_id(path.strip("/")))
        payload.update({
            "position": {"x": submitted[0], "y": submitted[1]},
            "channel": int(channel),
        })
        result = self.post(path, payload)
        if result.get("accepted") is not True:
            raise RuntimeError(f"{path}未被接受")
        # 保留现有客户端的记录语义：路径图用策略原始坐标。
        self.position = position
        self.virtual_time_s = float(result["virtual_time_s"])
        self.records.append({
            "seq": len(self.records) + 1,
            "phase": self.phase,
            "path": path,
            "x": position[0], "y": position[1], "channel": channel,
            "measure_result": result.get("measure_result", ""),
            "clear_result": result.get("clear_result", ""),
            "svd_deg": result.get("svd_deg", ""),
            "virtual_time_s": self.virtual_time_s,
            "delta_virtual_time_s": self.virtual_time_s - old_time,
            "move_distance_m": _distance(old_position, position),
        })
        return result

    def measure(self, position: Point, channel: int) -> dict:
        return self._action("/measure", position, channel)

    def clear(self, position: Point, channel: int) -> dict:
        return self._action("/clear", position, channel)

    def truth(self) -> dict:
        return self.core.truth()


class _DuplicateKey(ValueError):
    pass


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _json_depth(value, level: int = 1) -> int:
    if isinstance(value, dict):
        return max([level] + [_json_depth(v, level + 1) for v in value.values()])
    if isinstance(value, list):
        return max([level] + [_json_depth(v, level + 1) for v in value])
    return level


def make_http_handler(core: LocalSimulatorCore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "JammersOffline/1.0"

        def log_message(self, fmt, *args):
            print("HTTP", self.address_string(), fmt % args)

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(
                body, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            body = core._base_response(False)
            body["error"] = message
            self._send(status, body)

        def do_POST(self):
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment or parsed.path != self.path:
                self._error(404, "unknown_path")
                return
            if parsed.path not in ("/enter", "/measure", "/clear", "/exit"):
                self._error(404, "unknown_path")
                return
            content_type = self.headers.get("Content-Type", "")
            allowed_types = ("application/json", "application/json; charset=utf-8")
            if content_type.lower() not in allowed_types:
                self._error(415, "unsupported_content_type")
                return
            if self.headers.get("Content-Encoding", "identity").lower() != "identity":
                self._error(415, "unsupported_content_encoding")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(400, "invalid_content_length")
                return
            if size > 65536:
                self._error(413, "payload_too_large")
                return
            raw = self.rfile.read(size)
            if raw.startswith(b"\xef\xbb\xbf"):
                self._error(400, "utf8_bom_not_allowed")
                return
            try:
                payload = json.loads(
                    raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                )
                if not isinstance(payload, dict) or _json_depth(payload) > 16:
                    raise ValueError("invalid_root_or_depth")
                result = core.handle(parsed.path, payload)
                self._send(200, result)
                if parsed.path == "/exit" and result.get("accepted") is True:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
            except InterfaceClosed:
                self.close_connection = True
            except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, ValueError):
                self._error(400, "invalid_json")
            except LocalHTTPError as exc:
                self._error(exc.status, exc.message)

        def do_GET(self):
            self._error(405, "method_not_allowed")

    return Handler


def _self_test() -> None:
    # 附件2第10节的199秒完整计时例。
    core = LocalSimulatorCore(3, 1, "test")
    base = lambda rid: {"arena_id": "default", "robot_id": "test", "request_id": rid}
    assert core.handle("/enter", base("enter"))["virtual_time_s"] == 0
    p = base("m1") | {"position": {"x": 300, "y": 400}, "channel": 1}
    assert core.handle("/measure", p)["virtual_time_s"] == 105
    # 幂等重放不再推进时间。
    assert core.handle("/measure", p)["virtual_time_s"] == 105
    p = base("m2") | {"position": {"x": 300, "y": 400}, "channel": 2}
    assert core.handle("/measure", p)["virtual_time_s"] == 111
    p = base("c1") | {"position": {"x": 300, "y": 0}, "channel": 3}
    assert core.handle("/clear", p)["virtual_time_s"] == 194
    p = base("m3") | {"position": {"x": 300, "y": 0}, "channel": 2}
    assert core.handle("/measure", p)["virtual_time_s"] == 199
    assert core.handle("/exit", base("exit"))["virtual_time_s"] == 199

    # 全向、near、清除边界、清除后无信号。
    jammer = Jammer(1, 100.0, 0.0, 1000.0, error_seed=7)
    client = LocalSimulatorClient(3, 2, "test2")
    client.core.jammers = {1: jammer}
    client.enter()
    first = client.measure((0.0, 0.0), 1)
    again = client.measure((0.0, 0.0), 1)
    assert first["svd_deg"] == again["svd_deg"]
    assert client.measure((96.0, 0.0), 1)["measure_result"] == "near"
    assert client.clear((80.0, 0.0), 1)["clear_result"] == "success"
    assert client.measure((80.0, 0.0), 1)["measure_result"] == "no_signal"

    # 定向覆盖为中心方向两侧各90度且包含边界；清除不受方向限制。
    directional = Jammer(
        2, 0.0, 0.0, 1000.0, directional=True,
        direction_deg=0.0, error_seed=8,
    )
    core = LocalSimulatorCore(4, 3, "test3", jammers=[directional])
    core.handle("/enter", {
        "arena_id": "default", "robot_id": "test3", "request_id": "e"
    })
    def action(rid, x, y):
        return {
            "arena_id": "default", "robot_id": "test3", "request_id": rid,
            "position": {"x": x, "y": y}, "channel": 2,
        }
    assert core.handle("/measure", action("east", 100, 0))["measure_result"] == "direction"
    assert core.handle("/measure", action("north", 0, 100))["measure_result"] == "direction"
    assert core.handle("/measure", action("west", -100, 0))["measure_result"] == "no_signal"
    assert core.handle("/clear", action("clear-west", -20, 0))["clear_result"] == "success"

    # 缺少字段是HTTP 400；未声明字段是HTTP 200/accepted=false。
    invalid = {"arena_id": "default", "robot_id": "test3", "request_id": "bad"}
    try:
        core.handle("/measure", invalid)
        raise AssertionError("缺少动作字段未被拒绝")
    except LocalHTTPError as exc:
        assert exc.status == 400
    unknown = action("unknown", 0, 0) | {"extra": 1}
    assert core.handle("/measure", unknown)["accepted"] is False
    print(
        "自检通过：199秒计时例、幂等、固定测向误差、near、"
        "清除边界、定向覆盖和请求字段规则正常。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B题问题3/4本地离线模拟器")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test", help="执行附件规则回归测试")
    serve = sub.add_parser("serve", help="启动一局HTTP兼容模拟器")
    serve.add_argument("--problem", type=int, choices=(3, 4), default=3)
    serve.add_argument("--seed", type=int, default=2026)
    serve.add_argument("--robot-id", default="offline-robot")
    serve.add_argument("--port", type=int, default=2027)
    serve.add_argument("--count", type=int, default=None)
    serve.add_argument("--distribution", choices=("official_estimate", "robust"), default="official_estimate")
    serve.add_argument("--truth-file", default=None, help="退出后写入离线真值JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        _self_test()
        return
    core = LocalSimulatorCore(
        args.problem, args.seed, args.robot_id,
        CaseDistribution(mode=args.distribution), args.count,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_http_handler(core))
    print(
        f"离线模拟器已就绪：http://127.0.0.1:{args.port}  "
        f"问题{args.problem}  seed={args.seed}  源数={len(core.jammers)}"
    )
    print("这一进程只服务一局，收到/exit后自动停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        truth = core.truth()
        print(json.dumps({
            "jammer_count": truth["jammer_count"],
            "cleared_count": truth["cleared_count"],
            "virtual_time_s": core.virtual_time_s,
        }, ensure_ascii=False))
        if args.truth_file:
            path = Path(args.truth_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
