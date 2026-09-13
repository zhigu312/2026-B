#!/usr/bin/env python3
"""B题问题3/4：几何安全约束下的掩码PPO联合规划程序。

复用ceshi.py的接口、几何定位、清除恢复和绘图。PPO只在全局短路前沿
的安全候选中选择下一宏观任务，未成熟模型自动回退到确定性几何基线。
"""

import argparse
import importlib.util
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_FILE = SCRIPT_DIR.parent / "ceshi.py"
if not BASE_FILE.exists():
    BASE_FILE = SCRIPT_DIR / "ceshi.py"
spec = importlib.util.spec_from_file_location("b_problem_base", BASE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载基础算法：{BASE_FILE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


CANDIDATE_DIM = 14
STATE_DIM = 12
FEATURE_DIM = STATE_DIM + CANDIDATE_DIM
HIDDEN_DIM = 32
ALGORITHM_VERSION = "safe_residual_ppo_v6"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp / max(1e-12, float(np.sum(exp)))


class MaskedPPOPolicy:
    """变长安全候选集上的批量残差PPO-Clip。

    几何排序作为弱先验显式加到logit，神经网络只学习残差；
    多局轨迹合并后才计算GAE并更新，避免单局优势标准化消除案例间差异。
    """

    def __init__(
        self, path: Path, training: bool, min_policy_episodes: int = 20,
        gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_ratio: float = 0.20, actor_lr: float = 0.002,
        critic_lr: float = 0.008, update_epochs: int = 8,
        target_kl: float = 0.03, batch_episodes: int = 1,
        minibatch_size: int = 64, entropy_coef: float = 0.01,
        geometry_prior: float = 0.80, residual_warmup_episodes: int = 300,
    ):
        self.path = path
        self.training = training
        self.min_policy_episodes = max(0, min_policy_episodes)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_epochs = max(1, update_epochs)
        self.target_kl = max(0.0, target_kl)
        self.batch_episodes = max(1, batch_episodes)
        self.minibatch_size = max(1, minibatch_size)
        self.entropy_coef = max(0.0, entropy_coef)
        self.geometry_prior = max(0.0, geometry_prior)
        self.residual_warmup_episodes = max(1, residual_warmup_episodes)
        rng = np.random.default_rng(2026)
        self.actor_w1 = rng.normal(0.0, 0.025, (FEATURE_DIM, HIDDEN_DIM))
        self.actor_b1 = np.zeros(HIDDEN_DIM, dtype=float)
        self.actor_w2 = rng.normal(0.0, 0.025, HIDDEN_DIM)
        self.actor_b2 = 0.0
        self.critic = np.zeros(STATE_DIM, dtype=float)
        self.episodes = 0
        self.updates = 0
        self.pending: Optional[dict] = None
        self.trajectory: List[dict] = []
        self.episode_buffer: List[List[dict]] = []
        self.total_decisions = 0
        self.total_deviations = 0
        self.entropy_sum = 0.0
        self.last_entropy = 0.0
        self.last_choice_probability = 1.0
        self.last_baseline_probability = 1.0
        self.last_logit_margin = 0.0
        self.elite_episodes = 0
        self.adam_step = 0
        self.adam_m = {
            "actor_w1": np.zeros_like(self.actor_w1),
            "actor_b1": np.zeros_like(self.actor_b1),
            "actor_w2": np.zeros_like(self.actor_w2),
            "critic": np.zeros_like(self.critic),
        }
        self.adam_v = {key: np.zeros_like(value) for key, value in self.adam_m.items()}
        self.last_update_stats: Dict[str, float] = {}
        self._load()

    @property
    def ready(self) -> bool:
        return self.episodes >= self.min_policy_episodes

    @property
    def residual_scale(self) -> float:
        # 前期保留25%学习残差以开始探索，随经验增加至100%。
        progress = min(1.0, self.episodes / self.residual_warmup_episodes)
        return 0.25 + 0.75 * progress

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("algorithm") != ALGORITHM_VERSION:
                raise ValueError("模型算法或版本不匹配")
            self.actor_w1 = np.asarray(data["actor_w1"], dtype=float)
            self.actor_b1 = np.asarray(data["actor_b1"], dtype=float)
            self.actor_w2 = np.asarray(data["actor_w2"], dtype=float)
            self.actor_b2 = float(data["actor_b2"])
            self.critic = np.asarray(data["critic"], dtype=float)
            if self.actor_w1.shape != (FEATURE_DIM, HIDDEN_DIM):
                raise ValueError("actor_w1维度不匹配")
            if self.actor_b1.shape != (HIDDEN_DIM,):
                raise ValueError("actor_b1维度不匹配")
            if self.actor_w2.shape != (HIDDEN_DIM,):
                raise ValueError("actor_w2维度不匹配")
            if self.critic.shape != (STATE_DIM,):
                raise ValueError("critic维度不匹配")
            self.episodes = int(data.get("episodes", 0))
            self.updates = int(data.get("updates", 0))
            self.elite_episodes = int(data.get("elite_episodes", 0))
            # 推理必须复用训练时的先验强度和残差成熟度，否则同一权重会
            # 产生不同动作。命令行参数仅在新建模型时生效。
            self.geometry_prior = float(data.get(
                "geometry_prior", self.geometry_prior
            ))
            self.residual_warmup_episodes = max(1, int(data.get(
                "residual_warmup_episodes", self.residual_warmup_episodes
            )))
            self.adam_step = int(data.get("adam_step", 0))
            for key in self.adam_m:
                if key in data.get("adam_m", {}):
                    self.adam_m[key] = np.asarray(data["adam_m"][key], dtype=float)
                if key in data.get("adam_v", {}):
                    self.adam_v[key] = np.asarray(data["adam_v"][key], dtype=float)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"警告：PPO模型读取失败，将使用安全初始策略：{exc}")

    def _features(
        self, state: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        return np.concatenate([
            np.repeat(state[None, :], len(candidates), axis=0), candidates
        ], axis=1)

    def _forward(
        self, features: np.ndarray, residual_scale: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(features @ self.actor_w1 + self.actor_b1)
        scale = self.residual_scale if residual_scale is None else residual_scale
        learned = hidden @ self.actor_w2 + self.actor_b2
        geometry_rank = features[:, STATE_DIM + 7]
        logits = self.geometry_prior * geometry_rank + scale * learned
        return hidden, logits

    def _probabilities(
        self, features: np.ndarray, residual_scale: Optional[float] = None
    ) -> np.ndarray:
        return _softmax(self._forward(features, residual_scale)[1])

    def _flush_pending(self) -> None:
        if self.pending is not None and "reward" in self.pending:
            self.trajectory.append(self.pending)
        self.pending = None

    def select(self, state: np.ndarray, candidate_vectors: np.ndarray) -> int:
        if self.training:
            self._flush_pending()
        features = self._features(state, candidate_vectors)
        scale = self.residual_scale
        _, logits = self._forward(features, scale)
        probabilities = _softmax(logits)
        entropy = -float(np.sum(probabilities * np.log(np.maximum(1e-12, probabilities))))
        self.last_entropy = entropy
        self.total_decisions += 1
        if self.training:
            choice = int(np.random.choice(len(candidate_vectors), p=probabilities))
            self.pending = {
                "state": state.copy(),
                "features": features.copy(),
                "choice": choice,
                "old_log_probability": float(math.log(max(1e-12, probabilities[choice]))),
                "old_value": float(np.dot(self.critic, state)),
                "residual_scale": scale,
            }
        else:
            choice = int(np.argmax(probabilities))
        self.last_choice_probability = float(probabilities[choice])
        self.last_baseline_probability = float(probabilities[0])
        self.last_logit_margin = float(logits[choice] - logits[0])
        if choice != 0:
            self.total_deviations += 1
        self.entropy_sum += entropy
        return choice

    def reward(self, value: float) -> None:
        if self.pending is not None:
            self.pending["reward"] = self.pending.get("reward", 0.0) + float(value)

    @staticmethod
    def _clip_gradient(gradient: np.ndarray, max_norm: float = 2.0) -> np.ndarray:
        norm = float(np.linalg.norm(gradient))
        return gradient if norm <= max_norm else gradient * (max_norm / max(1e-12, norm))

    def _gae_for_episode(self, episode: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray([item["reward"] for item in episode], dtype=float)
        values = np.asarray([item["old_value"] for item in episode], dtype=float)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            delta = rewards[index] + self.gamma * next_value - values[index]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[index] = gae
            next_value = values[index]
        return advantages, advantages + values

    def _adam_update(self, name: str, parameter: np.ndarray, gradient: np.ndarray, lr: float) -> None:
        gradient = self._clip_gradient(gradient)
        self.adam_m[name] = 0.9 * self.adam_m[name] + 0.1 * gradient
        self.adam_v[name] = 0.999 * self.adam_v[name] + 0.001 * gradient * gradient
        m_hat = self.adam_m[name] / (1.0 - 0.9 ** self.adam_step)
        v_hat = self.adam_v[name] / (1.0 - 0.999 ** self.adam_step)
        parameter += lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def _update(self) -> None:
        episodes = [episode for episode in self.episode_buffer if episode]
        if not episodes:
            return
        items = [item for episode in episodes for item in episode]
        gae_pairs = [self._gae_for_episode(episode) for episode in episodes]
        advantages = np.concatenate([pair[0] for pair in gae_pairs])
        returns = np.concatenate([pair[1] for pair in gae_pairs])
        if len(advantages) > 1:
            advantages = (
                (advantages - float(np.mean(advantages)))
                / max(1e-8, float(np.std(advantages)))
            )

        completed_epochs = 0
        approx_kl = 0.0
        clip_fraction = 0.0
        mean_entropy = 0.0
        rng = np.random.default_rng(self.episodes + 2026)
        for _ in range(self.update_epochs):
            order = rng.permutation(len(items))
            epoch_logs: List[Tuple[float, float]] = []
            epoch_clipped = 0
            epoch_entropies: List[float] = []
            for start in range(0, len(items), self.minibatch_size):
                indices = order[start:start + self.minibatch_size]
                grad_w1 = np.zeros_like(self.actor_w1)
                grad_b1 = np.zeros_like(self.actor_b1)
                grad_w2 = np.zeros_like(self.actor_w2)
                critic_gradient = np.zeros_like(self.critic)
                for index in indices:
                    item = items[int(index)]
                    advantage = float(advantages[int(index)])
                    scale = float(item["residual_scale"])
                    features = item["features"]
                    hidden, logits = self._forward(features, scale)
                    probabilities = _softmax(logits)
                    choice = item["choice"]
                    new_log = float(math.log(max(1e-12, probabilities[choice])))
                    old_log = float(item["old_log_probability"])
                    epoch_logs.append((old_log, new_log))
                    ratio = float(math.exp(max(-10.0, min(10.0, new_log - old_log))))
                    saturated = (
                        (advantage >= 0.0 and ratio > 1.0 + self.clip_ratio)
                        or (advantage < 0.0 and ratio < 1.0 - self.clip_ratio)
                    )
                    coefficients = np.zeros_like(probabilities)
                    if saturated:
                        epoch_clipped += 1
                    else:
                        coefficients = -probabilities
                        coefficients[choice] += 1.0
                        coefficients *= advantage * ratio
                    entropy = -float(np.sum(
                        probabilities * np.log(np.maximum(1e-12, probabilities))
                    ))
                    epoch_entropies.append(entropy)
                    entropy_gradient = -probabilities * (
                        np.log(np.maximum(1e-12, probabilities)) + entropy
                    )
                    coefficients += self.entropy_coef * entropy_gradient
                    # 几何先验logit不参与训练，反向传播只经过残差支路。
                    coefficients *= scale
                    grad_w2 += hidden.T @ coefficients
                    hidden_gradient = coefficients[:, None] * self.actor_w2[None, :]
                    pre_gradient = hidden_gradient * (1.0 - hidden * hidden)
                    grad_w1 += features.T @ pre_gradient
                    grad_b1 += np.sum(pre_gradient, axis=0)
                    error = float(np.clip(
                        returns[int(index)] - np.dot(self.critic, item["state"]),
                        -50.0, 50.0,
                    ))
                    critic_gradient += error * item["state"]
                denominator = max(1, len(indices))
                self.adam_step += 1
                self._adam_update(
                    "actor_w1", self.actor_w1, grad_w1 / denominator, self.actor_lr
                )
                self._adam_update(
                    "actor_b1", self.actor_b1, grad_b1 / denominator, self.actor_lr
                )
                self._adam_update(
                    "actor_w2", self.actor_w2, grad_w2 / denominator, self.actor_lr
                )
                self._adam_update(
                    "critic", self.critic, critic_gradient / denominator, self.critic_lr
                )
            differences = np.asarray([old - new for old, new in epoch_logs])
            approx_kl = float(np.mean(differences)) if len(differences) else 0.0
            clip_fraction = epoch_clipped / max(1, len(items))
            mean_entropy = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0
            completed_epochs += 1
            if self.target_kl and approx_kl > 1.5 * self.target_kl:
                break
        self.updates += 1
        self.last_update_stats = {
            "batch_episodes": len(episodes),
            "trajectory_steps": len(items),
            "ppo_epochs": completed_epochs,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "policy_entropy": mean_entropy,
            "mean_reward": float(np.mean([
                item["reward"] for item in items
            ])),
            "mean_return": float(np.mean(returns)),
            "residual_scale": self.residual_scale,
        }

    def finish(self, terminal_reward: float = 0.0) -> None:
        if not self.training:
            return
        self.reward(terminal_reward)
        self._flush_pending()
        if self.trajectory:
            self.episode_buffer.append(self.trajectory)
            # 同一批次内对“比同种子几何基线更快”的在策略轨迹加权。
            # 只复制引用、不跨批次回放，仍保持PPO的on-policy约束。
            if terminal_reward > 10.5:
                self.episode_buffer.append(self.trajectory)
                self.elite_episodes += 1
        self.episodes += 1
        self.trajectory = []
        if len(self.episode_buffer) >= self.batch_episodes:
            self._update()
            self.episode_buffer.clear()

    def flush_update(self) -> None:
        if self.training and self.episode_buffer:
            self._update()
            self.episode_buffer.clear()

    def abort_episode(self) -> None:
        self.pending = None
        self.trajectory.clear()

    def save(self, path: Optional[Path] = None) -> None:
        if not self.training:
            return
        target = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": ALGORITHM_VERSION,
            "episodes": self.episodes,
            "updates": self.updates,
            "state_dim": STATE_DIM,
            "candidate_dim": CANDIDATE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "actor_w1": self.actor_w1.tolist(),
            "actor_b1": self.actor_b1.tolist(),
            "actor_w2": self.actor_w2.tolist(),
            "actor_b2": self.actor_b2,
            "critic": self.critic.tolist(),
            "geometry_prior": self.geometry_prior,
            "residual_warmup_episodes": self.residual_warmup_episodes,
            "entropy_coef": self.entropy_coef,
            "batch_episodes": self.batch_episodes,
            "minibatch_size": self.minibatch_size,
            "adam_step": self.adam_step,
            "adam_m": {key: value.tolist() for key, value in self.adam_m.items()},
            "adam_v": {key: value.tolist() for key, value in self.adam_v.items()},
            "last_update_stats": self.last_update_stats,
            "elite_episodes": self.elite_episodes,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)


class PPOSearchAgent(base.SearchAgent):
    def __init__(
        self, *args, scheduler, max_detour_s: float = 45.0,
        candidate_limit: int = 6, baseline_time_s: Optional[float] = None,
        true_jammer_count: Optional[int] = None,
        safety_probability: float = 0.54, safety_margin: float = 0.06,
        safety_two_step_extra_s: float = 12.0,
        max_consecutive_deviations: int = 1,
        tail_threshold_s: float = 20.0, tail_risk_coef: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.max_detour_s = max(0.0, max_detour_s)
        self.candidate_limit = max(2, candidate_limit)
        self.baseline_time_s = baseline_time_s
        self.true_jammer_count = true_jammer_count
        self.safety_probability = min(1.0, max(0.0, safety_probability))
        self.safety_margin = max(0.0, safety_margin)
        self.safety_two_step_extra_s = max(0.0, safety_two_step_extra_s)
        self.max_consecutive_deviations = max(0, max_consecutive_deviations)
        self.tail_threshold_s = max(0.0, tail_threshold_s)
        self.tail_risk_coef = max(0.0, tail_risk_coef)
        self.rl_decisions = 0
        self.policy_deviations = 0
        self.policy_proposed_deviations = 0
        self.safety_gate_blocks = 0
        self.consecutive_policy_deviations = 0
        self.policy_entropy_sum = 0.0
        self.rl_reward_total = 0.0

    def uncertainty_snapshot(self) -> Dict[int, float]:
        return {
            channel: base.estimate_target(state.observations)[1]
            for channel, state in self.channels.items()
            if state.observations and not state.cleared
        }

    def planning_uncertainty_snapshot(self) -> Dict[int, float]:
        """负观测排除域下的规划离散度；只参与选点与奖励，不放宽清除门。"""
        return {
            channel: self.planning_target(state)[1]
            for channel, state in self.channels.items()
            if state.observations and not state.cleared
        }

    def global_state_vector(self, node_count: int, remaining_coverage: int) -> np.ndarray:
        uncertainties = list(self.planning_uncertainty_snapshot().values())
        finite = [min(1500.0, value) for value in uncertainties if math.isfinite(value)]
        mean_uncertainty = (sum(finite) / len(finite)) if finite else 1500.0
        max_uncertainty = max(finite) if finite else 1500.0
        localized = sum(
            bool(state.observations) and self.localization_ready(state)
            for state in self.channels.values() if not state.cleared
        )
        recent_moves = [
            float(record["move_distance_m"]) for record in self.client.records[-12:]
        ]
        recent_max_move = max(recent_moves, default=0.0)
        x, y = self.client.position
        return np.asarray([
            np.clip(x / 1800.0, -1.0, 1.0),
            np.clip(y / 1800.0, -1.0, 1.0),
            1.0 - remaining_coverage / max(1, len(self.scan_waypoints)),
            self.cleared_count / 16.0,
            self.discovered_count / 20.0,
            localized / max(1, self.discovered_count - self.cleared_count),
            1.0 - min(1.0, mean_uncertainty / 1500.0),
            1.0 - min(1.0, max_uncertainty / 1500.0),
            min(1.0, node_count / 28.0),
            min(1.0, self.client.virtual_time_s / 4500.0),
            min(1.0, recent_max_move / 1800.0),
            1.0,
        ], dtype=float)

    def candidate_vector(self, node: dict, rank: int, total: int) -> np.ndarray:
        move = base.distance(self.client.position, node["point"])
        exit_leg = base.distance(node["point"], node["expected_exit"])
        kind = node["kind"]
        info = 0.0
        uncertainty_quality = 0.0
        coverage_gain = 0.0
        if kind == "coverage":
            info = min(1.0, self.expected_information_at(node["point"]) / 4.0)
            coverage_gain = len(self.coverage_gain_cells(node["point"])) / max(
                1, len(self.coverage_cells)
            )
        elif kind == "probe":
            state = node["state"]
            first = state.observations[0]
            target = node["expected_exit"]
            predicted = base.normalize_angle(math.degrees(math.atan2(
                target[1] - node["point"][1], target[0] - node["point"][0]
            )))
            info = abs(math.sin(math.radians(
                base.angle_difference(first.bearing_deg, predicted)
            )))
            uncertainty = base.estimate_target(state.observations)[1]
            uncertainty_quality = 0.0 if not math.isfinite(uncertainty) else (
                1.0 - min(1.0, uncertainty / 200.0)
            )
        else:
            info = 1.0
            uncertainty_quality = 1.0
        exit_probability = max(
            0.0, min(1.0, float(node.get("exit_probability", 1.0)))
        )
        expected_service = move + exit_probability * exit_leg
        return np.asarray([
            1.0,
            1.0 - min(1.0, move / 3600.0),
            1.0 - min(1.0, exit_leg / 1500.0),
            info,
            1.0 if kind == "coverage" else 0.0,
            1.0 if kind == "probe" else 0.0,
            1.0 if kind == "clear" else 0.0,
            1.0 - rank / max(1, total - 1),
            uncertainty_quality,
            min(1.0, 8.0 * coverage_gain),
            exit_probability,
            1.0 - min(1.0, expected_service / 3600.0),
            1.0 - min(1.0, max(0.0, move - 850.0) / 950.0),
            1.0 - min(1.0, (move + exit_leg) / 3600.0),
        ], dtype=float)

    def safe_rl_choice(self, nodes: Sequence[dict], order: Sequence[int]) -> int:
        baseline = order[0]
        baseline_move_s = base.distance(self.client.position, nodes[baseline]["point"]) / 5.0
        # 残差权重成熟前逐步扩大探索半径，避免早期跨场。
        progress = min(1.0, self.scheduler.residual_scale)
        effective_detour_s = min(self.max_detour_s, 15.0 + progress * self.max_detour_s)
        safe = [baseline]
        for index in order[1:self.candidate_limit]:
            move_s = base.distance(self.client.position, nodes[index]["point"]) / 5.0
            if move_s <= baseline_move_s + effective_detour_s:
                safe.append(index)
        state = self.global_state_vector(len(nodes), sum(n["kind"] == "coverage" for n in nodes))
        vectors = np.vstack([
            self.candidate_vector(nodes[index], rank, len(safe))
            for rank, index in enumerate(safe)
        ])
        # 最后一维替换为两步前瞻成本，使PPO能识别“当前近但下一步很远”。
        for row, index in enumerate(safe):
            point = nodes[index]["expected_exit"]
            following = min((
                base.distance(point, nodes[other]["point"])
                for other in safe if other != index
            ), default=0.0)
            current = base.distance(self.client.position, nodes[index]["point"])
            vectors[row, 13] = 1.0 - min(1.0, (current + following) / 4200.0)
        # 模型未达到成熟度门槛时严格继承几何路线。
        if not self.scheduler.training and not self.scheduler.ready:
            choice = 0
        else:
            choice = self.scheduler.select(state, vectors)
        if choice != 0:
            self.policy_proposed_deviations += 1
        # 训练时保留随机探索；验证和正式推理时，PPO必须同时通过置信度、
        # 两步额外成本和连续改策三道门，否则退回几何基线。
        if not self.scheduler.training and choice != 0:
            def two_step_cost_s(index: int) -> float:
                node = nodes[index]
                exit_point = node["expected_exit"]
                service_distance = (
                    base.distance(self.client.position, node["point"])
                    + base.distance(node["point"], exit_point)
                )
                following = min((
                    base.distance(exit_point, nodes[other]["point"])
                    for other in order if other != index
                ), default=0.0)
                return (service_distance + following) / 5.0

            confidence_ok = (
                self.scheduler.last_choice_probability >= self.safety_probability
                and self.scheduler.last_logit_margin >= self.safety_margin
            )
            route_extra_s = two_step_cost_s(safe[choice]) - two_step_cost_s(baseline)
            route_ok = route_extra_s <= self.safety_two_step_extra_s
            sequence_ok = (
                self.consecutive_policy_deviations
                < self.max_consecutive_deviations
            )
            if not (confidence_ok and route_ok and sequence_ok):
                choice = 0
                self.safety_gate_blocks += 1
        self.rl_decisions += 1
        if choice != 0:
            self.policy_deviations += 1
            self.consecutive_policy_deviations += 1
        else:
            self.consecutive_policy_deviations = 0
        self.policy_entropy_sum += self.scheduler.last_entropy
        return safe[choice]

    def transition_reward(
        self, old_time: float, old_cleared: int, old_discovered: int,
        old_uncertainty: Dict[int, float], old_planning_uncertainty: Dict[int, float],
        old_cells: int,
    ) -> float:
        elapsed = self.client.virtual_time_s - old_time
        gain = 0.0
        potential_gain = 0.0
        newly_clearable = 0
        negative_domain_gain = 0.0
        clearance_target = base.CLEAR_RADIUS - base.CLEARANCE_NUMERIC_MARGIN_M
        log_denominator = math.log(1500.0 / clearance_target)

        def clearance_potential(radius: float) -> float:
            if not math.isfinite(radius):
                return 0.0
            clipped = min(1500.0, max(clearance_target, radius))
            return max(0.0, min(
                1.0, math.log(1500.0 / clipped) / log_denominator
            ))

        for channel, previous in old_uncertainty.items():
            current = base.estimate_target(self.channels[channel].observations)[1]
            gain += base.information_gain(previous, current)
            potential_gain += max(
                0.0, clearance_potential(current) - clearance_potential(previous)
            )
            if previous > clearance_target and current <= clearance_target:
                newly_clearable += 1
        for channel, previous in old_planning_uncertainty.items():
            state = self.channels[channel]
            if state.cleared:
                current = clearance_target
            else:
                current = self.planning_target(state)[1]
            negative_domain_gain += base.information_gain(previous, current)
        new_cleared = self.cleared_count - old_cleared
        new_discovered = max(0, self.discovered_count - old_discovered)
        coverage_gain = max(
            0.0,
            (len(self.covered_cells) - old_cells) / max(1, len(self.coverage_cells)),
        )
        # 指导书GIRH思想的势函数塑形：时间仍是主目标，但明确奖励单位步骤内
        # 的发现、可行域收缩、跨入20m保证清除域及保守覆盖进展。
        tail_penalty = (max(0.0, elapsed - 120.0) / 120.0) ** 2
        reward = (
            -elapsed / 100.0
            + 0.8 * new_discovered + 1.2 * new_cleared
            + 0.30 * gain + 0.45 * negative_domain_gain
            + 4.0 * potential_gain
            + 3.0 * newly_clearable + 3.0 * coverage_gain
            - 0.75 * tail_penalty
        )
        self.rl_reward_total += reward
        return reward

    def run_joint_coverage_localization(self) -> None:
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
            for channel, plan in list(pending_plans.items()):
                state = self.channels[channel]
                if (
                    state.cleared or self.localization_ready(state)
                    or len(state.observations) != plan["observation_count"]
                ):
                    pending_plans.pop(channel, None)

            nodes: List[dict] = [{
                "kind": "coverage", "point": point,
                "expected_exit": point, "coverage_point": point,
            } for point in remaining_coverage]
            completed = len(self.scan_waypoints) - len(remaining_coverage)
            if completed < self.joint_coverage_warmup:
                order = self.optimized_plan_order(nodes, None)
                point = nodes[order[0]]["coverage_point"]
                step += 1
                print(f"\nPPO预热{completed + 1}/{self.joint_coverage_warmup}: {point}")
                self.client.phase = "ppo_coverage_warmup"
                self.scan_one_waypoint(point)
                remaining_coverage.remove(point)
                continue

            for channel, state in self.channels.items():
                if state.cleared or not state.observations:
                    continue
                if self.localization_ready(state):
                    if channel not in failed_joint_clear:
                        estimate, _ = base.estimate_target(state.observations)
                        if estimate is not None:
                            nodes.append({
                                "kind": "clear", "point": estimate,
                                "expected_exit": estimate, "state": state,
                            })
                    continue
                if probe_attempts[channel] >= self.max_probes:
                    continue
                if channel not in pending_plans:
                    plan = self.plan_probe(state, probe_attempts[channel])
                    if plan is not None:
                        pending_plans[channel] = plan
                if channel in pending_plans:
                    node = dict(pending_plans[channel])
                    node["kind"] = "probe"
                    nodes.append(node)

            if not nodes:
                break
            self.optimize_joint_probe_variants(nodes, None)
            order = self.optimized_plan_order(nodes, None)
            chosen = self.safe_rl_choice(nodes, order)
            node = nodes[chosen]
            step += 1
            old_time = self.client.virtual_time_s
            old_cleared = self.cleared_count
            old_discovered = self.discovered_count
            old_uncertainty = self.uncertainty_snapshot()
            old_planning_uncertainty = self.planning_uncertainty_snapshot()
            old_cells = len(self.covered_cells)

            if node["kind"] == "coverage":
                point = node["coverage_point"]
                print(f"\nPPO步骤{step}: 公共覆盖 {point}")
                self.client.phase = "ppo_joint_coverage"
                self.scan_one_waypoint(point)
                remaining_coverage.remove(point)
            elif node["kind"] == "probe":
                state = node["state"]
                print(f"\nPPO步骤{step}: 频道{state.channel}补测")
                self.client.phase = "ppo_joint_probe"
                rank = order.index(chosen)
                node["next_route_point"] = (
                    nodes[order[rank + 1]]["point"]
                    if rank + 1 < len(order) else None
                )
                node["future_coverage_points"] = list(remaining_coverage)
                self.execute_probe(node)
                probe_attempts[state.channel] += 1
                pending_plans.pop(state.channel, None)
            else:
                state = node["state"]
                print(f"\nPPO步骤{step}: 频道{state.channel}清除")
                self.client.phase = "ppo_joint_clear"
                if not self.clear_at(state, node["point"]):
                    self.client.phase = "ppo_local_recovery"
                    if not self.recover_clear_locally(state):
                        failed_joint_clear.add(state.channel)

            self.scheduler.reward(self.transition_reward(
                old_time, old_cleared, old_discovered, old_uncertainty,
                old_planning_uncertainty, old_cells
            ))

    def terminal_reward(self) -> float:
        if self.true_jammer_count is not None:
            missed = max(0, self.true_jammer_count - self.cleared_count)
        else:
            missed = sum(
                bool(state.observations) and not state.cleared
                for state in self.channels.values()
            )
        if self.baseline_time_s is not None:
            # 配对基线消除了地图难度和干扰源数量的影响：快60秒即+6，
            # 慢60秒即-6。全清仅给小常数，防止再次盖住时间信号。
            regret_reward = (
                self.baseline_time_s - self.client.virtual_time_s
            ) / 10.0
            regret_s = self.client.virtual_time_s - self.baseline_time_s
            if regret_s > self.tail_threshold_s:
                tail_excess = (regret_s - self.tail_threshold_s) / 10.0
                regret_reward -= self.tail_risk_coef * tail_excess * tail_excess
        else:
            regret_reward = -max(
                0.0, self.client.virtual_time_s - 3300.0
            ) / 10.0
        return regret_reward + (10.0 if missed == 0 else -250.0 * missed)

    def run(self) -> dict:
        summary = super().run()
        terminal = self.terminal_reward()
        self.rl_reward_total += terminal
        self.scheduler.finish(terminal)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B题问题3/4安全掩码PPO联合规划")
    parser.add_argument("--problem", type=int, choices=(3, 4), default=3)
    parser.add_argument("--base-url", default=base.DEFAULT_BASE_URL)
    parser.add_argument("--robot-id", default=base.DEFAULT_ROBOT_ID)
    parser.add_argument(
        "--train", action="store_true",
        help="收集本次完整轨迹并执行一次PPO更新；正式测试不要开启",
    )
    parser.add_argument("--joint-warmup", type=int, default=2)
    parser.add_argument("--max-probes", type=int, default=4)
    parser.add_argument("--max-detour-s", type=float, default=55.0)
    parser.add_argument("--candidate-limit", type=int, default=6)
    parser.add_argument("--safety-probability", type=float, default=0.54)
    parser.add_argument("--safety-margin", type=float, default=0.06)
    parser.add_argument("--safety-two-step-extra-s", type=float, default=12.0)
    parser.add_argument("--max-consecutive-deviations", type=int, default=1)
    parser.add_argument("--tail-threshold-s", type=float, default=20.0)
    parser.add_argument("--tail-risk-coef", type=float, default=1.0)
    parser.add_argument("--min-policy-episodes", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--actor-lr", type=float, default=0.002)
    parser.add_argument("--critic-lr", type=float, default=0.008)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--batch-episodes", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--geometry-prior", type=float, default=0.40)
    parser.add_argument("--residual-warmup-episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--model-path", default=None,
        help="指定PPO模型JSON；默认优先自动加载v6部署最佳模型",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.robot_id.strip():
        raise SystemExit("请使用 --robot-id 填入模拟器分配的团队号/机器狗编号")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    if args.joint_warmup <= 0 or args.max_probes <= 0:
        raise SystemExit("joint-warmup和max-probes必须为正整数")
    if args.min_policy_episodes < 0 or args.update_epochs <= 0:
        raise SystemExit("min-policy-episodes不能为负，update-epochs必须为正")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        raise SystemExit("gamma和gae-lambda参数范围无效")
    run_root = (
        SCRIPT_DIR / "ppo_runs" / f"problem_{args.problem}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        default_candidates = [
            SCRIPT_DIR / "offline_models"
            / f"ppo_problem_{args.problem}_residual_v6_deploy_best.json",
            SCRIPT_DIR / "offline_models"
            / f"ppo_problem_{args.problem}_residual_v6_safe_best.json",
            SCRIPT_DIR / "ppo_models" / f"ppo_problem_{args.problem}_policy.json",
        ]
        model_path = default_candidates[-1]
        for candidate in default_candidates[:-1]:
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
                if (
                    metadata.get("algorithm") == ALGORITHM_VERSION
                    and int(metadata.get("episodes", 0)) >= args.min_policy_episodes
                ):
                    model_path = candidate
                    break
            except (OSError, ValueError, TypeError):
                continue
    print(f"PPO模型：{model_path}")
    scheduler = MaskedPPOPolicy(
        model_path,
        training=args.train,
        min_policy_episodes=args.min_policy_episodes,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        update_epochs=args.update_epochs,
        target_kl=args.target_kl,
        batch_episodes=args.batch_episodes,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        geometry_prior=args.geometry_prior,
        residual_warmup_episodes=args.residual_warmup_episodes,
    )

    q_policy = base.QLearningPolicy(
        SCRIPT_DIR.parent / "rl_policy.json",
        epsilon=0.0,
        alpha=0.0,
        # PPO训练期间也冻结Q层，避免两个学习器同时更新导致环境非平稳。
        training=False,
    )
    client = base.SimulatorClient(args.base_url, args.robot_id)
    agent = PPOSearchAgent(
        client, q_policy, args.problem,
        scheduler=scheduler,
        max_detour_s=args.max_detour_s,
        candidate_limit=args.candidate_limit,
        safety_probability=args.safety_probability,
        safety_margin=args.safety_margin,
        safety_two_step_extra_s=args.safety_two_step_extra_s,
        max_consecutive_deviations=args.max_consecutive_deviations,
        tail_threshold_s=args.tail_threshold_s,
        tail_risk_coef=args.tail_risk_coef,
        max_probes=args.max_probes,
        joint_coverage_localization=True,
        joint_coverage_warmup=args.joint_warmup,
        directed_plan_order=True,
        joint_probe_variants=True,
        immediate_refine=True,
        information_gated_coverage=True,
        information_rate_planning=True,
    )
    summary = None
    try:
        summary = agent.run()
        summary.update({
            "rl_algorithm": ALGORITHM_VERSION,
            "ppo_training": args.train,
            "ppo_policy_active": scheduler.ready,
            "ppo_episodes": scheduler.episodes,
            "ppo_updates": scheduler.updates,
            "ppo_decisions": agent.rl_decisions,
            "ppo_policy_deviations": agent.policy_deviations,
            "ppo_proposed_deviations": agent.policy_proposed_deviations,
            "ppo_safety_gate_blocks": agent.safety_gate_blocks,
            "ppo_policy_changed_rate": (
                agent.policy_deviations / max(1, agent.rl_decisions)
            ),
            "ppo_mean_entropy": (
                agent.policy_entropy_sum / max(1, agent.rl_decisions)
            ),
            "ppo_reward": agent.rl_reward_total,
            "ppo_last_update": scheduler.last_update_stats,
            "max_detour_s": args.max_detour_s,
        })
        print("\nPPO运行汇总：")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("收到中断信号，尝试主动退出")
        client.exit()
    except Exception:
        client.exit()
        raise
    finally:
        q_policy.save()
        scheduler.save()
        if summary is None:
            summary = {
                "problem": args.problem,
                "cleared_count": agent.cleared_count,
                "virtual_time_s": client.virtual_time_s,
                "path_distance_m": sum(r["move_distance_m"] for r in client.records),
                "action_count": len(client.records),
                "rl_algorithm": ALGORITHM_VERSION,
                "ppo_training": args.train,
                "ppo_policy_active": scheduler.ready,
                "ppo_episodes": scheduler.episodes,
            }
        base.save_outputs(output_dir, client.records, summary, agent.channels)
        base.draw_path(
            output_dir / "ppo_path.png",
            client.records, summary, not args.no_show,
        )


if __name__ == "__main__":
    main()
